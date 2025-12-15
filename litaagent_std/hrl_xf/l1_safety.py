"""L1 安全护盾层 - 时序 ATP 算法实现.

L1 是确定性规则层，不含可训练参数。其职责：
1. 计算时序库容约束 Q_safe[δ]，维度 (H+1,)
2. 计算资金约束 B_free
3. 生成动作掩码 time_mask 供 L3 使用
4. 提供基准动作 baseline_action

核心算法：Available-To-Promise (ATP) 时序安全检查
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Optional
import numpy as np

if TYPE_CHECKING:
    from scml.std import StdAWI


@dataclass
class L1Output:
    """L1 安全护盾的输出结构.
    
    Attributes:
        Q_safe: 每个交货日的最大安全买入量，shape (H+1,)，δt ∈ {0, 1, ..., H}
        B_free: 可用资金上限
        time_mask: 时间掩码 (0 或 -inf)，shape (H+1,)，用于 L3 的 Masked Softmax
        baseline_action: 基准动作 (q_base, p_base, t_base)
        L_trajectory: 库存轨迹，shape (H,)
        C_total: 库容向量，shape (H,)
    """
    Q_safe: np.ndarray
    B_free: float
    time_mask: np.ndarray
    baseline_action: Tuple[float, float, int]
    L_trajectory: np.ndarray
    C_total: np.ndarray


def get_capacity_vector(awi: "StdAWI", horizon: int) -> np.ndarray:
    """获取未来 H 天的库容上限向量.
    
    决策逻辑：
    1. 优先使用 awi.profile.storage_capacity（如存在）
    2. 否则使用动态公式 C[k] = n_lines × (T_max - (t+k))
    
    Args:
        awi: Agent World Interface
        horizon: 规划视界 H
        
    Returns:
        C_total: shape (H,)
    """
    # 尝试使用 API 提供的静态容量
    static_cap = getattr(awi.profile, 'storage_capacity', None)
    
    if static_cap is not None:
        return np.full(horizon, static_cap, dtype=np.float32)
    
    # 动态容量：基于剩余天数 × 日产能
    n_lines = awi.profile.n_lines
    t_current = awi.current_step
    t_max = awi.n_steps
    
    C_total = np.zeros(horizon, dtype=np.float32)
    for k in range(horizon):
        remaining_days = t_max - (t_current + k)
        C_total[k] = n_lines * max(0, remaining_days)
    
    return C_total


def extract_commitments(awi: "StdAWI", horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """从已签合约中提取未来 H 天的入库和出库承诺.
    
    注意：仅使用 signed_contracts，不含 unsigned_contracts。
    未签署合约可能不会成交，纳入计算会导致过度保守。
    
    Args:
        awi: Agent World Interface
        horizon: 规划视界 H
        
    Returns:
        Q_in: shape (H,) - 每天的入库量
        Q_out: shape (H,) - 每天的出库量
    """
    Q_in = np.zeros(horizon, dtype=np.float32)
    Q_out = np.zeros(horizon, dtype=np.float32)
    
    t_current = awi.current_step
    agent_id = awi.agent.id
    
    # 仅遍历已签署的合约
    signed_contracts = getattr(awi, 'signed_contracts', []) or []
    
    for contract in signed_contracts:
        # 跳过已执行的合约
        if getattr(contract, 'executed', False):
            continue
        
        # 获取交货时间
        agreement = getattr(contract, 'agreement', {}) or {}
        delivery_time = agreement.get('time', getattr(contract, 'time', None))
        
        if delivery_time is None:
            continue
            
        delta = delivery_time - t_current
        
        if 0 <= delta < horizon:
            quantity = agreement.get('quantity', 0)
            
            # 判断是买入还是卖出
            annotation = getattr(contract, 'annotation', {}) or {}
            seller_id = annotation.get('seller')
            
            if seller_id != agent_id:
                # 我是买方 -> 入库
                Q_in[delta] += quantity
            else:
                # 我是卖方 -> 出库
                Q_out[delta] += quantity
    
    return Q_in, Q_out


def compute_inventory_trajectory(
    I_now: float,
    Q_in: np.ndarray,
    Q_out: np.ndarray,
    Q_prod: np.ndarray,
    horizon: int
) -> np.ndarray:
    """计算未来 H 天的库存水位轨迹.
    
    公式：L[k] = I_now + Σ_{j=0}^{k} (Q_in[j] - Q_out[j] - Q_prod[j])
    
    采用方案B（适度保守）：扣减 Q_out
    
    Args:
        I_now: 当前库存
        Q_in: 入库承诺向量，shape (H,)
        Q_out: 出库承诺向量，shape (H,)
        Q_prod: 生产消耗向量，shape (H,)
        horizon: 规划视界
        
    Returns:
        L: shape (H,) - 每天的预计库存
    """
    net_flow = Q_in - Q_out - Q_prod  # shape (H,)
    cumulative_flow = np.cumsum(net_flow)  # shape (H,)
    L = I_now + cumulative_flow
    return L


def compute_safe_buy_mask(
    C_total: np.ndarray,
    L: np.ndarray,
    horizon: int
) -> np.ndarray:
    """计算每个交货日 delta 的最大安全买入量.
    
    公式：Q_safe[delta] = min_{k=delta}^{H-1} (C_total[k] - L[k])
    
    使用逆向累积最小值高效计算。
    
    Args:
        C_total: 库容向量，shape (H,)
        L: 库存轨迹，shape (H,)
        horizon: 规划视界
        
    Returns:
        Q_safe: shape (H,)
    """
    raw_free = C_total - L  # shape (H,)
    
    # 逆向累积最小值
    reversed_free = raw_free[::-1]
    reversed_cummin = np.minimum.accumulate(reversed_free)
    Q_safe = reversed_cummin[::-1]
    
    # 非负约束
    Q_safe = np.maximum(Q_safe, 0)
    
    return Q_safe


def compute_budget_limit(
    B_current: float,
    Payables: np.ndarray,
    reserve: float = 1000.0
) -> float:
    """计算可用于采购的最大资金.
    
    公式：B_free = B_current - reserve - Σ Payables
    
    Args:
        B_current: 当前现金余额
        Payables: 未来应付款项向量，shape (H,)
        reserve: 安全储备金
        
    Returns:
        B_free: 可用资金上限
    """
    total_payables = np.sum(Payables)
    B_free = B_current - reserve - total_payables
    return max(0.0, B_free)


class L1SafetyLayer:
    """L1 时序安全护盾.
    
    核心职责：
    1. 计算时序库容约束 Q_safe[δ]
    2. 计算资金约束 B_free
    3. 生成时间掩码 time_mask
    4. 提供基准动作 baseline_action
    
    Args:
        horizon: 规划视界 H，默认 40
        reserve: 安全储备金，默认 1000.0
        min_tradable_qty: 最小可交易量阈值，默认 1.0
    """
    
    def __init__(
        self,
        horizon: int = 40,
        reserve: float = 1000.0,
        min_tradable_qty: float = 1.0
    ):
        self.horizon = horizon
        self.reserve = reserve
        self.min_tradable_qty = min_tradable_qty
    
    def compute(self, awi: "StdAWI", is_buying: bool) -> L1Output:
        """计算安全约束.
        
        Args:
            awi: Agent World Interface
            is_buying: 当前是买入还是卖出角色
            
        Returns:
            L1Output: 包含 Q_safe, B_free, time_mask, baseline_action 等
        """
        # 1. 获取库容向量
        C_total = get_capacity_vector(awi, self.horizon)
        
        # 2. 提取合约承诺
        Q_in, Q_out = extract_commitments(awi, self.horizon)
        
        # 3. 估计生产消耗（保守：假设满负荷）
        n_lines = getattr(awi.profile, 'n_lines', 1)
        Q_prod = np.full(self.horizon, n_lines, dtype=np.float32)
        
        # 4. 计算库存轨迹
        # current_inventory 返回 tuple (input_qty, output_qty) 或单个数值
        inventory = getattr(awi, 'current_inventory', None)
        if inventory is None:
            I_now = 0.0
        elif hasattr(inventory, '__iter__') and not isinstance(inventory, (str, dict)):
            # tuple 或 list
            I_now = float(sum(inventory))
        elif isinstance(inventory, dict):
            I_now = float(sum(inventory.values()))
        else:
            I_now = float(inventory)
        L = compute_inventory_trajectory(I_now, Q_in, Q_out, Q_prod, self.horizon)
        
        # 5. 计算安全买入量掩码 (H 维)
        Q_safe_h = compute_safe_buy_mask(C_total, L, self.horizon)
        
        # 6. 扩展为 H+1 维，支持 δt ∈ {0, 1, ..., H}
        # δt = H 的处理：复用 H-1 的值
        Q_safe = np.zeros(self.horizon + 1, dtype=np.float32)
        Q_safe[:self.horizon] = Q_safe_h
        Q_safe[self.horizon] = Q_safe_h[-1] if len(Q_safe_h) > 0 else 0.0
        
        # 7. 计算资金约束
        Payables = self._extract_payables(awi)
        wallet = getattr(awi, 'wallet', 0.0) or 0.0
        B_free = compute_budget_limit(wallet, Payables, self.reserve)
        
        # 8. 生成时间掩码 (用于 L3 的 Masked Softmax)，H+1 维
        time_mask = np.where(Q_safe >= self.min_tradable_qty, 0.0, -np.inf)
        
        # 9. 生成基准动作
        baseline_action = self._compute_baseline(awi, Q_safe, B_free, is_buying)
        
        return L1Output(
            Q_safe=Q_safe,
            B_free=B_free,
            time_mask=time_mask,
            baseline_action=baseline_action,
            L_trajectory=L,
            C_total=C_total
        )
    
    def _extract_payables(self, awi: "StdAWI") -> np.ndarray:
        """提取未来的应付款项."""
        Payables = np.zeros(self.horizon, dtype=np.float32)
        t_current = awi.current_step
        agent_id = awi.agent.id
        
        signed_contracts = getattr(awi, 'signed_contracts', []) or []
        
        for contract in signed_contracts:
            if getattr(contract, 'executed', False):
                continue
            
            annotation = getattr(contract, 'annotation', {}) or {}
            seller_id = annotation.get('seller')
            
            if seller_id != agent_id:
                # 我是买方，需要付款
                agreement = getattr(contract, 'agreement', {}) or {}
                delivery_time = agreement.get('time', getattr(contract, 'time', None))
                
                if delivery_time is None:
                    continue
                    
                delta = delivery_time - t_current
                
                if 0 <= delta < self.horizon:
                    quantity = agreement.get('quantity', 0)
                    unit_price = agreement.get('unit_price', 0)
                    Payables[delta] += quantity * unit_price
        
        return Payables
    
    def _compute_baseline(
        self,
        awi: "StdAWI",
        Q_safe: np.ndarray,
        B_free: float,
        is_buying: bool
    ) -> Tuple[float, float, int]:
        """计算基准动作.
        
        基准策略：选择最早的可行交货日，使用保守的数量和价格。
        
        Args:
            awi: Agent World Interface
            Q_safe: 安全买入量向量，shape (H+1,)
            B_free: 可用资金
            is_buying: 是否为买入角色
            
        Returns:
            (q_base, p_base, t_base): 基准动作
        """
        t_current = awi.current_step
        
        # 获取市场价格 (trading_prices 是 numpy 数组，按产品索引)
        trading_prices = getattr(awi, 'trading_prices', None)
        
        def get_price(product_id: int, default: float) -> float:
            """安全地从 trading_prices 获取价格."""
            if trading_prices is None:
                return default
            try:
                if hasattr(trading_prices, '__getitem__'):
                    if product_id < len(trading_prices):
                        return float(trading_prices[product_id])
            except (IndexError, TypeError):
                pass
            return default
        
        if is_buying:
            # 买入：使用输入产品价格
            input_product = awi.my_input_product
            if hasattr(input_product, '__len__') and not isinstance(input_product, (str, int)):
                input_product = input_product[0]
            market_price = get_price(input_product, 10.0)
            
            # 找到最早可行的交货日
            best_delta = 1  # 默认明天
            for delta in range(self.horizon + 1):
                if Q_safe[delta] >= self.min_tradable_qty:
                    best_delta = delta
                    break
            
            # 基准数量：安全量的一半（保守）
            q_base = min(Q_safe[best_delta] / 2, B_free / max(market_price, 1.0))
            q_base = max(1.0, q_base)
            
            # 基准价格：略低于市场价（买方希望便宜）
            p_base = market_price * 0.95
            
        else:
            # 卖出：使用输出产品价格
            output_product = awi.my_output_product
            if hasattr(output_product, '__len__') and not isinstance(output_product, (str, int)):
                output_product = output_product[0]
            market_price = get_price(output_product, 20.0)
            
            # 对于卖出，Q_safe 不直接适用，使用库存
            # current_inventory 返回 tuple (input_qty, output_qty) 或单个数值
            inventory = getattr(awi, 'current_inventory', None)
            if inventory is None:
                available = 0.0
            elif hasattr(inventory, '__iter__') and not isinstance(inventory, (str, dict)):
                available = float(sum(inventory))
            elif isinstance(inventory, dict):
                available = float(sum(inventory.values()))
            else:
                available = float(inventory)
            
            best_delta = 1
            q_base = max(1.0, available / 2)
            
            # 基准价格：略高于市场价（卖方希望贵）
            p_base = market_price * 1.05
        
        t_base = best_delta
        
        return (float(q_base), float(p_base), int(t_base))
    
    def clip_action(
        self,
        action: Tuple[float, float, int],
        Q_safe: np.ndarray,
        B_free: float,
        is_buying: bool,
        min_price: float = 0.0,
        max_price: float = float('inf')
    ) -> Tuple[int, float, int]:
        """裁剪动作到安全范围.
        
        所有最终动作必须经过此方法裁剪。
        
        Args:
            action: (quantity, price, delivery_time) 原始动作
            Q_safe: 安全买入量向量，shape (H+1,)
            B_free: 可用资金
            is_buying: 是否为买入角色
            min_price: 最小价格（卖方底价）
            max_price: 最大价格（买方顶价）
            
        Returns:
            (quantity, price, delivery_time): 裁剪后的动作
        """
        quantity, price, delivery_time = action
        
        # 确保 delivery_time 在有效范围内
        delivery_time = int(max(0, min(delivery_time, self.horizon)))
        
        # 价格裁剪
        if is_buying:
            price = min(price, max_price)
        else:
            price = max(price, min_price)
        price = max(0.0, price)
        
        # 数量裁剪
        if is_buying:
            # 不超过安全量
            max_qty = Q_safe[delivery_time] if delivery_time < len(Q_safe) else 0.0
            # 不超过资金允许
            if price > 0:
                max_qty = min(max_qty, B_free / price)
            quantity = min(quantity, max_qty)
        
        quantity = max(0, int(quantity))
        
        return (quantity, float(price), delivery_time)


# 向后兼容别名
TemporalSafetyShield = L1SafetyLayer


__all__ = [
    "L1Output",
    "L1SafetyLayer",
    "TemporalSafetyShield",
    "get_capacity_vector",
    "extract_commitments",
    "compute_inventory_trajectory",
    "compute_safe_buy_mask",
    "compute_budget_limit",
]
