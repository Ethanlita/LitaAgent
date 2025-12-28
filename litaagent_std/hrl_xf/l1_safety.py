"""L1 安全护盾层 - 时序 ATP 算法实现.

L1 是确定性规则层，不含可训练参数。其职责：
1. 计算时序库容约束 Q_safe[δ]，维度 (H+1,)
2. 计算资金约束 B_free
3. 生成动作掩码 time_mask 供 L3 使用

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
        Q_safe_sell: 每个交货日的最大安全卖出量，shape (H+1,)，δt ∈ {0, 1, ..., H}
        B_free: 可用资金上限
        time_mask: 时间掩码 (0 或 -inf)，shape (H+1,)，用于 L3 的 Masked Softmax
        L_trajectory: 原材料 backlog 轨迹（满载加工假设），shape (H,)
        C_total: 库容向量，shape (H,)
        Q_in: 已承诺入库量向量，shape (H,)
        I_input_now: 当前原材料库存
        n_lines: 生产线数量（每日最大加工量）
    """
    Q_safe: np.ndarray
    Q_safe_sell: np.ndarray
    B_free: float
    time_mask: np.ndarray
    L_trajectory: np.ndarray
    C_total: np.ndarray
    Q_in: Optional[np.ndarray] = None
    I_input_now: Optional[float] = None
    n_lines: Optional[float] = None


def get_capacity_vector(awi: "StdAWI", horizon: int) -> np.ndarray:
    """获取未来 H 天的库容上限向量（经济容量）.
    
    注意：SCML 2025 Standard 世界不存在 storage_capacity 属性，
    采用"经济容量" = n_lines × 剩余天数 作为产能约束。
    
    逻辑解释：买入的是原材料，需要加工才能出售。超出剩余可加工
    天数的原材料会因无法及时加工而被浪费，因此动态库容本质上是
    "可有效利用的原材料容量"。
    
    Args:
        awi: Agent World Interface
        horizon: 规划视界 H
        
    Returns:
        C_total: shape (H,)
    """
    # SCML 无 storage_capacity，使用经济容量
    n_lines = awi.profile.n_lines
    t_current = awi.current_step
    t_max = awi.n_steps
    
    C_total = np.zeros(horizon, dtype=np.float32)
    for k in range(horizon):
        remaining_days = t_max - (t_current + k)
        C_total[k] = n_lines * max(0, remaining_days)
    
    return C_total


def _sum_mapping_values(mapping) -> float:
    """对字典值求和（兼容 None 和空值）."""
    if mapping is None:
        return 0.0
    if isinstance(mapping, dict):
        return sum(float(v) for v in mapping.values() if v is not None)
    return 0.0


def _fill_future_values(future_map, current_step: int, target: np.ndarray) -> None:
    """从 future_* 字典填充目标数组."""
    if not isinstance(future_map, dict):
        return
    for step, per_partner in future_map.items():
        try:
            delta = int(step) - current_step
        except Exception:
            continue
        if 0 <= delta < len(target):
            target[delta] += _sum_mapping_values(per_partner)


def extract_commitments(awi: "StdAWI", horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """从 AWI 提取未来 H 天的入库和出库承诺.
    
    优先使用 AWI 的 supplies/sales/future_supplies/future_sales 接口，
    与 tracker_mixin.py 保持一致，确保线上/离线数据口径相同。
    
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
    
    # 优先使用 AWI 的 supplies/sales/future_* 接口（与 tracker_mixin 一致）
    supplies = getattr(awi, 'supplies', None)
    sales = getattr(awi, 'sales', None)
    future_supplies = getattr(awi, 'future_supplies', None)
    future_sales = getattr(awi, 'future_sales', None)
    
    has_awi_future = any(x is not None for x in (supplies, sales, future_supplies, future_sales))
    
    if has_awi_future:
        # 当前日 (delta=0)
        Q_in[0] = _sum_mapping_values(supplies)
        Q_out[0] = _sum_mapping_values(sales)
        
        # 未来日 (delta>0)
        _fill_future_values(future_supplies, t_current, Q_in)
        _fill_future_values(future_sales, t_current, Q_out)
        
        return Q_in, Q_out
    
    # 回退：使用 signed_contracts（兼容旧版本 negmas）
    agent_id = awi.agent.id
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
    n_lines: float,
    horizon: int
) -> np.ndarray:
    """计算未来 H 天的原材料 backlog 轨迹（满载加工假设）.
    
    递推：
    B[0] = I_now
    B[k+1] = max(0, B[k] + Q_in[k] - n_lines)
    
    语义：
    - B[k] 表示第 k 天开始前的未加工原材料库存
    - 假设生产线每日满载（最多 n_lines）
    - Q_out 是成品出库，不影响原材料库存，故不包含
    
    Args:
        I_now: 当前原材料库存
        Q_in: 入库承诺向量，shape (H,)
        n_lines: 每日最大加工量
        horizon: 规划视界
        
    Returns:
        B: shape (H,) - 每天的原材料 backlog
    """
    backlog = np.zeros(horizon, dtype=np.float32)
    b = float(I_now)
    for k in range(horizon):
        backlog[k] = max(0.0, b)
        b = b + float(Q_in[k]) - float(n_lines)
        if b < 0.0:
            b = 0.0
    return backlog


def compute_safe_buy_mask(
    C_total: np.ndarray,
    Q_in: np.ndarray,
    backlog: np.ndarray,
    horizon: int
) -> np.ndarray:
    """计算每个交货日 delta 的最大安全买入量.
    
    公式：
    Q_safe[δ] = max(0, C_total[δ] - backlog[δ] - Σ_{k=δ}^{H-1} Q_in[k])
    
    解释：
    - C_total[δ] 是交货日 δ 之后的剩余可加工总量
    - backlog[δ] 是交货日 δ 开始前已积压的原材料
    - Σ_{k=δ}^{H-1} Q_in[k] 是已承诺的未来入库量
    
    Args:
        C_total: 库容向量，shape (H,)
        Q_in: 入库承诺向量，shape (H,)
        backlog: 原材料 backlog 轨迹，shape (H,)
        horizon: 规划视界
        
    Returns:
        Q_safe: shape (H,)
    """
    suffix_in = np.cumsum(Q_in[::-1])[::-1]  # shape (H,)
    Q_safe = C_total - backlog - suffix_in
    
    # 非负约束
    Q_safe = np.maximum(Q_safe, 0)
    
    return Q_safe


def compute_safe_sell_mask(
    I_output_now: float,
    Q_prod: np.ndarray,
    Q_out: np.ndarray,
    horizon: int
) -> np.ndarray:
    """计算每个交货日 delta 的最大安全卖出量.
    
    卖侧约束：在交货日 δ 能交付的最大成品数量，且不能影响后续已签合约。
    
    算法分两步：
    1. 计算每天的"净可交付量"（成品轨迹）：
       S[k] = I_output_now + Σ_{j=0}^{k} Q_prod[j] - Σ_{j=0}^{k} Q_out[j]
    2. 使用逆向累积最小值确保不挤爆后续交付：
       Q_safe_sell[δ] = min_{k>=δ} S[k]
    
    与买侧 Q_safe 对称：如果在 δ 卖出 q，会影响从 δ 到 H 的所有可交付量。
    
    Args:
        I_output_now: 当前成品库存
        Q_prod: 每日产能向量，shape (H,)
        Q_out: 已承诺的出库量向量，shape (H,)
        horizon: 规划视界
        
    Returns:
        Q_safe_sell: shape (H,)
    """
    # 步骤1: 计算成品轨迹 S[k]（每天的可交付量）
    # S[k] = I_output + Σ_{j=0}^{k} (Q_prod[j] - Q_out[j])
    net_flow = Q_prod - Q_out  # 每天净产出
    
    S = np.zeros(horizon, dtype=np.float32)
    if horizon > 0:
        cumulative_net = np.cumsum(net_flow)  # 前 k 天的累计净产出
        S = I_output_now + cumulative_net
    
    # 步骤2: 逆向累积最小值
    # Q_safe_sell[δ] = min_{k>=δ} S[k]
    reversed_S = S[::-1]
    reversed_cummin = np.minimum.accumulate(reversed_S)
    Q_safe_sell = reversed_cummin[::-1]
    
    # 非负约束
    Q_safe_sell = np.maximum(Q_safe_sell, 0)
    
    return Q_safe_sell


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
            L1Output: 包含 Q_safe, Q_safe_sell, B_free, time_mask 等
        """
        # 1. 获取库容向量
        C_total = get_capacity_vector(awi, self.horizon)
        
        # 2. 提取合约承诺
        Q_in, Q_out = extract_commitments(awi, self.horizon)
        
        # 3. 估计生产消耗（卖侧用满负荷）
        n_lines = getattr(awi.profile, 'n_lines', 1)
        Q_prod = np.full(self.horizon, n_lines, dtype=np.float32)
        
        # 4. 计算 backlog 轨迹（原材料，满载加工假设）
        # 重要：使用原材料库存（current_inventory_input），而非总库存
        # 因为采购入库的是原材料，库容约束针对原材料存储空间
        # 注意：Q_out 是成品出库，不影响原材料库存，故不参与此计算
        I_input_now = float(getattr(awi, 'current_inventory_input', 0) or 0)
        backlog = compute_inventory_trajectory(I_input_now, Q_in, n_lines, self.horizon)
        
        # 5. 计算安全买入量掩码 (H 维)
        Q_safe_h = compute_safe_buy_mask(C_total, Q_in, backlog, self.horizon)
        
        # 6. 扩展为 H+1 维，支持 δt ∈ {0, 1, ..., H}
        # δt = H 的处理：复用 H-1 的值
        Q_safe = np.zeros(self.horizon + 1, dtype=np.float32)
        Q_safe[:self.horizon] = Q_safe_h
        Q_safe[self.horizon] = Q_safe_h[-1] if len(Q_safe_h) > 0 else 0.0
        
        # 7. 计算卖侧安全量（成品可交付量）
        I_output_now = float(getattr(awi, 'current_inventory_output', 0) or 0)
        Q_safe_sell_h = compute_safe_sell_mask(I_output_now, Q_prod, Q_out, self.horizon)
        
        # 扩展为 H+1 维
        Q_safe_sell = np.zeros(self.horizon + 1, dtype=np.float32)
        Q_safe_sell[:self.horizon] = Q_safe_sell_h
        Q_safe_sell[self.horizon] = Q_safe_sell_h[-1] if len(Q_safe_sell_h) > 0 else 0.0
        
        # 8. 计算资金约束
        Payables = self._extract_payables(awi)
        wallet = getattr(awi, 'wallet', 0.0) or 0.0
        B_free = compute_budget_limit(wallet, Payables, self.reserve)
        
        # 9. 生成时间掩码 (用于 L3 的 Masked Softmax)，H+1 维
        # 根据角色选择对应的安全量
        if is_buying:
            time_mask = np.where(Q_safe >= self.min_tradable_qty, 0.0, -np.inf)
        else:
            time_mask = np.where(Q_safe_sell >= self.min_tradable_qty, 0.0, -np.inf)
        
        return L1Output(
            Q_safe=Q_safe,
            Q_safe_sell=Q_safe_sell,
            B_free=B_free,
            time_mask=time_mask,
            L_trajectory=backlog,
            C_total=C_total,
            Q_in=Q_in,
            I_input_now=I_input_now,
            n_lines=float(n_lines)
        )
    
    def _extract_payables(self, awi: "StdAWI") -> np.ndarray:
        """提取未来的应付款项.
        
        优先使用 AWI 的 supplies_cost/future_supplies_cost 接口。
        注意：StdAWI 不提供 signed_contracts 属性。
        """
        Payables = np.zeros(self.horizon, dtype=np.float32)
        t_current = awi.current_step
        
        # 辅助函数：求和 mapping 的值
        def sum_mapping(mapping) -> float:
            if mapping is None:
                return 0.0
            if isinstance(mapping, dict):
                return sum(float(v) for v in mapping.values() if v is not None)
            return 0.0
        
        # 辅助函数：填充未来值
        def fill_from_future(future_map, target: np.ndarray):
            if not isinstance(future_map, dict):
                return
            for step, per_partner in future_map.items():
                try:
                    delta = int(step) - t_current
                except Exception:
                    continue
                if 0 <= delta < len(target):
                    target[delta] += sum_mapping(per_partner)
        
        # 使用 AWI 接口获取应付款
        supplies_cost = getattr(awi, 'supplies_cost', None)
        future_supplies_cost = getattr(awi, 'future_supplies_cost', None)
        
        # 当日应付款 (delta=0)
        Payables[0] = sum_mapping(supplies_cost)
        
        # 未来应付款 (delta>0)
        fill_from_future(future_supplies_cost, Payables)
        
        return Payables
    
    def clip_action(
        self,
        action: Tuple[float, float, int],
        Q_safe: np.ndarray,
        B_free: float,
        is_buying: bool,
        min_price: float = 0.0,
        max_price: float = float('inf'),
        Q_safe_sell: Optional[np.ndarray] = None,
        time_range: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, float, int]:
        """裁剪动作到安全范围（仅裁剪数量）.
        
        所有最终动作必须经过此方法裁剪。
        
        Args:
            action: (quantity, price, delivery_time) 原始动作
            Q_safe: 买方剩余安全量向量，shape (H+1,)
            B_free: 可用资金（此处不裁剪预算）
            is_buying: 是否为买入角色
            min_price: 价格下界（此处不裁剪价格）
            max_price: 价格上界（此处不裁剪价格）
            Q_safe_sell: 卖方剩余安全量向量，shape (H+1,)
            
        Returns:
            (quantity, price, delivery_time): 裁剪后的动作
        """
        quantity, price, delivery_time = action
        
        delivery_time = int(delivery_time)

        if time_range is not None:
            min_t, max_t = time_range
            if delivery_time < int(min_t) or delivery_time > int(max_t):
                return (0, float(price), delivery_time)

        # 数量裁剪（不改时间，不改价格）
        if is_buying:
            max_qty = Q_safe[delivery_time] if 0 <= delivery_time < len(Q_safe) else 0.0
            quantity = min(quantity, max_qty)
        else:
            max_qty = Q_safe_sell[delivery_time] if Q_safe_sell is not None and 0 <= delivery_time < len(Q_safe_sell) else 0.0
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
    "compute_safe_sell_mask",
    "compute_budget_limit",
]
