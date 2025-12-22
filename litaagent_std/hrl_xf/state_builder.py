"""状态空间构建器 - 构建 HRL-XF 的混合张量组状态.

状态空间结构：
- x_static: (12,) 静态特征向量
- X_temporal: (H+1, 10) 时序状态张量，覆盖 δ∈{0..H}
  通道 6-7 拆分为采购/销售两条 price_diff，8-9 为买/卖压力
- x_role: (2,) 角色嵌入 (One-hot)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from scml.std import StdAWI


@dataclass
class StateDict:
    """HRL-XF 状态空间结构.
    
    Attributes:
        x_static: 静态特征向量，shape (12,)
        X_temporal: 时序状态张量，shape (H+1, 10)，覆盖 δ∈{0..H}
        x_role: 角色嵌入 (One-hot)，shape (2,)
    """
    x_static: np.ndarray      # (12,)
    X_temporal: np.ndarray    # (H+1, 10)
    x_role: np.ndarray        # (2,)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """转换为字典格式."""
        return {
            "x_static": self.x_static,
            "X_temporal": self.X_temporal,
            "x_role": self.x_role,
        }
    
    def add_batch_dim(self) -> Dict[str, np.ndarray]:
        """添加批次维度 (B=1)."""
        return {
            "x_static": self.x_static[np.newaxis, :],       # (1, 12)
            "X_temporal": self.X_temporal[np.newaxis, :, :], # (1, H+1, 10)
            "x_role": self.x_role[np.newaxis, :],           # (1, 2)
        }


class StateBuilder:
    """状态空间构建器.
    
    负责从 AWI 提取特征并构建标准化的状态张量。
    
    归一化说明：
    - balance: 使用首次调用时从 AWI 获取的 initial_balance
    - inventory: 使用经济容量 n_lines × n_steps 作为归一化基准
    - 如果首次调用前未设置，则使用默认值
    
    Args:
        horizon: 规划视界 H，默认 40
        initial_balance: 初始资金（用于归一化），None 表示从 AWI 自动获取
        max_inventory: 最大库存（用于归一化），None 表示使用经济容量
        max_price: 最大价格（用于归一化），默认 50.0
    """
    
    # 静态特征索引
    STATIC_FEATURES = [
        "balance_norm",      # 0
        "inventory_raw",     # 1
        "inventory_product", # 2
        "step_progress",     # 3
        "n_lines",           # 4
        "production_cost",   # 5
        "spot_price_in",     # 6
        "spot_price_out",    # 7
        "pending_buy_qty",   # 8
        "pending_sell_qty",  # 9
        "pending_buy_value", # 10
        "pending_sell_value" # 11
    ]
    
    # 时序特征通道索引
    TEMPORAL_CHANNELS = [
        "vol_in",         # 0: 到货量
        "vol_out",        # 1: 发货量
        "prod_plan",      # 2: 生产消耗
        "inventory_proj", # 3: 库存投影
        "capacity_free",  # 4: 自由库容
        "balance_proj",   # 5: 资金投影
        "price_diff_in",  # 6: 采购侧期货溢价
        "price_diff_out", # 7: 销售侧期货溢价
        "buy_pressure",   # 8: 买方压力
        "sell_pressure",  # 9: 卖方压力
    ]
    
    def __init__(
        self,
        horizon: int = 40,
        initial_balance: Optional[float] = None,
        max_inventory: Optional[float] = None,
        max_price: float = 50.0
    ):
        self.horizon = horizon
        self._initial_balance = initial_balance  # None 表示从 AWI 自动获取
        self._max_inventory = max_inventory      # None 表示使用经济容量
        self.max_price = max_price
        self._initialized = False
    
    def _init_normalization(self, awi: "StdAWI") -> None:
        """初始化归一化基准（首次调用时从 AWI 获取）."""
        # 初始资金：如果未指定，从 AWI 获取
        if self._initial_balance is None:
            self._initial_balance = getattr(awi, 'initial_balance', None)
            if self._initial_balance is None:
                self._initial_balance = getattr(awi, 'wallet', 10000.0) or 10000.0
        
        # 经济容量：n_lines × n_steps（可加工的最大量）
        if self._max_inventory is None:
            n_lines = getattr(awi.profile, 'n_lines', 1)
            n_steps = getattr(awi, 'n_steps', 100)
            self._max_inventory = float(n_lines * n_steps)
        
        self._initialized = True
    
    @property
    def initial_balance(self) -> float:
        """获取归一化用的初始资金."""
        return self._initial_balance if self._initial_balance else 10000.0
    
    @property
    def max_inventory(self) -> float:
        """获取归一化用的最大库存（经济容量）."""
        return self._max_inventory if self._max_inventory else 100.0
    
    def build(self, awi: "StdAWI", is_buying: bool) -> StateDict:
        """从 AWI 构建完整状态.
        
        Args:
            awi: Agent World Interface
            is_buying: 当前角色是否为买方
            
        Returns:
            StateDict: 包含 x_static, X_temporal, x_role
        """
        # 首次调用时初始化归一化基准
        if not self._initialized:
            self._init_normalization(awi)
        
        x_static = self._build_static(awi)
        X_temporal = self._build_temporal(awi)
        x_role = self._build_role(is_buying)
        
        return StateDict(
            x_static=x_static,
            X_temporal=X_temporal,
            x_role=x_role
        )
    
    def _build_static(self, awi: "StdAWI") -> np.ndarray:
        """构建静态特征向量 (12维).
        
        | 索引 | 特征名 | 计算方式 |
        |------|--------|----------|
        | 0 | balance_norm | B_t / B_initial |
        | 1 | inventory_raw | 原材料库存 |
        | 2 | inventory_product | 成品库存 |
        | 3 | step_progress | t / T_max |
        | 4 | n_lines | 生产线数量 |
        | 5 | production_cost | 单位生产成本 |
        | 6 | spot_price_in | 当前原材料市场价 |
        | 7 | spot_price_out | 当前成品市场价 |
        | 8 | pending_buy_qty | 未执行采购合约总量 |
        | 9 | pending_sell_qty | 未执行销售合约总量 |
        | 10 | pending_buy_value | 未执行采购合约总价值 |
        | 11 | pending_sell_value | 未执行销售合约总价值 |
        """
        x = np.zeros(12, dtype=np.float32)
        
        # 基础信息
        wallet = getattr(awi, 'wallet', 0.0) or 0.0
        t_current = awi.current_step
        t_max = awi.n_steps
        
        # 库存信息
        # 注意：使用 AWI 提供的 current_inventory_input/output 属性
        # 这些属性直接返回原材料和成品的库存量，无需通过 product ID 查询
        inventory_raw = float(getattr(awi, 'current_inventory_input', 0) or 0)
        inventory_product = float(getattr(awi, 'current_inventory_output', 0) or 0)
        
        # 市场价格
        input_product = self._get_input_product(awi)
        output_product = self._get_output_product(awi)
        trading_prices = getattr(awi, 'trading_prices', None)
        
        if trading_prices is not None and hasattr(trading_prices, '__getitem__'):
            spot_price_in = trading_prices[input_product] if input_product < len(trading_prices) else 10.0
            spot_price_out = trading_prices[output_product] if output_product < len(trading_prices) else 20.0
        else:
            spot_price_in = 10.0
            spot_price_out = 20.0
        
        # Profile 信息
        profile = awi.profile
        n_lines = getattr(profile, 'n_lines', 1)
        production_cost = getattr(profile, 'cost', 1.0)
        if hasattr(production_cost, '__len__'):
            production_cost = float(np.mean(production_cost))
        
        # 计算未执行合约统计
        buy_qty, buy_value, sell_qty, sell_value = self._compute_pending_contracts(awi)
        
        # 填充特征向量（带归一化）
        x[0] = np.clip(wallet / max(self.initial_balance, 1.0), 0.0, 2.0)
        x[1] = np.clip(inventory_raw / max(self.max_inventory, 1.0), 0.0, 1.0)
        x[2] = np.clip(inventory_product / max(self.max_inventory, 1.0), 0.0, 1.0)
        x[3] = t_current / max(t_max, 1)
        x[4] = np.clip(n_lines / 10.0, 0.0, 1.0)  # 假设最大10条生产线
        x[5] = np.clip(production_cost / max(self.max_price, 1.0), 0.0, 1.0)
        x[6] = np.clip(spot_price_in / max(self.max_price, 1.0), 0.0, 2.0)
        x[7] = np.clip(spot_price_out / max(self.max_price, 1.0), 0.0, 2.0)
        x[8] = np.clip(buy_qty / max(self.max_inventory, 1.0), 0.0, 1.0)
        x[9] = np.clip(sell_qty / max(self.max_inventory, 1.0), 0.0, 1.0)
        x[10] = np.clip(buy_value / max(self.initial_balance, 1.0), 0.0, 1.0)
        x[11] = np.clip(sell_value / max(self.initial_balance, 1.0), 0.0, 1.0)
        
        return x
    
    def _build_temporal(self, awi: "StdAWI") -> np.ndarray:
        """构建时序状态张量 ((H+1) × 10).
        
        覆盖 δ∈{0..H}，共 H+1 个时间点。
        
        通道定义：
        | 0 | vol_in | 到货量 |
        | 1 | vol_out | 发货量 |
        | 2 | prod_plan | 生产消耗 |
        | 3 | inventory_proj | 库存投影 |
        | 4 | capacity_free | 自由库容 |
        | 5 | balance_proj | 资金投影 |
        | 6 | price_diff_in | 采购侧期货溢价 |
        | 7 | price_diff_out | 销售侧期货溢价 |
        | 8 | buy_pressure | 买方压力（基于谈判与合约数据） |
        | 9 | sell_pressure | 卖方压力（基于谈判与合约数据） |
        """
        # 使用 H+1 覆盖 δ∈{0..H}
        temporal_len = self.horizon + 1
        X = np.zeros((temporal_len, 10), dtype=np.float32)
        
        t_current = awi.current_step
        agent_id = awi.agent.id
        
        # 提取合约承诺（按 delta 聚合）
        Q_in = np.zeros(temporal_len, dtype=np.float32)  # 买入量
        Q_out = np.zeros(temporal_len, dtype=np.float32) # 卖出量
        Payables = np.zeros(temporal_len, dtype=np.float32)
        Receivables = np.zeros(temporal_len, dtype=np.float32)
        
        # 用于 price_diff 计算的合约价格数据（分采购/销售）
        signed_buy_contracts_by_delta: Dict[int, list] = {}
        signed_sell_contracts_by_delta: Dict[int, list] = {}
        
        signed_contracts = getattr(awi, 'signed_contracts', []) or []
        
        for contract in signed_contracts:
            if getattr(contract, 'executed', False):
                continue
            
            agreement = getattr(contract, 'agreement', {}) or {}
            delivery_time = agreement.get('time', getattr(contract, 'time', None))
            
            if delivery_time is None:
                continue
                
            delta = delivery_time - t_current
            
            if 0 <= delta <= self.horizon:  # 包含 δ=H
                quantity = agreement.get('quantity', 0)
                unit_price = agreement.get('unit_price', 0)
                
                annotation = getattr(contract, 'annotation', {}) or {}
                seller_id = annotation.get('seller')
                
                if seller_id != agent_id:
                    # 我是买方 -> 入库，付款
                    Q_in[delta] += quantity
                    Payables[delta] += quantity * unit_price
                    if delta not in signed_buy_contracts_by_delta:
                        signed_buy_contracts_by_delta[delta] = []
                    signed_buy_contracts_by_delta[delta].append((quantity, unit_price))
                else:
                    # 我是卖方 -> 出库，收款
                    Q_out[delta] += quantity
                    Receivables[delta] += quantity * unit_price
                    if delta not in signed_sell_contracts_by_delta:
                        signed_sell_contracts_by_delta[delta] = []
                    signed_sell_contracts_by_delta[delta].append((quantity, unit_price))
        
        # 生产消耗（保守估计：满负荷）
        n_lines = getattr(awi.profile, 'n_lines', 1)
        Q_prod = np.full(temporal_len, n_lines, dtype=np.float32)
        
        # 生产成本（从 profile 获取）
        production_cost = float(getattr(awi.profile, 'cost', 0.0) or 0.0)
        
        # 库存投影
        # 注意：使用原材料库存作为库存投影的起点，因为采购入库的是原材料
        I_now = float(getattr(awi, 'current_inventory_input', 0) or 0)
        net_flow = Q_in - Q_out - Q_prod
        I_proj = I_now + np.cumsum(net_flow)
        
        # 库容投影（经济容量：基于剩余可加工天数）
        # 注意：SCML 不提供 storage_capacity，使用 n_lines × remaining_days 作为经济容量
        C_total = self._get_capacity_vector(awi, length=temporal_len)
        C_free = C_total - I_proj
        
        # 资金投影（含生产成本）
        wallet = getattr(awi, 'wallet', 0.0) or 0.0
        # 现金流 = 收款 - 付款 - 生产成本
        cash_flow = Receivables - Payables - Q_prod * production_cost
        B_proj = wallet + np.cumsum(cash_flow)
        
        # 获取现货价格（采购侧：input_product，销售侧：output_product）
        input_product = self._get_input_product(awi)
        output_product = self._get_output_product(awi)
        trading_prices = getattr(awi, 'trading_prices', None)
        
        # trading_prices 是 numpy 数组，按产品索引访问
        if trading_prices is not None and hasattr(trading_prices, '__getitem__'):
            try:
                spot_price_in = float(trading_prices[input_product]) if input_product < len(trading_prices) else 10.0
            except (IndexError, TypeError):
                spot_price_in = 10.0
            try:
                spot_price_out = float(trading_prices[output_product]) if output_product < len(trading_prices) else 20.0
            except (IndexError, TypeError):
                spot_price_out = 20.0
        else:
            spot_price_in = 10.0
            spot_price_out = 20.0
        
        # 计算采购/销售两侧的压力与价格趋势
        # buy_pressure: 买方需求强度 → 我在 output market 卖出 → 用 Q_out + spot_price_out
        # sell_pressure: 卖方供给强度 → 我在 input market 买入 → 用 Q_in + spot_price_in
        buy_pressure = self._compute_buy_pressure(awi, Q_out, C_total, spot_price_out)
        sell_pressure = self._compute_sell_pressure(awi, Q_in, C_total, spot_price_in)
        price_diff_in = self._compute_price_diff_market(
            awi=awi,
            signed_contracts_by_delta=signed_buy_contracts_by_delta,
            spot_price=spot_price_in,
            active_offers_source="buy",
            temporal_len=temporal_len,
        )
        price_diff_out = self._compute_price_diff_market(
            awi=awi,
            signed_contracts_by_delta=signed_sell_contracts_by_delta,
            spot_price=spot_price_out,
            active_offers_source="sell",
            temporal_len=temporal_len,
        )
        
        # 填充张量（带归一化）
        X[:, 0] = Q_in / max(self.max_inventory, 1.0)
        X[:, 1] = Q_out / max(self.max_inventory, 1.0)
        X[:, 2] = Q_prod / max(self.max_inventory, 1.0)
        X[:, 3] = np.clip(I_proj / max(self.max_inventory, 1.0), -1.0, 2.0)
        X[:, 4] = np.clip(C_free / max(self.max_inventory, 1.0), -1.0, 2.0)
        X[:, 5] = np.clip(B_proj / max(self.initial_balance, 1.0), -1.0, 2.0)
        X[:, 6] = np.clip(price_diff_in / max(self.max_price, 1.0), -1.0, 1.0)
        X[:, 7] = np.clip(price_diff_out / max(self.max_price, 1.0), -1.0, 1.0)
        X[:, 8] = buy_pressure  # 已经是 [0, 1] 范围
        X[:, 9] = sell_pressure  # 已经是 [0, 1] 范围
        
        return X
    
    def _build_role(self, is_buying: bool) -> np.ndarray:
        """构建 L3 谈判角色嵌入 (One-hot).
        
        注意：这是 L3 层的谈判角色，表示当前这个具体谈判的方向。
        与 L2 层的 x_role（Multi-Hot 谈判能力）含义不同：
        
        L3 角色 (本方法，One-hot，每次谈判):
            [1, 0] = Buyer - 当前谈判中我方是买家
            [0, 1] = Seller - 当前谈判中我方是卖家
        
        L2 x_role (Multi-Hot，全局能力):
            [can_buy, can_sell] - 代理在供应链中的谈判能力
            [1, 1] = 中间层，买卖都需谈判
            [0, 1] = 第一层，只能谈判销售（采购外生）
            [1, 0] = 最后层，只能谈判采购（销售外生）
        """
        if is_buying:
            return np.array([1.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 1.0], dtype=np.float32)
    
    def _get_input_product(self, awi: "StdAWI") -> int:
        """获取输入产品 ID."""
        input_product = awi.my_input_product
        if hasattr(input_product, '__len__') and not isinstance(input_product, (str, int)):
            return input_product[0]
        return input_product
    
    def _get_output_product(self, awi: "StdAWI") -> int:
        """获取输出产品 ID."""
        output_product = awi.my_output_product
        if hasattr(output_product, '__len__') and not isinstance(output_product, (str, int)):
            return output_product[0]
        return output_product
    
    def _get_capacity_vector(self, awi: "StdAWI", length: Optional[int] = None) -> np.ndarray:
        """
        获取经济产能向量。
        
        注意：negmas/SCML 标准世界不存在 storage_capacity 属性，
        采用"经济容量" = n_lines × 剩余天数 作为产能约束。
        
        Args:
            awi: 代理世界接口
            length: 向量长度，默认为 horizon+1（覆盖 δ∈{0..H}）
            
        Returns:
            np.ndarray: 形状 (length,) 的经济产能向量
        """
        if length is None:
            length = self.horizon + 1
            
        # negmas/SCML 无 storage_capacity，使用经济容量
        n_lines = getattr(awi.profile, 'n_lines', 1)
        t_current = awi.current_step
        t_max = awi.n_steps
        
        C_total = np.zeros(length, dtype=np.float32)
        for k in range(length):
            remaining_days = t_max - (t_current + k)
            C_total[k] = n_lines * max(0, remaining_days)
        
        return C_total
    
    def _compute_buy_pressure(
        self, 
        awi: "StdAWI", 
        signed_sell_by_delta: np.ndarray,
        capacity: np.ndarray,
        spot_price_out: float = 10.0
    ) -> np.ndarray:
        """计算输出市场买方需求强度（价格加权）.
        
        买方压力 = 加权需求量 / 经济容量
        高于现货价的买单权重更大（表示更强的需求意愿）。
        
        语义：我在输出市场 (output market) 卖出产品，买方需求越强对我越有利。
        第 t+k 天买方对商品的需求强度。
        值越大表示"买方多、缺货风险高、可抬价"。
        
        数据来源：
            1. signed_sell_by_delta (Q_out): 已签销售合约按 delta 聚合的数量
            2. current_sell_offers: StdAWI 提供的当前卖出谈判最新出价
               格式为 dict[partner_id, Outcome]，Outcome = (quantity, time, unit_price)
            3. 若 current_sell_offers 不可用，退化为仅基于已签合约计算
            
        备选方案（如需更详细的谈判历史）：
            可从 awi.current_negotiation_details["sell"] 获取 NegotiationDetails，
            通过 details.nmi.state.current_offer 访问出价
        
        Args:
            awi: 代理世界接口
            signed_sell_by_delta: 已签销售合约按 delta 聚合的数量向量 (Q_out)
            capacity: 经济容量向量
            spot_price_out: 输出市场现货价格（用于价格加权）
            
        Returns:
            np.ndarray: 形状 (len,) 的买方压力向量，范围 [0, 1]
            
        TODO: 
            - 考虑结合当前库存/需求规模动态调整归一化分母
            - 添加 EMA 历史平滑以减少单步波动
        """
        temporal_len = len(signed_sell_by_delta)
        t_current = awi.current_step
        weighted_demand = signed_sell_by_delta.copy().astype(np.float32)
        
        # 从当前卖出谈判中获取活跃出价
        # （我方作为卖方，收到买方请求 -> 代表输出市场买方需求）
        # 高价买单权重更大（以 spot_price_out 为基准）
        # 注意：若 current_sell_offers 不可用，压力将仅基于已签合约计算
        try:
            current_sell_offers = getattr(awi, 'current_sell_offers', None)
            if current_sell_offers is None:
                current_sell_offers = {}
        except Exception:
            current_sell_offers = {}
        
        for partner_id, offer in current_sell_offers.items():
            if offer is None:
                continue
            try:
                # Outcome 格式: (quantity, time, unit_price)
                quantity = float(offer[0])
                delivery_time = int(offer[1])
                unit_price = float(offer[2])
                delta = delivery_time - t_current
                
                if 0 <= delta < temporal_len:
                    # 价格加权：高于输出市场现货价的买单权重 > 1
                    price_weight = max(0.5, min(2.0, unit_price / max(spot_price_out, 1.0)))
                    weighted_demand[delta] += quantity * price_weight
            except (IndexError, TypeError, ValueError):
                continue
        
        # 归一化
        # TODO: 考虑在大容量场景下添加平滑常数避免过度稀释
        safe_cap = np.maximum(capacity, 1.0)
        buy_pressure = np.clip(weighted_demand / safe_cap, 0.0, 1.0)
        
        return buy_pressure.astype(np.float32)
    
    def _compute_sell_pressure(
        self, 
        awi: "StdAWI", 
        signed_buy_by_delta: np.ndarray,
        capacity: np.ndarray,
        spot_price_in: float = 10.0
    ) -> np.ndarray:
        """计算输入市场卖方供给强度（价格加权）.
        
        卖方压力 = 加权供给量 / 经济容量
        低于现货价的卖单权重更大（表示更强的抛售意愿）。
        
        语义：我在输入市场 (input market) 采购原料，卖方供给越强对我越有利。
        第 t+k 天卖方的供给强度。
        值越大表示"供给多、价格承压、可压价"。
        
        数据来源：
            1. signed_buy_by_delta (Q_in): 已签采购合约按 delta 聚合的数量
            2. current_buy_offers: StdAWI 提供的当前买入谈判最新出价
               格式为 dict[partner_id, Outcome]，Outcome = (quantity, time, unit_price)
            3. 若 current_buy_offers 不可用，退化为仅基于已签合约计算
            
        备选方案（如需更详细的谈判历史）：
            可从 awi.current_negotiation_details["buy"] 获取 NegotiationDetails，
            通过 details.nmi.state.current_offer 访问出价
        
        Args:
            awi: 代理世界接口
            signed_buy_by_delta: 已签采购合约按 delta 聚合的数量向量 (Q_in)
            capacity: 经济容量向量
            spot_price_in: 输入市场现货价格（用于价格加权）
            
        Returns:
            np.ndarray: 形状 (len,) 的卖方压力向量，范围 [0, 1]
            
        TODO:
            - 考虑结合当前库存水平动态调整归一化分母
            - 添加 EMA 历史平滑以减少单步波动
        """
        temporal_len = len(signed_buy_by_delta)
        t_current = awi.current_step
        weighted_supply = signed_buy_by_delta.copy().astype(np.float32)
        
        # 从当前买入谈判中获取活跃出价
        # （我方作为买方，收到卖方报价 -> 代表输入市场卖方供给）
        # 低价卖单权重更大（以 spot_price_in 为基准）
        # 注意：若 current_buy_offers 不可用，压力将仅基于已签合约计算
        try:
            current_buy_offers = getattr(awi, 'current_buy_offers', None)
            if current_buy_offers is None:
                current_buy_offers = {}
        except Exception:
            current_buy_offers = {}
        
        for partner_id, offer in current_buy_offers.items():
            if offer is None:
                continue
            try:
                # Outcome 格式: (quantity, time, unit_price)
                quantity = float(offer[0])
                delivery_time = int(offer[1])
                unit_price = float(offer[2])
                delta = delivery_time - t_current
                
                if 0 <= delta < temporal_len:
                    # 价格加权：低于输入市场现货价的卖单权重 > 1
                    # 使用倒数关系：价格越低，权重越高
                    price_ratio = max(spot_price_in, 1.0) / max(unit_price, 1.0)
                    price_weight = max(0.5, min(2.0, price_ratio))
                    weighted_supply[delta] += quantity * price_weight
            except (IndexError, TypeError, ValueError):
                continue
        
        # 归一化
        # TODO: 考虑在大容量场景下添加平滑常数避免过度稀释
        safe_cap = np.maximum(capacity, 1.0)
        sell_pressure = np.clip(weighted_supply / safe_cap, 0.0, 1.0)
        
        return sell_pressure.astype(np.float32)
    
    def _compute_price_diff_market(
        self,
        awi: "StdAWI",
        signed_contracts_by_delta: Dict[int, list],
        spot_price: float,
        temporal_len: Optional[int] = None,
        active_offers_source: str = "buy",
    ) -> np.ndarray:
        """计算单个市场的价格趋势向量 (期货溢价/贴水).
        
        信号来源优先级：
        1. 已签成交 VWAP（按交货日聚合）
        2. 正在谈判的最新出价中位数（按交货日聚合）
        3. 回退到现货价（无数据时保持平坦）
        
        Args:
            awi: 代理世界接口
            signed_contracts_by_delta: 已签合约按 delta 聚合的 {delta: [(qty, price), ...]}
            spot_price: 当前现货价格（采购用 input，销售用 output）
            temporal_len: 输出向量长度，默认为 horizon+1
            active_offers_source: "buy" 使用 current_buy_offers；"sell" 使用 current_sell_offers
        """
        if temporal_len is None:
            temporal_len = self.horizon + 1
            
        t_current = awi.current_step
        price_diff = np.zeros(temporal_len, dtype=np.float32)
        
        # 权重配置（与离线重建保持一致）
        W_SIGNED = 0.6   # 已签成交权重
        W_ACTIVE = 0.3   # 活跃谈判权重
        W_SPOT = 0.1     # 现货回退权重
        
        # 获取活跃谈判数据
        try:
            if active_offers_source == "sell":
                active_offers = getattr(awi, 'current_sell_offers', None) or {}
            else:
                active_offers = getattr(awi, 'current_buy_offers', None) or {}
        except Exception:
            active_offers = {}
        
        for k in range(temporal_len):
            vwap_signed = None
            mid_active = None
            
            # 1) 已签合约 VWAP
            if k in signed_contracts_by_delta:
                contracts = signed_contracts_by_delta[k]
                if contracts:
                    try:
                        total_value = sum(qty * price for qty, price in contracts)
                        total_qty = sum(qty for qty, price in contracts)
                        if total_qty > 0:
                            vwap_signed = total_value / total_qty
                    except (TypeError, ValueError):
                        pass
            
            # 2) 活跃谈判中位价
            active_prices = []
            for offer in active_offers.values():
                if offer is None:
                    continue
                try:
                    delivery_time = int(offer[1])
                    unit_price = float(offer[2])
                    if delivery_time - t_current == k:
                        active_prices.append(unit_price)
                except (IndexError, TypeError, ValueError):
                    continue
            
            if active_prices:
                mid_active = float(np.median(active_prices))
            
            # 3) 融合计算
            if vwap_signed is not None and mid_active is not None:
                p_future = W_SIGNED * vwap_signed + W_ACTIVE * mid_active + W_SPOT * spot_price
            elif vwap_signed is not None:
                p_future = (W_SIGNED + W_ACTIVE/2) * vwap_signed + (W_SPOT + W_ACTIVE/2) * spot_price
            elif mid_active is not None:
                p_future = (W_ACTIVE + W_SIGNED/2) * mid_active + (W_SPOT + W_SIGNED/2) * spot_price
            else:
                p_future = spot_price
            
            price_diff[k] = p_future - spot_price
        
        return price_diff
    
    def _compute_pending_contracts(
        self, awi: "StdAWI"
    ) -> Tuple[float, float, float, float]:
        """计算未执行合约统计.
        
        Returns:
            (buy_qty, buy_value, sell_qty, sell_value)
        """
        buy_qty = 0.0
        buy_value = 0.0
        sell_qty = 0.0
        sell_value = 0.0
        
        agent_id = awi.agent.id
        signed_contracts = getattr(awi, 'signed_contracts', []) or []
        
        for contract in signed_contracts:
            if getattr(contract, 'executed', False):
                continue
            
            agreement = getattr(contract, 'agreement', {}) or {}
            quantity = agreement.get('quantity', 0)
            unit_price = agreement.get('unit_price', 0)
            
            annotation = getattr(contract, 'annotation', {}) or {}
            seller_id = annotation.get('seller')
            
            if seller_id != agent_id:
                # 我是买方
                buy_qty += quantity
                buy_value += quantity * unit_price
            else:
                # 我是卖方
                sell_qty += quantity
                sell_value += quantity * unit_price
        
        return buy_qty, buy_value, sell_qty, sell_value


__all__ = [
    "StateDict",
    "StateBuilder",
]
