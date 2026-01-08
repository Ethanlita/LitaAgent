 # HRL-XF 实施规范文档

> **生成日期**：2025年12月13日  
> **文档版本**：1.0  
> **基于设计文档**：
> - `HRL-X 期货市场适应性重构.md`
> - `HRL-XF 期货市场代理重构.md`

> **更新说明（2026-01）**：
> - 在线阶段先跑 **IPPO**：仅训练 L3，L2/L4 冻结，显式设 `α=0`
> - **L1 仅输出 mask**（`time_mask/Q_safe/Q_safe_sell/B_free`），不再输出 baseline
> - **L3 输出完整 AOP 动作**（`op/time/price/qty`），提供采样、logprob、entropy、value
> - **L4 = GlobalMonitor + GlobalCoordinator**，`α` 仅作为 L3 输入，不再做调制/动态预留
> - **MAPPO/集中式 critic** 保留为远期扩展
> - 若本文件后续章节与上述冲突，以此更新为准

---

## 1. 概述

### 1.1 HRL-XF 与 HRL-X 的核心区别

HRL-XF (Hybrid Residual Learner - Extended Framework, **Futures Edition**) 是针对 SCML 2025 期货市场环境的全面重构版本。与原 HRL-X 的关键区别：

| 维度 | HRL-X (现货) | HRL-XF (期货) |
|------|-------------|---------------|
| **动作空间** | $(q, p)$ | AOP: `op` + $(q, p, \delta_t)$ |
| **状态空间** | 扁平向量 | 混合张量组（含时序） |
| **L1 约束** | 静态库容检查 | 时序 ATP 算法 |
| **L2 输出** | 8维（仅买入） | 16维（买卖对称） |
| **L3 架构** | DT (2分量) | DT (AOP: op/time/price/qty，可采样分布) |
| **L4 机制** | 简单注意力 | Monitor + Coordinator（`α` 仅输入，不调制/动态预留） |

### 1.2 设计决策确认记录

以下决策已获用户确认：

1. **$C_{total}$ 定义**：统一使用经济容量 $C_{total}[k] = n\_lines \times (T_{max} - (t+k))$
2. **原材料 backlog 计算**：满载加工假设，$B[0]=I_{input}$，$B[k+1]=\\max(0, B[k]+Q_{in}[k]-n_{lines})$；$Q_{out}$ 是成品出库，不影响原材料库存
3. **L2 输出维度**：16维（4桶 × 4分量：$Q_{buy}, P_{buy}, Q_{sell}, P_{sell}$）
4. **L4 实现方式**：Transformer Encoder + 时间偏置掩码
5. **$Q_{safe}[\delta]$ 公式**：$Q_{safe}[\\delta]=\\max\\big(0, C_{total}[\\delta]-B[\\delta]-\\sum_{k=\\delta}^{H-1}Q_{in}[k]\\big)$
6. **时序特征维度**：10维（拆分采购/销售 price_diff，含买卖压力）
7. **角色嵌入**：L2-L4 全部需要
8. **状态空间组成**：$\{x_{static}, X_{temporal}, x_{role}\}$（无独立 $x_{market}$，市场信息分布于前两者）
9. **$\delta_t$ 取值范围**：$\{0, 1, ..., H\}$ 共 $H+1$ 个值
10. **L1 合约提取范围**：优先使用 AWI 的 `supplies/sales/future_supplies/future_sales`（必要时回退 `signed_contracts`）

---

## 2. 数学符号定义

### 2.1 基础符号

| 符号 | 含义 | 来源/计算方式 |
|------|------|---------------|
| $t$ | 当前仿真步（天） | `awi.current_step` |
| $T_{max}$ | 总仿真步数 | `awi.n_steps` |
| $H$ | 规划视界（天数） | 常量，建议 40 |
| $n\_lines$ | 生产线数量 | `awi.profile.n_lines` |
| $B_t$ | 当前现金余额 | `awi.wallet` 或 `awi.state.balance` |
| $I_{now}$ | 当前物理库存 | `awi.current_inventory` |

### 2.2 时序符号

| 符号 | 含义 | 维度 |
|------|------|------|
| $Q_{in}[k]$ | 第 $t+k$ 天的入库承诺量 | $(H,)$ |
| $Q_{out}[k]$ | 第 $t+k$ 天的出库承诺量 | $(H,)$ |
| $Q_{prod}[k]$ | 第 $t+k$ 天的预计生产消耗/产出 | $(H,)$ |
| $L(k)$ | 第 $t+k$ 天的库存水位投影 | 标量（某一天）或向量 |
| $C_{total}[k]$ | 第 $t+k$ 天的库容上限 | $(H,)$ |
| $C_{free}[k]$ | 第 $t+k$ 天的自由库容 | $(H,)$ |
| $Q_{safe}[\delta]$ | 交货日为 $\delta$ 时的最大安全买入量 | $(H+1,)$，$\delta \in \{0..H\}$ |

**注**：$Q_{safe}$、`time_mask` 等 L1 输出为 $(H+1,)$ 维，以支持 $\delta_t \in \{0, 1, ..., H\}$。其中 $\delta_t = H$ 的值沿用 $\delta_t = H-1$ 的估计（因为 Payables/Commitments 按 $H$ 天计算）。

---

## 3. 状态空间详细定义 ($\mathcal{S}^+$)

### 3.1 混合张量组结构

```python
S_t = {
    "x_static": Tensor[B, 12],      # 静态特征向量
    "X_temporal": Tensor[B, H+1, 10],  # 时序状态张量（10维特征通道，δ∈[0,H]）
    "x_role": Tensor[B, 2],         # 角色 Multi-Hot [can_buy, can_sell]
}
```

**注**：市场信息已分布于 `x_static`（当前现货价格）和 `X_temporal`（期货溢价、买卖压力）中，无需独立的 `x_market` 组件。

### 3.2 静态状态向量 `x_static` (12维)

| 索引 | 特征名 | 计算方式 | 归一化 |
|------|--------|----------|--------|
| 0 | `balance_norm` | $B_t / B_{initial}$ | [0, 2] |
| 1 | `inventory_raw` | 原材料库存 | [0, 1] |
| 2 | `inventory_product` | 成品库存 | [0, 1] |
| 3 | `step_progress` | $t / T_{max}$ | [0, 1] |
| 4 | `n_lines` | 生产线数量 | 归一化 |
| 5 | `production_cost` | 单位生产成本 | 归一化 |
| 6 | `spot_price_in` | 当前原材料市场价 | 归一化 |
| 7 | `spot_price_out` | 当前成品市场价 | 归一化 |
| 8 | `pending_buy_qty` | 未执行的采购合约总量 | 归一化 |
| 9 | `pending_sell_qty` | 未执行的销售合约总量 | 归一化 |
| 10 | `pending_buy_value` | 未执行采购合约总价值 | 归一化 |
| 11 | `pending_sell_value` | 未执行销售合约总价值 | 归一化 |

### 3.3 时序状态张量 `X_temporal` (H+1 × 10)

覆盖 $\delta \in [0, H]$ 共 $H+1$ 个时间点，特征通道定义如下：

| 通道 | 特征名 | 公式/说明 |
|------|--------|----------|
| 0 | `vol_in` | $Q_{in}[k]$ = 第 $t+k$ 天到货的采购量（已签署合约） |
| 1 | `vol_out` | $Q_{out}[k]$ = 第 $t+k$ 天发货的销售量（已签署合约） |
| 2 | `prod_plan` | $Q_{prod}[k]$ = 预计生产消耗（保守估计） |
| 3 | `inventory_proj` | $I_{proj}[k] = I_{now} + \sum_{j=0}^{k}(Q_{in}[j] - Q_{prod}[j])$（原材料库存投影） |
| 4 | `capacity_free` | $C_{free}[k] = C_{total}[k] - I_{proj}[k]$ |
| 5 | `balance_proj` | $B_{proj}[k] = B_t + \sum_{j=0}^{k}(Receivables[j] - Payables[j] - Q_{prod}[j]·cost)$ |
| 6 | `price_diff_in` | 采购侧期货溢价：$P^{buy}_{future}[k] - P^{buy}_{spot}$ |
| 7 | `price_diff_out` | 销售侧期货溢价：$P^{sell}_{future}[k] - P^{sell}_{spot}$ |
| 8 | `buy_pressure` | 买方需求压力（价格加权） |
| 9 | `sell_pressure` | 卖方供给压力（价格加权） |

**通道 6-9 计算说明**：

- **price_diff_in / price_diff_out**  
- 信号来源优先级：① 已签成交 VWAP（按交货日聚合）；② 正在谈判的活跃报价**轮次衰减加权均值**；③ 回退现货价  
  - 融合：`P_future = w_signed*VWAP + w_active*avg + w_spot*P_spot`（权重 0.6/0.3/0.1）  
  - 输出：`price_diff = P_future - P_spot`（分别使用 `spot_price_in`、`spot_price_out`）
- **buy_pressure[k]**（买方需求强度）  
  `demand[k] = signed_buy[k] + active_sell_offers[k]·weight·price_weight`，再除以经济容量 `C_total[k]` 并裁剪到 `[0,1]`
- **sell_pressure[k]**（卖方供给强度）  
  `supply[k] = signed_sell[k] + active_buy_offers[k]·weight·price_weight`，再除以 `C_total[k]` 并裁剪到 `[0,1]`

**设计说明**：
- 现货价格在 `x_static` 中提供（`spot_price_in`/`spot_price_out`），时序张量只存储期货溢价/贴水
- 买卖压力分离，便于 L3 根据谈判方向选择性关注

### 3.4 角色嵌入 `x_role`

#### L2 层：Multi-Hot 谈判能力编码

根据 SCML 2025 Standard 规则，代理在供应链中的位置决定了其谈判能力：

| 位置 | 采购 | 销售 | x_role | 说明 |
|------|------|------|--------|------|
| 第一层 | 外生 (SELLER) | 需谈判 | `[0, 1]` | 只能谈判销售 |
| 中间层 | 需谈判 | 需谈判 | `[1, 1]` | 买卖都需谈判 |
| 最后层 | 需谈判 | 外生 (BUYER) | `[1, 0]` | 只能谈判采购 |

```python
x_role = [can_negotiate_buy, can_negotiate_sell]  # Multi-Hot
```

**设计理由**：
- Multi-Hot 编码比 One-Hot 更适合表示两个独立能力（买/卖不是互斥的）
- `[1, 1]` 明确表示"两种能力都有"，而 `[0.5, 0.5]` 语义模糊
- 网络可学习：当 `dim[0]=1` 时关注 `Q_buy, P_buy`；当 `dim[1]=1` 时关注 `Q_sell, P_sell`

#### L3/L4 层：One-Hot 谈判角色编码

L3/L4 处理具体谈判，每个谈判有明确的买/卖方向：

- **One-hot 编码**：`[1, 0]` = Buyer（当前谈判中是买家）, `[0, 1]` = Seller（当前谈判中是卖家）
- **或可学习嵌入**：`Embedding(input_dim=2, output_dim=d_role)`

---

## 4. 动作空间详细定义 ($\mathcal{A}^+$)

### 4.1 完整动作元组

```python
a_offer = (q, p, delta_t)
```

| 分量 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `q` | 连续 | $[0, Q_{max}]$ | 数量 |
| `p` | 连续 | $[P_{min}, P_{max}]$ | 单价 |
| `delta_t` | 离散 | $\{0, 1, ..., H\}$ | 相对交货时间 |

### 4.2 为什么 $\delta_t$ 是离散分类

1. **语义稳定性**：$\delta_t = 3$ 总是表示"3天后"，不受绝对时间影响
2. **多峰分布**：实际谈判中，"明天"和"下周"可能都合理，中间无意义
3. **便于 Masking**：可以直接对不安全的时间点应用 $-\infty$ 掩码

---

## 5. L1 安全护盾层 - 完整实现规范

### 5.1 核心职责

L1 是**确定性规则层**，不含可训练参数。其职责：
1. 计算时序库容约束 $Q_{safe}[\delta]$
2. 计算资金约束 $B_{free}$
3. 生成动作掩码供 L3 使用
4. 提供基准动作 $a_{base}$

### 5.2 $C_{total}$ 的获取逻辑

```python
def get_capacity_vector(awi, horizon: int) -> np.ndarray:
    """
    获取未来 H 天的库容上限向量。
    
    Returns:
        C_total: shape (H,)
    """
    # 尝试使用 API 提供的静态容量
    # 经济容量：基于剩余天数 × 日产能
    # 逻辑：超出剩余可加工天数的原材料会被浪费
    n_lines = awi.profile.n_lines
    t_current = awi.current_step
    t_max = awi.n_steps
    t_max = awi.n_steps
    
    C_total = np.zeros(horizon, dtype=np.float32)
    for k in range(horizon):
        remaining_days = t_max - (t_current + k)
        C_total[k] = n_lines * max(0, remaining_days)
    
    return C_total
```

### 5.3 合约承诺量的提取

```python
def extract_commitments(awi, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 AWI 提取未来 H 天的入库和出库承诺。
    
    优先使用 AWI 的 supplies/sales/future_supplies/future_sales 接口，
    与 tracker_mixin 保持一致；必要时回退 signed_contracts。
    
    Returns:
        Q_in: shape (H,) - 每天的入库量
        Q_out: shape (H,) - 每天的出库量
    """
    Q_in = np.zeros(horizon, dtype=np.float32)
    Q_out = np.zeros(horizon, dtype=np.float32)
    
    t_current = awi.current_step
    
    supplies = getattr(awi, 'supplies', None)
    sales = getattr(awi, 'sales', None)
    future_supplies = getattr(awi, 'future_supplies', None)
    future_sales = getattr(awi, 'future_sales', None)
    has_awi_future = any(x is not None for x in (supplies, sales, future_supplies, future_sales))
    
    if has_awi_future:
        def sum_mapping(mapping):
            if mapping is None:
                return 0.0
            if isinstance(mapping, dict):
                return sum(float(v) for v in mapping.values() if v is not None)
            return 0.0
        
        def fill_from_future(future_map, target):
            if not isinstance(future_map, dict):
                return
            for step, per_partner in future_map.items():
                try:
                    delta = int(step) - t_current
                except Exception:
                    continue
                if 0 <= delta < len(target):
                    target[delta] += sum_mapping(per_partner)
        
        Q_in[0] = sum_mapping(supplies)
        Q_out[0] = sum_mapping(sales)
        fill_from_future(future_supplies, Q_in)
        fill_from_future(future_sales, Q_out)
        return Q_in, Q_out
    
    # 回退：使用 signed_contracts（兼容旧版本 negmas）
    signed_contracts = getattr(awi, 'signed_contracts', []) or []
    
    for contract in signed_contracts:
        if getattr(contract, 'executed', False):
            continue
        
        agreement = getattr(contract, 'agreement', {}) or {}
        delivery_time = agreement.get('time', getattr(contract, 'time', None))
        
        if delivery_time is None:
            continue
        
        delta = delivery_time - t_current
        if 0 <= delta < horizon:
            quantity = agreement.get('quantity', 0)
            annotation = getattr(contract, 'annotation', {}) or {}
            if annotation.get('seller') != awi.agent.id:
                Q_in[delta] += quantity
            else:
                Q_out[delta] += quantity
    
    return Q_in, Q_out
```

### 5.4 原材料 backlog 轨迹 $B(\tau)$

```python
def compute_inventory_trajectory(
    I_now: float,
    Q_in: np.ndarray,
    n_lines: float,
    horizon: int
) -> np.ndarray:
    """
    计算未来 H 天的原材料 backlog 轨迹（满载加工假设）。
    
    递推：
    B[0] = I_now
    B[k+1] = max(0, B[k] + Q_in[k] - n_lines)
    
    纯原材料模型：Q_out 是成品出库，不影响原材料库存
    
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
```

### 5.5 针对特定交货日的最大安全买入量 $Q_{safe}[\delta]$

```python
def compute_safe_buy_mask(
    C_total: np.ndarray,
    Q_in: np.ndarray,
    backlog: np.ndarray,
    horizon: int
) -> np.ndarray:
    """
    计算每个交货日 delta 的最大安全买入量。
    
    公式：
    Q_safe[δ] = max(0, C_total[δ] - backlog[δ] - Σ_{k=δ}^{H-1} Q_in[k])
    
    Returns:
        Q_safe: shape (H,)
    """
    suffix_in = np.cumsum(Q_in[::-1])[::-1]
    Q_safe = C_total - backlog - suffix_in
    
    # 非负约束
    Q_safe = np.maximum(Q_safe, 0)
    
    return Q_safe
```

### 5.5.1 （已弃用）动态预留后的 Q_safe 更新

> 当前方案已移除“动态预留/顺序裁剪”机制，L4 不再参与裁剪或重算 Q_safe。本节仅作历史参考。

当 L4 批次规划器为某个线程（交货日 $\delta_t$）预留 $q$ 单位时，需要更新 `Q_in` 并重算 Q_safe：

```python
def recompute_q_safe_after_reservation(
    Q_in: np.ndarray,
    delta_t: int,
    reserved_qty: float,
    I_input_now: float,
    n_lines: float,
    C_total: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    动态预留后重新计算 Q_safe。
    
    关键逻辑：在 delta_t 买入 q 单位，将其计入 Q_in 后
    重新计算 backlog 与 Q_safe。
    """
    Q_in = Q_in.copy()
    if 0 <= delta_t < len(Q_in):
        Q_in[delta_t] += reserved_qty
    
    backlog = compute_inventory_trajectory(I_input_now, Q_in, n_lines, len(Q_in))
    Q_safe_h = compute_safe_buy_mask(C_total, Q_in, backlog, len(Q_in))
    
    Q_safe = np.zeros(len(Q_safe_h) + 1, dtype=np.float32)
    Q_safe[:len(Q_safe_h)] = Q_safe_h
    Q_safe[len(Q_safe_h)] = Q_safe_h[-1] if len(Q_safe_h) > 0 else 0
    
    return Q_safe, Q_in
```

### 5.6 资金约束

```python
def compute_budget_limit(
    B_current: float,
    Payables: np.ndarray,
    reserve: float = 1000.0
) -> float:
    """
    计算可用于采购的最大资金。
    
    公式：B_free = B_current - reserve - Σ Payables
    """
    total_payables = np.sum(Payables)
    B_free = B_current - reserve - total_payables
    return max(0, B_free)
```

### 5.7 L1 完整接口

```python
@dataclass
class L1Output:
    """L1 安全护盾的输出"""
    Q_safe: np.ndarray          # shape (H+1,) - 每个交货日的最大安全买入量，δt ∈ {0, 1, ..., H}
    Q_safe_sell: np.ndarray     # shape (H+1,) - 每个交货日的最大安全卖出量，δt ∈ {0, 1, ..., H}
    B_free: float               # 可用资金上限
    time_mask: np.ndarray       # shape (H+1,) - 时间掩码 (0 或 -inf)，δt ∈ {0, 1, ..., H}
    
    # 调试信息
    L_trajectory: np.ndarray    # 原材料 backlog 轨迹
    C_total: np.ndarray         # 库容向量
    Q_in: np.ndarray            # 已承诺入库量向量
    I_input_now: float          # 当前原材料库存
    n_lines: float              # 生产线数量（每日最大加工量）


class L1SafetyLayer:
    """L1 安全层（时序 ATP + clip_action）"""
    
    def __init__(self, horizon: int = 40, reserve: float = 1000.0):
        self.horizon = horizon
        self.reserve = reserve
    
    def compute(self, awi, is_buying: bool) -> L1Output:
        """
        计算安全约束。
        
        Args:
            awi: Agent World Interface
            is_buying: 当前是买入还是卖出角色
        
        Returns:
            L1Output
        """
        # 1. 获取库容向量
        C_total = get_capacity_vector(awi, self.horizon)
        
        # 2. 提取合约承诺
        Q_in, Q_out = extract_commitments(awi, self.horizon)
        
        # 3. 估计生产消耗（卖侧用满负荷）
        Q_prod = np.full(self.horizon, awi.profile.n_lines, dtype=np.float32)
        
        # 4. 计算 backlog 轨迹（原材料）
        # 重要：使用原材料库存（current_inventory_input），而非总库存
        # Q_out 是成品出库，不影响原材料库存，故不参与此计算
        I_now = float(getattr(awi, 'current_inventory_input', 0) or 0)
        backlog = compute_inventory_trajectory(I_now, Q_in, awi.profile.n_lines, self.horizon)
        
        # 5. 计算安全买入量掩码 (H 维)
        Q_safe_h = compute_safe_buy_mask(C_total, Q_in, backlog, self.horizon)
        
        # 6. 扩展为 H+1 维，支持 δt ∈ {0, 1, ..., H}
        # δt = H 的处理：假设与 H-1 相同（或设为 0 禁止）
        Q_safe = np.zeros(self.horizon + 1, dtype=np.float32)
        Q_safe[:self.horizon] = Q_safe_h
        Q_safe[self.horizon] = Q_safe_h[-1] if len(Q_safe_h) > 0 else 0  # δt = H 复用 H-1 的值
        
        # 7. 计算卖侧安全量（成品可交付量）
        I_output_now = float(getattr(awi, 'current_inventory_output', 0) or 0)
        Q_safe_sell_h = compute_safe_sell_mask(I_output_now, Q_prod, Q_out, self.horizon)
        Q_safe_sell = np.zeros(self.horizon + 1, dtype=np.float32)
        Q_safe_sell[:self.horizon] = Q_safe_sell_h
        Q_safe_sell[self.horizon] = Q_safe_sell_h[-1] if len(Q_safe_sell_h) > 0 else 0
        
        # 8. 计算资金约束
        Payables = self._extract_payables(awi)
        B_free = compute_budget_limit(awi.wallet, Payables, self.reserve)
        
        # 9. 生成时间掩码 (用于 L3 的 Masked Softmax)，H+1 维
        threshold = 1.0  # 最小可交易量
        if is_buying:
            time_mask = np.where(Q_safe >= threshold, 0.0, -np.inf)
        else:
            time_mask = np.where(Q_safe_sell >= threshold, 0.0, -np.inf)
        
        return L1Output(
            Q_safe=Q_safe,
            Q_safe_sell=Q_safe_sell,
            B_free=B_free,
            time_mask=time_mask,
            L_trajectory=backlog,
            C_total=C_total,
            Q_in=Q_in,
            I_input_now=I_input_now,
            n_lines=awi.profile.n_lines
        )
    
    def _extract_payables(self, awi) -> np.ndarray:
        """提取未来的应付款项（优先 AWI supplies_cost 接口）"""
        Payables = np.zeros(self.horizon, dtype=np.float32)
        t_current = awi.current_step
        
        def sum_mapping(mapping):
            if mapping is None:
                return 0.0
            if isinstance(mapping, dict):
                return sum(float(v) for v in mapping.values() if v is not None)
            return 0.0
        
        def fill_from_future(future_map, target):
            if not isinstance(future_map, dict):
                return
            for step, per_partner in future_map.items():
                try:
                    delta = int(step) - t_current
                except Exception:
                    continue
                if 0 <= delta < len(target):
                    target[delta] += sum_mapping(per_partner)
        
        supplies_cost = getattr(awi, 'supplies_cost', None)
        future_supplies_cost = getattr(awi, 'future_supplies_cost', None)
        
        Payables[0] = sum_mapping(supplies_cost)
        fill_from_future(future_supplies_cost, Payables)
        
        return Payables
    
```

---

## 6. L2 战略规划层 - 完整实现规范

### 6.1 核心职责

L2 是**日级战略规划器**，基于宏观状态生成分桶目标向量，指导 L3 的微观决策。

### 6.2 网络架构

```
输入:
  - x_static: (B, 12)
  - X_temporal: (B, H+1, 10)  # H=40, 10通道: vol_in, vol_out, prod_plan, inventory_proj, capacity_free, balance_proj, price_diff_in, price_diff_out, buy_pressure, sell_pressure
  - x_role: (B, 2)  # 角色 Multi-Hot [can_buy, can_sell]

架构:
  1. 时序特征塔 (Temporal Tower)
     - Conv1D(32, kernel=3, padding='same', ReLU)
     - Conv1D(64, kernel=7, padding='same', ReLU)
     - GlobalMaxPooling1D() -> h_temp (B, 64)
  
  2. 静态特征嵌入
     - Dense(32, ReLU) -> h_static (B, 32)
  
  3. 角色嵌入
     - Linear(2, d_role) + ReLU -> h_role (B, d_role)  # 支持 Multi-Hot 输入
  
  4. 融合层
     - Concat([h_temp, h_static, h_role]) -> (B, 64+32+d_role)
     - Dense(128, ReLU)
  
  5. 策略头 (Actor)
     - Dense(16) -> 目标向量均值 μ
     - Dense(16) -> 目标向量对数标准差 log_σ
  
  6. 价值头 (Critic)
     - Dense(64, ReLU)
     - Dense(1) -> V(s)

输出:
  - goal_vector: (B, 16) - 采样自 N(μ, σ)
  - value: (B, 1)
```

### 6.3 输出向量解码 (16维 = 4桶 × 4分量)

| 索引 | 含义 | 桶范围 |
|------|------|--------|
| 0-3 | Bucket 0 (Urgent): $Q_{buy}^0, P_{buy}^0, Q_{sell}^0, P_{sell}^0$ | Days 0-2 |
| 4-7 | Bucket 1 (Short-term): $Q_{buy}^1, P_{buy}^1, Q_{sell}^1, P_{sell}^1$ | Days 3-7 |
| 8-11 | Bucket 2 (Medium-term): $Q_{buy}^2, P_{buy}^2, Q_{sell}^2, P_{sell}^2$ | Days 8-14 |
| 12-15 | Bucket 3 (Long-term): $Q_{buy}^3, P_{buy}^3, Q_{sell}^3, P_{sell}^3$ | Days 15+ |

### 6.4 桶索引函数

```python
def delta_to_bucket(delta: int) -> int:
    """将相对交货时间映射到桶索引"""
    if delta <= 2:
        return 0  # Urgent
    elif delta <= 7:
        return 1  # Short-term
    elif delta <= 14:
        return 2  # Medium-term
    else:
        return 3  # Long-term
```

### 6.5 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonManagerPPO(nn.Module):
    """L2 战略规划层 - 1D-CNN 时平规划器"""
    
    def __init__(
        self,
        horizon: int = 40,
        n_buckets: int = 4,
        d_static: int = 12,
        d_temporal: int = 10,  # 10通道: vol_in, vol_out, prod_plan, inventory_proj, capacity_free, balance_proj, price_diff_in, price_diff_out, buy_pressure, sell_pressure
        d_role: int = 16,
    ):
        super().__init__()
        
        self.horizon = horizon
        self.n_buckets = n_buckets
        self.output_dim = n_buckets * 4  # 16
        
        # 时序特征塔
        self.conv1 = nn.Conv1d(d_temporal, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 静态特征嵌入
        self.static_embed = nn.Linear(d_static, 32)
        
        # 角色嵌入：使用 Linear 支持 Multi-Hot [can_buy, can_sell]
        # [0,1]=第一层（只卖）, [1,1]=中间层, [1,0]=最后层（只买）
        self.role_embed = nn.Linear(2, d_role)
        
        # 融合层
        fusion_dim = 64 + 32 + d_role
        self.fusion = nn.Linear(fusion_dim, 128)
        
        # Actor 头
        self.actor_mean = nn.Linear(128, self.output_dim)
        self.actor_log_std = nn.Linear(128, self.output_dim)
        
        # Critic 头
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x_static, X_temporal, x_role):
        """
        Args:
            x_static: (B, 12) - 静态特征
            X_temporal: (B, H+1, 10) - 时序特征 (H=40, 10通道)
            x_role: (B, 2) - 角色 Multi-Hot [can_buy, can_sell]
        
        Returns:
            mean: (B, 16)
            log_std: (B, 16)
            value: (B, 1)
        """
        # 时序塔 (需要转置为 B, C, L)
        x_t = X_temporal.permute(0, 2, 1)  # (B, 10, H+1)
        x_t = F.relu(self.conv1(x_t))
        x_t = F.relu(self.conv2(x_t))
        h_temp = self.pool(x_t).squeeze(-1)  # (B, 64)
        
        # 静态嵌入
        h_static = F.relu(self.static_embed(x_static))  # (B, 32)
        
        # 角色嵌入 (Linear 接受 Multi-Hot)
        h_role = F.relu(self.role_embed(x_role))  # (B, d_role)
        
        # 融合
        h = torch.cat([h_temp, h_static, h_role], dim=-1)
        h = F.relu(self.fusion(h))  # (B, 128)
        
        # Actor
        mean = self.actor_mean(h)
        log_std = self.actor_log_std(h)
        log_std = torch.clamp(log_std, -20, 2)  # 数值稳定性
        
        # Critic
        value = self.critic(h)
        
        return mean, log_std, value
    
    def sample_action(self, x_static, X_temporal, x_role):
        """采样动作并计算 log_prob
        
        Args:
            x_role: (B, 2) - 角色 Multi-Hot [can_buy, can_sell]
        """
        mean, log_std, value = self.forward(x_static, X_temporal, x_role)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
```

---

## 7. L3 执行层 - 完整实现规范（AOP/IPPO）

### 7.1 核心职责

L3 是**轮级执行器**，直接输出 AOP 动作：`op ∈ {ACCEPT, REJECT, END}`，若为 `REJECT` 则生成 counter offer `(qty, price, delta_t)`。参数在所有 `negotiator_id` 间共享，支持在线 IPPO/MAPPO。

### 7.2 输入设计

```python
L3_Input = {
    "history": Tensor[B, T, 4],    # 谈判历史 (q, p, delta_t, who/flag)
    "context": Tensor[B, C],       # 本地状态 + GlobalBroadcast（含 L2 目标缺口）
    "role": Tensor[B,],            # 0=Buyer, 1=Seller
    "alpha": Tensor[B,],           # L4 优先级（IPPO 阶段固定 0）
    "time_mask": Tensor[B, H+1],   # 0 或 -inf，δt ∈ {0..H}
    "has_offer": Tensor[B,],       # 用于 op mask（无 offer 禁止 ACCEPT）
}
```

### 7.3 输出设计（动作分解）

```
op_logits   : (B, 3)     # ACCEPT / REJECT / END
time_logits : (B, H+1)   # Masked Categorical
price_ab    : (B, 2)     # Beta(α, β) -> 映射到 [min_price, max_price]
qty_logits  : (B, Nq)    # 数量分桶 Categorical
value       : (B, 1)
```

采样顺序：`op -> (if REJECT) delta_t -> price -> Q_max -> quantity`。  
仅用 `time_mask` 与资源上界保证可行性，避免 `clip_action()` 改写时间/价格导致 logprob 不一致。

### 7.4 关键接口（PPO/IPPO）

- `act(obs, deterministic=False) -> action, logp, value, info`
- `evaluate_actions(obs, action) -> logp, entropy, value`

`logp` 由 `op` +（若 REJECT）`time + price + qty` 组成；`entropy` 仅统计启用分支。

### 7.5 离线 BC 兼容

- `op` 监督：CE
- `time/price/qty` 仅在 `op==REJECT` 时监督
- `ACCEPT` 必须携带对手原始 `offer_in`；`END` 无 counter offer
- 需要在 tracker 中显式记录 AOP 动作，避免 END/ACCEPT 标签歧义

---

## 8. L4 全局协调层 - 完整实现规范

### 8.1 核心职责

L4 是**并发协调器**，处理多个谈判线程之间的资源冲突。当前实现强调 **顺序无关** 与 **输入可重建**：
- 输出连续优先级 $\\alpha$（线程数可变）作为全局调度信号；
- $\\alpha$ **仅作为 L3 输入**（不再调制输出、不做顺序裁剪/动态预留）；
- 每次决策先收集全部活跃线程特征，统一计算一次 L4 并缓存复用，减少回调顺序导致的差异。

### 8.2 时空注意力机制

```python
Attention(Q, K, V) = Softmax((QK^T / √d_k) + M_time) V
```

其中 $M_{time}[i,j]$ 是时间偏置：
- 如果线程 $i$ 和 $j$ 的意向交货时间相近 → $M$ 较大（冲突警示）
- 如果时间相距甚远 → $M \to 0$（无冲突）

### 8.3 PyTorch 实现

> 说明（与代码实现对齐）：L4 不再直接输入 L3 的 latent/隐状态，而是输入**可离线重建的显式特征**：  
> - `thread_feat`：每个 negotiation 的线程特征（交期切片的 `X_temporal[δ]`、L1/L2 约束、谈判进度/报价偏差等）  
> - `global_feat`：全局上下文（如 `goal_hat`、`x_static`、活跃线程数等）

```python
class GlobalCoordinator(nn.Module):
    """L4 全局协调层 - 时空注意力网络"""
    
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        horizon: int = 40,
        thread_feat_dim: int = 24,
        global_feat_dim: int = 30,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.horizon = horizon
        
        # 全局状态编码
        self.global_encoder = nn.Linear(global_feat_dim, d_model)  # 全局上下文特征
        
        # 线程特征投影
        self.thread_proj = nn.Linear(thread_feat_dim, d_model)
        
        # 时间嵌入 (H+1 维，支持 δt ∈ {0, 1, ..., H})
        self.time_embed = nn.Embedding(horizon + 1, d_model)
        
        # 角色嵌入
        self.role_embed = nn.Embedding(2, d_model)
        
        # 多头自注意力
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # 时间偏置矩阵生成器 (H+1 × H+1)
        self.time_bias = nn.Parameter(torch.zeros(horizon + 1, horizon + 1))
        nn.init.normal_(self.time_bias, std=0.1)
        
        # 输出门控
        self.gate = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, thread_feats, thread_times, thread_roles, global_feat, thread_mask=None):
        """
        Args:
            thread_feats: (B, K, thread_feat_dim) - K 个线程的线程特征
            thread_times: (B, K) - 每个线程的意向交货时间
            thread_roles: (B, K) - 每个线程的角色
            global_feat: (B, global_feat_dim) - 全局上下文特征
            thread_mask: (B, K) - True=有效线程，False=padding（可选）
        
        Returns:
            weights: (B, K) - 线程重要性权重
            gated_states: (B, K, d_model) - 门控后的状态
        """
        B, K, _ = thread_feats.shape
        n_heads = self.mha.num_heads
        
        # 1. 投影线程状态
        h = self.thread_proj(thread_feats)  # (B, K, d)
        
        # 2. 添加时间嵌入
        t_emb = self.time_embed(thread_times.long())  # (B, K, d)
        h = h + t_emb
        
        # 3. 添加角色嵌入
        r_emb = self.role_embed(thread_roles.long())  # (B, K, d)
        h = h + r_emb
        
        # 4. 全局上下文作为 Query
        q = self.global_encoder(global_feat).unsqueeze(1)  # (B, 1, d)
        q = q.expand(-1, K, -1)  # (B, K, d)
        
        # 5. 构建时间偏置掩码
        # 基于线程间的时间接近程度：接近的时间 -> 高偏置（表示冲突，需要相互关注）
        time_diff = thread_times.unsqueeze(-1) - thread_times.unsqueeze(-2)  # (B, K, K)
        time_diff = time_diff.abs().float()
        
        # 时间距离越近 -> 偏置越大（更强的注意力）
        # 时间距离越远 -> 偏置越小（更弱的注意力）
        # 使用负距离作为偏置，使得相近时间的线程相互关注更多
        attn_bias = -time_diff  # (B, K, K)
        
        # 扩展到多头格式: (B, K, K) -> (B * n_heads, K, K)
        attn_bias = attn_bias.unsqueeze(1).expand(-1, n_heads, -1, -1)
        attn_bias = attn_bias.reshape(B * n_heads, K, K)
        
        # 6. 多头注意力 (使用 h 作为 K, V，应用时间偏置掩码)
        attn_output, attn_weights = self.mha(q, h, h, attn_mask=attn_bias)
        
        # 7. 计算门控权重
        gate_values = self.gate(attn_output).squeeze(-1)  # (B, K)
        
        # 8. 归一化权重
        weights = F.softmax(gate_values, dim=-1)  # (B, K)
        
        return weights, attn_output
    
    # 已弃用：L4 不再调制 L3 输出（alpha 仅作为输入信号）
```

---

## 9. 奖励函数设计

> IPPO 起步建议使用**谈判级最小奖励**（成功奖励 + surplus - 时间惩罚），先保证在线可学，再逐步引入势能整形与复合奖励。

### 9.1 势能奖励函数 (解决跨期信用分配)

```python
def compute_potential(state: dict, gamma: float = 0.98) -> float:
    """
    计算状态势能 Φ(s)
    
    公式：Φ(s) = B_t + Val(I_now) + Σ γ^δ * Val(contract)
    """
    B_t = state['balance']
    
    # 库存价值
    I_raw = state['inventory_raw']
    I_product = state['inventory_product']
    P_raw = state['spot_price_in']
    P_product = state['spot_price_out']
    
    val_inventory = I_raw * P_raw + I_product * P_product
    
    # 未执行合约价值 (时间折扣)
    val_contracts = 0
    for contract in state['pending_contracts']:
        delta = contract['delivery_time'] - state['current_step']
        val = contract['quantity'] * contract['unit_price']
        val_contracts += (gamma ** delta) * val
    
    return B_t + val_inventory + val_contracts


def shaped_reward(r_env: float, s_t: dict, s_t1: dict, gamma: float = 0.98) -> float:
    """
    势能整形奖励
    
    R' = R_env + (Φ(s_{t+1}) - Φ(s_t))
    """
    phi_t = compute_potential(s_t, gamma)
    phi_t1 = compute_potential(s_t1, gamma)
    
    return r_env + (phi_t1 - phi_t)
```

### 9.2 复合奖励函数

```python
def composite_reward(
    r_env: float,
    s_t: dict,
    s_t1: dict,
    goal: np.ndarray,
    executed_qty: float,
    lambda_1: float = 0.1,
    lambda_2: float = 0.5,
    lambda_3: float = 0.1,
) -> float:
    """
    R_total = R_profit + λ1 * R_liquidity - λ2 * R_risk + λ3 * R_intrinsic
    """
    # 1. 势能利润
    R_profit = shaped_reward(r_env, s_t, s_t1)
    
    # 2. 流动性奖励 (成交即给分)
    R_liquidity = 1.0 if executed_qty > 0 else 0.0
    
    # 3. 风险惩罚 (未来库存低于 0)
    min_future_inv = np.min(s_t1['inventory_trajectory'])
    R_risk = np.exp(max(0, -min_future_inv)) - 1
    
    # 4. 内在一致性 (与 L2 目标对齐)
    target_qty = goal[0]  # Q_buy 或 Q_sell
    R_intrinsic = -((executed_qty - target_qty) ** 2) / (target_qty + 1)
    
    return R_profit + lambda_1 * R_liquidity - lambda_2 * R_risk + lambda_3 * R_intrinsic
```

---

## 10. 数据工程与取证流水线

### 10.1 时序目标重构（L2 标签，v2 口径）

```python
def reconstruct_l2_goals(daily_logs: List[dict], n_buckets: int = 4) -> List["MacroSample"]:
    """
    从日志中反推 L2 的 16 维目标向量（v2，保持 16 维，降低稀疏/噪声）。

    v2 标签要点：
    - 交易量目标 Q = 成交（软分桶） + 缺口补偿（needed_supplies/needed_sales） + “活跃未成交”弱意图（offers_snapshot）
    - 价格目标 P = 成交 VWAP + 活跃报价轮次衰减加权均值 + spot 回退（买 0.95、卖 1.05），避免 max/min 噪声
    - 不可谈判侧（x_role）目标压零，避免生成“不可谈判”的假目标

    说明：
    - daily_logs 需包含：deals、offers_snapshot、needed_supplies/needed_sales、spot_price_in/out 等字段
    - 详见实现：litaagent_std/hrl_xf/data_pipeline.py:reconstruct_l2_goals
    """
    ...
```

### 10.2 L3 动作标签提取（AOP）

```python
def compute_time_mask_offline(
    state_dict: dict,
    horizon: int = 40,
    min_tradable_qty: float = 1.0,
) -> np.ndarray:
    """
    从日志状态离线计算 time_mask（与 L1SafetyLayer 逻辑一致）。
    """
    Q_safe = compute_q_safe_offline(state_dict, horizon=horizon)
    return np.where(Q_safe >= min_tradable_qty, 0.0, -np.inf).astype(np.float32)


def fix_invalid_time_label(time_label: int, time_mask: np.ndarray) -> int:
    """若 time_label 被 L1 禁止，则投影到最近的合法 delta。"""
    if time_label < len(time_mask) and time_mask[time_label] == 0.0:
        return time_label
    valid_indices = np.where(time_mask == 0.0)[0]
    if len(valid_indices) == 0:
        return 0
    distances = np.abs(valid_indices - time_label)
    return int(valid_indices[np.argmin(distances)])


def extract_l3_actions_aop(
    negotiation_logs: List[dict],
    *,
    daily_states: Dict[int, dict],
    horizon: int = 40,
):
    """
    从谈判日志中提取 L3 AOP 动作标签。
    Returns: List of {history, role, time_mask, action_op, offer_in, offer_out}
    """
    samples = []
    for neg in negotiation_logs:
        history = np.array(neg['offer_history'])  # (T, 3)
        role = 0 if neg['is_buyer'] else 1
        day = int(neg.get('sim_step', neg.get('day', 0)) or 0)
        state = daily_states[day]
        time_mask = compute_time_mask_offline(state, horizon=horizon)

        action_op = neg['action_op']  # 0=ACCEPT,1=REJECT,2=END（需 tracker 显式记录）
        offer_in = neg.get('offer_in')    # 对手当前 offer
        offer_out = neg.get('offer_out')  # 我方 counter offer（仅 REJECT）

        if action_op == 1 and offer_out is not None:
            offer_out[2] = fix_invalid_time_label(int(offer_out[2]), time_mask)

        samples.append({
            'history': history,
            'role': role,
            'time_mask': time_mask,
            'action_op': action_op,
            'offer_in': offer_in,
            'offer_out': offer_out,
        })
    return samples
```

---

## 11. 训练流程

### 11.1 四阶段课程学习

| 阶段 | 名称 | 目标 | 方法 | 数据来源 |
|------|------|------|------|----------|
| 0 | Cold Start | 不崩盘 | 行为克隆 (BC) | PenguinAgent 日志 |
| 1 | Offline RL | 超越专家均值 | AWR (优势加权回归) | 筛选后的专家轨迹 |
| 2 | Online Warmup | 先跑通在线 | IPPO（仅 L3，α=0） | SCML 2025 模拟器 |
| 3 | Online CTDE | 稳定提升 | MAPPO + Centralized Critic | 对手池 |
| 4 | Self-Play | 逼近纳什均衡 | MAPPO + 对手池 | 历史版本对抗 |

### 11.2 训练循环伪代码

```python
class HRLXFTrainer:
    def __init__(self, l1, l2, l3, l4, config):
        self.l1 = l1  # 不训练，纯规则
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.config = config
    
    def _build_l2_loss_mask(self, x_role: torch.Tensor) -> torch.Tensor:
        """
        根据 x_role 构建 L2 损失掩码：不可谈判的分量应被忽略
        
        Args:
            x_role: (B, 2) - [can_buy, can_sell] Multi-Hot
        
        Returns:
            mask: (B, 16) - 1.0 表示有效，0.0 表示忽略
        """
        B = x_role.size(0)
        mask = torch.ones(B, 16, device=x_role.device)
        
        # 16 维目标结构: [Q_buy, P_buy, Q_sell, P_sell] × 4 个 bucket
        # 索引: 0,1=buy_bucket0, 2,3=sell_bucket0, 4,5=buy_bucket1, ...
        buy_indices = [0, 1, 4, 5, 8, 9, 12, 13]    # Q_buy, P_buy 在每个 bucket
        sell_indices = [2, 3, 6, 7, 10, 11, 14, 15]  # Q_sell, P_sell 在每个 bucket
        
        # can_buy=0 时，掩码 buy 分量
        mask[:, buy_indices] *= x_role[:, 0:1]
        # can_sell=0 时，掩码 sell 分量
        mask[:, sell_indices] *= x_role[:, 1:2]
        
        return mask
    
    def train_step(self, batch):
        # ========== L2 更新 (日级) ==========
        with torch.enable_grad():
            # x_role: Multi-Hot [can_buy, can_sell]
            x_role = batch['x_role']  # (B, 2)
            
            mean, log_std, value = self.l2(
                batch['macro_state'],
                batch['temporal_state'],
                x_role  # 传入 Multi-Hot 角色向量
            )
            
            # 构建损失掩码：忽略不可谈判的目标分量
            loss_mask = self._build_l2_loss_mask(x_role)  # (B, 16)
            
            # PPO Loss (带掩码)
            l2_loss = self.ppo_loss(
                mean, log_std, value,
                batch['l2_actions'],
                batch['l2_returns'],
                batch['l2_advantages'],
                loss_mask=loss_mask  # 新增：传入损失掩码
            )
        
        self.l2_optimizer.zero_grad()
        l2_loss.backward()
        self.l2_optimizer.step()
        
        # ========== L3 更新 (轮级, AOP) ==========
        with torch.enable_grad():
            # L4 仅输出 alpha（可选择冻结）
            alpha = self.l4(
                batch['thread_feats'],
                batch['thread_times'],
                batch['thread_roles'],
                batch['global_feat']
            )

            # L3 前向（AOP + time_mask）
            op_logits, time_logits, price_ab, qty_logits, values = self.l3(
                batch['history'],
                batch['context'],
                batch['l3_role'],
                alpha,
                batch['time_mask'],
                batch['has_offer'],
            )

            # 损失计算（仅对 REJECT 监督 time/price/qty）
            loss_op = F.cross_entropy(op_logits, batch['action_op'])
            reject_mask = batch['action_op'] == OP_REJECT
            loss_t = F.cross_entropy(time_logits[reject_mask], batch['target_time'][reject_mask])
            loss_p = beta_nll(price_ab[reject_mask], batch['target_price'][reject_mask])
            loss_q = F.cross_entropy(qty_logits[reject_mask], batch['target_qty'][reject_mask])
            l3_loss = loss_op + loss_t + loss_p + loss_q
        
        self.l3_optimizer.zero_grad()
        l3_loss.backward()
        self.l3_optimizer.step()
        
        return {'l2_loss': l2_loss.item(), 'l3_loss': l3_loss.item()}
```

---

## 12. 文件结构规划

```
litaagent_std/
└── hrl_xf/
    ├── __init__.py
    │
    ├── # 核心模块
    ├── agent.py            # HRL-XF Agent 主类（StdAgent 逐线程接口）
    ├── l1_safety.py        # L1 安全层（Q_safe/time_mask + clip_action）
    ├── l2_manager.py       # L2 日级目标层（启发式/神经）
    ├── l3_executor.py      # L3 执行层（AOP/IPPO）
    ├── l4_coordinator.py   # L4 并发协调层（启发式/神经，输出连续 α）
    ├── state_builder.py    # 状态张量构建器（在线）
    ├── rewards.py          # 奖励函数（在线 RL 预留）
    ├── data_pipeline.py    # 数据流水线（离线：宏/微/L4 蒸馏样本）
    └── training.py         # 训练（BC/AWR + L4 蒸馏；在线 IPPO/MAPPO 预留）
```

---

## 13. 实施检查清单

### Phase 0: 基础设施
- [x] 创建 `hrl_xf/` 目录结构
- [x] 统一使用经济容量（n_lines × 剩余天数）
- [x] 实现 `get_capacity_vector()` / `extract_commitments()`

### Phase 1: L1 安全护盾
- [x] 实现 `L1SafetyLayer`（在线）与 `compute_l1_masks_offline()`（离线对齐）
- [x] 实现库存轨迹投影与 $Q_{safe}[\delta]$ 计算
- [x] 实现时间掩码生成与 `clip_action`
- [ ] 单元测试：极端边界（爆仓、缺货）完整覆盖（目前仅基础/接口测试）

### Phase 2: L2 战略规划
- [x] 实现 `HorizonManagerPPO`（L2 模型骨架）
- [x] 实现 1D-CNN 时序特征提取与 16 维分桶输出
- [x] 实现角色嵌入（`x_role`）
- [ ] 单元测试：目标向量解码完整覆盖（当前仅最小接口/闭环测试）

### Phase 3: L3 执行（AOP）
- [x] 实现 L3 Actor-Critic 骨架与 AOP 头
- [x] 实现 op/time/price/qty 输出与 mask
- [ ] 单元测试：因果掩码/边界情况完整覆盖（当前仅最小接口测试）

### Phase 4: L4 全局协调
- [x] 实现 `GlobalCoordinator`（启发式 + 神经网络）
- [x] L4 输入语义对齐：`thread_feat_set + global_feat`（不依赖 L3 latent）
- [ ] 单元测试：更复杂的多线程冲突场景（当前仅覆盖核心接口）

### Phase 5: 数据工程
- [x] 实现 L2 v2 目标重构（`reconstruct_l2_goals()`）
- [x] 实现 goal_hat 回填（`load_tournament_data(goal_backfill="l2")`）
- [ ] 实现 L3 AOP 动作提取（`extract_l3_actions_aop()`）
- [x] 实现状态张量构建器（在线/离线语义对齐）
- [x] 提供 `hrl_data_runner` 采集专家日志（推荐配合 `--track-only-penguin`）

### Phase 6: 训练
- [x] 实现离线 BC（L2/L3）与 L3 AWR
- [x] 实现 L4 启发式蒸馏训练（监督学习）
- [ ] 实现在线 IPPO 采样/更新（仅 L3，α=0）
- [ ] 实现 MAPPO/自博弈训练流水线（后续里程碑）

### Phase 7: 集成测试
- [x] 基础单元测试（`tests/test_hrlxf_*`）
- [ ] 端到端测试（模拟器运行）
- [ ] 对抗 PenguinAgent 评估与性能调优

---

*本文档应随实施进度更新，完成项请打勾 ✅*
