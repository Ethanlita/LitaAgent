# HRL-XF L1&L3&L4 层重构详细行动方案

> **生成日期**：2025年12月25日
> **文档版本**：1.5（方案A：L4=网络输出α；BC/数据重建纳入 END_NEGOTIATION；L2 目标标签优先使用 first_proposals 聚合）
> **基于设计文档**：
> - `HRL-XF 实施规范.md`
> - `HRL-XF 期货市场代理重构.md`
> - `HRL-XF 期货市场适应性重构.md`

---

## 一、重构概述

### 1.1 核心变化总结

| 层级 | 当前设计 | 重构后设计 |
|------|----------|------------|
| **L1** | 输出 `Q_safe`, `time_mask`, `baseline_action` | 仅输出 `Safety Mask`（不再产生 baseline） |
| **L3** | 接收 baseline，输出残差 `(Δq, Δp, δt)` | 直接输出完整动作，支持 SAO(AOP) 协议（ACCEPT / REJECT_OFFER+counter / END） |
| **L4** | 旧实现（将完全移除）：神经网络输出权重调制 L3 残差 + `batch_planner` 动态预留/裁剪 | `GlobalMonitor` 广播全局状态；`GlobalCoordinator` 输出每线程优先级 α（不再调制 L3 残差；也不再绑定任何“动态预留/顺序裁剪”机制） |
| **Action** | `(q, p, δt)` 三元组 | **SCML/SAO 协议对齐**：`respond()` 输出 `SAOResponse(ResponseType.{ACCEPT_OFFER, REJECT_OFFER, END_NEGOTIATION}, outcome)`；`propose()` 输出 `outcome` 或 `None` |

### 1.2 新的信息流

```
AWI → L1 (Safety Mask)
    → L2 (日级目标)
    → L4GlobalMonitor (全局状态监控) → 广播 GlobalBroadcast
                   ↘ GlobalCoordinator (自注意力网络) → α(thread)
    → L3 (局部状态 + GlobalBroadcast + α) → 完整 AOP Action
```

### 1.3 重构动机

1. **L1 移除 Baseline**：Baseline 的概念与"残差学习"强绑定，移除后 L3 可以更灵活地学习完整策略
2. **L3 输出完整动作**：残差模型依赖 baseline 质量，直接输出完整动作可避免 baseline 偏差传播
3. **L4 从“权重调制”改为“监控 + 优先级”**：保留 L4 为可训练网络，但不再输出调制系数去改写 L3 残差；改为由 `L4GlobalMonitor` 广播全局状态、`GlobalCoordinator` 输出线程优先级 α，L3 将 α 作为输入自行调整行为（更清晰、更易调试）
4. **AOP 协议**：显式建模"接受/拒绝/出价"决策，与实际谈判语义对齐

---

## 二、Action 空间重定义（SCML / SAO 协议：仅三类动作）

> **关键语义更正（必须落实到 agent / tracker / pipeline）**  
> SCML 标准赛道的谈判回调来自 **negmas 的 SAO（Single-Alternating-Offers）协议**。对我们最关键的点只有两个：  
> 1) **在 `respond()` 里没有独立的 “OFFER” 动作**。所谓“还价/反报价（Counter Offer）”，在 API 上就是：  
> &nbsp;&nbsp;&nbsp;&nbsp;`SAOResponse(ResponseType.REJECT_OFFER, counter_offer)`  
> 2) **在 SCML 的实现里，“拒绝但不提出反报价”（`REJECT_OFFER` 且 `outcome=None`）是非法的**（你已核对）。  
> &nbsp;&nbsp;&nbsp;&nbsp;因此我们不再设计/重建 `DO_NOTHING` 动作。

---

### 2.1 统一动作抽象：L3 的“操作符 op + 可选报价 offer_out”

为了让同一套 L3 网络同时覆盖 `propose()` 和 `respond()`，我们把动作统一成：

- 一个离散操作符：\(op\in\{\texttt{ACCEPT},\texttt{COUNTER},\texttt{END}\}\)
- 以及一个可选的我方报价 \(offer\_out\)（仅当 \(op=\texttt{COUNTER}\) 时需要）

> **Outcome 的字段顺序**（SCML/negmas）：`(quantity, time, unit_price)`  
> 为避免混淆，本文统一使用：  
> - 数量：\(q\)  
> - 交货日（绝对时间）：\(t_{abs}\)  
> - 单价：\(p\)  
> 即：`outcome = (q, t_abs, p)`  
> 但在网络内部我们仍可用 \(\delta t = t_{abs}-t_{now}\) 做离散 bucket / time head（更容易 mask），落地到环境前再转回绝对时间。

#### 2.1.1 三类动作的含义

| op_id | op 名称 | 在 respond() 中含义 | 在 propose() 中含义 |
|---:|---|---|---|
| 0 | **ACCEPT** | `ACCEPT_OFFER`（接受对手当前 offer） | **无效（mask 掉）** |
| 1 | **COUNTER** | `REJECT_OFFER + counter_offer`（拒绝并还价） | 返回 `offer_out`（主动出价） |
| 2 | **END** | `END_NEGOTIATION`（结束谈判） | **允许**：返回 `None` / 或在批量返回字典中省略该 negotiator，以显式结束该谈判（用于复刻 PenguinAgent 的“筛掉对手”行为） |

> 重要更正：PenguinAgent 很少显式返回 `ResponseType.END_NEGOTIATION`，但它会通过“在 `first_proposals()` / `counter_all()` 的返回字典里 **省略某些 negotiator**”来**隐式终止**这些谈判；控制器会把缺失键补成 `END_NEGOTIATION`。  
> 因此 **BC 数据必须包含 END**（否则线上你一旦遇到需要“筛掉/终止某些谈判”的场景就会 OOD）。实现上建议在 **tracker 侧把“缺失响应/缺失提案”显式记录为 `action_op=END`**（见 7.2.4）。

---

### 2.2 与 SCML API 的一一映射（实现必须严格按此）

#### 2.2.1 respond() → SAOResponse

令对手当前报价为 `offer_in = state.current_offer`（若为 `None`，属于异常/机制边界情况，建议直接 END）。

- 若 `op==ACCEPT`：  
  `return SAOResponse(ResponseType.ACCEPT_OFFER, offer_in)`  
  > **注意**：`ACCEPT` 必须带上 **对手的 offer（相同 Outcome）**，不能传 `None` 或我方自定义 outcome。

- 若 `op==COUNTER`：  
  `return SAOResponse(ResponseType.REJECT_OFFER, offer_out)`  
  > **注意**：这里的 `REJECT_OFFER` 在 SCML 语义上就是“反报价/还价”，因此 **必须携带一个 counter offer**。

- 若 `op==END`：  
  `return SAOResponse(ResponseType.END_NEGOTIATION, None)`

#### 2.2.2 propose() → Outcome | None

- 通常只允许 `op==COUNTER`：`return offer_out`  
- 若你希望允许“不开局”，可把 `op==END` 映射为 `return None`（或由 controller 默认补 END）。  
  但由于你明确“不需要重建 DO_NOTHING”，BC 阶段不训练 `None`/END 的 propose 行为。

---

### 2.3 L3 的输出形状（神经网络头）

L3 不再输出残差，而是输出完整动作：

- `op_logits ∈ R^{3}`（Categorical：ACCEPT / COUNTER / END）
- `quantity_head`：例如 `q_ratio ∈ (0,1)` 或直接回归 `q`
- `price_head`：例如 `p_ratio ∈ (0,1)` 或直接回归 `p`
- `time_logits ∈ R^{H+1}`（离散的 \(\delta t\in\{0,1,\dots,H\}\)）

并通过 **掩码 + 兜底规则** 确保合法性：

1) **mode mask（上下文掩码）**  
   - `propose()`：mask 掉 `ACCEPT`（通常也 mask 掉 `END`）  
   - `respond()`：三类都允许（END 是否允许取决于策略/规则）

2) **L1 Safety Mask（只管“能不能履约”，不管价格好坏）**  
   - `ACCEPT`：仅在“接受后必然违约/浪费/资金不足”时屏蔽  
   - `COUNTER`：仅在“该交期/数量必然违约/爆仓/资金不足”时裁剪/改选  
   - **不**因为价格高低屏蔽动作（价格优劣交给在线 RL 的回报学习）

3) **关键合法性兜底（非常重要）**  
   - 如果策略选择 `COUNTER`，但经过 L1 裁剪/约束后无法产生任何合法 offer（例如所有可行 \(\delta t\) 都被 mask，或 \(q\le 0\)）：  
     **必须改为 `END`**（因为 “REJECT_OFFER 且 outcome=None” 非法）。

---

### 2.4 离线重建标签（BC 阶段）应当怎么理解

- 我方每次发出报价（`offer_made`：包括 first_proposal / counter_offer）→ `op=COUNTER`
- 我方接受对手报价（若日志中能明确识别）→ `op=ACCEPT`
- 我方结束谈判（若日志/控制器能明确识别）→ `op=END`

> 由于 PenguinAgent 的 `END` 多为**隐式 END**（`first_proposals()/counter_all()` 省略某些 negotiator → 控制器补 `END_NEGOTIATION`），我们需要在 **tracker/pipeline** 中把它显式补齐为 `op=END`。  
> 因此 BC 阶段将同时训练三类动作：`COUNTER`、`ACCEPT`、`END`（避免线上遇到“需要筛掉某些对手/终止谈判”的场景产生 OOD）。

## 三、L1 层重构：移除 Baseline，仅输出 Safety Mask

### 3.1 变更概述

| 文件 | 变更 |
|------|------|
| `litaagent_std/hrl_xf/l1_safety.py` | 移除 `_compute_baseline()` 和 `baseline_action` |
| `litaagent_std/hrl_xf/l1_safety.py` | 废弃 `recompute_q_safe_after_reservation()` —— 仅被 batch_planner 调用，动态预留机制移除后无调用方 |

### 3.2 新的 L1Output 结构

```python
@dataclass
class L1Output:
    """L1 安全护盾的输出（仅 Safety Mask）.

    Attributes:
        Q_safe: 每个交货日的最大安全买入量，shape (H+1,)，δt ∈ {0, 1, ..., H}
        Q_safe_sell: 每个交货日的最大安全卖出量，shape (H+1,)
        B_free: 可用资金上限
        time_mask: 时间掩码 (0 或 -inf)，shape (H+1,)，用于 L3 的 Masked Softmax

    调试信息（保留）:
        L_trajectory: 原材料 backlog 轨迹，shape (H,)
        C_total: 库容向量，shape (H,)
        Q_in: 已承诺入库量向量，shape (H,)
        I_input_now: 当前原材料库存
        n_lines: 生产线数量
    """
    Q_safe: np.ndarray
    Q_safe_sell: np.ndarray
    B_free: float
    time_mask: np.ndarray

    # 调试信息
    L_trajectory: np.ndarray
    C_total: np.ndarray
    Q_in: Optional[np.ndarray] = None
    I_input_now: Optional[float] = None
    n_lines: Optional[float] = None

    # 移除: baseline_action: Tuple[float, float, int]
```

### 3.3 具体修改内容

#### 3.3.1 删除 `_compute_baseline()` 方法

从 `L1SafetyLayer` 类中移除整个 `_compute_baseline()` 方法（约 80 行）。

#### 3.3.2 修改 `compute()` 方法

```python
class L1SafetyLayer:
    def compute(self, awi: "StdAWI", is_buying: bool) -> L1Output:
        """计算安全约束.

        Args:
            awi: Agent World Interface
            is_buying: 当前是买入还是卖出角色

        Returns:
            L1Output: 包含 Q_safe, Q_safe_sell, B_free, time_mask 等
                      不再包含 baseline_action
        """
        # 1-8 步骤保持不变：计算 Q_safe, Q_safe_sell, B_free, time_mask
        # ...

        # 移除: baseline_action = self._compute_baseline(...)

        return L1Output(
            Q_safe=Q_safe,
            Q_safe_sell=Q_safe_sell,
            B_free=B_free,
            time_mask=time_mask,
            # baseline_action=baseline_action,  # 移除此行
            L_trajectory=backlog,
            C_total=C_total,
            Q_in=Q_in,
            I_input_now=I_input_now,
            n_lines=float(n_lines)
        )
```

#### 3.3.3 保留 `clip_action()` 方法

L1 仍需提供 `clip_action()` 方法，用于在最终输出前裁剪动作到安全范围：

```python
def clip_action(
    self,
    action: Tuple[float, float, int],
    Q_safe: np.ndarray,
    B_free: float,
    is_buying: bool,
    min_price: float = 0.0,
    max_price: float = float('inf'),
    Q_safe_sell: Optional[np.ndarray] = None
) -> Tuple[int, float, int]:
    """裁剪动作到安全范围.

    所有最终动作必须经过此方法裁剪。
    此方法保持不变。
    """
    # ... 现有实现保持不变 ...
```

---

## 四、L4 层重构：监控器 + 优先级协调器（α 机制）

### 4.1 设计理念

L4 采用**双组件架构**：

| 组件 | 类型 | 功能 |
|------|------|------|
| `L4GlobalMonitor` | 确定性规则 | 追踪全局状态、成交统计、目标缺口计算、广播全局状态 |
| `GlobalCoordinator` | 自注意力网络 | 为每个活跃线程计算优先级 α ∈ (-1, 1) |

**α 优先级的语义**：

| α 值范围 | 语义 | 期望行为 |
|----------|------|----------|
| α → +1 | 高优先级/紧急 | 激进策略：加快让步、争取快速成交、可接受更多数量 |
| α ≈ 0 | 中性 | 标准策略：平衡价格与成交速度 |
| α → -1 | 低优先级/非紧急 | 保守策略：坚持价格、慢节奏谈判、限制数量 |

**设计动机**：

1. **资源协调**：替代原 batch_planner 的动态预留机制，用 α 信号引导 L3 行为
2. **CTDE 架构基础**：α 由集中式 L4 产生，L3 分散执行时根据 α 调整策略
3. **可训练**：α 网络可在 RL 阶段联合优化，学习最优资源分配策略

### 4.2 新的数据结构

#### 4.2.1 ThreadState 与 ThreadPriority

```python
@dataclass
class ThreadState:
    """单个谈判线程的状态（供 L4 自注意力使用）.

    Attributes:
        thread_id: 线程唯一标识
        is_buying: 谈判方向
        negotiation_step: 当前谈判轮次
        relative_time: 谈判相对进度 [0, 1]
        current_offer: 对手当前报价 (q, p, t) 或 None
        history_len: 历史轮次数

        # 目标相关
        target_bucket: 交货时间对应的桶索引
        goal_gap: 该桶的目标缺口

        # 资源相关
        Q_safe_at_t: 该交货时间的安全量
        B_remaining: 剩余预算（买侧）
    """
    thread_id: str
    is_buying: bool
    negotiation_step: int
    relative_time: float
    current_offer: Optional[Tuple[int, float, int]]
    history_len: int
    target_bucket: int
    goal_gap: float
    Q_safe_at_t: float
    B_remaining: float


@dataclass
class ThreadPriority:
    """L4 为单个线程计算的优先级.

    Attributes:
        thread_id: 线程唯一标识
        alpha: 优先级值 ∈ (-1, 1)
               +1 = 高优先级/紧急 → 激进策略
               0 = 中性
               -1 = 低优先级 → 保守策略
        attention_weight: 自注意力权重（调试用）
    """
    thread_id: str
    alpha: float  # ∈ (-1, 1)
    attention_weight: Optional[float] = None
```

#### 4.2.2 GlobalBroadcast

```python
@dataclass
class GlobalBroadcast:
    """L4 广播给每个 L3 线程的全局状态.

    此结构包含 L3 做决策时需要的全局上下文信息。
    """

    # ========== 时间与进度 ==========
    current_step: int              # 当前仿真天
    n_steps: int                   # 总天数
    step_progress: float           # t / T_max ∈ [0, 1]

    # ========== L2 目标与剩余缺口 ==========
    l2_goal: np.ndarray            # shape (16,)，原始 L2 目标向量
    goal_gap_buy: np.ndarray       # shape (4,)，每个桶的买入缺口
    goal_gap_sell: np.ndarray      # shape (4,)，每个桶的卖出缺口

    # ========== 当日已成交统计 ==========
    today_bought_qty: float        # 今日已买入总量
    today_bought_value: float      # 今日已买入总金额
    today_sold_qty: float          # 今日已卖出总量
    today_sold_value: float        # 今日已卖出总金额

    # 按桶细分
    today_bought_by_bucket: np.ndarray  # shape (4,)
    today_sold_by_bucket: np.ndarray    # shape (4,)

    # ========== 资源约束状态（动态更新） ==========
    B_remaining: float             # 剩余可用预算（扣除今日已承诺）
    Q_safe_remaining: np.ndarray   # shape (H+1,)，剩余安全买入量
    Q_safe_sell_remaining: np.ndarray  # shape (H+1,)，剩余安全卖出量

    # ========== 并发谈判概览 ==========
    n_active_buy_threads: int      # 活跃的买入谈判数
    n_active_sell_threads: int     # 活跃的卖出谈判数

    # ========== 状态张量（供 L3 神经网络使用） ==========
    x_static: np.ndarray           # shape (12,)，静态状态向量
    X_temporal: np.ndarray         # shape (H+1, 10)，时序状态张量
```

### 4.3 新的 L4GlobalMonitor 类

```python
from typing import List, Tuple, Optional
import numpy as np

# 桶定义（与 l2_manager.py 保持一致）
BUCKET_RANGES = [(0, 2), (3, 7), (8, 14), (15, 40)]

def delta_to_bucket(delta: int) -> int:
    """将相对交货时间映射到桶索引."""
    for i, (lo, hi) in enumerate(BUCKET_RANGES):
        if lo <= delta <= hi:
            return i
    return 3


class L4GlobalMonitor:
    """L4 全局监控器（无可训练参数）.

    职责：
    1. 追踪当日已成交的协议
    2. 计算 L2 目标的剩余缺口
    3. 维护资源约束的动态状态
    4. 广播全局状态给每个 L3 线程

    与原 L4 的区别：
    - 不再是神经网络，无需训练
    - 不再输出"权重"来调制 L3
    - 改为广播全局状态，让 L3 自主决策

    Args:
        horizon: 规划视界 H
    """

    def __init__(self, horizon: int = 40):
        self.horizon = horizon
        self._reset_daily_stats()

        # 外部引用（由 on_step_begin 设置）
        self._awi = None
        self._l1_buy = None
        self._l1_sell = None
        self._l2_output = None
        self._state_dict = None

    def _reset_daily_stats(self):
        """重置每日统计（每个 step 开始时调用）."""
        # 今日成交记录: [(qty, price, delta_t), ...]
        self.today_bought: List[Tuple[float, float, int]] = []
        self.today_sold: List[Tuple[float, float, int]] = []

        # 今日已承诺的资源
        self._committed_buy_qty = np.zeros(self.horizon + 1, dtype=np.float32)
        self._committed_sell_qty = np.zeros(self.horizon + 1, dtype=np.float32)
        self._committed_buy_budget = 0.0

    def on_step_begin(
        self,
        awi,
        l1_buy: "L1Output",
        l1_sell: "L1Output",
        l2_output: "L2Output",
        state_dict: "StateDict"
    ):
        """每个仿真步开始时调用.

        Args:
            awi: Agent World Interface
            l1_buy: 买侧 L1 输出
            l1_sell: 卖侧 L1 输出
            l2_output: L2 日级目标
            state_dict: 当前状态张量
        """
        self._reset_daily_stats()
        self._awi = awi
        self._l1_buy = l1_buy
        self._l1_sell = l1_sell
        self._l2_output = l2_output
        self._state_dict = state_dict

    def on_contract_signed(
        self,
        quantity: int,
        unit_price: float,
        delivery_time: int,
        is_buying: bool
    ):
        """合约签署时更新状态.

        由 agent.on_negotiation_success() 调用。

        Args:
            quantity: 成交数量
            unit_price: 成交单价
            delivery_time: 交货时间（绝对时间）
            is_buying: 是否为买入
        """
        delta_t = delivery_time - self._awi.current_step
        delta_t = max(0, min(delta_t, self.horizon))

        if is_buying:
            self.today_bought.append((quantity, unit_price, delta_t))
            self._committed_buy_budget += quantity * unit_price
            self._committed_buy_qty[delta_t] += quantity
        else:
            self.today_sold.append((quantity, unit_price, delta_t))
            self._committed_sell_qty[delta_t] += quantity

    def compute_broadcast(self) -> GlobalBroadcast:
        """计算并返回全局广播状态.

        Returns:
            GlobalBroadcast: 包含所有全局状态信息
        """
        # 计算目标缺口
        goal_gap_buy, goal_gap_sell = self._compute_goal_gaps()

        # 计算剩余资源
        B_remaining = max(0.0, self._l1_buy.B_free - self._committed_buy_budget)
        Q_safe_remaining = np.maximum(
            0.0,
            self._l1_buy.Q_safe - self._committed_buy_qty
        )
        Q_safe_sell_remaining = np.maximum(
            0.0,
            self._l1_sell.Q_safe_sell - self._committed_sell_qty
        )

        # 统计今日成交（按桶分类）
        today_bought_by_bucket = np.zeros(4, dtype=np.float32)
        today_sold_by_bucket = np.zeros(4, dtype=np.float32)

        for qty, price, delta_t in self.today_bought:
            bucket = delta_to_bucket(delta_t)
            today_bought_by_bucket[bucket] += qty

        for qty, price, delta_t in self.today_sold:
            bucket = delta_to_bucket(delta_t)
            today_sold_by_bucket[bucket] += qty

        return GlobalBroadcast(
            # 时间与进度
            current_step=self._awi.current_step,
            n_steps=self._awi.n_steps,
            step_progress=self._awi.current_step / max(1, self._awi.n_steps),

            # L2 目标与缺口
            l2_goal=self._l2_output.goal_vector.copy(),
            goal_gap_buy=goal_gap_buy,
            goal_gap_sell=goal_gap_sell,

            # 当日成交统计
            today_bought_qty=sum(q for q, _, _ in self.today_bought),
            today_bought_value=sum(q * p for q, p, _ in self.today_bought),
            today_sold_qty=sum(q for q, _, _ in self.today_sold),
            today_sold_value=sum(q * p for q, p, _ in self.today_sold),
            today_bought_by_bucket=today_bought_by_bucket,
            today_sold_by_bucket=today_sold_by_bucket,

            # 资源约束状态
            B_remaining=B_remaining,
            Q_safe_remaining=Q_safe_remaining,
            Q_safe_sell_remaining=Q_safe_sell_remaining,

            # 并发谈判概览（需要从 agent 获取）
            n_active_buy_threads=0,  # 由调用方填充
            n_active_sell_threads=0,

            # 状态张量
            x_static=self._state_dict.x_static.copy(),
            X_temporal=self._state_dict.X_temporal.copy(),
        )

    def _compute_goal_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算每个桶的目标缺口.

        Returns:
            (goal_gap_buy, goal_gap_sell): 各 shape (4,)
        """
        goal = self._l2_output.goal_vector  # (16,)

        # 统计今日成交按桶分布
        bought_by_bucket = np.zeros(4, dtype=np.float32)
        sold_by_bucket = np.zeros(4, dtype=np.float32)

        for qty, price, delta_t in self.today_bought:
            bucket = delta_to_bucket(delta_t)
            bought_by_bucket[bucket] += qty

        for qty, price, delta_t in self.today_sold:
            bucket = delta_to_bucket(delta_t)
            sold_by_bucket[bucket] += qty

        # 计算缺口 = 目标 - 已成交
        # L2 目标结构: [Q_buy^0, P_buy^0, Q_sell^0, P_sell^0, ...] × 4 桶
        goal_gap_buy = np.zeros(4, dtype=np.float32)
        goal_gap_sell = np.zeros(4, dtype=np.float32)

        for i in range(4):
            goal_gap_buy[i] = max(0.0, goal[i * 4] - bought_by_bucket[i])      # Q_buy^i
            goal_gap_sell[i] = max(0.0, goal[i * 4 + 2] - sold_by_bucket[i])   # Q_sell^i

        return goal_gap_buy, goal_gap_sell
```

### 4.4 GlobalCoordinator 自注意力网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalCoordinator(nn.Module):
    """L4 全局协调器：为每个活跃线程计算优先级 α.

    采用自注意力机制，让每个线程"看到"其他线程的状态，
    从而做出全局最优的优先级分配。

    架构：
    1. 线程状态编码：将每个 ThreadState 编码为向量
    2. 自注意力层：线程之间相互关注
    3. 优先级头：输出 α ∈ (-1, 1)

    训练阶段：
    - 离线 BC：用启发式规则生成伪标签预训练
    - 在线 RL：与 L3 联合优化，学习最优分配

    Args:
        d_thread: 线程状态嵌入维度
        d_global: 全局状态嵌入维度
        n_heads: 自注意力头数
        n_layers: Transformer 层数
        max_threads: 最大并发线程数
    """

    def __init__(
        self,
        d_thread: int = 64,
        d_global: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_threads: int = 20,
    ):
        super().__init__()
        self.d_thread = d_thread
        self.d_global = d_global
        self.max_threads = max_threads

        # 线程状态编码器
        # 输入: [is_buying, neg_step, rel_time, offer_q, offer_p, offer_t,
        #        history_len, target_bucket, goal_gap, Q_safe, B_remaining]
        self.thread_encoder = nn.Sequential(
            nn.Linear(11, d_thread),
            nn.ReLU(),
            nn.Linear(d_thread, d_thread),
        )

        # 全局状态编码器（从 GlobalBroadcast 提取关键信息）
        # 输入: [step_progress, B_remaining, goal_gap_buy(4), goal_gap_sell(4),
        #        n_active_buy, n_active_sell]
        self.global_encoder = nn.Sequential(
            nn.Linear(12, d_global),
            nn.ReLU(),
            nn.Linear(d_global, d_global),
        )

        # 位置编码（线程顺序无关，使用可学习 token）
        self.thread_token = nn.Parameter(torch.randn(1, 1, d_thread) * 0.02)

        # 自注意力 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_thread + d_global,
            nhead=n_heads,
            dim_feedforward=(d_thread + d_global) * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 优先级输出头
        self.alpha_head = nn.Sequential(
            nn.Linear(d_thread + d_global, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # 输出 ∈ (-1, 1)
        )

    def forward(
        self,
        thread_states: torch.Tensor,      # (B, N, 11) 线程状态
        global_state: torch.Tensor,       # (B, 12) 全局状态
        thread_mask: torch.Tensor = None, # (B, N) 有效线程掩码
    ) -> torch.Tensor:
        """计算每个线程的优先级 α.

        Args:
            thread_states: 线程状态张量，shape (B, N, 11)
            global_state: 全局状态张量，shape (B, 12)
            thread_mask: 有效线程掩码，shape (B, N)，True = 有效

        Returns:
            alphas: 优先级向量，shape (B, N)，值 ∈ (-1, 1)
        """
        B, N, _ = thread_states.shape

        # 编码线程状态
        thread_emb = self.thread_encoder(thread_states)  # (B, N, d_thread)

        # 编码全局状态并广播
        global_emb = self.global_encoder(global_state)   # (B, d_global)
        global_emb = global_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, d_global)

        # 拼接线程 + 全局
        combined = torch.cat([thread_emb, global_emb], dim=-1)  # (B, N, d_thread + d_global)

        # 构造注意力掩码
        if thread_mask is not None:
            # Transformer 掩码：True = 忽略
            attn_mask = ~thread_mask  # (B, N)
        else:
            attn_mask = None

        # 自注意力
        hidden = self.transformer(
            combined,
            src_key_padding_mask=attn_mask,
        )  # (B, N, d_thread + d_global)

        # 输出优先级
        alphas = self.alpha_head(hidden).squeeze(-1)  # (B, N)

        # 对无效线程置零
        if thread_mask is not None:
            alphas = alphas * thread_mask.float()

        return alphas


class HeuristicAlphaGenerator:
    """启发式 α 生成器（用于离线预训练伪标签）.

    基于规则生成优先级，用于 L4 神经网络的监督预训练。

    规则逻辑：
    1. 资源紧张时（库存低、资金少）→ 高 α（紧急采购）
    2. 目标缺口大时 → 高 α
    3. 谈判进度高时 → 高 α（deadline 压力）
    4. 资源充裕 & 缺口小 → 低 α
    """

    def __init__(self, urgency_threshold: float = 0.3):
        self.urgency_threshold = urgency_threshold

    def compute_alpha(
        self,
        thread_state: ThreadState,
        global_broadcast: GlobalBroadcast,
    ) -> float:
        """为单个线程计算启发式 α.

        Returns:
            alpha: ∈ (-1, 1)
        """
        alpha = 0.0

        # 1. 资源紧张度：库存/预算越低，α 越高
        if thread_state.is_buying:
            resource_ratio = thread_state.B_remaining / max(1.0, global_broadcast.l2_goal[0] * 10)
            if resource_ratio < self.urgency_threshold:
                alpha += 0.3 * (1 - resource_ratio / self.urgency_threshold)

        # 2. 目标缺口紧迫度
        bucket = thread_state.target_bucket
        if thread_state.is_buying:
            gap = global_broadcast.goal_gap_buy[bucket]
        else:
            gap = global_broadcast.goal_gap_sell[bucket]

        if gap > 0:
            alpha += 0.3 * min(1.0, gap / 10.0)

        # 3. 谈判进度（deadline 压力）
        alpha += 0.2 * thread_state.relative_time

        # 4. 紧急桶（bucket 0-1）额外加权
        if bucket <= 1:
            alpha += 0.2

        # 归一化到 (-1, 1)
        alpha = max(-1.0, min(1.0, alpha * 2 - 0.5))

        return alpha
```

### 4.5 L4 组合使用

```python
class L4Layer:
    """L4 层的统一接口：监控器 + 协调器.

    组合使用 L4GlobalMonitor 和 GlobalCoordinator。

    Args:
        horizon: 规划视界
        use_neural_alpha: 是否使用神经网络计算 α（False = 启发式）
        model_path: 预训练模型路径
    """

    def __init__(
        self,
        horizon: int = 40,
        use_neural_alpha: bool = True,
        model_path: Optional[str] = None,
    ):
        self.monitor = L4GlobalMonitor(horizon=horizon)
        self.use_neural_alpha = use_neural_alpha

        if use_neural_alpha:
            self.coordinator = GlobalCoordinator()
            if model_path:
                self.coordinator.load_state_dict(torch.load(model_path))
            self.coordinator.eval()
        else:
            self.heuristic_alpha = HeuristicAlphaGenerator()

        self._thread_states: Dict[str, ThreadState] = {}

    def on_step_begin(self, awi, l1_buy, l1_sell, l2_output, state_dict):
        """每日初始化."""
        self.monitor.on_step_begin(awi, l1_buy, l1_sell, l2_output, state_dict)
        self._thread_states.clear()

    def register_thread(self, thread_id: str, thread_state: ThreadState):
        """注册/更新线程状态."""
        self._thread_states[thread_id] = thread_state

    def on_contract_signed(self, quantity, unit_price, delivery_time, is_buying):
        """合约签署时更新."""
        self.monitor.on_contract_signed(quantity, unit_price, delivery_time, is_buying)

    def compute_broadcast_and_alpha(
        self,
        thread_id: str,
    ) -> Tuple[GlobalBroadcast, float]:
        """计算全局广播和该线程的优先级 α.

        Returns:
            (global_broadcast, alpha): 全局状态 + 该线程的优先级
        """
        broadcast = self.monitor.compute_broadcast()

        if not self.use_neural_alpha:
            # 启发式
            thread_state = self._thread_states.get(thread_id)
            if thread_state:
                alpha = self.heuristic_alpha.compute_alpha(thread_state, broadcast)
            else:
                alpha = 0.0
        else:
            # 神经网络：需要所有线程状态
            alpha = self._compute_neural_alpha(thread_id, broadcast)

        return broadcast, alpha

    def _compute_neural_alpha(self, thread_id: str, broadcast: GlobalBroadcast) -> float:
        """使用神经网络计算 α."""
        # 构造输入张量
        thread_list = list(self._thread_states.values())
        if not thread_list:
            return 0.0

        # 编码线程状态
        thread_tensor = self._encode_threads(thread_list)  # (1, N, 11)
        global_tensor = self._encode_global(broadcast)     # (1, 12)

        with torch.no_grad():
            alphas = self.coordinator(thread_tensor, global_tensor)  # (1, N)

        # 找到目标线程
        for i, ts in enumerate(thread_list):
            if ts.thread_id == thread_id:
                return alphas[0, i].item()

        return 0.0

    def _encode_threads(self, threads: List[ThreadState]) -> torch.Tensor:
        """将 ThreadState 列表编码为张量."""
        features = []
        for t in threads:
            offer = t.current_offer or (0, 0, 0)
            features.append([
                float(t.is_buying),
                t.negotiation_step / 20.0,
                t.relative_time,
                offer[0] / 10.0,
                offer[1] / 100.0,
                offer[2] / 40.0,
                t.history_len / 20.0,
                t.target_bucket / 3.0,
                t.goal_gap / 10.0,
                t.Q_safe_at_t / 100.0,
                t.B_remaining / 10000.0,
            ])
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _encode_global(self, gb: GlobalBroadcast) -> torch.Tensor:
        """将 GlobalBroadcast 编码为张量."""
        features = [
            gb.step_progress,
            gb.B_remaining / 10000.0,
            *[g / 10.0 for g in gb.goal_gap_buy],
            *[g / 10.0 for g in gb.goal_gap_sell],
            gb.n_active_buy_threads / 10.0,
            gb.n_active_sell_threads / 10.0,
        ]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
```

### 4.6 文件修改清单

| 操作 | 文件/类 |
|------|---------|
| 保留/重构 | `l4_coordinator.py` → `L4GlobalMonitor` + `GlobalCoordinator` |
| 新增 | `GlobalBroadcast` dataclass |
| 新增 | `ThreadState` dataclass |
| 新增 | `ThreadPriority` dataclass |
| 新增 | `L4Layer` 组合类 |
| 新增 | `HeuristicAlphaGenerator` 启发式 α 生成器 |
| 删除 | `HeuristicL4Coordinator`（被新设计替代） |
| 更新 | `__init__.py` 导出 |

---

## 五、L3 层重构：Decision Transformer 输出完整 AOP Action

### 5.1 核心变化

| 方面 | 当前设计 | 重构后设计 |
|------|----------|------------|
| **输入** | `baseline + history + goal` | `local_state + global_broadcast + α + history` |
| **输出** | 残差 `(Δq, Δp, δt)` | **完整动作 `SAOAction`**（不输出 α） |
| **决策** | 仅调整 baseline | 自主决定 accept/counter/end，行为受 α 调节 |
| **架构** | MLP / Transformer Encoder | **Decision Transformer** |

**α 信号流向**：
- **L4 产生 α** → **L3 接收 α 作为输入** → **L3 输出 SAOAction**
- L3 不产生也不输出 α，仅根据 α 调节自身行为

**α 对 L3 行为的影响**：

| α 值 | 期望行为特征 |
|------|-------------|
| α → +1 | 加快让步速度、倾向于 accept、报价数量更大 |
| α ≈ 0 | 标准行为（BC 阶段的默认行为） |
| α → -1 | 坚持价格、倾向于 reject/慢节奏谈判、报价数量保守 |

### 5.2 新的 L3 输入结构

```python
@dataclass
class L3Input:
    """L3 执行器的输入.

    包含三部分：
    1. 本线程局部状态：当前谈判的具体信息
    2. L4 广播的全局状态：全局资源和目标信息
    3. L4 计算的线程优先级 α

    注意：α 由 L4 产生，L3 仅接收作为输入，不输出 α。
    """

    # ========== 本线程局部状态 ==========
    is_buying: bool                      # 当前谈判方向
    history: List["NegotiationRound"]    # 谈判历史 (对手报价 + 我方报价)
    current_offer: Optional[Tuple[int, float, int]]  # 对手当前报价 (q, p, t)
    negotiation_step: int                # 谈判轮次（从 0 开始）
    relative_time: float                 # 谈判相对进度 [0, 1]
    partner_id: str                      # 对手 ID

    # ========== L4 广播的全局状态 ==========
    global_broadcast: GlobalBroadcast

    # ========== L4 产生的线程优先级（L3 仅接收） ==========
    alpha: float                         # ∈ (-1, 1)
                                         # BC 阶段固定为 0
                                         # RL 阶段由 L4 动态计算

    # ========== L1 安全掩码（根据 is_buying 选择） ==========
    time_mask: np.ndarray          # shape (H+1,)，0 或 -inf
    Q_safe: np.ndarray             # shape (H+1,)，当前方向的安全量
    B_free: float                  # 可用资金（仅买侧有意义）
```

### 5.3 新的 L3 输出结构

```python
@dataclass
class L3Output:
    """L3 执行器的输出.

    核心输出是 SAOAction，表示 op∈{ACCEPT, COUNTER, END} 的决策。
    """
    action: SAOAction

    # 辅助信息（用于调试和训练）
    time_probs: Optional[np.ndarray] = None       # shape (H+1,)，时间选择概率
    op_probs: Optional[np.ndarray] = None  # shape (3,)，op 概率（ACCEPT/COUNTER/END）
    confidence: float = 1.0                        # 决策置信度
```

### 5.4 启发式 L3 实现

```python
class HeuristicL3Actor:
    """启发式 L3 执行器（无需训练）.

    决策逻辑：

    1. Accept 条件：
       - 价格在可接受范围（买方：≤ P_limit × 1.1；卖方：≥ P_floor × 0.9）
       - 数量在安全范围
       - 时间可行（time_mask 允许）

    2. Reject 条件（终止谈判）：
       - 谈判即将超时 (relative_time > 0.9) 且对方报价完全不可接受
       - 安全约束不允许任何交易

    3. Offer 逻辑：
       - 选择最优交货时间（基于 L2 目标缺口）
       - 数量 = min(目标缺口, 安全量)
       - 价格 = 基于市场价和谈判进度的让步曲线
    """

    def __init__(self, horizon: int = 40):
        self.horizon = horizon

    def compute(self, l3_input: L3Input) -> L3Output:
        """计算 L3 动作.

        Args:
            l3_input: L3 输入（局部状态 + 全局广播）

        Returns:
            L3Output: 包含 SAOAction
        """
        # 1. 检查是否应该接受
        if self._should_accept(l3_input):
            return L3Output(
                action=SAOAction(action_type="accept"),
                confidence=0.9
            )

        # 2. 检查是否应该拒绝（终止）
        if self._should_reject(l3_input):
            return L3Output(
                action=SAOAction(action_type="end"),
                confidence=0.8
            )

        # 3. 生成报价
        return self._generate_offer(l3_input)

    def _should_accept(self, l3_input: L3Input) -> bool:
        """判断是否应该接受对方报价."""
        offer = l3_input.current_offer
        if offer is None:
            return False

        q, p, t = offer
        gb = l3_input.global_broadcast
        delta_t = t - gb.current_step

        if delta_t < 0 or delta_t > self.horizon:
            return False

        # 检查时间可行性
        if l3_input.time_mask[delta_t] == -np.inf:
            return False

        # 检查数量安全性
        if q > l3_input.Q_safe[delta_t]:
            return False

        # 检查资金（买侧）
        if l3_input.is_buying and q * p > l3_input.B_free:
            return False

        # 获取目标价格
        bucket = delta_to_bucket(delta_t)
        if l3_input.is_buying:
            target_price = gb.l2_goal[bucket * 4 + 1]  # P_buy
            return p <= target_price * 1.1  # 允许 10% 溢价
        else:
            target_price = gb.l2_goal[bucket * 4 + 3]  # P_sell
            return p >= target_price * 0.9  # 允许 10% 折扣

    def _should_reject(self, l3_input: L3Input) -> bool:
        """判断是否应该拒绝并终止谈判."""
        # 如果没有任何可行的交货时间，拒绝
        valid_times = np.where(l3_input.time_mask > -np.inf)[0]
        if len(valid_times) == 0:
            return True

        # 如果快超时且对方报价偏离太大，拒绝
        if l3_input.relative_time > 0.95:
            offer = l3_input.current_offer
            if offer is not None:
                q, p, t = offer
                gb = l3_input.global_broadcast
                delta_t = t - gb.current_step
                bucket = delta_to_bucket(max(0, min(delta_t, self.horizon)))

                if l3_input.is_buying:
                    target_price = gb.l2_goal[bucket * 4 + 1]
                    if p > target_price * 1.5:  # 偏离 50% 以上
                        return True
                else:
                    target_price = gb.l2_goal[bucket * 4 + 3]
                    if p < target_price * 0.5:
                        return True

        return False

    def _generate_offer(self, l3_input: L3Input) -> L3Output:
        """生成报价."""
        gb = l3_input.global_broadcast

        # 选择最优交货时间（目标缺口最大的桶）
        if l3_input.is_buying:
            gaps = gb.goal_gap_buy
        else:
            gaps = gb.goal_gap_sell

        # 按缺口大小排序，选择最大且可行的
        bucket_order = np.argsort(-gaps)  # 降序

        best_delta = None
        for bucket in bucket_order:
            lo, hi = BUCKET_RANGES[bucket]
            # 在桶范围内找第一个可行时间
            for delta in range(lo, min(hi + 1, self.horizon + 1)):
                if delta < len(l3_input.time_mask) and l3_input.time_mask[delta] > -np.inf:
                    best_delta = delta
                    break
            if best_delta is not None:
                break

        if best_delta is None:
            # 没有可行时间，拒绝
            return L3Output(action=SAOAction(action_type="end"))

        best_bucket = delta_to_bucket(best_delta)

        # 计算数量
        gap = gaps[best_bucket]
        safe_qty = l3_input.Q_safe[best_delta]
        q = int(max(1, min(gap, safe_qty)))

        # 买侧还需检查资金
        if l3_input.is_buying:
            base_price = gb.l2_goal[best_bucket * 4 + 1]
            max_qty_by_budget = l3_input.B_free / max(base_price, 1.0)
            q = int(max(1, min(q, max_qty_by_budget)))

        # 价格让步曲线
        concession = l3_input.relative_time * 0.1  # 最多让步 10%
        if l3_input.is_buying:
            base_price = gb.l2_goal[best_bucket * 4 + 1]  # P_buy
            p = base_price * (1 + concession)  # 买方逐渐提高出价
        else:
            base_price = gb.l2_goal[best_bucket * 4 + 3]  # P_sell
            p = base_price * (1 - concession)  # 卖方逐渐降低要价

        return L3Output(
            action=SAOAction(
                action_type="counter",
                quantity=q,
                unit_price=float(p),
                delivery_time=best_delta + gb.current_step,
            ),
            confidence=0.7
        )
```

### 5.5 神经网络 L3 架构

### 5.5.1 架构选择：Decision Transformer

**设计决策（2025-12-25 更新）**：采用 **Decision Transformer**。

#### 架构说明

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Decision Transformer (采用)                       │
├─────────────────────────────────────────────────────────────────────┤
│  输入序列（每个 timestep 三元组）:                                    │
│  [α, s₁, a₁] → [α, s₂, a₂] → ... → [α, sₜ, ?]                       │
│                                                                     │
│  其中:                                                               │
│  - α: 线程优先级（由 L4 产生，替代传统 DT 的 Return-to-go）           │
│  - sₜ: 当前状态（报价、谈判进度、全局广播等）                          │
│  - aₜ: 动作（accept/counter/end）                                  │
│                                                                     │
│                           ↓                                         │
│            GPT-style Decoder (因果注意力，只看过去)                   │
│                           ↓                                         │
│                    预测 aₜ (SAOAction)                              │
│                           ↓                                         │
│    ┌──────────┬──────────┬──────────┬──────────┐                   │
│    │action_   │quantity  │price     │time      │                   │
│    │type_head │head      │head      │head      │                   │
│    └──────────┴──────────┴──────────┴──────────┘                   │
│                                                                     │
│  特点：α 作为条件变量控制策略激进/保守程度                            │
└─────────────────────────────────────────────────────────────────────┘
```

#### 选择 Decision Transformer 的理由

| 维度 | 说明 |
|------|------|
| **条件变量** | 用 α（优先级）替代传统 Return-to-go，α 由 L4 产生 |
| **α 作用** | 高 α → 激进策略（快速成交）；低 α → 保守策略（坚持价格） |
| **因果注意力** | 只看过去的 (s, a) 对，符合在线决策场景 |
| **序列建模** | 天然支持变长谈判历史 |
| **训练兼容** | BC 阶段 α=0，RL 阶段 α 由 L4 动态产生 |

**α 与 Return-to-go 的对比**：

| 方面 | 传统 Return-to-go | α 优先级（本设计） |
|------|-------------------|-------------------|
| 语义 | 期望累积回报 | 行为激进/保守程度 |
| 来源 | 离线计算或人工设定 | L4 实时计算 |
| 训练 | 需要轨迹回报标注 | BC 阶段固定为 0 |
| 可控性 | 回报数值抽象 | 直观的行为控制 |

### 5.5.2 L3DecisionTransformer 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class L3DecisionTransformer(nn.Module):
    """L3 Decision Transformer：基于 α 条件的动作生成器.

    架构遵循 Decision Transformer 论文，但用 α（优先级）替代 Return-to-go。
    α 由 L4 产生，L3 仅接收作为输入，不输出 α。

    输入序列结构（每个 timestep）：
    [α_embed, state_embed, action_embed] → 3 tokens per step

    输出：
    - op_logits: (B, 4) - reject/accept/offer
    - quantity_ratio: (B, 1) - 数量比例
    - price_ratio: (B, 1) - 价格比例
    - time_logits: (B, H+1) - 交货时间

    训练阶段：
    - BC 阶段：α 固定为 0
    - RL 阶段：α 由 L4 动态产生

    Args:
        horizon: 规划视界
        d_model: 模型隐藏维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        max_seq_len: 最大序列长度（历史轮数 × 3）
    """

    def __init__(
        self,
        horizon: int = 40,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 60,  # 20轮 × 3 tokens
        dropout: float = 0.1,
    ):
        super().__init__()

        self.horizon = horizon
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # ========== Token 嵌入 ==========
        # α 嵌入（线程优先级，由 L4 产生）
        self.alpha_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )

        # 状态嵌入
        # state = [is_buying, rel_time, offer_q, offer_p, offer_t, goal_gap, Q_safe, B_remaining]
        self.state_embed = nn.Sequential(
            nn.Linear(8, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 动作嵌入
        # action = [op_onehot(4), q_ratio, p_ratio, time_onehot(H+1)]
        self.action_embed = nn.Sequential(
            nn.Linear(3 + 2 + horizon + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 全局状态嵌入（GlobalBroadcast 精简版）
        self.global_embed = nn.Linear(32, d_model)  # 精简后的全局特征

        # 角色嵌入
        self.role_embed = nn.Embedding(2, d_model)  # 0=buyer, 1=seller

        # Token 类型嵌入 (0=α, 1=state, 2=action)
        self.token_type_embed = nn.Embedding(3, d_model)

        # 位置编码
        self.pos_embed = nn.Embedding(max_seq_len + 2, d_model)

        # ========== GPT-style Decoder ==========
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 用于 cross-attention 的 memory（角色 + 全局状态）
        self.memory_proj = nn.Linear(d_model * 2, d_model)

        # LayerNorm
        self.ln = nn.LayerNorm(d_model)

        # ========== 输出头 ==========
        # 动作类型：reject=0, accept=1, offer=2
        self.op_head = nn.Linear(d_model, 4)

        # 数量比例：[0, 1]
        self.quantity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # 价格比例：[0, 1]
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # 时间 logits
        self.time_head = nn.Linear(d_model, horizon + 1)

    def forward(
        self,
        alpha: torch.Tensor,            # (B,): 线程优先级，由 L4 产生
        history_states: torch.Tensor,   # (B, T, 8): 历史状态序列
        history_actions: torch.Tensor,  # (B, T-1, action_dim): 历史动作序列
        current_state: torch.Tensor,    # (B, 8): 当前状态
        role: torch.Tensor,             # (B,): 0=buyer, 1=seller
        global_feat: torch.Tensor,      # (B, 32): 精简全局特征
        time_mask: torch.Tensor,        # (B, H+1): 0 or -inf
        Q_safe: torch.Tensor,           # (B, H+1)
        price_range: torch.Tensor,      # (B, 2): [min_price, max_price]
    ):
        """前向传播.

        构建序列: [α, s₁, a₁, α, s₂, a₂, ..., α, sₜ] → 预测 aₜ

        注意：α 在每个 timestep 都作为条件输入，但 L3 不输出 α。

        Returns:
            op_logits: (B, 4)
            quantity: (B, 1) 缩放后的数量
            price: (B, 1) 缩放后的价格
            time_logits: (B, H+1) masked
        """
        B = alpha.shape[0]
        T = history_states.shape[1]

        # 1. 构建输入序列
        tokens = []
        token_types = []

        alpha_emb = self.alpha_embed(alpha.unsqueeze(-1))  # (B, d)

        for t in range(T - 1):
            # α token
            tokens.append(alpha_emb)
            token_types.append(torch.zeros(B, dtype=torch.long, device=alpha.device))

            # state token
            s_emb = self.state_embed(history_states[:, t, :])
            tokens.append(s_emb)
            token_types.append(torch.ones(B, dtype=torch.long, device=alpha.device))

            # action token (如果不是最后一个)
            if t < history_actions.shape[1]:
                a_emb = self.action_embed(history_actions[:, t, :])
                tokens.append(a_emb)
                token_types.append(torch.full((B,), 2, dtype=torch.long, device=alpha.device))

        # 最后一个 timestep：α, current_state（预测 action）
        tokens.append(alpha_emb)
        token_types.append(torch.zeros(B, dtype=torch.long, device=alpha.device))

        tokens.append(self.state_embed(current_state))
        token_types.append(torch.ones(B, dtype=torch.long, device=alpha.device))

        # 堆叠
        seq = torch.stack(tokens, dim=1)  # (B, L, d)
        token_types = torch.stack(token_types, dim=1)  # (B, L)

        L = seq.shape[1]

        # 添加 token 类型嵌入和位置编码
        seq = seq + self.token_type_embed(token_types)
        positions = torch.arange(L, device=seq.device)
        seq = seq + self.pos_embed(positions)

        # 2. 构建 memory（角色 + 全局状态）
        role_emb = self.role_embed(role)  # (B, d)
        global_emb = self.global_embed(global_feat)  # (B, d)
        memory = self.memory_proj(torch.cat([role_emb, global_emb], dim=-1))  # (B, d)
        memory = memory.unsqueeze(1)  # (B, 1, d)

        # 3. 因果掩码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(seq.device)

        # 4. Transformer Decoder
        h = self.transformer(
            seq,
            memory,
            tgt_mask=causal_mask,
        )
        h = self.ln(h[:, -1, :])  # 取最后位置 (B, d)

        # 5. 输出头
        op_logits = self.op_head(h)

        q_ratio = self.quantity_head(h)
        p_ratio = self.price_head(h)
        time_logits = self.time_head(h)

        # 应用 time_mask
        time_logits = time_logits + time_mask

        # 缩放数量
        time_probs = F.softmax(time_logits, dim=-1)
        expected_Q_safe = (time_probs * Q_safe).sum(dim=-1, keepdim=True)
        quantity = q_ratio * expected_Q_safe

        # 缩放价格
        price = price_range[:, 0:1] + p_ratio * (price_range[:, 1:2] - price_range[:, 0:1])

        return op_logits, quantity, price, time_logits
```

---

## 六、Agent 层修改（对齐“方案A：L4=网络 + α”）

> **重要更正**：本方案已确认 **不选方案B（不保留 batch_planner 动态预留/裁剪）**，但 **L4 仍然是网络**（输出 α），因此 Agent 层需要同时接入：
> - `L4GlobalMonitor`：负责广播 GlobalBroadcast（目标缺口、已签约量、Q_safe/B_remaining 等）
> - `GlobalCoordinator`：自注意力网络，输入所有活跃线程显式状态，输出每线程 α

### 6.1 agent.py 变更清单（与旧代码逐项对照）

| 变更 | 旧实现（当前代码） | 新实现（本方案） |
|------|-------------------|------------------|
| L4 成员 | `self.l4 = L4ThreadCoordinator(...)`（输出 weights+modulation） | `self.l4_monitor = L4GlobalMonitor(...)` + `self.l4_coord = GlobalCoordinator(...)`（输出 α） |
| L4 输入 | `thread_feat + target_time + role + global_feat` | **仍然是显式特征**（不接 L3 hidden），但需**去除 baseline 依赖**（见 6.3） |
| 线程调度 | `batch_planner` 依赖 L4 weights 顺序裁剪 Q/B | **移除**。不再做“按权重顺序扣减并裁剪动作” |
| L3 输出 | `(Δq, Δp, Δt)` 残差 + hidden_state | **完整 SAOAction**：`accept/counter/end(q,p,t_abs)` |
| L1 输出 | baseline + mask | **仅 mask**（time_mask、Q_safe、Q_safe_sell、B_free），不再有 baseline |
| 安全裁剪 | clip_action 输入为 `(q,p,delta_t)` | **仍用 clip_action**，但必须做 **abs_t ↔ delta_t** 转换（见 6.2） |
| 接受动作可行性 | `_should_accept()` 内含价格阈值 + 资源检查 | **接受动作的 mask/校验仅做硬约束**（不含价格），价格优劣交给策略/在线学习（见 6.2.2） |

### 6.2 核心实现框架（建议写法）

#### 6.2.1 before_step：构建 L1/L2/L4 的“当日快照”

```python
def before_step(self):
    super().before_step()

    # 1) 全局状态（原有）
    self._current_state = self.state_builder.build(self.awi)

    # 2) L1（新：不再有 baseline_action）
    self._step_l1_buy  = self.l1.compute(self.awi, is_buying=True)
    self._step_l1_sell = self.l1.compute(self.awi, is_buying=False)

    # 3) L2（原有）
    self._current_l2_output = self.l2.compute(
        self._current_state.x_static,
        self._current_state.X_temporal,
        is_buying=True,     # L2 输出包含买卖两侧桶目标
        awi=self.awi,
    )

    # 4) L4 monitor：初始化当日统计（必须）
    self.l4_monitor.on_step_begin(
        awi=self.awi,
        l1_buy=self._step_l1_buy,
        l1_sell=self._step_l1_sell,
        l2_output=self._current_l2_output,
        state_dict=self._current_state,
    )
```

#### 6.2.2 respond/propose：每次决策前“快照化” α 与广播

关键点有三个：

1) **L4 计算 α** 必须看到 **全部活跃线程的显式状态**（不依赖调用顺序）  
2) **L3 输入** 包含 `GlobalBroadcast` 与本线程 `α`  
3) **L1.clip_action** 仍然按 `delta_t` 索引，因此对 `SAOAction.delivery_time` 需要转换

```python
def _compute_alpha_map(self) -> dict[str, float]:
    """返回 {thread_id: alpha}，对所有活跃线程一次性计算。"""
    threads = self._gather_thread_states()   # List[ThreadState]（新：不含 baseline 依赖）
    if not threads:
        return {}

    # 旧代码里 global_feat 是 (16 goal + 12 x_static + 2 counts) = 30 维
    global_feat = self._build_l4_global_feat(threads)     # np.ndarray (30,)

    # 组装 L4 输入（pad+mask）
    thread_feats, thread_times, thread_roles, attn_mask, thread_ids =         pack_threads_for_l4(threads)

    # 输出 alpha (N,)
    alpha = self.l4_coord.compute_alpha(
        thread_feats, thread_times, thread_roles, global_feat, attn_mask
    )
    return {tid: float(a) for tid, a in zip(thread_ids, alpha)}
```

```python
def respond(self, negotiator_id: str, state: SAOState) -> SAOResponse:
    offer = state.current_offer
    if offer is None:
        return SAOResponse(ResponseType.END_NEGOTIATION, None)  # SCML: REJECT_OFFER 必须带 counter_offer

    # 1) L4 广播 + α
    gb = self.l4_monitor.compute_broadcast()
    alpha_map = self._compute_alpha_map()
    alpha = alpha_map.get(negotiator_id, 0.0)

    # 2) L3 输入
    l3_input = self._build_l3_input(negotiator_id, state, gb=gb, alpha=alpha)
    l3_out = self.l3.compute(l3_input)
    action = l3_out.action

    # 3) “接受动作”的硬约束校验（只管能不能履约，不管划不划算）
    if action.op == 0:  # ACCEPT
        if not self._is_accept_feasible(negotiator_id, offer, gb):
            # 不可行则强制拒绝（或改为 offer）
            return SAOResponse(ResponseType.END_NEGOTIATION, None)  # SCML: REJECT_OFFER 必须带 counter_offer
        return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

    if action.op == 2:  # END (END_NEGOTIATION)
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    # 4) offer：转 Outcome + L1 安全裁剪（注意 time 变换）
    outcome = action.to_outcome()
    if outcome is None or outcome[0] <= 0:
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    q, abs_t, p = outcome
    delta_t = int(abs_t) - int(gb.current_step)
    delta_t = max(0, min(delta_t, self.horizon))

    is_buying = self._is_buying(negotiator_id)
    l1_out = self._step_l1_buy if is_buying else self._step_l1_sell

    # 关键：clip_action 期望 (q,p,delta_t)
    q2, p2, dt2 = self.l1.clip_action(
        action=(q, p, delta_t),
        Q_safe=(l1_out.Q_safe if is_buying else np.zeros_like(l1_out.Q_safe)),
        B_free=l1_out.B_free,
        is_buying=is_buying,
        Q_safe_sell=(l1_out.Q_safe_sell if not is_buying else None),
    )
    if q2 <= 0:
        return SAOResponse(ResponseType.END_NEGOTIATION, None)

    clipped_abs_t = int(gb.current_step) + int(dt2)
    return SAOResponse(ResponseType.REJECT_OFFER, (int(q2), clipped_abs_t, float(p2)))
```

#### 6.2.3 接受动作“只屏蔽必然违约/浪费”的实现点

这部分对应你提出的口径（**不考虑价格好坏**）：

- 买入：`qty <= Q_safe_remaining[delta_t]` 且 `qty * price <= B_remaining`
- 卖出：`qty <= Q_safe_sell_remaining[delta_t]`
- 交期：`0 <= delta_t <= H` 且 `time_mask[delta_t]` 可行

```python
def _is_accept_feasible(self, negotiator_id: str, offer: Outcome, gb: GlobalBroadcast) -> bool:
    q, abs_t, p = offer
    delta_t = int(abs_t) - int(gb.current_step)
    if delta_t < 0 or delta_t > self.horizon:
        return False

    is_buying = self._is_buying(negotiator_id)
    l1_out = self._step_l1_buy if is_buying else self._step_l1_sell
    if l1_out.time_mask[delta_t] == -np.inf:
        return False

    if is_buying:
        if q > gb.Q_safe_remaining[delta_t]:
            return False
        if float(q) * float(p) > float(gb.B_remaining):
            return False
    else:
        if q > gb.Q_safe_sell_remaining[delta_t]:
            return False

    return True
```

### 6.3 与旧实现的关键冲突点（必须改）

旧代码里 `_build_l4_thread_state()` 会使用：

- `baseline_q, baseline_p, baseline_t = ctx.l1_output.baseline_action`

但新 L1 已移除 baseline，因此该函数必须重写。建议用下列信息替代 baseline：

- 本线程 **对手最近报价**（q,p,delta_t）  
- 本线程 **我方最近一次报价**（如果存在）  
- 目标缺口（bucket gap）与当前交期 bucket  
- 安全量比例（Q_safe at delta）与预算比例（B_remaining）  
- 谈判进度（round, relative_time）

这样仍然是“显式特征”，并可从在线状态与离线日志中重建。

### 6.4 ThreadState v2（去 baseline）——需要在文档与代码中显式落地的修改点

> 你提到的 `_build_l4_thread_state()` **确实必须改**：旧实现把 `baseline_action=(q_base,p_base,t_base)` 当作 L4 的输入特征之一（同时还用 `t_base` 做 `target_delta` 的兜底），而新设计里 **L1 不再输出 baseline**，因此旧字段不可能再被可靠重建。

下面给出**最小侵入**的改法：尽量不改 `thread_feat_dim=24`，只把 baseline 相关的维度替换成“可从 AOP 谈判历史直接重建”的字段；并把 `target_delta` 的兜底逻辑改为“由 L2 目标 + L1 可行性”推导。

#### 6.4.1 旧实现中 baseline 被用在什么地方

旧实现（`agent.py::_build_l4_thread_state`）对 baseline 的依赖主要有 2 处：

1) **特征向量里直接塞 baseline 的 (q,p)**  
`feat[4] = baseline_q / max_inv`  
`feat[5] = baseline_p / max_price`

2) **当 `ctx.last_delta_t` 缺失时，用 baseline_t 兜底得到 `target_delta`**  
```python
if ctx.last_delta_t is None:
    target_delta = int(baseline_t)
else:
    target_delta = int(ctx.last_delta_t)
```

这两处在“无 baseline”的新设计里都会失效，因此必须改。

#### 6.4.2 推荐的 ThreadState/thread_feat 改法（保持 24 维不变）

我们保持 `thread_feat ∈ R^{24}` 的整体结构不变（减少对 `GlobalCoordinator` 的连锁修改），仅替换 baseline 相关维度：

- `feat[4]`：从 **baseline_q** 改为 **last_opp_q（或 last_offer_q）**
- `feat[5]`：从 **baseline_p** 改为 **last_opp_p（或 last_offer_p）**
- `target_delta` 兜底：从 **baseline_t** 改为 **由 L2 目标推断的 fallback_delta**

**last_opp_q/last_opp_p 如何取？**

- 若处在 `respond()`（有 `state.current_offer`），直接用“对手当前报价”作为 last_opp_offer
- 若处在 `propose()`（对手未报），则用 `ctx.history` 中最新一条 `is_my_turn=False` 的记录；若不存在则退化为“最新一条报价”（不区分我方/对手）；再不行则置 0。

这样做的关键好处是：**完全不依赖 baseline**，并且可以从 strict JSON 日志（offer_made/offer_received 序列）离线重建。

#### 6.4.3 `target_delta` 的新兜底逻辑（不依赖 baseline_t）

当 `ctx.last_delta_t` 缺失时，我们需要一个“合理的交期关注点”，否则 L4 的线程特征里会出现大量不稳定噪声。

一种简单且可解释的兜底方法：

1) 先从 L2 目标里选一个“最重要桶”  
- 买方线程：`bucket* = argmax_b Q_buy[b]`
- 卖方线程：`bucket* = argmax_b Q_sell[b]`

2) 在该桶的 δ 范围内，选一个 **最早可行** 的 δ  
- 可行标准只看 L1：`time_mask[δ]` 可行 且（买方）`Q_safe[δ] > 0`（卖方可用 `Q_safe_sell[δ] > 0`）

3) 如果桶内都不可行，则在全局 δ∈[0,H] 扫描最早可行者；若仍无可行则默认 δ=0

伪代码：

```python
def fallback_target_delta(is_buying, l1_out, l2_out, H):
    # 1) pick bucket
    goals = l2_out.goal_vector.reshape(4, 4)  # [bucket, (Q_buy,P_buy,Q_sell,P_sell)]
    q_idx = 0 if is_buying else 2
    bucket = int(np.argmax(goals[:, q_idx]))

    # 2) scan delta inside bucket
    dmin, dmax = BUCKET_RANGES[bucket]
    dmin, dmax = max(0, dmin), min(H, dmax)
    for d in range(dmin, dmax + 1):
        if l1_out.time_mask[d] != -np.inf:
            if is_buying and l1_out.Q_safe[d] > 0:
                return d
            if (not is_buying) and l1_out.Q_safe_sell[d] > 0:
                return d

    # 3) scan globally
    for d in range(0, H + 1):
        if l1_out.time_mask[d] != -np.inf:
            if is_buying and l1_out.Q_safe[d] > 0:
                return d
            if (not is_buying) and l1_out.Q_safe_sell[d] > 0:
                return d
    return 0
```

> 注意：这个兜底只是为了让 L4 的输入“稳定且可重建”，不是为了替代 L3 的决策。

#### 6.4.4 这是否需要同步修改 L4 输入与 data pipeline？

**需要。**原因很简单：你改了 `_build_l4_thread_state()` 的显式特征定义，那么：

- 在线推理时 L4 网络看到的是新 thread_feat
- 离线训练/蒸馏时也必须构造**同样定义**的 thread_feat，否则会出现“训练—推理分布不一致”（distribution shift）

因此文档里必须写清楚：
- thread_feat 每一维的语义（至少把 baseline 相关维度替换规则写清）
- 离线 pipeline 如何在 strict JSON ONLY 条件下重建 last_opp_offer 与 fallback_target_delta

（下面 Data Pipeline 章节会补上对应修改点。）

## 七、Data Pipeline 修改

### 7.1 新的 MicroSample 结构

```python
@dataclass
class MicroSample:
    """轮级样本（AOP 格式）.

    变化：
    - 移除 baseline, residual（不再是残差学习）
    - 新增 action_op 标签（ACCEPT/COUNTER/END）
    - 新增全局状态特征
    """
    negotiation_id: str

    # ========== 输入特征 ==========
    history: np.ndarray          # shape (T, 4): [q, p, delta_t, is_my_turn]
    role: int                    # 0=Buyer, 1=Seller
    current_offer: np.ndarray    # shape (3,): [q, p, delta_t]
    relative_time: float

    # 全局状态
    l2_goal: np.ndarray          # shape (16,)
    goal_gap_buy: np.ndarray     # shape (4,)
    goal_gap_sell: np.ndarray    # shape (4,)
    today_bought_qty: float
    today_sold_qty: float
    today_bought_by_bucket: np.ndarray  # shape (4,)
    today_sold_by_bucket: np.ndarray    # shape (4,)

    x_static: np.ndarray         # shape (12,)
    X_temporal: np.ndarray       # shape (H+1, 10)
    time_mask: np.ndarray        # shape (H+1,)
    Q_safe: np.ndarray           # shape (H+1,)
    B_free: float

    # ========== 标签 ==========
    action_op: int               # 0=ACCEPT, 1=COUNTER, 2=END
    # 说明：PenguinAgent 的 END 多为“隐式 END”（first_proposals/counter_all 省略键 → 控制器补 END）。因此 BC 必须包含 END，且需要 tracker 显式补齐该标签（见 7.2.4）。
    target_quantity: Optional[int] = None
    target_price: Optional[float] = None
    target_time: Optional[int] = None  # delta_t

    reward: Optional[float] = None
```

### 7.2 提取函数修改（对齐现有 tracker JSON 聚合结果）

> **重要更正**：当前代码的数据管线（`data_pipeline.py`）会先把 tracker 的离散事件聚合为 `neg_logs`，其谈判结构包含：
> - `offer_history`: 列表，每条含 `{quantity, price, delta_t, source(=self/opponent), round}`
> - `final_status`: `succeeded/failed`
> - `sim_step`: 当天 step
>
> 这意味着离线阶段 **不应该依赖** `final_action["type"]` 之类的字段（旧格式不存在），而应当通过 `offer_history.source` 与 `final_status` **推断 action_op**。

> **迁移目标**：为了让 BC 覆盖 **END/ACCEPT** 并避免线上 OOD，我们最终必须在 **tracker** 中新增 `aop_action` 事件来**显式记录动作**（含 PenguinAgent 的“隐式 END”），并让 pipeline **优先使用**该事件（见 7.2.4）。
> 在迁移完成前，下文的“后验推断”规则仅用于兼容旧数据。

#### 7.2.1 目标：从同一条谈判生成“多条”AOP 样本

- **每次我方出价（offer_made）** → 产生一个 `action_op=COUNTER` 的样本（propose 或 counter-offer）  
- **每次我方收到对手报价（offer_received）** → 产生一个 respond 样本，其 `action_op` 由我方当轮响应决定：  
  - 当轮出现 `accept` → `action_op=ACCEPT`  
  - 当轮出现我方 `offer_made(reason==counter_offer)` → `action_op=COUNTER`（还价）  
  - 否则（既未 accept 也未 counter）：在 SCML 语义下应视为 **结束谈判** → `action_op=END`  
    > 因为“拒绝但不提出还价（REJECT_OFFER 且 outcome=None）”是不合法的。若控制器采用“未响应=END”的默认补全逻辑，则应在 tracker/pipeline 中把这类情况标注为 END。

> 说明：如果你使用的日志只记录 `offer_made/offer_received/success/failure`，没有显式 `accept/end`，仍可通过 **同一 round 内是否出现 counter_offer、以及成功时最后一条 offer 的 source** 做推断，但会有少量歧义。

  - **数据量与可行性提醒（结合当前代码/日志现状）**：
    - 你目前给的示例 strict-JSON 日志中，`negotiation` 事件只有 `started/offer_made/offer_received/success/failure`，**没有显式的 accept/counter/end 动作事件（也没有 reject_no_counter 这种显式事件）**（因此 `END` 样本在现有日志里几乎无法可靠构造）。
    - 同时，现版本 `tracker_mixin.py::patched_on_negotiation_failure()` 记录的 `partner` 可能等于 self.id（示例日志里确实如此），这会进一步破坏“按 partner 复原谈判轨迹”的推断方法。
    - 结论：**如果不改日志采集，`END` 标签要么数量≈0，要么只能靠非常噪声的后验推断**（不建议在 BC 初期做）。

  - **必须做法：在采集阶段直接记录 AOP 动作类型（用于 END/ACCEPT 标签，避免 OOD）**  
    既然新 L3 会显式输出 `op∈{ACCEPT, COUNTER, END}`，最稳妥的做法是在 `respond()/propose()` 里（或在 tracker 里 hook 住它们）写入一条自定义事件，例如：
    ```json
    {"category":"custom","event":"aop_action",
     "data":{"negotiator_id": "...", "partner":"...", "role":"buyer/seller",
             "round": r, "action_op":"ACCEPT/COUNTER/END",
             "offer": {"q":..,"p":..,"delta_t":..}}}
    ```
    这样离线 pipeline 就能**无歧义**统计并抽取 END（结束谈判）样本，并且可以做 class-weight/重采样。

  - **样本量怎么快速统计？**（一行公式 + 一个脚本就够）
    设三类动作样本数为 `N_end, N_accept, N_counter`，则：
    \[
        \text{end\_ratio}=\frac{N_{end}}{N_{end}+N_{accept}+N_{counter}}
    \]
    你只要把所有日志里 `event=="aop_action"` 的 `action_op` 计数即可。



#### 7.2.2 样本构造规则（伪代码）

```python
def extract_l3_samples_aop(neg_logs, daily_states, horizon=40):
    samples = []
    for neg in neg_logs:
        day = int(neg.get("sim_step", 0))
        hist = neg.get("offer_history", [])
        if not hist:
            continue

        # 1) 构建当天的 L1 mask（离线可重建）
        #    说明：只要能重建 time_mask / Q_safe / Q_safe_sell / B_free 即可，
        #    不要求 baseline。
        l1_buy  = compute_l1_masks_offline(daily_states[day], is_buying=True,  horizon=horizon)
        l1_sell = compute_l1_masks_offline(daily_states[day], is_buying=False, horizon=horizon)

        is_buyer = bool(neg.get("is_buyer", True))
        l1_out = l1_buy if is_buyer else l1_sell

        # 2) 预先计算 max_round 以获得 relative_time（近似）
        rounds = [h.get("round") for h in hist if h.get("round") is not None]
        max_round = max(rounds) if rounds else max(1, len(hist) - 1)

        # 3) 扫描历史：每次我方出价都形成一个 “offer” 监督样本
        last_opponent_offer = None
        for i, h in enumerate(hist):
            if h.get("source") == "opponent":
                last_opponent_offer = h

            if h.get("source") != "self":
                continue

            # current_offer：用“我方出价之前最近一次对手报价”
            cur = None
            if last_opponent_offer is not None:
                cur = (
                    int(last_opponent_offer["quantity"]),
                    float(last_opponent_offer["price"]),
                    int(last_opponent_offer["delta_t"]),
                )

            rel_time = float((h.get("round", i)) / max_round)
            samples.append(MicroSampleAOP(
                history=hist[:i],                 # 不含本次动作
                current_offer=cur,
                is_buyer=is_buyer,
                sim_step=day,
                relative_time=rel_time,
                time_mask=l1_out.time_mask,
                Q_safe=(l1_out.Q_safe if is_buyer else l1_out.Q_safe_sell),
                B_free=float(l1_out.B_free),
                action_op=1,                    # COUNTER（出价/还价）
                target_q=int(h["quantity"]),
                target_p=float(h["price"]),
                target_delta_t=int(h["delta_t"]),
            ))

        # 4) accept 样本：成功且最后报价来自对手 → 我方 accept
        if neg.get("final_status") == "succeeded" and hist[-1].get("source") == "opponent":
            h = hist[-1]
            rel_time = 1.0
            samples.append(MicroSampleAOP(
                history=hist,
                current_offer=(int(h["quantity"]), float(h["price"]), int(h["delta_t"])),
                is_buyer=is_buyer,
                sim_step=day,
                relative_time=rel_time,
                time_mask=l1_out.time_mask,
                Q_safe=(l1_out.Q_safe if is_buyer else l1_out.Q_safe_sell),
                B_free=float(l1_out.B_free),
                action_op=0,          # ACCEPT
                target_q=0, target_p=0.0, target_delta_t=0,
            ))

        # END（结束谈判）样本：**必须**纳入 BC（用于复刻 PenguinAgent 的“筛掉对手/终止谈判”行为）。
        # - 新数据：直接使用 tracker 记录的 aop_action.action_op==END
        # - 旧数据：按 7.2.1 的“收到 offer 但无 accept/无 counter”后验推断为 END（噪声更大）
    return samples
```

#### 7.2.3 为什么这样能对齐现有日志与实现

- 当前 tracker 聚合出来的 `offer_history` 已经带有 `source=self/opponent`，因此可以稳定地把“我方动作时刻”抽出来。
- `accept` 的推断规则（最后报价来自对手且最终成功）在 AOP 协议中基本成立，且不要求额外日志字段。
- `relative_time` 离线无法精确重建（在线来自 `SAOState.relative_time`），这里用 `round/max_round` 的近似即可；在线阶段再用真实 `relative_time`。



#### 7.2.4 Tracker 必须修改：显式记录 `aop_action`（含 PenguinAgent 的“隐式 END”）

> 结论先说清楚：**要在 BC 阶段覆盖 `END_NEGOTIATION`，我们必须改 tracker**。  
> 仅靠 `offer_history + final_status` 的后验推断，`ACCEPT` 勉强可行，`END` 会非常噪声（且会漏掉 PenguinAgent 在 first_proposals 阶段“直接不谈”的隐式筛选）。

我们建议把“**我方在每一轮对每个 negotiator 的最终响应**”写成一条显式事件（这也是你后续做 MAPPO/CTDE 时最需要的 ground-truth action 记录）。

**建议事件格式（严格 JSON，仅新增一个 event，不破坏旧字段）：**

```json
{
  "category": "negotiation",
  "event": "aop_action",
  "data": {
    "sim_step": 12,
    "negotiation_id": "neg_...",
    "partner": "agent_...",
    "role": "buyer|seller",
    "round": 3,
    "action_op": 0,
    "response_type": "ACCEPT_OFFER|REJECT_OFFER|END_NEGOTIATION",
    "offer": {
      "quantity": 10,
      "unit_price": 23.5,
      "delivery_day": 15
    },
    "reason": "respond|propose|first_proposal|omit_in_first_proposals|omit_in_counter_all"
  }
}
```

- `action_op` 采用统一编码：`0=ACCEPT, 1=COUNTER, 2=END`  
- `offer` 的语义：
  - `ACCEPT`：`offer` **必须等于**对手当前 offer（SCML 语义要求：接受要带相同 outcome）
  - `COUNTER`：`offer` 为我方 counter-offer / propose 的 outcome
  - `END`：`offer = null`

##### 7.2.4.1 对我们的 HRL 代理怎么打点（StdAgent / respond/propose）

- 在 `respond()` 里：在返回 `SAOResponse` 之前写一条 `aop_action`：
  - `ACCEPT_OFFER` → `action_op=0`，`offer=opponent_offer`
  - `REJECT_OFFER` → `action_op=1`，`offer=counter_offer`
  - `END_NEGOTIATION` → `action_op=2`，`offer=null`
- 在 `propose()` 里：
  - 返回 `Outcome` → `action_op=1`（COUNTER/主动出价）
  - 返回 `None` → `action_op=2`（END：不开局/不出价/直接结束该谈判）

##### 7.2.4.2 对 PenguinAgent 怎么打点（StdSyncAgent / first_proposals + counter_all）

PenguinAgent 的“结束谈判”大多不是显式 `END_NEGOTIATION`，而是：

- `first_proposals()`：**不在返回字典里给某些 negotiator 提 proposal**（隐式筛选，不谈这些对手）
- `counter_all()`：**不在返回字典里包含某些 negotiator 的响应**（控制器会补 `END_NEGOTIATION`）

因此我们需要在 tracker/mixin 里对这两个函数做 hook，并把“缺失键”显式记录为 `END`：

**(A) hook first_proposals():**

- 令 `active_ids = list(self.negotiators.keys())`
- 令 `out = super().first_proposals()`（或原函数返回 dict）
- 对每个 `nid in active_ids`：
  - 若 `nid in out`：记录 `aop_action(action_op=COUNTER, reason="first_proposal", offer=out[nid])`
  - 若 `nid not in out`：记录 `aop_action(action_op=END, reason="omit_in_first_proposals", offer=null)`

**(B) hook counter_all(offers, states):**

- `active_ids = offers.keys()`（或与 states 对齐）
- `out = super().counter_all(offers, states)` 返回 `dict[nid -> SAOResponse]`
- 对每个 `nid in active_ids`：
  - 若 `nid in out`：读取 `resp = out[nid]`
    - `resp.response == ResponseType.ACCEPT_OFFER` → `action_op=ACCEPT`，`offer=resp.outcome`（应等于 `offers[nid]`）
    - `resp.response == ResponseType.REJECT_OFFER` → `action_op=COUNTER`，`offer=resp.outcome`（必须非空；为空则当作 END）
    - `resp.response == ResponseType.END_NEGOTIATION` → `action_op=END`
  - 若 `nid not in out`：记录 `action_op=END, reason="omit_in_counter_all"`

> 这样一来：**BC 阶段的 END 样本就是“真实的 PenguinAgent 筛选行为”，不是噪声推断**。

---

#### 7.2.5 Pipeline 侧修改：优先用 `aop_action` 生成 MicroSample（推荐）

当我们有 `aop_action` 事件后，MicroSample 的构造会非常直接：

- `history`：仍由 `offer_made/offer_received` 聚合（或直接从你现有的 `offer_history` 复用）
- `action_op`：直接来自 `aop_action.action_op`
- `target(q,p,t)`：仅在 `action_op==COUNTER` 时读取 `aop_action.offer`
- `ACCEPT/END`：不需要 `q,p,t` 标签（对应 head loss 直接 mask）

伪代码要点：

```python
if action_op == COUNTER:
    loss = CE(op) + Huber(q) + Huber(p) + CE(time_bucket)
else:
    loss = CE(op)  # q/p/t 不回传梯度
```

---

### 7.3 L2 目标标签怎么建：推荐用 `first_proposals` 聚合（而不是 signed contracts）

你提出的关键问题是：**L2 的“目标”到底应该对齐 PenguinAgent 的什么量？**

#### 7.3.1 对比：signed contracts vs. first_proposals vs. 其它

- **signed contracts（当天真实成交）**  
  - 优点：容易从日志拿到，且一定“可实现”（毕竟发生了）。
  - 缺点：更像“结果”，强烈受对手策略/随机性影响。市场不利时，成交量可能远小于 PenguinAgent 实际想买/想卖的量 → L2 会学得更保守。

- **first_proposals（开局的第一轮出价集合）✅ 推荐**  
  - 这是 PenguinAgent 在看到当日状态后**主动给出的计划**：  
    1) 它决定“要谈哪些对手”（没出现在返回 dict 的 negotiator 就被隐式 END）；  
    2) 它决定“每个交货期大概想要多少量/什么价格锚点”（至少对近几天有明确出价）。  
  - 因此 first_proposals 比 signed contracts 更像“目标/意图”，也更适合做 L2 的监督信号。

- **needs_at(step)（根据库存/产能算出来的需求）**  
  - 语义很像“目标”，但你需要确保它与日志里的真实行为一致（PenguinAgent 对未来 step 的 first_proposal 里会做 `/3` 分摊，直接用 needs_at 会让 L2 标签偏大，反而可能让 BC 难对齐）。

#### 7.3.2 用 first_proposals 聚合成 16 维目标向量（数学定义）

记当天为 \(d\)，桶区间为 \(\mathcal{B}_b = [\Delta_b^{lo}, \Delta_b^{hi}]\)。

从日志中抽取当天所有 `offer_made(reason=="first_proposal")`，并用 `negotiation.started.role` 区分买/卖方向：

- 买侧 first proposals 集合：\(\mathcal{F}^{buy}_d\)
- 卖侧 first proposals 集合：\(\mathcal{F}^{sell}_d\)

对每条 proposal \(i\)，其 outcome 为 \((q_i, t_i, p_i)\)，其中 \(t_i\) 是**绝对交货日**（SCML 的 issue 是绝对时间）。定义相对交货期：

\[
\delta_i = t_i - d
\]

分桶：

\[
b(i) = \texttt{delta\_to\_bucket}(\delta_i)
\]

则 L2 的数量目标（按桶）：

\[
Q^{buy}_{d,b} = \sum_{i \in \mathcal{F}^{buy}_d: b(i)=b} q_i,\quad
Q^{sell}_{d,b} = \sum_{i \in \mathcal{F}^{sell}_d: b(i)=b} q_i
\]

价格目标（建议用**数量加权平均**，并且只在 \(Q>0\) 时监督）：

\[
P^{buy}_{d,b} =
\begin{cases}
\frac{\sum_{i \in \mathcal{F}^{buy}_d: b(i)=b} q_i p_i}{\sum_{i \in \mathcal{F}^{buy}_d: b(i)=b} q_i}, & Q^{buy}_{d,b} > 0\\
0, & Q^{buy}_{d,b} = 0
\end{cases}
\]

\[
P^{sell}_{d,b} =
\begin{cases}
\frac{\sum_{i \in \mathcal{F}^{sell}_d: b(i)=b} q_i p_i}{\sum_{i \in \mathcal{F}^{sell}_d: b(i)=b} q_i}, & Q^{sell}_{d,b} > 0\\
0, & Q^{sell}_{d,b} = 0
\end{cases}
\]

最终 16 维目标向量：

\[
g_d = [Q^{buy}_{d,0}, P^{buy}_{d,0}, Q^{sell}_{d,0}, P^{sell}_{d,0},\dots,Q^{buy}_{d,3}, P^{buy}_{d,3}, Q^{sell}_{d,3}, P^{sell}_{d,3}]
\]

**训练时的 loss mask：**

- 价格项只在对应桶 \(Q>0\) 时训练（否则 price label 没意义）；
- 数量项永远训练（哪怕是 0，0 也是“今天这个桶不谈”的明确信号）。

#### 7.3.3 这如何解决“过量购买”的担忧？

当 L2 目标来自 PenguinAgent 的 first_proposals 时，它的 \(Q^{buy}_{d,b}\) 本身就对齐“PenguinAgent 打算买多少”。

L3 线程在每一轮都能看到：

- `goal_gap_buy[b] = max(0, Q^{buy}_{d,b} - bought_by_bucket[b])`

并且在生成 counter-offer 时我们**硬裁剪**：

\[
q \leftarrow \min(\text{goal\_gap\_buy}[b], Q_{safe}[\delta])
\]

所以即使 \(Q_{safe}\) 在剩余天数很多时很大，只要 L2 的目标 \(Q^{buy}\) 是 50，那么所有线程共享的缺口会很快被填满到 0，后续线程自然会停止加量（或直接 END）。



## 八、Training 修改：三阶段训练流程

### 8.1 训练流程概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         训练阶段流程图                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │ 阶段一：    │    │ 阶段二：    │    │ 阶段三：                │ │
│  │ 离线训练    │ →  │ α 热身训练   │ →  │ 在线强化学习 (CTDE)    │ │
│  │ (BC)       │    │             │    │                         │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│                                                                     │
│  - L3: BC (α=0)     - 简化环境       - Centralized Critic           │
│  - L4: 伪标签预训练  - 强化 α 响应    - 奖励塑形 + 正则化            │
│                                       - L4 联合优化（可选）         │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 阶段一：离线训练

#### 8.2.1 专家数据收集与安全规则

```python
# 收集专家数据
expert_trajectories = collect_expert_demonstrations(
    expert_agent="PenguinAgent",  # 或人类专家
    num_episodes=1000,
)

# 安全规则（构建 L1 安全掩码）
safety_rules = {
    "no_exceed_Q_safe": True,      # 不超过安全量
    "no_exceed_budget": True,       # 不超预算
    "respect_time_mask": True,      # 遵守时间掩码
}
```

#### 8.2.2 L3 行为克隆（α=0）

```python
class BCTrainer:
    """L3 行为克隆训练器.

    关键：α 固定为 0（中性），L3 学习专家的标准策略。
    L3 输入包含 α，但不输出 α。
    """

    def train_l3_bc(self, batch):
        # α 固定为 0
        alpha = torch.zeros(batch['states'].shape[0])

        # L3 前向传播（Decision Transformer）
        op_logits, quantity, price, time_logits = self.l3(
            alpha=alpha,  # 固定为 0
            history_states=batch['history_states'],
            history_actions=batch['history_actions'],
            current_state=batch['current_state'],
            role=batch['role'],
            global_feat=batch['global_feat'],
            time_mask=batch['time_mask'],
            Q_safe=batch['Q_safe'],
            price_range=batch['price_range'],
        )

        # 多任务损失
        loss = self.compute_multitask_loss(
            op_logits, quantity, price, time_logits,
            batch['targets'],
        )

        # 应用 L1 安全掩码（屏蔽违规动作）
        loss = self.apply_safety_mask(loss, batch)

        return loss
```

#### 8.2.3 L4 预训练（启发式伪标签）

```python
class L4Pretrainer:
    """L4 GlobalCoordinator 预训练.

    使用启发式规则生成 α 伪标签，监督预训练 L4。
    """

    def generate_pseudo_labels(self, thread_states, global_broadcast):
        """根据启发式规则生成 α 伪标签."""
        alphas = []
        for ts in thread_states:
            alpha = self.heuristic_alpha.compute_alpha(ts, global_broadcast)
            alphas.append(alpha)
        return torch.tensor(alphas)

    def train_l4_step(self, batch):
        # 生成伪标签
        target_alphas = self.generate_pseudo_labels(
            batch['thread_states'],
            batch['global_broadcast'],
        )

        # L4 前向传播
        pred_alphas = self.l4_coordinator(
            batch['thread_states_tensor'],
            batch['global_state_tensor'],
        )

        # MSE 损失
        loss = F.mse_loss(pred_alphas, target_alphas)
        return loss
```

### 8.3 阶段二：α 服从热身训练

#### 8.3.1 目的

在正式 RL 之前，用简化环境强化 L3 对 α 的响应。

#### 8.3.2 简化环境与奖励设计

```python
class AlphaWarmupEnv:
    """α 热身简化环境.

    目的：让 L3 快速学会"高 α = 快成交，低 α = 坚持价格"。
    """

    def compute_reward(self, alpha, action, outcome):
        """根据 α 计算塑形奖励."""

        # 高 α 场景：奖励快速成交
        if alpha > 0.5:
            if outcome == "accepted":
                return +1.0  # 快速成交奖励
            elif action == "end":
                return -0.3  # 不应该在高 α 时拒绝
            else:
                return -0.1 * self.round_number  # 每轮延迟惩罚

        # 低 α 场景：奖励坚持价格
        elif alpha < -0.5:
            if action == "accept" and self.price_disadvantageous():
                return -0.3  # 不应该在低 α 时接受不利价格
            elif outcome == "deal_with_good_price":
                return +1.0  # 坚持后获得好价格
            else:
                return +0.1  # 每轮坚持奖励

        # 中性 α：标准奖励
        else:
            return self.compute_standard_reward(outcome)
```

### 8.4 阶段三：在线强化学习（CTDE）

#### 8.4.1 CTDE 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Centralized Training, Decentralized Execution        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  训练时（Centralized）：                                             │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │  Centralized Critic                                     │        │
│  │  输入：全局状态 + 所有线程状态 + 所有 α                   │        │
│  │  输出：全局价值估计 V(s_global)                          │        │
│  └─────────────────────────────────────────────────────────┘        │
│                          ↓                                          │
│         为每个线程的 L3 策略提供梯度                                  │
│                                                                     │
│  执行时（Decentralized）：                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │ Thread 1 │  │ Thread 2 │  │ Thread 3 │  ...                     │
│  │ L3(s,α₁) │  │ L3(s,α₂) │  │ L3(s,α₃) │                          │
│  └──────────┘  └──────────┘  └──────────┘                          │
│  各线程独立决策，仅依赖本地观察和 α                                   │
└─────────────────────────────────────────────────────────────────────┘
```

#### 8.4.2 奖励函数设计

```python
def compute_reward(self, thread_id, alpha, action, outcome, global_state):
    """综合奖励 = 主奖励 + α 塑形奖励."""

    # 1. 主奖励：经济效益
    main_reward = self.compute_profit_reward(outcome)

    # 2. α 塑形奖励（权重较小，仅作引导）
    shaping_weight = 0.1

    if alpha > 0.3:  # 高 α
        # 高优先级却未成交 → 轻微惩罚
        if outcome == "no_deal":
            shaping = -0.2
        # 高优先级且快速成交 → 奖励
        elif outcome == "deal" and self.rounds_taken < 5:
            shaping = +0.2
        else:
            shaping = 0.0

    elif alpha < -0.3:  # 低 α
        # 低优先级却激进成交 → 轻微惩罚
        if outcome == "deal" and self.price_too_compromised():
            shaping = -0.2
        # 低优先级且坚持价格 → 奖励
        elif outcome == "deal" and self.price_favorable():
            shaping = +0.2
        else:
            shaping = 0.0

    else:  # 中性 α
        shaping = 0.0

    return main_reward + shaping_weight * shaping
```

#### 8.4.3 行为-α 正则化（可选）

```python
def compute_alpha_alignment_loss(self, alpha, aggressiveness_metric):
    """
    正则化损失：鼓励行为激进度与 α 正相关.

    aggressiveness_metric 示例：
    - 平均让步幅度
    - 成交所需轮数的倒数
    - accept 概率
    """
    # 目标：aggressiveness ≈ k * α + b
    k, b = 0.5, 0.5  # 可调参数
    target_aggressiveness = k * alpha + b

    loss = F.mse_loss(aggressiveness_metric, target_aggressiveness)
    return loss
```

#### 8.4.4 L4 联合优化（可选）

```python
class CTDETrainer:
    """CTDE 训练器，支持 L3/L4 联合优化."""

    def train_step(self, episode_data):
        # 1. 更新 Centralized Critic
        critic_loss = self.update_critic(episode_data)

        # 2. 更新各线程的 L3 策略
        for thread_data in episode_data['threads']:
            l3_loss = self.update_l3_policy(
                thread_data,
                self.critic,
            )

        # 3. （可选）更新 L4 GlobalCoordinator
        if self.train_l4:
            # 使用回合总回报作为 L4 的梯度信号
            l4_loss = self.update_l4_policy(
                episode_data['global_return'],
                episode_data['alpha_decisions'],
            )

        return {'critic': critic_loss, 'l3': l3_loss, 'l4': l4_loss}
```

### 8.5 安全约束：持续 L1 掩码

```python
def apply_safety_during_training(self, action_logits, time_mask, Q_safe, B_free):
    """在训练和执行中持续应用 L1 安全掩码."""

    # 1. 时间掩码
    action_logits['time'] = action_logits['time'] + time_mask

    # 2. 数量裁剪
    max_qty = Q_safe[action_logits['time'].argmax()]
    action_logits['quantity'] = torch.clamp(
        action_logits['quantity'],
        max=max_qty,
    )

    # 3. 预算约束
    # ...

    return action_logits
```

---

## 九、文档修改清单

| 文档 | 修改内容 |
|------|----------|
| `HRL-XF 实施规范.md` | 更新 L1/L3/L4 设计、移除 baseline 相关内容 |
| `HRL-XF 期货市场代理重构.md` | 更新架构图、信息流描述 |
| `HRL-XF 期货市场适应性重构.md` | 更新 L3/L4 实现细节 |
| `README.md` | 更新架构说明（如有） |

---

## 十、实施检查清单（修正版：方案A + L4 仍为网络）

> **重要更正**：原 v1.0 的第十部分把 L4 当作“纯规则监控器”，并要求删除 `GlobalCoordinator` 与相关训练代码。  
> 但你已确认：**不选方案B**，且 **L4 仍是网络（输出 α）**。因此本检查清单以“监控器 + α 协调器”作为唯一实现口径。

### Phase 1：L1 改造（移除 baseline，仅保留 mask）

- [ ] 修改 `l1_safety.py`：`L1Output` 移除 `baseline_action`
- [ ] `L1SafetyLayer.compute()` 不再返回 baseline
- [ ] 清理所有调用 `l1_output.baseline_action` 的地方（至少包括 `agent.py`、`l3_executor.py`、`data_pipeline.py`）
- [ ] 保留并验证：`time_mask / Q_safe / Q_safe_sell / B_free` 的离线可重建性

### Phase 2：L3 改造（残差 → 完整 SAOAction）

- [ ] 新增/重写 `l3_executor.py`：输出 `L3Output(action=SAOAction(...))`
- [ ] 动作头：`op_logits (3,)` + `time_logits (H+1,)` + `quantity_ratio` + `price_ratio`
- [ ] 训练损失：对 `action_op` 与 `time` 用 CE；对 `q/p` 用 MSE（仅在 `action_op==COUNTER`（还价/出价）时计算）
- [ ] 推理：实现 `argmax`/采样两套策略；并支持 mask（time_mask；accept_feasible mask）

### Phase 3：L4 改造（保留网络，但不再调制 L3 残差）

- [ ] 代码结构拆分：
  - [ ] `L4GlobalMonitor`：纯规则，维护成交/承诺、计算 `GlobalBroadcast`
  - [ ] `GlobalCoordinator`：自注意力网络，输入所有线程显式状态，输出每线程 `α`
- [ ] 删除或停用旧的：
  - [ ] `L4ThreadCoordinator.compute()` 中的 **modulation_factors** 路径
  - [ ] `batch_planner.py` 以及与 “按权重顺序扣减 Q/B” 相关的逻辑
- [ ] 明确 L4 输入来自“显式特征”（不读 L3 hidden），且**不再依赖 baseline**

### Phase 4：Agent 集成（对齐新动作语义）

- [ ] `agent.py`：
  - [ ] `before_step()` 中调用 `l4_monitor.on_step_begin(...)`
  - [ ] 每次 `respond/propose` 前计算 `alpha_map = l4_coord.compute_alpha(all_threads, global_state)`
  - [ ] 构造 `L3Input`：包含 `GlobalBroadcast` 与本线程 `α`
  - [ ] **abs_t ↔ delta_t 转换**：调用 `l1.clip_action()` 前把 `delivery_time(abs)` 变为 `delta_t`
  - [ ] **接受动作硬约束校验**（只管履约/预算，不含价格）：不可行时强制 reject/offer
- [ ] `on_negotiation_success()`：
  - [ ] 通知 `l4_monitor.on_contract_signed(...)` 更新当日已签量/预算与 broadcast

### Phase 5：Data Pipeline（strict JSON ONLY 下可重建）

- [ ] `data_pipeline.py`：
  - [ ] 保留 tracker entries → `neg_logs` 聚合（已有）
  - [ ] 新增 `extract_l3_samples_aop()`：从 `offer_history.source` 推断 `offer/accept` 标签（见 7.2）
  - [ ] 新增 `compute_l1_masks_offline()`：离线重建 `time_mask/Q_safe/B_free`（不要求 baseline）
  - [ ] 仍保留 L2 goal 回填机制（macro → micro），用于构造 `GlobalBroadcast.goal_gap_*`

### Phase 6：训练（BC + L4 蒸馏/监督）

> 这部分是原第十部分缺失/错误的核心：**L4 仍需要训练入口**（至少预训练），否则在线阶段 α 会随机干扰 L3。

- [ ] BC（Phase 0/1）：
  - [ ] **L2（必须）**：监督学习日级目标 `g_d ∈ R^{16}`（标签来自 7.3 的 `first_proposals` 聚合）
    - [ ] loss（示例）：
      \[
      L_{L2}=\sum_{b=0}^{3} \Bigl(\mathrm{Huber}(\hat Q^{buy}_b,Q^{buy}_b)+\mathrm{Huber}(\hat Q^{sell}_b,Q^{sell}_b)\Bigr)
      +\lambda_p\sum_{b=0}^{3}\Bigl(\mathbb{1}[Q^{buy}_b>0]\,\mathrm{Huber}(\hat P^{buy}_b,P^{buy}_b)+\mathbb{1}[Q^{sell}_b>0]\,\mathrm{Huber}(\hat P^{sell}_b,P^{sell}_b)\Bigr)
      \]
    - [ ] 输出约束：`Q≥0`（ReLU/softplus），必要时可在输出端 **clip 到 L1 的上限**（如 `Q <= Q_safe_bucket_sum`，`buy_cost <= B_free`），但不在 mask 阶段用价格筛选
  - [ ] **L3（必须）**：用专家（PenguinAgent）AOP 动作做 BC（`α=0` 或固定中性值），学习三类动作：
    - `op ∈ {ACCEPT, COUNTER, END}`（其中 END 包含 PenguinAgent 的“隐式 END”，由 tracker 补齐）
    - 仅当 `op==COUNTER` 时回归/分类 `(q,p,delta_t)`（其它 op 对应 head loss mask 掉）
  - [ ] **L2→L3 的输入对齐（建议）**：
    - 起步：用 **label 的 `g_d`（teacher forcing）** 当作 L3 的输入（更容易收敛、对齐专家轨迹）
    - 稳定后：逐步混入 `\hat g_d = L2(state_d)`（scheduled sampling），减少上线时 L2 预测误差带来的分布偏移
  - [ ] 安全：训练/推理时始终启用 L1 的 mask/裁剪（避免违约样本污染）
- [ ] 预训练 L4（可选但强烈推荐）：
  - [ ] 实现 `HeuristicAlphaGenerator`（从离线轨迹反推 α 伪标签）
    - 输入：报价数量变化、价格移动、round/relative_time、目标缺口、交期紧迫度等
    - 输出：`α_hat(thread, round)`（允许同一谈判内随轮次变化）
  - [ ] 用监督学习/蒸馏训练 `GlobalCoordinator` 拟合 `α_hat`
  - [ ] 训练目标：`L4Loss = MSE(α_pred, α_hat)` 或 `HuberLoss`
- [ ] 在线 RL 热身（α 依从 warmup，可选）：
  - [ ] 在简化奖励下跑短期 PPO/MAPPO，让 L3 对 α 敏感（避免 RL 初期把 α 当噪声）

### Phase 7：在线学习（MAPPO + Centralized Critic，CTDE）

- [ ] Actor：每个线程一份共享参数的 L3（输入含 α 与 GlobalBroadcast）
- [ ] Critic：集中式价值网络 `V(s_global)`，可观察所有线程状态、目标剩余、已签约量等（不要过复杂）
- [ ] 奖励：以真实利润为主；必要时加轻度 shaping（如高 α 线程迟迟不成交的惩罚）

### Phase 8：回归测试（必须做）

- [ ] 单线程谈判：ACCEPT / REJECT_OFFER(带counter offer) / END_NEGOTIATION 三动作语义是否正确
- [ ] 多线程并发：某线程成交后，`GlobalBroadcast.goal_gap` 与 `Q_safe_remaining/B_remaining` 是否即时变化
- [ ] 安全性：任何时刻 accept/offer 都不会导致“必然违约或浪费”
- [ ] 训练可跑：strict JSON ONLY pipeline 下能生成 AOP micro samples 且损失收敛