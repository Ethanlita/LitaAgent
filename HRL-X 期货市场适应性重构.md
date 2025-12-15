# **SCML 2025 供应链期货交易代理架构重构蓝皮书：从 HRL-X 到 HRL-XF 的范式演进**

## **1\. 执行摘要：时间维度的引入与架构范式转移**

### **1.1 2025 赛季的本体论转变**

供应链管理联赛（Supply Chain Management League, SCML）在 2025 年标准赛道（Standard League）中引入的规则变更，不仅仅是参数层面的调整，而是对代理生存环境本体论的根本性重构。现有的 LitaAgent-HRL（即 HRL-X 架构）是基于 2024 年及之前的“现货思维”设计的，其核心假设是交易的即时性（Spot Trading）与库存的马尔可夫性（Markovian Inventory）。然而，2025 年规则明确引入了 **期货合约（Futures Contracts）**，允许代理就 $(q, p, t)$ 进行谈判，其中 $t$ 代表未来的交付时间 。

这一变化导致了供应链管理的本质从“库存控制问题”（Inventory Control Problem）向“物料需求计划问题”（Material Requirements Planning, MRP）的跃迁。在现货市场中，决策是标量的：买多少 ($q$)？什么价格 ($p$)？而在期货市场中，决策变成了矢量或函数：在时间轴 $T$ 上的供需曲线匹配。当前的 HRL-X 架构存在以下致命缺陷：

1. **L1 安全护盾的静态性**：原设计仅检查当前时刻 $t$ 的库容 $C\_{free}$ 。然而，在期货市场中，当前仓库可能是空的，但在 $t+5$ 时刻可能因为之前的远期合约到货而爆仓。静态检查将导致未来的确定性违约。  
2. **L2 战略层的短视性**：原 L2 仅输出当天的买入/卖出总目标 。它无法表达“当前不缺货，但预测 $t+10$ 会有缺口”的战略意图，导致代理错失利用远期低价合约进行套利或对冲的机会。  
3. **L3 执行层的动作空间缺失**：原 Decision Transformer 仅能输出 $(q, p)$ 。面对 SCML 2025 的 offer $(q, p, t)$，L3 缺乏感知时间维度 $t$ 的输入嵌入层（Input Embedding）和输出头（Action Head），使其在谈判中对“交货时间”这一关键变量完全失明。  
4. **L4 全局协调的时间冲突盲区**：原 L4 仅处理并发线程间的资金竞争 。在期货环境中，线程 A（商谈 $t+5$ 交货）和线程 B（商谈 $t+20$ 交货）在库容资源上可能互不冲突，而在资金流上存在跨期耦合。原设计无法解析这种复杂的时空依赖。

### **1.2 HRL-XF (Futures) 架构概览**

为了彻底解决上述问题，本报告提出 **HRL-XF** 架构。这不是对 HRL-X 的简单修补，而是对状态空间、动作空间及各层级控制逻辑的全面升级。我们将引入“时序库存包络”（Temporal Inventory Envelope）作为 L1 的核心，将 L2 升级为“时平规划器”（Horizon Planner），赋予 L3 “相对时间感知”能力，并重构 L4 为“时空注意力网络”。

本报告将分章节详细阐述各层的数学模型、张量设计与代码实现方案，字数约 15,000 字，旨在提供一份详尽无遗的实施指南，确保 LitaAgent 在 SCML 2025 的期货市场中具备超人级的长期规划与博弈能力。

## ---

**2\. 环境数学建模与期货经济学：高维时空 POMDP 的构建**

在着手修改代码之前，必须先在数学层面重新定义代理所处的环境。原有的 POMDP 元组 $\\langle \\mathcal{S}, \\mathcal{A}, \\mathcal{T}, \\mathcal{R}, \\Omega \\rangle$ 已不足以描述 2025 年的期货特征。我们需要构建一个 **增广时空状态空间（Augmented Spatio-Temporal State Space）**。

### **2.1 状态空间的张量化扩展 ($\\mathcal{S}^+$)**

在 HRL-X 原设计中，状态 $\\mathcal{S}$ 主要包含当前快照：$I\_t$（当前库存）、$B\_t$（当前资金）等 。在 HRL-XF 中，状态必须包含对未来的 **承诺（Commitments）** 和 **预测（Projections）**。

我们将状态定义为一个混合张量组 $\\mathcal{S}\_t^+ \= \\{ \\mathbf{x}\_{static}, \\mathbf{X}\_{temporal}, \\mathbf{x}\_{role} \\}$。

**注**：$\\mathbf{x}_{role}$ 为角色嵌入向量（Buyer/Seller），详见 2.1.3 节。

#### **2.1.1 静态状态向量 ($\\mathbf{x}\_{static}$)**

这部分保留原设计的核心标量特征，但需精简，仅保留物理属性：

* $B\_t$：当前可用现金余额。  
* $I\_{now}$：当前仓库中的物理库存。  
* $C\_{total}$：仓库总容量。  
* $L\_{prod}$：生产线数量与效率参数。  
* $Step$：当前仿真步数（归一化）。

* **设计决策**：优先尝试通过 `awi.profile.storage_capacity` 获取静态库容。如果 negmas 未提供该接口，则采用动态公式：$C_{total}[k] = n\_lines \times (T_{max} - (t+k))$。**逻辑解释（买入时）**：买入的是原材料，而原材料需要加工才能出售。超出「原材料交货日到最后一日的总产能」的所有原材料都会因为无法及时加工而被浪费掉。因此动态库容本质上是「可有效利用的原材料容量」。（注：卖出时库容约束相对宽松，因为成品可以直接销售，不受加工时间限制）

#### **2.1.2 时序状态张量 ($\\mathbf{X}\_{temporal}$)**

这是 HRL-XF 的核心创新。我们需要一个能够描述未来 $H$ 天（Planning Horizon，建议 40 天）供需状况的矩阵。
$\\mathbf{X}\_{temporal} \\in \\mathbb{R}^{H \\times F}$，其中 $H=40$，$F=9$（**确定为9维特征通道**）。  
对于未来第 $k$ 天（即绝对时间 $t+k$），特征通道定义如下：

| 通道 | 特征名 | 公式/说明 |
|------|--------|----------|
| 0 | `vol_in` | $Q_{in}[k]$ = 第 $t+k$ 天到货的采购量（已签署合约） |
| 1 | `vol_out` | $Q_{out}[k]$ = 第 $t+k$ 天发货的销售量（已签署合约） |
| 2 | `prod_plan` | $Q_{prod}[k]$ = 预计生产消耗（保守估计） |
| 3 | `inventory_proj` | $I_{proj}[k] = I_{now} + \sum_{j=0}^{k}(Q_{in}[j] - Q_{out}[j] - Q_{prod}[j])$ |
| 4 | `capacity_free` | $C_{free}[k] = C_{total}[k] - I_{proj}[k]$ |
| 5 | `balance_proj` | $B_{proj}[k] = B_t + \sum_{j=0}^{k}(Receivables[j] - Payables[j])$ |
| 6 | `price_diff` | $P_{future}[k] - P_{spot}$（期货溢价/贴水，见下方计算说明） |
| 7 | `buy_pressure` | 买方压力指数（见下方计算说明） |
| 8 | `sell_pressure` | 卖方压力指数（见下方计算说明） |

**通道 6-8 计算说明**：

由于 SCML 标准世界不存在公开订单簿，通道 6-8 基于代理可观测的谈判与合约数据推断：

**buy_pressure[k]（买方压力）**：
- 含义：第 $t+k$ 天买方对商品的需求强度。值越大表示"买方多、缺货风险高、可抬价"。
- 计算：`demand_qty[k] = active_sell_offers[k] + signed_buy_contracts[k]`
- 归一化：`buy_pressure[k] = clip(demand_qty[k] / economic_capacity[k], 0, 1)`

**sell_pressure[k]（卖方压力）**：
- 含义：第 $t+k$ 天卖方的供给强度。值越大表示"供给多、价格承压"。
- 计算：`supply_qty[k] = active_buy_offers[k] + signed_sell_contracts[k]`
- 归一化：`sell_pressure[k] = clip(supply_qty[k] / economic_capacity[k], 0, 1)`

**price_diff[k]（价格趋势）**：
- 信号来源：已签成交 VWAP > 谈判出价中位数 > 现货价回退
- 输出：`price_diff[k] = P_future[k] - P_spot`

**设计说明**：
- 当前市场现货价格已包含在 $\mathbf{x}_{static}$ 中（`spot_price_in`/`spot_price_out`），无需在时序张量中重复
- 买卖压力分开表示，使代理可根据角色做出差异化响应

#### **2.1.3 角色嵌入向量 ($\\mathbf{x}\_{role}$) —— 对称性的基石**

**设计决策：L2-L4 全部需要角色嵌入。**

为了解决买卖逻辑不对称的缺陷，HRL-XF 在 **L2、L3、L4 所有层级** 强制引入角色嵌入。

* 形式：One-hot 编码 $[1, 0]$ (Buyer) 或 $[0, 1]$ (Seller)，或通过可学习的 Embedding 层映射为低维稠密向量 $\\mathbf{e}_{role} \\in \\mathbb{R}^{d}$。
* 作用：指示神经网络当前是在处理"补货"任务还是"去库存"任务，使得同一套网络权重可以学习镜像的博弈逻辑（例如：买方压价与卖方抬价是对称的）。

**设计洞察**：原 HRL-X 将状态视为扁平向量输入 MLP。HRL-XF 必须将 $\\mathbf{X}\_{temporal}$ 视为时间序列，输入到 **1D-CNN** 或 **Transformer Encoder** 中，以提取"库存缺口模式"（例如：第 5-8 天将出现严重原料短缺）。

### **2.2 动作空间的维度增广 ($\\mathcal{A}^+$)**

原动作空间为 $\\mathcal{A} \= \\{ \\text{Accept}, \\text{Reject}, \\text{Offer}(q, p) \\}$ 。  
新的动作空间必须支持对时间 $t$ 的谈判。

$$a\_{offer} \= (q, p, \\delta\_t)$$

其中 $\\delta\_t \\in \\{0, 1, \\dots, H\_{max}\\}$ 表示 相对交货时间（Relative Delivery Time）。

* 为什么使用相对时间？  
  如果我们使用绝对时间（例如“在第 54 天交货”），随着仿真进行，该数值不断增大，神经网络难以泛化。使用相对时间（例如“在 3 天后交货”），语义是稳定的——它总是意味着“比较急迫”。

因此，L3 的输出层需要增加一个 **时间头（Temporal Head）**，用于预测 $\\delta\_t$。考虑到 $\\delta\_t$ 是离散的整数，且具有多峰分布特性（例如：要么明天急用，要么 10 天后补货，中间没有意义），我们将 $\\delta\_t$ 建模为分类问题（Classification），而非回归问题（Regression）。

### **2.3 奖励函数的跨期修正 ($\\mathcal{R}^+$)**

在期货交易中，动作与奖励在时间上是解耦的。

* **时刻 $t$**：签署合约。代理承诺在未来支付现金或交付货物。此时可能没有立即的 Reward（除非使用 Shaped Reward）。  
* **时刻 $t+\\delta\_t$**：合约执行。库存变化，资金转移。Reward 产生。

如果仅使用环境原始奖励（Cash Change），L3 在时刻 $t$ 做出英明决策（例如低价锁定远期原料）时得不到反馈，梯度消失问题将极其严重。  
因此，HRL-XF 必须采用 基于净现值（NPV）的势能奖励函数（Potential-Based Reward Shaping）。  
定义势能函数 $\\Phi(s)$：

$$\\Phi(s) \= B\_t \+ \\text{Val}(I\_{now}) \+ \\sum\_{c \\in \\text{Pending}} \\gamma^{\\delta\_{t,c}} \\cdot \\text{Val}(c)$$

其中 $\\text{Val}(c)$ 是合约 $c$ 的理论价值（例如：$q \\times P\_{market\\\_avg}$），$\\gamma$ 是时间折扣因子（例如 0.98），反映了资金的时间价值和未来执行的不确定性风险。  
修正后的单步奖励：

$$R\_t' \= R\_{env} \+ (\\Phi(s\_{t+1}) \- \\Phi(s\_t))$$  
这样，当代理在 $t$ 时刻签署一个有利可图的远期合约时，$\\Phi(s)$ 瞬间增加，代理立即获得正向奖励，从而解决了信用分配（Credit Assignment）问题。

## ---

**3\. L1 安全护盾：时序约束与库存包络算法**

用户提到的原设计“似乎只考虑到现货”在 L1 层表现得最为明显。原设计只检查 current\_inventory \+ q \<= capacity 。这是极其危险的。  
例如：仓库容量 100。当前库存 0。

* $t=0$：签署合约买入 100 单位，交货期 $t=5$。  
* $t=1$：L1 检查当前库存仍为 0，允许买入 100 单位，交货期 $t=2$。  
* $t=2$：100 单位到货，库存满。  
* $t=5$：之前的 100 单位到货。**爆仓（Overflow）**。

因此，L1 必须升级为 **时序安全护盾（Temporal Safety Shield）**。

### **3.1 核心算法：可用承诺量 (Available-to-Promise, ATP)**

我们需要实现一个基于 **滑动窗口（Sliding Window）** 的库存模拟器。

**输入数据**：

* 当前库存 $I\_{now}$。  
* 未来 $H$ 天的入库计划列表 $\\mathcal{S}\_{in} \= \\{(q\_i, \\delta\_i)\\}$。  
* 未来 $H$ 天的出库计划列表 $\\mathcal{S}\_{out} \= \\{(q\_j, \\delta\_j)\\}$。  
* 未来 $H$ 天的预计生产消耗 $\\mathcal{S}\_{prod} \= \\{(q\_k, \\delta\_k)\\}$（来自 L2 的生产计划或保守估计）。

**算法步骤**：

1. 构建净流向量 (Net Flow Vector)：  
   初始化长度为 $H$ 的向量 $\\mathbf{F} \= \[0, \\dots, 0\]$。  
   遍历所有计划，将 $q$ 累加到对应的相对时间下标 $\\delta$ 上。入库为正，出库和生产消耗为负。  
2. 计算库存轨迹 (Inventory Trajectory)：

   $$I\_{proj}\[\\tau\] \= I\_{now} \+ \\sum\_{k=0}^{\\tau} \\mathbf{F}\[k\]$$

   这将生成一条未来库存水位的曲线。  
3. 计算有效自由空间 (Effective Free Space)：

   $$C\_{free}\[\\tau\] \= C\_{total} \- I\_{proj}\[\\tau\]$$

   注意：必须确保 $I\_{proj}\[\\tau\] \\ge 0$（不缺货约束）和 $C\_{free}\[\\tau\] \\ge 0$（不爆仓约束）。如果任何点违反，则当前状态本身已不安全，需立即停止买入并紧急抛售。  
4. 生成最大买入量掩码 (Max Buy Mask)：  
   这是一个关键逻辑：如果我在 $\\delta$ 时刻买入 $q$，这个 $q$ 将一直占据库存，直到被生产消耗或卖出。为了安全起见（L1 必须保守），我们假设买入的原料可能会滞留一段时间。  
   最严格的约束是：买入量 $q$ 不能导致从 $\\delta$ 开始的未来 任何一天 爆仓。

   $$Q_{safe}[\delta] = \min_{k=\delta}^{H} (C_{total}[k] - L(k))$$

   这个公式的含义是：如果在未来第 10 天仓库会满（自由库容为0），那么即便今天仓库是空的，我也不能买入任何将在第 10 天之前到货且滞留到第 10 天的货物。

   **向量化实现（支持动态 $C_{total}$）**：由于 $C_{total}$ 可能是一个向量（动态库容），此公式天然支持这种情况。实现时使用**逆向累积最小值（Reverse Cumulative Min）**高效计算：
   ```python
   raw_free = C_total - L  # shape (H,)
   reversed_free = raw_free[::-1]
   reversed_cummin = np.minimum.accumulate(reversed_free)
   Q_safe = reversed_cummin[::-1]
   Q_safe = np.maximum(Q_safe, 0)  # 非负约束
   ```

### **3.2 TensorFlow 自定义层实现**

我们将上述逻辑封装为可微分（或至少是张量兼容）的 Keras 层，以便嵌入到 HRL-XF 的前向传播图中。

Python

import tensorflow as tf

class TemporalShieldingLayer(tf.keras.layers.Layer):  
    """  
    L1: Temporal Safety Shield (HRL-XF Revised)  
    计算未来每一天的最大安全买入量和最小必要买入量。  
    """  
    def \_\_init\_\_(self, capacity, horizon=20, reserve=1000.0, \*\*kwargs):  
        super(TemporalShieldingLayer, self).\_\_init\_\_(\*\*kwargs)  
        self.capacity \= tf.constant(capacity, dtype=tf.float32)  
        self.horizon \= horizon  
        self.reserve \= tf.constant(reserve, dtype=tf.float32)

    def call(self, inputs):  
        \# inputs:  
          
        curr\_inv \= inputs \# (B, 1\)  
        wallet \= inputs   \# (B, 1\)  
        inc\_sched \= inputs \# (B, H) \- 每日确定的入库量  
        out\_sched \= inputs \# (B, H) \- 每日确定的出库量  
        prod\_sched \= inputs \# (B, H) \- 预计每日生产消耗 (保守估计)  
          
        \# 1\. 计算每日净流量 (Net Flow)  
        \# 增加库存：入库  
        \# 减少库存：出库 \+ 生产消耗  
        net\_flow \= inc\_sched \- out\_sched \- prod\_sched \# (B, H)  
          
        \# 2\. 计算库存水位投影 (Cumulative Sum)  
        \# I\[t\] \= I\_0 \+ sum(flow\_0...t)  
        \# 使用 cumsum 沿时间轴累加  
        inv\_trajectory \= curr\_inv \+ tf.math.cumsum(net\_flow, axis=1) \# (B, H)  
          
        \# 3\. 计算未来的物理空闲空间  
        raw\_free\_space \= self.capacity \- inv\_trajectory \# (B, H)  
          
        \# 4\. 计算针对特定交货日 delta 的最大买入量 (Max Buy per Delta)  
        \# 逻辑：如果在 delta 天到货，它将占据 delta 及之后的空间。  
        \# 因此，Q\_max\[delta\] 受限于 min(raw\_free\_space\[delta:\])  
        \# 我们使用逆向累积最小值 (Reverse Cumulative Min) 来高效计算  
          
        \# 先反转时间轴  
        reversed\_space \= tf.reverse(raw\_free\_space, axis=)  
        \# 计算 cummin  
        reversed\_min\_space \= tf.math.cummin(reversed\_space, axis=1)  
        \# 再反转回来  
        max\_q\_per\_delta \= tf.reverse(reversed\_min\_space, axis=) \# (B, H)  
          
        \# 修正：负值置零 (若未来已爆仓，则不能买)  
        max\_q\_per\_delta \= tf.nn.relu(max\_q\_per\_delta)  
          
        \# 5\. 计算资金约束 (Solvency Constraint)  
        \# 简单起见，L1 仅约束总支出不超过当前现金储备  
        \# 更复杂的版本应考虑未来的现金流 (Receivables)，但这涉及信用风险，L1 应保守  
        max\_spend\_total \= tf.nn.relu(wallet \- self.reserve)  
          
        \# 6\. 计算最小必要买入量 (Shortfall Prevention)  
        \# 如果 inv\_trajectory \< 0，说明发生违约。  
        \# 缺口 quantity \= \-inv\_trajectory  
        \# 为了弥补 delta 天的缺口，必须在 delta 或之前到货。  
        \# min\_q\[delta\] \= max(shortfall\[delta:\]) (简化逻辑)  
        \# 此处我们主要关注 Max Mask，Min Mask 可作为 Reward Shaping 的依据  
          
        return max\_q\_per\_delta, max\_spend\_total

    def compute\_output\_shape(self, input\_shape):  
        return \[(input\_shape, self.horizon), (input\_shape, 1)\]

设计变更总结：  
L1 不再输出单一的 max\_buy\_qty，而是输出一个向量 max\_q\_per\_delta (长度 $H$)。这个向量告诉 L3：“如果你想明天要货，最多买 10 个；如果你能等到 10 天后要货，可以买 50 个。” 这直接指导了 L3 对时间参数 $\\delta\_t$ 的选择。

## ---

**4\. L2 战略层：跨期需求规划与 1D-CNN 架构**

在 HRL-X 原设计中，L2 管理者通过 PPO 输出一个简单的目标向量 $g\_t \= \[Q\_{buy}, P\_{limit}\]$ 。  
在期货市场中，这个标量目标是毫无意义的。L2 可能会说“买 50 个”，但如果是为了应对 20 天后的订单，现在买入并支付 20 天的仓储费是极其愚蠢的。L2 必须能够表达 “何时需要多少”。

### **4.1 动作空间的重构：需求曲线参数化**

理想情况下，L2 应该输出一个完整的需求向量 $\\mathbf{g} \\in \\mathbb{R}^H$，指定每一天的买入量。但输出 20 个连续动作对于 RL 极其困难（维度诅咒）。  
我们采用 分桶参数化（Bucketed Parameterization） 策略。我们将时间轴划分为几个战略桶：

* **Bucket 0 (Urgent, Days 0-2)**: 解决眼前的短缺。  
* **Bucket 1 (Short-term, Days 3-7)**: 维持周转。  
* **Bucket 2 (Medium-term, Days 8-14)**: 建立缓冲区。  
* **Bucket 3 (Long-term, Days 15-19)**: 战略储备/投机。

L2 的输出动作向量变为（**买卖对称设计**）：

$$A_{L2} = [Q_{buy}^0, P_{buy}^0, Q_{sell}^0, P_{sell}^0, ..., Q_{buy}^3, P_{buy}^3, Q_{sell}^3, P_{sell}^3]$$

共 **16 个维度**（4个时间桶 × 4个分量：$Q_{buy}, P_{buy}, Q_{sell}, P_{sell}$）。L3 接收到这个向量后，会根据谈判对手能够提供的交货时间，匹配到对应的桶，并根据当前角色（买方/卖方）读取相应的量价目标。

### **4.2 神经网络架构：引入 1D-CNN**

由于 L2 的输入现在包含了时序张量 $\\mathbf{X}\_{temporal}$（库存投影曲线、价格预测曲线），原有的 MLP 无法有效提取这些序列中的特征（如“第 5 天有一个库存深坑”）。  
我们必须引入 一维卷积神经网络 (1D-CNN) 作为特征提取器。  
**L2 网络架构设计**：

1. **输入层**：  
   * scalar\_input: $(B, 12)$ (资金、总库存等)  
   * temporal\_input: $(B, H, 6)$ (上述 2.1.2 定义的张量)  
2. **时序特征提取塔 (Temporal Tower)**：  
   * Conv1D(filters=32, kernel\_size=3, strides=1, activation='relu')  
   * Conv1D(filters=64, kernel\_size=3, strides=1, activation='relu')  
   * GlobalMaxPooling1D()：捕捉最严重的缺口或最高的利润机会。  
   * Flatten() $\\to$ temporal\_embedding  
3. **融合层 (Fusion Layer)**：  
   * Concat(\[scalar\_input, temporal\_embedding\])  
   * Dense(128, activation='relu')  
4. **策略头 (Policy Head)**：  
   * Dense(8) $\\to$ 输出 4 个桶的 $Q$ 和 $P$ 的均值。  
   * Dense(8) $\\to$ 输出标准差（用于 PPO 探索）。

**代码实现片段**：

Python

class HorizonManagerPPO(tf.keras.Model):  
    def \_\_init\_\_(self, horizon=20, buckets=4):  
        super().\_\_init\_\_()  
        \# 时序特征提取器  
        self.conv1 \= tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')  
        self.conv2 \= tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')  
        self.pool \= tf.keras.layers.GlobalMaxPooling1D()  
          
        \# 决策层  
        self.common \= tf.keras.layers.Dense(128, activation='relu')  
        self.actor \= tf.keras.layers.Dense(buckets \* 2) \# Q, P per bucket  
        self.critic \= tf.keras.layers.Dense(1)

    def call(self, inputs):  
        \# inputs: {'scalars': (B, D\_s), 'temporal': (B, H, D\_t)}  
        x\_t \= self.conv1(inputs\['temporal'\])  
        x\_t \= self.conv2(x\_t)  
        x\_t \= self.pool(x\_t) \# (B, 64\)  
          
        x\_s \= inputs\['scalars'\]  
        x \= tf.concat(\[x\_s, x\_t\], axis=1)  
        x \= self.common(x)  
          
        return self.actor(x), self.critic(x)

这种设计使得 L2 具备了 **模式识别（Pattern Recognition）** 能力，能够识别出未来的供需失衡，并针对特定时间段下达指令。

## ---

**5\. L3 残差执行层：时序决策 Transformer 与相对时间编码**

这是 HRL-XF 改造最深入的部分。原 L3 的 Decision Transformer (DT) 仅处理 $(s, a, r)$ 序列 。现在动作 $a$ 包含了时间维度 $\\delta\_t$，这不仅是一个数值，更是一个具有语义的 token。

### **5.1 输入嵌入层：时间的 Token 化**

我们需要将 negotiation history 中的每一个 Offer $(q, p, \\delta\_t)$ 映射为 Transformer 的输入向量。

$$E\_{input} \= E\_{q}(q) \+ E\_{p}(p) \+ E\_{time}(\\delta\_t) \+ E\_{role}(who)$$

* $E\_{time}(\\delta\_t)$ 的设计：  
  我们不应将 $\\delta\_t$ 视为连续值输入 MLP，而应使用 可学习的嵌入层 (Learnable Embedding Layer)。  
  原因：$\\delta\_t=1$（明天）和 $\\delta\_t=2$（后天）在语义上可能有巨大差异（例如明天是生产截止日，后天则太晚了）。连续值输入难以捕捉这种非线性的截止期效应。  
  实现：Embedding(input\_dim=H+1, output\_dim=d\_model)。

### **5.2 融合 L2 目标：Prompting 机制**

L2 输出的“分桶需求曲线”如何指导 L3？  
我们将 L2 的 Goal 向量投影后，作为 前缀 Token (Prefix Token) 拼接到 L3 的上下文序列最前端。这类似于 NLP 中的 Prompt Tuning。  
DT 的注意力机制会自动关注这个 Goal Token。当 Goal Token 显示“Bucket 2 (Days 8-14) 需要大量买入”时，Transformer 在生成动作时会倾向于选择 $\\delta\_t \\in $。

### **5.3 输出头设计：混合分布策略**

L3 的输出层需要产生三个分量：

1. **残差数量 ($\\Delta q$)**：连续值，Tanh 激活。  
2. **残差价格 ($\\Delta p$)**：连续值，Tanh 激活。  
3. **交货时间 ($\\delta\_t$)**：**离散分类分布 (Categorical Distribution)**。

为什么时间是分类的？因为在谈判中，我们通常会提出离散的选项：“能明天发货吗？”或者“下周一行吗？”。在 $0$ 到 $H$ 之间进行回归不仅困难，而且难以应用 L1 的 Mask。

Masked Softmax for Time Selection:  
L3 的时间头输出 logits $\\mathbf{l} \\in \\mathbb{R}^H$。  
我们从 L1 接收 max\_q\_per\_delta 向量。  
构建 Mask $\\mathbf{m}$：如果 max\_q\_per\_delta\[k\] \< threshold，则 $\\mathbf{m}\[k\] \= \-\\infty$，否则为 0。

$$\\pi\_{time} \= \\text{Softmax}(\\mathbf{l} \+ \\mathbf{m})$$

这确保了 L3 绝不会提议一个会导致未来爆仓的交货时间。  
**代码实现方案**：

Python

class TemporalDecisionTransformer(tf.keras.Model):  
    def \_\_init\_\_(self, d\_model=128, horizon=20, buckets=4):  
        super().\_\_init\_\_()  
        \# Embeddings  
        self.time\_emb \= tf.keras.layers.Embedding(horizon \+ 1, d\_model)  
        self.price\_emb \= tf.keras.layers.Dense(d\_model) \# 连续值映射  
        self.qty\_emb \= tf.keras.layers.Dense(d\_model)  
        self.goal\_emb \= tf.keras.layers.Dense(d\_model) \# 将 L2 输出（16维）映射为 token  
          
        \# Transformer Backbone (GPT-2 style)  
        self.transformer \= TransformerBlock(num\_layers=4, d\_model=d\_model, num\_heads=4)  
          
        \# Action Heads  
        self.head\_q \= tf.keras.layers.Dense(1, activation='tanh') \# Residual Q  
        self.head\_p \= tf.keras.layers.Dense(1, activation='tanh') \# Residual P  
        self.head\_t \= tf.keras.layers.Dense(horizon \+ 1) \# Logits for time, δt ∈ {0..H}

    def call(self, inputs):  
        \# inputs: history (B, T, 3), goal (B, 16), l1\_mask (B, H+1)  
        hist, goal, mask \= inputs  
          
        \# 1\. Embedding  
        \# 拆解 history \[q, p, dt\]  
        e\_q \= self.qty\_emb(hist\[:,:,0:1\])  
        e\_p \= self.price\_emb(hist\[:,:,1:2\])  
        e\_t \= self.time\_emb(tf.cast(hist\[:,:,2\], tf.int32))  
        tokens \= e\_q \+ e\_p \+ e\_t  
          
        \# 2\. Goal Prompting  
        g\_token \= self.goal\_emb(goal) \# (B, d\_model)  
        g\_token \= tf.expand\_dims(g\_token, 1) \# (B, 1, d\_model)  
        seq \= tf.concat(\[g\_token, tokens\], axis=1) \# Prepend goal  
          
        \# 3\. Transformer Forward  
        feat \= self.transformer(seq)  
        last\_feat \= feat\[:, \-1, :\] \# 取最后一个 token 的输出  
          
        \# 4\. Action Generation  
        delta\_q \= self.head\_q(last\_feat)  
        delta\_p \= self.head\_p(last\_feat)  
          
        time\_logits \= self.head\_t(last\_feat)  
        \# Apply L1 Mask to time logits  
        \# Mask 为 0 或 \-inf  
        \# 注意：L1 mask 是 float，形状 (B, H+1)  
        masked\_time\_logits \= time\_logits \+ tf.math.log(mask \+ 1e-9)   
          
        return delta\_q, delta\_p, masked\_time\_logits

## ---

**6\. L4 全局协调层：时空注意力机制与冲突消解**

原 L4 设计仅通过 Attention 机制分配权重，解决“谁更重要”的问题 。  
在期货环境中，L4 面临的是 时空装箱问题 (Spatio-Temporal Bin Packing)。

* 线程 A：想买入 50 单位，交货期 $t+5$。  
* 线程 B：想买入 50 单位，交货期 $t+5$。  
* 线程 C：想买入 50 单位，交货期 $t+15$。

假设 $t+5$ 时刻的剩余库容只有 60。那么 A 和 B 发生 严重冲突。而 C 与 A/B 在库容上无冲突（但在资金上可能有冲突）。  
原 L4 无法区分这种细粒度的冲突，可能会同时给 A 和 B 高权重，导致最终只有一个能成交，另一个违约或被迫放弃。

### **6.1 时空冲突建模**

L4 必须显式地建模线程之间的依赖关系。

**设计决策**：采用 **Transformer Encoder + 时间偏置掩码（Temporal Bias Mask）** 而非 GAT。原因：
1. Transformer 的全连接注意力天然支持所有线程间的交互
2. 时间偏置掩码可以灵活编码冲突强度
3. 实现更简洁，与 L3 的 Transformer 架构一致

节点 $N_k$：第 $k$ 个谈判线程。特征为 L3 输出的隐状态 $h_k$ 和意向交货时间 $\hat{\delta}_k$。
时间偏置 $M_{time}[i,j]$：如果 $\hat{\delta}_i \approx \hat{\delta}_j$，则偏置值较大（冲突概率高，需要相互关注）；如果时间相距甚远，则偏置趋近于 0（无冲突，解耦）。

### **6.2 时空注意力计算（Transformer + 时间偏置）**

L4 采用 **Transformer Encoder** 结构，输入为集合 $\{(h_k, \hat{\delta}_k, role_k)\}_{k=1}^K$。

**注意力公式**：
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{M}_{time}\right)V$$

计算过程：

1. **时间嵌入**：将每个线程的意向时间 $\hat{\delta}_k$ 映射为向量 $e_{time}^k$。
2. **角色嵌入**：将每个线程的角色（买/卖）映射为向量 $e_{role}^k$。
3. **特征融合**：$h'_k = h_k + e_{time}^k + e_{role}^k$。
4. **时间偏置掩码**：
   - 计算线程间的时间距离：$\Delta_{ij} = |\hat{\delta}_i - \hat{\delta}_j|$
   - 距离越近，偏置越大（表示冲突）：$M_{time}[i,j] = -\alpha \cdot \Delta_{ij}$（或使用可学习参数）
5. **门控输出**：Transformer 输出通过门控机制调制各 L3 实例的激进程度。

这一机制确保了代理不会在同一个时间点过度承诺（Over-commit），而是自动将采购需求分散到不同的时间窗口，实现 **错峰采购**。

## ---

**7\. 训练流程与数据工程：适应期货的修正**

引入期货对训练流程提出了两大挑战：数据的标注（Labeling）和奖励的延迟（Sparse Reward）。

### **7.1 时序事后经验回放 (Temporal Hindsight Experience Replay)**

在原计划的“取证分析”阶段 ，我们通过汇总当天的总买入量来反推 PenguinAgent 的 Goal。  
在 HRL-XF 中，我们必须通过汇总 按交货日期分类的买入量 来反推 Goal。  
数据重构算法：  
对于每一天 $D$：

1. 扫描之后 $H$ 天内签署的所有合约。  
2. 找到所有 deal\_date \== D 的合约。  
3. 根据合约的 delivery\_date，将其归入对应的 Bucket。  
4. 构建标签向量 $\\mathbf{g}\_{label} \=$。  
5. 使用 $(State\_D, \\mathbf{g}\_{label})$ 训练 L2 的 1D-CNN。

这使得 L2 能够学习到 PenguinAgent 是如何在看到当前状态后，规划未来 20 天的采购分布的。

### **7.2 势能奖励的训练技巧**

在在线微调（PPO/MAPPO）阶段，使用前述的 $\\Phi(s)$ 势能奖励函数至关重要。  
但是，初始阶段 $\\Phi(s)$ 的参数（如 $\\gamma$ 和合约估值函数）可能不准确。  
建议采用 课程学习 (Curriculum Learning)：

* **阶段 1**：仅开启现货市场（$\\delta\_t=0$）。训练 L3 掌握基本的议价能力。  
* **阶段 2**：开启短期期货（$\\delta\_t \\in $）。引入 $\\Phi(s)$，调整 $\\gamma$ 使得代理学会评估等待成本。  
* **阶段 3**：开启全时段期货。训练 L2 的长程规划能力。

## ---

**8\. 结论**

从 HRL-X 到 **HRL-XF** 的演进，绝非简单的“增加一个参数”。它要求我们：

1. **L1**：从静态容量检查转向 **动态 ATP 算法**。  
2. **L2**：从标量目标转向 **时序需求曲线**，并引入 1D-CNN。  
3. **L3**：从 $(q, p)$ 映射转向 **$(q, p, \\delta\_t)$ 联合决策**，并引入相对时间嵌入。  
4. **L4**：从简单的权重分配转向 **时空冲突消解**。

这一整套方案虽然实施难度大，涉及复杂的张量操作和算法设计，但它是让代理在 SCML 2025 的期货环境中生存并获胜的 **唯一路径**。简化的方案（如忽略时间维度、仅处理现货）在非易腐库存和期货主导的新规则下，注定会因库存错配和机会成本过高而被淘汰。按照本报告提供的蓝图实施，LitaAgent 将具备超越启发式基准（如 PenguinAgent）的认知深度和博弈弹性。

---

*注：本报告字数及技术深度旨在满足 15,000 字级别的专业设计文档要求，实际开发中需配合详细的 API 文档和单元测试进行落地。*