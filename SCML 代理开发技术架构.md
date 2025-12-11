# **LitaAgent-HRL (HRL-X) 技术架构与实施蓝皮书：面向 SCML 2025 标准联赛的深度强化学习解决方案**

## **1\. 绪论：供应链管理的认知飞跃与 SCML 2025 的挑战**

### **1.1 赛道背景与范式转移**

国际自动协商代理竞赛（ANAC）中的供应链管理联赛（SCML）长期以来被视为多智能体系统（MAS）与自动谈判技术的顶级试验场。随着比赛进入 2025 赛季，标准联赛（Standard League）迎来了一次深刻的规则重构，这标志着自动代理设计从“战术型单次博弈”向“战略型长期规划”的范式转移。此次变革的核心在于引入了非易腐库存（Non-Perishable Inventory）、期货合约（Future Contracts）以及更为严苛的短缺惩罚（Shortfall Penalties）机制 1。

在早期的 SCML 赛季中，库存往往在每日结束时清零，这种设定将复杂的供应链问题简化为一系列相互独立的单日“报童问题”（Newsvendor Problem）。代理只需关注当天的供需平衡，无需考虑跨日的库存结转成本或未来的市场波动。然而，SCML 2025 的新规则打破了这一假设。库存的非易腐性意味着代理不仅是生产者，更是资产管理者；期货合约的引入使得当前的决策后果可能在数十个仿真日之后才显现，极大地拉长了信用分配（Credit Assignment）的链条；而严厉的短缺惩罚则将环境从单纯的利润最大化转变为带有硬约束的风险敏感型博弈 1。

回顾 2024 年及之前的赛果，以 PenguinAgent 和 AS0 为代表的启发式代理（Heuristic Agents）凭借其稳健的成本控制逻辑和确定性的风险规避策略，占据了排行榜的前列 4。然而，这种“硬编码”的智慧在面对 2025 年日益复杂的动态市场时显露出了明显的局限性：它们缺乏对非平稳市场环境的自适应能力，难以利用长尾分布中的投机机会（如跨期套利），且无法有效处理由期货交易带来的状态空间指数级膨胀问题。为了突破这一策略天花板，我们提出了 **HRL-X（Hierarchical Residual Learner \- Extended）** 架构，并基于此开发 **LitaAgent-HRL**。

### **1.2 本报告的宗旨与结构**

本报告旨在为 LitaAgent-HRL 的开发提供一份详尽无遗的技术蓝图。不同于常规的概要设计，本文档将深入代码实现层面，结合 TensorFlow 2.x 的具体特性，详细阐述 HRL-X 架构中每一个组件的数学原理、张量流向、工程实现细节以及完整的训练路线图。我们将通过严谨的数学建模，剖析 SCML 2025 环境下的部分可观察马尔可夫决策过程（POMDP）特性，并论证 HRL-X 如何通过分层控制、残差学习和全局注意力机制，系统性地解决库存耦合、并发协商和探索风险这三大核心难题 5。

报告结构如下：

* **第 2 章**：对 SCML 2025 环境进行数学解构，定义状态空间、动作空间及核心耦合问题。  
* **第 3 章**：概述 HRL-X 的总体架构拓扑与数据流。  
* **第 4-7 章**：深入剖析 L1 安全护盾、L2 战略管理者、L3 残差执行者及 L4 全局协调器的技术细节与 TensorFlow 代码实现。  
* **第 8 章**：详述奖励函数工程与特征提取方法。  
* **第 9 章**：制定分阶段的课程学习训练路线图，包括离线 ROL 算法与在线 MAPPO 的具体应用。  
* **第 10 章**：总结与展望。

## ---

**2\. 环境数学建模：高维 POMDP 下的耦合博弈**

在着手代码实现之前，必须透彻理解 LitaAgent-HRL 所处的数学环境。SCML 2025 本质上是一个高维、随机、部分可观察的马尔可夫博弈，我们需要将其形式化为一个 POMDP 元组 $\\langle \\mathcal{S}, \\mathcal{A}, \\mathcal{T}, \\mathcal{R}, \\Omega, \\mathcal{O}, \\gamma \\rangle$。

### **2.1 状态空间的非完全可观察性与信息不对称**

**全局状态 $\\mathcal{S}$** 包含了上帝视角下的所有市场信息：每一个竞争对手 $i \\in \\{1, \\dots, N\\}$ 的私有库存水平 $I\_t^i$、资金状况 $B\_t^i$、生产线闲置率、隐藏的生产成本函数 $C^i(\\cdot)$，以及未来 $T$ 天的真实市场外生需求曲线 $D(p, t)$。显然，对于单一代理 LitaAgent-HRL 而言，这些关键信息是完全屏蔽的 6。

**观测空间 $\\Omega$** 是代理在时刻 $t$ 能够获取的局部信息切片，定义为 $o\_t \= \\langle I\_t, B\_t, \\mathcal{H}\_t, \\mathcal{M}\_t \\rangle$：

* $I\_t \\in \\mathbb{R}^{|Products|}$：当前的私有库存向量（包括原材料、中间品与成品）。  
* $B\_t \\in \\mathbb{R}$：当前的现金余额。  
* $\\mathcal{H}\_t \= \\{h\_t^1, h\_t^2, \\dots, h\_t^K\\}$：当前所有 $K$ 个活跃协商线程的历史记录集合。每个线程的历史 $h\_t^k$ 是一个序列，包含了从谈判开始至今的所有出价 $(p, q)$、对手的响应动作及时间戳。这是推断对手意图的关键数据源。  
* $\\mathcal{M}\_t$：公告板（Bulletin Board）上的公开市场数据，如昨日的加权平均交易价格、总交易量等。这些数据存在滞后性且含有噪声 6。

这种严重的信息不对称导致了环境的非平稳性（Non-Stationarity）。从 LitaAgent-HRL 的局部视角看，环境的转移概率 $\\mathcal{T}(s'|s, a)$ 似乎随时间动态变化，因为对手的策略 $\\pi^{-i}$ 在不断演变，而这些变化无法被直接观测。因此，我们的架构必须具备信念状态重构（Belief State Reconstruction）的能力，即利用序列模型（如 Transformer）从观测历史 $\\mathcal{H}\_t$ 中提取隐变量 $z\_t$，以逼近真实状态。

### **2.2 动作空间的组合爆炸与混合特性**

SCML 的动作空间 $\\mathcal{A}$ 是典型的混合型空间（Hybrid Action Space），兼具离散与连续特征。对于每一个协商线程 $k$，代理需要做出如下决策：

* **离散动作**：$a\_{disc} \\in \\{\\text{Accept}, \\text{Reject}, \\text{End Negotiation}, \\text{Offer}\\}$。  
* **连续参数**：如果是 Offer 动作，需指定价格 $p \\in \\mathbb{R}^+$ 和数量 $q \\in \\mathbb{N}$。

更致命的挑战在于**并发协商（Concurrent Negotiation）**。代理在同一时刻可能维持着 $K$ 个并行的谈判线程（$K$ 可达数十甚至上百）。这意味着联合动作空间的大小随 $K$ 呈指数级增长，即 $|\\mathcal{A}\_{joint}| \\propto |\\mathcal{A}\_{single}|^K$。传统的强化学习方法（如 DQN 或标准 Actor-Critic）在面对如此巨大的动作空间时往往难以收敛。因此，HRL-X 架构必须采用分解（Decomposition）与协调（Coordination）相结合的策略 1。

### **2.3 2025 规则引发的“三重耦合难题”**

LitaAgent-HRL 的核心设计逻辑旨在解决由 2025 年新规则引发的三重深度耦合问题。

#### **2.3.1 库存-时间的跨期耦合 (Inventory-Time Coupling)**

非易腐库存规则的引入彻底改变了库存管理的逻辑。在旧规则下，库存每日清零，代理只需关注当天的供需平衡（Just-In-Time）。而在新规则下，库存可以累积，这引入了跨期套利（Inter-temporal Arbitrage）的可能性。  
代理可以在原材料价格低廉时（$t$ 时刻）大量囤积，承担持有成本，待市场价格回升后（$t+H$ 时刻）生产并出售成品。这种决策具有极长的因果链条：今日的买入动作可能旨在满足十天后的市场需求。这意味着奖励信号在时间维度上极度稀疏且延迟，传统的扁平化 RL 难以处理这种长达数百步的信用分配（Credit Assignment）问题。因此，必须引入分层强化学习（HRL），利用高层管理者进行跨天规划 5。

#### **2.3.2 并发-资源的竞争耦合 (Concurrency-Resource Coupling)**

多个并发的协商线程虽然在通信上是独立的，但在资源上却是紧密耦合的。它们竞争同一个有限的资源池：

* **资金耦合**：如果在上游过度激进地锁定原材料，可能导致现金流枯竭，无法支付下游交易的违约金或生产成本。  
* **产能耦合**：生产线的产能是有限的。如果承诺了过多的下游订单，超出生产能力，将触发高额的短缺惩罚。  
* **库存耦合**：上游买入的原料如果没有下游订单消化，将导致爆仓，产生高额仓储费。

单一线程的最优解（例如以极低价格买入大量原料）如果缺乏全局视角的配合，可能对整体效用是灾难性的。这要求架构中必须包含一个**全局协调器（Global Coordinator）**，利用注意力机制在不同线程间动态分配权重 5。

#### **2.3.3 风险-探索的生存耦合 (Risk-Exploration Coupling)**

取消现货市场并引入严厉的短缺惩罚，使得环境具有极端的风险不对称性。对于 RL 代理而言，这意味着**探索（Exploration）的代价极其高昂。在训练初期，基于 $\\epsilon$-greedy 或高熵策略的随机探索极易触发连续违约，导致代理破产并提前终止回合。这种“死亡陷阱”使得代理难以收集到成功的正样本，容易陷入“不敢交易”的次优局部极小值。因此，必须引入安全护盾（Safety Shield）**，通过确定性规则强制约束探索边界 5。

## ---

**3\. LitaAgent-HRL (HRL-X) 总体架构**

针对上述数学模型与挑战，我们设计了 HRL-X 架构。这是一个集成了规则控制、分层规划、序列建模与多智能体协调的混合系统。本章将概述其拓扑结构与数据流向。

### **3.1 架构拓扑：四层混合控制系统**

LitaAgent-HRL 由四个逻辑层级组成，自底向上分别负责安全约束、战术执行、全局协调与战略规划：

| 层级 | 组件名称 | 核心技术 | 时间尺度 | 核心职责 |
| :---- | :---- | :---- | :---- | :---- |
| **L1** | **安全护盾 (Safety Shield)** | 启发式规则 (Penguin Logic) | 实时 (Real-time) | 生成动作掩码 $\\mathcal{M}\_{safe}$，提供基准动作 $a\_{base}$，防止破产与违约。 |
| **L2** | **战略管理者 (Strategic Manager)** | PPO (Proximal Policy Optimization) | 天 (Daily) | 基于宏观状态设定每日战略目标向量 $g\_t$（买卖量与限价）。 |
| **L3** | **残差执行者 (Residual Actor)** | 决策 Transformer (Decision Transformer) | 轮 (Round) | 基于微观协商历史与目标 $g\_t$，输出相对于基准的动作残差 $\\Delta a$。 |
| **L4** | **全局协调器 (Global Coordinator)** | 注意力机制 (Centralized Attention) | 轮 (Round) | 处理并发谈判，计算各线程的重要性权重 $\\alpha\_k$，调节 L3 的激进程度。 |

### **3.2 数据流与张量计算图**

在 TensorFlow 的计算图中，每一决策步 $t$ 的数据流向如下：

1. **观测解析**：原始观测 $o\_t$ 被拆解。宏观特征（总库存、资金、天数）流向 L2；微观特征（各线程的报价序列）流向 L3。  
2. **安全计算 (L1)**：  
   * L1 根据当前 $I\_t, B\_t$ 及已签合约，计算硬约束。  
   * 输出：动作掩码张量 mask\_tensor（用于 Logits 裁剪）和基准动作张量 base\_action。  
3. **战略规划 (L2 \- 仅在每日开始时激活)**：  
   * L2 接收宏观特征，通过 MLP 网络输出高斯分布参数。  
   * 采样得到当天的目标向量 goal\_vector（如：今日需买入 50 单位，最高价 100）。此向量被广播（Broadcast）给所有 L3 实例。  
4. **微观编码 (L3)**：  
   * 每个活跃线程的 L3 决策 Transformer 接收各自的历史序列 history\_seq 和 goal\_vector。  
   * Transformer 输出当前线程的隐状态 hidden\_state。  
5. **全局协调 (L4)**：  
   * L4 接收所有线程的 hidden\_state 集合。  
   * 通过多头自注意力（Multi-Head Self-Attention）计算各线程的注意力权重 attention\_weights。  
   * 高权重的线程意味着其对全局目标至关重要（如急需的原料采购）。  
6. **动作生成与合成**：  
   * L3 结合 hidden\_state 和 L4 反馈的 attention\_weights，输出残差动作 delta\_action。  
   * 最终动作合成：final\_action \= clip(base\_action \+ delta\_action, mask\_tensor)。  
   * final\_action 被发送至 SCML 模拟器执行。

## ---

**4\. L1 安全护盾层：确定性逻辑与 TensorFlow 实现**

L1 安全护盾是 LitaAgent-HRL 的基石。它不涉及梯度更新，而是通过硬编码的领域知识（Domain Knowledge）确保代理在任何时候都不会采取自杀式行动。我们借鉴了 2024 年冠军 PenguinAgent 的核心逻辑 5。

### **4.1 核心逻辑：Penguin 范式的数学表达**

安全护盾的核心任务是计算**可行且安全**的动作边界。

1\. 最大安全买入量 ($Q\_{max\\\_buy}$)  
为了防止爆仓（导致仓储费剧增），买入量必须受限于剩余库容。考虑到已签署但未交付的入库量 $I\_{incoming}$ 和已承诺发货的出库量 $O\_{committed}$：

$$Q\_{max\\\_buy} \= \\text{Capacity} \- I\_{current} \- I\_{incoming} \+ O\_{committed}$$

任何 $q \> Q\_{max\\\_buy}$ 的买入 Offer 都是非法的。  
2\. 最小必要买入量 ($Q\_{min\\\_buy}$)  
为了防止违约（Shortfall Penalty），必须确保有足够的原料来生产已承诺的订单：

$$Q\_{min\\\_buy} \= \\max(0, O\_{committed} \- I\_{current} \- I\_{incoming})$$

这是满足当前合约的底线。  
3\. 破产保护价格 ($P\_{limit}$)  
为了防止现金流断裂，总支出不能超过当前可用资金减去保留金（Reserve）：

$$P\_{limit}(q) \= \\frac{B\_t \- \\text{Reserve}}{q}$$  
4\. 基准动作 ($a\_{base}$)  
L1 不仅提供约束，还提供一个“及格”的动作建议。基于 PenguinAgent 的保守策略，基准动作通常是：

* **买入**：报价量 $q \= Q\_{min\\\_buy}$，价格 $p \= \\text{Cost}\_{production} \\times (1 \+ \\text{Margin}\_{min})$。  
* **卖出**：报价量 $q \= I\_{free}$，价格 $p \= \\text{Market}\_{avg} \\times (1 \+ \\text{Premium})$。

### **4.2 TensorFlow 自定义层实现**

我们将 L1 封装为一个标准的 Keras Layer，以便无缝嵌入模型图。虽然其内部逻辑不可微，但它可以作为张量预处理步骤。

Python

import tensorflow as tf

class SafetyMaskingLayer(tf.keras.layers.Layer):  
    """  
    L1: Safety Shield Layer.  
    负责计算动作掩码与基准动作，不含可训练参数。  
    """  
    def \_\_init\_\_(self, warehouse\_capacity, bankruptcy\_reserve=1000.0, \*\*kwargs):  
        super(SafetyMaskingLayer, self).\_\_init\_\_(\*\*kwargs)  
        self.capacity \= tf.constant(warehouse\_capacity, dtype=tf.float32)  
        self.reserve \= tf.constant(bankruptcys\_reserve, dtype=tf.float32)

    def call(self, inputs):  
        \# inputs shape: (batch\_size, 5\)  
        \# \[current\_inventory, wallet\_balance, incoming\_committed, outgoing\_committed, market\_price\]  
        inventory \= inputs\[:, 0\]  
        wallet \= inputs\[:, 1\]  
        incoming \= inputs\[:, 2\]  
        outgoing \= inputs\[:, 3\]  
        market\_price \= inputs\[:, 4\]

        \# \--- 1\. 计算约束边界 \---  
          
        \# 预测库存峰值：当前 \+ 并在路 \- 拟出库  
        \# 注意：这里做保守估计，假设出库可能失败，故计算最大买入时不扣减 outgoing 可能更安全，  
        \# 但 Penguin 逻辑允许扣减以最大化流转。  
        projected\_inventory \= inventory \+ incoming \- outgoing  
          
        \# Max Buy: 剩余空间  
        max\_buy\_qty \= tf.nn.relu(self.capacity \- projected\_inventory)  
          
        \# Min Buy: 缺口  
        min\_buy\_qty \= tf.nn.relu(outgoing \- (inventory \+ incoming))  
          
        \# Max Spend Price per unit (simplified for tensor op)  
        \# 实际操作中价格上限通常依赖于数量，这里计算一个针对单单位的理论上限  
        max\_price\_total \= tf.nn.relu(wallet \- self.reserve)

        \# \--- 2\. 生成基准动作 (Baseline Action) \---  
        \# 策略：只买必须的 (Min Buy)，如果无需买则为 0  
        base\_buy\_qty \= min\_buy\_qty  
        \# 策略：基准价格设为市场价  
        base\_buy\_price \= market\_price

        \# 组合基准动作向量 \[qty, price\]  
        baseline\_action \= tf.stack(\[base\_buy\_qty, base\_buy\_price\], axis=1)

        \# \--- 3\. 生成掩码边界 (Bounds) \---  
        \# 返回 \[min\_q, max\_q, max\_total\_spend\]  
        bounds \= tf.stack(\[min\_buy\_qty, max\_buy\_qty, max\_price\_total\], axis=1)

        return baseline\_action, bounds

    def compute\_output\_shape(self, input\_shape):  
        return \[(input\_shape, 2), (input\_shape, 3)\]

**实现细节说明**：

* 使用 tf.nn.relu 替代 tf.maximum(0, x) 以保持图的简洁性。  
* bounds 张量并不直接作为 Softmax 的 mask，而是作为后续 L3 输出层 tf.clip\_by\_value 的参数。这是一种针对连续动作空间的软掩码机制。

## ---

**5\. L2 战略管理者：分层 PPO 代理设计**

L2 管理者是 HRL-X 的大脑，负责跨天的时间抽象（Temporal Abstraction）。它解决的是“今天该采取何种库存姿态”的问题，而不关心具体的谈判话术。

### **5.1 状态与动作空间定义**

管理者状态空间 $\\mathcal{S}\_{mgr}$：  
我们需要构建能够反映宏观供需和长期趋势的特征向量：

1. **库存势能特征**：$\\Phi(s) \= I\_{total} \\times P\_{avg}$。反映当前持仓的总市值。  
2. **资金健康度**：$B\_t / B\_{initial}$。  
3. **期货承诺向量**：一个长度为 $H$（如 10 天）的向量，表示未来每天的净合约量。这需要通过 1D 卷积层（Conv1D）进行特征提取，以感知未来的供需峰谷 5。  
4. **市场趋势**：过去 10 天的市场均价与成交量。

管理者动作空间（目标向量 $g\_t$）：  
管理者输出的是一个指导性的目标向量，而非直接的控制指令：

$$g\_t \= \[Q\_{target\\\_buy}, P\_{limit\\\_buy}, Q\_{target\\\_sell}, P\_{limit\\\_sell}\]$$

* $Q\_{target\\\_buy}$：今日计划总买入量。  
* $P\_{limit\\\_buy}$：买入最高限价（超出此价 L3 将极其谨慎）。  
  这些目标通过 $R\_{intrinsic}$（内在奖励）与 L3 的行为挂钩。

### **5.2 PPO 算法适用性与实现**

对于 L2，我们选择 **PPO (Proximal Policy Optimization)** 算法。原因如下：

1. **稳定性**：管理者的决策频率低（每天一次），样本相对稀缺，PPO 的 Clip 机制能防止策略剧烈震荡。  
2. **连续控制**：目标向量是连续变量，PPO 天然支持高斯分布输出。

以下是使用 TensorFlow 构建 PPO Actor-Critic 网络的实现：

Python

import tensorflow as tf  
import tensorflow\_probability as tfp

class ManagerPPOAgent(tf.keras.Model):  
    def \_\_init\_\_(self, state\_dim, action\_dim=4, clip\_ratio=0.2):  
        super(ManagerPPOAgent, self).\_\_init\_\_()  
        self.clip\_ratio \= clip\_ratio  
        self.action\_dim \= action\_dim  
          
        \# \--- 特征提取层 \---  
        \# 专门处理期货向量的卷积层  
        self.future\_conv \= tf.keras.layers.Conv1D(filters=16, kernel\_size=3, activation='relu')  
        self.flatten \= tf.keras.layers.Flatten()  
          
        \# 共享特征层  
        self.common\_dense \= tf.keras.layers.Dense(128, activation='relu')  
          
        \# \--- Actor 网络 (策略) \---  
        self.actor\_dense \= tf.keras.layers.Dense(64, activation='relu')  
        \# 输出均值和对数标准差  
        self.actor\_out \= tf.keras.layers.Dense(action\_dim \* 2)   
          
        \# \--- Critic 网络 (价值) \---  
        self.critic\_dense \= tf.keras.layers.Dense(64, activation='relu')  
        self.critic\_out \= tf.keras.layers.Dense(1)

    def call(self, inputs):  
        \# inputs: {'scalars': (batch, n), 'futures': (batch, H, 1)}  
        scalars \= inputs\['scalars'\]  
        futures \= inputs\['futures'\]  
          
        \# 处理期货序列特征  
        fut\_feat \= self.future\_conv(futures)  
        fut\_feat \= self.flatten(fut\_feat)  
          
        \# 拼接特征  
        concat\_state \= tf.concat(\[scalars, fut\_feat\], axis=-1)  
        x \= self.common\_dense(concat\_state)  
          
        \# Actor 前向传播  
        a\_x \= self.actor\_dense(x)  
        actor\_params \= self.actor\_out(a\_x)  
        mean, log\_std \= tf.split(actor\_params, 2, axis=-1)  
        \# 限制 log\_std 防止数值不稳定  
        log\_std \= tf.clip\_by\_value(log\_std, \-20, 2)  
        std \= tf.exp(log\_std)  
          
        \# 构建高斯分布  
        dist \= tfp.distributions.Normal(mean, std)  
        action \= dist.sample()  
          
        \# Critic 前向传播  
        c\_x \= self.critic\_dense(x)  
        value \= self.critic\_out(c\_x)  
          
        return action, dist.log\_prob(action), value

    def train\_step(self, data):  
        \# 标准 PPO 训练循环 (伪代码逻辑)  
        \# 计算优势函数 Advantage  
        \# 计算 actor\_loss \= \-min(surr1, surr2)  
        \# 计算 critic\_loss \= MSE(return, value)  
        \# 应用梯度更新  
        pass

## ---

**6\. L3 残差执行者：决策 Transformer 与残差学习**

L3 是系统的战术引擎，负责在每一轮（Round）的微观博弈中与对手周旋。我们采用 **Decision Transformer (DT)** 结合 **残差学习（Residual Learning）** 的方案 1。

### **6.1 为什么要用 Decision Transformer？**

SCML 的协商过程本质上是一个序列决策问题。对手的每一次出价、每一次犹豫（响应时间）、每一次让步，都构成了上下文（Context）。传统的 MLP 或 LSTM 在处理长序列依赖（Long-term Dependency）和因果推理方面能力有限。DT 通过自注意力机制（Self-Attention），能够直接关注到序列中关键的历史节点（例如对手在 Deadline 前的突然降价），从而捕捉其隐性的心理状态（Belief State）。

### **6.2 残差学习机制：$A \= A\_{base} \+ \\Delta A$**

为了解决 RL 代理在初期探索时的不稳定性，L3 并不直接输出绝对动作，而是输出相对于 L1 基准动作的偏置（Residual/Delta）。

$$A\_{final} \= \\text{Clip}(A\_{base} \+ \\Delta A, \\text{Bounds})$$

* **训练初期**：网络权重接近随机，$\\Delta A \\approx 0$，代理表现等同于 PenguinAgent（及格线）。  
* **训练后期**：网络学会了在何时微调价格以榨取更多利润（如 $\\Delta p \= \+5$）或为了成交而快速让步（$\\Delta p \= \-2$）。

### **6.3 TensorFlow Decision Transformer 实现**

我们需要实现一个因果 GPT 风格的 Transformer。

Python

class ResidualDecisionTransformer(tf.keras.Model):  
    def \_\_init\_\_(self, d\_model=128, n\_heads=4, n\_layers=2, max\_len=20, action\_dim=2):  
        super().\_\_init\_\_()  
        self.d\_model \= d\_model  
          
        \# \--- 嵌入层 \---  
        self.state\_emb \= tf.keras.layers.Dense(d\_model)  
        self.goal\_emb \= tf.keras.layers.Dense(d\_model)  
        self.pos\_emb \= tf.keras.layers.Embedding(max\_len, d\_model)  
        self.dropout \= tf.keras.layers.Dropout(0.1)  
          
        \# \--- Transformer Blocks \---  
        self.blocks \=  
        for \_ in range(n\_layers):  
            self.blocks.append({  
                'attn': tf.keras.layers.MultiHeadAttention(num\_heads=n\_heads, key\_dim=d\_model),  
                'ln1': tf.keras.layers.LayerNormalization(),  
                'ffn': tf.keras.Sequential(),  
                'ln2': tf.keras.layers.LayerNormalization()  
            })  
              
        \# \--- 残差输出头 \---  
        \# 输出 \[delta\_qty, delta\_price\]  
        self.action\_head \= tf.keras.layers.Dense(action\_dim, activation='tanh')  
        \# 可学习的缩放因子，控制残差幅度  
        self.residual\_scale \= tf.Variable(\[5.0, 10.0\], trainable=True, dtype=tf.float32)

    def call(self, inputs, training=False):  
        \# inputs: {'history': (B, T, F), 'goal': (B, G\_dim), 'baseline': (B, A\_dim)}  
        history \= inputs\['history'\]  
        goal \= inputs\['goal'\]  
        baseline \= inputs\['baseline'\]  
          
        seq\_len \= tf.shape(history)  
          
        \# 1\. 嵌入  
        x \= self.state\_emb(history) \# (B, T, d\_model)  
          
        \# 2\. 融合 Goal (将 L2 的目标注入到每一个时间步)  
        g \= self.goal\_emb(goal)     \# (B, d\_model)  
        g \= tf.expand\_dims(g, 1)    \# (B, 1, d\_model)  
        x \= x \+ g                   \# 广播相加  
          
        \# 3\. 位置编码  
        positions \= tf.range(start=0, limit=seq\_len, delta=1)  
        x \= x \+ self.pos\_emb(positions)  
        x \= self.dropout(x, training=training)  
          
        \# 4\. 因果掩码 (Causal Mask)  
        \# 确保 t 时刻只能看到 0...t  
        mask \= 1 \- tf.linalg.band\_part(tf.ones((seq\_len, seq\_len)), \-1, 0)  
          
        \# 5\. Transformer Pass  
        for block in self.blocks:  
            \# Self Attention  
            attn\_out \= block\['attn'\](query=x, value=x, key=x, attention\_mask=mask)  
            x \= block\['ln1'\](x \+ attn\_out)  
            \# FFN  
            ffn\_out \= block\['ffn'\](x)  
            x \= block\['ln2'\](x \+ ffn\_out)  
              
        \# 6\. 取最后一个时间步的隐状态  
        last\_hidden \= x\[:, \-1, :\] \# (B, d\_model)  
          
        \# 7\. 计算残差  
        delta \= self.action\_head(last\_hidden) \* self.residual\_scale  
          
        \# 8\. 合成 (这里仅计算合成前的 raw 值，Clip 在外部模型循环中做)  
        final\_raw \= baseline \+ delta  
          
        return final\_raw, delta, last\_hidden

## ---

**7\. L4 全局协调器：基于注意力的多线程调度**

SCML 2025 的核心痛点在于并发协商中的资源冲突。L4 协调器不直接产生动作，而是通过调节 L3 的隐状态或输出权重来解决冲突 5。

### **7.1 集中式注意力机制原理**

协调器接收所有活跃线程 L3 Transformer 输出的隐状态向量集合 $\\{h\_1, h\_2, \\dots, h\_K\\}$。它采用多头注意力机制来计算每个线程的重要性权重 $\\alpha\_k$：

$$Q \= W\_q S\_{global}, \\quad K\_k \= W\_k h\_k$$

$$\\alpha \= \\text{Softmax}\\left(\\frac{Q K^T}{\\sqrt{d\_k}}\\right)$$

* $S\_{global}$：全局状态（库存、资金）。  
* $\\alpha\_k$：线程 $k$ 的注意力权重。

**逻辑解释**：如果 $S\_{global}$ 显示原料库存告急，且线程 A 是唯一能提供该原料的供应商，$Q$ 与 $K\_A$ 的点积将极大，导致 $\\alpha\_A \\to 1$。

### **7.2 策略修正与资源倾斜**

计算出的权重 $\\alpha\_k$ 如何影响决策？我们将其作为一种调制信号反馈给 L3。  
在 L3 的 call 方法中，我们可以修改最后一步：

Python

\# 修改 L3 输出逻辑以接受 L4 的调制  
delta \= self.action\_head(last\_hidden) \* self.residual\_scale \* (1 \+ alpha\_k)

或者，更复杂地，高权重 $\\alpha\_k$ 可以触发 L3 进入“不惜代价成交”模式（放宽对利润的追求），而低权重则触发“高风险博弈”模式。

## ---

**8\. 奖励函数工程与特征提取**

### **8.1 复合奖励函数设计**

为了解决 SCML 极度稀疏的奖励信号，我们设计了如下复合奖励函数 $R\_t$ 5：

$$R\_t \= R\_{profit} \+ \\lambda\_1 R\_{liquidity} \- \\lambda\_2 R\_{risk} \+ \\lambda\_3 R\_{intrinsic}$$

1. 利润奖励 ($R\_{profit}$) 与 势能函数 ($\\Phi$)  
   SCML 2025 中买入原料会导致现金减少（负奖励），若直接优化会导致代理拒绝买入。我们引入势能函数 $\\Phi(s)$ 来衡量库存的潜在价值：

   $$\\Phi(s) \= I\_{inventory} \\times P\_{market\\\_avg}$$

   重塑后的利润奖励为：

   $$R\_{profit} \= (B\_{t+1} \- B\_t) \+ \\gamma \\Phi(s\_{t+1}) \- \\Phi(s\_t)$$

   根据 Ng et al. 的奖励塑形定理，这种形式不会改变最优策略，但能提供即时的正向反馈，鼓励“投资”行为。  
2. \*\*流动性奖励 ($R\_{liquidity}$) \*\*  
   为了防止代理在训练初期因害怕亏损而冻结操作，只要达成任何交易（无论盈亏），给予一个微小的正奖励 $\\epsilon$。

   $$R\_{liquidity} \= \\mathbb{I}(\\text{Deal Executed}) \\times \\epsilon$$  
3. 前瞻性风险惩罚 ($R\_{risk}$)  
   基于 L1 中的预测模型，如果预测未来 $t+H$ 天库存为负（违约风险）：

   $$R\_{risk} \= \\text{Prob}(Shortfall) \\times \\text{PenaltyUnit}$$

   这迫使代理在危机发生前采取行动。  
4. 内在一致性奖励 ($R\_{intrinsic}$)  
   用于对齐 L2 Manager 与 L3 Worker：

   $$R\_{intrinsic} \= \-\\| q\_{executed} \- Q\_{target} \\|^2$$

### **8.2 特征工程公式**

我们需要计算关键的统计特征喂给网络 6：

* **库存压力指数 (IPI)**：$IPI \= I\_{current} / I\_{capacity}$。高 IPI 指示应降价抛售。  
* **相对价格指数 (RPI)**：$RPI \= (P\_{offer} \- P\_{market\\\_avg}) / \\sigma\_{price}$。用于归一化价格输入。

## ---

**9\. 训练路线图与方法论**

训练 LitaAgent-HRL 是一个系统工程，我们制定了严格的四阶段课程学习计划 6。

### **阶段 0: 行为克隆与热启动 (Phase 0: Behavior Cloning & Warm Start)**

* **目标**：解决 RL 冷启动问题，获得一个性能等同于 PenguinAgent 的神经网络副本。  
* **数据源**：SCML 2023-2024 冠军日志 (scml-agents repository)。  
* **算法**：**Reward-on-the-Line (ROL)** 离线强化学习算法 5。  
  * **集合一致性 (Ensemble Agreement)**：训练 $N=5$ 个 Decision Transformer 网络。对于任意状态动作对 $(s, a)$，计算 $N$ 个网络 Q 值的方差。如果方差过大（说明该动作在数据集中未出现，属 OOD），则在 Loss 中施加惩罚。  
  * **加权行为克隆 (WBC)**：只克隆那些 Advantage $A(s, a) \> 0$ 的样本，即只模仿 PenguinAgent 赚钱的操作，过滤掉其亏钱的操作。  
* **产出**：预训练好的 L3 Residual Actor 权重。

### **阶段 1: 无约束环境下的单点突破 (Phase 1: Single-Thread Optimization)**

* **环境设置**：修改 SCML 模拟器，设置无限库存容量，移除短缺惩罚。  
* **目标**：训练 L3 Actor 掌握纯粹的谈判技巧（如让步曲线优化），而不必担心生存问题。  
* **算法**：标准 PPO。冻结 L2 和 L4，仅优化 L3。  
* **训练量**：约 10,000 个 Episodes。

### **阶段 2: 引入库存约束与多线程 (Phase 2: Hierarchical Coordination)**

* **环境设置**：恢复库存限制与短缺惩罚，开启多线程并发。  
* **目标**：训练 L2 Manager 和 L4 Coordinator。  
* **算法**：**MAPPO (Multi-Agent PPO)**。  
  * 此时 L3 Actor 的权重以较小的学习率微调（Fine-tuning）。  
  * 重点优化 L2 的目标设定能力和 L4 的注意力权重分配。  
  * L2 的回报基于每日的总利润；L4 的回报基于全线程的总效用。

### **阶段 3: 全压力对抗演练 (Phase 3: Adversarial Self-Play)**

* **环境设置**：完整的 SCML 2025 标准环境。  
* **对手池 (Opponent Pool)**：  
  * 静态对手：PenguinAgent, AS0 (基准)。  
  * 动态对手：历史版本的 LitaAgent-HRL (Self-Play)。  
* **方法**：随着训练进行，将当前的 LitaAgent-HRL 存入对手池。代理必须不断战胜过去的自己。这有助于发现策略中的漏洞并提升鲁棒性。  
* **Nash 均衡**：目标是逼近博弈的纳什均衡点，使得策略在面对任何对手时都不被利用（Unexploitable）。

## ---

**10\. 结论**

LitaAgent-HRL (HRL-X) 方案代表了应对 SCML 2025 复杂挑战的前沿探索。通过**数学建模**，我们识别了库存、并发与风险的三重耦合；通过**分层架构**，我们将长期战略规划（Manager）与短期战术执行（Worker）解耦；通过**残差学习**与**安全护盾**，我们将领域专家的确定性智慧与深度学习的泛化能力有机融合；最后，通过**全局协调器**，我们解决了多线程资源争夺的难题。

本报告提供的 TensorFlow 代码框架和训练路线图，不仅在理论上自洽，更在工程上具备高度的可操作性。随着 Phase 0 到 Phase 3 的逐步实施，LitaAgent-HRL 有望在 SCML 2025 标准联赛中展现出超越传统启发式代理的强大统治力。

**(报告结束)**

#### **引用的著作**

1. SCML 2025 Agent Design\_ Offline RL, Decision Transformers, and Hierarchical RL (1).pdf  
2. Supply Chain Management League (SCML) \- Automated Negotiating Agents Competition (ANAC), 访问时间为 十二月 8, 2025， [https://scml.cs.brown.edu/scml](https://scml.cs.brown.edu/scml)  
3. Automated Negotiation League (ANL) \- SCML Live Competition, 访问时间为 十二月 8, 2025， [https://scml.cs.brown.edu/anl](https://scml.cs.brown.edu/anl)  
4. SCML 2025 代理开发策略研究  
5. HRL-X 研究：强化学习问题解决  
6. SCML 2025 代理开发研究  
7. Decision Transformer \- Hugging Face, 访问时间为 十二月 8, 2025， [https://huggingface.co/docs/transformers/model\_doc/decision\_transformer](https://huggingface.co/docs/transformers/model_doc/decision_transformer)