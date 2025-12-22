# **HRL-XF：面向SCML 2025期货市场的混合残差分层代理架构深度实施报告**

## **1\. 执行摘要：从现货思维到期货本体论的范式重构**

供应链管理联赛（Supply Chain Management League, SCML）2025标准赛道（Standard League）的规则迭代，标志着自动协商代理竞赛（ANAC）从传统的单次、短视博弈向长周期、高维度的企业资源规划（ERP）模拟环境的根本性跨越。随着非易腐库存（Non-Perishable Inventory）机制的确立、期货合约（Future Contracts）的引入以及非线性短缺惩罚（Shortfall Penalties）的强化，参赛代理面临的决策空间呈现出爆炸性增长。

回顾过往设计，原有的HRL-X架构虽然引入了分层强化学习（HRL）的理念，但其本质仍深受现货市场思维的束缚。而在随后的实施方案尝试中（即所谓的“B方案”），我们遗憾地发现其在工程落地过程中出现了严重的逻辑退化：L2层放弃了对时序特征的深度提取，回退到了扁平化的MLP结构；L3层忽略了期货交易中最核心的时间维度（Delivery Time）博弈能力。这种“旧瓶装新酒”的妥协方案无法应对SCML 2025中跨期套利（Inter-temporal Arbitrage）与长鞭效应（Bullwhip Effect）的双重挑战。

本报告旨在呈现**HRL-XF (Hybrid Residual Learner \- Extended Framework, Futures Edition)** 的最终实施形态。本方案基于“方案A”的理论高度，明确拒绝了“方案B”的简化主义倾向，并在L1安全层计算、买卖逻辑对称性以及角色嵌入（Role Embedding）等方面进行了决定性的补全与修正。HRL-XF不仅仅是一个算法模型的堆砌，而是一个具有自我保护能力、长期因果推理能力和微观战术执行能力的有机控制系统。

本报告将分八个章节，详尽阐述从数学原理到张量维度的每一个工程细节，字数约15,000字，旨在为开发团队提供一份无死角、可执行的“超人级”代理构建蓝图。

## ---

**2\. 环境本体论与高维时空POMDP建模**

在着手具体的代码实现之前，必须首先对HRL-XF所处的数学环境进行严格的定义。SCML 2025不再是一个简单的马尔可夫决策过程（MDP），而是一个具有高维状态空间、连续-离散混合动作空间以及延迟奖励特征的**增广时空部分可观察马尔可夫决策过程（Augmented Spatio-Temporal POMDP）**。

### **2.1 状态空间的张量化重构 ($\\mathcal{S}^+$)**

在期货市场中，代理的"状态"不再仅仅是当前的资产负债表快照，更包含了对未来时间窗口内所有承诺（Commitments）与预期（Projections）的完整描述。HRL-XF摒弃了扁平化的状态向量设计，转而采用结构化的混合张量组：

$$\\mathcal{S}\_t^+ \= \\{ \\mathbf{x}\_{static}, \\mathbf{X}\_{temporal}, \\mathbf{x}\_{role} \\}$$

**注**：市场信息已分布于 $\mathbf{x}_{static}$（当前现货价格）和 $\mathbf{X}_{temporal}$（期货溢价、买卖压力）中，无需独立的 $\mathbf{x}_{market}$ 组件。

#### **2.1.1 静态状态向量 ($\\mathbf{x}\_{static}$)**

该向量编码了与时间无关或仅与当前时刻相关的物理快照信息，用于L2与L3的辅助输入。虽然称为“静态”，其数值随仿真步进变化，但不具备规划视界内的时序结构。

* $B\_t$：当前可用现金余额（归一化）。  
* $I\_{now}$：当前仓库中的物理库存（原材料 \+ 成品）。  
* $C\_{total}$：仓库总物理上限（静态常数）。  
* $L\_{state}$：生产线状态（闲置率/效率）。  
* $t\_{norm}$：全局归一化时间 $t/T\_{max}$。  
* $P\_{idx}$：当前公告板上的即时市场指数。

#### **2.1.2 时序状态张量 ($\\mathbf{X}\_{temporal}$) —— 核心感知单元**

这是HRL-XF区别于传统架构的核心。覆盖未来 $H$ 天（含 $\\delta=0$ 当日）的供需/价格轨迹。  
维度：$\\mathbb{R}^{(H+1) \\times 10}$。通道定义如下：

| 通道 | 特征名 | 公式/说明 |
|------|--------|----------|
| 0 | `vol_in` | $Q_{in}[k]$ = 第 $t+k$ 天到货的采购量（已签署合约） |
| 1 | `vol_out` | $Q_{out}[k]$ = 第 $t+k$ 天发货的销售量（已签署合约） |
| 2 | `prod_plan` | $Q_{prod}[k]$ = 预计生产消耗（保守估计） |
| 3 | `inventory_proj` | $I_{proj}[k] = I_{now} + \sum_{j=0}^{k}(Q_{in}[j] - Q_{out}[j] - Q_{prod}[j])$ |
| 4 | `capacity_free` | $C_{free}[k] = C_{total}[k] - I_{proj}[k]$ |
| 5 | `balance_proj` | $B_{proj}[k] = B_t + \sum_{j=0}^{k}(Receivables[j] - Payables[j] - Q_{prod}[j]\cdot cost)$ |
| 6 | `price_diff_in` | 采购侧期货溢价：$P^{buy}_{future}[k] - P^{buy}_{spot}$ |
| 7 | `price_diff_out` | 销售侧期货溢价：$P^{sell}_{future}[k] - P^{sell}_{spot}$ |
| 8 | `buy_pressure` | 买方需求压力（价格加权） |
| 9 | `sell_pressure` | 卖方供给压力（价格加权） |

**通道 6-9 计算说明**：

由于 SCML 标准世界不存在公开订单簿，通道 6-9 基于代理可观测的谈判与合约数据推断：

- `price_diff_in/out`：成交 VWAP 与活跃报价**加权均值**融合（权重 0.6/0.3/0.1），分别基于买入/卖出谈判和 `spot_price_in/out`
- `buy_pressure[k]`：输出市场买方需求强度 = 已签销售量 (Q_out) + 卖出谈判中买家的报价量（按轮次/数量加权，spot_price_out 为基准高价权重更大），除以经济容量裁剪到 `[0,1]`
- `sell_pressure[k]`：输入市场卖方供给强度 = 已签采购量 (Q_in) + 买入谈判中卖家的报价量（按轮次/数量加权，spot_price_in 为基准低价权重更大），除以经济容量裁剪到 `[0,1]`

**设计说明**：
- 当前市场现货价格已包含在 $\mathbf{x}_{static}$ 中（`spot_price_in`/`spot_price_out`），无需在时序张量中重复
- 买卖压力分开表示，使代理可根据角色做出差异化响应

#### **2.1.3 角色嵌入向量 ($\\mathbf{x}\_{role}$) —— 对称性的基石**

**设计决策：L2-L4 全部需要角色嵌入，但语义因层级而异。**

为了解决"B方案"中买卖逻辑不对称的缺陷，HRL-XF在 **L2、L3、L4 所有层级** 强制引入角色嵌入，但根据层级职责采用不同的编码方式：

##### L2 层：Multi-Hot 谈判能力编码

根据 SCML 2025 Standard 规则，代理在供应链中的位置决定了其谈判能力：

* **形式**：Multi-Hot 编码 $[\text{can\_buy}, \text{can\_sell}]$
  * $[0, 1]$：第一层代理，采购来自外生合约 (SELLER)，只能谈判销售
  * $[1, 1]$：中间层代理，买卖都需要谈判
  * $[1, 0]$：最后一层代理，销售来自外生合约 (BUYER)，只能谈判采购
* **作用**：L2 输出 16 维目标向量（4 桶 × 买卖各 2 分量），Multi-Hot 编码让网络知道应关注哪些分量
* **优势**：$[1,1]$ 明确表示"两种能力都有"，比 $[0.5, 0.5]$ 语义更清晰

##### L3/L4 层：One-Hot 谈判角色编码

* **形式**：One-hot 编码 $[1, 0]$ (Buyer) 或 $[0, 1]$ (Seller)，或通过可学习 Embedding 映射为 $\\mathbf{e}\_{role} \\in \\mathbb{R}^{d}$
* **作用**：指示当前这个具体谈判的方向，使网络学习镜像博弈逻辑（买方压价 vs 卖方抬价）

### **2.2 动作空间的混合编码与时间维度 ($\\mathcal{A}^+$)**

HRL-XF必须具备对交易条件的全面掌控力。我们定义了一个混合动作空间，明确包含时间维度的博弈：

$$a\_{offer} \= (q, p, \\delta\_t)$$

* **数量 ($q$)**：连续值，由L3残差头调节。  
* **价格 ($p$)**：连续值，由L3残差头调节。  
* **交货时间 ($\\delta\_t$)**：离散分类分布 (Categorical Distribution)，$\\delta\_t \\in \\{0, 1, \\dots, H\\}$。  
  * **设计依据**：在“方案A”中明确指出，时间选择具有多峰分布特性（例如：要么急用明天交货，要么囤货20天后交货，中间日期可能无意义）。因此，不能将其视为回归问题，必须使用Softmax分类头进行预测 1。

### **2.3 势能奖励函数：解决跨期信用分配**

在期货交易中，动作（签约）与奖励（实物交割与结算）在时间上是解耦的。为了解决稀疏奖励问题，我们引入基于净现值（NPV）的势能奖励函数（Potential-Based Reward Shaping）：

$$\\Phi(s) \= B\_t \+ \\text{Val}(I\_{now}) \+ \\sum\_{c \\in \\text{Pending}} \\gamma^{\\delta\_{t,c}} \\cdot \\text{Val}(c)$$  
修正后的单步奖励 $R\_t' \= R\_{env} \+ (\\Phi(s\_{t+1}) \- \\Phi(s\_t))$ 能够确保代理在签署有利可图的远期合约时立即获得正反馈，从而学会“延迟满足” 3。

## ---

**3\. L1 安全护盾层：基于保守轨迹推演的“宪法”**

L1安全护盾层（Safety Shield Layer）是HRL-XF架构的基石。它的核心职责不是优化，而是**约束**。针对用户提出的关于 $C\_{free}$ 和 $C\_{total}$ 计算不明确的问题，本章提供精确的数学定义与实现逻辑。

### **3.1 $C\_{total}$ 的定义与获取**

$C\_{total}$ 代表工厂的 **总物理仓储容量（Total Warehouse Capacity）**。

* **物理意义**：这是环境赋予代理的硬性物理上限，代表原材料和成品共享的总空间。在标准联赛中，一旦库存总量超过此值，将产生高额的\*\*处置成本（Disposal Cost）\*\*或导致爆仓惩罚。  
* **获取方式**：在代理初始化（init）阶段，通过AWI接口读取静态配置。  
  Python  
  self.c\_total \= self.awi.profile.storage\_capacity

* **数学性质**：在单次模拟中为常数 $C\_{total} \= \\text{Const}$。  
* **设计决策**：优先尝试通过 `awi.profile.storage_capacity` 获取静态库容。如果 negmas 未提供该接口，则采用动态公式：$C_{total}[k] = n\_lines \times (T_{max} - (t+k))$。
* **逻辑解释（买入时）**：买入的是原材料，而原材料需要加工才能出售。超出「原材料交货日到最后一日的总产能」的所有原材料都会因为无法及时加工而被浪费掉。因此动态库容本质上是「可有效利用的原材料容量」。（注：卖出时库容约束相对宽松，因为成品可以直接销售，不受加工时间限制）

### **3.2 $C\_{free}$ 的动态计算逻辑：基于最坏情况的峰值预测**

$C\_{free}(t)$ 代表 **当前时刻 $t$ 的有效剩余安全库容**。在引入期货后，简单的 $C\_{total} \- I\_{current}$ 计算法是极其危险的，因为它忽略了未来即将到货的合约流（Incoming Futures）。为了确保绝对安全，L1层必须采用 **“企鹅原则”（The Penguin Principle）**，即极度保守主义。

#### **3.2.1 悲观假设原则**

1. **入库必达**：假设所有已签署的采购合约都会按时、全量到货（供应商不违约）。  
2. **出库受阻**：假设所有已签署的销售合约可能因生产线故障或客户破产而无法执行，或者为了安全起见，计算库容时不扣减未来的出库量。

#### **3.2.2 库存轨迹投影函数 $L(\\tau)$**

定义当前时刻为 $t$，规划视界为 $H$。我们需要构建未来每一天 $\\tau \\in \[t, t+H\]$ 的累积净流入曲线。

$$L(\\tau) \= I\_{hand}(t) \+ \\sum\_{k=t+1}^{\\tau} Q\_{in}(k) \- Q\_{out}(k)$$  
其中：

* $I\_{hand}(t)$：当前实际在库库存。  
* $Q\_{in}(k)$：将在未来时刻 $k$ 到货的合约总量。  
* **设计决策（适度保守）**：公式中**减去** $Q_{out}(k)$。虽然存在"出库受阻"的风险（供应商违约、客户破产），但采用极度悲观假设（完全不扣减出库）会导致策略过于保守，错失大量交易机会。方案B的逻辑是：即使出现交割失败，代理可在下一日立即调整策略（紧急抛售），同时代理可能会学习减少从声誉不好的市场参与者处的购买。因此适度信任已签署的出库合约是合理的。

#### **3.2.3 $C\_{free}$ 的精确计算公式**

为了防止未来任何一个时间点的爆仓，当前的可买入量受限于未来库存轨迹的最高水位（High Water Mark）。

1. 寻找峰值库存：

   $$I\_{peak} \= \\max\_{\\tau \\in \[t, t+H\]} (L(\\tau))$$

   这个 $I\_{peak}$ 代表了在“只进不出”的最坏情况下，仓库在未来一段时间内可能达到的最大占用量。  
2. 计算全局安全库容：

   $$C\_{free}(t) \=  \\min\_{\\tau \\in \[t, t+H\]}(C\_{total}(\\tau) \- L(\\tau))$$  

#### **3.2.4 针对特定交货日 $\\delta$ 的细粒度掩码**

L3层在选择交货时间 $\\delta\_t$ 时，需要更精细的约束。如果我决定在 $\\delta$ 天后进货，这批货物会占据 $\[\\delta, H\]$ 时间段内的空间。因此，针对特定交货日 $\\delta$ 的最大买入量 $Q\_{safe}\[\\delta\]$ 计算如下：

$$Q_{safe}[\delta] = \min_{k=\delta}^{H} (C_{total}[k] - L(k))$$

这意味着：在 $\delta$ 天进货的数量，不能导致从 $\delta$ 开始的未来**任何一天**发生爆仓。

**向量化实现**：使用**逆向累积最小值（Reverse Cumulative Min）**高效计算，支持动态 $C_{total}[k]$ 向量：
```python
raw_free = C_total - L  # shape (H,)
reversed_free = raw_free[::-1]
reversed_cummin = np.minimum.accumulate(reversed_free)
Q_safe = reversed_cummin[::-1]
Q_safe = np.maximum(Q_safe, 0)  # 非负约束
```

### **3.3 资金维度的延伸：$B\_{limit}$**

除了空间，L1还需计算资金约束，防止“过度承诺（Over-commitment）”。

$$B\_{free}(t) \= B\_{current}(t) \- B\_{reserved} \- \\sum\_{\\tau=t}^{t+H} \\text{Payable}(\\tau)$$

其中 $B\_{reserved}$ 是为了防止破产而预留的“救命钱”（通常为生产线运行成本的倍数）。

### **3.4 L1层的代码实现架构**

L1层应被封装为一个无状态的计算模块，接受AWI状态，输出两个掩码张量：

1. **Quantity Mask**：形状为 $(H+1,)$，对应每一天交货的最大允许买入量，$\delta_t \in \{0, 1, ..., H\}$。其中 $\delta_t = H$ 的值沿用 $\delta_t = H-1$ 的估计。  
2. **Price Mask**：基于 $B\_{free}$ 和数量计算出的最高允许单价。

## ---

**4\. L2 战略规划层：1D-CNN 时平规划器与对称逻辑**

HRL-XF确立了以 **1D-CNN** 为核心的时序特征提取架构，并根据用户要求引入了 **角色嵌入** 和 **完全对称的买卖逻辑**。

### **4.1 1D-CNN 双塔网络架构设计**

L2的核心任务是从 $\\mathbf{X}\_{temporal}$ 张量中识别供需模式（如：原料断供的"深坑"、成品积压的"山峰"）。

* **输入**：  
  * 时序张量 $\\mathbf{X}\_{temporal} \\in \\mathbb{R}^{H \\times 9}$。  
  * 静态向量 $\\mathbf{x}\_{static} \\in \\mathbb{R}^{12}$。  
  * **角色嵌入 $\\mathbf{x}\_{role}$**：虽然L2通常同时规划买卖，但在某些特定模式下（如紧急采购模式），可以通过角色嵌入强化某种倾向。但在标准模式下，L2输出双向目标，角色嵌入可作为全局状态的一部分。
* **时序特征塔 (Temporal Tower)**：  
  * **Conv1D Block 1**：Filters=32, Kernel=3, Stride=1, ReLU。捕捉短期（3天内）波动。  
  * **Conv1D Block 2**：Filters=64, Kernel=7, Stride=1, ReLU。捕捉周度趋势。  
  * **Global Max/Avg Pooling**：提取整个规划视界内的极值特征（最大缺口、最大盈余）。  
  * **输出**：$h\_{temp} \\in \\mathbb{R}^{64}$。  
* **融合策略头 (Policy Head)**：  
  * 拼接 $h\_{temp}$ 与 $\\mathbf{x}\_{static}$ 的Embedding。  
  * 输出层设计为 **分桶参数化（Bucketed Parameterization）**。

### **4.2 对称的8维度（实为16维度）输出与分桶算法**

为了彻底解决“只买不卖”的逻辑缺失，我们将L2的输出空间扩展为**对称的买卖双向规划**。我们将规划视界 $H$ 划分为4个战略时间桶（Buckets）：

1. **Urgent (0-2天)**：解决眼前生存问题。  
2. **Short-term (3-7天)**：维持正常周转。  
3. **Medium-term (8-14天)**：建立缓冲区。  
4. **Long-term (15+天)**：战略储备与套利。

对于每一个桶 $i \\in \\{0, 1, 2, 3\\}$，L2输出四个分量：

1. $Q\_{buy}^i$：该时间段的采购目标量。  
2. $P\_{buy}^i$：该时间段的采购限价系数。  
3. $Q\_{sell}^i$：该时间段的销售目标量（**新增对称逻辑**）。  
4. $P\_{sell}^i$：该时间段的销售底价系数（**新增对称逻辑**）。

总输出维度为 $4 \\times 4 \= 16$ 维。这使得L2不仅是一个采购经理，更是一个能够利用期货溢价进行“囤货待涨”（High $Q\_{sell}^{long}$）或“清仓回笼”（High $Q\_{sell}^{urgent}$）的战略家 1。

分桶算法实现：  
L2输出的 $16$ 维向量 $\mathbf{g}\_t$ 将被广播给所有的L3实例。L3接收完整的16维目标向量，通过神经网络自学习如何利用各时间桶的目标信息。这使得L3能够看到全局规划，作出更协调的谈判决策。

## ---

**5\. L3 残差执行层：混合头决策Transformer与角色嵌入**

L3层是微观博弈的执行者。针对“方案B”的缺陷，我们在此恢复了 **$(q, p, \\delta\_t)$ 三分量输出**，并强制加入了 **角色嵌入**。

### **5.1 输入层设计的增强**

L3 Decision Transformer (DT) 的输入不再仅仅是历史序列，而是包含了显式的条件编码：

1. **State Embedding**：编码对手的当前Offer $(q, p, \delta\_t)$ 和剩余谈判时间。  
2. **Goal Embedding**：将L2输出的完整目标向量 $\mathbf{g} \in \mathbb{R}^{16}$（4桶 × 4分量）投影后，作为前缀Token加入序列。L3通过学习自动关注与当前谈判相关的时间桶目标。  
3. **Role Embedding (关键修正)**：
   * 这是一个可学习的向量 $\\mathbf{e}\_{role} \\in \\mathbb{R}^{d\_{model}}$。  
   * 当L3作为**买家**（Buyer）运行时，输入 $\\mathbf{e}\_{buyer}$。此时网络倾向于压低价格，且关注 $Q\_{buy}$ 目标。  
   * 当L3作为**卖家**（Seller）运行时，输入 $\\mathbf{e}\_{seller}$。此时网络倾向于抬高价格，且关注 $Q\_{sell}$ 目标。  
   * 这种设计实现了**一套权重，双向博弈**，极大提高了参数利用率和泛化能力。

### **5.2 混合输出头拓扑 (Hybrid Head Topology)**

L3的最后一层隐状态 $h\_{last}$ 被送入三个独立的输出头，以生成完整的动作 $a\_{offer}$：

1. **价格残差头 (Price Residual Head)**：  
   * 结构：Dense \-\> Tanh \-\> Scale。  
   * 输出：$\\Delta p \\in \[-1, 1\]$。  
   * 逻辑：最终价格 $P \= P\_{base} \+ \\alpha \\cdot \\Delta p$。如果是买家，$\\Delta p \< 0$ 表示进一步压价；如果是卖家，$\\Delta p \> 0$ 表示尝试溢价。  
2. **数量残差头 (Quantity Residual Head)**：  
   * 结构：Dense \-\> Tanh \-\> Scale。  
   * 输出：$\\Delta q \\in \[-1, 1\]$。  
   * 逻辑：微调成交量以匹配L2的目标或L1的约束。  
3. **时序分类头 (Temporal Classification Head) —— 方案A的核心回归**：  
   * 结构：Dense \-\> Masked Softmax。  
   * 输出：$\\mathbb{P}(\\delta\_t | context) \\in \\mathbb{R}^{H+1}$。  
   * **Masking机制**：此处必须应用L1计算出的 $Q\_{safe}$ 掩码。对于任何 $Q\_{safe}\[\\delta\] \< \\text{Current\\\_Q}$ 的时间点，其Logits被置为 $-\\infty$。这确保了L3绝不会提出一个会导致爆仓的交货时间建议。  
   * **意义**：这一头赋予了L3“讨价还日”的能力（例如：“价格好商量，能不能晚两天发货？”），这是期货谈判的精髓。

## ---

**6\. L4 全局协调层：时空注意力网络**

L4解决了“并发资源耦合”问题。针对“方案B”中L4逻辑的模糊，我们在此明确其 **时空注意力（Spatio-Temporal Attention）** 的实现。

### **6.1 时空冲突图的构建**

L4不仅要看“谁重要”，还要看“谁和谁冲突”。在期货中，冲突是基于时间的。

* 输入：所有活跃L3线程的隐状态集合 $\\{h\_1, \\dots, h\_N\\}$，以及它们当前的意向交货时间 $\\{\\delta\_1, \\dots, \\delta\_N\\}$。  
* **角色嵌入**：同样，L4输入中包含每个线程的角色（买/卖），以便分别统计资金流入（卖）和流出（买）。

### **6.2 注意力机制的数学表达**

L4采用Transformer Encoder结构，引入 **时间偏置掩码 (Temporal Bias Mask)**。

$$\\text{Attention}(Q, K, V) \= \\text{Softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}} \+ \\mathbf{M}\_{time}\\right)V$$

* $\\mathbf{M}\_{time}\[i, j\]$：如果线程 $i$ 和线程 $j$ 的意向交货时间 $\\delta\_i \\approx \\delta\_j$，则 $M$ 值较大（冲突概率高，需要相互关注）；如果时间相距甚远，则 $M \\to \-\\infty$（无冲突，解耦）。

### **6.3 资源压力的双向传导**

L4输出的权重 $\\alpha\_k$ 直接作用于L3的Gate：

* **资金危机模式**：当 $B\_{proj}$ 显示未来某日资金断裂，L4将注意力集中在该日之前的所有 **Seller** 线程上，迫使其降价回款（Fire Sale）。  
* **库存告急模式**：当 $I\_{proj}$ 显示未来某日缺货，L4激活该日对应的 **Buyer** 线程，允许其溢价采购。

## ---

**7\. 数据工程与取证流水线 (Forensics Pipeline)**

为了训练上述复杂的神经网络，我们不能仅依赖在线强化学习（效率太低）。我们需要从历年冠军（如PenguinAgent）的日志中通过 **“事后诸葛亮”（Hindsight/Forensics）** 机制提取训练数据。

### **7.1 L2目标的逆向重构 (Target Reconstruction)**

PenguinAgent是基于规则的，其日志中没有显式的 $g\_t$。我们需要从其行为中推断其意图。

**买入目标重构 ($Q\_{buy}^{target}$)**：

* 假设：专家在一天结束时实际买入的总量，就是它在一天开始时“想要”买入的量。  
* 公式：$Q\_{buy}^{bucket\\\_i} \= \\sum\_{c \\in \\text{Deals}} q\_c \\cdot \\mathbb{I}(\\text{time}(c) \\in \\text{Bucket}\_i)$。

**卖出目标重构 ($Q\_{sell}^{target}$) —— 对称性补全**：

* 逻辑：PenguinAgent遵循“零库存等待”原则。  
* 公式：$Q\_{sell}^{target} \= I\_{finished} \+ I\_{production\\\_today}$。这意味着它的目标总是清空当前成品库存。  
* 这种重构为L2提供了明确的“清仓”信号标签。

### **7.2 L3残差的提取**

* 基准动作 ($a\_{base}$)：由L1逻辑计算出的保守报价。  
* 专家动作 ($a\_{expert}$)：日志中PenguinAgent的实际报价。  
* 残差标签 ($\\Delta a\_{label}$)：$a\_{expert} \- a\_{base}$。  
* **角色嵌入生成**：根据日志中Agent是Buyer还是Seller，生成对应的One-hot向量作为L3输入的标签。

## ---

**8\. 训练系统与课程学习方案**

我们设计了 HRLXFTrainer，采用四阶段课程学习（Curriculum Learning）策略，从模仿到超越。

### **8.1 训练器架构设计**

Python

class HRLXFTrainer:  
    def train\_step(self, macro\_batch, micro\_batch):  
        \# 1\. L2 Update (Daily Scale)  
        \# 输入包含 Role Embedding 和 X\_temporal  
        \# Loss: PPO Loss (Phase 2\) or MLE (Phase 0\)  
        with tf.GradientTape() as tape\_l2:  
            g\_pred \= self.l2\_model(macro\_batch\['state'\], macro\_batch\['role'\])  
            loss\_l2 \= self.ppo\_loss(g\_pred, macro\_batch\['actions'\])  
          
        \# 2\. L3 & L4 Update (Round Scale)  
        \# 输入包含 Role Embedding  
        \# Loss: Hybrid (MSE for p/q \+ CrossEntropy for time)  
        with tf.GradientTape() as tape\_l3:  
            weights \= self.l4\_model(micro\_batch\['hidden\_states'\], micro\_batch\['roles'\])  
            \# L3 forward conditioned on L4 weights, L2 goals, and Role  
            pred\_res, pred\_time \= self.l3\_model(micro\_batch\['context'\], weights, micro\_batch\['role'\])  
              
            loss\_res \= mse(pred\_res, micro\_batch\['target\_residuals'\])  
            loss\_time \= cross\_entropy(pred\_time, micro\_batch\['target\_time'\])  
            total\_loss \= loss\_res \+ loss\_time  
              
        \# Apply Gradients  
        self.opt\_l2.apply(...)  
        self.opt\_l3.apply(...)

### **8.2 详细训练阶段**

#### **Phase 0: 行为克隆 (Cold Start)**

* **目标**：获得一个不崩盘的基准策略。  
* **数据**：PenguinAgent 2024日志。  
* **方法**：监督学习。L2学习预测Penguin的日交易分布（买/卖双向）；L3学习预测Penguin的出价残差。  
* **关键点**：必须开启L1安全护盾，防止神经网络初期的随机输出导致环境重置。

#### **Phase 1: 离线 ROL (Reward-on-the-Line)**

* **目标**：超越专家的平均水平。  
* **方法**：仅筛选PenguinAgent那些利润高于平均值的轨迹进行训练（Advantage Filtering）。  
* **算法**：AWR (Advantage-Weighted Regression)。L2学会只在“赚钱的日子”模仿专家的库存规划。

#### **Phase 2: 分层联合在线微调 (Exploration)**

* **环境**：SCML 2025 模拟器。  
* **奖励工程**：启用势能函数 $\\Phi(s)$。  
  * 当L2指令买入期货时，现金减少，但 $\\Phi$ 增加，总奖励平衡。这解决了“买入即亏损”的短视问题。  
* **探索**：L2的PPO探索噪声 $\\sigma$ 逐渐衰减。L3在L1划定的 $C\_{free}$ 范围内尝试不同的 $\\delta\_t$（例如：尝试比专家更早或更晚交货以换取价格优势）。

#### **Phase 3: 对抗自博弈 (Robustness)**

* **目标**：防止策略退化，逼近纳什均衡。  
* **对手池**：放入Phase 2训练出的HRL-XF快照。  
* **机制**：新一代代理必须战胜上一代代理才能获得奖励。这将迫使L2学会应对“恶意挤兑”和“虚假报价”。

## ---

**9\. 结论**

HRL-XF架构是对SCML 2025复杂期货环境的系统性回应。通过废弃“方案B”中的现货逻辑回归，我们重新确立了以 **1D-CNN时序感知**、**买卖对称角色嵌入** 以及 **混合动作头（含时间维度）** 为核心的技术路线。

本报告提供的实施方案中：

1. **L1层** 提供了基于“企鹅原则”的严密 $C\_{free}$ 计算，构筑了物理安全的底座。  
2. **L2层** 通过16维分桶目标和双向规划，实现了跨期套利的战略意图。  
3. **L3层** 通过 $q, p, \\delta\_t$ 三分量输出和角色嵌入，具备了完整的微观谈判能力。  
4. **L4层** 通过时空注意力，解决了并发合约的资源冲突。

这套架构不仅在理论上自洽，在工程上也通过详细的TensorFlow/PyTorch定义变得切实可行。立即启动Phase 0的数据取证工作，是通往SCML 2025冠军的第一步。

#### **引用的著作**

1. HRL-X 期货市场适应性重构  
2. HRL-XF 代理实施与训练方案-1  
3. L1-L4 层设计与离线强化学习  
4. HRL-X 研究：强化学习问题解决  
5. HRL-XF 代理实施与训练方案
