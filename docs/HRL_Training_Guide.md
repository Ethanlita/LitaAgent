# LitaAgent-HRL (HRL-XF) 训练与架构详解

> 目标：利用 SCML 2025 Standard 赛道数据，训练基于 HRL-XF 架构的分层强化学习代理 LitaAgent-HRL（模式 B：L1 安全基准，L2 日级目标，L3 残差微调，L4 并发协调）。

> ⚠️ **重要限制**：当前 HRL-XF 数据管道仅支持 **SCML Standard** 协议。OneShot 协议的报价事件追踪尚未实现，无法生成 L3 训练数据。

## 目录
1. 核心设计哲学  
2. 架构概览与数学定义  
3. 动作空间与状态空间  
4. 数据流水线：从锦标赛到训练样本  
5. 训练流程：预训练与在线微调  
6. 如何运行数据采集 runner（HRL-XF）  
7. 如何训练模型并接入代理（HRL-XF）  
8. 常见问题与数据需求  
9. 技术路线：后续探索方向

---
*** 所有AGENTS需注意：1.1、1.2、1.4节是用户确认的，如果有任何文档或实现和这里有差异，以这里为准。不可以做出违反这里的设计逻辑的代码修改。***
## 1. 核心设计哲学

### 1.1 为什么选择残差学习？

HRL-XF 架构的核心理念是：**用简单的 L1 baseline 基础，用神经网络 L2-L4 学习"专家相对于 baseline 的改进"**。

```
┌─────────────────────────────────────────────────────────────────┐
│                      核心公式                                   │
│                                                                 │
│   a_final = Clip(L1_baseline(s) + L3_residual(s), L1_mask)     │
│                                                                 │
│   其中:                                                         │
│   - L1_baseline: 简单启发式规则，无需学习，保证安全             │
│   - L3_residual: 神经网络学习的"改进量"                        │
│   - L1_mask: 安全裁剪，防止越界                                │
└─────────────────────────────────────────────────────────────────┘
```

**设计优势**：
1. **冷启动安全**：L3 网络随机初始化时，输出接近 0，代理表现 ≈ L1 baseline
2. **学习目标简化**：L3 只需学习"如何比 baseline 更好"，在离线学习阶段重点是”复刻专家代理的行为“，而非"如何从零开始谈判"
3. **训练/推理一致**：L1 baseline 在训练和推理时使用相同的计算方法

### 1.2 数据收集策略

**关键问题**：我们应该用什么 Agent 收集训练数据？

**答案**：使用 **Expert Agent**（如 PenguinAgent、DecentralizingAgent），而非未训练的 HRL-XF。

```
┌─────────────────────────────────────────────────────────────────┐
│                    数据收集流程                                 │
│                                                                 │
│   Step 1: 运行 Tournament                                       │
│           PenguinAgent vs DecentralizingAgent vs ...           │
│                         ↓                                       │
│   Step 2: 从 Tracker 日志读取 Expert 的动作                     │
│           expert_action = (quantity, price, time)              │
│                         ↓                                       │
│   Step 3: 离线计算 L1 baseline（使用日志中的状态）              │
│           baseline = compute_l1_baseline_offline(state_dict)   │
│                         ↓                                       │
│   Step 4: 计算 residual label                                   │
│           residual = expert_action - baseline                  │
│                         ↓                                       │
│   Step 5: 训练 L3 模型                                          │
│           L3.learn(history, goal, state → residual)            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 L1 Baseline 的生命周期

**L1 Baseline 是无状态的**：只依赖当前时刻的 `Q_safe`, `B_free`, `inventory`, `market_price`。

| 阶段 | 组件 | 方法 | 时机 |
|------|------|------|------|
| **在线推理** | L1SafetyLayer | `compute()` → `_compute_baseline()` | 每次谈判开始 |
| **离线训练** | data_pipeline | `compute_l1_baseline_offline()` | 从日志重建状态后计算 |

**Baseline 计算逻辑**（两者相同）：
```python
# 买入
q_base = min(Q_safe[δ]/2, B_free/price)
p_base = market_price * 0.95

# 卖出
q_base = inventory / 2
p_base = market_price * 1.05
```

**一致性保证**：
- `compute_l1_baseline_offline()` 复现了 `L1SafetyLayer._compute_baseline()` 的逻辑
- 训练时的 baseline 与推理时的 baseline 使用相同的计算方法
- 这确保了 L3 网络在训练和推理时看到一致的输入分布

### 1.4 训练目标：模仿 Expert，超越 Baseline

```
┌─────────────────────────────────────────────────────────────────┐
│                    离线模仿学习阶段                              │
│                                                                 │
│   Expert Agent (如 PenguinAgent)                                │
│         ↓ 参加比赛，生成数据                                     │
│   expert_action = (q_expert, p_expert, t_expert)                │
│         ↓                                                       │
│   baseline = compute_l1_baseline_offline(state)  ← 离线计算     │
│         ↓                                                       │
│   residual_label = expert_action - baseline                     │
│         ↓                                                       │
│   L3 学习: residual = f(history, goal, state)                   │
│                                                                 │
│   目标: 让 L3 输出的 residual 使得                              │
│         baseline + residual ≈ expert_action                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    在线强化学习阶段                              │
│                                                                 │
│   L3 输出 residual → final_action = baseline + residual        │
│         ↓ 与环境交互                                            │
│   reward → 更新 L3 策略                                         │
│         ↓                                                       │
│   Agent 探索更优策略（可能超越 Expert）                          │
└─────────────────────────────────────────────────────────────────┘



┌─────────────────────────────────────────────────────────────────┐
│                    阶段 1: 离线预训练                           │
│                                                                 │
│   L2: BC (使用 Hindsight 反推的目标)                            │
│   L3: BC (使用 residual = expert - baseline)                   │
│   L4: 不训练 (无数据)                                           │
│                                                                 │
│   目标: 让 L2/L3 获得"像 Expert 一样"的初始能力                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    阶段 2: 在线微调 (可选分步)                  │
│                                                                 │
│   步骤 A: 冻结 L3，用 PPO 微调 L2                               │
│           → L2 学会设定更好的目标                               │
│                                                                 │
│   步骤 B: 解冻 L3，用 MAPPO 联合微调 L2/L3/L4                   │
│           → 整个系统协同优化                                    │
│                                                                 │
│   目标: 超越 Expert，适应 2025 新规则                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 架构概览与数学定义
LitaAgent-HRL 由四层组成，时间尺度与职责不同：
- **L1 安全/基准层**：提炼自 Penguin 的微观谈判逻辑，输出安全掩码与基准报价，防止超量、超价、超资金。  
- **L2 战略管理层**：每日决策（Day 级），生成目标向量  
  \( g_t = [Q^{buy}_t, P^{buy}_t, Q^{sell}_t, P^{sell}_t] \)。  
- **L3 残差执行层**：轮级决策（Round 级），在 L1 基准上输出残差  
  \( \Delta a = [\Delta q, \Delta p] \)，合成最终报价  
  \( a_{final} = \text{clip}(a_{base} + \Delta a, \text{mask}) \)。  
- **L4 并发协调层（预留）**：注意力网络，在多线程并发谈判中给每个线程分配权重 \( \alpha_k \)，调节其激进程度。

时间顺序：  
1) 每日开始：L2 读取宏观状态，输出 \( g_t \)；L1 不改宏观，仅提供安全边界。  
2) 每场谈判：L3 在 L1 基准上微调；L4 可调节各线程的残差缩放。  
3) 所有动作最终经 L1 裁剪，保证安全。

## 3. 动作空间与状态空间
- **L2 动作（连续）**：\( g_t \in \mathbb{R}^{16} \)，4 桶 × \([Q^{buy}, P^{buy}, Q^{sell}, P^{sell}]\)。  
- **L3 动作（连续 + 接受/拒绝）**：报价 \((q, t_{deliv}, p)\) 或 Accept/End；实现中以残差方式输出数量/价格，交付时间由 L3 time-head 与 L1 time_mask 联合决定。  
- **状态（宏观）**：库存 \(I\)、资金 \(B\)、市场均价 \(P_{mkt}\)、生产线 \(n_{lines}\)、已承诺合约。  
- **状态（微观序列）**：谈判历史序列 \((q, p, \delta_t)\)、剩余轮次、L2 目标 \(g_t\)。

## 4. 数据流水线：从锦标赛到训练样本

### 4.1 Tracker 记录字段（HRL-XF 训练所需）

为了支持离线计算 L1 baseline，`tracker_mixin.py` 在每日状态中记录以下字段：

| 字段 | 用途 | 示例值 |
|------|------|--------|
| `balance` | 当前余额，用于计算 B_free | 10000.0 |
| `initial_balance` | 初始余额，用于归一化 | 10000.0 |
| `inventory_input` | 原材料库存 | 5 |
| `inventory_output` | 成品库存 | 3 |
| `spot_price_in` | 原材料市场价 | 10.0 |
| `spot_price_out` | 成品市场价 | 20.0 |
| `trading_prices` | 完整价格数组 | [10.0, 15.0, 20.0] |
| `n_lines` | 生产线数量 | 5 |
| `n_steps` | 总仿真步数 | 100 |
| `current_step` | 当前步数 | 10 |
| `commitments` | 合同承诺（Q_in, Q_out, Payables, Receivables） | {...} |
| `my_input_product` | 输入产品索引 | 0 |
| `my_output_product` | 输出产品索引 | 1 |

**代理身份字段（用于精确过滤）**：  
`agent_initialized` 事件会记录 `agent_type_raw/base` 及其 `*_full` 版本（模块路径），
data_pipeline 会优先使用这些字段做匹配，避免 `agent_type` 为分组名导致的混合/漏采。

**离线 baseline 计算依赖**：
- `balance` + `commitments.Payables` → `B_free`
- `n_lines` + `n_steps` + `commitments.Q_in/Q_out` → `Q_safe`
- `spot_price_in/out` → `market_price`
- `inventory_input/output` → 卖出时的可用量

### 4.2 采集数据
运行 `runners/hrl_data_runner.py`：  
- 参赛：全部 LitaAgent（tracked 版，除 HRL）、2025 标准前 5、2024 标准前 5、RandomStdAgent/SyncRandomStdAgent（全部动态 Tracked）。  
- 开启 `log_negotiations=True`、`log_ufuns=True`，输出至 `tournament_history/hrl_data_<timestamp>_std`（可用 `--output-dir` 覆盖）。  
- 若安装 `scml_analyzer`，自动记录 Tracker。

### 4.3 解析日志
使用 `litaagent_std/hrl_xf/data_pipeline.py`：  
```python
from litaagent_std.hrl_xf.data_pipeline import load_tournament_data, save_samples

# 数据格式：Tracker JSON
# Tracker JSON 位置：
#   tournament_dir/tracker_logs/agent_*.json  (hrl_data_runner 默认输出)
#   tournament_dir/agent_*.json               (也支持)

# 使用 PenguinAgent 作为专家示范
macro_ds, micro_ds = load_tournament_data(
    "./tournament_history/hrl_data_<timestamp>_std",
    agent_name="Pe",  # 'Pe' 匹配 PenguinAgent，'Li' 匹配 LitaAgent
    num_workers=4,    # 并行解析（默认自动）
)
# 保存样本，horizon 应与训练配置一致（默认 40）
save_samples(macro_ds, micro_ds, "./data/hrlxf_samples", horizon=40)

# 预期输出：
# [INFO] Found 752 world directories
# [INFO] Extracted 114552 macro samples, 64752 micro samples
```
- Macro 样本（L2）：`MacroSample(day, state_static, state_temporal, x_role, goal)`，goal 为反推的日级目标。
- Micro 样本（L3）：`MicroSample(negotiation_id, history, baseline, residual, goal, time_label, time_mask)`，time_mask 为 L1 安全约束。

> ✅ **Tracker JSON 格式**：`data_pipeline.py` 会自动递归搜索 `tracker_logs/` 子目录中的 Tracker JSON 文件。

## 5. 训练流程：预训练与在线微调
### 5.1 预训练（离线）
- **L3 残差监督**：
   - 标签：`residual = action_expert - baseline`（baseline 由 L1 生成）。
   - 损失：MSE + 安全正则（越界惩罚），可加 ROL/ensemble 方差惩罚防 OOD。
- **L2 目标生成 BC/CQL**：
   - 标签：`g_t`（从成交反推）。
   - 奖励塑形（在线用）：势能函数 \( \Phi = I \times P_{mkt} \)，风险惩罚（短缺预测）。

占位示例（`litaagent_std/hrl_xf/training.py`）：  
```python
from litaagent_std.hrl_xf import training
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

cfg = training.TrainConfig(l2_epochs=10, l3_epochs=10, output_dir="./out_hrlxf")
l2_model = HorizonManagerPPO()
l3_model = TemporalDecisionTransformer()
trainer = training.HRLXFTrainer(l2_model, l3_model, None, cfg)

# 离线 BC训练
trainer.train_phase0_bc(macro_ds, micro_ds)
trainer.save_all("bc_example")
```
> “占位模型”意为简单可运行的线性/MLP 示例；实际项目应替换为更强的 Transformer/PPO 等模型。

### 5.2 在线微调（分层/多智能体 PPO 示意）
目标：在真实环境（2025 规则：非易腐、期货、短缺惩罚，多线程并发）下，联合更新 L2/L3/L4，使收益更高、违约率更低、库存/资金曲线更稳。

**核心思路：CTDE（集中式训练、分布式执行）**
- L2（Day 粒度）单独 PPO；L3/L4（Round 粒度）用 MAPPO，critic 可观察全局。
- 动作安全由 L1 裁剪，始终开启。

**步骤流程**
1) 环境与初始化  
   - 载入离线预训练好的 L2/L3 权重，L4 初始均匀或简单启发式。  
   - 对手池：Penguin/AS0/随机/历史自我版本，提升鲁棒性。
2) 采样与拆分轨迹  
   - 一次迭代跑 3-5 场标准赛（n_steps≈90-100），收集轨迹。  
   - 按日拆分 macro 轨迹给 L2；按谈判轮拆分 micro 轨迹给 L3/L4。
3) 奖励设计  
   - L2：\( R_{L2} = \Delta cash + \gamma(\Phi_{t+1}-\Phi_t) - \text{risk\_penalty} + \epsilon \cdot \text{liquidity} \)，其中 \(\Phi = I \times P_{mkt}\)；risk_penalty 基于短缺预测；liquidity 为成交微奖。  
   - L3：目标对齐（罚 \(|q_{exec}-Q_{target}|\)、\(|p_{deal}-P_{target}|\)），成交优势（买入价低于市场/卖出价高于市场给正奖励），流动性微奖。  
   - L4：共享全局回报；注意力权重 \(\alpha\) 调节 L3 激进度，高 \(\alpha\) 成功成交则正反馈，盲目高 \(\alpha\) 导致亏损/违约则负反馈。
4) 更新规则  
   - L2：PPO clip 更新，输入日级状态 \(s_{macro}\)，动作 \(g_t\)。  
   - L3/L4：MAPPO 更新，输入轮级状态（历史序列、L2 目标、L4 权重），动作（残差/权重），critic 看到全局。  
   - 学习率：L3 小于 L2（保护已学微观技巧），L4 可略大以快速适应并发调度。
5) 断点续训  
   - 定期保存权重 + 优化器状态 + 种子；记录已完成场次/epoch。  
   - 恢复时加载最新快照继续采样-更新。  
   - 监控：平均利润、违约率、库存爆仓次数、对手池胜率；指标变差则回退。  
6) 终止/早停  
   - 5-10 轮锦标赛后若收益趋稳且违约低，可停止；若过拟合则增加对手多样性或回退权重。

### 5.3 L4（并发协调层）训练要点
- **状态设计**：对每个活跃谈判线程 \(k\) 构造特征向量（如：报价与目标差值、剩余轮次、对手让步速度、线程已成交量占比、历史成交价格均值），并拼成集合；全局上下文包含 L2 目标、库存/资金、尚未完成的总目标量。  
- **动作**：输出注意力权重 \(\alpha_k\) 或门控值，通常经 softmax 归一化。权重可以直接调制 L3 的残差缩放或接受阈值。  
- **奖励**：共享全局回报（与 L3 一致），并可加入线程级形状奖励：高权重线程若成交盈利则加分，盲目高权重导致亏损/违约则扣分。  
- **训练方式**：  
  - 主路径：与 L3 联合的 MAPPO，critic 观察全局；L4 行为梯度来自全局奖励。  
  - 可选监督预训练：从日志中选出“最佳线程”作为软标签（如买入时最低价成交的线程/卖出时最高价的线程），对 \(\alpha\) 施加交叉熵或排名损失，使其初始能偏向高收益线程。  
- **超参建议**：L4 学习率可略大于 L3，鼓励快速调度；动作熵系数可小一些，防止过度随机分配。  
- **安全性**：最终成交仍受 L1/L2/L3 约束，L4 只调度注意力，不直接突破安全裁剪。

#### 更详细的 L4 训练流程示例（与 L3 联合 MAPPO）
1) **数据准备**  
   - 采样一批完整对局，记录每一步的：全局状态、每个 negotiation_id 的线程特征、L2 目标、L3 动作、L4 当前权重 \(\alpha\)、全局奖励。  
   - 线程特征可包含：\(\Delta p = p_{offer} - P_{target}\)、\(\Delta q = q_{offer} - Q_{target}\)、剩余轮次、对手最近让步幅度、线程已成交量占比、线程盈利/亏损估计。  
2) **模型结构**  
   - 输入：线程嵌入序列 + 全局上下文，经过多头自注意力；输出每个线程的 \(\alpha_k\)（softmax）。  
   - Critic：集中式，看到所有线程特征 + 全局上下文，输出全局价值 \(V(s)\)。可与 L3 共享或单独一个 critic。  
3) **损失函数**（PPO/MAPPO 风格）  
   - Actor 损失：\(\mathcal{L}_{\text{actor}} = -\mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)]\)，其中 \(r_t = \frac{\pi(\alpha|s)}{\pi_{\text{old}}(\alpha|s)}\)，优势 \(A_t\) 来自全局回报（与 L3 一致）。  
   - Critic 损失：\(\mathcal{L}_{\text{critic}} = \| V(s) - R_t \|^2\)，\(R_t\) 为折扣回报。  
   - 形状奖励/惩罚：对高权重且盈利的线程给正偏置，对高权重且亏损/违约的线程给负偏置，加入 \(A_t\) 或额外正则项。  
   - 熵正则：\(-\beta \sum_k \alpha_k \log \alpha_k\)，\(\beta\) 可小以减少随机性。  
4) **监督预训练（可选）**  
   - 从日志中选择“好线程”标签：买入方选最低价成交线程，卖出方选最高价成交线程；若无成交，以最佳出价为软标签。  
   - 交叉熵损失：\(\mathcal{L}_{\text{sup}} = - \sum_k y_k \log \alpha_k\)，其中 \(y_k\) 为 one-hot/soft 标签。  
   - 预训练若干 epoch 后，再切换到 RL 更新。  
5) **更新与联合训练**  
   - 与 L3 共享采样的 micro 轨迹，同步更新（MAPPO）；L3 负责残差报价，L4 负责注意力权重。  
   - 每个 update 循环：计算优势 -> 更新 Actor/Critic -> 可选监督 loss 融合：\(\mathcal{L} = \mathcal{L}_{\text{actor}} + c_v \mathcal{L}_{\text{critic}} + c_e \mathcal{L}_{\text{entropy}} + c_s \mathcal{L}_{\text{sup}}\)。  
6) **评价与早停**  
   - 监控：全局收益、违约率、线程分配的多样性（避免一条线程总是独占）、关键线程成交率。  
   - 若发现 L4 过度偏向单线程导致其他线程空转，可提高熵系数或加入多样性正则（如最大化 \(\alpha\) 的均匀度或对低权重线程给予小额探索奖励）。

## 6. 如何运行数据采集 runner（HRL-XF）
推荐使用 `hrl_data_runner` 并开启 Tracker 记录：
```bash
cd /Users/lita/Documents/GitHub/LitaAgent
nohup python3 runners/hrl_data_runner.py --configs 3 --runs 1 \
   > tournament_run.log 2>&1 &
```
输出目录示例：`tournament_history/hrl_data_<timestamp>_std/`（默认），包含 `tracker_logs/agent_*.json` 等 Tracker JSON 文件。

> ✅ 数据管道使用 Tracker JSON 格式，包含完整状态快照，可直接使用 `load_tournament_data()` 加载。

## 7. 如何训练模型并接入代理（HRL-XF）
1) 比赛结束后，解析日志生成 `macro_ds`/`micro_ds`。  
2) 使用 `litaagent_std/hrl_xf/training.py` 训练 L2/L3/L4，并保存权重。  
3) 在 `litaagent_std/hrl_xf/agent.py` 中加载权重：L2/L3/L4 初始化传入 `mode="neural"`。  
4) 在 runner 中注册 `LitaAgentHRL` 进行仿真验证。

训练示例：
```python
from pathlib import Path
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "tournament_history/hrl_data_<timestamp>_std"

# 使用 PenguinAgent 作为专家示范
macro_ds, micro_ds = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="Pe",
    num_workers=4,  # 并行解析（默认自动）
)
# [INFO] Found 752 world directories
# [INFO] Extracted 114552 macro samples, 64752 micro samples

# 保存处理后的样本
# 注意：horizon 参数应与训练配置一致（默认 40）
data_pipeline.save_samples(macro_ds, micro_ds, f"{data_dir}/processed_samples", horizon=40)

# 开始训练
# 注意：TrainConfig.horizon 应与 save_samples 的 horizon 一致
cfg = training.TrainConfig(output_dir="./checkpoints_hrlxf", horizon=40)
l2_model = HorizonManagerPPO()
l3_model = TemporalDecisionTransformer()
trainer = training.HRLXFTrainer(l2_model, l3_model, None, cfg)

# 离线行为克隆
trainer.train_phase0_bc(macro_ds, micro_ds)
trainer.save_all("bc")
```

> ✅ 数据管道已完成 Tracker JSON 格式适配，可直接运行上述训练脚本。

### 7.1 离线训练策略建议
1) **先 BC 后 AWR**  
   - L2：行为克隆（macro 样本）  
   - L3：行为克隆（micro 残差）  
   - 若有奖励：继续 L3 AWR（同一模型权重上微调）  
2) **是否需要多次训练**  
   - 一次 `train_phase0_bc()` 仅完成一次 BC；  
   - 若新增数据或想继续优化，可加载 `*.ckpt.pt` 后继续训练。  
3) **权重保存路径**  
   - `output_dir/l2_bc_epoch{N}.pt`、`output_dir/l3_bc_epoch{N}.pt`、`output_dir/l3_awr_epoch{N}.pt`  
   - 断点文件：`output_dir/l2_bc_epoch{N}.ckpt.pt` 等  

## 8. 常见问题与数据需求
- **需要多少数据？** 建议至少 30-50 场完整标准赛（每场 ~90-100 天），总谈判记录量达到数千到上万条。数据越多越有利于离线预训练；在线微调时可在 10-20 场后开始观察收敛。  
- **需要训练多少轮？**  
  - 离线预训练：L2/L3 各训练 10-50 epoch（视模型复杂度和数据量），以验证集指标稳定为准。  
  - 在线微调：跑 5-10 轮锦标赛检查收益/违约率，若仍在上升可继续，若过拟合则提前停止。  
- **如何中断续训？**  
  - 训练阶段会保存 `*.ckpt.pt`（模型 + 优化器 + epoch），通过 `TrainConfig.l2_resume_path / l3_bc_resume_path / l3_awr_resume_path` 恢复。  
  - 建议把 `save_every` 调小（如 1-5），以减少中断带来的训练损失。  
- **为什么分层？** 解决长时间跨度的信用分配、并发耦合、安全探索问题。L1 保证安全，L2 管跨日规划，L3 做微调，L4 管并发资源分配。  
- **如果没有 negotiations.csv？** 必须重跑锦标赛并开启 `log_negotiations=True`，否则无法训练微观模型。
- **L4 线程上下文为什么在训练样本中没有？**  
  L4 协调层用于在线推理时分配并发资源（多条谈判线程的注意力权重），其参数在**在线微调阶段**通过自博弈学习。BC（行为克隆）阶段只训练 L2/L3，因为专家日志（如 PenguinAgent）不使用 HRL 架构，无法提供 L4 上下文数据。
- **离线状态重建与在线状态构建如何对齐？**  
  `data_pipeline.py` 的 `extract_macro_state()` 与 `state_builder.py` 的 `StateBuilder.build()` 使用**相同的状态结构**：
  
  | 状态组件 | 在线 (`state_builder`) | 离线 (`data_pipeline`) |
  |---------|----------------------|----------------------|
  | x_static | (12,) 静态特征 | (12,) 静态特征 |
  | X_temporal | (H+1, 10) 时序通道 | (H+1, 10) 时序通道 |
  | x_role (L2) | (2,) Multi-Hot 谈判能力 | (2,) Multi-Hot 谈判能力 |
  | x_role (L3) | (2,) One-hot 谈判角色 | (2,) One-hot 谈判角色 |
  
  **x_role 编码说明**：
  
  | 层级 | 编码方式 | 含义 |
  |------|---------|------|
  | L2 | Multi-Hot `[can_buy, can_sell]` | 代理的谈判能力：第一层`[0,1]`、中间层`[1,1]`、最后层`[1,0]` |
  | L3/L4 | One-Hot `[is_buyer, is_seller]` | 当前谈判角色：买方`[1,0]`、卖方`[0,1]` |
  
  **归一化参数**对齐：
  - `initial_balance = 10000.0`
  - `max_price = 50.0`
  - `max_inventory = n_lines × n_steps`（经济容量）
  
  **通道 5-9（balance_proj, price_diff_in, price_diff_out, buy_pressure, sell_pressure）**：
  - 在线模式：实时计算采购/销售两个 price_diff（成交 VWAP + 活跃报价加权均值 + spot）以及买/卖压力
  - 离线模式：优先使用 tracker JSON 的 `offers_snapshot`；若快照为空，则从谈判日志按**轮次/数量加权**聚合活跃报价回填；仍缺失时回退 commitments/0
  - **建议**：使用 tracked runner 输出 JSON 格式，才能保留 price_diff_in/out 与压力通道
  
- **L1 baseline 如何在离线数据中计算？**  
  `data_pipeline.py` 提供 `compute_l1_baseline_offline()` 函数，复现 `L1SafetyLayer._compute_baseline()` 的计算逻辑：
  
  ```python
  # 买入 baseline
  q_base = min(Q_safe[δ]/2, B_free/price)
  p_base = market_price * 0.95
  
  # 卖出 baseline
  q_base = inventory / 2
  p_base = market_price * 1.05
  ```
  
  **优先级**：
  1. 如果日志包含 `l1_baseline` 字段（tracker 记录），直接使用
  2. 如果提供 `state_dict`，调用 `compute_l1_baseline_offline()` 计算
  3. 回退：使用专家第一轮出价（会产生警告）

- **time_mask 无可交易日期时怎么办？**  
  - 若 `time_mask` 全为 `-inf`（极端情况下可能发生），离线管道会**强制保留** `t_base` 为可行，以避免训练时数值异常。  
  - 推理时可将该情况视为“无法交易/终止谈判”，由 L1/L3 共同约束。  

- **为什么库存投影不区分原料和成品？**  
  这是**有意的设计决策**，而非遗漏。理由如下：
  1. **与 `state_builder.py` 一致**：在线推理时的状态构建器使用 `total_inventory = raw + finished` 作为统一库存指标，离线数据管道保持相同语义。
  2. **信息压缩**：L2 战略层关注的是"是否需要补货"和"是否需要清仓"，这两个决策可以通过**总库存水位**和**承诺向量**（Q_in/Q_out）联合推断，无需区分原料/成品的精确库存。
  3. **生产假设**：SCML 2025 Standard 赛道的生产是确定性的（输入原料 → 输出成品），L1 安全层已经处理了生产能力约束。L2 只需知道"库存紧张还是充裕"即可制定目标。
  4. **复杂度权衡**：区分原料/成品需要额外的特征通道和更复杂的投影逻辑，但收益有限。如果未来发现这影响了模型性能，可以扩展为 `(H+1, 12)` 的双轨投影。

  **注意**：如果需要原料/成品分离的精细投影，可以修改 `extract_macro_state()` 中的通道 0-1，但需同步更新 `state_builder.py` 以保持在线/离线一致性。

- **L3 残差的 baseline 是什么？为什么不是 PenguinAgent 的完整输出？**  
  这是 HRL-XF 架构的**核心设计决策**。详细说明如下：

  **设计选择（方案 E）**：`baseline = L1SafetyLayer._compute_baseline()` 的输出，即一个简化的安全保守报价。
  
  **语义**：
  - `baseline`：L1 层计算的"傻瓜式安全报价"（不会破产、不会违约）
  - `expert`：PenguinAgent 的实际报价
  - `residual = expert - baseline`：PenguinAgent 相对于安全基准的"策略调整"
  
  **为什么不用 PenguinAgent 的完整输出作为 baseline？**
  1. **OOD 风险**：如果训练时用 Penguin 输出作为 baseline，推理时也必须实时调用 Penguin。但 L3 残差会修改动作，导致状态轨迹偏离训练分布，产生分布漂移。
  2. **架构耦合**：PenguinAgent 是有状态的多轮谈判策略，无法简单地拆解为"baseline + residual"。
  3. **训练/推理一致性**：L1 baseline 在训练和推理时使用**相同的计算方法**，确保 L3 的输入分布一致。
  
  **残差=0 时的表现**：
  - 当前设计：表现 = L1 baseline（简单安全策略），**不等于** PenguinAgent
  - 这是可接受的权衡：牺牲"冷启动=Penguin"以换取训练/推理分布一致性
  
  **一致性要求**：
  - 训练时的 `baseline` 必须与推理时的 `L1SafetyLayer._compute_baseline()` 使用**完全相同的逻辑**
  - Tracker JSON 包含完整状态快照，可准确重建 L1 baseline
  - **建议**：在 tracker 中记录 `l1_baseline` 字段，或在 data_pipeline 中调用 L1 重建 baseline

---

## 9. 技术路线：后续探索方向

### 9.1 当前方案（方案 E：残差学习 + L1 baseline）

**架构**：
```
a_final = Clip(L1_baseline(s) + L3_residual(s), L1_mask)
```

**特点**：
- 残差=0 时表现为 L1 baseline（安全但简单）
- 训练/推理分布一致，无 OOD 风险
- L3 学习的是"如何从安全基准调整到专家水平"

**验证指标**：
- L3 预测残差的 MSE
- 与纯 L1 baseline 相比的利润提升
- 与 PenguinAgent 相比的胜率

### 9.2 后续探索（方案 F：直接 BC，放弃残差）

如果方案 E 验证有效（L3 能有效学习残差），可以探索更激进的方案：

**架构**：
```
a_final = Clip(L3_policy(s), L1_mask)  # L3 直接输出动作
```

**变更点**：
1. L3 输出层从残差 `(Δq, Δp)` 改为直接动作 `(q, p, t)`
2. 训练损失从 `MSE(pred_residual, expert - baseline)` 改为 `MSE(pred_action, expert)`
3. L1 只提供安全裁剪，不再提供 baseline

**优势**：
- L3 有更大的表达自由度
- 不受 L1 baseline 质量的限制
- 可以学习 Penguin 的完整策略

**风险**：
- 动作空间更大，学习难度更高
- 冷启动时没有安全保底（需要预训练充分）
- 需要更多训练数据

**实现步骤**（待验证方案 E 后）：
1. 修改 `L3ResidualModel` 为 `L3PolicyModel`，输出层改为 `(q, p, t)`
2. 修改训练损失为直接 BC
3. 添加冷启动保护（如：低置信度时使用启发式策略）
4. 对比方案 E 和方案 F 的表现

**决策依据**：
- 如果方案 E 的 L3 残差预测准确，但整体表现仍不如 Penguin → L1 baseline 质量不足 → 尝试方案 F
- 如果方案 E 表现接近或超过 Penguin → 残差范式有效 → 保持方案 E 并优化 L1
