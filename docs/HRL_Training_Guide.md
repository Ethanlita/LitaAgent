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

**答案**：使用 **专家代理**（如 PenguinAgent、DecentralizingAgent），而非未训练的 HRL-XF。

```
┌─────────────────────────────────────────────────────────────────┐
│                    数据收集流程                                 │
│                                                                 │
│   1) 运行 Tournament                                            │
│           PenguinAgent vs DecentralizingAgent vs ...           │
│                         ↓                                       │
│   2) 从 Tracker 日志读取专家动作                                │
│           expert_action = (quantity, price, time)              │
│                         ↓                                       │
│   3) 离线计算 L1 基准（使用日志中的状态）                       │
│           baseline = compute_l1_baseline_offline(state_dict)   │
│                         ↓                                       │
│   4) 计算残差标签                                               │
│           residual = expert_action - baseline                  │
│                         ↓                                       │
│   5) 训练 L3 模型                                               │
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

### 1.4 训练目标：模仿专家，超越基准

```
┌─────────────────────────────────────────────────────────────────┐
│                    离线模仿学习阶段（冷启动）                      │
│                                                                 │
│   专家代理（如 PenguinAgent）参加比赛 → 生成 Tracker 日志         │
│                         ↓                                       │
│   解析日志：                                                    │
│   - 宏观（L2）：state → v2_goal（reconstruct_l2_goals）          │
│   - 微观（L3）：expert_action、history、state                    │
│                         ↓                                       │
│   训练 L2（BC）：g_hat = L2(state)                               │
│                         ↓                                       │
│   用 L2 预测 goal_hat 回填到 micro.goal                          │
│   （解决 Tracker 谈判事件通常不写入 l2_goal 的断层）              │
│                         ↓                                       │
│   baseline = compute_l1_baseline_offline(state)  ← 离线计算       │
│         ↓                                                       │
│   residual_label = expert_action - baseline                      │
│         ↓                                                       │
│   训练 L3（BC/AWR）：residual = f(history, goal_hat, role, baseline) │
│                                                                 │
│   蒸馏/预训练 L4：教师=启发式 L4，学习连续 α 向量（线程数可变）     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    在线强化学习阶段（联合优化）                    │
│                                                                 │
│   (L1 + L2 + L3 + L4) 参加比赛：                                 │
│   - 先冻结 L2/L3，优先训练 L4（稳定并发调度）                    │
│   - 再逐步解冻联合训练 L2/L3/L4（小学习率）                      │
└─────────────────────────────────────────────────────────────────┘



┌─────────────────────────────────────────────────────────────────┐
│                    阶段 1: 离线预训练 L2                         │
│                                                                 │
│   L2: BC（v2 目标：成交 + 缺口 + 活跃意图 + 软分桶 + 稳定价格）    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    阶段 2: 离线预训练 L3                         │
│                                                                 │
│   goal 来源：用已训练 L2 对 daily_state 预测 goal_hat 回填         │
│   L3: BC/AWR（residual = expert - baseline，条件化 goal_hat）     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    阶段 3: 蒸馏/预训练 L4                         │
│                                                                 │
│   教师=启发式 L4 → 输出连续 α（全局、K 线程可变）                │
│   监督学习： (thread_feat_set, global_feat) -> α                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    阶段 4: 在线联合训练（预留/后续）                │
│                                                                 │
│   A: 冻结 L2/L3，先训 L4                                          │
│   B: 解冻并小学习率联合微调 L2/L3/L4                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 架构概览与数学定义
LitaAgent-HRL 由四层组成，时间尺度与职责不同：
- **L1 安全/基准层**：提炼自 Penguin 的微观谈判逻辑，输出安全掩码与基准报价，防止超量、超价、超资金。  
- **L2 战略管理层**：每日决策（Day 级），生成 16 维分桶目标向量  
  \( g_t \in \mathbb{R}^{16} = 4桶 \times [Q^{buy}, P^{buy}, Q^{sell}, P^{sell}] \)。为便于叙述，后文常以单桶四元组表示。  
- **L3 残差执行层**：轮级决策（Round 级），在 L1 基准上输出残差  
  \( \Delta a = [\Delta q, \Delta p] \)，合成最终报价  
  \( a_{final} = \text{clip}(a_{base} + \Delta a, \text{mask}) \)。  
- **L4 并发协调层**：对所有活跃谈判线程统一计算连续权重 \( \alpha_k \)（线程数可变），用于全局调度与冲突消解：既可调制 L3 残差的激进程度，也可在生成报价时做“动态预留”（按 \( \alpha \) 排序、逐线程扣减 `B_free/Q_safe`），避免顺序依赖与粗暴切块。

时间顺序：  
1) 每日开始：L2 读取宏观状态，输出 \( g_t \)；L1 不改宏观，仅提供安全边界。  
2) 每次决策：L3 在 L1 基准上输出残差；L4 在当日/当轮对所有活跃线程**统一**计算 \( \alpha \) 并缓存复用；买侧在生成报价时按 \( \alpha \) 从高到低对各线程动作做 `clip_action`，并动态扣减剩余 `B_free/Q_safe`（不做固定切块），使 L4 的作用点发生在裁剪之前，同时减少回调顺序导致的差异。  
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
| `n_products` | 产品数量（用于正确判断 is_last_level 与 x_role） | 3 |
| `needed_supplies` | 今日需要通过协商获得的采购量（缺口信号） | 12 |
| `needed_sales` | 今日需要通过协商获得的销售量（缺口信号） | 8 |
| `offers_snapshot` | 活跃谈判报价快照（用于压力/价格趋势通道、活跃意图） | {'buy': [...], 'sell': [...]} |

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
- 支持 `--resumable` 续跑（复用同一 `--output-dir`）。
- 可选 `--no-csv` 减少 negmas CSV 写盘（仍会保留必要文件，如 contracts/negotiations/stats）。

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
    goal_backfill="none",  # 默认不回填（此时 micro.goal 往往为占位 0）
)
# 保存样本，horizon 应与训练配置一致（默认 40）
save_samples(macro_ds, micro_ds, "./data/hrlxf_samples", horizon=40)

# 预期输出：
# [INFO] Found 752 world directories
# [INFO] Extracted 114552 macro samples, 64752 micro samples
```
> Windows 注意：如果你在交互式环境/内联脚本里运行（如 `python -c` / stdin），建议先把 `num_workers=1` 跑通闭环；要开启多进程并行解析，请将代码保存成 `.py` 文件并用 `if __name__ == "__main__":` 保护后运行。
- Macro 样本（L2）：`MacroSample(day, state_static, state_temporal, x_role, goal)`，goal 为反推的日级目标。
- Micro 样本（L3）：`MicroSample(negotiation_id, history, baseline, residual, goal, time_label, time_mask)`，time_mask 为 L1 安全约束。

> ⚠️ 注意：Tracker 的谈判事件里通常不会真正写入 `l2_goal`，因此 `micro_ds` 的 `goal` 往往是全 0 占位。  
> 按最新训练计划，应先用 v2 标签训练 L2，再用 L2 对每日宏观状态预测 `goal_hat`，并在生成 micro 样本时回填到 `micro.goal` 后再训练 L3（解决“回填断层”）。

> ✅ **Tracker JSON 格式**：`data_pipeline.py` 会自动递归搜索 `tracker_logs/` 子目录中的 Tracker JSON 文件。

## 5. 训练流程：预训练与在线微调
### 5.1 预训练（离线）
离线阶段的目标：把“宏观目标 → 微观残差 → 并发调度”拆开学，保证冷启动稳定，并尽量让训练/推理输入分布一致。

**推荐顺序：离线训练 L2 → 离线训练 L3 → 蒸馏/预训练 L4**

1) **离线训练 L2（BC，v2 标签）**
   - 标签来自 `reconstruct_l2_goals()` 的 v2 口径（成交 + 缺口补偿 + 活跃意图 + 软分桶 + 稳定价格），替代“无成交即全 0 / 价格 max/min”的高噪声标签。  
2) **离线训练 L3（BC/AWR，使用 goal_hat 回填）**
   - `residual = expert_action - L1_baseline(state)`（保持残差学习核心设计）。
   - 关键：用已训练好的 L2 对每日宏观状态预测 `goal_hat`，并在生成 micro 样本时回填到 `micro.goal`，让 L3 学到“目标条件下”的残差策略（见 `load_tournament_data(goal_backfill="l2")`）。
3) **蒸馏/预训练 L4（监督学习，教师=启发式 L4）**
   - 专家日志无法直接提供 L4 的监督信号（α/权重），因此先用启发式 L4 作为教师，离线蒸馏出神经 L4；在线阶段再联合训练。

最小可运行示例（仅 L2/L3 的 BC；若未做 goal_hat 回填，则 L3 的 goal 条件等价于常量 0）：  
```python
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "./tournament_history/hrl_data_<timestamp>_std"
macro_ds, _ = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="Pe",
    num_workers=4,
    goal_backfill="none",
)

cfg = training.TrainConfig(l2_epochs=10, l3_epochs=10, output_dir="./out_hrlxf")
l2_model = HorizonManagerPPO(horizon=cfg.horizon)
l3_model = TemporalDecisionTransformer(horizon=cfg.horizon)
trainer = training.HRLXFTrainer(l2_model, l3_model, None, cfg)

# 1) 先训 L2（micro.goal 此时往往为占位 0，不建议直接训 L3）
trainer.train_phase0_bc(macro_ds, [])
training.save_model(l2_model, cfg.output_dir, "l2_bc.pt")

# 2) 用已训练 L2 回填 micro.goal（goal_hat）后再训 L3
_, micro_goalhat = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="Pe",
    num_workers=4,
    goal_backfill="l2",
    l2_model_path=f"{cfg.output_dir}/l2_bc.pt",
)
trainer.train_phase0_bc([], micro_goalhat)
trainer.save_all("bc_example")
```
> “占位模型”意为简单可运行的线性/MLP 示例；实际项目应替换为更强的 Transformer/PPO 等模型。

### 5.2 在线微调（预留）
当前代码实现层面：  
- `litaagent_std/hrl_xf/training.py` 已实现离线 **BC**（L2/L3）与 L3 的 **AWR**；  
- **PPO 仅提供缓冲区与损失计算框架**，`train_phase2_ppo()` 仍是占位提示（尚未接入 SCML 仿真采样）；  
- **MAPPO 未实现**。  

因此短期建议：先把离线闭环跑通（含 `goal_hat` 回填与 L4 启发式蒸馏），在线微调作为后续里程碑。

### 5.3 L4（并发协调层）训练要点
- **输入语义（与实现一致）**：L4 不再使用 L3 的隐状态作为输入，而是使用可离线重建的显式特征：  
  - `thread_feat`：每个线程一条（交期/角色、谈判进度、与 L2 桶目标的差值、该交期的 `X_temporal[δ]` 切片、L1 baseline 等）；  
  - `global_feat`：全局上下文（`goal_hat`、`x_static`、活跃买卖线程数等）。  
- **动作语义**：输出连续权重 \(\alpha_k\)（softmax 归一化）。在方案 B 中，\(\alpha\) 主要用于**全局调度**：买侧按 \(\alpha\) 从高到低对各线程动作进行裁剪并动态扣减剩余 `B_free/Q_safe`（不做 `B_free_k=\alpha_k B_free` 这类固定切块）。  
- **顺序无关性（与实现一致）**：每次决策先收集全部活跃线程特征，统一算一次 L4 并缓存，随后各线程从缓存中取本轮结果，减少回调顺序差异。  
- **当前可行训练路径**：先离线训练 L2/L3，再用 `data_pipeline.load_l4_distill_data()` 从专家日志离线构建 `(thread_feat_set, global_feat) -> α_teacher` 样本并监督蒸馏 L4；在线 RL 微调作为后续工作。

## 6. 如何运行数据采集 runner（HRL-XF）
推荐使用 `runners/hrl_data_runner.py` 采集专家日志（默认会自动启用 Tracker 并输出到 `tournament_history/`）：

```bash
# Linux/macOS（默认后台模式会自动重定向 stdout/stderr 到日志文件）
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner --configs 3 --runs 1

# 调试时可前台输出
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner --foreground

# 仅追踪 PenguinAgent（如果离线训练只用 Penguin 作专家示范，可显著降低 tracker_logs 体积）
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner --track-only-penguin
```

Windows（PowerShell）示例：
```powershell
$env:MPLCONFIGDIR = ".\\.mpl_cache"
.\\.venv\\Scripts\\python.exe -m runners.hrl_data_runner --configs 3 --runs 1
```

输出目录示例：`tournament_history/hrl_data_<timestamp>_std/`（默认，可用 `--output-dir` 覆盖），包含 `tracker_logs/agent_*.json` 等 Tracker JSON 文件。

> 更完整的参数与规模说明见：`docs/HRL_XF_Runbook.md`。

> ✅ 数据管道使用 Tracker JSON 格式，包含完整状态快照，可直接使用 `load_tournament_data()` 加载。

## 7. 如何训练模型并接入代理（HRL-XF）
1) 比赛结束后，解析日志生成 `macro_ds`/`micro_ds`。  
2) 使用 `litaagent_std/hrl_xf/training.py` 训练 L2/L3（BC/AWR），并保存权重。  
3) 使用 `data_pipeline.load_l4_distill_data()` 提取 L4 蒸馏数据，并用 `training.train_l4_distill()`（或 `HRLXFTrainer.train_l4_distill()`）训练 L4 权重。  
4) 在 runner 中注册 `LitaAgentHRL` 进行仿真验证（neural 模式可通过 `l2_model_path/l3_model_path/l4_model_path` 加载权重）。

训练示例：
```python
from pathlib import Path
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "tournament_history/hrl_data_<timestamp>_std"

# 使用 PenguinAgent 作为专家示范
macro_ds, _ = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="Pe",
    num_workers=4,  # 并行解析（默认自动）
    goal_backfill="none",
)
# [INFO] Found 752 world directories
# [INFO] Extracted 114552 macro samples, 64752 micro samples

# 开始训练
# 注意：TrainConfig.horizon 应与 save_samples 的 horizon 一致
cfg = training.TrainConfig(output_dir="./checkpoints_hrlxf", horizon=40)
l2_model = HorizonManagerPPO(horizon=cfg.horizon)
l3_model = TemporalDecisionTransformer(horizon=cfg.horizon)
trainer = training.HRLXFTrainer(l2_model, l3_model, None, cfg)

# 1) 先训 L2（v2 标签）
trainer.train_phase0_bc(macro_ds, [])
training.save_model(l2_model, cfg.output_dir, "l2_bc.pt")

# 2) 用已训练 L2 预测 goal_hat 回填 micro.goal，再训 L3
_, micro_goalhat = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="Pe",
    num_workers=4,
    goal_backfill="l2",
    l2_model_path=f"{cfg.output_dir}/l2_bc.pt",
)
trainer.train_phase0_bc([], micro_goalhat)
training.save_model(l3_model, cfg.output_dir, "l3_bc.pt")
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
  专家日志（如 PenguinAgent）不使用 HRL 架构，因此不会产生“线程集合 → 权重/α”的 L4 监督信号，无法直接从专家日志训练 L4。  
  按最新训练计划：先离线训练好 L2/L3，然后在仿真中运行 HRL（L1 + 已训 L2 + 已训 L3 + 启发式 L4），把启发式 L4 的输出当作教师信号，采集 `(thread_feat_set, global_feat) -> α_teacher` 数据离线蒸馏出神经 L4；最后进入在线联合训练阶段。  
  其中 `global_feat` 可包含 `goal_hat`、`x_static` 等全局上下文；`thread_feat` 包含每个 negotiation 的交期/角色、在该交期切片的 `X_temporal[δ]`、L1/L2 约束与谈判进度等可离线重建特征。
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
  - 在线模式：实时计算采购/销售两个 price_diff（成交 VWAP + 活跃报价轮次衰减加权均值 + spot）以及买/卖压力
  - 离线模式：若有谈判日志，优先按**轮次衰减/数量加权**重建活跃报价；否则使用 tracker JSON 的 `offers_snapshot`；仍缺失时回退 commitments/0
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

- **为什么库存投影只使用原材料库存？**  
  这是**有意的设计决策**，而非遗漏。理由如下：
  1. **与 L1 安全层一致**：L1 的库容约束针对原材料入库与产能消耗，库存投影使用原材料更能反映“可加工”约束。
  2. **因果链一致**：Q_out 是成品出库，不影响原材料库存；将其混入原材料投影会造成语义混淆。
  3. **信息压缩**：L2 关注“是否需要补货/清仓”，原材料投影 + Q_out 承诺已足够提供趋势信号。
  4. **复杂度权衡**：若未来需要更精细的双轨投影，可扩展为 `(H+1, 12)` 并同步更新在线/离线实现。
 
  **注意**：若扩展为原料/成品双轨投影，需同步修改 `extract_macro_state()` 与 `state_builder.py`，保持在线/离线一致性。

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

### 9.3 P2 里程碑：在线训练与 Sync 化（工程侧）
当前仓库的“P2”不仅包括 PPO/MAPPO 的在线训练接入，也包括 **迁移到 `StdSyncAgent`**：
- **在线训练（PPO/MAPPO）**：需要把采样、回放缓冲区、优势估计与参数更新接入 SCML 仿真循环；当前 `training.py` 的 `train_phase2_ppo()` 仍是占位提示，MAPPO 未实现。
- **迁移到 `StdSyncAgent`**：把逐线程 `propose()/respond()` 的回调改为同步接口 `first_proposals()/counter_all()`，使得 L4 的“批次统一规划”在接口层天然成立，避免因为回调顺序/线程推进不一致带来的残余顺序依赖（也更利于后续做在线多智能体训练与对齐观测）。
