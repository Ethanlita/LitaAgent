# HRL-XF 运行指引

本指引覆盖：数据采集 → pipeline → 离线训练 → 中断恢复与规模说明。

## 0. 前置环境
- 进入仓库根目录：`/Users/lita/Documents/GitHub/LitaAgent`
- 使用虚拟环境：`.venv`
- 推荐设置：`MPLCONFIGDIR=./.mpl_cache`（避免 matplotlib 缓存权限问题）

## 1) 运行比赛（数据采集）
使用 `hrl_data_runner`，默认官方规模（可通过参数缩放）：
```bash
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner
```

小规模示例（快速验证）：
```bash
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner \
  --configs 1 --runs 1 --max-top-2025 2 --max-top-2024 2 \
  --max-worlds-per-config 11 --steps 50 \
  --output-dir tournament_history/hrl_data_smoke_std \
  --parallelism loky --foreground
```

常见参数：
- `--configs` / `--runs`：配置数 / 每配置运行次数  
- `--max-top-2025` / `--max-top-2024`：Top Agents 数量上限  
- `--max-worlds-per-config`：每配置最大 world 数  
- `--steps`：固定步数（用于快速验证）  
- `--track-only-penguin`：仅追踪 `PenguinAgent`（适合只用 Penguin 做专家示范时，显著降低 `tracker_logs` 体积与 pipeline 解析开销）  
- `--no-csv`：尽量减少 negmas CSV 输出（仍会保留必要文件，如 contracts/negotiations/stats）  
- `--forced-logs-fraction`：强制保留详细日志的 world 比例（默认 0.1；即使 `--no-csv` 也会保留这部分详细日志）  
- `--resumable`：启用断点续跑（复用 `--output-dir`；比赛完成后会自动清理 resumable 中间数据）  
- `--output-dir`：输出目录  
- `--parallelism`：并行后端（推荐 `loky`）  
- `--foreground`：前台输出

注意：`max-worlds-per-config` 必须 **≥ 参赛代理数量**，否则会报“公平分配”错误。

## 2) 中断恢复（resume）
`hrl_data_runner` 支持断点续跑：使用同一个 `--output-dir` 并加上 `--resumable` 即可自动跳过已完成 world。
```bash
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner \
  --resumable --output-dir tournament_history/hrl_data_resume_std
```
中断后再执行同一命令即可继续（已完成的 world 会被跳过）。
注意：比赛完成后会自动清理 resumable 中间数据，因此**完成后无法再继续追加**。

如果想分批采集数据：  
- 仍可跑到**新的输出目录**  
- pipeline 可递归读取多个目录（把父目录作为输入即可）

## 3) 运行 pipeline
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import data_pipeline

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, _ = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="LitaAgent",
    num_workers=4,  # 并行解析（默认自动）
)
data_pipeline.save_samples(macro_ds, [], f"{data_dir}/processed_samples", horizon=40)
PY
```

## 4) 离线训练
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, _ = data_pipeline.load_tournament_data(data_dir, agent_name="LitaAgent")

cfg = training.TrainConfig(l2_epochs=10, l3_epochs=10, output_dir=f"{data_dir}/checkpoints", horizon=40)
l2_model = HorizonManagerPPO(horizon=cfg.horizon)
training.train_l2_bc(l2_model, macro_ds, cfg)

# 用已训练 L2 预测 goal_hat 回填 micro.goal，再训练 L3
goal_predictor = training.make_l2_goal_predictor(l2_model, device=cfg.device)
# 若 micro_ds 为 0：尝试 strict_json_only=False 允许使用 negotiations.csv（需同时存在 stats.csv/negotiations.csv）
micro_ds = data_pipeline.load_tournament_micro_samples(
    data_dir,
    agent_name="LitaAgent",
    goal_predictor=goal_predictor,
    horizon=cfg.horizon,
)
l3_model = TemporalDecisionTransformer(horizon=cfg.horizon)
training.train_l3_bc(l3_model, micro_ds, cfg)
PY
```

### 训练策略建议（离线阶段）
1) **先 BC 后 AWR**  
   - L2：行为克隆（macro 样本）  
   - L3：行为克隆（micro 残差）  
   - 若有奖励：继续 L3 AWR（同一模型权重上微调）  
2) **训练是否需要多次运行**  
   - 单次 `train_l2_bc()` / `train_l3_bc()` 只是完成一次离线训练；  
   - 若新增数据或想继续优化：加载 checkpoint 后继续训练。  
3) **推荐轮数（起点）**  
   - L2 BC：10–50 epoch  
   - L3 BC：10–50 epoch  
   - L3 AWR：5–20 epoch（有 reward 才做）  
   - L4：推荐先做启发式蒸馏（监督），在线再联合训练  

### 权重与断点文件路径
- 模型权重：`output_dir/l2_bc_epoch{N}.pt`、`output_dir/l3_bc_epoch{N}.pt`、`output_dir/l3_awr_epoch{N}.pt`  
- 断点文件：`output_dir/l2_bc_epoch{N}.ckpt.pt` 等（包含 optimizer + epoch）  
- 如需额外手动保存，可使用 `training.save_model(...)`

### 训练中断是否可恢复？
支持。训练过程会周期性保存 `*.ckpt.pt`（模型 + 优化器 + epoch）。
恢复示例（继续 L3 BC）：
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, _ = data_pipeline.load_tournament_data(data_dir, agent_name="LitaAgent", num_workers=4)

cfg = training.TrainConfig(
    l2_epochs=10,
    l3_epochs=10,
    output_dir=f"{data_dir}/checkpoints",
    horizon=40,
    l2_resume_path=f"{data_dir}/checkpoints/l2_bc_epoch10.ckpt.pt",
    l3_bc_resume_path=f"{data_dir}/checkpoints/l3_bc_epoch10.ckpt.pt",
)
# 1) 先加载/恢复 L2（用于生成 goal_hat）
l2_model = HorizonManagerPPO(horizon=cfg.horizon)
training.train_l2_bc(l2_model, macro_ds, cfg, resume_path=cfg.l2_resume_path)

# 2) 用已训练 L2 生成 micro_ds（goal_hat 回填），再恢复 L3 BC
goal_predictor = training.make_l2_goal_predictor(l2_model, device=cfg.device)
# 若 micro_ds 为 0：尝试 strict_json_only=False 允许使用 negotiations.csv（需同时存在 stats.csv/negotiations.csv）
micro_ds = data_pipeline.load_tournament_micro_samples(
    data_dir,
    agent_name="LitaAgent",
    goal_predictor=goal_predictor,
    horizon=cfg.horizon,
)
l3_model = TemporalDecisionTransformer(horizon=cfg.horizon)
training.train_l3_bc(l3_model, micro_ds, cfg, resume_path=cfg.l3_bc_resume_path)
PY
```
建议将 `save_every` 设小（如 1-5），以减少中断损失。

## 5) 官方标准规模的 world 数
默认官方规模（`run_default_std.py` / `hrl_data_runner` 默认参数）：
```
worlds ≈ n_configs × n_runs × max_worlds_per_config
```
当 `n_competitors_per_world = 全部参赛者` 且 `max_worlds_per_config = 参赛者数量` 时：
```
worlds ≈ 40 × 参赛者数量
```
例如参赛者 11 个时，worlds 约为 440。
