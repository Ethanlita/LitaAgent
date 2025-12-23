# HRL-XF 运行指引

本指引覆盖：数据采集 → pipeline → 离线训练 → 中断恢复与规模说明。

## 0. 前置环境
- 进入仓库根目录（示例：Windows `D:\SCML_initial`）
- 使用虚拟环境：`.venv`
- 推荐设置：`MPLCONFIGDIR=./.mpl_cache`（避免 matplotlib 缓存权限问题）

## 1) 运行比赛（数据采集）
使用 `hrl_data_runner`，默认官方规模（可通过参数缩放）：
```bash
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner
```
> 说明：默认是“后台模式”，会自动把 stdout/stderr 重定向到日志文件；加 `--foreground` 可在终端前台输出。

Windows（PowerShell）示例：
```powershell
$env:MPLCONFIGDIR = ".\\.mpl_cache"
.\\.venv\\Scripts\\python.exe -m runners.hrl_data_runner
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
- `--n-competitors-per-world`：每个 world 的参赛者数量（默认：全部参赛者）
- `--max-worlds-per-config`：限制每个配置的最大 world 数  
- `--target-worlds`：目标总 world 数（自动折算为 `max_worlds_per_config`，需同时指定 `--n-competitors-per-world`）
- `--round-robin` / `--no-round-robin`：是否启用 round-robin（默认启用；禁用可更快但分配更随机）
- `--steps`：固定步数（用于快速验证）  
- `--track-only-penguin`：仅追踪 `PenguinAgent`（适合只用 Penguin 做专家示范时，显著降低 `tracker_logs` 体积与 pipeline 解析开销）  
- `--no-csv`：尽量减少 negmas CSV 输出（仍会保留必要文件，如 contracts/negotiations/stats）  
- `--resumable`：启用断点续跑（复用 `--output-dir`）  
- `--output-dir`：输出目录  
- `--parallelism`：并行模式（可填 `parallel/serial/dask/loky`；其中 `loky`/`loky:<fraction>` 会通过 `SCML_PARALLELISM` 启用 loky 执行器）  
- `--foreground`：前台输出
- `--no-auto-collect`：禁用自动归集
- `--quiet` / `-q`：静默模式

注意：当你指定了 `--n-competitors-per-world K` 时，为了保证公平分配，需要 `--max-worlds-per-config >= K`，否则会报“公平分配”错误（Cannot guarantee fair assignment）。

## 2) 中断恢复（resume）
`hrl_data_runner` 支持断点续跑：使用同一个 `--output-dir` 并加上 `--resumable` 即可自动跳过已完成 world。
```bash
MPLCONFIGDIR=./.mpl_cache .venv/bin/python -m runners.hrl_data_runner \
  --resumable --output-dir tournament_history/hrl_data_resume_std
```
中断后再执行同一命令即可继续（已完成的 world 会被跳过）。

如果想分批采集数据：  
- 仍可跑到**新的输出目录**  
- pipeline 可递归读取多个目录（把父目录作为输入即可）

## 3) 运行 pipeline
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import data_pipeline

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, micro_ds = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="Pe",  # 'Pe' 匹配 PenguinAgent（专家示范）
    num_workers=4,  # 并行解析（默认自动）
    goal_backfill="none",
)
data_pipeline.save_samples(macro_ds, micro_ds, f"{data_dir}/processed_samples", horizon=40)
PY
```

> Windows 注意：如果你用“内联脚本”（如 `python -c` / stdin）运行上述代码，`num_workers>1` 可能触发 multiprocessing 的 `<stdin>` 报错；此时请先用 `num_workers=1` 跑通闭环，或把脚本保存为 `.py` 文件并用 `if __name__ == "__main__":` 保护后再开并行。

## 4) 离线训练
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, _ = data_pipeline.load_tournament_data(data_dir, agent_name="Pe", goal_backfill="none")

cfg = training.TrainConfig(l2_epochs=10, l3_epochs=10, output_dir=f"{data_dir}/checkpoints", horizon=40)
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
    goal_backfill="l2",
    l2_model_path=f"{cfg.output_dir}/l2_bc.pt",
)
trainer.train_phase0_bc([], micro_goalhat)
training.save_model(l3_model, cfg.output_dir, "l3_bc.pt")
PY
```

> ⚠️ 注意：Tracker 的谈判事件里通常不会真正写入 `l2_goal`，因此 `micro_ds` 的 `goal` 往往是全 0 占位。  
> 现在可以通过 `load_tournament_data(goal_backfill="l2", l2_model_path=...)` 用已训练 L2 预测 `goal_hat` 并回填到 `micro.goal`（推荐口径，训练/推理一致）。

## 4.1) L4 蒸馏（推荐）
L4 的监督信号（soft α）来自**启发式 L4 教师**，可直接从专家日志离线构建样本并训练神经 L4：
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l4_coordinator import GlobalCoordinator

data_dir = "tournament_history/hrl_data_<timestamp>_std"
l2_path = f"{data_dir}/checkpoints/l2_bc.pt"

l4_samples = data_pipeline.load_l4_distill_data(
    data_dir,
    agent_name="Pe",
    goal_source="l2",
    l2_model_path=l2_path,
    num_workers=4,
)

cfg = training.TrainConfig(output_dir=f"{data_dir}/checkpoints", horizon=40, l4_epochs=10)
l4_model = GlobalCoordinator(horizon=cfg.horizon, thread_feat_dim=24, global_feat_dim=30)
trainer = training.HRLXFTrainer(None, None, l4_model, cfg)
trainer.train_l4_distill(l4_samples)
training.save_model(l4_model, cfg.output_dir, "l4_distill.pt")
PY
```

> Windows 注意：同样建议先用 `num_workers=1` 验证；如需并行抽取 L4 蒸馏数据，请把脚本保存为 `.py` 文件运行（避免 `<stdin>` spawn 问题）。

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
macro_ds, micro_ds = data_pipeline.load_tournament_data(data_dir, agent_name="Pe", num_workers=4)

cfg = training.TrainConfig(
    l2_epochs=10,
    l3_epochs=10,
    output_dir=f"{data_dir}/checkpoints",
    horizon=40,
    l2_resume_path=f"{data_dir}/checkpoints/l2_bc_epoch10.ckpt.pt",
    l3_bc_resume_path=f"{data_dir}/checkpoints/l3_bc_epoch10.ckpt.pt",
)
l2_model = HorizonManagerPPO(horizon=cfg.horizon)
l3_model = TemporalDecisionTransformer(horizon=cfg.horizon)
trainer = training.HRLXFTrainer(l2_model, l3_model, None, cfg)
trainer.train_phase0_bc(macro_ds, micro_ds)
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

## 6) P2 里程碑（后续工作）
- **在线训练（PPO/MAPPO）**：当前 `litaagent_std/hrl_xf/training.py` 仅提供离线 BC/AWR 与（可用的）L4 蒸馏；PPO 的在线采样接入仍是占位提示，MAPPO 未实现。
- **迁移到 `StdSyncAgent`**：把逐线程 `propose()/respond()` 改为同步接口 `first_proposals()/counter_all()`，让 L4 的“批次统一规划 + 缓存复用”在接口层天然成立，进一步消除回调顺序造成的残余差异（也更利于后续做在线多智能体训练与观测对齐）。
