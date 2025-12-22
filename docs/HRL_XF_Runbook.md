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
- `--output-dir`：输出目录  
- `--parallelism`：并行后端（推荐 `loky`）  
- `--foreground`：前台输出

注意：`max-worlds-per-config` 必须 **≥ 参赛代理数量**，否则会报“公平分配”错误。

## 2) 中断恢复（resume）
`hrl_data_runner` 本身 **不支持续跑**。如需续跑，建议使用可断点的默认 runner：
```bash
MPLCONFIGDIR=./.mpl_cache .venv/bin/python runners/run_default_std.py \
  --output-dir tournament_history/std_default_resume
```
中断后再执行同一命令即可继续。

如果仍需用 `hrl_data_runner` 采集数据：  
- 重新跑到**新的输出目录**  
- pipeline 可递归读取多个目录（把父目录作为输入即可）

## 3) 运行 pipeline
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import data_pipeline

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, micro_ds = data_pipeline.load_tournament_data(
    data_dir,
    agent_name="LitaAgent",
    num_workers=4,  # 并行解析（默认自动）
)
data_pipeline.save_samples(macro_ds, micro_ds, f"{data_dir}/processed_samples", horizon=40)
PY
```

## 4) 离线训练
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, micro_ds = data_pipeline.load_tournament_data(data_dir, agent_name="LitaAgent")

cfg = training.TrainConfig(l2_epochs=10, l3_epochs=10, output_dir=f"{data_dir}/checkpoints", horizon=40)
l2_model = HorizonManagerPPO(horizon=40)
l3_model = TemporalDecisionTransformer(horizon=40)
trainer = training.HRLXFTrainer(l2_model, l3_model, None, cfg)
trainer.train_phase0_bc(macro_ds, micro_ds)
PY
```

### 训练策略建议（离线阶段）
1) **先 BC 后 AWR**  
   - L2：行为克隆（macro 样本）  
   - L3：行为克隆（micro 残差）  
   - 若有奖励：继续 L3 AWR（同一模型权重上微调）  
2) **训练是否需要多次运行**  
   - 单次 `trainer.train_phase0_bc()` 只是完成一次 BC；  
   - 若新增数据或想继续优化：加载 checkpoint 后继续训练。  
3) **推荐轮数（起点）**  
   - L2 BC：10–50 epoch  
   - L3 BC：10–50 epoch  
   - L3 AWR：5–20 epoch（有 reward 才做）  
   - L4：当前仅设计在线阶段联合训练（离线暂不做）  

### 权重与断点文件路径
- 模型权重：`output_dir/l2_bc_epoch{N}.pt`、`output_dir/l3_bc_epoch{N}.pt`、`output_dir/l3_awr_epoch{N}.pt`  
- 断点文件：`output_dir/l2_bc_epoch{N}.ckpt.pt` 等（包含 optimizer + epoch）  
- `trainer.save_all("bc")` 会保存 `output_dir/l2_bc.pt`、`output_dir/l3_bc.pt` 等

### 训练中断是否可恢复？
支持。训练过程会周期性保存 `*.ckpt.pt`（模型 + 优化器 + epoch）。
恢复示例（继续 L3 BC）：
```bash
MPLCONFIGDIR=./.mpl_cache MPLBACKEND=Agg .venv/bin/python - <<'PY'
from litaagent_std.hrl_xf import training, data_pipeline
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import TemporalDecisionTransformer

data_dir = "tournament_history/hrl_data_<timestamp>_std"
macro_ds, micro_ds = data_pipeline.load_tournament_data(data_dir, agent_name="LitaAgent", num_workers=4)

cfg = training.TrainConfig(
    l2_epochs=10,
    l3_epochs=10,
    output_dir=f"{data_dir}/checkpoints",
    horizon=40,
    l3_bc_resume_path=f"{data_dir}/checkpoints/l3_bc_epoch10.ckpt.pt",
)
l2_model = HorizonManagerPPO(horizon=40)
l3_model = TemporalDecisionTransformer(horizon=40)
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
