# 训练记录（HRL-XF）

> 说明：本文记录实际运行过的训练及其参数、结果、权重与日志路径。若日志未显式写入参数，则以日志文件名或当时命令推断，并在条目内注明“推断”。

## 训练入口与数据
- 训练入口：`runners/run_hrl_bc_awr_train.py`
- CLI 参数总表：`runners/run_hrl_bc_awr_train.py`（argparse 定义）
- 日志目录：`diagnose_logs/`
- 权重输出目录：`training_runs/`
- 数据目录：`tournament_history/`
- 默认 agent 过滤：`PenguinAgent`（脚本默认）
- world 切分（来自日志）：
  - L2：train=80112，val=9424，test=4712（合计 94248 world）
  - L3：train=84824，val=9424，test=0（合计 94248 world）

## 已完成的训练

### L2 基线（旧 loss，原始 L2 BC）
- 状态：完成
- 训练内容：L2 行为克隆（原始 loss / MSE），未使用 Q 变换或加权。
- 输出目录：`training_runs/l2_only_cuda_20251231_010729`
- 设备：CUDA（从输出目录命名与运行设定推断）
- 关键参数（日志/推断）：
  - `--phases l2`
  - `--num-workers 31`
  - `--l2-batch-size 512`（推断：来自日志名 `bs512` 与当时设定）
  - `--l2-epochs 40`（后续阶段继续到 50/60/70）
  - `--l2-lr`：阶段 1 未记录（默认 3e-4，推断）；阶段 2/3/4 见下
- 训练阶段与结果：
  - 阶段 1：Epoch 1-40（lr 推断为 3e-4）
    - 末次：`Epoch 40/40 | Loss 0.6901 | MSE 0.6848`
    - Eval：`val loss 0.6928 / mse 0.6884`，`test loss 0.6837 / mse 0.6801`
    - 日志：`diagnose_logs/l2_only_cuda_20251231_010729_bs512_20251231_014534.out`
  - 阶段 2：Epoch 41-50（lr=1e-4）
    - 末次：`Epoch 50/50 | Loss 0.6788 | MSE 0.6728`
    - Eval：`val loss 0.6900 / mse 0.6837`，`test loss 0.6783 / mse 0.6731`
    - 日志：`diagnose_logs/l2_only_cuda_20251231_010729_lr1e-4_e50_20251231_103518.out`
  - 阶段 3：Epoch 51-60（lr=5e-5）
    - 末次：`Epoch 60/60 | Loss 0.6699 | MSE 0.6635`
    - Eval：`val loss 0.6940 / mse 0.6855`，`test loss 0.6756 / mse 0.6694`
    - 日志：`diagnose_logs/l2_only_cuda_20251231_010729_lr5e-5_e60_20251231_103518.out`
  - 阶段 4：Epoch 61-70（lr=5e-5）
    - 末次：`Epoch 70/70 | Loss 0.6624 | MSE 0.6554`
    - Eval：`val loss 0.6695 / mse 0.6634`，`test loss 0.6634 / mse 0.6577`
    - 日志：`diagnose_logs/l2_only_cuda_20251231_010729_lr5e-5_e70_20251231_205749.out`
- 主要权重：
  - 最终权重：`training_runs/l2_only_cuda_20251231_010729/l2_bc.pt`
  - 各 epoch 权重：`training_runs/l2_only_cuda_20251231_010729/l2_bc_epoch*.pt`

### L2 改损失（Q log1p + 权重 2）
- 状态：完成
- 训练内容：L2 BC，Q 分支使用 `log1p` 变换并加权（权重 2），其余保持原始 loss 结构。
- 输出目录：`training_runs/l2_qlog1p_w2_20251231_220338`
- 关键参数（日志/推断）：
  - `--phases l2`
  - `--num-workers 32`
  - `--l2-q-transform log1p`
  - `--l2-q-weight 2`
  - `--l2-batch-size 512`
  - `--l2-epochs 40`
  - `--l2-lr` 默认 3e-4（推断，日志未显式记录）
- 进度与结果：
  - 断点恢复：从 `epoch=6` 继续到 `epoch=40`
  - 末次：`Epoch 40/40 | Loss 0.0568 | MSE 0.0667`
  - Eval：`val loss 0.0623 / mse 0.0716`，`test loss 0.0547 / mse 0.0652`
  - 后续阶段（学习率分段）：
    - 阶段 2：Epoch 41-50（lr=1e-4）
      - 末次：`Epoch 50/50 | Loss 0.0553 | MSE 0.0651`
      - Eval：`val loss 0.0607 / mse 0.0698`，`test loss 0.0527 / mse 0.0630`
      - 日志：`diagnose_logs/l2_qlog1p_w2_lr1e-4_e50_20260104_203711.out`
    - 阶段 3：Epoch 51-60（lr=5e-5）
      - 末次：`Epoch 60/60 | Loss 0.0549 | MSE 0.0646`
      - Eval：`val loss 0.0602 / mse 0.0692`，`test loss 0.0519 / mse 0.0622`
      - 日志：`diagnose_logs/l2_qlog1p_w2_lr5e-5_e60_20260104_203711.out`
- 量纲评估（验证集，原始单位）：
  - Q：`MAE 1.2644`，`RMSE 1.9643`，`mean_target 7.6046`，`mean_rel_err 0.8650`
  - P：`MAE 0.1496`，`RMSE 0.3459`，`mean_target 18.8000`，`mean_rel_err 0.0098`
- 主要日志：
  - `diagnose_logs/l2_qlog1p_w2_20251231_220338.out`
  - `diagnose_logs/l2_qlog1p_w2_resume_20260101_205639.out`
  - `diagnose_logs/l2_qlog1p_w2_resume_20260104_155710.out`
  - `diagnose_logs/l2_qlog1p_w2_lr1e-4_e50_20260104_203711.out`
  - `diagnose_logs/l2_qlog1p_w2_lr5e-5_e60_20260104_203711.out`
- 主要权重：
  - `training_runs/l2_qlog1p_w2_20251231_220338/l2_bc.pt`
  - `training_runs/l2_qlog1p_w2_20251231_220338/l2_bc_epoch*.pt`

### L3（旧 L2 回填，GPU + 预张量化 + 回填批处理）
- 状态：完成
- 输出目录：`training_runs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_210620`
- 旧 L2 权重：`training_runs/l2_only_cuda_20251231_010729/l2_bc.pt`
- 训练内容：L3 BC，输出 AOP（ACCEPT/REJECT/END + q/p/t），`goal_hat` 来自旧 L2 回填。
- 关键参数（两段训练）：
  - 初始运行（`diagnose_logs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_210620.out`）：
    - `--phases l3`
    - `--num-workers 30`
    - `--l2-backfill-device cuda`
    - `--l2-backfill-batch-size 512`
    - `--l3-pre-tensorize`、`--tensorize-workers 8`
    - `--dataloader-workers 8`、`--dataloader-pin-memory`
    - `--dataloader-persistent-workers False`
    - `--dataloader-prefetch-factor 2`
    - `--backfill-world-chunk 256`
    - `--batch-log-every 50`
    - `--l3-batch-size 64`（推断：来自日志名）
  - 降学习率续训（`diagnose_logs/l3_bc_oldl2_gpu_batchbackfill_bs256_lr5e-5_e13_20260104_100413.out`）：
    - `--num-workers 32 (auto)`
    - `--l2-backfill-batch-size 1024`
    - `--dataloader-workers 16`
    - `--dataloader-persistent-workers True`
    - `--dataloader-prefetch-factor 4`
    - `--batch-log-every 200`
    - `--l3-lr 5e-5`
    - `--l3-epochs 13`（从 checkpoint 继续，实际为 5 个 epoch）
    - `--l3-batch-size 256`（推断：由日志名与默认值）
- 结果（续训阶段 Epoch 9-13）：
  - Epoch 9/13：`Loss 0.0357 (q=0.0001, p=0.0023, t=0.0001) op_acc=0.989`
  - Epoch 10/13：`Loss 0.0348 (q=0.0001, p=0.0020, t=0.0001) op_acc=0.989`
  - Epoch 11/13：`Loss 0.0341 (q=0.0001, p=0.0017, t=0.0001) op_acc=0.990`
  - Epoch 12/13：`Loss 0.0338 (q=0.0001, p=0.0017, t=0.0001) op_acc=0.990`
  - Epoch 13/13：`Loss 0.0338 (q=0.0001, p=0.0019, t=0.0001) op_acc=0.990`
- 主要权重：
  - `training_runs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_210620/l3_bc_epoch13.pt`
  - `training_runs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_210620/l3_bc_epoch13.ckpt.pt`
  - `training_runs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_210620/l3_bc.pt`
- 主要日志：
  - 初始运行：`diagnose_logs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_210620.out`
  - 续训完成：`diagnose_logs/l3_bc_oldl2_gpu_batchbackfill_bs256_lr5e-5_e13_20260104_100413.out`

### L3（新 L2 回填，按 chunk 解析→回填）
- 状态：完成
- 输出目录：`training_runs/l3_bc_newl2_chunk_20260105_070123`
- 新 L2 权重：`training_runs/l2_qlog1p_w2_20251231_220338/l2_bc.pt`
- world 切分：train=80112，val=9424，test=4712
- 训练内容：L3 BC，输出 AOP（ACCEPT/REJECT/END + q/p/t），`goal_hat` 来自新 L2 回填；解析/回填按 chunk 顺序执行，避免流水线积压。
- 关键参数（两段训练）：
  - 阶段 1（`diagnose_logs/l3_bc_newl2_chunk_e8_lr1e-4_20260105_070123.out`）：
    - `--phases l3`
    - `--num-workers 32 (auto)`
    - `--l2-backfill-device cuda`
    - `--l2-backfill-batch-size 1024`
    - `--backfill-world-chunk 256`
    - `--l3-pre-tensorize`、`--tensorize-workers 8`
    - `--dataloader-workers 16`、`--dataloader-pin-memory`
    - `--dataloader-persistent-workers True`
    - `--dataloader-prefetch-factor 4`
    - `--l3-batch-size 256`
    - `--l3-epochs 8`、`--l3-lr 1e-4`
  - 阶段 2（`diagnose_logs/l3_bc_newl2_chunk_e13_lr5e-5_20260105_070123.out`）：
    - `--l3-epochs 13`（从 epoch 8 续训到 13）
    - `--l3-lr 5e-5`
- 结果：
  - Epoch 8/8：`Loss 0.0386 (q=0.0001, p=0.0030, t=0.0000) op_acc=0.988`
  - Eval：`L3/val loss 0.0381`，`L3/test loss 0.0386`
  - Epoch 13/13：`Loss 0.0352 (q=0.0001, p=0.0018, t=0.0001) op_acc=0.989`
  - Eval：`L3/val loss 0.0348`，`L3/test loss 0.0358`
- 主要权重：
  - `training_runs/l3_bc_newl2_chunk_20260105_070123/l3_bc_epoch13.pt`
  - `training_runs/l3_bc_newl2_chunk_20260105_070123/l3_bc_epoch13.ckpt.pt`
  - `training_runs/l3_bc_newl2_chunk_20260105_070123/l3_bc.pt`

## 未完成/中断的训练

### L3（旧 L2 回填）——历史尝试
- 共同目标：使用旧 L2 模型回填 `goal_hat`，训练 L3 BC（残差 + 接受/拒绝/结束）。
- 旧 L2 权重：`training_runs/l2_only_cuda_20251231_010729/l2_bc.pt`
- 试运行记录（均为中断/未完成）：
  - `training_runs/l3_bc_oldl2_20251231_222743`（CPU 推理阶段极慢）
    - 日志：`diagnose_logs/l3_bc_oldl2_20251231_222743.out`
  - `training_runs/l3_bc_oldl2_gpu_20260101_003044`（首次启用 GPU 回填）
    - 关键参数：`l2_backfill_device=cuda`，`l2_backfill_batch_size=256`
    - 日志：`diagnose_logs/l3_bc_oldl2_gpu_20260101_003044.out`
  - `training_runs/l3_bc_oldl2_gpu_20260101_205647` / `..._retry_20260101_225951`（GPU 回填尝试与重试）
    - 日志：`diagnose_logs/l3_bc_oldl2_gpu_20260101_205647.out`、`diagnose_logs/l3_bc_oldl2_gpu_20260101_205647_retry_20260101_225951.out`
  - `training_runs/l3_bc_oldl2_gpu_pre_tensorize_bs64_20260103_131041`（引入预张量化）
    - 日志：`diagnose_logs/l3_bc_oldl2_gpu_pre_tensorize_bs64_20260103_131041.out`
  - `training_runs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_165146`（回填批处理初版）
    - 日志：`diagnose_logs/l3_bc_oldl2_gpu_batchbackfill_bs64_20260103_165146.out`

## 计划中的训练

### 新 L2（改损失）
- 目标：使用更新后的 loss（Q log1p + 权重 2）重新训练 L2。
- 计划参数：`--l2-q-transform log1p --l2-q-weight 2 --l2-batch-size 512 --device cuda`
- 训练完成后将产出新 L2 权重，用于下一阶段 L3 回填。

### 新 L3（基于新 L2 回填）
- 目标：使用新 L2 权重回填 `goal_hat`，重新训练 L3，和当前旧 L2 回填模型对比效果。
- 计划保留：预张量化、回填批处理、多进程解析与 GPU 回填。

### AWR / PPO / 蒸馏
- AWR：基于 L3 BC 权重继续进行 Advantage-Weighted Regression。
- PPO：在线训练，需配合比赛跑数据与稳定评估。
- 蒸馏：将启发式或多模型输出蒸馏到 L4 或紧凑模型。

## 评估脚本说明
- 脚本：`scripts/eval_l2_val_error.py`
- 作用：读取 `world_split.json` 的指定 split，计算 L2 预测在原始量纲下的 Q/P 误差（MAE/RMSE/mean_target/mean_rel_err），并按 `x_role` 掩码过滤不可谈判分量。
- 用法示例：
  ```bash
  PYTHONPATH=/home/ecs-user/LitaAgent /home/ecs-user/LitaAgent/venv/bin/python scripts/eval_l2_val_error.py \
    --output-dir training_runs/l2_qlog1p_w2_20251231_220338 \
    --split val \
    --agent-name PenguinAgent \
    --num-workers 32 \
    --batch-size 1024
  ```
- 备注：`--output-dir` 必须包含 `world_split.json` 与 `l2_bc.pt`；`--split` 可选 `train/val/test`。

## 附录：数据尺度参考
- 目标价格（target_price）抽样统计（300 个 tracker 文件）：
  - 样本数：53,567
  - min=1, max=56, mean=16.189, std=8.5425
  - 分位数：p01=1.0, p05=4.2, p25=11, p50=14, p75=20, p95=34, p99=44
