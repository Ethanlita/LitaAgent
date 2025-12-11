# LitaAgent-HRL 快速指引（模式 B）

## 目录
1. 准备环境
2. 数据采集与预处理
3. 训练 L2/L3 占位模型
4. 模型加载与集成
5. 后续扩展（L4/自博弈/真实 RL）

## 1. 准备环境
- Python 3.10/3.11，安装 `scml`, `scml-agents`, `negmas`, `pandas`, `numpy`。
- 代码入口：`litaagent_std/hrl_x/agent.py`（提供 Tracked 版本）。

## 2. 数据采集与预处理
1) 运行锦标赛（含冠军代理）并开启日志：
```python
# 示例：negmas.tournaments.run_tournament(...)
# 关键参数：log_negotiations=True, log_ufuns=True, save_path=./scml_logs
```
2) 解析日志生成数据集：
```python
from litaagent_std.hrl_x.data_pipeline import (
    load_negotiation_csv, build_macro_dataset, build_micro_dataset
)
df = load_negotiation_csv("./scml_logs")
macro_ds = build_macro_dataset(df)   # 日级标签 -> L2
micro_ds = build_micro_dataset(df)   # 轮级序列 -> L3
```

## 3. 训练 L2/L3 占位模型
> 当前为可运行骨架，后续可替换为 torch/tf 模型。
```python
from litaagent_std.hrl_x.training import TrainConfig, train_l2_bc, train_l3_bc, save_model

cfg = TrainConfig(lr=1e-3, epochs=10, batch_size=32)
l2_model = train_l2_bc(macro_ds, cfg)
l3_model = train_l3_bc(micro_ds, cfg)
save_model(l2_model, "./out/l2_baseline.npz")
save_model(l3_model, "./out/l3_baseline.npz")
```

## 4. 模型加载与集成
- 目前 `agent.py` 中 L2 为启发式，L3 残差为 0。接入模型时：
  - 替换 `_heuristic_manager` 调用：加载 L2 模型 `predict`，用日级特征生成目标。
  - 在 `propose/respond` 中，将 L1 基准结果加上 L3 残差模型输出，再经过 `clip_offer`。
  - 所有最终动作必须经过 L1 安全裁剪。

## 5. 后续扩展
- L4：加入注意力权重调制并发线程激进度。
- 在线微调：采用 MAPPO/分层 PPO，奖励含势能/风险/流动性/对齐项。
- 对手池自博弈：周期性快照代理，混合冠军与自我版本提高鲁棒性。
