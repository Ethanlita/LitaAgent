## 运行脚本一览（Runners Overview）

| 文件 (File) | 作用 (Purpose) | 赛道/规模 (Track & Size) | 备注 (Notes) |
|-------------|----------------|--------------------------|--------------|
| `runners/run_std_full.py` | 完整 SCML 2025 Standard 比赛 / Full Standard tournament | `track=std`, `n_configs=20`, `n_runs_per_world=2`, `n_steps=(50,200)` | 默认使用 loky 并行，现已开启 verbose 进度条 |
| `runners/run_std_quick.py` | 快速 Standard 试跑 / Quick Standard smoke test | `track=std`, `n_configs=3`, `n_runs_per_world=1`, `n_steps=50`, `max_worlds_per_config=10` | loky 并行，较小规模 |
| `runners/run_full_std_tournament.py` | 包含多年份 Top Agents 的完整 Standard / Full Standard with Top Agents | `track=std`, 参数可调 | loky 并行 |
| `runners/run_full_tournament.py` | 综合 Standard + OneShot 混合入口 / Mixed (std + oneshot) driver | 依内部参数 | loky 并行 |
| `runners/run_oneshot_full.py` | 完整 OneShot 比赛 / Full OneShot tournament | `track=oneshot`, `n_configs=10`, `n_runs_per_world=2`, `n_steps=50` | loky 并行 |
| `runners/run_oneshot_quick.py` | 快速 OneShot 试跑 / Quick OneShot smoke test | `track=oneshot`, `n_configs=3`, `n_runs_per_world=1`, `n_steps=50`, `max_worlds_per_config=10` | loky 并行 |
| `runners/run_scml_analyzer.py` | 带 tracker/可视化的多模式入口 / Analyzer entry with tracker & visualizer | 根据 `--mode` (standard/full/oneshot) | loky 并行 |
| `runners/SCML_quick_test.py` | 小规模快速验证 / Small quick test | 预置少量 agents | loky 并行 |
| `runners/SCML_small_test_tournament.py` | 小规模对比测试 / Small comparison tournament | 预置少量 agents | loky 并行 |
| `SCML_2025_tourment_runner.py` | 官方示例式运行器 / Legacy tournament runner | 按内部配置 | 适用于手工触发 |

说明 (Notes):
- 所有 runner 已默认启用 `loky` 执行器（通过 `runners.loky_patch.enable_loky_executor()`），可用环境变量 `SCML_PARALLELISM=loky[:fraction]` 调节并发。
- 运行后需告知用户输出路径，等待确认再做后续处理（详见 docs/Agents.md）。***
