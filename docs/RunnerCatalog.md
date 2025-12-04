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

## 诊断/测试脚本（Diagnostic / Test Scripts）
以下脚本也会启动比赛（多为 Standard，用于复现/调试）。默认未强制 loky；如需 loky，可设置 `SCML_PARALLELISM=loky` 再运行。

| 文件 (File) | 作用 (Purpose) | 赛道/规模 (Track & Size) |
|-------------|----------------|--------------------------|
| `diagnose_deep.py` | 深度监控并行，记录 worker/future 追踪 | Standard，n_configs=3，n_steps=50 |
| `diagnose_exact.py` | 精确复现特定卡死场景 | Standard |
| `diagnose_futures.py` | future 状态回调/监控 | Standard |
| `diagnose_isolate.py` | 单 world 隔离调试 | Standard |
| `diagnose_parallel.py` / `_full.py` / `_hang.py` | 并行行为/挂起复现 | Standard |
| `diagnose_progressive.py` | 渐进规模并行测试 | Standard |
| `diagnose_reproduce.py` | 快速复现挂起 | Standard |
| `diagnose_workers.py` | Worker 级监控与日志 | Standard |
| `reproduce_deadlock.py` | 最小复现挂死用例 | Standard |
| `test_non_tracked.py` / `_large.py` | 不使用 tracker 的基准运行 | Standard |
| `test_dask_full.py` | Dask 后端并行测试 | Standard |
| `test_parallel_quick.py` / `_injected.py` | 并行快速/注入测试 | Standard |
| `test_parallel_step.py` | OneShot 并行 step 测试 | OneShot |
| `test_progressive.py` / `test_progressive2.py` | 渐进规模 std 测试 | Standard |
| `test_alternatives.py` | 并行后端/参数替代实验 | Standard |
| `test_isolate_p.py` | 进程隔离测试 | Standard |
| `test_std_with_top_agents.py` | Top Agents 标准赛测试 | Standard |
| `test_tournament_direct.py` | 直接调用生成器/分配器的 tournament 测试 | Standard |
| `litaagent_std/helpers/runner.py` | std/oneshot 通用封装 | Standard/OneShot |
| `litaagent_std/team_miyajima_oneshot/helpers/runner.py` | OneShot/Standard helper | Standard/OneShot |

## 示例与 Agent 自检（Examples & Agent Self-tests）
这些脚本通常运行小规模示例或 agent 行为测试，可能启动简化比赛或单局模拟。

| 文件 (File) | 作用 (Purpose) | 说明 (Notes) |
|-------------|----------------|--------------|
| `examples/run_std_example.py` | Standard 示例比赛 | 官方 demo 规模 |
| `examples/run_oneshot_example.py` | OneShot 示例比赛 | 官方 demo 规模 |
| `litaagent_std/agent_logger.py` | Agent 日志示例 | 可能运行小型交互 |
| `litaagent_std/test_im_full.py` / `test_im_material_only.py` / `test_inventory_manager_cir.py` / `unit_test_agent_im_cir.py` | 库存管理/产线 Agent 自测 | 小规模仿真/单元测试 |
| `litaagent_std/inventory_manager_cir.py` / `_cirs.py` | 库存管理组件，内含 main 用于快速验证 | 非大规模比赛 |
| `litaagent_std/litaagent_*.py`（Y/YR/N/P/CIR/YS/CIRS） | Agent 定义，main 可做简单自检 | 仅单 Agent 检验 |
| `litaagent_std/team_miyajima_oneshot/*.py`（cautious/myagent/sampleagents） | OneShot Agent 示例/自测 | 小型仿真 |
| `Agent_tester_.py` / `Agent_tester_1.py` | 手动选择 agent 运行单个 Standard world（带图表/Pareto 分析） | 单局、步数可配置 |
| `P_tester_vary_ptoday.py` | 迭代不同 `_ptoday` 配置运行 SCML2024StdWorld | 多次单局仿真 |

## Analyzer/可视化工具（Analyzer & Visualization）
| 文件 (File) | 作用 (Purpose) |
|-------------|----------------|
| `scml_analyzer/analyze_failures.py` | 分析比赛失败/异常 |
| `scml_analyzer/browser.py` | 浏览比赛结果 |
| `scml_analyzer/history.py` | 操作历史记录 |
| `scml_analyzer/visualizer.py` | 启动/操作可视化界面 |

## 其它仿真/最小复现脚本（Misc simulations & minimal repro）
| 文件 (File) | 作用 (Purpose) | 说明 (Notes) |
|-------------|----------------|--------------|
| `test_mp_minimal.py` | 使用 multiprocessing 直接跑 SCML2024StdWorld（spawn） | 最小并行复现 |
| `test_executor.py` | 用 ProcessPoolExecutor 运行多个 std world，验证 as_completed 行为 | 4 个 world，10 步，进程池 |
| `test_tracker_debug.py` | 启用 tracker 的 OneShot 小型仿真，验证日志 | SCML2024OneShotWorld，n_steps=3 |
