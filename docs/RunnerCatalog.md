## 运行脚本一览（Runners Overview）

| 文件 (File) | 作用 (Purpose) | 赛道/规模 (Track & Size) | 备注 (Notes) |
|-------------|----------------|--------------------------|--------------|
| `runners/run_std_full.py` | 官方尺度 std 锦标赛，覆盖 Lita+Top5；验证 loky 稳定性并保留 tracker/可视化后处理 | `track=std`, `n_configs=20`, `n_runs_per_world=2`, `n_steps=(50,200)` | loky 并行，verbose 进度条 |
| `runners/run_std_quick.py` | 快速 smoke，最少配置验证环境/依赖是否正常，观测 loky 是否能跑完一轮 | `track=std`, `n_configs=3`, `n_runs_per_world=1`, `n_steps=50`, `max_worlds_per_config=10` | loky 并行，小规模 |
| `runners/run_full_std_tournament.py` | 含历届 Top Agents 的完整 std 赛，验证多版本 agent 共存及 tracker 输出 | `track=std`, 参数可调 | loky 并行 |
| `runners/run_full_tournament.py` | 同时支持 std/oneshot 混合运行的入口，用于一次性生成全套结果 | 依内部参数 | loky 并行 |
| `runners/run_oneshot_full.py` | 完整 oneshot 锦标赛，测试 Lita 变体与 Top5 的对局稳定性 | `track=oneshot`, `n_configs=10`, `n_runs_per_world=2`, `n_steps=50` | loky 并行 |
| `runners/run_oneshot_quick.py` | oneshot 快速自检，确保最小配置也能跑通并输出日志 | `track=oneshot`, `n_configs=3`, `n_runs_per_world=1`, `n_steps=50`, `max_worlds_per_config=10` | loky 并行 |
| `runners/run_scml_analyzer.py` | 集成 tracker + postprocess + visualizer 的多模式入口，适合日常分析 | 根据 `--mode` (standard/full/oneshot) | loky 并行 |
| `runners/SCML_quick_test.py` | 排除噪声 agent 的快速对比，检查得分/日志链路 | 预置少量 agents | loky 并行 |
| `runners/SCML_small_test_tournament.py` | 小规模对比赛，验证 Lita 与获奖代理的相对表现 | 预置少量 agents | loky 并行 |
| `SCML_2025_tourment_runner.py` | 官方示例式运行器，保留原有参数，便于对标官方行为 | 按内部配置 | 手工触发 |
| `runners/run_std_full_tracked.py` | 官方规模 std 赛（20 配置 ×2，步长 50-200），LitaAgents 注入 tracker，尝试加载最多 8 个 Top Agents，loky 并行+进度条 | `track=std`, `n_configs=20`, `n_runs_per_world=2`, `n_steps=(50,200)` | tracker_logs 写入输出目录 |
| `runners/run_std_full_tracked_penguin.py` | 官方规模 std 赛，包含 PenguinAgent，LitaAgents tracker 版 + 最多 8 个 Top Agents，loky 并行 + 进度 | `track=std`, `n_configs=20`, `n_runs_per_world=2`, `n_steps=(50,200)` | tracker_logs 写入输出目录 |
| `runners/run_std_medium_tracked.py` | 中等规模 std 赛（5 配置 ×1，步长 50-100），LitaAgents 注入 tracker，加载最多 6 个 Top Agents，loky 并行+进度条 | `track=std`, `n_configs=5`, `n_runs_per_world=1`, `n_steps=(50,100)` | tracker_logs 写入输出目录 |

说明 (Notes):
- 所有 runner 已默认启用 `loky` 执行器（通过 `runners.loky_patch.enable_loky_executor()`），可用环境变量 `SCML_PARALLELISM=loky[:fraction]` 调节并发。
- 运行后需告知用户输出路径，等待确认再做后续处理（详见 docs/Agents.md）。***

## 诊断/测试脚本（Diagnostic / Test Scripts）
以下脚本也会启动比赛（多为 Standard，用于复现/调试）。默认未强制 loky；如需 loky，可设置 `SCML_PARALLELISM=loky` 再运行。

| 文件 (File) | 作用 (Purpose) | 赛道/规模 (Track & Size) |
|-------------|----------------|--------------------------|
| `diagnose_deep.py` | 全量追踪（worker/future/进程监控）复现挂死，用于收集日志/栈 | Standard，n_configs=3，n_steps=50 |
| `diagnose_exact.py` | 复现特定挂死点，聚焦单一配置，便于 gdb/strace | Standard |
| `diagnose_futures.py` | 在父进程记录 futures 完成/异常，定位丢信号问题 | Standard |
| `diagnose_isolate.py` | 只跑单个 world，剥离并行因素，定位单局异常 | Standard |
| `diagnose_parallel.py` / `_full.py` / `_hang.py` | 分别测试并行、全量并行及特定挂起路径 | Standard |
| `diagnose_progressive.py` | 逐步放大规模观察并行饱和与崩溃点 | Standard |
| `diagnose_reproduce.py` | 最小步骤快速复现挂起，便于反复采样 | Standard |
| `diagnose_workers.py` | 记录子进程生命周期与输出，核对 worker 是否正常退出 | Standard |
| `reproduce_deadlock.py` | 最小复现挂死用例 | Standard |
| `test_non_tracked.py` / `_large.py` | 去除 tracker 的 std 基准，确认问题是否由追踪器引入 | Standard |
| `test_dask_full.py` | 换 Dask 后端的大规模并行验证，观察分布式序列化问题 | Standard |
| `test_parallel_quick.py` / `_injected.py` | 小规模并行与故障注入测试，验证 as_completed 行为 | Standard |
| `test_parallel_step.py` | OneShot 并行 step 执行测试 | OneShot |
| `test_progressive.py` / `test_progressive2.py` | 渐进放大 std 规模，观察资源/稳定性拐点 | Standard |
| `test_alternatives.py` | 评估不同并行后端/参数的效果与稳定性 | Standard |
| `test_isolate_p.py` | 进程隔离/启动方式测试，排查 spawn/fork 差异 | Standard |
| `test_std_with_top_agents.py` | 含 Top Agents 的标准赛基准，验证兼容性和分数 | Standard |
| `test_tournament_direct.py` | 直接调用 config/generator 的 tournament，验证管线完整性 | Standard |
| `litaagent_std/helpers/runner.py` | std/oneshot 通用封装 | Standard/OneShot |
| `litaagent_std/team_miyajima_oneshot/helpers/runner.py` | OneShot/Standard helper | Standard/OneShot |

## 示例与 Agent 自检（Examples & Agent Self-tests）
这些脚本通常运行小规模示例或 agent 行为测试，可能启动简化比赛或单局模拟。

| 文件 (File) | 作用 (Purpose) | 说明 (Notes) |
|-------------|----------------|--------------|
| `examples/run_std_example.py` | Standard 示例比赛 | 官方 demo 规模 |
| `examples/run_oneshot_example.py` | OneShot 示例比赛 | 官方 demo 规模 |
| `litaagent_std/agent_logger.py` | Agent 日志示例，演示如何记录交互，可能触发小型仿真 | 可能运行小型交互 |
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
| `start_visualizer.py` | 一键启动可视化服务器，默认监听 0.0.0.0:8081（可用 VIS_HOST/VIS_PORT 覆盖） |
| 静态报告 (report.html) | `python -m scml_analyzer.visualizer --static <tournament_id>` 生成；也可在 Python 中 `generate_static_report(tid)`；生成后文件保存在对应比赛目录下并可复用 |

## 其它仿真/最小复现脚本（Misc simulations & minimal repro）
| 文件 (File) | 作用 (Purpose) | 说明 (Notes) |
|-------------|----------------|--------------|
| `test_mp_minimal.py` | 使用 multiprocessing 直接跑 SCML2024StdWorld（spawn） | 最小并行复现 |
| `test_executor.py` | 用 ProcessPoolExecutor 运行多个 std world，验证 as_completed 行为 | 4 个 world，10 步，进程池 |
| `test_tracker_debug.py` | 启用 tracker 的 OneShot 小型仿真，验证日志 | SCML2024OneShotWorld，n_steps=3 |
