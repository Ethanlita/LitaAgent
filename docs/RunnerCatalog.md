## 运行脚本一览

| 文件 | 参与代理 | 是否可续跑 | 赛道 | 默认规模 | 参数 | 自动启动可视化 | 自动归集数据 |
|---|---|---|---|---|---|---|---|
| `runners/run_default_std.py` | `LitaAgentY/YR/CIR/N/P`（动态 Tracked）<br>2025 Std Top5 + 2024 Std Top5（动态 Tracked） | 否 | Std | n_configs=20, n_runs=2, n_competitors_per_world=全部<br>max_worlds_per_config=全部（=n_per_world）<br>n_steps=(50,200)<br>输出: `tournament_history/std_default_<timestamp>/`（含 `tracker_logs/agent_*.json`、`tournament_results.json`） | `--configs --runs --max-top-2025 --max-top-2024 --n-competitors-per-world --max-worlds-per-config --target-worlds --round-robin --quick --output-dir --parallelism --quiet/--verbose` | 否 | 是 |
| `runners/run_std_quick.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025 前 5 代理<br>不足时补 `RandomStdAgent/GreedyStdAgent/SyncRandomStdAgent` | 否 | Std | n_configs=3, n_runs=1, n_steps=50<br>max_worlds_per_config=10 | `--output-dir --port --no-server` | 默认是（`--no-server` 关闭） | 是 |
| `runners/run_std_full.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025 前 5 代理 | 否 | Std | n_configs=20, n_runs=2, n_steps=(50,200) | `--output-dir --port --no-server` | 默认是（`--no-server` 关闭） | 是 |
| `runners/run_std_full_tracked.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025 前 8 代理 | 否 | Std | n_configs=20, n_runs=2, n_steps=(50,200) | `--output-dir --port --no-server` | 默认是（`--no-server` 关闭） | 是 |
| `runners/run_std_medium_tracked.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025 前 6 代理（可用 `--max-top`） | 否 | Std | n_configs=5, n_runs=1, n_steps=(50,100) | `--output-dir --port --no-server --max-top` | 默认是（`--no-server` 关闭） | 是 |
| `runners/run_std_full_tracked_penguin.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>`PenguinAgent`<br>2025 前 8 代理 | 否 | Std | n_configs=20, n_runs=2, n_steps=(50,200) | `--configs --runs --max-top --output-dir --port --no-server` | 默认是（`--no-server` 关闭） | 是 |
| `runners/run_std_full_tracked_penguin_logs.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>`PenguinAgent`<br>2025 前 8 代理 + `RandomStdAgent` | 否 | Std | n_configs=20, n_runs=2, n_steps=(50,200)<br>forced_logs=1.0 | `--configs --runs --max-top --output-dir` | 否 | 是 |
| `runners/run_std_full_tracked_penguin_logs_resumable.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>`PenguinAgent`（Tracked）<br>2025 前 8 代理（尽量 Tracked） + `RandomStdAgent` | 是 | Std | n_configs=20, n_runs=2, n_steps=(50,200)<br>forced_logs=1.0 | `--configs --runs --max-top --parallelism --output-dir --no-postprocess` | 否 | 默认是（`--no-postprocess` 关闭） |
| `runners/run_std_full_untracked_resumable.py` | `LitaAgentY/YR/CIR/N/P`（未追踪）<br>`PenguinAgent`<br>2025 前 8 代理 + `RandomStdAgent` | 是 | Std | n_configs=20, n_runs=2, n_steps=(50,200)<br>forced_logs=1.0 | `--configs --runs --max-top --parallelism --output-dir --no-postprocess` | 否 | 默认是（`--no-postprocess` 关闭） |
| `runners/run_oneshot_quick.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025 OneShot 前 5 代理 | 否 | OneShot | n_configs=3, n_runs=1, n_steps=20 | `--output-dir --port --no-server` | 默认是（`--no-server` 关闭） | 是 |
| `runners/run_oneshot_full.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025 OneShot 前 5 代理 | 否 | OneShot | n_configs=20, n_runs=2, n_steps=(50,200) | `--output-dir --port --no-server` | 默认是（`--no-server` 关闭） | 是 |
| `runners/run_full_std_tournament.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025/2024/2023 Std 前 5 代理<br>`GreedyStdAgent/RandomStdAgent/SyncRandomStdAgent` | 否 | Std | 单 world，n_steps=50 | `--n-steps --output-dir` | 是 | 是（import_tournament） |
| `runners/run_full_tournament.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>2025/2024/2023 Std/OneShot 前 5 代理<br>`Greedy/Random/SyncRandom`（Std/OneShot） + `RandDistOneShotAgent` | 否 | OneShot/Std（`--track` 控制） | 单 world，n_steps=20 | `--track oneshot/std/both --n-steps --output-dir` | 是 | 是（import_tournament） |
| `runners/run_scml_analyzer.py` | `LitaAgentY/LitaAgentYR/LitaAgentN`（Tracked）<br>`PenguinAgent` + `AS0` | 否 | Std | quick: n_configs=2, n_runs=1, n_steps=30<br>standard: 5x2x50<br>full: 10x2x100 | `--mode quick/standard/full --visualize <log_dir> --auto-visualize --no-browser --silent --port` | 可选（`--auto-visualize`/`--visualize`） | 可选（配合可视化导入） |
| `runners/hrl_data_runner.py` | `LitaAgentY/YR/CIR/N/P`（动态 Tracked）<br>2025 Std Top5 + 2024 Std Top5 + `RandomStdAgent/SyncRandomStdAgent`（动态 Tracked） | 是（`--resumable`） | Std | n_configs=20, n_runs=2, n_competitors_per_world=全部<br>max_worlds_per_config=全部（=n_per_world）<br>n_steps=(50,200)<br>输出: `tournament_history/hrl_data_<timestamp>_std/`（含 `tracker_logs/agent_*.json`、`tournament_run.log`） | `--configs --runs --max-top-2025 --max-top-2024 --n-competitors-per-world --max-worlds-per-config --target-worlds --round-robin --steps --output-dir --parallelism --foreground --quiet --no-auto-collect --resumable --no-csv` | 否 | 默认是（`--no-auto-collect` 关闭） |
| `runners/run_batched_hrl_logs.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>`PenguinAgent`（Tracked）<br>2025 前 8 代理（尽量 Tracked） + `RandomStdAgent` | 是 | Std | 自适应（CPU 决定 configs/runs/steps/batches） | `--batches --configs --runs --steps-min --steps-max --parallelism --max-top --output-base` | 否 | 否 |
| `runners/run_manifest_hrl_logs.py` | `LitaAgentY/YR/CIR/N/P`（Tracked）<br>`PenguinAgent`（Tracked）<br>2025 前 8 代理（尽量 Tracked） + `RandomStdAgent` | 是 | Std | 每 world: n_configs=1, n_runs=1, steps=100<br>min_comp=5（默认 `--generate=0`） | `--manifest --generate --steps --min-comp --max-comp --max-top --seed-base --parallelism --output-base --no-run` | 否 | 否 |
| `runners/SCML_quick_test.py` | `LitaAgentY/LitaAgentYR` + `AS0` + `PenguinAgent` | 否 | Std | n_worlds=2, n_steps=30 | 无 | 是 | 是（import_tournament） |
| `runners/SCML_small_test_tournament.py` | `LitaAgentY/YR/N/P/CIR` + `AS0` + `PenguinAgent` | 否 | Std | n_configs=3, n_runs=1, n_steps=50 | 无 | 是 | 是（import_tournament） |
| `SCML_2025_tourment_runner.py` | `SyncRandomStdAgent` + `RandDistOneShotAgent` + `GreedyOneShotAgent` + `RandomStdAgent` + `LitaAgentCIR` + `LitaAgentY` | 否 | Std | 单 world，n_steps=50 | 无 | 否（仅 matplotlib 绘图） | 否 |

说明：
- “自动归集数据”指调用 `scml_analyzer.postprocess.postprocess_tournament` 或 `scml_analyzer.history.import_tournament`。
- “自动启动可视化”指启动 `scml_analyzer.visualizer.start_server`，`matplotlib` 绘图不计入。
- 标注“未显式设置”的步数来自 `scml.utils` 默认值（当前为 `(50,200)`）。
- 参与代理可能受可选依赖影响（如 `scml_agents`、`stable_baselines3`）。

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

## Resumable runners（可断点续跑）
- `runners/run_std_full_tracked_penguin_logs_resumable.py`  
  用法：`python runners/run_std_full_tracked_penguin_logs_resumable.py --output-dir <目录> [--configs 20 --runs 2 --max-top 8 --parallelism parallel]`  
  特点：官方 std 全规模，Lita tracker + Penguin + Top Agents，强制谈判日志，loky 并行，使用相同 `--output-dir` 重跑即可续跑（已完成 world 跳过，自动识别 *-stage-0001）。
- `runners/run_std_full_untracked_resumable.py`  
  用法：`python runners/run_std_full_untracked_resumable.py --output-dir <目录> [--configs 20 --runs 2 --max-top 8 --parallelism parallel]`  
  特点：官方 std 全规模，未追踪 Lita + Penguin + 全部 Top Agents，强制谈判日志，loky 并行，使用相同 `--output-dir` 重跑即可续跑（已完成 world 跳过，自动识别 *-stage-0001）。

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
