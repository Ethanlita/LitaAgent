# SCML Parallel 模式死锁问题调查报告

**日期**: 2025年11月29日  
**状态**: 调查中  
**影响**: Windows 平台上使用 `parallelism="parallel"` 运行 SCML 锦标赛时会卡死

---

## 1. 问题描述

### 1.1 现象

在 Windows 平台上运行 SCML 2024/2025 锦标赛时，使用 `parallel` 模式会导致程序卡死：

- **CPU 使用率降到 0%** - 不是计算慢，而是真正的死锁/等待状态
- **进度条停止在固定位置** - 相同配置下，每次都在相同进度卡死
- **Serial 模式完全正常** - 只有 Parallel 模式有问题

### 1.2 卡死位置的规律

| 测试配置 | 卡死进度 |
|---------|---------|
| 4 agents, n_configs=1 | 40% |
| 10 agents (非Tracked), n_configs=2 | 17% |
| 12 agents (Tracked), n_configs=2 | 4-12% |
| 9 agents | 63% |

**关键发现**: 卡死位置是**确定性的**，相同配置每次都在相同位置卡死。

### 1.3 环境信息

- **操作系统**: Windows 11
- **Python**: 3.12
- **CPU**: 16 核
- **SCML 版本**: 最新版（使用 `anac2024_std` API）
- **NegMas 版本**: 最新版

---

## 2. 调查过程

### 2.1 测试 1: 基础 Multiprocessing 机制

**测试文件**: `test_mp_minimal.py`

**测试方法**:
```python
from multiprocessing import Pool
from scml.std import SCML2024StdWorld

def run_single_world(config):
    world = SCML2024StdWorld(**config, construct_graphs=False)
    world.run()
    return ("success", world.current_step, world.name)

# 测试 Pool(4) 运行 4 个 worlds
with Pool(4) as pool:
    results = pool.map(run_single_world, configs)
```

**结果**: ✅ **完全正常**
- Pool(1): 成功，11.3秒
- Pool(2): 成功，14.2秒
- Pool(4): 成功，14.8秒

**结论**: 基础的 `multiprocessing.Pool` 没有问题。

---

### 2.2 测试 2: ProcessPoolExecutor

**测试文件**: `test_executor.py`

**测试方法**:
```python
from concurrent import futures

with futures.ProcessPoolExecutor(max_workers=4) as executor:
    future_results = [executor.submit(run_single_world, cfg) for cfg in configs]
    for future in futures.as_completed(future_results):
        result = future.result(timeout=60)
```

**结果**: ✅ **完全正常**
- 4 workers, 4 tasks: 成功，16.6秒
- 4 workers, 8 tasks: 成功，18.5秒

**结论**: `ProcessPoolExecutor` + `as_completed` 本身没有问题。

---

### 2.3 测试 3: Agent 类的 Pickle 序列化

**测试文件**: `test_pickle.py`

**测试方法**:
```python
import pickle

# 测试每个 Agent 类是否可以 pickle
for agent_class in all_agents:
    data = pickle.dumps(agent_class)
    pickle.loads(data)
```

**结果**: ✅ **所有 Agent 类都可以正常 pickle**
- LitaAgentY: 55 bytes
- LitaAgentYTracked: 62 bytes
- AX, CautiousStdAgent, DogAgent 等: 全部成功

**结论**: Agent 类的序列化没有问题。

---

### 2.4 测试 4: NegMas 传递给子进程的对象

**测试文件**: `test_pickle_negmas.py`

**测试方法**:
```python
# 测试 World 配置和 Generator 函数
config = SCML2024StdWorld.generate(agent_types=agents, n_steps=10)
pickle.dumps(config)  # 测试配置
pickle.dumps(anac2024_std_world_generator)  # 测试函数
```

**结果**: ✅ **全部正常**
- config (整个配置): 10,622 bytes
- anac2024_std_world_generator: 103 bytes
- balance_calculator_std: 56 bytes

**结论**: NegMas 传递给子进程的对象都可以正确序列化。

---

### 2.5 测试 5: Agent 实例的 Pickle

**测试文件**: `test_pickle_instance.py`

**测试方法**:
```python
world = SCML2024StdWorld(**config)
for agent in world.agents.values():
    pickle.dumps(agent)  # 测试实例
```

**结果**: ❌ **失败 - RecursionError**
```
RecursionError: maximum recursion depth exceeded
```

**发现**: Agent 实例包含循环引用（agent → world → agent），无法直接 pickle。

**但这不是问题原因**: NegMas 传递的是配置字典，不是 Agent 实例。Agent 实例在子进程中重新创建。

---

### 2.6 测试 6: 子进程中 Import Agents

**测试文件**: `diagnose_spawn.py`

**测试方法**:
```python
from multiprocessing import Process, Queue

def worker_import_test(queue, agent_module, agent_name):
    module = __import__(agent_module, fromlist=[agent_name])
    agent_class = getattr(module, agent_name)
    queue.put(("success", agent_name))

# 在子进程中测试 import
p = Process(target=worker_import_test, args=(queue, module, name))
p.start()
p.join(timeout=30)
```

**结果**: ✅ **所有 Agent 都可以在子进程中正常 import**
- LitaAgentYTracked: 2.08秒
- LitaAgentNTracked: 9.17秒（较慢但成功）
- AX, CautiousStdAgent 等: 全部成功

**结论**: 子进程中的模块导入没有问题。

---

### 2.7 测试 7: 隔离测试每对 Agents

**测试文件**: `diagnose_deep.py` (测试 4)

**测试方法**:
```python
# 逐一测试每个 agent 与基准 agent 的组合
for agent in test_agents:
    results = anac2024_std(
        competitors=[base, agent],
        n_configs=2,
        parallelism="parallel",
    )
```

**结果**: ✅ **所有单独的 agent 对都正常完成**

**结论**: 问题不是某个特定 Agent 导致的。

---

### 2.8 测试 8: 非 Tracked 版本的 Agents

**测试文件**: `test_non_tracked_large.py`

**测试方法**:
```python
# 使用不带 TrackerMixin 的原始 Agent
all_agents = [
    LitaAgentY,  # 不是 LitaAgentYTracked
    LitaAgentYR,
    LitaAgentN,
    ...
]

results = anac2024_std(
    competitors=all_agents,
    n_configs=2,
    parallelism="parallel:0.75",
)
```

**结果**: ❌ **仍然卡死（在 17% 位置）**

**结论**: **问题不在 TrackerMixin 的线程锁序列化上**。

---

### 2.9 测试 9: Dask Distributed 模式

**测试文件**: `test_alternatives.py`, `test_dask_full.py`

**测试方法**:
```python
from dask.distributed import Client

client = Client(n_workers=4)
results = anac2024_std(
    competitors=agents,
    parallelism="distributed",
)
```

**结果**: 
- 4 agents: ✅ 成功，17.37秒
- 12 agents: ❌ 出现内存错误
  ```
  Unable to allocate 3.84 EiB for an array with shape (4428796755203867975,)
  ```

**发现**: Dask 模式出现数据损坏，尝试分配不可能的内存大小，说明序列化/反序列化过程中有问题。

---

### 2.10 测试 10: 渐进式增加 Agents 数量

**测试文件**: `test_progressive.py`, `test_progressive2.py`

**测试方法**:
```python
# 从 2 个 agents 开始，逐步增加到 12 个
for n in range(2, 13):
    agents = ALL_AGENTS[:n]
    results = anac2024_std(competitors=agents, ...)
```

**结果**:
| Agents 数量 | 结果 | 耗时 |
|------------|------|------|
| 2 | ✅ 成功 | 15.7s |
| 3 | ✅ 成功 | 15.9s |
| 4 | ✅ 成功 | 24.2s |
| 5 | ✅ 成功 | 38.5s |
| 6 | ✅ 成功 | 53.4s |
| 7 | ✅ 成功 | 80.4s |
| 8 | ✅ 成功 | 66.7s |
| 9 | ❌ 卡死 | - |

**发现**: 问题在 9 个 agents 时开始出现，但这可能与 world 组合数量有关，而不是 agent 数量本身。

---

## 3. 已排除的问题

| 可能原因 | 状态 | 证据 |
|---------|------|------|
| multiprocessing.Pool 问题 | ❌ 已排除 | 测试 1 完全正常 |
| ProcessPoolExecutor 问题 | ❌ 已排除 | 测试 2 完全正常 |
| Agent 类 pickle 问题 | ❌ 已排除 | 测试 3 全部成功 |
| NegMas 参数 pickle 问题 | ❌ 已排除 | 测试 4 全部成功 |
| 子进程 import 问题 | ❌ 已排除 | 测试 6 全部成功 |
| 特定 Agent 的 bug | ❌ 已排除 | 测试 7 所有组合正常 |
| TrackerMixin 线程锁问题 | ❌ 已排除 | 测试 8 非 Tracked 版本也卡死 |
| Worker 数量太多 | ❌ 已排除 | 0.25 和 0.75 都会卡死 |

---

## 4. 关键发现

### 4.1 确定性死锁

死锁位置是**确定性的** - 相同配置每次都在相同进度卡死。这意味着：
- 不是随机的竞态条件
- 不是 Agent 的随机行为导致
- 很可能是 NegMas/SCML 内部的某个确定性逻辑问题

### 4.2 问题层级

```
✅ multiprocessing (底层) - 正常
✅ ProcessPoolExecutor (中层) - 正常  
✅ 我们的代码 (Agent/Tracker) - 正常
❌ NegMas tournament() (上层) - 有问题
```

问题出在 **NegMas 的 `tournament()` 函数** 或其调用的内部函数中。

### 4.3 Serial vs Parallel

- **Serial 模式**: 永远正常，任何配置都能完成
- **Parallel 模式**: 在足够多的 world 组合时会死锁

---

## 5. 可能的根本原因（待验证）

### 5.1 NegMas 的 `_run_parallel` 函数

位置: `negmas/tournaments/tournaments.py`

```python
for i, future in track(enumerate(as_completed(future_results)), ...):
    result = future.result(timeout=timeout)
```

`futures.as_completed()` 本身没有全局超时机制。如果某个子进程卡死，整个循环会无限等待。

### 5.2 可能的死锁点

1. **World 运行中的某个步骤** - 特定的 world 配置在特定步骤卡住
2. **谈判机制** - NegMas 的谈判可能在某些条件下无限等待
3. **资源竞争** - 多个 world 同时访问某些共享资源

---

## 6. 下一步计划

### 6.1 短期方案

1. **使用 Serial 模式** - 虽然慢但可靠
2. **减少 n_configs** - 减少 world 组合数量

### 6.2 进一步调查

1. **在 NegMas 代码中加日志** - 确定具体是哪个 world/step 导致卡死
2. **检查 NegMas GitHub issues** - 搜索类似的 Windows parallel 问题
3. **向 NegMas 提交 issue** - 报告这个 bug

### 6.3 长期方案

1. **等待 NegMas 修复**
2. **实现自己的并行执行逻辑** - 绕过 NegMas 的 tournament 函数

---

## 7. 相关文件

| 文件 | 用途 |
|------|------|
| `test_mp_minimal.py` | 测试基础 multiprocessing |
| `test_executor.py` | 测试 ProcessPoolExecutor |
| `test_pickle.py` | 测试 Agent 类 pickle |
| `test_pickle_instance.py` | 测试 Agent 实例 pickle |
| `test_pickle_negmas.py` | 测试 NegMas 参数 pickle |
| `diagnose_spawn.py` | 测试子进程 import |
| `diagnose_deep.py` | 综合诊断测试 |
| `test_non_tracked_large.py` | 测试非 Tracked agents |
| `test_progressive.py` | 渐进式增加 agents |
| `test_alternatives.py` | 测试 Dask 替代方案 |

---

## 8. 参考资料

- SCML 2025 官方文档: `scml2025.pdf`, `overview2025.pdf`
- NegMas 源码: `.venv/Lib/site-packages/negmas/tournaments/tournaments.py`
- SCML 源码: `.venv/Lib/site-packages/scml/utils.py`

---

**文档维护者**: GitHub Copilot  
**最后更新**: 2025年12月1日

---

## 9. 第二阶段调查：Linux 环境复现 (2025-11-29)

### 9.1 环境信息

问题在 Linux (Ubuntu) 环境下同样复现，证明**不是 Windows 特有问题**。

- **操作系统**: Linux (Ubuntu)
- **Python**: 3.12
- **SCML 版本**: 0.7.3
- **NegMas 版本**: 0.10.21

### 9.2 详细监控数据

通过 `diagnose_deep.py` 脚本进行深入监控：

**配置**：
- 9 个 Agents (5 LitaAgent + 1 TopAgent + 3 内置Agent)
- `n_configs=3`, `n_steps=50`
- `parallelism='parallel'`, `verbose=False`
- 无 `max_worlds_per_config` 限制 → 生成 756 个 worlds

**时间线**：
```
20:45:xx  开始运行，32个工作进程启动
20:45-21:02  进度正常推进，子进程数量保持在30+
21:02:26  工作进程数量骤降到只剩 resource_tracker (1个)
21:02:26 - 21:41:xx  主进程空等，CPU使用率接近0，系统负载降到接近0
```

### 9.3 Future 状态追踪

使用 `diagnose_futures.py` 脚本 Monkey-patch `as_completed()` 进行监控：

**Future 状态监控**：
```
[22:13:18] as_completed yielded future 320/756 after 910.9s, status=success
[22:13:27] [Monitor 920s] Total=756 Done=320 Running=33 Pending=436 Cancelled=0
... (状态停止变化，持续8分钟以上)
[22:21:37] [Monitor 1410s] Total=756 Done=320 Running=33 Pending=436 Cancelled=0
```

**进程状态** (挂起时)：
```bash
$ ps -ef | grep python
# 只有主进程和 resource_tracker
# 没有任何工作子进程

$ pstree -p 105310
python(105310)─┬─python(105566)    # resource_tracker
               ├─{python}(105314)  # 主进程的线程池 (69个线程)
               └─...
```

### 9.4 堆栈跟踪分析

通过 `kill -USR1` 获取的堆栈跟踪：

```
Thread QueueFeederThread:
  File 'multiprocessing/connection.py', line 384, in _send
    n = write(self._handle, buf)
  # ⚠️ 卡在 write() - 管道另一端已关闭

Thread Thread-1 (ProcessPoolExecutor 管理线程):
  File 'concurrent/futures/process.py', line 426, in wait_result_broken_or_wakeup
    ready = mp.connection.wait(readers + worker_sentinels)
  # ⚠️ 等待已退出的 worker

Thread MainThread:
  File 'negmas/tournaments/tournaments.py', line 1395, in _run_parallel
    for i, future in track(enumerate(as_completed(future_results)), ...)
  File 'concurrent/futures/_base.py', line 243, in as_completed
    waiter.event.wait(wait_timeout)
  # ⚠️ 卡在 as_completed() - 等待永远不会完成的 futures
```

### 9.5 根因确认

| 问题 | 答案 |
|------|------|
| 1. 真的有未完成的 future 吗？ | ✅ 是的，469 个未完成 (436 Pending + 33 Running) |
| 2. 子进程都结束了，为什么 future 未完成？ | ProcessPoolExecutor 没有正确检测到 worker 退出 |
| 3. 主进程真的卡在等待 Future 吗？ | ✅ 是的，堆栈确认卡在 `as_completed()` |
| 4. timeout 为什么没效果？ | negmas 没有给 `as_completed()` 传 timeout 参数 |

### 9.6 negmas 源码问题

问题代码位于 `negmas/tournaments/tournaments.py`:

```python
# Line 1395 - _run_parallel 函数
for i, future in track(
    enumerate(as_completed(future_results)),  # ⚠️ 没有 timeout 参数！
    total=n_world_configs,
    description="Simulating ...",
):
    if total_timeout is not None and time.perf_counter() - strt > total_timeout:
        break  # ⚠️ 这行永远执行不到，因为 as_completed 已经阻塞了
```

---

## 10. 第三阶段调查：排除 scml_analyzer 影响 (2025-12-01)

### 10.1 环境版本检查

**SCML 官方要求** (来自 scml2025.web.app)：
> "We only support python 3.10 and 3.11. The reason python 3.12 is not yet supported is that stable_baselines3 is not supporting it yet."

**当前环境**：
- Python 版本：3.12 ⚠️ (官方不推荐)
- scml 版本：0.7.3
- negmas 版本：0.10.21

### 10.2 干净运行测试

为排除 `scml_analyzer` 模块导致问题的可能性，创建了不加载任何自定义代码的测试脚本。

**测试脚本**: `test_clean_run.py`
- 只使用 scml 内置 agents (RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent)
- 不导入任何 LitaAgent 或 scml_analyzer 代码

**小规模测试结果** (27-54 worlds)：
```
✓ 测试成功完成
✓ 没有发生挂起
⚠️ 但观察到 worker 进程异常终止的警告
```

### 10.3 待验证测试

需要进行大规模测试（756 worlds）来确认问题来源：

| 测试 | 配置 | 目的 |
|------|------|------|
| 纯内置 agents 大规模测试 | 756 worlds, 无 scml_analyzer | 确认是否是 scml_analyzer 的问题 |

### 10.4 后续排查计划

**如果纯内置 agents 大规模测试不会挂起**：
- 问题在 scml_analyzer，需要检查其多进程安全性

**如果纯内置 agents 大规模测试仍然挂起**：
1. **考虑将 Python 版本切换到 3.11** - 官方推荐版本
2. **尝试使用 `dask` 作为并行后端** - `parallelism='dask'`
3. **尝试使用 `loky` 替代 `multiprocessing`** - 更健壮的进程池实现
4. **检查 scml/negmas 是否提供配置选项** - 在不修改源码的情况下设置 `as_completed()` 的 timeout

---

## 11. 相关文件（更新）

| 文件 | 用途 |
|------|------|
| `diagnose_deep.py` | 深度监控脚本 |
| `diagnose_futures.py` | Future 状态追踪脚本 |
| `test_clean_run.py` | 不加载 scml_analyzer 的干净测试 |
| `test_clean_run_large.py` | 大规模干净测试脚本 |
| `diagnose_logs/` | 监控日志输出目录 |
