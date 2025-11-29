# SCML Parallel 模式死锁问题调查报告

**日期**: 2025年11月29日  
**状态**: 🔴 调查中（第二阶段）  
**环境**: Linux (Ubuntu) 和 Windows（均有复现，非系统环境问题）  
**影响**: 在使用 `parallelism="parallel"` 运行 SCML 锦标赛时，程序会在运行一段时间后挂起 (Hang)。

---

## 1. 问题描述

### 1.1 现象

在运行 SCML 2025 锦标赛时，程序会在运行一段时间后挂起：

**第一阶段问题（已解决）**：
- **原因**: `verbose=True` 导致 stdout buffer 溢出
- **解决**: 设置 `verbose=False`

**第二阶段问题（当前调查中）**：
即使设置了 `verbose=False`，程序仍然会挂起。具体表现为：

1. **进度条停止更新** - `rich` 的进度条停在某个百分比不再前进
2. **所有工作子进程消失** - 通过 `top` 或 `ps` 查看，只剩下1个 Python 进程（主进程 + `multiprocessing.resource_tracker`）
3. **CPU 使用率接近 0** - 系统负载从运行时的 10+ 下降到接近 0
4. **主进程空等** - 主进程持续运行但不做任何有意义的工作

### 1.2 详细监控数据

通过 `diagnose_deep.py` 脚本进行深入监控，记录了以下关键数据：

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

**进程状态监控（摘录）**：
```
[1015s] Load: 13.93 12.42 7.50 | Children: 34
[1026s] Load: 13.64 12.41 7.55 | Children: 34
...
[1708s] Load: 2.23 5.32 6.08 | Children: 2   # 工作进程开始消失
[1719s] Load: 1.82 5.06 5.97 | Children: 2
...
[3344s] Load: 0.00 0.00 1.27 | Children: 2   # 系统完全空闲
[3355s] Load: 0.00 0.00 1.18 | Children: 2   # 只剩 resource_tracker
```

### 1.3 关键发现

1. **所有 futures 已提交，所有工作进程已退出**：
   - 756 个 worlds 的任务被提交到 ProcessPoolExecutor
   - 32个工作进程在某个时间点全部正常退出
   - 但主进程中的 `as_completed()` 迭代器没有返回所有结果

2. **主进程卡在 `as_completed()` 循环中**：
   - negmas 的 `tournaments.py` 使用 `for future in as_completed(future_results)` 收集结果
   - 当某些 futures 的结果通知丢失时，`as_completed()` 会无限等待

3. **跨平台复现**：
   - 此问题在 Ubuntu 和 Windows 下均有复现
   - 排除了操作系统特定问题的可能性

### 1.4 原因分析

#### A. Stdout Buffer Overflow (第一阶段，已解决)
当 `anac2024_std` 设置为 `verbose=True` 时，大量日志导致管道缓冲区溢出。

#### B. Multiprocessing Safety (已加固)
`threading.Lock` 的 pickle 和 fork 安全问题已通过 `__getstate__`/`__setstate__` 解决。

#### C. Future 结果丢失 (第二阶段，调查中)
疑似问题：
- `ProcessPoolExecutor` 的某些 worker 可能异常退出但未正确报告错误
- `as_completed()` 等待的某些 futures 永远不会收到完成通知
- 可能与 negmas 内部的异常处理或进程间通信有关

---

## 2. 已实施的解决方案（第一阶段）

### 2.1 禁用详细日志 (关键修复)

在所有并行运行的脚本中，将 `anac2024_std` 或 `anac2024_oneshot` 的 `verbose` 参数设置为 `False`。

```python
results = anac2024_std(
    # ...
    verbose=False,  # 关键：防止 stdout buffer 溢出
    # ...
)
```

已更新的文件：
- `runners/run_std_full.py`
- `runners/run_std_quick.py`
- `runners/run_oneshot_full.py`
- `runners/run_oneshot_quick.py`
- `reproduce_deadlock.py`

### 2.2 代码加固 (Code Hardening)

为了防止未来的死锁和兼容 Spawn 模式，我们对 `scml_analyzer/auto_tracker.py` 进行了以下修复：

#### 修复 1: Pickle 支持 (针对 Spawn 模式)
在 `AgentLogger` 类中实现了 `__getstate__` 和 `__setstate__`，在序列化时排除 `_lock` 对象，在反序列化时重新创建锁。

```python
def __getstate__(self):
    state = self.__dict__.copy()
    if '_lock' in state:
        del state['_lock']  # 锁不能被 pickle
    return state

def __setstate__(self, state):
    self.__dict__.update(state)
    self._lock = threading.Lock()  # 重建锁
```

#### 修复 2: Fork 安全 (针对 Linux Fork 模式)
在 `TrackerManager` 中引入了 PID 检查机制。每次获取锁或 Logger 时，检查当前 PID 是否与创建时的 PID 一致。如果不一致（说明发生了 fork），则重置锁和 Logger，确保子进程拥有干净的状态。

```python
@classmethod
def _get_lock(cls):
    # 检测 fork：如果 PID 变化，重置锁
    if os.getpid() != cls._pid:
        cls._reset_for_new_process()
    return cls._lock
```

---

## 3. 第一阶段验证结果

使用 `reproduce_deadlock.py` 进行验证（小规模测试）：

- **配置**: 2 configs, 50 steps, `verbose=False`
- **结果**: ✅ 成功完成
- **耗时**: 约 40-50 秒

**但是**：在更大规模的测试中（如 `run_std_quick.py` 默认配置，756 个 worlds），问题仍然存在。

---

## 4. 第二阶段调查（进行中）

### 4.1 调查工具

1. **`diagnose_deep.py`** - 深度监控脚本
   - 每10秒记录子进程状态到日志文件
   - 记录系统负载、进程数量、进程状态
   - 将stdout和监控日志分别输出到不同文件

2. **`diagnose_futures.py`** - Future 状态追踪脚本
   - Monkey-patch `concurrent.futures.as_completed()` 监控每个 future 的状态
   - 每10秒报告 `Done/Running/Pending/Cancelled` 统计
   - 支持 `SIGUSR1` 信号打印所有线程的堆栈跟踪

### 4.2 关键实验结果 (2025-11-29 22:17)

使用 `diagnose_futures.py` 进行监控，在程序挂起时获得了以下数据：

**Future 状态监控**：
```
[22:13:18] as_completed yielded future 320/756 after 910.9s, status=success
[22:13:27] [Monitor 920s] Total=756 Done=320 Running=33 Pending=436 Cancelled=0
... (状态停止变化，持续2分钟以上)
[22:15:17] [Monitor 1030s] Total=756 Done=320 Running=33 Pending=436 Cancelled=0
```

**进程状态**：
```bash
$ ps -ef | grep python
# 只有主进程 (105310) 和 resource_tracker (105566)
# 没有任何工作子进程

$ pstree -p 105310
python(105310)─┬─python(105566)    # resource_tracker
               ├─{python}(105314)  # 主进程的线程池 (69个线程)
               └─...
```

**堆栈跟踪** (通过 `kill -USR1 105310` 获取)：
```
Thread 135753438852800 (QueueFeederThread):
  File 'multiprocessing/queues.py', line 270, in _feed
    send_bytes(obj)
  File 'multiprocessing/connection.py', line 384, in _send
    n = write(self._handle, buf)
  # ⚠️ QueueFeederThread 卡在 write() 调用！

Thread 135753452484288 (Thread-1):  # ProcessPoolExecutor 的管理线程
  File 'concurrent/futures/process.py', line 353, in run
    result_item, is_broken, cause = self.wait_result_broken_or_wakeup()
  File 'concurrent/futures/process.py', line 426, in wait_result_broken_or_wakeup
    ready = mp.connection.wait(readers + worker_sentinels)
  File 'selectors.py', line 415, in select
    fd_event_list = self._selector.poll(timeout)
  # ⚠️ 在等待 worker 结果，但 worker 已全部退出

Thread 135762625311488 (MainThread):
  File 'negmas/tournaments/tournaments.py', line 1395, in _run_parallel
    for i, future in track(
  File 'concurrent/futures/_base.py', line 243, in as_completed
    waiter.event.wait(wait_timeout)
  # ⚠️ 主线程卡在 as_completed()
```

### 4.3 问题根因分析

**核心问题：Worker 进程死亡但 Future 状态未更新**

1. **现象**：
   - 756 个任务，完成了 320 个 (42%)
   - 33 个 futures 标记为 `Running`，436 个标记为 `Pending`
   - 但实际上 **没有任何 worker 进程在运行**

2. **根因**：
   - ProcessPoolExecutor 的 worker 进程在某个时刻全部退出
   - 但 executor 内部没有检测到这些进程的异常退出
   - 导致对应的 futures 永远保持在 `Running` 状态
   - `as_completed()` 永远等待这些 "running" 的 futures

3. **QueueFeederThread 阻塞**：
   - `QueueFeederThread` 卡在 `write()` 调用
   - 可能是因为管道的另一端（worker）已关闭，但 feeder 线程没有收到通知
   - 这是一个典型的 **管道破裂 (Broken Pipe)** 场景，但信号被忽略了

4. **系统日志检查结果**：
   - ❌ 没有发现 OOM (Out of Memory) killer 记录
   - ❌ 没有发现 segfault 或其他内核级别的进程终止记录
   - Worker 进程可能是"正常退出"但未正确通知 executor

5. **状态冻结确认**：
   - 从 22:13:18 最后一个任务完成到 22:21:37（8分钟+）
   - `Done=320 Running=33 Pending=436` 完全没有变化
   - 主进程线程数：69个（主要是 ProcessPoolExecutor 的线程池）
   - 子进程：只剩 resource_tracker

### 4.4 回答最初的四个问题

| 问题 | 答案 |
|------|------|
| 1. 真的有未完成的 future 吗？ | ✅ 是的，436 个 Pending + 33 个 Running = 469 个未完成 |
| 2. 子进程都结束了，为什么 future 未完成？ | ⚠️ ProcessPoolExecutor 没有正确检测到 worker 退出，导致 futures 状态未更新 |
| 3. 主进程真的卡在等待 Future 的循环吗？ | ✅ 是的，堆栈跟踪确认主线程在 `as_completed()` 的 `waiter.event.wait()` |
| 4. 即使加入了 timeout，为什么没有解决问题？ | ⚠️ negmas 的 `_run_parallel` 中 `as_completed()` 没有传入 timeout 参数，total_timeout 只在循环体内检查，但循环根本进不去 |

### 4.5 下一步调查方向

1. **检查 worker 退出原因**：修改 negmas 或使用 `loky` 替代 `multiprocessing`
2. **添加 BrokenProcessPool 检测**：在 as_completed 循环中添加 executor 状态检查
3. **减少并发度**：使用 `parallelism='parallel:0.5'` 减少同时运行的 worker 数量
4. **使用 `dask` 替代**：`parallelism='dask'` 使用更健壮的分布式执行框架
5. **给 as_completed 添加 timeout**：修改 negmas 源码，让 as_completed 有超时机制

### 4.6 negmas 源码分析

问题代码位于 `/venv/lib/python3.12/site-packages/negmas/tournaments/tournaments.py`:

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

`as_completed()` 支持 `timeout` 参数，但 negmas 没有使用它。这是导致主进程无法退出的直接原因。

---

## 5. 最佳实践建议

1. **并行运行时务必设置 `verbose=False`**。如果需要调试，请减少并发数或使用 `serial` 模式。
2. **避免在全局对象中持有锁**。如果必须持有，请处理好 pickle 和 fork 的情况。
3. **使用文件日志代替 stdout**。`TrackerManager` 已经配置为将日志写入文件，这比打印到控制台更安全且性能更好。
4. **限制 world 数量**：使用 `max_worlds_per_config` 参数限制每个配置的 world 数量，减少出问题的概率。

---

## 6. 相关文件

- `diagnose_deep.py` - 深度监控脚本
- `diagnose_futures.py` - Future 状态追踪脚本
- `diagnose_progressive.py` - 渐进式测试脚本
- `diagnose_parallel_hang.py` - 并行挂起诊断脚本
- `diagnose_logs/` - 监控日志输出目录
  - `futures_trace.log` - Future 状态追踪日志
  - `futures_run.log` - 完整运行输出

---

**文档维护者**: GitHub Copilot  
**最后更新**: 2025年11月29日 22:25
