# SCML Parallel 模式死锁问题调查报告

**日期**: 2025年11月29日  
**状态**: 调查中  
**影响**: Windows 和 Linux 平台上使用 `parallelism="parallel"` 运行 SCML 锦标赛时会挂起

---

## 0. SCML 2025 Standard Track 背景知识

### 0.1 什么是 SCML？

**SCML (Supply Chain Management League)** 是 ANAC (Automated Negotiating Agents Competition) 国际竞赛的一部分，自2019年起每年举办。该竞赛模拟一个供应链管理场景，参赛者需要设计自主代理（Agent）来管理工厂、与其他代理进行谈判以采购原材料和销售产品，目标是最大化利润。

**官方网站**: https://scml.cs.brown.edu/  
**文档**: https://scml.readthedocs.io/  
**源码**: https://github.com/yasserfarouk/scml

### 0.2 SCML 2025 与 SCML 2024 的关系

> ⚠️ **重要说明**: SCML 2025 **沿用了 SCML 2024 的规则和 API**。
> 
> 官方PDF文档（scml2025.pdf, overview2025.pdf）发布于2025年3月，但规则内容仍标注为"SCML 2024"。文档中明确说：*"There are two tracks in SCML 2024. This document pertains only to the Standard track."*

因此：
- **运行比赛**: 使用 `anac2024_std()` 函数（没有 `anac2025_std`）
- **World 类**: 使用 `SCML2024StdWorld`（没有 `SCML2025StdWorld`）
- **Agent 基类**: 使用 `StdAgent` / `StdSyncAgent`
- **2025年参赛Agents**: 存在于 `scml_agents.scml2025.standard.*`，但运行在 `SCML2024StdWorld` 上

### 0.3 SCML 的两个赛道

| 赛道 | 说明 | World 类 | Agent 基类 |
|------|------|----------|------------|
| **Standard** | 完整游戏，代理需要考虑长期规划、生产调度和多日谈判 | `SCML2024StdWorld` | `StdAgent` / `StdSyncAgent` |
| **OneShot** | 简化游戏，专注于单日内的多对多并发谈判 | `SCML2024OneShotWorld` | `OneShotAgent` |

### 0.4 SCML 2025 Standard Track 规则要点

基于官方文档 (scml2025.pdf)：

1. **产品与生产图**: 
   - n 种产品类型：原材料(product 0) → 中间产品(products 1:n-2) → 最终产品(product n-1)
   - n-1 个制造过程，每个将 product i 转换为 product i+1
   - 工厂组织在 n-1 层 (L₀ 到 Lₙ₋₂)

2. **外生合约 (Exogenous Contracts)**:
   - L₀ 工厂收到外生**买入**合约（原材料供应）
   - Lₙ₋₁ 工厂收到外生**卖出**合约（最终产品需求）

3. **谈判议题**:
   - **数量 (Quantity)**: 1 到 σ×λₐ（σ是配置参数，λₐ是生产线数量）
   - **交付日期 (Delivery Day)**: 0（当天）到 H-1（H是谈判地平线）
   - **单价 (Unit Price)**: 基于交易价格 tp(s) 的 ±κ 范围内

4. **Standard vs OneShot 的主要区别**:
   - 产品**不易腐**：可以累积库存（支付存储成本而非丢弃）
   - 可以谈判**未来合约**：不仅限当天交付
   - **价格范围更大**：需要认真考虑价格策略
   - **生产图可以更深**：代理可能同时与供应商和消费者谈判

5. **评估标准**: 使用 truncated mean（截断均值）进行排名

### 0.5 如何运行一场 SCML 比赛

#### 方法一：使用 `anac2024_std` 函数运行锦标赛

```python
from scml.utils import anac2024_std
from scml.std import RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent

results = anac2024_std(
    competitors=[RandomStdAgent, GreedyStdAgent, SyncRandomStdAgent],
    n_configs=5,            # 生成的世界配置数量
    n_runs_per_world=1,     # 每个配置重复运行的次数
    n_steps=50,             # 每场模拟的步数（天数）
    parallelism="parallel", # 并行模式: "parallel", "serial", "dask"
    print_exceptions=True,
)

# 查看结果
print(f"Winners: {results.winners}")
print(results.total_scores)
```

#### 方法二：运行单个 World

```python
from scml.std import SCML2024StdWorld, RandomStdAgent, GreedyStdAgent

agent_types = [RandomStdAgent, GreedyStdAgent]
world = SCML2024StdWorld(
    **SCML2024StdWorld.generate(agent_types=agent_types, n_steps=50),
    construct_graphs=True,
)
world.run()

# 查看统计信息
world.plot_stats()
```

### 0.6 SCML 2025 Standard Track 获胜者

| 名次 | Agent 名称 | 开发者 | 所属机构 |
|------|-----------|--------|----------|
| 🥇 1st | AS0 | Atsunaga Sadahiro | TUAT (东京农工大学) |
| 🥈 2nd | XenoSotaAgent | Sota Sakaguchi, Takanobu Otsuka | NIT (名古屋工业大学) |
| 🥉 3rd | UltraSuperMiracleSoraFinalAgentZ | Sora Nishizaki, Takanobu Otsuka | NIT |

**获取2025年参赛Agents**:
```python
from scml_agents import get_agents

# 获取2025年Standard赛道前5名
top_agents = get_agents(2025, track="std", top_only=5, as_class=True)
# 返回: [XenoSotaAgent, UltraSuperMiracleSoraFinalAgentZ, PonponAgent, ...]

# 获取2025年获胜者
winners = get_agents(2025, track="std", winners_only=True, as_class=True)
```

### 0.7 相关软件包

| 包名 | 当前版本 | 用途 | 安装命令 |
|------|----------|------|----------|
| `scml` | 0.7.7 | SCML 核心库 | `pip install scml` |
| `negmas` | 0.10.21 | 多代理谈判系统底层库 | `pip install negmas` |
| `scml-agents` | 0.4.13 | 历届参赛 Agent 集合 | `pip install scml-agents` |
| `scml-vis` | - | 可视化工具 | `pip install scml-vis` |

**注意**: 官方推荐 Python 3.10 或 3.11，因为 stable_baselines3 尚不完全支持 Python 3.12。

---

## 1. 问题描述

### 1.1 现象

在运行 SCML 锦标赛时，使用 `parallel` 模式会导致程序挂起（Windows 和 Linux 均受影响）：

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
**需要运行大规模干净测试**：我们现在的工作基本上集中于创建了一个新的scml_analyzer来跟踪agent的运行情况。然而，这一工具本身也有可能导致问题，我们必须排除这种可能性。
具体而言，我们需要进行一次大规模的、完整的”干净运行“：即在不使用scml_analyzer追踪agent的情况下，运行一场完整的SCML 2025 Standard比赛，且应当有以下Agent参加： 
  - Negmas内置agent
  - 所有的LitaAgent
  - 所有参加SCML 2025的Agent（先选Top 5，如果未能复现Hung的问题，则进一步扩大规模到全部）
以之前的经验，这种规模的比赛一定会Hung。

**如果 agents 大规模测试不会挂起**：
- 问题在 scml_analyzer，需要检查其多进程安全性

**如果 agents 大规模测试仍然挂起**：
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


## 12. 最新排查（2025-12-01）

### 12.1 干净运行大规模测试（无 scml_analyzer）
- **脚本**: `diagnose_deep.py`（新增 `tournament_path` → `results/clean_run_<timestamp>`，Top Agents 使用 `get_agents(2025, top_only=5, track='std')`）
- **配置**: 13 Agents（5 Lita + 2025 Top5 + Random/Greedy/SyncRandom），`n_configs=3`，`n_steps=50`，`parallelism='parallel'`，不加载 scml_analyzer
- **运行命令**: `PYTHONUNBUFFERED=1 ./venv/bin/python diagnose_deep.py > diagnose_logs/clean_run.out 2>&1`
- **现象**: 运行约 16 分钟后卡死。`ps --ppid <主进程>` 仅剩 `resource_tracker`，所有 worker 退出，主进程 CPU≈0。
- **日志**:
  - 监控: `diagnose_logs/monitor_20251201_112232.log`
  - 主日志: `diagnose_logs/main_20251201_112232.log`
  - 输出目录: `results/clean_run_20251201_112232/20251201H112236910233Kqg-stage-0001/`

### 12.2 gdb/strace 定位
- 安装了 `gdb`、`python3.12-dbg`，在 full access 环境下调试。
- **gdb (py-bt) 主线程栈**：
  ```
  diagnose_deep.py:243 main
  → scml.utils.anac2024_std
  → negmas.tournaments.tournament/_run_eval/run_tournament/_run_parallel
  → rich.progress.track
  → concurrent.futures.as_completed
  → threading.Event.wait  ← 卡住
  ```
- **关键发现**: `as_completed()` 在等待 futures，worker 全部退出后未标记完成，导致无限等待（无全局超时）。
- 线程概况：
  - 大量 OpenBLAS/Scipy 线程在 `pthread_cond_wait`（空闲）。
  - 两个 `rich` 进度线程在 futex 等待。
  - CUDA 线程在 poll 等待。
- strace (`/home/ecs-user/strace_10914.log`) 也显示主线程和等待线程长期 futex，未有子进程活动。

### 12.3 结论
- **确认挂点**: negmas `_run_parallel` 内 `as_completed()` 无超时，worker 意外退出后主进程永远等待。
- **已排除**: scml_analyzer 影响；纯干净运行也会挂死。
- **下一步建议**（对应 10.4）：
 1) 尝试 Python 3.11 复现；
 2) 尝试 `parallelism='dask'` 或 joblib/loky；
 3) 在 negmas `_run_parallel` 增加超时/日志，定位崩溃的 worker。

## 13. 最新排查（二次 Hung，worker 追踪）

### 13.1 新增追踪机制
- 在 `diagnose_deep.py` 对 negmas `_run_worlds` 做 monkeypatch，记录子进程的 `worker_start/worker_done/worker_error/worker_exit` 到 `diagnose_logs/worker_trace_<timestamp>.log`，包含 run_id 与 world 名称（config_id/name）。使用 `spawn` 保持与官方行为一致，追踪函数定义在顶层以便 pickle。

### 13.2 本次运行（clean_run_20251201_121933，已 Hung）
- 配置同前：13 agents，n_configs=3，n_steps=50，parallelism='parallel'，无 tracker。
- 运行约 16 分钟后再次挂起。监控：`diagnose_logs/monitor_20251201_121933.log`，子进程仅剩 `resource_tracker`，CPU≈0，进度停在检查 #199。
- worker 追踪：`diagnose_logs/worker_trace_20251201_121933.log`
  - 记录 320 个 world：`start=320, done=320, err=0, exit=320`，无遗漏的 stuck run_id。
  - 说明所有 `_run_worlds` 都正常返回/退出后才进入 hung。

### 13.3 调试采样
- **strace** (PID 35849): futex ~56%、wait4 ~28%，主线程处于 futex 等待。
- **gdb py-bt** (PID 35849): 主线程仍卡在 `concurrent.futures.as_completed()` 的 `waiter.event.wait()`，链路：
  ```
  diagnose_deep.py -> scml.utils.anac2024_std -> negmas._run_parallel
  -> rich.progress.track -> concurrent.futures.as_completed -> threading.Event.wait
  ```
- 子进程状态：仅 resource_tracker 存活，所有 worker 已退出。

### 13.4 结论更新
- Hung 不是单个 world 崩溃：所有 `_run_worlds` 已完成且记录 `worker_done/exit`，但主线程仍阻塞在 as_completed，推测 executor/future 完成信号丢失或队列异常。
- 如需继续定位且不改 negmas，可在父进程对 futures 增加 done 回调或用 `wait(..., timeout)` 包裹；或自建 executor 替换，观察信号是否正常。

### 13.5 可能的信号缺失原因
- ProcessPoolExecutor 管道/队列异常（序列化失败、BrokenPipe、队列关闭），导致 future 不完成。
- futures 集合不为空但 executor 已回收/崩溃，as_completed 没收到完成事件。
- 结果对象无法 pickle（曾见过 “Can’t pickle local object ...”），使 `_feed` 中断。
- executor 清理顺序异常，worker 全退出但 future 状态未标记异常。
- 外部信号/OOM 杀掉 worker，完成信号未送达。

### 13.6 后续方案评估
1) 捕捉 executor/future 状态  
   - 在父进程拿到 futures 后加 `add_done_callback` 记录 result/exception，或监控 executor 队列；可判断是未完成还是完成信号丢失。需获取 negmas 内部 futures，可通过自建 executor/包装 `_run_parallel` 实现。
   - **已完成**：参见13.7和13.8
2) 自建 executor（可选 loky），设置 max_workers/mp_context=spawn，替换 negmas 内置  
   - 好处：可控的 futures + 回调，绕过默认 executor 的潜在 bug，并可启用 maxtasksperchild。需 monkeypatch `_get_executor` 或直接改 `_run_parallel`。
3) 切换 Python 3.11  
   - **已完成**：Python 3.11.14（deadsnakes）新环境复测，仍然 Hung（进度停在 #274，worker_trace 320/320 完成）。strace（`strace_45756_py311.log`）与 gdb（futex 等待）显示与 3.12 相同症状，未解决问题。
4) 使用 dask 并行后端  
   - 作为旁路方案再试，需控制序列化体积；此前大规模有反序列化超大数组的异常，可靠性存疑。
5) 使用 loky 替代 multiprocessing  
   - 更健壮的进程池，可与方案 2 结合：自建基于 loky 的 executor 替换 negmas 内置，监控 futures 完成信号。

### 13.7 追加采样与追踪
- Python 3.11 Hung 时 strace/gdb：`/home/ecs-user/strace_45756_py311.log`（futex/ wait4 主导），gdb 栈顶在 futex 等待（py-bt 读不到 Python 行号，但症状同前）。
- 为捕捉 future 状态，已在 `diagnose_deep.py` monkeypatch `negmas._submit_all`，在提交时记录 `future_submitted`，并通过回调记录 `future_done/cancelled/error` 到 `diagnose_logs/future_trace_<timestamp>.log`（当前运行示例：`future_trace_20251201_134500.log`）。worker 事件仍写入 `worker_trace_<timestamp>.log`。

### 13.8 Python 3.11 + Future 追踪的最新发现（clean_run_20251201_134500，已 Hung）
- 监控：`monitor_20251201_134500.log` 停在检查 #218，子进程仅 `resource_tracker`，CPU≈0。
- worker_trace：320 个 world 全部完成/退出（start=320, done=320, err=0, exit=320）。
- future_trace：`future_submitted=2574`，`future_done=320`，无 error/cancel，剩余 2254 个 futures 未完成，直接导致 `as_completed` 永久等待。说明完成信号在 executor/future 层丢失，而不是 `_run_worlds` 崩溃。

## 14. loky 替换与运行注意事项（2025-12-04）
- 所有 `runners/` 脚本已在开头调用 `runners.loky_patch.enable_loky_executor()`，默认改用 loky 的 `ProcessPoolExecutor`，不用调整传入 negmas 的 `parallelism`（仍用 `parallel` 即可）。
- 通过环境变量 `SCML_PARALLELISM` 控制：`loky`（默认）或 `loky:<fraction>`（按 CPU 比例限制并发，至少 1）。未设置时默认启用 loky。
- 在自定义脚本里，如果需要 loky，同样在调用比赛前 `from runners.loky_patch import enable_loky_executor; enable_loky_executor()`。
- 比赛完成后，Agent/脚本应告知用户结果路径并等待用户查看或确认下一步（见 Agents.md 注意事项）。
