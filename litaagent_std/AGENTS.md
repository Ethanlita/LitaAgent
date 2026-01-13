这是根据Skeleton创建的LitaAgent
其路径位于./litaagent_std/litaagent_y.py
类名为LitaAgentY

---

## 终端与编码（Windows）

- Windows 下若出现乱码，优先用 PowerShell 7.x 的 `pwsh` 运行命令，默认 UTF-8。
- macOS/Ubuntu 默认 UTF-8，此项不适用。

## SCML 2024+ API 重要变更

> **关键信息**: 从 2024 年开始，SCML 统一了 OneShot 和 Standard 赛道的 API 接口。
> 
> 官方文档原文: "In 2024, we introduced a new implementation of the SCML-Standard track which simplified its API **making it exactly the same as the simpler SCML-OneShot track**."
> 
> 来源: https://scml.readthedocs.io/en/latest/tutorials/04.develop_agent_scml2024_std.html

### Agent 基类选择

两种 Agent 基类可供选择：

| 基类 | 协商方法 | 特点 |
|------|----------|------|
| `StdAgent` / `OneShotAgent` | `propose()`, `respond()` | 独立处理每个协商 |
| `StdSyncAgent` / `OneShotSyncAgent` | `counter_all()`, `first_proposals()` | 同步处理所有协商，可统一决策 |

### 当前 Agent 使用情况

- **LitaAgentY** 等: 继承自 `StdSyncAgent`，使用 `counter_all()` 和 `first_proposals()`
- **Tracker Mixin**: 需要同时支持两种模式的方法注入

### 关键回调方法

无论使用哪种基类，都有以下共同回调：
- `init()`: 模拟开始时调用一次
- `before_step()`: 每天开始时调用，此时 `ufun` 已设置
- `step()`: 每天结束时调用
- `on_negotiation_success()` / `on_negotiation_failure()`: 协商结束时调用

### Standard 与 OneShot 的主要区别

1. **合同时间范围**: Standard 可协商未来合同，OneShot 只能协商当天交付
2. **价格范围**: Standard 价格范围更大，需要更多价格策略
3. **供应链深度**: Standard 的生产图可以更深，中间层 Agent 需要同时与供应商和消费者协商

---

## Tracker 系统注意事项

### 并行执行问题

> **⚠️ 重要**: 动态注入的 Tracker（`inject_tracker_to_agents`）在并行模式下无法工作！
>
> 原因：Windows 上 Python multiprocessing 使用 `spawn` 模式，子进程会重新导入模块，动态修改的类不会被保留。

### 解决方案：使用静态定义的 Tracked 版本

我们为每个 Agent 类提供了静态定义的 `Tracked` 版本，支持并行模式：

**可用的 Tracked 版本：**

| Agent 文件 | Tracked 类 | 基类 |
|------------|-----------|------|
| `litaagent_y.py` | `LitaAgentYTracked` | `LitaAgentY` |
| `litaagent_p.py` | `LitaAgentPTracked` | `LitaAgentP` |
| `litaagent_yr.py` | `LitaAgentYRTracked` | `LitaAgentYR` |
| `litaagent_ys.py` | `LitaAgentYSTracked` | `LitaAgentYR` (文件内定义) |
| `litaagent_n.py` | `LitaAgentNTracked` | `LitaAgentN` |
| `litaagent_cir.py` | `LitaAgentCIRTracked` | `LitaAgentCIR` |
| `litaagent_cirs.py` | `LitaAgentCIRSTracked` | `LitaAgentCIR` (文件内定义) |

### 新增：LitaAgent-HRL（模式 B）

- 位置：`litaagent_std/hrl_xf/agent.py`
- 基类：`StdAgent`
- 特点：HRL-XF 四层架构（L1 安全护盾 → L2 16 维分桶目标 → L3 残差执行（含 baseline 条件化）→ L4 并发协调），并用“批次统一规划 + 动态预留”降低顺序依赖。
- Tracked 版本：`LitaAgentHRLTracked`

```python
# 推荐方式：使用静态定义的 Tracked 版本（支持并行模式）
import os
os.environ['SCML_TRACKER_LOG_DIR'] = os.path.abspath('./tracker_logs')

# 导入需要的 Tracked 版本
from litaagent_std.litaagent_y import LitaAgentYTracked
from litaagent_std.litaagent_p import LitaAgentPTracked
from litaagent_std.litaagent_yr import LitaAgentYRTracked
from litaagent_std.litaagent_ys import LitaAgentYSTracked
from litaagent_std.litaagent_n import LitaAgentNTracked
from litaagent_std.litaagent_cir import LitaAgentCIRTracked
from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked
from litaagent_std.hrl_xf import LitaAgentHRLTracked

# 在比赛中使用 Tracked 版本
results = anac2024_oneshot(
    competitors=[LitaAgentYTracked, LitaAgentYRTracked],
    parallelism='parallel',  # 并行模式也能工作！
    # ...
)
```

### 为新 Agent 添加 Tracker 支持

如果你开发了新的 Agent 类，可以按照以下模式添加 Tracked 版本：

```python
# 在 Agent 文件末尾添加 Tracked 版本

# 1. 导入 Tracker
try:
    from scml_analyzer.auto_tracker import TrackerManager, AgentLogger
    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False
    TrackerManager = None
    AgentLogger = None


# 2. 定义 Tracked 版本
class MyAgentTracked(MyAgent):
    """带有 Tracker 功能的 MyAgent（支持并行模式）"""
    
    _tracker_logger = None
    
    @property
    def tracker(self):
        if not _TRACKER_AVAILABLE:
            return None
        if self._tracker_logger is None:
            self._tracker_logger = TrackerManager.get_logger(self.id, 'MyAgent')
        return self._tracker_logger
    
    def init(self):
        super().init()
        if self.tracker:
            self.tracker.custom("agent_initialized", 
                n_steps=self.awi.n_steps,
                level=self.awi.level,
            )
    
    def before_step(self):
        super().before_step()
        if self.tracker:
            self.tracker.set_day(self.awi.current_step)
            # 记录每日状态...
    
    def step(self):
        super().step()
        if self.tracker:
            # 最后一步保存数据
            if self.awi.current_step >= self.awi.n_steps - 1:
                world_id = getattr(self.awi._world, 'id', 'unknown')
                self.tracker.world_id = world_id
                
                log_dir = os.environ.get('SCML_TRACKER_LOG_DIR')
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                    safe_id = self.id.replace("@", "_at_")
                    self.tracker.save(f"{log_dir}/agent_{safe_id}.json")
    
    def counter_all(self, offers, states):
        responses = super().counter_all(offers, states)
        if self.tracker:
            # 记录报价和响应...
        return responses
    
    # ... 其他方法
```

### Tracker 可记录的事件类型

| 方法 | 用途 |
|------|------|
| `tracker.set_day(day)` | 设置当前天数 |
| `tracker.custom(event, **data)` | 记录自定义事件 |
| `tracker.inventory_state(raw, product, balance)` | 记录库存状态 |
| `tracker.negotiation_started(partner, issues, is_seller)` | 协商开始 |
| `tracker.negotiation_offer_made(partner, offer, reason)` | 发出报价 |
| `tracker.negotiation_offer_received(partner, offer)` | 收到报价 |
| `tracker.negotiation_accept(partner, offer, reason)` | 接受报价 |
| `tracker.negotiation_reject(partner, offer, reason)` | 拒绝报价 |
| `tracker.negotiation_success(partner, agreement)` | 协商成功 |
| `tracker.negotiation_failure(partner, reason)` | 协商失败 |
| `tracker.contract_signed(id, partner, qty, price, day, is_seller)` | 合同签署 |
| `tracker.decision(name, result, reason)` | 记录决策 |
| `tracker.save(filepath)` | 保存到文件 |

### 旧方式（仅单进程模式）

如果确定只使用单进程模式，可以使用动态注入：

```python
# 仅适用于 parallelism='serial' 模式
from litaagent_std.tracker_mixin import inject_tracker_to_agents
from scml_analyzer.auto_tracker import TrackerConfig

TrackerConfig.configure(log_dir='./tracker_logs', enabled=True)
agents = inject_tracker_to_agents([LitaAgentY, LitaAgentP])

results = anac2024_oneshot(
    competitors=agents,
    parallelism='serial',  # 必须使用 serial 模式！
)
```

### Tracker Mixin 工作原理

`tracker_mixin.py` 通过动态包装以下方法来记录事件：

| 方法 | 记录的事件 |
|------|-----------|
| `init()` | `agent_initialized` |
| `before_step()` | `state`, `daily_status` |
| `first_proposals()` | `started`, `offer_made` (初始报价) |
| `counter_all()` | `offer_received`, `offer_made` (还价), `accept`, `reject` |
| `on_negotiation_success()` | `signed`, `success` |
| `on_negotiation_failure()` | `failure` |

---

## Tracker 日志分析工具

### `scripts/analyze_shortfall.py` - 每日平衡分析

分析 LOS（LitaAgentOS）BUYER/SELLER 每日供需平衡情况，检测 shortfall、exact、overfull 的天数分布。

**用途**：
- 诊断 BUYER 或 SELLER 过度购买/销售的问题
- 比较不同版本 Agent 的表现

**用法**：
```bash
# 自动发现最近2场 oneshot 比赛并分析
python scripts/analyze_shortfall.py

# 分析指定比赛
python scripts/analyze_shortfall.py results/20260110_212745_oneshot

# 比较两场比赛
python scripts/analyze_shortfall.py results/20260110_191341_oneshot results/20260110_212745_oneshot
```

**输出示例**：
```
================================================================================
📊 [20260110_212745_oneshot]
================================================================================

📦 SELLER (Level 0): 205 days
  ❌ Shortfall (sold < supply):   31 days ( 15.1%)
  ✅ Exact (sold == supply):     141 days ( 68.8%)
  ⚠️ Overfull (sold > supply):   33 days ( 16.1%)

🛒 BUYER (Level 1): 215 days
  ❌ Shortfall (bought < demand): 11 days (  5.1%)
  ✅ Exact (bought == demand):     9 days (  4.2%)
  ⚠️ Overfull (bought > demand): 195 days ( 90.7%)  ⚠️ 过度购买！
```

**关键指标解读**：
- **Shortfall**: 协商获得的量 < 外生需求量（供应不足）
- **Exact**: 协商获得的量 == 外生需求量（完美匹配）
- **Overfull**: 协商获得的量 > 外生需求量（过度采购/销售）

**典型问题诊断**：
- BUYER Overfull 过高 → 过度购买，可能导致库存积压和成本损失
- SELLER Shortfall 过高 → 销售不足，可能导致外生供应浪费
