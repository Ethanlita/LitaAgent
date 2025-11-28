# SCML Agent Analyzer 使用指南

## 概述

SCML Agent Analyzer 是一个用于 SCML（Supply Chain Management League）比赛的智能分析工具套件，提供：

- **自动追踪系统** (Auto Tracker) - 自动记录 Agent 运行时数据
- **比赛运行器** (Tournament Runner) - 方便运行 OneShot 和 Standard 赛道比赛
- **数据分析器** (Analyzer) - 分析 Agent 表现并生成报告
- **错误检测器** (Detectors) - 检测常见问题模式

---

## 快速开始

### 1. 基本用法

```python
from scml_analyzer.auto_tracker import (
    TrackerConfig, 
    TrackerManager, 
    TrackedAgent
)
from litaagent_std.tracker_mixin import inject_tracker_to_agents

# 1. 配置 Tracker
TrackerConfig.configure(
    enabled=True,
    log_dir="./tracker_logs",
    console_echo=False
)

# 2. 注入 Tracker 到你的 Agent
from litaagent_std.litaagent_y import LitaAgentY
tracked_agents = inject_tracker_to_agents([LitaAgentY])

# 3. 运行比赛后保存数据
TrackerManager.save_all("./tracker_logs")
```

### 2. 运行 OneShot 比赛

```python
from scml.oneshot import SCML2024OneShotWorld

world = SCML2024OneShotWorld(
    **SCML2024OneShotWorld.generate(
        agent_types=tracked_agents + [GreedyOneShotAgent],
        n_steps=20,
        n_processes=2,
    ),
    construct_graphs=True,
)

# 带进度条运行
world.run_with_progress()

# 获取结果
scores = world.scores()
```

### 3. 运行 Standard 比赛

```python
from scml.std import SCML2024StdWorld

world = SCML2024StdWorld(
    **SCML2024StdWorld.generate(
        agent_types=tracked_agents + [GreedyStdAgent],
        n_steps=50,
        n_processes=2,
    ),
    construct_graphs=True,
)

world.run_with_progress()
scores = world.scores()
```

---

## 详细配置

### TrackerConfig 配置项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | True | 是否启用追踪 |
| `log_dir` | str | None | 日志保存目录 |
| `console_echo` | bool | False | 是否同时输出到控制台 |
| `log_level` | str | "INFO" | 日志级别 (DEBUG/INFO/WARN/ERROR) |
| `auto_log_negotiations` | bool | True | 自动记录谈判事件 |
| `auto_log_contracts` | bool | True | 自动记录合同事件 |
| `auto_log_inventory` | bool | True | 自动记录库存状态 |
| `auto_log_production` | bool | True | 自动记录生产计划 |
| `auto_log_decisions` | bool | True | 自动记录决策事件 |

### 配置示例

```python
TrackerConfig.configure(
    enabled=True,
    log_dir="./my_experiment/logs",
    console_echo=True,  # 调试时启用
    log_level="DEBUG",
    auto_log_negotiations=True,
    auto_log_contracts=True,
    auto_log_inventory=True,
)
```

---

## 记录的数据类型

### 1. 谈判事件 (negotiation)
- `started` - 谈判开始
- `success` - 谈判成功
- `failure` - 谈判失败
- `offer_made` - 发出报价
- `offer_received` - 收到报价

### 2. 合同事件 (contract)
- `signed` - 合同签署
- `executed` - 合同执行
- `breached` - 合同违约

### 3. 库存事件 (inventory)
- `state` - 库存状态快照

### 4. 生产事件 (production)
- `scheduled` - 计划生产
- `executed` - 执行生产

### 5. 自定义事件 (custom)
- `daily_status` - 每日状态汇总
- 任意自定义事件

---

## 保存的文件格式

### tracker_summary.json
```json
{
  "world_id": "world_001",
  "agents": {
    "agent_01": {
      "type": "LitaAgentY",
      "stats": {
        "negotiations_started": 50,
        "negotiations_success": 30,
        "negotiations_failed": 20,
        "contracts_signed": 30,
        "contracts_breached": 2
      }
    }
  },
  "timestamp": "2025-11-28T10:00:00"
}
```

### agent_xxx.json
```json
{
  "agent_id": "agent_01",
  "agent_type": "LitaAgentY",
  "entries": [
    {
      "timestamp": "2025-11-28T10:00:00.123",
      "agent_id": "agent_01",
      "agent_type": "LitaAgentY",
      "world_id": "world_001",
      "day": 5,
      "category": "negotiation",
      "event": "success",
      "data": {
        "partner": "agent_02",
        "quantity": 10,
        "unit_price": 15
      }
    }
  ]
}
```

---

## 数据分析

### 使用内置分析器

```python
from scml_analyzer import LogParser, FailureAnalyzer, ReportGenerator

# 解析日志
parser = LogParser()
sim_data = parser.parse_directory("./tracker_logs")

# 分析失败原因
analyzer = FailureAnalyzer(sim_data)
results = analyzer.analyze_all_agents()

# 生成报告
reporter = ReportGenerator(results)
reporter.to_console()
reporter.to_json("./analysis_report.json")
```

### 分析 Tracker 数据

```python
import json
from collections import Counter

# 读取追踪数据
with open("tracker_logs/agent_01.json") as f:
    data = json.load(f)

# 统计事件类型
categories = Counter(e['category'] for e in data['entries'])
events = Counter(e['event'] for e in data['entries'])

print("Event Categories:", categories.most_common())
print("Event Types:", events.most_common())

# 分析谈判成功率
negotiations = [e for e in data['entries'] if e['category'] == 'negotiation']
success_count = sum(1 for n in negotiations if n['event'] == 'success')
total_count = len([n for n in negotiations if n['event'] in ['success', 'failure']])
success_rate = success_count / total_count if total_count > 0 else 0
print(f"Negotiation Success Rate: {success_rate:.2%}")
```

---

## 示例文件

完整示例请参考：
- `examples/run_oneshot_example.py` - OneShot 赛道示例
- `examples/run_std_example.py` - Standard 赛道示例
- `run_full_tournament.py` - 完整比赛运行器 (OneShot)
- `run_full_std_tournament.py` - 完整比赛运行器 (Standard)

---

## 命令行使用

```bash
# 运行 OneShot 比赛
python run_full_tournament.py --track oneshot --n-steps 20

# 运行 Standard 比赛
python run_full_std_tournament.py --n-steps 50

# 同时运行两种赛道
python run_full_tournament.py --track both --n-steps 30
```

---

## 常见问题

### Q: 为什么没有进度条？
A: 确保使用 `world.run_with_progress()` 而不是 `world.run()`。

### Q: 追踪数据文件为空？
A: 检查 `TrackerConfig.configure(enabled=True)` 是否正确设置，并确保调用了 `TrackerManager.save_all()`。

### Q: 如何自定义记录？
A: 在 Agent 中使用 `self.log(event, **data)` 方法：
```python
class MyAgent(TrackedAgent, StdSyncAgent):
    def my_method(self):
        self.log("my_custom_event", reason="custom", value=42)
```

---

## API 参考

### TrackerConfig

```python
TrackerConfig.configure(**kwargs)  # 配置全局设置
TrackerConfig.get()               # 获取当前配置
```

### TrackerManager

```python
TrackerManager.get_logger(agent_id, agent_type)  # 获取 Agent 的 Logger
TrackerManager.save_all(output_dir)              # 保存所有数据
TrackerManager.clear()                           # 清除所有 Logger
```

### AgentLogger

```python
logger = TrackerManager.get_logger("agent_01", "MyAgent")
logger.log(category, event, **data)  # 记录事件
logger.save(filepath)                 # 保存到文件
logger.get_stats()                    # 获取统计信息
```

---

## 版本历史

- **v0.2.0** - 添加 AutoTracker 系统，支持自动数据采集
- **v0.1.0** - 初始版本，基础日志解析和分析功能
