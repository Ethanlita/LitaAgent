# SCML 2025 比赛运行器

这个目录包含多个比赛运行器，用于测试 LitaAgent 系列与顶级 Agents 的对比。

## 🏗️ 系统架构

```
Runner 运行流程:
┌─────────────────────────────────────────────────────────────────┐
│ 1. 加载 Agents                                                   │
│    └─ inject_tracker_to_agents() 自动注入 Tracker               │
├─────────────────────────────────────────────────────────────────┤
│ 2. 运行比赛 (negmas)                                             │
│    └─ Tracker 自动记录所有协商数据                               │
├─────────────────────────────────────────────────────────────────┤
│ 3. 保存数据                                                      │
│    ├─ Tracker 日志 → results/xxx/tracker_logs/                  │
│    └─ 比赛结果 → results/xxx/tournament_results.json            │
├─────────────────────────────────────────────────────────────────┤
│ 4. 自动导入到 tournament_history/                                │
│    └─ 合并 negmas 数据 + Tracker 日志 (使用移动模式)             │
├─────────────────────────────────────────────────────────────────┤
│ 5. 启动 Visualizer（无参数）                                     │
│    └─ 自动从 tournament_history/ 读取所有比赛                    │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 运行器列表

### ⭐ 默认运行器（强烈推荐）

| 运行器 | 赛道 | 说明 |
|--------|------|------|
| **`run_default_std.py`** | Standard | 🎯 **默认 Runner**：官方规模、可断点续跑、自动归集、支持全部配置参数 |

```bash
# 官方规模（默认）
python runners/run_default_std.py

# 快速测试
python runners/run_default_std.py --quick

# 自定义规模 + 启用 Tracker
python runners/run_default_std.py --configs 10 --runs 1 --tracker

# 断点续跑（使用同一目录）
python runners/run_default_std.py --output-dir tournament_history/my_run
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--configs` | 20 | World 配置数量 |
| `--runs` | 2 | 每配置运行次数 |
| `--max-top` | 5 | Top Agents 数量 |
| `--quick` | - | 快速测试 (configs=3, runs=1) |
| `--tracker` | 否 | 启用 Tracker |
| `--visualizer` | 否 | 完成后启动 Visualizer |
| `--no-auto-collect` | 否 | 禁用自动归集 |
| `--output-dir` | 自动 | 输出目录（复用可续跑） |
| `--quiet` | 否 | 静默模式 |
| `--verbose` | 否 | 详细模式 |

### 🔹 其他运行器

| 运行器 | 赛道 | 配置数 | 每配置运行 | 步数 | 总比赛数 | 预计时间 |
|--------|------|--------|-----------|------|---------|---------|
| `run_oneshot_quick.py` | OneShot | 3 | 1 | 20 | 3 | 5-10 分钟 |
| `run_oneshot_full.py` | OneShot | 10 | 2 | 50 | 20 | 30-60 分钟 |
| `run_std_quick.py` | Standard | 3 | 1 | 50 | 3 | 10-20 分钟 |
| `run_std_full.py` | Standard | 10 | 2 | 100 | 20 | 60-120 分钟 |
| `run_std_full_tracked_penguin_logs_resumable.py` | Standard | 20（可调） | 2（可调） | 50-200（官方） | 官方全配 | 可断点续跑，强制谈判日志 |

这些运行器使用 negmas tournament API (`anac2024_oneshot()`/`anac2024_std()`)，数据会自动导入到 `tournament_history/`。

### 🔸 早期/测试运行器

| 运行器 | 说明 | API |
|--------|------|-----|
| `run_full_tournament.py` | OneShot + Std 完整比赛 | World.run_with_progress() |
| `run_full_std_tournament.py` | Std 比赛 + 详细分析 | World.run_with_progress() |
| `SCML_quick_test.py` | 快速功能测试 | World.run() |
| `SCML_small_test_tournament.py` | 小规模比赛测试 | World.run() |
| `run_scml_analyzer.py` | 完整分析流程 | Tournament wrapper |

这些早期运行器不使用 negmas tournament API，数据同样会导入到 `tournament_history/`。

## 🤖 参赛 Agents

### LitaAgent 系列 (5 个)
- **LitaAgentY** - 基础版
- **LitaAgentYR** - 增强版（动态利润率）
- **LitaAgentN** - N 变体
- **LitaAgentP** - P 变体
- **LitaAgentCIR** - CIR 变体（循环库存）

### 2025 Top Agents (排名前 5)
- 从 `scml_agents` 包自动加载 2025 年排名前 5 的 Agents

## 🚀 快速开始

### 1. 运行比赛

```bash
# 进入项目根目录
cd SCML_initial

# 运行快速 OneShot 比赛（推荐首次测试）
python runners/run_oneshot_quick.py

# 运行完整 OneShot 比赛
python runners/run_oneshot_full.py

# 运行快速 Standard 比赛
python runners/run_std_quick.py

# 运行完整 Standard 比赛
python runners/run_std_full.py
```

### 2. 查看结果

比赛完成后会自动：
1. ✅ 保存 Tracker 追踪数据
2. ✅ 导入数据到 `tournament_history/`
3. ✅ 启动 Visualizer 可视化服务器

浏览器会自动打开 http://localhost:8080 显示比赛列表。

### 3. 单独启动 Visualizer

```bash
# 不需要任何参数！自动从 tournament_history 读取
python -m scml_analyzer.visualizer
```

## 📊 数据流说明

### Tracker 数据记录

Runner 会自动为所有 LitaAgent 注入 Tracker：

```python
from litaagent_std.tracker_mixin import inject_tracker_to_agents

# 原始 Agents
lita_agents = [LitaAgentY, LitaAgentYR, LitaAgentN, LitaAgentP, LitaAgentCIR]

# 注入 Tracker（自动记录协商过程）
tracked_agents = inject_tracker_to_agents(lita_agents)
```

Tracker 自动记录：
- 协商开始/成功/失败
- 每轮出价（我方/对方）
- 合同签署/违约
- 每日状态（库存、余额、分数等）

### 数据导入

比赛完成后自动导入到 `tournament_history/`：

```
tournament_history/
├── 20251128_160240_oneshot/           # 比赛 ID = 日期_时间_赛道
│   ├── tournament_info.json           # 比赛元信息
│   ├── params.json                    # negmas 参数
│   ├── total_scores.csv               # 总分排名
│   ├── winners.csv                    # 冠军信息
│   ├── world_stats.csv                # 每场统计
│   ├── score_stats.csv                # 分数统计
│   ├── scores.csv                     # 详细分数
│   └── tracker_logs/                  # Tracker 数据
│       ├── agent_00LitaAgentY.json
│       ├── agent_01LitaAgentYR.json
│       └── tracker_summary.json
└── 20251128_180000_std/
    └── ...
```

### 手动导入历史数据

如果有之前运行的比赛数据，可以手动导入：

```bash
# 导入单个比赛
python -m scml_analyzer.history import "C:\Users\xxx\negmas\tournaments\xxx-stage-0001"

# 导入所有未导入的比赛
python -m scml_analyzer.history import-all

# 列出已导入的比赛
python -m scml_analyzer.history list
```

## ⚙️ 命令行参数

所有运行器都支持以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 输出目录 | `results/<track>_<mode>_<timestamp>` |
| `--port` | 可视化服务器端口 | 8080 |
| `--no-server` | 不启动可视化服务器 | False |

## 📝 示例

```bash
# 运行快速测试，指定输出目录
python runners/run_oneshot_quick.py --output-dir results/my_test

# 运行完整比赛，不启动服务器（用于批量测试）
python runners/run_std_full.py --no-server

# 运行比赛，使用不同端口
python runners/run_std_quick.py --port 9000
```

## 🔑 新的可断点续跑 runner
- `run_std_full_tracked_penguin_logs_resumable.py`：完整 Standard 比赛（Lita tracker + Penguin + Top Agents，强制谈判日志），支持中断后继续，使用 loky 并行。  
  - 用法：`python runners/run_std_full_tracked_penguin_logs_resumable.py --output-dir <目录> [--configs 20 --runs 2 --max-top 8 --parallelism parallel]`  
  - 断点恢复：保持同一 `--output-dir` 重新运行即可，已完成的 world（有 results.json）会跳过；若生成了 `*-stage-0001` 路径，脚本会自动识别并继续。

## 🔧 自定义 Agent

如果要添加自己的 Agent 参加比赛：

```python
# 1. 在 get_all_agents() 函数中添加
from your_module import YourAgent

def get_all_agents():
    # LitaAgents
    lita_agents = [LitaAgentY, LitaAgentYR, ..., YourAgent]
    
    # 注入 Tracker
    tracked_agents = inject_tracker_to_agents(lita_agents)
    
    return tracked_agents + list(TOP_AGENTS_2025)
```

## 📖 相关文档

- [Tracker 使用指南](../scml_analyzer/USAGE.md)
- [LitaAgent 系列说明](../litaagent_std/README.md)
- [Visualizer 设计文档](../scml_analyzer/DESIGN.md)
