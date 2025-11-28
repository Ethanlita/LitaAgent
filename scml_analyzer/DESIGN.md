# SCML Analyzer 设计文档

## 核心设计原则

### 1. 独立性原则

**scml_analyzer 是完全独立的分析工具，不依赖于任何特定的 runner 实现。**

- 所有数据应从 **negmas tournament 目录** 自动提取
- 不需要 runner 传递任何参数或预处理数据
- 任何使用 `anac2024_std()` 或 `anac2024_oneshot()` 运行的比赛都可以被分析

### 2. 零配置原则

**使用方式应该极其简单：**

```bash
# 启动浏览器（从 tournament_history 读取）
python -m scml_analyzer.browser

# 导入所有比赛后启动
python -m scml_analyzer.browser --import-all

# 直接查看某个 negmas 目录
python -m scml_analyzer.visualizer --data C:\Users\xxx\negmas\tournaments\xxx-stage-0001
```

---

## 数据存储架构

### 三层数据结构

```
1. negmas 原始数据 (自动生成，保留)
   ~/negmas/tournaments/
   └── 20251128Hxxxxxx-stage-0001/
       ├── params.json
       ├── total_scores.csv
       └── ...

2. tracker 原始数据 (runner 生成，保留)
   项目目录/results/
   └── oneshot_quick_20251128_xxxxxx/
       ├── tournament_results.json
       └── tracker_logs/
           └── agent_xxx.json

3. tournament_history (统一整合，推荐使用)
   项目目录/tournament_history/
   └── 20251128_130949_oneshot/
       ├── tournament_info.json    # 元信息 (自动生成)
       ├── params.json             # 从 negmas 复制
       ├── total_scores.csv        # 从 negmas 复制
       ├── score_stats.csv         # 从 negmas 复制
       ├── world_stats.csv         # 从 negmas 复制
       └── tracker_logs/           # 从 results 复制 (自动匹配)
           └── agent_xxx.json
```

### 数据流

```
negmas tournament ─────┐
(~/negmas/tournaments) │
                       ├──► history.py ──► tournament_history/
tracker logs ──────────┘       │                    │
(results/xxx/tracker_logs)     │                    ▼
                               │              browser.py
                               │                    │
                               └──────────────► visualizer.py
```

---

## 模块职责

### history.py - 历史管理器

| 功能 | 函数 | 说明 |
|------|------|------|
| 导入单个比赛 | `import_tournament()` | 将 negmas 目录导入到 tournament_history |
| 自动导入 | `auto_import_tournament()` | 自动匹配 tracker 数据 |
| 批量导入 | `scan_and_import_all()` | 导入所有比赛 |
| 列出比赛 | `list_tournaments()` | 列出已导入的比赛 |
| 查找 tracker | `find_matching_tracker_dir()` | 根据时间戳匹配 tracker 目录 |

**自动匹配逻辑**：
- negmas 目录名: `20251128H130949613919Kqg-stage-0001`
- 提取时间戳: `20251128_130949`
- 在 `results/` 中查找包含该时间戳的目录
- 找到对应的 `tracker_logs/` 目录

### browser.py - 比赛浏览器

| 功能 | 说明 |
|------|------|
| 列出所有比赛 | 从 tournament_history 读取 |
| 查看比赛详情 | 调用 visualizer 显示 |
| 一键导入 | 调用 history.scan_and_import_all() |
| 双模式支持 | history 模式 / negmas 模式 |

### visualizer.py - 数据可视化

| 功能 | 说明 |
|------|------|
| 加载数据 | 从指定目录加载 CSV/JSON |
| 生成报告 | 生成 HTML 可视化页面 |
| HTTP 服务 | 提供 Web 访问 |

---

## 数据来源

所有数据都从 negmas 生成的 tournament 目录中提取：

```
tournament_directory/
├── params.json           # 比赛参数（n_configs, n_steps, competitors 等）
├── total_scores.csv      # 总分排名
├── winners.csv           # 冠军信息
├── world_stats.csv       # 每个 world 的统计（执行时间、异常等）
├── agent_stats.csv       # Agent 统计数据
├── score_stats.csv       # 分数统计（mean, std, min, max）
├── type_stats.csv        # 按 Agent 类型的统计
├── scores.csv            # 每个 world 每个 agent 的分数
├── stats.csv             # 详细统计
└── [world_directories]/  # 每个 world 的目录（可选）
```

### 4. 不应该做的事情

- ❌ 不要让 runner 保存特殊格式的 JSON 文件
- ❌ 不要要求 runner 计算或传递统计数据
- ❌ 不要依赖任何 runner 特有的输出格式
- ❌ 不要在 visualizer 中接受除路径以外的配置参数

---

## 数据提取规范

### 从 params.json 提取

| 字段 | 说明 |
|------|------|
| `name` | Tournament 名称 |
| `n_configs` | 配置数 |
| `n_runs_per_world` | 每配置运行次数 |
| `n_steps` | 每场步数 |
| `n_worlds` | 总 world 数 |
| `competitors` | 参赛 Agent 列表 |
| `parallelism` | 并行模式 |
| `oneshot_world` | 是否为 OneShot 赛道 |

### 从 total_scores.csv 提取

| 字段 | 说明 |
|------|------|
| `agent_type` | Agent 完整类型名 |
| `score` | 总分 |

### 从 winners.csv 提取

| 字段 | 说明 |
|------|------|
| `agent_type` | 冠军 Agent 类型 |
| `score` | 冠军分数 |

### 从 world_stats.csv 提取

| 字段 | 说明 |
|------|------|
| `name` | World 名称 |
| `execution_time` | 执行时间（秒）|
| `planned_n_steps` | 计划步数 |
| `executed_n_steps` | 实际执行步数 |
| `n_contracts_executed` | 执行的合同数 |
| `n_contracts_breached` | 违约的合同数 |
| ... | 其他统计字段 |

### 从 score_stats.csv 提取

| 字段 | 说明 |
|------|------|
| `agent_type` | Agent 类型 |
| `mean` | 平均分 |
| `std` | 标准差 |
| `min` | 最低分 |
| `max` | 最高分 |
| `count` | 场次 |

---

## Visualizer 显示内容

### 比赛概览卡片

- **完成的世界**: 从 `world_stats.csv` 行数提取
- **总耗时**: 从 `world_stats.csv` 的 `execution_time` 列求和
- **参赛 Agent 数**: 从 `params.json` 的 `competitors` 列表长度
- **冠军**: 从 `winners.csv` 提取

### 排名表

从 `total_scores.csv` 和 `score_stats.csv` 合并：

| 排名 | Agent 类型 | 平均分 | 标准差 | 最低分 | 最高分 | 场次 |
|------|-----------|--------|--------|--------|--------|------|
| 1 | MATAgent | 1.046 | 0.12 | 0.85 | 1.23 | 90 |
| 2 | LitaAgentY | 0.947 | 0.15 | 0.72 | 1.15 | 90 |
| ... | ... | ... | ... | ... | ... | ... |

### 得分分布图

从 `scores.csv` 提取每个 agent 在每个 world 的分数，生成箱线图或直方图。

---

## API 设计

### VisualizerData 类

```python
class VisualizerData:
    """从 negmas tournament 目录加载所有数据"""
    
    def __init__(self, tournament_dir: str):
        """
        Args:
            tournament_dir: negmas tournament 目录路径
                           (例如 C:\\Users\\xxx\\negmas\\tournaments\\xxx-stage-0001)
        """
        self.tournament_dir = Path(tournament_dir)
        self.load_all()
    
    def load_all(self):
        """自动加载所有数据文件"""
        self._load_params()        # params.json
        self._load_total_scores()  # total_scores.csv
        self._load_winners()       # winners.csv
        self._load_world_stats()   # world_stats.csv
        self._load_score_stats()   # score_stats.csv
        self._load_scores()        # scores.csv
    
    def get_summary(self) -> Dict:
        """获取比赛概览"""
        return {
            "name": self._params.get("name"),
            "track": "oneshot" if self._params.get("oneshot_world") else "std",
            "n_configs": self._params.get("n_configs"),
            "n_steps": self._params.get("n_steps"),
            "n_worlds": self._params.get("n_worlds"),
            "n_worlds_completed": len(self._world_stats),
            "total_duration_seconds": sum(w["execution_time"] for w in self._world_stats),
            "winner": self._extract_winner_name(),
            "winner_score": self._winners[0]["score"] if self._winners else None,
        }
    
    def get_rankings(self) -> List[Dict]:
        """获取排名数据（合并 total_scores 和 score_stats）"""
        ...
    
    def get_score_distribution(self, agent_type: str) -> List[float]:
        """获取某个 Agent 类型的分数分布"""
        ...
```

### start_server 函数

```python
def start_server(tournament_dir: str, port: int = 8080, open_browser: bool = True):
    """
    启动可视化服务器
    
    Args:
        tournament_dir: negmas tournament 目录路径（唯一必需参数）
        port: 服务器端口（可选，默认 8080）
        open_browser: 是否自动打开浏览器（可选，默认 True）
    """
    data = VisualizerData(tournament_dir)
    # 启动 HTTP 服务器...
```

---

## 兼容性说明

### 支持的数据源

1. **negmas tournament 目录**（推荐）
   - 直接指向 `C:\Users\xxx\negmas\tournaments\xxx-stage-0001`
   - 包含所有 CSV 和 JSON 文件

2. **自定义输出目录**（向后兼容）
   - 如果目录中包含 `tournament_results.json`，使用旧格式
   - 如果没有，尝试查找 negmas 格式的文件

### Agent 类型名称处理

negmas 生成的 agent_type 格式为：
```
scml.oneshot.sysagents.DefaultOneShotAdapter:litaagent_std.litaagent_y.LitaAgentY
```

显示时应提取简短名称：
```
LitaAgentY
```

---

## 未来扩展

1. **World 详情查看**: 点击某个 world 查看详细的合同、谈判、库存变化
2. **Agent 对比**: 选择两个 Agent 进行详细对比分析
3. **时间序列分析**: 查看 Agent 分数随比赛进程的变化
4. **异常检测**: 自动检测表现异常的 world 或 agent

---

## 快速使用指南

### 1. 导入并浏览所有比赛

```bash
python -m scml_analyzer.browser --import-all
```

这会：
- 扫描 `~/negmas/tournaments/` 中的所有比赛
- 自动匹配 `results/` 中的 tracker 数据
- 复制到 `tournament_history/`（保留原始数据）
- 启动浏览器界面

### 2. 仅启动浏览器（已导入的数据）

```bash
python -m scml_analyzer.browser
```

### 3. 直接扫描 negmas 原始数据

```bash
python -m scml_analyzer.browser --mode negmas
```

### 4. 命令行管理历史

```bash
# 列出已导入的比赛
python -m scml_analyzer.history list

# 导入单个比赛
python -m scml_analyzer.history import <negmas_dir>

# 导入所有比赛
python -m scml_analyzer.history import-all
```
