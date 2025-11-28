# LitaAgent - SCML 2025 竞赛代理

[![SCML 2025](https://img.shields.io/badge/SCML-2025-blue)](https://scml.cs.brown.edu)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)

SCML (Supply Chain Management League) 2025 竞赛的 LitaAgent 代理系列实现。

## 📋 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [比赛运行器](#比赛运行器)
- [LitaAgent 变体](#litaagent-变体)
- [分析工具](#分析工具)
- [命令行参数](#命令行参数)

## 📁 项目结构

```
SCML_initial/
├── litaagent_std/          # LitaAgent 代理实现
│   ├── litaagent_y.py      # LitaAgentY - 基础版
│   ├── litaagent_yr.py     # LitaAgentYR - 增强版（动态利润率）
│   ├── litaagent_n.py      # LitaAgentN - N 变体
│   ├── litaagent_p.py      # LitaAgentP - P 变体
│   ├── litaagent_cir.py    # LitaAgentCIR - 循环库存变体
│   ├── inventory_manager_*.py  # 库存管理器
│   └── tracker_mixin.py    # Tracker 混入类
├── runners/                # 比赛运行器 ⭐
│   ├── run_oneshot_full.py   # 完整 OneShot 比赛
│   ├── run_oneshot_quick.py  # 快速 OneShot 比赛
│   ├── run_std_full.py       # 完整 Standard 比赛
│   └── run_std_quick.py      # 快速 Standard 比赛
├── scml_analyzer/          # 分析工具包
│   ├── auto_tracker.py     # 自动追踪系统
│   └── visualizer.py       # 可视化服务器
├── examples/               # 示例脚本
│   ├── run_std_example.py    # Standard 赛道示例
│   └── run_oneshot_example.py # OneShot 赛道示例
└── results/                # 比赛结果输出目录
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行快速测试

```bash
# 快速 Standard 比赛（约 10-20 分钟）
.venv\Scripts\python.exe runners\run_std_quick.py

# 快速 OneShot 比赛（约 5-10 分钟）
.venv\Scripts\python.exe runners\run_oneshot_quick.py
```

### 3. 查看结果

比赛完成后会自动启动可视化服务器并在浏览器中打开分析报告。

## 🏆 比赛运行器

位于 `runners/` 目录下，包含 4 个运行器：

| 运行器 | 赛道 | 配置数 | 每配置运行 | 步数 | 总比赛 | 预计时间 |
|--------|------|--------|-----------|------|--------|---------|
| `run_oneshot_full.py` | OneShot | 10 | 2 | 50 | 20 场 | 30-60 分钟 |
| `run_oneshot_quick.py` | OneShot | 3 | 1 | 20 | 3 场 | 5-10 分钟 |
| `run_std_full.py` | Standard | 10 | 2 | 100 | 20 场 | 60-120 分钟 |
| `run_std_quick.py` | Standard | 3 | 1 | 50 | 3 场 | 10-20 分钟 |

### 参赛 Agents

每个运行器都包含：
- **LitaAgent 系列** (5 个): LitaAgentY, LitaAgentYR, LitaAgentN, LitaAgentP, LitaAgentCIR
- **2025 Top 5 Agents**: 自动从 `scml_agents` 包加载

### 使用方法

```bash
# 快速测试（推荐先用这个）
.venv\Scripts\python.exe runners\run_oneshot_quick.py

# 完整比赛
.venv\Scripts\python.exe runners\run_std_full.py

# 指定输出目录
.venv\Scripts\python.exe runners\run_std_quick.py --output-dir results/my_test

# 不启动服务器（用于批量测试）
.venv\Scripts\python.exe runners\run_std_full.py --no-server

# 使用不同端口
.venv\Scripts\python.exe runners\run_std_quick.py --port 9000
```

## 🤖 LitaAgent 变体

| Agent | 说明 | 特点 |
|-------|------|------|
| **LitaAgentY** | 基础版 | 标准实现 |
| **LitaAgentYR** | 增强版 | 动态利润率调整 |
| **LitaAgentN** | N 变体 | 优化谈判策略 |
| **LitaAgentP** | P 变体 | 优化价格策略 |
| **LitaAgentCIR** | CIR 变体 | 循环库存管理 |

## 📊 分析工具

### Tracker 系统

自动追踪 Agent 的：
- 谈判过程（报价、成功/失败）
- 合同签署情况
- 生产计划
- 库存状态

### 可视化服务器

比赛完成后自动启动 HTTP 服务器，提供：
- 排名表
- 得分图表
- Agent 详细分析
- 谈判统计

手动启动可视化：
```bash
python -m scml_analyzer.visualizer --data "results/xxx"
```

## ⚙️ 命令行参数

所有运行器都支持以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 输出目录 | `results/<track>_<mode>_<timestamp>` |
| `--port` | 可视化服务器端口 | 8080 |
| `--no-server` | 不启动可视化服务器 | False |

## 📝 示例脚本

`examples/` 目录下的示例脚本用于演示单场比赛：

```bash
# Standard 赛道单场演示
.venv\Scripts\python.exe examples\run_std_example.py

# OneShot 赛道单场演示
.venv\Scripts\python.exe examples\run_oneshot_example.py
```

## 📄 输出文件

比赛完成后会在输出目录生成：

```
results/std_quick_20251128_120000/
├── tracker_logs/           # Tracker 追踪数据
│   ├── agent_LitaAgentY.json
│   ├── agent_LitaAgentYR.json
│   └── ...
├── tournament_results.json # 比赛结果
└── analysis_report.html    # HTML 分析报告（可选）
```

## 🔧 开发

### 运行测试

```bash
.venv\Scripts\python.exe -m pytest tests/
```

### 添加新 Agent

1. 在 `litaagent_std/` 目录下创建新的 Agent 文件
2. 继承 `StdAgent` 或现有的 LitaAgent
3. 在 `runners/` 中的运行器里添加新 Agent

## 📚 参考

- [SCML 官方网站](https://scml.cs.brown.edu)
- [SCML 文档](https://scml.readthedocs.io)
- [NegMAS 文档](https://negmas.readthedocs.io)

## 📧 联系方式

- 团队名称: LitaAgent Team
- 竞赛: SCML 2025
