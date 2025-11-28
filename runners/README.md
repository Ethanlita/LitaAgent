# SCML 2025 比赛运行器

这个目录包含 4 个比赛运行器，用于测试 LitaAgent 系列与 2025 年顶级 Agents 的对比。

## 运行器列表

| 运行器 | 赛道 | 配置数 | 每配置运行 | 步数 | 总比赛数 | 预计时间 |
|--------|------|--------|-----------|------|---------|---------|
| `run_oneshot_full.py` | OneShot | 10 | 2 | 50 | 20 | 30-60 分钟 |
| `run_oneshot_quick.py` | OneShot | 3 | 1 | 20 | 3 | 5-10 分钟 |
| `run_std_full.py` | Standard | 10 | 2 | 100 | 20 | 60-120 分钟 |
| `run_std_quick.py` | Standard | 3 | 1 | 50 | 3 | 10-20 分钟 |

## 参赛 Agents

### LitaAgent 系列 (5 个)
- LitaAgentY - 基础版
- LitaAgentYR - 增强版（动态利润率）
- LitaAgentN - N 变体
- LitaAgentP - P 变体
- LitaAgentCIR - CIR 变体（循环库存）

### 2025 Top Agents (排名前 5)
- 从 `scml_agents` 包自动加载 2025 年排名前 5 的 Agents

## 使用方法

```bash
# 进入项目根目录
cd SCML_initial

# 运行快速 OneShot 比赛（推荐用于快速测试）
.venv\Scripts\python.exe runners\run_oneshot_quick.py

# 运行完整 OneShot 比赛
.venv\Scripts\python.exe runners\run_oneshot_full.py

# 运行快速 Standard 比赛
.venv\Scripts\python.exe runners\run_std_quick.py

# 运行完整 Standard 比赛
.venv\Scripts\python.exe runners\run_std_full.py
```

## 命令行参数

所有运行器都支持以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output-dir` | 输出目录 | `results/<track>_<mode>_<timestamp>` |
| `--port` | 可视化服务器端口 | 8080 |
| `--no-server` | 不启动可视化服务器 | False |

## 输出

比赛完成后会：
1. 保存 Tracker 追踪数据到 `tracker_logs/` 目录
2. 保存比赛结果到 `tournament_results.json`
3. 自动启动可视化服务器并在浏览器中打开分析报告

## 示例

```bash
# 运行快速测试，指定输出目录
.venv\Scripts\python.exe runners\run_oneshot_quick.py --output-dir results/my_test

# 运行完整比赛，不启动服务器（用于批量测试）
.venv\Scripts\python.exe runners\run_std_full.py --no-server

# 运行比赛，使用不同端口
.venv\Scripts\python.exe runners\run_std_quick.py --port 9000
```
