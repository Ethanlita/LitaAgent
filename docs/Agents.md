## 🎯 默认 Runner

**推荐使用 `run_default_std.py`** 作为 SCML 2025 Standard 比赛的默认 runner。

### 特性
- ✅ **Resumable**: 支持断点续跑，中断后重新运行同一目录即可继续
- ✅ **官方规模**: 默认使用 SCML 2025 Standard 官方环境 (20 configs × 2 runs)
- ✅ **完整参赛池**: 包含所有 LitaAgent + PenguinAgent + SCML 2025 Top 5
- ✅ **自动归集**: 运行完成后自动归集数据到 `tournament_history/`
- ✅ **灵活配置**: 支持 Tracker、Visualizer、规模、verbose 等参数

### 用法

```bash
# 1. 官方规模（默认，用于正式数据采集）
python runners/run_default_std.py

# 2. 快速测试（3 configs × 1 run）
python runners/run_default_std.py --quick

# 3. 自定义规模
python runners/run_default_std.py --configs 10 --runs 1

# 4. 启用 Tracker 和 Visualizer
python runners/run_default_std.py --tracker --visualizer

# 5. 断点续跑
python runners/run_default_std.py --output-dir tournament_history/my_run

# 6. 静默模式
python runners/run_default_std.py --quiet
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--configs` | 20 | World 配置数量 |
| `--runs` | 2 | 每配置运行次数 |
| `--max-top` | 5 | Top Agents 数量（前 N 名）|
| `--quick` | - | 快速测试模式 |
| `--tracker` | 否 | 启用 Tracker（记录协商过程）|
| `--visualizer` | 否 | 完成后启动可视化服务器 |
| `--no-auto-collect` | 否 | 禁用自动归集 |
| `--output-dir` | 自动生成 | 输出目录（复用可续跑）|
| `--parallelism` | parallel | 并行模式 |
| `--quiet` / `-q` | 否 | 静默模式 |
| `--verbose` / `-v` | 否 | 详细模式 |

---

## 注意事项

- 运行比赛后，务必向用户明确告知比赛的并行模式/结果输出路径，并等待用户观察或给出下一步指令后再继续后处理或新一轮运行，不需要轮询。
- 运行比赛时应该将stdout和stderr转到log文件中保持后台运行，确保不会一直占用终端。运行后告诉用户log文件路径。
