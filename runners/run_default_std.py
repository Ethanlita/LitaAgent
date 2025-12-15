#!/usr/bin/env python
"""
🎯 默认 SCML 2025 Standard 比赛 Runner（用于 HRL 训练数据采集）

这是 LitaAgent 项目的默认 runner，用于运行官方规模的 SCML 2025 Standard 比赛
并采集谈判日志以供 HRL 训练使用。

核心特性：
- ✅ Resumable: 支持断点续跑，中断后重新运行同一目录即可继续
- ✅ 官方规模: 默认使用 SCML 2025 Standard 官方环境和规模
- ✅ 完整参赛池: 包含所有 LitaAgent、PenguinAgent 及 SCML 2025 Top 5 Agents
- ✅ 可配置规模: 支持通过参数指定更小的比赛规模用于测试
- ✅ 自动归集: 运行完成后自动归集数据到 tournament_history/
- ✅ 默认不使用 Tracker: 避免额外开销，适合大规模训练数据采集
- ✅ 默认不使用 Visualizer: 无需人工观察时节省资源

用法：
    # 1. 默认官方规模（推荐用于正式数据采集）
    python runners/run_default_std.py
    
    # 2. 快速测试（3 个配置，1 轮）
    python runners/run_default_std.py --quick
    
    # 3. 自定义规模
    python runners/run_default_std.py --configs 10 --runs 1 --steps 50
    
    # 4. 启用 Tracker 和 Visualizer
    python runners/run_default_std.py --tracker --visualizer
    
    # 5. 断点续跑（使用同一输出目录）
    python runners/run_default_std.py --output-dir tournament_history/my_run
    
    # 6. 静默模式（减少输出）
    python runners/run_default_std.py --quiet

环境：
- SCML 2025 Standard World（Futures Market）
- 步数范围: 50-200（官方随机）
- 工厂数: 5-15（官方随机）
- 层级数: 3-5（官方随机）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Type

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 应用 loky 执行器补丁（避免 pickle 问题）
from runners.loky_patch import enable_loky_executor
enable_loky_executor()

from negmas.helpers.inout import load
from negmas.helpers.numeric import truncated_mean
from negmas.tournaments.tournaments import (
    ASSIGNED_CONFIGS_JSON_FILE,
    ASSIGNED_CONFIGS_PICKLE_FILE,
    RESULTS_FILE,
    evaluate_tournament,
    run_tournament,
)
from scml.utils import (
    anac2024_std,
    anac2024_std_world_generator,
    balance_calculator_std,
)
from scml.std.agents import RandomStdAgent

# ============================================================================
# LitaAgent 系列（使用延迟导入避免可选依赖问题）
# ============================================================================
LITA_AGENTS: List[Type] = []
LITA_AGENT_NAMES: List[str] = []

def _load_lita_agents() -> Tuple[List[Type], List[str]]:
    """延迟加载 LitaAgent，避免可选依赖问题。"""
    global LITA_AGENTS, LITA_AGENT_NAMES
    if LITA_AGENTS:
        return LITA_AGENTS, LITA_AGENT_NAMES
    
    agents = []
    names = []
    
    # 核心 LitaAgent（应该总是可用）
    try:
        from litaagent_std.litaagent_y import LitaAgentY
        agents.append(LitaAgentY)
        names.append("LitaAgentY")
    except ImportError as e:
        print(f"[警告] 无法加载 LitaAgentY: {e}")
    
    try:
        from litaagent_std.litaagent_yr import LitaAgentYR
        agents.append(LitaAgentYR)
        names.append("LitaAgentYR")
    except ImportError as e:
        print(f"[警告] 无法加载 LitaAgentYR: {e}")
    
    try:
        from litaagent_std.litaagent_cir import LitaAgentCIR
        agents.append(LitaAgentCIR)
        names.append("LitaAgentCIR")
    except ImportError as e:
        print(f"[警告] 无法加载 LitaAgentCIR: {e}")
    
    # 可选 LitaAgent（依赖 stable_baselines3 等）
    try:
        from litaagent_std.litaagent_n import LitaAgentN
        agents.append(LitaAgentN)
        names.append("LitaAgentN")
    except ImportError:
        pass  # 静默跳过，可能缺少 stable_baselines3
    
    try:
        from litaagent_std.litaagent_p import LitaAgentP
        agents.append(LitaAgentP)
        names.append("LitaAgentP")
    except ImportError:
        pass  # 静默跳过
    
    # HRL Agent（可选）
    try:
        from litaagent_std.hrl_xf import LitaAgentHRL
        agents.append(LitaAgentHRL)
        names.append("LitaAgentHRL")
    except ImportError:
        pass  # 静默跳过
    
    LITA_AGENTS = agents
    LITA_AGENT_NAMES = names
    return agents, names

# ============================================================================
# 外部 Agent
# ============================================================================
# PenguinAgent (2024 冠军)
try:
    from scml_agents.scml2024.standard.team_penguin.penguinagent import PenguinAgent
    PENGUIN_AVAILABLE = True
except ImportError:
    PenguinAgent = None
    PENGUIN_AVAILABLE = False

# 2025 Top Agents
try:
    from scml_agents import get_agents
    TOP_AGENTS_2025 = get_agents(2025, as_class=True, track="std")
except Exception as exc:
    print(f"[警告] 无法加载 2025 Top Agents: {exc}")
    TOP_AGENTS_2025: List[Type] = []

# ============================================================================
# Tracker（可选）
# ============================================================================
try:
    from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager
    from litaagent_std.tracker_mixin import create_tracked_agent
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    TrackerConfig = None
    TrackerManager = None
    create_tracked_agent = None


# ============================================================================
# 官方默认配置 (SCML 2025 Standard)
# ============================================================================
DEFAULT_CONFIGS = 20          # 官方配置数
DEFAULT_RUNS = 2              # 每配置运行次数
DEFAULT_MAX_TOP = 5           # Top Agents 数量（前 5 名）
FORCED_LOGS = 1.0             # 强制保存所有谈判日志（用于训练）
DEFAULT_PARALLELISM = "parallel"


def build_competitors(
    max_top: int = DEFAULT_MAX_TOP,
    use_tracker: bool = False,
    tracker_log_dir: str = ".",
) -> Tuple[List[Type], List[str]]:
    """
    构建参赛代理池。
    
    Args:
        max_top: 包含的 Top Agents 数量
        use_tracker: 是否为 LitaAgent 启用 Tracker
        tracker_log_dir: Tracker 日志目录
        
    Returns:
        (competitors, lita_names): 参赛者列表和 LitaAgent 名称列表
    """
    # 加载 LitaAgent（延迟加载，避免可选依赖问题）
    lita_bases, lita_names = _load_lita_agents()
    
    if not lita_bases:
        print("[警告] 没有可用的 LitaAgent！")
    
    # 是否包装 Tracker
    if use_tracker and TRACKER_AVAILABLE and create_tracked_agent is not None:
        lita_agents = [create_tracked_agent(cls, log_dir=tracker_log_dir) for cls in lita_bases]
    else:
        lita_agents = list(lita_bases)
    
    # 构建完整参赛池
    competitors: List[Type] = list(lita_agents)
    
    # PenguinAgent
    if PENGUIN_AVAILABLE and PenguinAgent is not None:
        competitors.append(PenguinAgent)
    
    # Top Agents (截断到 max_top)
    tops = TOP_AGENTS_2025[:max_top] if max_top else TOP_AGENTS_2025
    competitors.extend(tops)
    
    # RandomStdAgent 作为基准
    competitors.append(RandomStdAgent)
    
    # 去重（保持顺序）
    seen = set()
    unique = []
    for c in competitors:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    
    return unique, lita_names


def has_existing_tournament(tournament_dir: Path) -> bool:
    """判断是否已有配置（决定新建/恢复）。"""
    return any(
        (tournament_dir / fname).exists()
        for fname in (
            ASSIGNED_CONFIGS_PICKLE_FILE,
            ASSIGNED_CONFIGS_JSON_FILE,
            "assigned_configs",
        )
    )


def load_assignments(tournament_dir: Path):
    """加载已分配的 world 配置。"""
    for fname in (
        ASSIGNED_CONFIGS_PICKLE_FILE,
        ASSIGNED_CONFIGS_JSON_FILE,
        "assigned_configs",
    ):
        fpath = tournament_dir / fname
        if not fpath.exists():
            continue
        try:
            data = load(fpath)
            if data:
                return data
        except Exception:
            continue
    return []


def summarize_progress(tournament_dir: Path) -> Tuple[int, int]:
    """返回 (已完成 world 数, 总 world 数)。"""
    assignments = load_assignments(tournament_dir)
    if not assignments:
        return 0, 0
    total = len(assignments)
    done = 0
    for config_set in assignments:
        if not config_set:
            continue
        dir_name = config_set[0].get("__dir_name")
        if not dir_name:
            continue
        run_root = Path(dir_name).parent
        if (run_root / RESULTS_FILE).exists():
            done += 1
    return done, total


def find_existing_root(base: Path) -> Path | None:
    """查找已存在的比赛目录（支持 stage 目录）。"""
    if has_existing_tournament(base):
        return base
    stage_candidate = base.parent / f"{base.name}-stage-0001"
    if has_existing_tournament(stage_candidate):
        return stage_candidate
    for p in base.parent.glob(f"{base.name}-stage-*"):
        if has_existing_tournament(p):
            return p
    return None


def prepare_tournament(
    tournament_dir: Path,
    competitors: List[Type],
    n_configs: int,
    n_runs_per_world: int,
    forced_logs_fraction: float,
    parallelism: str,
    verbose: bool,
) -> Tuple[bool, Path]:
    """
    创建或恢复比赛配置。
    
    Returns:
        (created, tournament_root): 是否新创建，以及实际比赛目录
    """
    existing_root = find_existing_root(tournament_dir)
    if existing_root:
        if verbose:
            print(f"[恢复] 已发现配置，使用 {existing_root}")
        return False, existing_root
    
    if tournament_dir.exists():
        raise RuntimeError(
            f"{tournament_dir} 已存在但缺少配置。\n"
            f"请更换 --output-dir，或确认安全后手动删除该目录。"
        )

    base_dir = tournament_dir.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[创建] 生成比赛配置: {tournament_dir}")
        print(f"       n_configs={n_configs}, n_runs_per_world={n_runs_per_world}")
    
    # 使用 anac2024_std 生成配置（configs_only=True 不运行）
    configs_path = anac2024_std(
        competitors=competitors,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        tournament_path=str(base_dir),
        name=tournament_dir.name,
        forced_logs_fraction=forced_logs_fraction,
        parallelism=parallelism,
        compact=False,
        configs_only=True,
        verbose=verbose,
        print_exceptions=True,
    )
    
    # 确定实际的比赛根目录
    root = tournament_dir
    try:
        if configs_path is not None:
            configs_path = Path(configs_path)
            root = configs_path.parent
    except Exception:
        pass
    
    return True, root


def setup_tracker(tracker_dir: Path) -> None:
    """配置 Tracker。"""
    if not TRACKER_AVAILABLE:
        return
    tracker_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)
    if TrackerManager is not None:
        TrackerManager._loggers.clear()
    if TrackerConfig is not None:
        TrackerConfig.configure(enabled=True, log_dir=str(tracker_dir), console_echo=False)


def save_results(
    output_dir: Path,
    results,
    competitors: List[Type],
    lita_names: List[str],
    config: dict,
) -> None:
    """保存比赛结果摘要。"""
    data = {
        "runner": "run_default_std",
        "tournament_path": str(output_dir),
        "competitors": [c.__name__ for c in competitors],
        "lita_agents": lita_names,
        "winners": [w.split(".")[-1] for w in getattr(results, "winners", [])],
        "timestamp": datetime.now().isoformat(),
        **config,
    }
    (output_dir / "tournament_results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_rankings(results, lita_names: List[str], verbose: bool) -> None:
    """打印排名结果。"""
    if not verbose:
        return
    if getattr(results, "total_scores", None) is None:
        return
    
    print("\n" + "=" * 60)
    print("📊 比赛结果排名")
    print("=" * 60)
    
    sorted_scores = results.total_scores.sort_values("score", ascending=False)
    for rank, (_, row) in enumerate(sorted_scores.iterrows(), 1):
        agent_name = row["agent_type"].split(".")[-1]
        # 标记 LitaAgent
        tag = " ⭐" if any(name in agent_name for name in lita_names) else ""
        print(f"  {rank:2d}. {agent_name:30s} {row['score']:.4f}{tag}")
    
    print("=" * 60)


def run_tournament_resumable(
    n_configs: int,
    n_runs: int,
    max_top: int,
    output_dir: Path | None,
    parallelism: str,
    use_tracker: bool,
    use_visualizer: bool,
    auto_collect: bool,
    verbose: bool,
) -> Path:
    """
    运行可断点续跑的比赛。
    
    Returns:
        实际比赛目录
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_dir = output_dir or Path(f"tournament_history/std_default_{ts}")
    
    # 构建参赛池
    tracker_dir = tournament_dir / "tracker_logs" if use_tracker else Path(".")
    competitors, lita_names = build_competitors(
        max_top=max_top,
        use_tracker=use_tracker,
        tracker_log_dir=str(tracker_dir),
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("🎯 SCML 2025 Standard 默认 Runner")
        print("=" * 60)
        print(f"📋 参赛代理: {len(competitors)} 个")
        print(f"   LitaAgent: {lita_names}")
        print(f"   外部 Agent: {[c.__name__ for c in competitors if c.__name__ not in lita_names]}")
        print(f"📊 配置: n_configs={n_configs}, n_runs={n_runs}")
        print(f"🔧 选项: tracker={use_tracker}, visualizer={use_visualizer}, auto_collect={auto_collect}")
        print("=" * 60 + "\n")
    
    # 准备比赛配置
    created, tournament_root = prepare_tournament(
        tournament_dir=tournament_dir,
        competitors=competitors,
        n_configs=n_configs,
        n_runs_per_world=n_runs,
        forced_logs_fraction=FORCED_LOGS,
        parallelism=parallelism,
        verbose=verbose,
    )
    
    # 配置 Tracker（如果启用）
    if use_tracker:
        tracker_dir = tournament_root / "tracker_logs"
        os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)
        setup_tracker(tracker_dir)
        if verbose:
            print(f"[Tracker] 启用，日志目录: {tracker_dir}")
    
    # 显示进度
    done, total = summarize_progress(tournament_root)
    if total and verbose:
        print(f"[进度] 已完成 {done}/{total} 个 world ({done/total:.1%})")
    
    if verbose:
        print(f"[运行] 启动比赛: {tournament_root}")
        print(f"       parallelism={parallelism}")
    
    # 运行比赛
    run_tournament(
        tournament_path=str(tournament_root),
        world_generator=anac2024_std_world_generator,
        score_calculator=balance_calculator_std,
        parallelism=parallelism,
        verbose=verbose,
        compact=False,
        print_exceptions=True,
    )
    
    # 评估结果
    if verbose:
        print("[评估] 汇总比赛结果...")
    
    results = evaluate_tournament(
        tournament_path=str(tournament_root),
        metric=truncated_mean,
        verbose=verbose,
        recursive=True,
    )
    
    # 打印排名
    print_rankings(results, lita_names, verbose)
    
    # 保存结果摘要
    save_results(
        output_dir=tournament_root,
        results=results,
        competitors=competitors,
        lita_names=lita_names,
        config={
            "n_configs": n_configs,
            "n_runs_per_world": n_runs,
            "max_top": max_top,
            "parallelism": parallelism,
            "use_tracker": use_tracker,
            "use_visualizer": use_visualizer,
            "auto_collect": auto_collect,
        },
    )
    
    # 自动归集（后处理）
    if auto_collect:
        try:
            from scml_analyzer.postprocess import postprocess_tournament
            if verbose:
                print("[归集] 汇总日志到 tournament_history/...")
            postprocess_tournament(
                output_dir=tournament_root,
                start_visualizer=False,
                visualizer_port=None,
            )
        except ImportError:
            if verbose:
                print("[归集] scml_analyzer.postprocess 不可用，跳过自动归集")
        except Exception as e:
            if verbose:
                print(f"[归集] 后处理失败: {e}")
    
    # 启动 Visualizer（如果启用）
    if use_visualizer:
        try:
            from scml_analyzer.visualizer import start_visualizer
            if verbose:
                print("[Visualizer] 启动可视化服务器...")
            start_visualizer(port=8080)
        except ImportError:
            if verbose:
                print("[Visualizer] scml_analyzer.visualizer 不可用")
        except Exception as e:
            if verbose:
                print(f"[Visualizer] 启动失败: {e}")
    
    if verbose:
        print(f"\n✅ 比赛完成！结果保存在: {tournament_root}")
    
    return tournament_root


def main():
    parser = argparse.ArgumentParser(
        description="🎯 SCML 2025 Standard 默认 Runner（用于 HRL 训练数据采集）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 官方规模（默认）
  python runners/run_default_std.py
  
  # 快速测试
  python runners/run_default_std.py --quick
  
  # 自定义规模
  python runners/run_default_std.py --configs 10 --runs 1
  
  # 启用 Tracker 和 Visualizer
  python runners/run_default_std.py --tracker --visualizer
  
  # 断点续跑
  python runners/run_default_std.py --output-dir tournament_history/my_run
        """,
    )
    
    # 规模参数
    parser.add_argument(
        "--configs", type=int, default=DEFAULT_CONFIGS,
        help=f"World 配置数量 (default: {DEFAULT_CONFIGS})",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"每个 world 运行次数 (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--max-top", type=int, default=DEFAULT_MAX_TOP,
        help=f"包含的 Top Agents 数量 (default: {DEFAULT_MAX_TOP})",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式 (configs=3, runs=1)",
    )
    
    # 功能开关
    parser.add_argument(
        "--tracker", action="store_true",
        help="启用 Tracker（记录 LitaAgent 协商过程）",
    )
    parser.add_argument(
        "--visualizer", action="store_true",
        help="完成后启动 Visualizer 可视化服务器",
    )
    parser.add_argument(
        "--no-auto-collect", action="store_true",
        help="禁用自动归集（不执行 postprocess）",
    )
    
    # 输出控制
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（复用即可断点续跑）",
    )
    parser.add_argument(
        "--parallelism", type=str, default=DEFAULT_PARALLELISM,
        help=f"并行模式 (default: {DEFAULT_PARALLELISM})",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="静默模式（减少输出）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细模式（增加输出）",
    )
    
    args = parser.parse_args()
    
    # 快速模式覆盖
    if args.quick:
        args.configs = 3
        args.runs = 1
    
    # verbose 优先级：--verbose > 默认 > --quiet
    verbose = True  # 默认显示
    if args.quiet:
        verbose = False
    if args.verbose:
        verbose = True
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_tournament_resumable(
        n_configs=args.configs,
        n_runs=args.runs,
        max_top=args.max_top,
        output_dir=output_dir,
        parallelism=args.parallelism,
        use_tracker=args.tracker,
        use_visualizer=args.visualizer,
        auto_collect=not args.no_auto_collect,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
