#!/usr/bin/env python
"""
🎯 默认 SCML 2025 Standard 比赛 Runner（用于 HRL 训练数据采集）

这是 LitaAgent 项目的默认 runner，用于运行官方规模的 SCML 2025 Standard 比赛
并采集谈判日志以供 HRL 训练使用。

核心特性：
- ✅ 官方规模: 默认使用 SCML 2025 Standard 官方环境和规模
- ✅ 完整参赛池: 所有 LitaAgent（不含 HRL）+ SCML 2025 Top5 + SCML 2024 Top5
- ✅ 可配置规模: 支持通过参数指定更小的比赛规模用于测试
- ✅ 自动归集: 运行完成后自动归集数据到 tournament_history/
- ✅ 强制启用 Tracker: 所有代理均为动态生成的 Tracked 版本
- ✅ 默认不启用 Visualizer: 无需人工观察时节省资源（不提供启动开关）

用法：
    # 1. 默认官方规模（推荐用于正式数据采集）
    python runners/run_default_std.py
    
    # 2. 快速测试（3 个配置，1 轮）
    python runners/run_default_std.py --quick
    
    # 3. 自定义规模
    python runners/run_default_std.py --configs 10 --runs 1 --max-worlds-per-config 10
    
    # 4. 自定义输出目录
    python runners/run_default_std.py --output-dir tournament_history/my_run
    
    # 5. 静默模式（减少输出）
    python runners/run_default_std.py --quiet

环境：
- SCML 2025 Standard World（Futures Market）
- 步数范围: 50-200（官方随机）
- 工厂数: 5-15（官方随机）
- 层级数: 3-5（官方随机）
"""

from __future__ import annotations

import argparse
import math
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
from scml_agents import get_agents

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
    except ImportError as e:
        print(f"[警告] 无法加载 LitaAgentN: {e}")
    
    try:
        from litaagent_std.litaagent_p import LitaAgentP
        agents.append(LitaAgentP)
        names.append("LitaAgentP")
    except ImportError as e:
        print(f"[警告] 无法加载 LitaAgentP: {e}")
    
    LITA_AGENTS = agents
    LITA_AGENT_NAMES = names
    return agents, names

# ============================================================================
# 外部 Agent
# ============================================================================
# 2025/2024 Top Agents
try:
    TOP_AGENTS_2025 = list(get_agents(2025, as_class=True, track="std", top_only=5))
    TOP_AGENTS_2024 = list(get_agents(2024, as_class=True, track="std", top_only=5))
except Exception as exc:
    raise RuntimeError(f"无法加载 SCML Top Agents: {exc}")

# ============================================================================
# Tracker（必须）
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
DEFAULT_MAX_TOP_2025 = 5      # 2025 Top Agents 数量
DEFAULT_MAX_TOP_2024 = 5      # 2024 Top Agents 数量
FORCED_LOGS = 1.0             # 强制保存所有谈判日志（用于训练）
DEFAULT_PARALLELISM = "loky"
DEFAULT_MAX_WORLDS_PER_CONFIG: int | None = None


def _filter_legacy_agents(agents: List[Type]) -> List[Type]:
    filtered: List[Type] = []
    for cls in agents:
        module = getattr(cls, "__module__", "")
        if "scml2020" in module.lower():
            continue
        filtered.append(cls)
    return filtered


def _ensure_tracked(base_cls: Type, tracked_cls: Type) -> Type:
    if tracked_cls is base_cls or tracked_cls.__name__ == base_cls.__name__:
        raise RuntimeError(f"无法为 {base_cls.__name__} 创建动态 Tracked 版本")
    return tracked_cls


def _estimate_competitor_sets(
    n_competitors: int,
    n_per_world: int,
    round_robin: bool,
) -> int:
    if n_per_world >= n_competitors:
        return 1
    if round_robin:
        return math.comb(n_competitors, n_per_world)
    return math.ceil(n_competitors / n_per_world)


def _strip_adapter_prefix(agent_type: str) -> str:
    if not isinstance(agent_type, str):
        return agent_type
    if "DefaultOneShotAdapter" in agent_type and ":" in agent_type:
        return agent_type.split(":", 1)[1]
    if "DefaultStdAdapter" in agent_type and ":" in agent_type:
        return agent_type.split(":", 1)[1]
    return agent_type


def _patch_score_calculator() -> None:
    import scml.utils as scml_utils

    if getattr(scml_utils, "_litaagent_score_patch", False):
        return

    original = scml_utils.balance_calculator_std

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            if result is not None and getattr(result, "types", None):
                result.types = [_strip_adapter_prefix(t) for t in result.types]
        except Exception:
            pass
        return result

    scml_utils.balance_calculator_std = wrapped
    scml_utils._litaagent_score_patch = True


def build_competitors(
    max_top_2025: int = DEFAULT_MAX_TOP_2025,
    max_top_2024: int = DEFAULT_MAX_TOP_2024,
    tracker_log_dir: str = ".",
) -> Tuple[List[Type], List[str]]:
    """
    构建参赛代理池。
    
    Args:
        max_top_2025: 2025 Top Agents 数量
        max_top_2024: 2024 Top Agents 数量
        tracker_log_dir: Tracker 日志目录
        
    Returns:
        (competitors, lita_names): 参赛者列表和 LitaAgent 名称列表
    """
    if not TRACKER_AVAILABLE or create_tracked_agent is None:
        raise RuntimeError("必须安装 scml_analyzer 以启用全量 Tracker")

    # 加载 LitaAgent（不含 HRL）
    lita_bases, lita_names = _load_lita_agents()
    
    if not lita_bases:
        print("[警告] 没有可用的 LitaAgent！")
    expected = {"LitaAgentY", "LitaAgentYR", "LitaAgentCIR", "LitaAgentN", "LitaAgentP"}
    missing = expected - set(lita_names)
    if missing:
        raise RuntimeError(f"LitaAgent 缺失: {sorted(missing)}，请确认依赖已安装")

    # 所有 LitaAgent 使用动态 Tracked 版本
    lita_agents = [
        _ensure_tracked(cls, create_tracked_agent(cls, log_dir=tracker_log_dir))
        for cls in lita_bases
    ]
    lita_display_names = [c.__name__ for c in lita_agents]
    
    # 构建完整参赛池
    competitors: List[Type] = list(lita_agents)
    
    # Top Agents (2025/2024)
    tops_2025 = TOP_AGENTS_2025[:max_top_2025] if max_top_2025 else TOP_AGENTS_2025
    tops_2024 = TOP_AGENTS_2024[:max_top_2024] if max_top_2024 else TOP_AGENTS_2024
    tops_2025 = _filter_legacy_agents(tops_2025)
    tops_2024 = _filter_legacy_agents(tops_2024)
    lita_base_names = {c.__name__ for c in lita_bases}
    tops = [cls for cls in list(tops_2025) + list(tops_2024) if cls.__name__ not in lita_base_names]
    for cls in tops:
        try:
            competitors.append(create_tracked_agent(cls, log_dir=tracker_log_dir))
        except Exception as exc:
            raise RuntimeError(f"无法为 {cls.__name__} 创建动态 Tracked 版本: {exc}")
    
    # 去重（保持顺序）
    seen = set()
    unique = []
    for c in competitors:
        key = (c.__module__, c.__name__)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique, lita_display_names


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
    max_worlds_per_config: int | None,
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
        if max_worlds_per_config is not None:
            print(f"       max_worlds_per_config={max_worlds_per_config}")
    
    # 使用 anac2024_std 生成配置（configs_only=True 不运行）
    configs_path = anac2024_std(
        competitors=competitors,
        n_configs=n_configs,
        n_runs_per_world=n_runs_per_world,
        max_worlds_per_config=max_worlds_per_config,
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
    max_top_2025: int,
    max_top_2024: int,
    n_competitors_per_world: int | None,
    round_robin: bool,
    output_dir: Path | None,
    parallelism: str,
    parallelism_label: str,
    max_worlds_per_config: int | None,
    verbose: bool,
) -> Path:
    """
    运行可断点续跑的比赛。
    
    Returns:
        实际比赛目录
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_dir = output_dir or Path(f"tournament_history/std_default_{ts}")
    tournament_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建参赛池
    tracker_dir = tournament_dir / "tracker_logs"
    competitors, lita_names = build_competitors(
        max_top_2025=max_top_2025,
        max_top_2024=max_top_2024,
        tracker_log_dir=str(tracker_dir),
    )
    n_per_world = n_competitors_per_world or len(competitors)
    if max_worlds_per_config is None:
        max_worlds_per_config = n_per_world
    if not round_robin and len(competitors) % n_per_world != 0:
        raise RuntimeError(
            f"n_competitors_per_world={n_per_world} 不能整除参赛数量 {len(competitors)}，"
            f"请调整或启用 --round-robin"
        )
    
    if verbose:
        print("\n" + "=" * 60)
        print("🎯 SCML 2025 Standard 默认 Runner")
        print("=" * 60)
        print(f"📋 参赛代理: {len(competitors)} 个")
        print(f"   LitaAgent: {lita_names}")
        print(f"   外部 Agent: {[c.__name__ for c in competitors if c.__name__ not in lita_names]}")
        print(f"📊 配置: n_configs={n_configs}, n_runs={n_runs}")
        if max_worlds_per_config is not None:
            n_sets = _estimate_competitor_sets(len(competitors), n_per_world, round_robin)
            approx_worlds = n_configs * n_runs * max_worlds_per_config * n_sets
            print(f"🧮 约束: max_worlds_per_config={max_worlds_per_config} (≈ {approx_worlds} worlds)")
        print("🔧 选项: tracker=True, visualizer=False, auto_collect=True")
        print(f"⚙️  并行: {parallelism_label}")
        print("=" * 60 + "\n")
    
    # 配置 Tracker（必须启用）
    tracker_dir = tournament_dir / "tracker_logs"
    os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)
    setup_tracker(tracker_dir)
    if verbose:
        print(f"[Tracker] 启用，日志目录: {tracker_dir}")
    
    if verbose:
        print(f"[运行] 启动比赛: {tournament_dir}")
        print(f"       parallelism={parallelism_label}")
    
    _patch_score_calculator()
    
    results = anac2024_std(
        competitors=competitors,
        n_configs=n_configs,
        n_runs_per_world=n_runs,
        max_worlds_per_config=max_worlds_per_config,
        tournament_path=str(tournament_dir),
        forced_logs_fraction=FORCED_LOGS,
        parallelism=parallelism,
        round_robin=round_robin,
        n_competitors_per_world=n_per_world,
        name=f"StdDefault_{ts}",
        verbose=verbose,
        compact=False,
        print_exceptions=True,
    )
    
    # 打印排名
    print_rankings(results, lita_names, verbose)
    
    # 保存结果摘要
    save_results(
        output_dir=tournament_dir,
        results=results,
        competitors=competitors,
        lita_names=lita_names,
        config={
            "n_configs": n_configs,
            "n_runs_per_world": n_runs,
            "max_worlds_per_config": max_worlds_per_config,
            "n_competitors_per_world": n_per_world,
            "round_robin": round_robin,
            "max_top_2025": max_top_2025,
            "max_top_2024": max_top_2024,
            "parallelism": parallelism_label,
            "tracker": True,
            "visualizer": False,
            "auto_collect": True,
        },
    )
    
    # 自动归集（后处理）
    try:
        from scml_analyzer.postprocess import postprocess_tournament
        if verbose:
            print("[归集] 汇总日志到 tournament_history/...")
        postprocess_tournament(
            output_dir=tournament_dir,
            start_visualizer=False,
            visualizer_port=None,
        )
    except ImportError:
        if verbose:
            print("[归集] scml_analyzer.postprocess 不可用，跳过自动归集")
    except Exception as e:
        if verbose:
            print(f"[归集] 后处理失败: {e}")
    
    if verbose:
        print(f"\n✅ 比赛完成！结果保存在: {tournament_dir}")
    
    return tournament_dir


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
  
  # 自定义输出目录
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
        "--max-top-2025", type=int, default=DEFAULT_MAX_TOP_2025,
        help=f"2025 Top Agents 数量 (default: {DEFAULT_MAX_TOP_2025})",
    )
    parser.add_argument(
        "--max-top-2024", type=int, default=DEFAULT_MAX_TOP_2024,
        help=f"2024 Top Agents 数量 (default: {DEFAULT_MAX_TOP_2024})",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式 (configs=3, runs=1)",
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
        "--max-worlds-per-config", type=int, default=DEFAULT_MAX_WORLDS_PER_CONFIG,
        help="限制每个配置的最大 world 数量（用于压缩总规模）",
    )
    parser.add_argument(
        "--n-competitors-per-world", type=int, default=None,
        help="每个 world 的参赛者数量（默认使用全部参赛者）",
    )
    parser.add_argument(
        "--round-robin", action="store_true",
        help="启用 round-robin（组合爆炸，慎用）",
    )
    parser.add_argument(
        "--target-worlds", type=int, default=None,
        help="目标总 world 数量（自动折算为 max_worlds_per_config）",
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
    
    if args.target_worlds and args.max_worlds_per_config is None:
        denom = max(1, args.configs * args.runs)
        args.max_worlds_per_config = max(1, math.ceil(args.target_worlds / denom))

    parallelism_label = args.parallelism
    parallelism = args.parallelism
    if args.parallelism.startswith("loky"):
        os.environ["SCML_PARALLELISM"] = args.parallelism
        parallelism_label = f"{args.parallelism}（通过 SCML_PARALLELISM）"
        parallelism = "parallel"

    run_tournament_resumable(
        n_configs=args.configs,
        n_runs=args.runs,
        max_top_2025=args.max_top_2025,
        max_top_2024=args.max_top_2024,
        n_competitors_per_world=args.n_competitors_per_world,
        round_robin=args.round_robin,
        output_dir=output_dir,
        parallelism=parallelism,
        parallelism_label=parallelism_label,
        max_worlds_per_config=args.max_worlds_per_config,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
