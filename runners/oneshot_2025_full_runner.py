"""
SCML 2025 OneShot 赛道比赛运行器
================================

本脚本用于运行 SCML 2025 OneShot 赛道的比赛。

## 官方比赛规模 (Official Competition Scale)
根据 SCML 官方文档 (https://scml.readthedocs.io/en/latest/tutorials/01.run_scml2020.html):

    "Note that in the real competition we use thousands of configurations and longer
    simulation steps (e.g. 50 ≤ n_steps ≤ 500)."

因此官方比赛规模参数为:
    - n_configs: 1000+ (官方用 "thousands of configurations")
    - n_runs_per_world: 1 (通常每个配置运行一次)
    - n_steps: 50-500 (每场仿真的天数/步数)
    - round_robin: True (使所有参赛者两两对阵)

## 运行官方规模比赛
要运行一场和官方规模一致的完整比赛:

    python runners/oneshot_2025_full_runner.py --official

这将使用以下参数:
    - n_configs=1000
    - n_runs_per_world=1  
    - n_steps=100
    - round_robin=True

注意: 官方规模比赛需要大量时间(数小时到数天)和计算资源。

## 快速测试 (Quick Test)
要进行快速测试:

    python runners/oneshot_2025_full_runner.py --configs 5 --runs 1 --steps 10

## 中等规模测试 (Medium Scale)
要进行中等规模测试:

    python runners/oneshot_2025_full_runner.py --configs 50 --runs 1 --steps 50
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Type

from runners.loky_patch import enable_loky_executor

enable_loky_executor()

from negmas.helpers import get_full_type_name
from negmas.tournaments.tournaments import tournament
from scml.oneshot.agent import OneShotAgent
from scml.utils import (
    DefaultAgentsOneShot,
    anac_assigner_oneshot,
    anac_config_generator_oneshot,
    anac_oneshot_world_generator,
    balance_calculator_oneshot,
)

from scml.oneshot.agents import (
    GreedyOneShotAgent,
    SyncRandomOneShotAgent,
    GreedySyncAgent,
    SingleAgreementAspirationAgent,
)

# scml_agents 包的可用性标志 (延迟导入以避免模块级错误)
SCML_AGENTS_AVAILABLE = None  # 将在运行时检查


def _get_scml_agents_module():
    """延迟导入 scml_agents 以避免模块加载时的错误。"""
    global SCML_AGENTS_AVAILABLE
    if SCML_AGENTS_AVAILABLE is None:
        try:
            from scml_agents import get_agents
            SCML_AGENTS_AVAILABLE = True
            return get_agents
        except (ImportError, Exception) as e:
            print(f"[WARN] Failed to import scml_agents: {e}")
            SCML_AGENTS_AVAILABLE = False
            return None
    elif SCML_AGENTS_AVAILABLE:
        from scml_agents import get_agents
        return get_agents
    return None


from litaagent_os.agent import LitaAgentOS
from litaagent_std.tracker_mixin import (
    TRACKER_AVAILABLE,
    create_tracked_agent,
    save_tracker_data,
    setup_tracker_for_tournament,
)


def _load_oneshot_2025_agents() -> List[Type[OneShotAgent]]:
    """加载 2025 年提交的 OneShot 代理。
    
    注意: scml_agents 包可能有依赖问题，此函数会优雅地处理失败情况。
    """
    agents: List[Type[OneShotAgent]] = []
    
    try:
        from scml_agents.scml2025 import oneshot as oneshot_pkg
        
        names = getattr(oneshot_pkg, "__all__", [])
        for name in names:
            try:
                obj = getattr(oneshot_pkg, name)
            except Exception:
                continue
            if inspect.isclass(obj) and issubclass(obj, OneShotAgent):
                module_name = getattr(obj, "__module__", "")
                if "litaagent_std" in module_name:
                    continue
                agents.append(obj)
    except ImportError as e:
        print(f"[WARN] Failed to load scml_agents.scml2025: {e}")
        print("[INFO] Continuing without 2025 agents from scml_agents package")
    except Exception as e:
        print(f"[WARN] Unexpected error loading 2025 agents: {e}")

    unique: List[Type[OneShotAgent]] = []
    seen = set()
    for cls in agents:
        key = (cls.__module__, cls.__name__)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cls)
    return unique


def _resolve_oneshot_year(preferred: int = 2025) -> int:
    try:
        import scml.oneshot.world as world_mod
    except Exception:
        return 2024
    candidates = [preferred, 2024, 2023, 2022, 2021, 2020]
    for year in candidates:
        if hasattr(world_mod, f"SCML{year}OneShotWorld"):
            return year
    return 2024


def _find_stage_dir(base_dir: Path, name_prefix: str) -> Path | None:
    candidates = list(base_dir.glob(f"{name_prefix}-stage-*"))
    if not candidates:
        candidates = list(base_dir.glob("*-stage-*"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _import_to_history(stage_dir: Path, tracker_dir: Path | None) -> str | None:
    try:
        from scml_analyzer.history import import_tournament
    except Exception:
        return None
    if tracker_dir is not None and tracker_dir.exists():
        return import_tournament(str(stage_dir), tracker_dir=str(tracker_dir))
    return import_tournament(str(stage_dir))


def _create_tournament_info(base_dir: Path, run_name: str, tracker_dir: Path | None, competitors: list = None) -> None:
    """在 base_dir 创建 tournament_info.json (与 history.py import_tournament 格式对齐)"""
    import json
    import csv
    
    # 从 run_name 解析时间戳 (如 20260110_021745_oneshot -> 2026-01-10 02:17:45)
    timestamp_str = ""
    if len(run_name) >= 15 and run_name[:8].isdigit():
        date_part = run_name[:8]  # 20260110
        time_part = run_name[9:15]  # 021745
        timestamp_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
    
    # 查找 stage 目录
    stage_dirs = list(base_dir.glob("*-stage-*"))
    source_dir = str(stage_dirs[0]) if stage_dirs else str(base_dir)
    
    info = {
        "id": run_name,
        "source_dir": source_dir,
        "imported_at": datetime.now().isoformat(),
        "track": "oneshot",
        "settings": {},
        "competitors": [],
        "n_competitors": 0,
        "results": {
            "n_completed": 0,
            "total_duration_seconds": 0,
            "winner": "N/A",
            "winner_score": 0.0,
        },
        "timestamp": timestamp_str,
    }
    
    # 从 params.json 读取 settings
    params_file = base_dir / "params.json"
    if params_file.exists():
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            info["settings"] = {
                "n_configs": params.get("n_configs"),
                "n_runs_per_world": params.get("n_runs_per_world"),
                "n_steps": params.get("n_steps"),
                "n_worlds": params.get("n_worlds"),
                "n_processes": params.get("n_processes"),
                "parallelism": params.get("parallelism"),
                "world_generator": params.get("world_generator"),
                "score_calculator": params.get("score_calculator"),
                "publish_exogenous_summary": params.get("publish_exogenous_summary"),
                "publish_trading_prices": params.get("publish_trading_prices"),
                "min_factories_per_level": params.get("min_factories_per_level"),
                "n_agents_per_competitor": params.get("n_agents_per_competitor"),
                "n_competitors_per_world": params.get("n_competitors_per_world"),
            }
        except Exception:
            pass
    
    # 设置 competitors 列表
    if competitors:
        info["competitors"] = [c.__name__ if hasattr(c, '__name__') else str(c) for c in competitors]
        info["n_competitors"] = len(competitors)
    
    # 优先用 score_stats.csv 的 mean 做排名
    score_stats_file = base_dir / "score_stats.csv"
    if score_stats_file.exists():
        try:
            with open(score_stats_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                rows.sort(key=lambda r: float(r.get("mean", 0) or 0), reverse=True)
                info["n_competitors"] = len(rows)
                info["results"]["winner"] = rows[0].get("agent_type", "unknown")
                info["results"]["winner_score"] = float(rows[0].get("mean", 0) or 0)
                if not competitors:
                    info["competitors"] = [r.get("agent_type", "").split(":")[-1] for r in rows]
        except Exception:
            pass
    # 回退 total_scores.csv
    scores_file = base_dir / "total_scores.csv"
    if scores_file.exists() and info["results"]["winner"] == "N/A":
        try:
            with open(scores_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    info["n_competitors"] = len(rows)
                    info["results"]["winner"] = rows[0].get("agent_type", "unknown")
                    info["results"]["winner_score"] = float(rows[0].get("score", 0))
                    if not competitors:
                        info["competitors"] = [r.get("agent_type", "").split(":")[-1] for r in rows]
        except Exception:
            pass
    
    # 从 world_stats.csv 读取 world 数量和运行时间
    world_stats_file = base_dir / "world_stats.csv"
    if world_stats_file.exists():
        try:
            with open(world_stats_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                info["results"]["n_completed"] = len(rows)
                # 尝试计算总时间
                total_time = sum(float(r.get("time", 0)) for r in rows if r.get("time"))
                if total_time > 0:
                    info["results"]["total_duration_seconds"] = total_time
        except Exception:
            pass
    
    info_file = base_dir / "tournament_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SCML 2025 OneShot tournament",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Official scale (takes hours):
    python runners/oneshot_2025_full_runner.py --official
  
  Quick test:
    python runners/oneshot_2025_full_runner.py --configs 5 --runs 1 --steps 10
  
  Medium scale:
    python runners/oneshot_2025_full_runner.py --configs 50 --runs 1 --steps 50
"""
    )
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: tournament_history/<timestamp>_oneshot)")
    parser.add_argument("--name", type=str, default=None,
                       help="Tournament name")
    
    # 官方规模预设
    parser.add_argument("--official", action="store_true",
                       help="Use official competition scale: 1000 configs, 1 run, 100 steps")
    
    parser.add_argument("--configs", type=int, default=5,
                       help="Number of configurations (official: 1000+)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs per world (official: 1)")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of simulation steps/days (official: 50-500)")
    parser.add_argument("--n-competitors-per-world", type=int, default=0,
                       help="Competitors per world (0=all)")
    parser.add_argument("--max-worlds-per-config", type=int, default=None,
                       help="Max worlds per config")
    parser.add_argument("--parallelism", type=str, default="loky")
    parser.add_argument("--round-robin", dest="round_robin", action="store_true")
    parser.add_argument("--no-round-robin", dest="round_robin", action="store_false")
    parser.set_defaults(round_robin=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--log-negotiations", action="store_true")
    parser.add_argument("--log-ufuns", action="store_true")
    parser.add_argument("--forced-logs-fraction", type=float, default=1.0)
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # 如果使用 --official 预设，覆盖相关参数
    if args.official:
        print("[INFO] Using official competition scale parameters")
        args.configs = 1000
        args.runs = 1
        args.steps = 100
        args.round_robin = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 默认使用 tournament_history 目录，便于自动保存和管理
    default_history_dir = Path(__file__).resolve().parent.parent / "tournament_history"
    base_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (default_history_dir / f"{timestamp}_oneshot").resolve()
    )
    run_name = args.name or f"{timestamp}_oneshot"
    log_file = base_dir / "tournament_run.log"

    if not args.foreground:
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] background run, log: {log_file}")
        log_handle = open(log_file, "w", buffering=1, encoding="utf-8")
        sys.stdout = log_handle
        sys.stderr = log_handle
        print(f"[INFO] start: {timestamp}")
        print(f"[INFO] output_dir={base_dir}")
        print(f"[INFO] name={run_name}")

    if args.parallelism.startswith("loky"):
        loky_mode = args.parallelism if ":" in args.parallelism else "loky:1.0"
        os.environ["SCML_PARALLELISM"] = loky_mode
        parallelism = "parallel"
    else:
        parallelism = args.parallelism

    # 加载 2025 OneShot 代理 + CautiousOneShotAgent (2024)
    past_agents: List[Type[OneShotAgent]] = []
    get_agents = _get_scml_agents_module()
    if get_agents is not None:
        try:
            # 加载 2025 所有 OneShot 代理
            try:
                agents_2025 = get_agents(2025, track="oneshot", winners_only=False, as_class=True)
                if agents_2025:
                    for agent_cls in agents_2025:
                        module_name = getattr(agent_cls, "__module__", "")
                        # 排除自己的 litaagent 系列
                        if "litaagent" in module_name.lower():
                            continue
                        if issubclass(agent_cls, OneShotAgent):
                            past_agents.append(agent_cls)
                    print(f"[INFO] Loaded {len(past_agents)} agents from 2025 OneShot")
            except Exception as e:
                print(f"[WARN] Failed to load 2025 agents: {e}")
            
            # 加载 CautiousOneShotAgent (2024 冠军)
            try:
                agents_2024 = get_agents(2024, track="oneshot", winners_only=False, as_class=True)
                cautious_agent = None
                for agent_cls in agents_2024:
                    if agent_cls.__name__ == "CautiousOneShotAgent":
                        cautious_agent = agent_cls
                        break
                if cautious_agent:
                    past_agents.append(cautious_agent)
                    print(f"[INFO] Loaded CautiousOneShotAgent from 2024")
            except Exception as e:
                print(f"[WARN] Failed to load CautiousOneShotAgent: {e}")
                
            print(f"[INFO] Total {len(past_agents)} past OneShot agents loaded")
        except Exception as e:
            print(f"[WARN] Failed to load past agents: {e}")
    else:
        print("[INFO] scml_agents not available, using built-in agents only")
    
    if TRACKER_AVAILABLE:
        setup_tracker_for_tournament(str(base_dir), enabled=True)
        tracker_dir = base_dir / "tracker_logs"
        TrackedLita = create_tracked_agent(LitaAgentOS, log_dir=str(tracker_dir))
        # Track LitaAgentOS, CautiousOneShotAgent 和 RChan (如果存在)
        tracked_past_agents = []
        agents_to_track = ["CautiousOneShotAgent", "RChan", "RChanAgent"]  # 可能的 RChan 名称
        for agent_cls in past_agents:
            if agent_cls.__name__ in agents_to_track:
                # Track 这些代理
                try:
                    tracked_agent = create_tracked_agent(agent_cls, log_dir=str(tracker_dir))
                    tracked_past_agents.append(tracked_agent)
                    print(f"[INFO] Tracking {agent_cls.__name__}")
                except Exception as e:
                    print(f"[WARN] Failed to create tracked agent for {agent_cls.__name__}: {e}")
                    tracked_past_agents.append(agent_cls)
            else:
                # 其他代理不 track
                tracked_past_agents.append(agent_cls)
    else:
        tracker_dir = None
        TrackedLita = LitaAgentOS
        tracked_past_agents = past_agents

    agents_2025 = _load_oneshot_2025_agents()
    # 构建参赛者列表: TrackedLita + past agents (包含 tracked CautiousOneShotAgent)
    # 注意: agents_2025 已经包含在 past_agents 中 (year=2025 已加载)
    competitors: List[Type[OneShotAgent]] = [TrackedLita]
    competitors.extend(tracked_past_agents)
    # 去重 (agents_2025 可能与 past_agents 有重叠)
    seen_names = {c.__name__ for c in competitors}
    for agent_cls in agents_2025:
        if agent_cls.__name__ not in seen_names:
            competitors.append(agent_cls)
            seen_names.add(agent_cls.__name__)
    competitor_params = [dict() for _ in competitors]

    n_per_world = args.n_competitors_per_world
    if n_per_world <= 0:
        n_per_world = len(competitors)
    if n_per_world > len(competitors):
        n_per_world = len(competitors)

    if not args.quiet:
        names = [get_full_type_name(c) for c in competitors]
        print(f"[INFO] competitors={len(competitors)}")
        print(f"[INFO] n_per_world={n_per_world} round_robin={args.round_robin}")
        print(f"[INFO] configs={args.configs} runs={args.runs} steps={args.steps}")
        print(f"[INFO] max_worlds_per_config={args.max_worlds_per_config}")
        print(f"[INFO] parallelism={parallelism}")
        print(f"[INFO] forced_logs_fraction={args.forced_logs_fraction}")
        if not args.compact:
            print(f"[INFO] log_negotiations={args.log_negotiations} log_ufuns={args.log_ufuns}")
        if args.log_negotiations:
            print("[WARN] log_negotiations enabled (large output)")
        if args.log_ufuns:
            print("[WARN] log_ufuns enabled (large output)")

    oneshot_year = _resolve_oneshot_year(2025)
    if oneshot_year != 2025:
        print(f"[WARN] SCML{oneshot_year}OneShotWorld 可用，已回退 year={oneshot_year}")
    config_generator = partial(anac_config_generator_oneshot, year=oneshot_year)
    world_generator = partial(anac_oneshot_world_generator, year=oneshot_year)

    tournament_kwargs = dict(
        storage_cost=0,
        storage_cost_dev=0,
        perishable=True,
        oneshot_world=True,
        std_world=False,
        n_processes=2,
    )
    if args.steps is not None and args.steps > 0:
        tournament_kwargs["n_steps"] = args.steps
    if args.log_negotiations:
        tournament_kwargs["log_negotiations"] = True
    if args.log_ufuns:
        tournament_kwargs["log_ufuns"] = True

    results = tournament(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=DefaultAgentsOneShot,
        non_competitor_params=[dict() for _ in DefaultAgentsOneShot],
        config_generator=config_generator,
        config_assigner=anac_assigner_oneshot,
        world_generator=world_generator,
        score_calculator=balance_calculator_oneshot,
        n_competitors_per_world=n_per_world,
        n_configs=args.configs,
        n_runs_per_world=args.runs,
        max_worlds_per_config=args.max_worlds_per_config,
        tournament_path=str(base_dir),
        name=run_name,
        parallelism=parallelism,
        round_robin=args.round_robin,
        verbose=not args.quiet,
        compact=args.compact,
        forced_logs_fraction=args.forced_logs_fraction,
        agent_names_reveal_type=True,
        ignore_agent_exceptions=True,
        ignore_contract_execution_exceptions=True,
        ignore_simulation_exceptions=True,
        ignore_negotiation_exceptions=True,
        **tournament_kwargs,
    )

    if TRACKER_AVAILABLE:
        save_tracker_data(str(tracker_dir))

    # 查找 stage 目录（tournament 函数生成的实际结果目录）
    stage_dir = _find_stage_dir(base_dir, run_name)
    
    # 将 stage 目录中的结果文件复制到 base_dir
    if stage_dir and stage_dir.exists() and stage_dir != base_dir:
        print(f"[INFO] copying results from stage_dir: {stage_dir}")
        csv_files = [
            "agent_stats.csv", "params.json", "scores.csv", "score_stats.csv",
            "total_scores.csv", "type_stats.csv", "winners.csv", "world_stats.csv"
        ]
        for filename in csv_files:
            src = stage_dir / filename
            if src.exists():
                shutil.copy2(str(src), str(base_dir / filename))
        print(f"[INFO] results copied to: {base_dir}")
    elif not stage_dir:
        print(f"[WARN] stage_dir not found, checking base_dir for results")
    
    # 创建 tournament_info.json (传入 competitors 列表)
    _create_tournament_info(base_dir, run_name, tracker_dir, competitors)

    # copy world logs for DatasetBuilder (non-destructive)
    if stage_dir and stage_dir.exists():
        try:
            from scripts.copy_world_logs import copy_world_logs
            dataset_root = Path(__file__).resolve().parent.parent / "dataset_logs"
            result = copy_world_logs(stage_dir, dataset_root / run_name)
            print(
                f"[INFO] dataset logs copied: worlds={result['world_dirs']} "
                f"files_copied={result['files_copied']} files_skipped={result['files_skipped']}"
            )
            print(f"[INFO] dataset_logs_dir={result['output_dir']}")
        except Exception as e:
            print(f"[WARN] failed to copy dataset logs: {e}")
    
    # 打印最终结果摘要
    score_stats_file = base_dir / "score_stats.csv"
    total_scores_file = base_dir / "total_scores.csv"
    if score_stats_file.exists():
        print("\n" + "=" * 60)
        print("TOURNAMENT RESULTS (MEAN)")
        print("=" * 60)
        try:
            import csv
            with open(score_stats_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            rows.sort(key=lambda r: float(r.get("mean", 0) or 0), reverse=True)
            for i, row in enumerate(rows):
                agent_type = row.get("agent_type", "unknown")
                score = float(row.get("mean", 0) or 0)
                short_name = agent_type.split(".")[-1] if "." in agent_type else agent_type
                rank_marker = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"{i+1}."))
                print(f"  {rank_marker} {short_name}: {score:.4f}")
        except Exception as e:
            print(f"[WARN] failed to read results: {e}")
        print("=" * 60 + "\n")
    elif total_scores_file.exists():
        print("\n" + "=" * 60)
        print("TOURNAMENT RESULTS")
        print("=" * 60)
        try:
            import csv
            with open(total_scores_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    agent_type = row.get("agent_type", "unknown")
                    score = float(row.get("score", 0))
                    # 简化类名显示
                    short_name = agent_type.split(".")[-1] if "." in agent_type else agent_type
                    rank_marker = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"{i+1}."))
                    print(f"  {rank_marker} {short_name}: {score:.4f}")
        except Exception as e:
            print(f"[WARN] failed to read results: {e}")
        print("=" * 60 + "\n")

    print(f"[INFO] completed: {base_dir}")
    if not args.foreground:
        log_handle.flush()
        log_handle.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"[INFO] background job launched, log: {log_file}")


if __name__ == "__main__":
    main()
