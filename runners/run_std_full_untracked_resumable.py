#!/usr/bin/env python
"""
可断点续跑的 SCML 2025 Standard 赛道 runner（不带 tracker 的 Lita + Penguin + 全部 2025 Top Agents）。
Resumable std runner without tracker (Lita untracked + Penguin + all 2025 Top Agents).

特点 / Features:
- 使用 anac2024_std 官方流水线，显式传入 world_generator/score_calculator，保持官方行为。
- 强制保存谈判日志（forced_logs_fraction=1.0），loky 并行，支持中断后继续。
- 无 tracker 注入：适合仅需要外部交互日志的场景。

用法 / Usage:
    python runners/run_std_full_untracked_resumable.py --output-dir <目录> [--configs 20 --runs 2 --max-top 8 --parallelism parallel]
    # 中断后用同样的命令重跑，已完成的 world（有 results.json）会跳过；若生成了 *-stage-0001 会自动识别。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
from scml_agents.scml2024.standard.team_penguin.penguinagent import PenguinAgent
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP

try:
    from scml_agents import get_agents

    TOP_AGENTS_2025 = get_agents(2025, as_class=True, track="std")
except Exception as exc:  # pragma: no cover - 可选依赖
    print(f"[警告] 无法加载 2025 Top Agents: {exc}")
    TOP_AGENTS_2025: list[Type] = []


DEFAULT_CONFIGS = 20
DEFAULT_RUNS = 2
FORCED_LOGS = 1.0


def build_competitors(max_top: int | None) -> List[Type]:
    """构建参赛池（未追踪 Lita + Penguin + Top Agents + Random） / Build competitor pool."""
    lita_bases = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    tops = TOP_AGENTS_2025 if max_top is None else TOP_AGENTS_2025[:max_top]
    competitors = lita_bases + [PenguinAgent] + list(tops) + [RandomStdAgent]
    seen = set()
    uniq = []
    for c in competitors:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def has_existing_tournament(tournament_dir: Path) -> bool:
    """判断是否已有配置（决定新建/恢复） / Detect existing configs."""
    return any(
        (tournament_dir / fname).exists()
        for fname in (
            ASSIGNED_CONFIGS_PICKLE_FILE,
            ASSIGNED_CONFIGS_JSON_FILE,
            "assigned_configs",
        )
    )


def load_assignments(tournament_dir: Path):
    """加载已分配的 world 配置（用于进度统计） / Load assigned configs for progress."""
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
    """返回 (已完成 world 数, 总 world 数) / Return (done, total)."""
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


def prepare_tournament(
    tournament_dir: Path,
    competitors: List[Type],
    n_configs: int,
    n_runs_per_world: int,
    forced_logs_fraction: float,
    parallelism: str,
) -> Tuple[bool, Path]:
    """
    若不存在则创建赛程配置；存在则用于恢复。
    Create tournament configs if missing; otherwise resume.
    """

    def _find_existing_root(base: Path) -> Path | None:
        if has_existing_tournament(base):
            return base
        stage_candidate = base.parent / f"{base.name}-stage-0001"
        if has_existing_tournament(stage_candidate):
            return stage_candidate
        for p in base.parent.glob(f"{base.name}-stage-*"):
            if has_existing_tournament(p):
                return p
        return None

    existing_root = _find_existing_root(tournament_dir)
    if existing_root:
        print(f"[恢复] 已发现配置，使用 {existing_root}")
        return False, existing_root
    if tournament_dir.exists():
        raise RuntimeError(
            f"{tournament_dir} 已存在但缺少配置，请更换 --output-dir，或确认安全后手动清理。"
        )

    base_dir = tournament_dir.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[创建] 配置生成目录 {tournament_dir} "
        f"(n_configs={n_configs}, n_runs_per_world={n_runs_per_world})"
    )
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
        verbose=True,
        print_exceptions=True,
    )
    root = tournament_dir
    try:
        if configs_path is not None:
            configs_path = Path(configs_path)
            root = configs_path.parent
    except Exception:
        pass
    return True, root


def save_results(
    output_dir: Path,
    results,
    competitors: List[Type],
    created: bool,
    n_configs: int,
    n_runs: int,
    parallelism: str,
) -> None:
    """保存简要结果摘要 / Persist a small summary."""
    data = {
        "runner": "run_std_full_untracked_resumable",
        "created_now": created,
        "tournament_path": str(output_dir),
        "n_configs": n_configs,
        "n_runs_per_world": n_runs,
        "parallelism": parallelism,
        "competitors": [c.__name__ for c in competitors],
        "winners": [w.split(".")[-1] for w in getattr(results, "winners", [])],
        "timestamp": datetime.now().isoformat(),
    }
    (output_dir / "tournament_results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_rankings(results) -> None:
    if getattr(results, "total_scores", None) is None:
        return
    print("\n[成绩] 排名 / Ranking:")
    sorted_scores = results.total_scores.sort_values("score", ascending=False)
    for rank, (_, row) in enumerate(sorted_scores.iterrows(), 1):
        agent_name = row["agent_type"].split(".")[-1]
        print(f"  {rank}. {agent_name}: {row['score']:.4f}")


def run_resumable(
    n_configs: int,
    n_runs: int,
    max_top: int,
    output_dir: Path | None,
    parallelism: str,
    skip_postprocess: bool,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_dir = output_dir or Path(f"tournament_history/std_resumable_untracked_{ts}")

    competitors = build_competitors(max_top=max_top)
    created, tournament_root = prepare_tournament(
        tournament_dir=tournament_dir,
        competitors=competitors,
        n_configs=n_configs,
        n_runs_per_world=n_runs,
        forced_logs_fraction=FORCED_LOGS,
        parallelism=parallelism,
    )

    done, total = summarize_progress(tournament_root)
    if total:
        print(f"[进度] 已完成 {done}/{total} 个 world ({done/total:.1%})")

    print(f"[运行] 启动/恢复比赛 {tournament_root} (parallelism={parallelism})")
    run_tournament(
        tournament_path=str(tournament_root),
        world_generator=anac2024_std_world_generator,
        score_calculator=balance_calculator_std,
        parallelism=parallelism,
        verbose=True,
        compact=False,
        print_exceptions=True,
    )

    print("[评估] 汇总结果")
    results = evaluate_tournament(
        tournament_path=str(tournament_root),
        metric=truncated_mean,
        verbose=True,
        recursive=True,
    )
    print_rankings(results)
    save_results(
        output_dir=tournament_root,
        results=results,
        competitors=competitors,
        created=created,
        n_configs=n_configs,
        n_runs=n_runs,
        parallelism=parallelism,
    )

    if not skip_postprocess:
        from scml_analyzer.postprocess import postprocess_tournament

        print("[后处理] 汇总日志（negotiations 等）")
        postprocess_tournament(
            output_dir=tournament_root,
            start_visualizer=False,
            visualizer_port=None,
        )
    else:
        print("[后处理] 已跳过 (--no-postprocess)")

    return tournament_root


def main():
    parser = argparse.ArgumentParser(
        description="Resumable SCML 2025 Standard (untracked Lita + Penguin + all Top Agents)"
    )
    parser.add_argument("--configs", type=int, default=DEFAULT_CONFIGS, help="world 配置数量 / n_configs (default: 20)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="每个 world 运行次数 / n_runs_per_world (default: 2)")
    parser.add_argument("--max-top", type=int, default=8, help="Top Agents 截断数量 / max top agents")
    parser.add_argument(
        "--parallelism",
        type=str,
        default="parallel",
        help="传给 negmas 的并行模式（默认 loky 补丁） / parallelism",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="比赛输出目录（复用即可恢复） / output dir to resume")
    parser.add_argument(
        "--no-postprocess", action="store_true", help="跳过 postprocess（仅跑 world 与得分） / skip postprocess"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    run_resumable(
        n_configs=args.configs,
        n_runs=args.runs,
        max_top=args.max_top,
        output_dir=output_dir,
        parallelism=args.parallelism,
        skip_postprocess=args.no_postprocess,
    )


if __name__ == "__main__":
    main()
