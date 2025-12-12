#!/usr/bin/env python
"""
分批采集 HRL 训练日志的 runner（自适应本地/服务器资源，支持断点续跑）。

特点：
- 默认开启 log_negotiations / log_ufuns，生成训练所需日志。
- 分批运行，每批独立 save_path，批次目录存在则跳过（断点续跑）。
- 自适应并行度与规模：根据 CPU 核心数给出默认 configs/runs/steps/parallelism，可 CLI 覆盖。
- 参赛名单：LitaAgents（tracked，非 HRL）、Penguin、2025 Top Agents（最多 8）、RandomAgent。
"""

from __future__ import annotations

import argparse
import os
import sys
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from runners.loky_patch import enable_loky_executor

enable_loky_executor()

from scml.utils import anac2024_std
from scml.std.agents import RandomStdAgent
from litaagent_std.tracker_mixin import create_tracked_agent
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP
from scml_agents.scml2024.standard.team_penguin.penguinagent import PenguinAgent

try:  # optional tracker
    from scml_analyzer.auto_tracker import TrackerConfig, TrackerManager

    _TRACKER_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRACKER_AVAILABLE = False
    TrackerConfig = None
    TrackerManager = None

try:
    from scml_agents import get_agents

    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=8, track="std")
except Exception:
    TOP_AGENTS_2025 = []


def auto_defaults() -> dict:
    cpu = mp.cpu_count()
    if cpu >= 128:
        return dict(configs=10, runs=2, steps_min=90, steps_max=100, parallelism=min(96, cpu - 4), batches=20)
    if cpu >= 32:
        return dict(configs=6, runs=2, steps_min=80, steps_max=90, parallelism=min(24, cpu - 4), batches=10)
    return dict(configs=4, runs=1, steps_min=70, steps_max=80, parallelism=min(8, cpu - 2), batches=5)


def build_competitors(max_top: int | None = None):
    log_dir = os.environ.get("SCML_TRACKER_LOG_DIR", ".")
    lita_bases = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    tracked_lita = [create_tracked_agent(cls, log_dir=log_dir) for cls in lita_bases]
    penguin = [PenguinAgent]
    tops = TOP_AGENTS_2025 if max_top is None else TOP_AGENTS_2025[: max_top]
    competitors = tracked_lita + penguin + list(tops) + [RandomStdAgent]
    # 去重保持顺序
    seen = set()
    uniq = []
    for c in competitors:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq, [c.__name__ for c in lita_bases]


def run_one_batch(batch_dir: Path, configs: int, runs: int, steps_min: int, steps_max: int, parallelism: int, max_top: int | None):
    batch_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir = batch_dir / "tracker_logs"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)

    if _TRACKER_AVAILABLE:
        TrackerManager._loggers.clear()
        TrackerConfig.configure(enabled=True, log_dir=str(tracker_dir), console_echo=False)

    competitors, lita_names = build_competitors(max_top=max_top)
    print(f"[Batch {batch_dir.name}] competitors: {[c.__name__ for c in competitors]}")

    results = anac2024_std(
        name=f"Batch_{batch_dir.name}",
        competitors=competitors,
        n_runs_per_world=runs,
        n_configs=configs,
        # n_steps 由世界生成器控制，这里通过 kwargs 传递
        n_steps=(steps_min, steps_max),
        tournament_path=str(batch_dir),
        parallelism="parallel",
        verbose=True,
        compact=False,
        forced_logs_fraction=1.0,  # 强制保存谈判日志
        n_competitors_per_world=None,  # 默认随机
    )

    # 简单保存结果
    summary = {
        "competitors": [c.__name__ for c in competitors],
        "winners": [w.split(".")[-1] for w in getattr(results, "winners", [])],
    }
    (batch_dir / "tournament_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if getattr(results, "total_scores", None) is not None:
        print(f"[Batch {batch_dir.name}] top scores:")
        sorted_scores = results.total_scores.sort_values("score", ascending=False)
        for rank, (_, row) in enumerate(sorted_scores.iterrows(), 1):
            agent_name = row["agent_type"].split(".")[-1]
            tag = "★" if agent_name in lita_names else ""
            print(f"  {rank}. {agent_name}: {row['score']:.4f} {tag}")


def main():
    defaults = auto_defaults()
    parser = argparse.ArgumentParser(description="分批采集 HRL 日志的 runner（自适应并行，断点续跑）")
    parser.add_argument("--batches", type=int, default=defaults["batches"], help="批次数量（默认自适应）")
    parser.add_argument("--configs", type=int, default=defaults["configs"], help="每批 n_configs")
    parser.add_argument("--runs", type=int, default=defaults["runs"], help="每批 n_runs_per_world")
    parser.add_argument("--steps-min", type=int, default=defaults["steps_min"], help="最小步数")
    parser.add_argument("--steps-max", type=int, default=defaults["steps_max"], help="最大步数")
    parser.add_argument("--parallelism", type=int, default=defaults["parallelism"], help="并行度（n_jobs）")
    parser.add_argument("--max-top", type=int, default=8, help="加载的 Top Agents 上限")
    parser.add_argument("--output-base", type=str, default=None, help="输出基目录，未提供则自动生成")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_base or f"results/hrl_batches_{ts}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[Info] 分批采集 HRL 日志（Penguin + Top Agents + Random + Lita tracker）")
    print(f"[Info] 资源检测：CPU={mp.cpu_count()}，默认并行度={args.parallelism}")
    print(f"[Info] 批次数量={args.batches}, 每批 configs={args.configs}, runs={args.runs}, steps=({args.steps_min},{args.steps_max})")
    print(f"[Info] 输出目录: {base_dir}")
    print("=" * 70)

    for idx in range(args.batches):
        batch_dir = base_dir / f"batch_{idx:03d}"
        if batch_dir.exists() and any(batch_dir.iterdir()):
            print(f"[Skip] {batch_dir} 已存在，视为完成，跳过。")
            continue
        print(f"[Run] 开始批次 {idx} -> {batch_dir}")
        run_one_batch(
            batch_dir=batch_dir,
            configs=args.configs,
            runs=args.runs,
            steps_min=args.steps_min,
            steps_max=args.steps_max,
            parallelism=args.parallelism,
            max_top=args.max_top,
        )
    print("[Done] 所有批次处理完毕（或已跳过已存在批次）。")


if __name__ == "__main__":
    main()
