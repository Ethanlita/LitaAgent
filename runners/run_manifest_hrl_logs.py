#!/usr/bin/env python
"""
基于 manifest 的 HRL 日志采集 runner（可断点续跑，控制组合规模，保持多样性）。

特性：
- 预生成 manifest（json），每个条目定义一个 world（seed、对手子集、步数）。
- 逐 world 运行，状态标记为 done 后跳过，支持断点续跑；manifest 不够时可追加生成。
- 强制保存谈判日志（log_negotiations/log_ufuns），避免全排列爆量，固定对手子集大小。
- 自适应并行度默认使用全部 CPU，可通过参数覆盖。

用法：
  1) 默认生成并运行（如本地测试 10 个 world）：
     python runners/run_manifest_hrl_logs.py --generate 10 --steps 100 --min-comp 5 --max-comp 10
  2) 仅生成 manifest 不运行：
     python runners/run_manifest_hrl_logs.py --generate 200 --no-run
  3) 继续运行已有 manifest（跳过 status=done 的 world）：
     python runners/run_manifest_hrl_logs.py --manifest results/manifest_hrl.json
     # 如需追加更多 world 后继续：
     python runners/run_manifest_hrl_logs.py --manifest results/manifest_hrl.json --generate 100
  4) 控制并行度：
     python runners/run_manifest_hrl_logs.py --parallelism 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Type

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runners.loky_patch import enable_loky_executor

enable_loky_executor()

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from scml.std import SCML2024StdWorld
from scml.std.agents import RandomStdAgent
from scml.utils import anac2024_std
from litaagent_std.tracker_mixin import create_tracked_agent
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_p import LitaAgentP
from scml_agents.scml2024.standard.team_penguin.penguinagent import PenguinAgent

try:
    from scml_agents import get_agents

    TOP_AGENTS_2025 = get_agents(2025, as_class=True, top_only=8, track="std")
except Exception:
    TOP_AGENTS_2025 = []


def competitor_pool(max_top: int | None = None) -> List[Type]:
    """返回候选代理池（类），含 tracked Lita、Penguin、Top Agents、RandomStdAgent。"""
    log_dir = os.environ.get("SCML_TRACKER_LOG_DIR", ".")
    lita_bases = [LitaAgentY, LitaAgentYR, LitaAgentCIR, LitaAgentN, LitaAgentP]
    tracked_lita = [create_tracked_agent(cls, log_dir=log_dir) for cls in lita_bases]
    
    # PenguinAgent 也需要 Tracker 包装，确保收集完整数据
    tracked_penguin = create_tracked_agent(PenguinAgent, log_dir=log_dir)
    
    # Top Agents 也需要 Tracker 包装
    tops = TOP_AGENTS_2025 if max_top is None else TOP_AGENTS_2025[: max_top]
    tracked_tops = []
    for cls in tops:
        try:
            tracked_tops.append(create_tracked_agent(cls, log_dir=log_dir))
        except Exception:
            tracked_tops.append(cls)  # 包装失败则使用原始类
    
    pool = tracked_lita + [tracked_penguin] + tracked_tops + [RandomStdAgent]
    # 去重保持顺序
    seen = set()
    uniq = []
    for c in pool:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def gen_manifest(
    path: Path,
    n_worlds: int,
    steps: int,
    min_comp: int,
    max_comp: int | None,
    max_top: int,
    seed_base: int,
) -> None:
    """生成 manifest（若已存在则追加未完成部分）。"""
    pool = competitor_pool(max_top=max_top)
    # 使用 Tracked 版本的 PenguinAgent（在 pool 中找到它）
    log_dir = os.environ.get("SCML_TRACKER_LOG_DIR", ".")
    tracked_penguin = create_tracked_agent(PenguinAgent, log_dir=log_dir)
    # 找到 pool 中的 tracked penguin（通过基类名匹配）
    penguin_cls = None
    for c in pool:
        if 'Penguin' in c.__name__ or (hasattr(c, '__bases__') and any('Penguin' in b.__name__ for b in c.__bases__)):
            penguin_cls = c
            break
    if penguin_cls is None:
        penguin_cls = tracked_penguin  # 回退
    pool_no_penguin = [c for c in pool if c is not penguin_cls]
    entries = []
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
    start_id = len(entries)
    rng = random.Random(seed_base + start_id)
    pool_size = len(pool)
    eff_max = pool_size if max_comp is None else min(max_comp, pool_size)
    if min_comp < 1:
        raise ValueError("min_comp 必须 >=1（至少包含 PenguinAgent）")
    if min_comp > eff_max:
        raise ValueError(f"min_comp={min_comp} 大于候选池大小 {eff_max}，请调小 min_comp 或增大池子。")
    if eff_max - 1 > len(pool_no_penguin):
        # 不影响抽样，但避免不必要警告
        eff_max = len(pool_no_penguin) + 1
    for i in range(n_worlds):
        k = rng.randint(min_comp, eff_max)
        if k - 1 > len(pool_no_penguin):
            raise ValueError("对手池不足以满足 min/max_comp 要求（除 Penguin 外的池子太小）")
        subset = [penguin_cls] + rng.sample(pool_no_penguin, k - 1)
        rng.shuffle(subset)
        seed = seed_base + start_id + i
        entries.append(
            {
                "id": start_id + i,
                "seed": seed,
                "steps": steps,
                "competitors": [f"{c.__module__}.{c.__name__}" for c in subset],
                "status": "todo",
                "output_dir": None,
            }
        )
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Manifest] 生成/更新 {path}, 总 world={len(entries)} (新增 {n_worlds})", flush=True)


def load_class(qualname: str):
    mod, cls = qualname.rsplit(".", 1)
    module = __import__(mod, fromlist=[cls])
    return getattr(module, cls)


def run_world(entry: dict, base_dir: Path, parallelism: int | None):
    seed = entry["seed"]
    steps = entry["steps"]
    comp_classes = [load_class(q) for q in entry["competitors"]]

    random.seed(seed)
    np.random.seed(seed)

    world_dir = base_dir / f"world_{entry['id']:05d}"
    os.environ["SCML_TRACKER_LOG_DIR"] = str(world_dir / "tracker_logs")
    world_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[World {entry['id']}] seed={seed}, comps={len(comp_classes)}, steps={steps}, dir={world_dir}",
        flush=True,
    )

    # 使用官方 anac2024_std 管线生成 world（包含多工厂分配），强制保存日志
    # 将每个 manifest 条目限制为 1 个 world，避免全排列爆量
    anac2024_std(
        competitors=comp_classes,
        n_configs=1,
        max_worlds_per_config=len(comp_classes),
        n_runs_per_world=1,
        forced_logs_fraction=1.0,
        tournament_path=str(world_dir),
        parallelism="serial",
        n_steps=steps,
        verbose=False,
    )

    entry["status"] = "done"
    entry["output_dir"] = str(world_dir)
    print(f"[World {entry['id']}] 完成，日志目录={world_dir / 'logs'}", flush=True)
    return world


def main():
    parser = argparse.ArgumentParser(description="基于 manifest 的 HRL 日志采集（可断点续跑、可控规模）")
    parser.add_argument("--manifest", type=str, default="tournament_history/manifest_hrl.json", help="manifest 路径")
    parser.add_argument("--generate", type=int, default=0, help="生成的 world 数量（追加）")
    parser.add_argument("--steps", type=int, default=100, help="每个 world 步数（贴合 2025 Standard）")
    parser.add_argument("--min-comp", type=int, default=5, help="每个 world 最少对手数")
    parser.add_argument("--max-comp", type=int, default=None, help="每个 world 最多对手数（None 表示不限制）")
    parser.add_argument("--max-top", type=int, default=8, help="Top Agents 截断数")
    parser.add_argument("--seed-base", type=int, default=12345, help="生成 manifest 的随机种子基值")
    parser.add_argument("--parallelism", type=str, default="auto", help="并行度：数字或 auto")
    parser.add_argument("--output-base", type=str, default=None, help="输出基目录，默认 tournament_history/hrl_manifest_<ts>")
    parser.add_argument("--no-run", action="store_true", help="仅生成 manifest，不运行")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_base or f"tournament_history/hrl_manifest_{ts}")
    base_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if args.generate > 0 or not manifest_path.exists():
        gen_manifest(
            path=manifest_path,
            n_worlds=args.generate if args.generate > 0 else 0,
            steps=args.steps,
            min_comp=args.min_comp,
            max_comp=args.max_comp,
            max_top=args.max_top,
            seed_base=args.seed_base,
        )

    if args.no_run:
        print("[Info] 仅生成 manifest，未运行。")
        return

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    todo = [e for e in entries if e.get("status") != "done"]
    print(f"[Info] 待运行 world 数: {len(todo)} / {len(entries)}")

    try:
        parallel_setting = int(args.parallelism)
    except (TypeError, ValueError):
        parallel_setting = None

    for entry in todo:
        run_world(entry, base_dir=base_dir, parallelism=parallel_setting)
        # 每跑完一个 world 即刻持久化状态，便于断点续跑
        manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] manifest 中的待运行 world 已处理完毕。")


if __name__ == "__main__":
    main()
