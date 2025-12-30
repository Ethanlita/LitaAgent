# -*- coding: utf-8 -*-
"""全量训练脚本：数据处理 -> 统计 -> L2/L3 BC + L3 AWR(可选) + L4 蒸馏。"""
from __future__ import annotations

import argparse
import gc
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from litaagent_std.hrl_xf import data_pipeline
from litaagent_std.hrl_xf import training as training_mod
from litaagent_std.hrl_xf.training import TrainConfig, HRLXFTrainer, save_model
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import L3DecisionTransformer
from litaagent_std.hrl_xf.l4_coordinator import GlobalCoordinator


PHASE_CHOICES = ("l2", "l3", "l4")


def _default_num_workers() -> int:
    # Windows 下多进程容易触发 <stdin> 等问题，默认串行更稳。
    if os.name == "nt":
        return 1
    cpu_cnt = os.cpu_count() or 1
    return max(1, cpu_cnt - 1)


def _parse_phases(value: str) -> Tuple[str, ...]:
    raw = [t.strip().lower() for t in (value or "").split(",") if t.strip()]
    if not raw:
        return tuple()
    invalid = sorted({t for t in raw if t not in PHASE_CHOICES})
    if invalid:
        raise ValueError(f"Invalid phases: {invalid}. Valid: {PHASE_CHOICES}")
    return tuple(dict.fromkeys(raw))


def _find_latest_tournament_dir(base_dir: Path) -> Path:
    # 未指定目录时，自动选择最近的带 Tracker JSON 的比赛目录。
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")
    candidates = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "tracker_logs").exists():
            candidates.append(child)
            continue
        if any(child.glob("agent_*.json")):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No tournament dir with tracker logs under: {base_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_ckpt(output_dir: Path, pattern: str) -> Optional[Path]:
    if not output_dir.exists():
        return None
    regex = re.compile(pattern)
    best_epoch = -1
    best_path = None
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = regex.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = path
    return best_path


def _resolve_l2_model_path(output_dir: Path, explicit_path: Optional[str]) -> Optional[Path]:
    # 优先稳定权重，其次最新 epoch 权重/断点。
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        return path if path.exists() else None
    stable = output_dir / "l2_bc.pt"
    if stable.exists():
        return stable
    latest = _find_latest_ckpt(output_dir, r"l2_bc_epoch(\d+)\.pt$")
    if latest:
        return latest
    latest_ckpt = _find_latest_ckpt(output_dir, r"l2_bc_epoch(\d+)\.ckpt\.pt$")
    return latest_ckpt


def _summarize_samples(
    macro_samples,
    micro_samples,
    *,
    sample_limit: int = 0,
) -> None:
    # 轻量质量检查：规模、动作分布、time_mask、历史长度。
    print("=" * 60)
    print("[INFO] Data summary")
    print(f"[INFO] Macro samples: {len(macro_samples)}")
    print(f"[INFO] Micro samples: {len(micro_samples)}")

    if not micro_samples:
        print("=" * 60)
        return

    view = micro_samples
    if sample_limit > 0 and len(micro_samples) > sample_limit:
        view = micro_samples[:sample_limit]
        print(f"[INFO] Micro sample stats based on first {sample_limit} samples")

    action_counts = [0, 0, 0]
    missing_offer = 0
    all_inf_time_mask = 0
    history_len_sum = 0

    for s in view:
        try:
            op = int(s.action_op)
        except Exception:
            op = -1
        if 0 <= op <= 2:
            action_counts[op] += 1
        if op in (0, 1):
            if s.target_quantity is None or s.target_price is None or s.target_time is None:
                missing_offer += 1
        tm = s.time_mask
        if tm is not None and len(tm) > 0:
            tm_arr = np.asarray(tm, dtype=np.float32)
            if np.all(np.isneginf(tm_arr)):
                all_inf_time_mask += 1
        if s.history is not None:
            history_len_sum += len(s.history)

    denom = max(1, len(view))
    avg_hist = history_len_sum / denom
    print(f"[INFO] action_op counts: ACCEPT={action_counts[0]} REJECT={action_counts[1]} END={action_counts[2]}")
    print(f"[INFO] missing offer (ACCEPT/REJECT): {missing_offer}")
    print(f"[INFO] time_mask all -inf: {all_inf_time_mask}")
    print(f"[INFO] avg history length: {avg_hist:.2f}")
    print("=" * 60)


def _train_l2(
    macro_samples,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    resume_path: Optional[Path],
) -> Path:
    if not macro_samples:
        raise RuntimeError("No macro samples found for L2 training")
    config = TrainConfig(
        output_dir=str(output_dir),
        l2_epochs=args.l2_epochs,
        l2_batch_size=args.l2_batch_size,
        device=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
        l2_resume_path=str(resume_path) if resume_path else None,
    )
    model = HorizonManagerPPO(horizon=config.horizon)
    trainer = HRLXFTrainer(l2_model=model, l3_model=None, l4_model=None, config=config)
    trainer.train_phase0_bc(macro_samples, [])
    return Path(save_model(model, config.output_dir, "l2_bc.pt"))


def _train_l3(
    micro_samples,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    resume_path: Optional[Path],
    awr_resume_path: Optional[Path],
    run_awr: bool,
) -> Path:
    if not micro_samples:
        raise RuntimeError("No micro samples found for L3 training")
    config = TrainConfig(
        output_dir=str(output_dir),
        l3_epochs=args.l3_epochs,
        l3_batch_size=args.l3_batch_size,
        device=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
        l3_bc_resume_path=str(resume_path) if resume_path else None,
        l3_awr_resume_path=str(awr_resume_path) if awr_resume_path else None,
    )
    model = L3DecisionTransformer(horizon=config.horizon)
    trainer = HRLXFTrainer(l2_model=None, l3_model=model, l4_model=None, config=config)
    trainer.train_phase0_bc([], micro_samples)
    if run_awr:
        trainer.train_phase1_awr(micro_samples)
    return Path(save_model(model, config.output_dir, "l3_bc.pt"))


def _train_l4(
    l4_samples,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    resume_path: Optional[Path],
) -> Path:
    if not l4_samples:
        raise RuntimeError("No L4 distill samples found")
    config = TrainConfig(
        output_dir=str(output_dir),
        l4_epochs=args.l4_epochs,
        l4_batch_size=args.l4_batch_size,
        device=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
        l4_resume_path=str(resume_path) if resume_path else None,
    )
    model = GlobalCoordinator()
    trainer = HRLXFTrainer(l2_model=None, l3_model=None, l4_model=model, config=config)
    trainer.train_l4_distill(l4_samples)
    return Path(save_model(model, config.output_dir, "l4_distill.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="HRL BC/AWR training runner (data -> stats -> L2/L3/L4)")
    parser.add_argument("--tournament-dir", type=str, default=None, help="tournament_history/<dir>")
    parser.add_argument("--output-dir", type=str, default=None, help="checkpoint output dir")
    parser.add_argument("--agent-name", type=str, default="PenguinAgent", help="agent name filter")
    parser.add_argument("--phases", type=str, default="l2,l3,l4", help="comma list: l2,l3,l4")
    parser.add_argument("--allow-csv", action="store_true", help="allow CSV fallback (not recommended)")
    parser.add_argument("--num-workers", type=int, default=None, help="data pipeline workers")
    parser.add_argument("--stats-sample", type=int, default=0, help="limit stats to first N micro samples")
    parser.add_argument("--stats-only", action="store_true", help="only run data + stats")
    parser.add_argument("--no-resume", action="store_true", help="disable auto-resume from checkpoints")

    parser.add_argument("--l3-goal-backfill", type=str, default="l2", choices=["none", "v2", "l2"])
    parser.add_argument("--l4-goal-source", type=str, default="l2", choices=["none", "v2", "l2"])
    parser.add_argument("--l2-model-path", type=str, default=None, help="L2 model path for backfill")
    parser.add_argument("--l3-awr", action="store_true", help="run L3 AWR after BC")

    parser.add_argument("--l2-epochs", type=int, default=10)
    parser.add_argument("--l3-epochs", type=int, default=10)
    parser.add_argument("--l4-epochs", type=int, default=10)
    parser.add_argument("--l2-batch-size", type=int, default=64)
    parser.add_argument("--l3-batch-size", type=int, default=32)
    parser.add_argument("--l4-batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)

    args = parser.parse_args()
    phases = _parse_phases(args.phases)

    if not training_mod.TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    base_dir = Path("tournament_history").resolve()
    if args.tournament_dir:
        tournament_dir = Path(args.tournament_dir).expanduser().resolve()
    else:
        if not base_dir.exists():
            raise FileNotFoundError(f"tournament_history not found: {base_dir}")
        # 默认使用 tournament_history 下全部比赛日志。
        tournament_dir = base_dir
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (tournament_dir / "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers if args.num_workers is not None else _default_num_workers()
    strict_json_only = not args.allow_csv

    print(f"[INFO] tournament_dir: {tournament_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] phases: {phases}")
    print(f"[INFO] num_workers: {num_workers}")

    # 先做一次不回填的解析与统计，便于快速发现结构性异常。
    macro_samples, micro_samples = data_pipeline.load_tournament_data(
        str(tournament_dir),
        agent_name=args.agent_name,
        strict_json_only=strict_json_only,
        num_workers=num_workers,
        goal_backfill="none",
    )
    _summarize_samples(macro_samples, micro_samples, sample_limit=args.stats_sample)

    if args.stats_only or not phases:
        return

    l2_model_path = _resolve_l2_model_path(output_dir, args.l2_model_path)

    if "l2" in phases:
        resume_path = None
        if not args.no_resume:
            resume_path = _find_latest_ckpt(output_dir, r"l2_bc_epoch(\d+)\.ckpt\.pt$")
        l2_model_path = _train_l2(
            macro_samples,
            output_dir=output_dir,
            args=args,
            resume_path=resume_path,
        )

    del macro_samples
    del micro_samples
    gc.collect()

    if "l3" in phases:
        if args.l3_goal_backfill == "l2" and not l2_model_path:
            raise ValueError("l3-goal-backfill=l2 but no L2 model path found")

        resume_path = None
        awr_resume_path = None
        if not args.no_resume:
            resume_path = _find_latest_ckpt(output_dir, r"l3_bc_epoch(\d+)\.ckpt\.pt$")
            awr_resume_path = _find_latest_ckpt(output_dir, r"l3_awr_epoch(\d+)\.ckpt\.pt$")

        # L3 训练需要回填后的 micro.goal。
        _, micro_samples = data_pipeline.load_tournament_data(
            str(tournament_dir),
            agent_name=args.agent_name,
            strict_json_only=strict_json_only,
            num_workers=num_workers,
            goal_backfill=args.l3_goal_backfill,
            l2_model_path=str(l2_model_path) if l2_model_path else None,
        )
        _train_l3(
            micro_samples,
            output_dir=output_dir,
            args=args,
            resume_path=resume_path,
            awr_resume_path=awr_resume_path,
            run_awr=args.l3_awr,
        )
        del micro_samples
        gc.collect()

    if "l4" in phases:
        if args.l4_goal_source == "l2" and not l2_model_path:
            raise ValueError("l4-goal-source=l2 but no L2 model path found")

        resume_path = None
        if not args.no_resume:
            resume_path = _find_latest_ckpt(output_dir, r"l4_distill_epoch_(\d+)\.pt$")

        # L4 蒸馏基于启发式教师信号，需要单独抽取样本。
        l4_samples = data_pipeline.load_l4_distill_data(
            str(tournament_dir),
            agent_name=args.agent_name,
            strict_json_only=strict_json_only,
            num_workers=num_workers,
            goal_source=args.l4_goal_source,
            l2_model_path=str(l2_model_path) if l2_model_path else None,
        )
        _train_l4(
            l4_samples,
            output_dir=output_dir,
            args=args,
            resume_path=resume_path,
        )

    print(f"[INFO] Training finished. Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
