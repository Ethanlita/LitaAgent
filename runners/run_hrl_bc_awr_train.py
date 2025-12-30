# -*- coding: utf-8 -*-
"""全量训练脚本：数据处理 -> 统计 -> L2/L3 BC + L3 AWR(可选) + L4 蒸馏。"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import itertools
import json
import multiprocessing as mp
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from litaagent_std.hrl_xf import data_pipeline
from litaagent_std.hrl_xf import training as training_mod
from litaagent_std.hrl_xf.training import (
    TrainConfig,
    HRLXFTrainer,
    L2Dataset,
    L3Dataset,
    L4Dataset,
    collate_l4,
    save_model,
)
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


def _load_world_dirs(tournament_dir: Path) -> List[str]:
    return data_pipeline._find_world_dirs(str(tournament_dir))


def _split_world_dirs(
    world_dirs: List[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_ratio + test_ratio 必须在 [0, 1) 内")
    total = len(world_dirs)
    if total == 0:
        return [], [], []

    rng = random.Random(seed)
    shuffled = list(world_dirs)
    rng.shuffle(shuffled)

    val_n = int(total * val_ratio)
    test_n = int(total * test_ratio)
    if val_ratio > 0 and val_n == 0 and total >= 2:
        val_n = 1
    if test_ratio > 0 and test_n == 0 and total >= 3:
        test_n = 1

    if val_n + test_n >= total:
        val_n = min(val_n, max(0, total - 1))
        test_n = min(test_n, max(0, total - 1 - val_n))

    val_dirs = shuffled[:val_n]
    test_dirs = shuffled[val_n: val_n + test_n]
    train_dirs = shuffled[val_n + test_n :]
    return train_dirs, val_dirs, test_dirs


def _load_split_cache(
    cache_path: Path,
    tournament_dir: Path,
) -> Optional[Tuple[List[str], List[str], List[str]]]:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if payload.get("tournament_dir") != str(tournament_dir):
        return None
    splits = payload.get("splits", {})
    train_dirs = splits.get("train", [])
    val_dirs = splits.get("val", [])
    test_dirs = splits.get("test", [])
    return train_dirs, val_dirs, test_dirs


def _save_split_cache(
    cache_path: Path,
    tournament_dir: Path,
    train_dirs: List[str],
    val_dirs: List[str],
    test_dirs: List[str],
) -> None:
    payload = {
        "tournament_dir": str(tournament_dir),
        "splits": {
            "train": train_dirs,
            "val": val_dirs,
            "test": test_dirs,
        },
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_samples_from_world_dirs(
    world_dirs: List[str],
    *,
    agent_name: str,
    strict_json_only: bool,
    goal_backfill: str,
    l2_model_path: Optional[str],
    num_workers: int,
) -> Tuple[List[data_pipeline.MacroSample], List[data_pipeline.MicroSample]]:
    macro_samples: List[data_pipeline.MacroSample] = []
    micro_samples: List[data_pipeline.MicroSample] = []
    if not world_dirs:
        return macro_samples, micro_samples

    def _merge(result: Dict[str, Any]) -> None:
        macro_samples.extend(result.get("macro_samples", []))
        micro_samples.extend(result.get("micro_samples", []))

    if num_workers <= 1:
        for world_dir in world_dirs:
            _merge(
                data_pipeline._process_world_dir(
                    world_dir,
                    agent_name,
                    strict_json_only,
                    goal_backfill=goal_backfill,
                    l2_model_path=l2_model_path,
                )
            )
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            results_iter = executor.map(
                data_pipeline._process_world_dir,
                world_dirs,
                itertools.repeat(agent_name),
                itertools.repeat(strict_json_only),
                itertools.repeat(goal_backfill),
                itertools.repeat(l2_model_path),
                chunksize=1,
            )
            for result in results_iter:
                _merge(result)

    return macro_samples, micro_samples


def _load_l4_samples_from_world_dirs(
    world_dirs: List[str],
    *,
    agent_name: str,
    strict_json_only: bool,
    goal_source: str,
    l2_model_path: Optional[str],
    num_workers: int,
) -> List[data_pipeline.L4DistillSample]:
    l4_samples: List[data_pipeline.L4DistillSample] = []
    if not world_dirs:
        return l4_samples

    def _merge(result: Dict[str, Any]) -> None:
        l4_samples.extend(result.get("l4_samples", []))

    if num_workers <= 1:
        for world_dir in world_dirs:
            _merge(
                data_pipeline._process_world_dir_l4(
                    world_dir,
                    agent_name,
                    strict_json_only,
                    goal_source=goal_source,
                    l2_model_path=l2_model_path,
                )
            )
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            results_iter = executor.map(
                data_pipeline._process_world_dir_l4,
                world_dirs,
                itertools.repeat(agent_name),
                itertools.repeat(strict_json_only),
                itertools.repeat(goal_source),
                itertools.repeat(l2_model_path),
                chunksize=1,
            )
            for result in results_iter:
                _merge(result)

    return l4_samples


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
    title: str = "all",
) -> None:
    # 轻量质量检查：规模、动作分布、time_mask、历史长度。
    print("=" * 60)
    print(f"[INFO] Data summary ({title})")
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


def _eval_l2(model, samples, config: TrainConfig, batch_size: int) -> Dict[str, float]:
    import torch

    dataset = L2Dataset(samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(config.device)
    model.eval()

    total_samples = 0
    sum_loss = 0.0
    sum_mse = 0.0
    sum_mask = 0.0

    with torch.no_grad():
        for batch in loader:
            state_static = batch["state_static"].to(config.device)
            state_temporal = batch["state_temporal"].to(config.device)
            x_role = batch["x_role"].to(config.device)
            goal_target = batch["goal"].to(config.device)

            goal_pred = model(state_static, state_temporal, x_role)
            if isinstance(goal_pred, tuple):
                goal_pred = goal_pred[0]

            B = x_role.size(0)
            loss_mask = torch.ones(B, 16, device=config.device)
            can_buy = x_role[:, 0:1]
            can_sell = x_role[:, 1:2]
            buy_indices = [0, 1, 4, 5, 8, 9, 12, 13]
            sell_indices = [2, 3, 6, 7, 10, 11, 14, 15]
            for idx in buy_indices:
                loss_mask[:, idx] = can_buy.squeeze(-1)
            for idx in sell_indices:
                loss_mask[:, idx] = can_sell.squeeze(-1)

            squared_error = (goal_pred - goal_target) ** 2
            masked_error = squared_error * loss_mask
            n_valid = loss_mask.sum(dim=1).clamp(min=1.0)
            per_sample_loss = masked_error.sum(dim=1) / n_valid

            sum_loss += float(per_sample_loss.sum().item())
            sum_mse += float(masked_error.sum().item())
            sum_mask += float(loss_mask.sum().item())
            total_samples += int(B)

    avg_loss = sum_loss / max(1, total_samples)
    avg_mse = sum_mse / max(1.0, sum_mask)
    return {"loss": avg_loss, "mse": avg_mse, "samples": float(total_samples)}


def _eval_l3(model, samples, config: TrainConfig, batch_size: int) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F

    dataset = L3Dataset(samples, horizon=config.horizon)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(config.device)
    model.eval()

    total_samples = 0
    sum_op = 0.0
    sum_q = 0.0
    sum_p = 0.0
    sum_t = 0.0
    op_correct = 0
    q_count = 0
    t_count = 0

    with torch.no_grad():
        for batch in loader:
            history_seq = batch["history"].to(config.device)
            context = batch["context"].to(config.device)
            action_op = batch["action_op"].squeeze(-1).to(config.device)
            target_q = batch["target_q"].to(config.device)
            target_p = batch["target_p"].to(config.device)
            target_t = batch["target_t"].squeeze(-1).to(config.device)
            time_mask = batch["time_mask"].to(config.device)
            time_valid = batch["time_valid"].squeeze(-1).to(config.device)

            op_logits, quantity, price, time_logits = model(history_seq, context, time_mask)

            sum_op += float(F.cross_entropy(op_logits, action_op, reduction="sum").item())
            preds = torch.argmax(op_logits, dim=1)
            op_correct += int((preds == action_op).sum().item())
            total_samples += int(action_op.numel())

            counter_mask = action_op == 1
            valid_mask = counter_mask & (time_valid > 0.5)
            if valid_mask.any():
                count = int(valid_mask.sum().item())
                sum_q += float(F.mse_loss(quantity[valid_mask], target_q[valid_mask], reduction="sum").item())
                sum_p += float(F.mse_loss(price[valid_mask], target_p[valid_mask], reduction="sum").item())
                sum_t += float(F.cross_entropy(time_logits[valid_mask], target_t[valid_mask], reduction="sum").item())
                q_count += count
                t_count += count

    avg_op = sum_op / max(1, total_samples)
    avg_q = sum_q / max(1, q_count)
    avg_p = sum_p / max(1, q_count)
    avg_t = sum_t / max(1, t_count)
    acc = op_correct / max(1, total_samples)
    total_loss = avg_op + avg_q + avg_p + avg_t
    return {
        "loss": total_loss,
        "loss_op": avg_op,
        "loss_q": avg_q,
        "loss_p": avg_p,
        "loss_t": avg_t,
        "op_acc": acc,
        "samples": float(total_samples),
    }


def _eval_l4(model, samples, config: TrainConfig, batch_size: int) -> Dict[str, float]:
    import torch

    dataset = L4Dataset(samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_l4,
    )

    model.to(config.device)
    model.eval()

    sum_loss = 0.0
    sum_mask = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            global_state = batch["global_state"].to(config.device)
            thread_states = batch["thread_states"].to(config.device)
            teacher = batch["teacher_alpha"].to(config.device)
            mask = batch["thread_mask"].to(config.device)

            pred = model(thread_states, global_state, thread_mask=mask)
            diff = pred - teacher
            sum_loss += float((diff * diff * mask.float()).sum().item())
            sum_mask += float(mask.float().sum().item())
            total_batches += 1

    avg_loss = sum_loss / max(1.0, sum_mask)
    return {"loss": avg_loss, "batches": float(total_batches)}


def _report_eval(title: str, metrics: Dict[str, float]) -> None:
    fields = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if k != "samples" and k != "batches"])
    extra = []
    if "samples" in metrics:
        extra.append(f"samples={int(metrics['samples'])}")
    if "batches" in metrics:
        extra.append(f"batches={int(metrics['batches'])}")
    extra_str = f" ({', '.join(extra)})" if extra else ""
    print(f"[EVAL] {title}: {fields}{extra_str}")

def _train_l2(
    macro_samples,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    resume_path: Optional[Path],
    eval_sets: Optional[Dict[str, List[data_pipeline.MacroSample]]] = None,
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

    if eval_sets:
        for name, samples in eval_sets.items():
            if samples:
                metrics = _eval_l2(model, samples, config, batch_size=args.l2_batch_size)
                _report_eval(f"L2/{name}", metrics)
    return Path(save_model(model, config.output_dir, "l2_bc.pt"))


def _train_l3(
    micro_samples,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    resume_path: Optional[Path],
    awr_resume_path: Optional[Path],
    run_awr: bool,
    eval_sets: Optional[Dict[str, List[data_pipeline.MicroSample]]] = None,
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

    if eval_sets:
        for name, samples in eval_sets.items():
            if samples:
                metrics = _eval_l3(model, samples, config, batch_size=args.l3_batch_size)
                _report_eval(f"L3/{name}", metrics)
    return Path(save_model(model, config.output_dir, "l3_bc.pt"))


def _train_l4(
    l4_samples,
    *,
    output_dir: Path,
    args: argparse.Namespace,
    resume_path: Optional[Path],
    eval_sets: Optional[Dict[str, List[data_pipeline.L4DistillSample]]] = None,
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

    if eval_sets:
        for name, samples in eval_sets.items():
            if samples:
                metrics = _eval_l4(model, samples, config, batch_size=args.l4_batch_size)
                _report_eval(f"L4/{name}", metrics)
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
    parser.add_argument("--val-ratio", type=float, default=0.1, help="validation split ratio (world-level)")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="test split ratio (world-level)")
    parser.add_argument("--split-seed", type=int, default=42, help="split random seed")
    parser.add_argument("--regen-split", action="store_true", help="re-generate world split cache")

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

    world_dirs = _load_world_dirs(tournament_dir)
    if not world_dirs:
        raise RuntimeError(f"No world directories found in {tournament_dir}")

    split_cache = output_dir / "world_split.json"
    if args.val_ratio > 0 or args.test_ratio > 0:
        cached = None if args.regen_split else _load_split_cache(split_cache, tournament_dir)
        if cached:
            train_dirs, val_dirs, test_dirs = cached
        else:
            train_dirs, val_dirs, test_dirs = _split_world_dirs(
                world_dirs,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.split_seed,
            )
            _save_split_cache(split_cache, tournament_dir, train_dirs, val_dirs, test_dirs)
    else:
        train_dirs, val_dirs, test_dirs = world_dirs, [], []

    print(f"[INFO] world splits: train={len(train_dirs)} val={len(val_dirs)} test={len(test_dirs)}")

    # 先做一次不回填的解析与统计，便于快速发现结构性异常。
    train_macro, train_micro = _load_samples_from_world_dirs(
        train_dirs,
        agent_name=args.agent_name,
        strict_json_only=strict_json_only,
        goal_backfill="none",
        l2_model_path=None,
        num_workers=num_workers,
    )
    _summarize_samples(train_macro, train_micro, sample_limit=args.stats_sample, title="train")

    val_macro: List[data_pipeline.MacroSample] = []
    val_micro: List[data_pipeline.MicroSample] = []
    test_macro: List[data_pipeline.MacroSample] = []
    test_micro: List[data_pipeline.MicroSample] = []

    if val_dirs:
        val_macro, val_micro = _load_samples_from_world_dirs(
            val_dirs,
            agent_name=args.agent_name,
            strict_json_only=strict_json_only,
            goal_backfill="none",
            l2_model_path=None,
            num_workers=num_workers,
        )
        _summarize_samples(val_macro, val_micro, sample_limit=args.stats_sample, title="val")

    if test_dirs:
        test_macro, test_micro = _load_samples_from_world_dirs(
            test_dirs,
            agent_name=args.agent_name,
            strict_json_only=strict_json_only,
            goal_backfill="none",
            l2_model_path=None,
            num_workers=num_workers,
        )
        _summarize_samples(test_macro, test_micro, sample_limit=args.stats_sample, title="test")

    if args.stats_only or not phases:
        return

    l2_model_path = _resolve_l2_model_path(output_dir, args.l2_model_path)

    if "l2" in phases:
        resume_path = None
        if not args.no_resume:
            resume_path = _find_latest_ckpt(output_dir, r"l2_bc_epoch(\d+)\.ckpt\.pt$")
        eval_sets = {}
        if val_macro:
            eval_sets["val"] = val_macro
        if test_macro:
            eval_sets["test"] = test_macro
        l2_model_path = _train_l2(
            train_macro,
            output_dir=output_dir,
            args=args,
            resume_path=resume_path,
            eval_sets=eval_sets,
        )

    del train_macro
    del val_macro
    del test_macro
    del train_micro
    del val_micro
    del test_micro
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
        _, train_micro = _load_samples_from_world_dirs(
            train_dirs,
            agent_name=args.agent_name,
            strict_json_only=strict_json_only,
            goal_backfill=args.l3_goal_backfill,
            l2_model_path=str(l2_model_path) if l2_model_path else None,
            num_workers=num_workers,
        )
        eval_sets = {}
        if val_dirs:
            _, val_micro = _load_samples_from_world_dirs(
                val_dirs,
                agent_name=args.agent_name,
                strict_json_only=strict_json_only,
                goal_backfill=args.l3_goal_backfill,
                l2_model_path=str(l2_model_path) if l2_model_path else None,
                num_workers=num_workers,
            )
            eval_sets["val"] = val_micro
        if test_dirs:
            _, test_micro = _load_samples_from_world_dirs(
                test_dirs,
                agent_name=args.agent_name,
                strict_json_only=strict_json_only,
                goal_backfill=args.l3_goal_backfill,
                l2_model_path=str(l2_model_path) if l2_model_path else None,
                num_workers=num_workers,
            )
            eval_sets["test"] = test_micro

        _train_l3(
            train_micro,
            output_dir=output_dir,
            args=args,
            resume_path=resume_path,
            awr_resume_path=awr_resume_path,
            run_awr=args.l3_awr,
            eval_sets=eval_sets,
        )
        del train_micro
        if "val" in eval_sets:
            del eval_sets["val"]
        if "test" in eval_sets:
            del eval_sets["test"]
        gc.collect()

    if "l4" in phases:
        if args.l4_goal_source == "l2" and not l2_model_path:
            raise ValueError("l4-goal-source=l2 but no L2 model path found")

        resume_path = None
        if not args.no_resume:
            resume_path = _find_latest_ckpt(output_dir, r"l4_distill_epoch_(\d+)\.pt$")

        # L4 蒸馏基于启发式教师信号，需要单独抽取样本。
        train_l4 = _load_l4_samples_from_world_dirs(
            train_dirs,
            agent_name=args.agent_name,
            strict_json_only=strict_json_only,
            goal_source=args.l4_goal_source,
            l2_model_path=str(l2_model_path) if l2_model_path else None,
            num_workers=num_workers,
        )
        eval_sets = {}
        if val_dirs:
            val_l4 = _load_l4_samples_from_world_dirs(
                val_dirs,
                agent_name=args.agent_name,
                strict_json_only=strict_json_only,
                goal_source=args.l4_goal_source,
                l2_model_path=str(l2_model_path) if l2_model_path else None,
                num_workers=num_workers,
            )
            eval_sets["val"] = val_l4
        if test_dirs:
            test_l4 = _load_l4_samples_from_world_dirs(
                test_dirs,
                agent_name=args.agent_name,
                strict_json_only=strict_json_only,
                goal_source=args.l4_goal_source,
                l2_model_path=str(l2_model_path) if l2_model_path else None,
                num_workers=num_workers,
            )
            eval_sets["test"] = test_l4

        _train_l4(
            train_l4,
            output_dir=output_dir,
            args=args,
            resume_path=resume_path,
            eval_sets=eval_sets,
        )
        del train_l4
        if "val" in eval_sets:
            del eval_sets["val"]
        if "test" in eval_sets:
            del eval_sets["test"]
        gc.collect()

    print(f"[INFO] Training finished. Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
