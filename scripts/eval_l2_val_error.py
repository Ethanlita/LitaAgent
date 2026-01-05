#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


def _ensure_repo_root() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="training_runs/l2_qlog1p_w2_20251231_220338")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--agent-name", type=str, default="PenguinAgent")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--strict-json-only", action="store_true", default=True)
    parser.add_argument("--allow-csv", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    _ensure_repo_root()

    from runners import run_hrl_bc_awr_train as runner
    from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
    from litaagent_std.hrl_xf.training import TrainConfig, L2Dataset
    from litaagent_std.hrl_xf import training as training_mod

    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_path = output_dir / "world_split.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    data = json.loads(split_path.read_text())
    splits = data.get("splits", {})
    world_dirs = splits.get(args.split, [])
    if not world_dirs:
        raise RuntimeError(f"split is empty: {args.split}")

    strict_json_only = True
    if args.allow_csv:
        strict_json_only = False
    if not args.strict_json_only:
        strict_json_only = False

    macro_samples, _ = runner._load_samples_from_world_dirs(
        world_dirs,
        agent_name=args.agent_name,
        strict_json_only=strict_json_only,
        goal_backfill="none",
        l2_model_path=None,
        l2_backfill_device="cpu",
        l2_backfill_batch_size=256,
        num_workers=args.num_workers,
        progress_label=f"{args.split}(eval)",
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainConfig(
        output_dir=str(output_dir),
        device=device,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=2,
        dataloader_drop_last=False,
    )

    dataset = L2Dataset(macro_samples)
    loader = training_mod.build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        config=config,
    )

    model = HorizonManagerPPO(horizon=config.horizon)
    model_path = output_dir / "l2_bc.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    q_idx = [0, 2, 4, 6, 8, 10, 12, 14]
    p_idx = [1, 3, 5, 7, 9, 11, 13, 15]
    buy_idx = [0, 1, 4, 5, 8, 9, 12, 13]
    sell_idx = [2, 3, 6, 7, 10, 11, 14, 15]

    sum_abs_q = 0.0
    sum_sq_q = 0.0
    sum_target_q = 0.0
    sum_rel_q = 0.0
    cnt_q = 0.0

    sum_abs_p = 0.0
    sum_sq_p = 0.0
    sum_target_p = 0.0
    sum_rel_p = 0.0
    cnt_p = 0.0

    with torch.no_grad():
        for batch in loader:
            state_static = batch["state_static"].to(device)
            state_temporal = batch["state_temporal"].to(device)
            x_role = batch["x_role"].to(device)
            goal_target = batch["goal"].to(device)

            goal_pred = model(state_static, state_temporal, x_role)
            if isinstance(goal_pred, tuple):
                goal_pred = goal_pred[0]

            B = x_role.size(0)
            loss_mask = torch.ones(B, 16, device=device)
            can_buy = x_role[:, 0:1]
            can_sell = x_role[:, 1:2]
            for idx in buy_idx:
                loss_mask[:, idx] = can_buy.squeeze(-1)
            for idx in sell_idx:
                loss_mask[:, idx] = can_sell.squeeze(-1)

            err = goal_pred - goal_target
            abs_err = err.abs()
            sq_err = err * err

            mask_q = loss_mask[:, q_idx]
            if mask_q.sum() > 0:
                sum_abs_q += float((abs_err[:, q_idx] * mask_q).sum().item())
                sum_sq_q += float((sq_err[:, q_idx] * mask_q).sum().item())
                sum_target_q += float((goal_target[:, q_idx] * mask_q).sum().item())
                denom_q = torch.clamp(goal_target[:, q_idx].abs(), min=1.0)
                sum_rel_q += float(((abs_err[:, q_idx] / denom_q) * mask_q).sum().item())
                cnt_q += float(mask_q.sum().item())

            mask_p = loss_mask[:, p_idx]
            if mask_p.sum() > 0:
                sum_abs_p += float((abs_err[:, p_idx] * mask_p).sum().item())
                sum_sq_p += float((sq_err[:, p_idx] * mask_p).sum().item())
                sum_target_p += float((goal_target[:, p_idx] * mask_p).sum().item())
                denom_p = torch.clamp(goal_target[:, p_idx].abs(), min=1.0)
                sum_rel_p += float(((abs_err[:, p_idx] / denom_p) * mask_p).sum().item())
                cnt_p += float(mask_p.sum().item())

    def _safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    mae_q = _safe_div(sum_abs_q, cnt_q)
    rmse_q = math.sqrt(_safe_div(sum_sq_q, cnt_q)) if cnt_q > 0 else float("nan")
    mean_q = _safe_div(sum_target_q, cnt_q)
    mre_q = _safe_div(sum_rel_q, cnt_q)

    mae_p = _safe_div(sum_abs_p, cnt_p)
    rmse_p = math.sqrt(_safe_div(sum_sq_p, cnt_p)) if cnt_p > 0 else float("nan")
    mean_p = _safe_div(sum_target_p, cnt_p)
    mre_p = _safe_div(sum_rel_p, cnt_p)

    print("\n=== L2 val error (raw scale) ===")
    print(f"Q: MAE={mae_q:.4f}, RMSE={rmse_q:.4f}, mean_target={mean_q:.4f}, mean_rel_err={mre_q:.4f}")
    print(f"P: MAE={mae_p:.4f}, RMSE={rmse_p:.4f}, mean_target={mean_p:.4f}, mean_rel_err={mre_p:.4f}")
    print(f"count_q={int(cnt_q)}, count_p={int(cnt_p)}")


if __name__ == "__main__":
    main()
