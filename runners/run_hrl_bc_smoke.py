# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from litaagent_std.hrl_xf.data_pipeline import load_tournament_data
from litaagent_std.hrl_xf.training import TrainConfig, HRLXFTrainer
from litaagent_std.hrl_xf.l2_manager import HorizonManagerPPO
from litaagent_std.hrl_xf.l3_executor import L3DecisionTransformer
import litaagent_std.hrl_xf.training as training_mod


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("tournament_history") / f"hrl_bc_smoke_{timestamp}"


def _run_tournament(args: argparse.Namespace, output_dir: Path) -> None:
    if args.skip_run:
        if not output_dir.exists():
            raise RuntimeError(f"skip-run 但输出目录不存在: {output_dir}")
        return

    cmd = [
        sys.executable,
        "-m",
        "runners.hrl_data_runner",
        "--output-dir",
        str(output_dir),
        "--configs",
        str(args.configs),
        "--runs",
        str(args.runs),
        "--agent-set",
        args.agent_set,
        "--parallelism",
        args.parallelism,
    ]

    if args.n_competitors_per_world is not None:
        cmd += ["--n-competitors-per-world", str(args.n_competitors_per_world)]
    if args.max_worlds_per_config is not None:
        cmd += ["--max-worlds-per-config", str(args.max_worlds_per_config)]
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    if not args.track_all:
        cmd.append("--track-only-penguin")
    if not args.allow_csv:
        cmd.append("--no-csv")
    if not args.auto_collect:
        cmd.append("--no-auto-collect")
    if not args.background:
        cmd.append("--foreground")

    subprocess.run(cmd, check=True)


def _run_pipeline(args: argparse.Namespace, output_dir: Path):
    strict_json_only = not args.allow_csv
    macro_samples, micro_samples = load_tournament_data(
        str(output_dir),
        agent_name=args.agent_name,
        strict_json_only=strict_json_only,
        num_workers=1,
        goal_backfill=args.goal_backfill,
        l2_model_path=args.l2_model_path,
    )
    if not macro_samples:
        raise RuntimeError("宏观样本为空，无法训练 L2")
    if not micro_samples:
        raise RuntimeError("微观样本为空，无法训练 L3")
    return macro_samples, micro_samples


def _run_training(args: argparse.Namespace, output_dir: Path, macro_samples, micro_samples) -> None:
    if not training_mod.TORCH_AVAILABLE:
        raise RuntimeError("PyTorch 不可用，无法进行训练")

    train_dir = output_dir / "_train_smoke"
    config = TrainConfig(
        output_dir=str(train_dir),
        l2_epochs=args.l2_epochs,
        l3_epochs=args.l3_epochs,
        l2_batch_size=args.l2_batch_size,
        l3_batch_size=args.l3_batch_size,
        device=args.device,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    l2_model = HorizonManagerPPO(horizon=config.horizon)
    l3_model = L3DecisionTransformer(horizon=config.horizon)
    trainer = HRLXFTrainer(l2_model=l2_model, l3_model=l3_model, config=config)
    trainer.train_phase0_bc(macro_samples, micro_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="HRL 小规模 BC 闭环验证脚本")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认自动生成）")
    parser.add_argument("--skip-run", action="store_true", help="跳过比赛，仅处理已有目录")
    parser.add_argument("--agent-name", type=str, default="PenguinAgent", help="数据提取的代理名")
    parser.add_argument("--goal-backfill", type=str, default="v2", choices=["none", "v2", "l2"])
    parser.add_argument("--l2-model-path", type=str, default=None, help="goal-backfill=l2 时使用的 L2 权重")

    parser.add_argument("--configs", type=int, default=1, help="World 配置数量")
    parser.add_argument("--runs", type=int, default=1, help="每配置运行次数")
    parser.add_argument("--n-competitors-per-world", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--max-worlds-per-config", type=int, default=2, help="每配置最大 world 数")
    parser.add_argument("--steps", type=int, default=10, help="每个 world 的步数（小规模验证）")
    parser.add_argument("--agent-set", type=str, default="phh", choices=["default", "phh"])
    parser.add_argument("--parallelism", type=str, default="serial")

    parser.add_argument("--track-all", action="store_true", help="追踪所有参赛者")
    parser.add_argument("--allow-csv", action="store_true", help="允许 CSV 降级")
    parser.add_argument("--auto-collect", action="store_true", help="启用自动归集")
    parser.add_argument("--background", action="store_true", help="比赛输出写入日志文件")

    parser.add_argument("--l2-epochs", type=int, default=1)
    parser.add_argument("--l3-epochs", type=int, default=1)
    parser.add_argument("--l2-batch-size", type=int, default=8)
    parser.add_argument("--l3-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir().resolve()

    print(f"[INFO] 输出目录: {output_dir}")
    _run_tournament(args, output_dir)
    macro_samples, micro_samples = _run_pipeline(args, output_dir)
    _run_training(args, output_dir, macro_samples, micro_samples)
    print(f"[INFO] 闭环验证完成，训练输出目录: {output_dir / '_train_smoke'}")


if __name__ == "__main__":
    main()
