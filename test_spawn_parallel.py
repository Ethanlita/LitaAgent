"""在 Windows 默认的 spawn 启动方式下做并行冒烟测试。

目的：
- 强制使用 multiprocessing 的 ``spawn`` 启动方式（模拟 Windows）。
- 运行一个极小的并行 anac2024_oneshot 赛局，验证 Tracker 的序列化不会卡死。
- 退出前检查是否写出了追踪日志文件。
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile
from pathlib import Path


def run_spawn_parallel() -> int:
    # 在 Linux 下默认是 ``fork``，这里强制切到 ``spawn`` 来模拟 Windows 行为。
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    with tempfile.TemporaryDirectory(prefix="tracker_spawn_") as tmpdir:
        log_dir = Path(tmpdir) / "tracker_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SCML_TRACKER_LOG_DIR"] = str(log_dir.resolve())

        from litaagent_std.litaagent_y import LitaAgentYTracked
        from litaagent_std.litaagent_p import LitaAgentP
        from scml.utils import anac2024_oneshot

        results = anac2024_oneshot(
            competitors=[LitaAgentYTracked, LitaAgentP],
            n_configs=1,
            n_runs_per_world=1,
            n_steps=5,
            parallelism="parallel",
            print_exceptions=True,
            verbose=False,
        )

        # 只要返回结果列表非空且日志文件生成，就认为序列化链路畅通。
        tracker_files = list(log_dir.glob("agent_*.json"))
        print(f"Tracker logs generated: {len(tracker_files)}")
        if not tracker_files:
            print("❌ 未生成任何 Tracker 日志文件")
            return 1

        if not results:
            print("❌ anac2024_oneshot 未返回结果")
            return 1

    return 0


def main() -> None:
    code = run_spawn_parallel()
    if code == 0:
        print("✓ spawn 并行冒烟测试通过")
    sys.exit(code)


if __name__ == "__main__":
    main()
