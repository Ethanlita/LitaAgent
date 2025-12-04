"""
调试挂死用的 monkeypatch。

当环境变量 SCML_PATCH_WORKER_TRACE=1 时：
- 将 negmas.tournaments._run_worlds 包装，记录 worker 的开始/结束/异常
- 日志写到 SCML_WORKER_TRACE_FILE 指定的路径

不设置环境变量时不做任何修改，避免影响正常运行。
"""

import os
import json
import time
import atexit
from pathlib import Path


def _log_trace(filepath: Path, event: str, **data):
    payload = {"event": event, "pid": os.getpid(), "ts": time.time()}
    payload.update(data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


if os.environ.get("SCML_PATCH_WORKER_TRACE") == "1":
    trace_file = Path(
        os.environ.get("SCML_WORKER_TRACE_FILE", "./diagnose_logs/worker_trace.log")
    )
    try:
        from negmas.tournaments import tournaments as nt

        original_run_worlds = nt._run_worlds

        def traced_run_worlds(
            worlds_params,
            world_generator,
            score_calculator,
            world_progress_callback,
            dry_run,
            save_world_stats,
            override_ran_worlds,
            save_progress_every,
            attempts_path,
            max_attempts,
            verbose,
        ):
            try:
                run_id = nt._hash(worlds_params)
            except Exception:
                run_id = None
            names = []
            try:
                for wp in worlds_params:
                    name = None
                    if isinstance(wp, dict):
                        wp_info = wp.get("world_params") or {}
                        name = wp_info.get("name") or wp_info.get("config_id")
                    names.append(name)
            except Exception:
                pass
            _log_trace(trace_file, "worker_start", run_id=run_id, names=names)
            atexit.register(lambda: _log_trace(trace_file, "worker_exit", run_id=run_id))
            try:
                result = original_run_worlds(
                    worlds_params,
                    world_generator,
                    score_calculator,
                    world_progress_callback,
                    dry_run,
                    save_world_stats,
                    override_ran_worlds,
                    save_progress_every,
                    attempts_path,
                    max_attempts,
                    verbose,
                )
                _log_trace(trace_file, "worker_done", run_id=run_id)
                return result
            except Exception as e:
                _log_trace(trace_file, "worker_error", run_id=run_id, error=str(e))
                raise

        nt._run_worlds = traced_run_worlds
    except Exception as e:  # pragma: no cover - 仅调试
        _log_trace(trace_file, "patch_fail", error=str(e))
