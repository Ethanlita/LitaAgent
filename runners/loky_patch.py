"""
loky 执行器补丁（幂等）。

作用：
- 替换 negmas `tournaments._get_executor`，当并行模式是 loky（或 loky:<fraction>）时，返回
  `joblib.externals.loky.ProcessPoolExecutor`，并继续使用 `concurrent.futures.as_completed`。
- 支持通过环境变量控制，默认读取 `SCML_PARALLELISM`（默认值 `loky`）。
- 支持 `loky:<fraction>` 限制并发度（按 CPU * fraction，至少 1）。
- 设置环境变量避免 PyTorch/OpenMP 在多进程中的冲突。

用法（在任何 runner 开头调用）:
    from runners.loky_patch import enable_loky_executor
    enable_loky_executor()
    # 之后保持 parallelism='parallel' 传入 negmas 即可。
"""

from __future__ import annotations

import os
import concurrent.futures as cf

# 在导入任何可能使用 OpenMP 的库之前设置环境变量
# 这些设置可以避免 PyTorch + loky 多进程在 macOS 上的 segfault
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# PyTorch 特定设置
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

from joblib.externals.loky import ProcessPoolExecutor as LokyExecutor
from joblib.externals.loky import process_executor as loky_process_executor

if not getattr(loky_process_executor, "_litaagent_patch", False):
    _orig_check = loky_process_executor._check_system_limits

    def _safe_check_system_limits():
        try:
            return _orig_check()
        except PermissionError:
            return

    loky_process_executor._check_system_limits = _safe_check_system_limits
    loky_process_executor._litaagent_patch = True


def _parse_max_workers(mode: str | None) -> int | None:
    """解析 loky:<fraction> 形式的并发数。"""
    if not mode or ":" not in mode:
        return None
    try:
        frac = float(mode.split(":", 1)[1])
        if 0 < frac <= 1:
            return max(1, int((os.cpu_count() or 1) * frac))
    except Exception:
        return None
    return None


def enable_loky_executor(env_var: str = "SCML_PARALLELISM", default_mode: str = "loky") -> None:
    """
    启用 loky 替换。可重复调用（幂等）。

    - 若 env_var 设置为 loky 或 loky:<fraction>，或未设置时使用 default_mode，
      则返回 loky 的 ProcessPoolExecutor。
    - 否则回退 negmas 原始 executor。
    """
    import negmas.tournaments.tournaments as nt  # 延迟导入避免副作用

    if getattr(nt, "_loky_patched", False):
        return

    original_get_executor = nt._get_executor

    def patched_get_executor(parallelism, verbose, total_timeout=None, scheduler_ip=None, scheduler_port=None):
        requested = os.environ.get(env_var, default_mode)
        effective = requested or parallelism
        if isinstance(effective, str) and effective.startswith("loky"):
            max_workers = _parse_max_workers(effective)
            kwargs = {}
            if max_workers:
                kwargs["max_workers"] = max_workers
            executor = LokyExecutor(**kwargs)
            return executor, cf.as_completed
        return original_get_executor(parallelism, verbose, total_timeout, scheduler_ip, scheduler_port)

    nt._get_executor = patched_get_executor
    nt._loky_patched = True
