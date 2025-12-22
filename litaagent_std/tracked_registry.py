from __future__ import annotations

import importlib
import json
import os
from typing import Dict

REGISTRY_ENV = "LITA_TRACKED_REGISTRY"


def _load_registry() -> Dict[str, str]:
    raw = os.environ.get(REGISTRY_ENV, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def register(tracked_name: str, base_path: str) -> None:
    registry = _load_registry()
    registry[tracked_name] = base_path
    os.environ[REGISTRY_ENV] = json.dumps(registry, ensure_ascii=False)


def _import_base(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def __getattr__(name: str):
    registry = _load_registry()
    base_path = registry.get(name)
    if not base_path:
        raise AttributeError(f"{__name__} has no attribute {name!r}")
    from litaagent_std.tracker_mixin import create_tracked_agent

    base_cls = _import_base(base_path)
    log_dir = os.environ.get("SCML_TRACKER_LOG_DIR", ".")
    tracked_cls = create_tracked_agent(
        base_cls,
        log_dir=log_dir,
        agent_name_suffix="",
        register=False,
        register_module=__name__,
        class_name=name,
    )
    return tracked_cls
