from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class LitaOSConfig:
    """LitaAgent-OS 超参数与阈值配置"""

    # 模型工件与配置
    model_root_dir: str = "assets/models"
    accept_model_dir: str = "assets/models/accept"
    breach_model_dir: str = "assets/models/breach"
    agent_config_path: str = "assets/agent_config.json"

    # 时间与分桶
    round_bucket_T: int = 5

    # BOU 相关
    prior_strength_s: float = 8.0
    lcb_delta_accept: float = 0.2
    lcb_delta_fulfill: float = 0.2
    w_end: float = 0.2
    add_terminal_negative: bool = False

    # 概率数值安全
    mu_eps: float = 1e-4

    # probe 机制
    probe_q: int = 1
    probe_q_min: int = 1  # probe 时每个对手的最小 q
    probe_steps: int = 2
    probe_lcb_threshold: float = 0.30  # 降低以避免先验触发 probe
    q_candidate: int = 2

    # 组合优化
    buffer_min: float = 0.05
    buffer_max: float = 0.35
    buffer_exp: float = 2.0
    portfolio_k: int = 6
    risk_lambda: float = 0.0
    overfill_penalty_ratio: float = 0.1

    # 价格策略
    use_trading_price: bool = True
    price_concession_gamma: float = 2.0
    counter_anchor_eta_exp: float = 1.0  # eta = round_rel^k，用于 counter 价格锚定
    counter_monotonic: bool = True  # 是否强制单调让步

    # Panic 模式
    panic_enabled: bool = True
    panic_penalty_ratio_threshold: float = 3.0  # R > 此值触发 panic
    panic_round_rel_threshold: float = 0.6  # round_rel > 此值才检查 panic

    # 训练模型默认强度
    default_accept_strength: float = 8.0

    def load_overrides(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        self.apply_overrides(data)

    def apply_overrides(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, bool):
                    setattr(self, key, bool(value))
                elif isinstance(current, int):
                    setattr(self, key, int(value))
                elif isinstance(current, float):
                    setattr(self, key, float(value))
                elif isinstance(current, str):
                    setattr(self, key, str(value))
                else:
                    setattr(self, key, value)
            except Exception:
                continue
