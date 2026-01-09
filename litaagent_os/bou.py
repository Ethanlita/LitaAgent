from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scipy.stats import beta as beta_dist

from .config import LitaOSConfig
from .utils import q_bucket_coarse, clamp


@dataclass
class BetaState:
    alpha: float
    beta: float


class BOUEstimator:
    def __init__(self, config: LitaOSConfig) -> None:
        self._cfg = config
        self._state: dict[tuple[str, str, int, int, int], BetaState] = {}

    def _key(
        self, pid: str, role: str, round_bucket: int, p_bin: int, q: int
    ) -> Optional[tuple[str, str, int, int, int]]:
        coarse = q_bucket_coarse(q)
        if coarse is None:
            return None
        return (pid, role, round_bucket, p_bin, coarse)

    def _init_state(self, mu: float, strength: float) -> BetaState:
        eps = self._cfg.mu_eps
        mu = clamp(mu, eps, 1.0 - eps)
        strength = max(strength, 2.0)
        alpha = mu * strength
        beta = (1.0 - mu) * strength
        return BetaState(alpha=alpha, beta=beta)

    def get_state(
        self,
        pid: str,
        role: str,
        round_bucket: int,
        p_bin: int,
        q: int,
        mu: float,
        strength: float,
    ) -> Optional[BetaState]:
        key = self._key(pid, role, round_bucket, p_bin, q)
        if key is None:
            return None
        if key not in self._state:
            self._state[key] = self._init_state(mu, strength)
        return self._state[key]

    def update(
        self,
        pid: str,
        role: str,
        round_bucket: int,
        p_bin: int,
        q: int,
        mu: float,
        strength: float,
        accepted: bool,
        terminal_negative: bool = False,
    ) -> None:
        state = self.get_state(pid, role, round_bucket, p_bin, q, mu, strength)
        if state is None:
            return
        if terminal_negative:
            state.beta += self._cfg.w_end
            return
        if accepted:
            state.alpha += 1.0
        else:
            state.beta += 1.0

    def lcb(
        self,
        pid: str,
        role: str,
        round_bucket: int,
        p_bin: int,
        q: int,
        mu: float,
        strength: float,
        delta: float,
    ) -> Optional[float]:
        state = self.get_state(pid, role, round_bucket, p_bin, q, mu, strength)
        if state is None:
            return None
        return float(beta_dist.ppf(delta, state.alpha, state.beta))
