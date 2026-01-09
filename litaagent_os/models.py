from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional

from .config import LitaOSConfig


@dataclass
class PartnerStatsFeature:
    accept_rate_last_k: float = 0.0
    counter_rate_last_k: float = 0.0
    breach_rate_last_k: float = 0.0


@dataclass
class ContextFeatures:
    round_rel: float
    round_bucket: int
    role: str  # "BUYER" / "SELLER"
    need_norm: float
    trading_price: float | None
    pen_shortfall_norm: float | None
    cost_disposal_norm: float | None
    price_min: float
    price_max: float
    system_breach_prob: float | None
    system_breach_level: float | None
    partner_stats: PartnerStatsFeature


@dataclass
class OfferFeatures:
    q: int
    q_norm: float
    q_bucket: int
    p: float
    p_bin: int
    p_norm: float
    t: int


class AcceptModelInterface:
    def predict_accept_mu(
        self, context: ContextFeatures, offer: OfferFeatures, history_tokens: list[Any]
    ) -> float:
        raise NotImplementedError

    def predict_accept_strength(
        self, context: ContextFeatures, offer: OfferFeatures, history_tokens: list[Any]
    ) -> float:
        raise NotImplementedError


class BreachModelInterface:
    def predict_breach_mu(self, context: ContextFeatures, partner_features: dict[str, Any]) -> Optional[float]:
        raise NotImplementedError


class LinearLogisticAcceptModel(AcceptModelInterface):
    """线性 Logistic 接受概率模型（可从 JSON 权重加载）。"""

    def __init__(self, cfg: LitaOSConfig) -> None:
        self._cfg = cfg
        self._weights: dict[str, float] = {}
        self._bias = 0.0
        self._strength = cfg.prior_strength_s

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self._weights = {str(k): float(v) for k, v in data.get("weights", {}).items()}
        self._bias = float(data.get("bias", 0.0))
        self._strength = float(data.get("strength", self._strength))

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _feature_dict(self, context: ContextFeatures, offer: OfferFeatures) -> dict[str, float]:
        role_is_seller = 1.0 if context.role == "SELLER" else 0.0
        feats = {
            "round_rel": context.round_rel,
            "role_is_seller": role_is_seller,
            "need_norm": context.need_norm,
            "q_norm": offer.q_norm,
            "p_bin": float(offer.p_bin),
            "p_norm": offer.p_norm,
            "partner_accept_rate_last_k": context.partner_stats.accept_rate_last_k,
            "partner_counter_rate_last_k": context.partner_stats.counter_rate_last_k,
            "partner_breach_rate_last_k": context.partner_stats.breach_rate_last_k,
        }
        if context.trading_price is not None:
            feats["trading_price"] = context.trading_price
        if context.pen_shortfall_norm is not None:
            feats["pen_shortfall_norm"] = context.pen_shortfall_norm
        if context.cost_disposal_norm is not None:
            feats["cost_disposal_norm"] = context.cost_disposal_norm
        if context.system_breach_prob is not None:
            feats["system_breach_prob"] = context.system_breach_prob
        if context.system_breach_level is not None:
            feats["system_breach_level"] = context.system_breach_level
        return feats

    def predict_accept_mu(
        self, context: ContextFeatures, offer: OfferFeatures, history_tokens: list[Any]
    ) -> float:
        feats = self._feature_dict(context, offer)
        score = self._bias
        for name, weight in self._weights.items():
            score += weight * feats.get(name, 0.0)
        return self._sigmoid(score)

    def predict_accept_strength(
        self, context: ContextFeatures, offer: OfferFeatures, history_tokens: list[Any]
    ) -> float:
        return self._strength
