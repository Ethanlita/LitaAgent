from __future__ import annotations

from typing import Optional

from scml.oneshot.common import QUANTITY, TIME, UNIT_PRICE

from .config import LitaOSConfig
from .models import ContextFeatures, OfferFeatures, PartnerStatsFeature
from .partner_stats import PartnerStatsSnapshot
from .utils import p_bin, p_norm, q_bucket, q_norm, round_bucket, safe_round_rel


def build_offer_features(offer: tuple[int, int, float], q_max: int, p_min: float, p_max: float) -> OfferFeatures:
    q, t, p = int(offer[QUANTITY]), int(offer[TIME]), float(offer[UNIT_PRICE])
    return OfferFeatures(
        q=q,
        q_norm=q_norm(q, q_max),
        q_bucket=q_bucket(q, q_max),
        p=p,
        p_bin=p_bin(p, p_min, p_max),
        p_norm=p_norm(p, p_min, p_max),
        t=t,
    )


def build_context_features(
    cfg: LitaOSConfig,
    role: str,
    round_rel: float,
    p_min: float,
    p_max: float,
    need_remaining: int,
    q_max: int,
    trading_price: Optional[float],
    pen_shortfall: Optional[float],
    cost_disposal: Optional[float],
    system_breach_prob: Optional[float],
    system_breach_level: Optional[float],
    partner_stats: PartnerStatsSnapshot,
) -> ContextFeatures:
    need_norm = 0.0 if q_max <= 0 else need_remaining / float(q_max)
    pen_norm = float(pen_shortfall) if pen_shortfall is not None else None
    cost_norm = float(cost_disposal) if cost_disposal is not None else None
    return ContextFeatures(
        round_rel=round_rel,
        round_bucket=round_bucket(round_rel, cfg.round_bucket_T),
        role=role,
        need_norm=need_norm,
        trading_price=trading_price,
        pen_shortfall_norm=pen_norm,
        cost_disposal_norm=cost_norm,
        price_min=p_min,
        price_max=p_max,
        system_breach_prob=system_breach_prob,
        system_breach_level=system_breach_level,
        partner_stats=PartnerStatsFeature(
            accept_rate_last_k=partner_stats.accept_rate_last_k,
            counter_rate_last_k=partner_stats.counter_rate_last_k,
            breach_rate_last_k=partner_stats.breach_rate_last_k,
        ),
    )


def compute_round_rel(cfg: LitaOSConfig, state) -> float:
    relative_time = getattr(state, "relative_time", None)
    step = getattr(state, "step", 0)
    n_steps = getattr(state, "n_steps", 1)
    return safe_round_rel(relative_time, step, n_steps)
