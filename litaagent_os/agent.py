from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Optional

from negmas import Contract, Outcome, SAOResponse, SAOState, ResponseType
from scml.oneshot import OneShotSyncAgent, QUANTITY, TIME, UNIT_PRICE

from .bou import BOUEstimator
from .breach import BreachInfoProvider, CapabilityProbe
from .config import LitaOSConfig
from .features import build_context_features, build_offer_features, compute_round_rel
from .history import HistoryStore, HistoryToken
from .models import LinearLogisticAcceptModel
from .partner_stats import PartnerStatsStore
from .utils import clamp, price_marginal_gain, match_price, round_bucket


@dataclass
class SentOfferInfo:
    offer: tuple[int, int, float]
    role: str
    round_bucket: int
    p_bin: int
    mu: float
    strength: float


class LitaAgentOS(OneShotSyncAgent):
    """LitaAgent-OS 主体实现。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.cfg = LitaOSConfig()
        self._load_agent_config()
        self._cap_probe = CapabilityProbe()
        self._cap_status = None
        self._breach_provider: Optional[BreachInfoProvider] = None
        self._accept_model = LinearLogisticAcceptModel(self.cfg)
        self._bou = BOUEstimator(self.cfg)
        self._history = HistoryStore(max_len=100)
        self._partner_stats = PartnerStatsStore(k=32)
        self._last_offer_sent: dict[str, SentOfferInfo] = {}
        self._last_price_sent: dict[str, float] = {}  # 用于 counter_monotonic
        self._awaiting_response: dict[str, bool] = {}
        self._neg_seen: dict[str, bool] = {}
        self._accepted_by_me: dict[str, bool] = {}
        self._ended_by_me: dict[str, bool] = {}

    # =====================
    # 生命周期回调
    # =====================

    def init(self) -> None:
        self._cap_status = self._cap_probe.probe(self.awi)
        self._breach_provider = BreachInfoProvider(self.awi, self._cap_status)
        self._load_accept_model()

    def _resolve_path(self, path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        return os.path.abspath(os.path.join(self._base_dir, path_str))

    def _load_agent_config(self) -> None:
        config_path = self._resolve_path(self.cfg.agent_config_path)
        self.cfg.load_overrides(Path(config_path))

    def _load_accept_model(self) -> None:
        model_dir = os.environ.get("LITA_ACCEPT_MODEL_DIR") or self.cfg.accept_model_dir
        model_dir = self._resolve_path(model_dir)
        meta_path = os.path.join(model_dir, "model_meta.json")
        model_path = os.path.join(model_dir, "model.bin")

        meta = {}
        if os.path.exists(meta_path):
            try:
                meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            except Exception as exc:
                self.awi.logwarning_agent(f"读取 model_meta.json 失败，将直接尝试加载模型权重: {exc}")
        model_type = str(meta.get("model_type", "")).lower() if isinstance(meta, dict) else ""
        if model_type and model_type != "logistic":
            self.awi.logwarning_agent("当前仅支持 logistic accept 模型，将使用默认权重")
            return

        loaded = False
        if os.path.exists(model_path):
            try:
                self._accept_model.load(Path(model_path))
                loaded = True
            except Exception as exc:
                self.awi.logwarning_agent(f"加载 accept 模型失败，将使用默认权重: {exc}")

        if not loaded:
            legacy_path = os.environ.get("LITA_OS_ACCEPT_MODEL_PATH")
            if legacy_path and os.path.exists(legacy_path):
                try:
                    self._accept_model.load(Path(legacy_path))
                    loaded = True
                except Exception as exc:
                    self.awi.logwarning_agent(f"加载旧路径 accept 模型失败，将使用默认权重: {exc}")

        if not loaded:
            self.awi.logwarning_agent("未找到可用 accept 模型，将使用默认权重")

    def before_step(self) -> None:
        self._neg_seen.clear()
        self._awaiting_response.clear()
        self._last_offer_sent.clear()
        self._last_price_sent.clear()
        self._accepted_by_me.clear()
        self._ended_by_me.clear()
        self._history.clear()

    def step(self) -> None:
        pass

    # =====================
    # 谈判回调
    # =====================

    def first_proposals(self) -> dict[str, Outcome | None]:
        responses: dict[str, Outcome | None] = {}
        partners = list(self.active_negotiators.keys())
        if not partners:
            return responses
        for role, role_partners in self._split_by_role(partners).items():
            if not role_partners:
                continue
            responses.update(
                self._propose_for_role(
                    role,
                    role_partners,
                    is_first=True,
                    round_rel_override=0.0,
                )
            )
        return responses

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        responses: dict[str, SAOResponse] = {}
        if not offers:
            return responses

        offers_now: dict[str, tuple[int, int, float]] = {}
        for pid, offer in offers.items():
            normalized = self._normalize_offer(offer)
            if normalized is None:
                continue
            role = "BUYER" if pid in self.awi.my_suppliers else "SELLER"
            issues = self._issues_for_role(role)
            t_min, t_max = self._time_bounds(issues)
            if normalized[TIME] < t_min or normalized[TIME] > t_max:
                continue
            offers_now[pid] = normalized

        for pid, offer in offers_now.items():
            state = states.get(pid)
            self._record_offer_received(pid, offer, state)
            # 对方用新 offer 作为回复，视为拒绝我方上一条 offer
            if self._awaiting_response.get(pid) and pid in self._last_offer_sent:
                self._update_bou_on_reject(pid, state)
                self._awaiting_response[pid] = False

        # 分角色做子集选择
        for role, role_partners in self._split_by_role(list(offers_now.keys())).items():
            if not role_partners:
                continue
            subset = self._select_subset(role, role_partners, offers_now, states)
            # --- 1.1: Effective q for END decision ---
            # 使用 q × fulfill_prob 作为有效接受量
            accepted_q_eff = 0.0
            for pid in subset:
                q = offers_now[pid][QUANTITY]
                fulfill = self._breach_provider.get_fulfill_prob(pid) if self._breach_provider else None
                fulfill = fulfill if fulfill is not None else 1.0
                accepted_q_eff += q * fulfill
            need_remaining_eff = max(0.0, float(self._need_remaining(role)) - accepted_q_eff)
            need_remaining = max(0, int(need_remaining_eff))
            remaining_partners = [pid for pid in role_partners if pid not in subset]
            for pid in role_partners:
                offer = offers_now[pid]
                if pid in subset:
                    responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    self._record_accept(pid, offer, states.get(pid), speaker="ME")
                    self._accepted_by_me[pid] = True
                    self._awaiting_response[pid] = False
            if not remaining_partners:
                continue
            # --- 1.2: Reachability constraint ---
            # 检查是否还有可能完成目标
            if need_remaining_eff <= 0:
                for pid in remaining_partners:
                    responses[pid] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                    self._record_end(pid, states.get(pid), speaker="ME")
                    self._ended_by_me[pid] = True
                    self._awaiting_response[pid] = False
                continue
            round_rel = self._round_rel_from_states(remaining_partners, states)
            # --- 修改点 2: 传递 opp_offers 给 _propose_for_role ---
            opp_offers_for_role = {pid: offers_now[pid] for pid in remaining_partners}
            counter_offers = self._propose_for_role(
                role,
                remaining_partners,
                is_first=False,
                need_override=need_remaining,
                round_rel_override=round_rel,
                states=states,
                opp_offers=opp_offers_for_role,
            )
            for pid in remaining_partners:
                counter_offer = counter_offers.get(pid)
                if counter_offer is None and need_remaining > 0:
                    # 避免返回 REJECT + None（会被视作 END），提供最小可用反报价
                    state = states.get(pid)
                    opp_offer = offers_now.get(pid)
                    counter_offer = self._counter_offer_for_partner_fallback(
                        role,
                        pid,
                        state,
                        need_override=need_remaining,
                        round_rel_override=round_rel,
                        opp_offer=opp_offer,
                    )
                    if counter_offer is not None:
                        issues = self._issues_for_role(role)
                        p_min, p_max = self._price_bounds(issues)
                        q_max = self._q_max(issues)
                        context = self._build_context(
                            pid,
                            role,
                            round_rel,
                            p_min,
                            p_max,
                            q_max,
                            need_remaining,
                        )
                        of = build_offer_features(counter_offer, q_max, p_min, p_max)
                        mu = self._accept_model.predict_accept_mu(context, of, [])
                        strength = self._accept_model.predict_accept_strength(context, of, [])
                        self._record_offer_sent(
                            pid,
                            counter_offer,
                            state,
                            is_first=False,
                            is_counter=True,
                            round_rel_override=round_rel,
                        )
                        self._last_offer_sent[pid] = self._pack_sent_offer(
                            pid, counter_offer, role, state, mu, strength, round_rel_override=round_rel
                        )
                        self._last_price_sent[pid] = counter_offer[UNIT_PRICE]
                        self._awaiting_response[pid] = True
                if counter_offer is None:
                    responses[pid] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                    self._record_end(pid, states.get(pid), speaker="ME")
                    self._ended_by_me[pid] = True
                    self._awaiting_response[pid] = False
                else:
                    responses[pid] = SAOResponse(ResponseType.REJECT_OFFER, counter_offer)
        return responses

    def on_negotiation_success(self, contract: Contract, mechanism) -> None:  # type: ignore
        pid = self._partner_from_contract(contract)
        if not pid:
            return
        agreement = self._normalize_offer(contract.agreement)
        if agreement is None:
            return
        if self._accepted_by_me.pop(pid, False):
            return
        self._record_accept(pid, agreement, None, speaker="OPP")
        # 更新对方接受我方报价
        last = self._last_offer_sent.get(pid)
        if last and self._awaiting_response.get(pid):
            if self._match_offer(last.offer, agreement):
                self._update_bou_on_accept(pid, last)
                self._awaiting_response[pid] = False

    def on_negotiation_failure(self, partners, annotation, mechanism, state) -> None:  # type: ignore
        for pid in partners:
            self._accepted_by_me.pop(pid, None)
            if self._ended_by_me.pop(pid, False):
                continue
            self._record_end(pid, state, speaker="OPP")
            last = self._last_offer_sent.get(pid)
            if last and self._awaiting_response.get(pid):
                if self.cfg.add_terminal_negative:
                    self._bou.update(
                        pid=pid,
                        role=last.role,
                        round_bucket=last.round_bucket,
                        p_bin=last.p_bin,
                        q=last.offer[QUANTITY],
                        mu=last.mu,
                        strength=last.strength,
                        accepted=False,
                        terminal_negative=True,
                    )
                self._awaiting_response[pid] = False

    # =====================
    # 内部工具
    # =====================

    def _split_by_role(self, partners: list[str]) -> dict[str, list[str]]:
        buyers, sellers = [], []
        for pid in partners:
            if pid in self.awi.my_suppliers:
                buyers.append(pid)
            elif pid in self.awi.my_consumers:
                sellers.append(pid)
        return {"BUYER": buyers, "SELLER": sellers}

    def _issues_for_role(self, role: str):
        return self.awi.current_input_issues if role == "BUYER" else self.awi.current_output_issues

    def _price_bounds(self, issues) -> tuple[float, float]:
        p_issue = issues[UNIT_PRICE]
        return float(p_issue.min_value), float(p_issue.max_value)

    def _q_max(self, issues) -> int:
        q_issue = issues[QUANTITY]
        return int(q_issue.max_value) if q_issue is not None else int(self.awi.n_lines)

    def _time_bounds(self, issues) -> tuple[int, int]:
        try:
            t_issue = issues[TIME]
            t_min = int(t_issue.min_value)
            t_max = int(t_issue.max_value)
        except Exception:
            step = int(getattr(self.awi, "current_step", 0))
            return step, step
        if t_min > t_max:
            t_min, t_max = t_max, t_min
        return t_min, t_max

    def _select_time(self, issues) -> int:
        t_min, t_max = self._time_bounds(issues)
        if t_min == t_max:
            return t_min
        step = int(getattr(self.awi, "current_step", 0))
        return max(t_min, min(step, t_max))

    def _round_rel_from_state(self, state: Optional[SAOState]) -> float:
        if state is None:
            rel = getattr(self.awi, "relative_time", None)
            step = self.awi.current_step
            n_steps = self.awi.n_steps
            return clamp(rel if rel is not None else step / max(1, n_steps - 1), 0.0, 1.0)
        return compute_round_rel(self.cfg, state)

    def _round_rel_from_states(self, partners: list[str], states: dict[str, SAOState]) -> float:
        rels = []
        for pid in partners:
            st = states.get(pid)
            if st is not None:
                rels.append(compute_round_rel(self.cfg, st))
        if not rels:
            return 0.0
        return clamp(min(rels), 0.0, 1.0)

    def _price_for_role(self, role: str, p_min: float, p_max: float, round_rel: float) -> float:
        if p_max <= p_min:
            return p_min
        concession = clamp(round_rel, 0.0, 1.0) ** self.cfg.price_concession_gamma
        if role == "BUYER":
            price = p_min + concession * (p_max - p_min)
        else:
            price = p_max - concession * (p_max - p_min)
        return clamp(price, p_min, p_max)

    def _trading_price(self, role: str) -> Optional[float]:
        if not self.cfg.use_trading_price:
            return None
        prices = getattr(self.awi, "trading_prices", None)
        if prices is None:
            return None
        product = self.awi.my_output_product if role == "SELLER" else self.awi.my_input_product
        try:
            return float(prices[product])
        except Exception:
            return None

    def _penalty_scale(self, is_input: bool, unit_price: Optional[float]) -> float:
        try:
            return float(self.awi.penalty_multiplier(is_input, unit_price))
        except Exception:
            if unit_price is not None:
                return float(unit_price)
            return 1.0

    def _need_remaining(self, role: str) -> int:
        return int(self.awi.needed_sales if role == "SELLER" else self.awi.needed_supplies)

    def _buffer(self, round_rel: float) -> float:
        span = self.cfg.buffer_max - self.cfg.buffer_min
        return self.cfg.buffer_min + span * (round_rel ** self.cfg.buffer_exp)

    def _build_context(
        self,
        pid: str,
        role: str,
        round_rel: float,
        p_min: float,
        p_max: float,
        q_max: int,
        need_remaining: int,
    ):
        stats = self._partner_stats.get(pid, role).snapshot()
        system_breach_prob = self._breach_provider.get_breach_prob(pid) if self._breach_provider else None
        system_breach_level = self._breach_provider.get_breach_level(pid) if self._breach_provider else None
        return build_context_features(
            self.cfg,
            role,
            round_rel,
            p_min,
            p_max,
            need_remaining,
            q_max,
            self._trading_price(role),
            self.awi.current_shortfall_penalty,
            self.awi.current_disposal_cost,
            system_breach_prob,
            system_breach_level,
            stats,
        )

    def _pack_sent_offer(
        self,
        pid: str,
        offer: tuple[int, int, float],
        role: str,
        state: Optional[SAOState],
        mu: float,
        strength: float,
        round_rel_override: Optional[float] = None,
    ) -> SentOfferInfo:
        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        of = build_offer_features(offer, q_max, p_min, p_max)
        round_rel = round_rel_override if round_rel_override is not None else self._round_rel_from_state(state)
        rb = round_bucket(round_rel, self.cfg.round_bucket_T)
        return SentOfferInfo(
            offer=offer,
            role=role,
            round_bucket=rb,
            p_bin=of.p_bin,
            mu=mu,
            strength=strength,
        )

    def _match_offer(self, a: tuple[int, int, float], b: tuple[int, int, float]) -> bool:
        return a[QUANTITY] == b[QUANTITY] and a[TIME] == b[TIME] and match_price(a[UNIT_PRICE], b[UNIT_PRICE])

    def _propose_for_role(
        self,
        role: str,
        partners: list[str],
        is_first: bool,
        need_override: Optional[int] = None,
        round_rel_override: Optional[float] = None,
        states: Optional[dict[str, SAOState]] = None,
        opp_offers: Optional[dict[str, tuple[int, int, float]]] = None,
    ) -> dict[str, Outcome | None]:
        """
        为指定角色的伙伴生成报价。
        修改点:
        - 0.1: Feasibility-first probe disable
        - 0.2: Probe q allocation (ceil(target / n_partners))
        - Q: q consistency fix (使用实际 q 计算 p_eff)
        - 2: Counter price anchoring (使用 opp_offers)
        - 3: Panic mode
        """
        import math

        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        round_rel = round_rel_override if round_rel_override is not None else self._round_rel_from_state(None)
        base_price = self._price_for_role(role, p_min, p_max, round_rel)
        trading_price = self._trading_price(role)
        need_remaining = need_override if need_override is not None else self._need_remaining(role)
        if need_remaining <= 0 or q_max <= 0:
            return {pid: None for pid in partners}

        t_value = self._select_time(issues)
        n_partners = max(1, len(partners))
        target = int(need_remaining * (1.0 + self._buffer(round_rel)))

        # --- 0.1: Feasibility-first probe disable ---
        # 计算 probe 模式下的最大有效数量
        probe_q_per_partner = max(self.cfg.probe_q_min, self.cfg.probe_q)
        eff_probe_max = n_partners * probe_q_per_partner
        # 若 probe 模式无法满足需求，则禁用 probe
        probe_disabled = eff_probe_max < need_remaining

        # --- 3: Panic mode check ---
        panic_active = self._should_panic(role, need_remaining, round_rel)

        candidate_q = min(q_max, max(1, min(self.cfg.q_candidate, need_remaining)))

        # 收集每个 partner 的初步信息 (用于排序)
        partner_info: list[tuple[str, Any, float, float, float]] = []  # (pid, context, p_eff_cand, score, fulfill)
        for pid in partners:
            context = self._build_context(pid, role, round_rel, p_min, p_max, q_max, need_remaining)
            offer_cand = build_offer_features((candidate_q, t_value, base_price), q_max, p_min, p_max)
            mu = self._accept_model.predict_accept_mu(context, offer_cand, [])
            strength = self._accept_model.predict_accept_strength(context, offer_cand, [])
            lcb_sign = self._bou.lcb(
                pid=pid,
                role=role,
                round_bucket=context.round_bucket,
                p_bin=offer_cand.p_bin,
                q=offer_cand.q,
                mu=mu,
                strength=strength,
                delta=self.cfg.lcb_delta_accept,
            )
            lcb_sign = lcb_sign if lcb_sign is not None else mu
            # 系统 fulfill 概率直接使用，不做 LCB
            fulfill = self._breach_provider.get_fulfill_prob(pid) if self._breach_provider else None
            fulfill = fulfill if fulfill is not None else 1.0
            p_eff = lcb_sign * fulfill
            score = price_marginal_gain(base_price, trading_price, role == "SELLER") * p_eff
            partner_info.append((pid, context, p_eff, score, fulfill))

        partner_info.sort(key=lambda x: x[3], reverse=True)
        remaining_eff = float(target)
        offers: dict[str, Outcome | None] = {}

        for pid, context, p_eff_cand, _, fulfill in partner_info:
            if remaining_eff <= 0:
                offers[pid] = None
                continue

            # 计算期望 q
            q = min(q_max, max(1, int(round(remaining_eff / max(p_eff_cand, 1e-6)))))

            # --- 0.2: Probe q allocation ---
            # 若处于 probe 模式且未被禁用，限制 q
            in_probe_mode = (
                not probe_disabled
                and not panic_active
                and (self.awi.current_step < self.cfg.probe_steps or p_eff_cand < self.cfg.probe_lcb_threshold)
            )
            if in_probe_mode:
                # 使用 ceil(target / n_partners) 作为 probe q 上限
                probe_q_limit = max(self.cfg.probe_q_min, math.ceil(target / n_partners))
                q = min(q, probe_q_limit)

            # Panic 模式下直接用较大 q
            if panic_active:
                q = min(q_max, max(1, need_remaining))

            # --- Q: q consistency fix ---
            # 用实际 q 重新计算 p_eff
            # 获取用于该 partner 的实际价格 (考虑 anchoring)
            price = self._counter_price_for_partner(role, pid, base_price, p_min, p_max, opp_offers)

            offer = (q, t_value, price)
            of = build_offer_features(offer, q_max, p_min, p_max)
            mu = self._accept_model.predict_accept_mu(context, of, [])
            strength = self._accept_model.predict_accept_strength(context, of, [])
            lcb_sign = self._bou.lcb(
                pid=pid,
                role=role,
                round_bucket=context.round_bucket,
                p_bin=of.p_bin,
                q=of.q,
                mu=mu,
                strength=strength,
                delta=self.cfg.lcb_delta_accept,
            )
            lcb_sign = lcb_sign if lcb_sign is not None else mu
            # 用实际 q 的 p_eff
            p_eff_actual = lcb_sign * fulfill
            offers[pid] = offer
            remaining_eff -= of.q * p_eff_actual

        for pid in partners:
            if pid not in offers:
                offers[pid] = None

        # 记录发送的 offer
        for pid, offer in offers.items():
            if offer is None:
                continue
            state = states.get(pid) if states else None
            self._record_offer_sent(
                pid,
                offer,
                state,
                is_first=is_first,
                is_counter=not is_first,
                round_rel_override=round_rel,
            )
            context = self._build_context(pid, role, round_rel, p_min, p_max, q_max, need_remaining)
            of = build_offer_features(offer, q_max, p_min, p_max)
            mu = self._accept_model.predict_accept_mu(context, of, [])
            strength = self._accept_model.predict_accept_strength(context, of, [])
            self._last_offer_sent[pid] = self._pack_sent_offer(
                pid, offer, role, state, mu, strength, round_rel_override=round_rel
            )
            self._last_price_sent[pid] = offer[UNIT_PRICE]
            self._awaiting_response[pid] = True
        return offers

    def _counter_price_for_partner(
        self,
        role: str,
        pid: str,
        base_price: float,
        p_min: float,
        p_max: float,
        opp_offers: Optional[dict[str, tuple[int, int, float]]],
    ) -> float:
        """
        计算针对特定 partner 的反报价价格。
        修改点 2: Counter price anchoring
        p_counter = (1 - η) × base_price + η × p_opp
        其中 η = round_rel^k (k = counter_anchor_eta_exp)
        """
        if opp_offers is None or pid not in opp_offers:
            return base_price
        opp_offer = opp_offers[pid]
        opp_price = opp_offer[UNIT_PRICE]

        # 计算 round_rel (这里简化处理，使用全局 round_rel)
        round_rel = self._round_rel_from_state(None)
        eta = round_rel ** self.cfg.counter_anchor_eta_exp
        price = (1.0 - eta) * base_price + eta * opp_price

        # Monotonic 约束: BUYER 价格只能升，SELLER 价格只能降
        if self.cfg.counter_monotonic and pid in self._last_price_sent:
            last_price = self._last_price_sent[pid]
            if role == "BUYER":
                price = max(price, last_price)
            else:  # SELLER
                price = min(price, last_price)

        return clamp(price, p_min, p_max)

    def _should_panic(self, role: str, need_remaining: int, round_rel: float) -> bool:
        """
        判断是否应进入 panic 模式。
        修改点 3: Panic mode (区分 buyer/seller)
        触发条件: R = penalty/cost > threshold AND round_rel > threshold AND remaining > 0
        """
        if not self.cfg.panic_enabled:
            return False
        if need_remaining <= 0:
            return False
        if round_rel < self.cfg.panic_round_rel_threshold:
            return False

        # 获取惩罚/成本
        spot_in = self._trading_price("BUYER")
        spot_out = self._trading_price("SELLER")
        shortfall_penalty = (self.awi.current_shortfall_penalty or 0.0) * self._penalty_scale(False, spot_out)
        disposal_cost = (self.awi.current_disposal_cost or 0.0) * self._penalty_scale(True, spot_in)

        if role == "BUYER":
            # 买方关心 shortfall penalty
            penalty = shortfall_penalty
            cost_ref = disposal_cost if disposal_cost > 0 else 1.0
        else:
            # 卖方关心 disposal cost
            penalty = disposal_cost
            cost_ref = shortfall_penalty if shortfall_penalty > 0 else 1.0

        ratio = penalty / max(cost_ref, 1e-6)
        return ratio > self.cfg.panic_penalty_ratio_threshold

    def _counter_offer_for_partner_fallback(
        self,
        role: str,
        pid: str,
        state: Optional[SAOState],
        need_override: Optional[int] = None,
        round_rel_override: Optional[float] = None,
        opp_offer: Optional[tuple[int, int, float]] = None,
    ) -> Optional[tuple[int, int, float]]:
        """
        生成 fallback 反报价（当 _propose_for_role 返回 None 时使用）。
        重命名自 _counter_offer_for_partner，添加 opp_offer 支持。
        """
        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        round_rel = round_rel_override if round_rel_override is not None else self._round_rel_from_state(state)
        base_price = self._price_for_role(role, p_min, p_max, round_rel)
        need_remaining = need_override if need_override is not None else self._need_remaining(role)
        if need_remaining <= 0:
            return None

        # 使用 opp_offer 做价格锚定
        opp_offers_dict = {pid: opp_offer} if opp_offer is not None else None
        price = self._counter_price_for_partner(role, pid, base_price, p_min, p_max, opp_offers_dict)

        q = min(q_max, max(1, min(need_remaining, q_max)))
        t_value = self._select_time(issues)
        return (q, t_value, price)

    def _select_subset(self, role: str, partners: list[str], offers: dict[str, Outcome], states: dict[str, SAOState]) -> set[str]:
        need_remaining = self._need_remaining(role)
        if need_remaining <= 0:
            return set()
        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        round_rel = self._round_rel_from_states(partners, states)
        trading_price = self._trading_price(role)
        spot_in = self._trading_price("BUYER")
        spot_out = self._trading_price("SELLER")
        shortfall_unit = (self.awi.current_shortfall_penalty or 0.0) * self._penalty_scale(False, spot_out)
        disposal_unit = (self.awi.current_disposal_cost or 0.0) * self._penalty_scale(True, spot_in)
        penalty_unit = shortfall_unit if role == "BUYER" else disposal_unit
        partner_scores = []
        for pid in partners:
            offer = offers[pid]
            stats = self._partner_stats.get(pid, role).snapshot()
            system_breach_prob = self._breach_provider.get_breach_prob(pid) if self._breach_provider else None
            system_breach_level = self._breach_provider.get_breach_level(pid) if self._breach_provider else None
            context = build_context_features(
                self.cfg,
                role,
                round_rel,
                p_min,
                p_max,
                need_remaining,
                q_max,
                trading_price,
                self.awi.current_shortfall_penalty,
                self.awi.current_disposal_cost,
                system_breach_prob,
                system_breach_level,
                stats,
            )
            of = build_offer_features(offer, q_max, p_min, p_max)
            fulfill = self._breach_provider.get_fulfill_prob(pid) if self._breach_provider else None
            p_eff = fulfill if fulfill is not None else 1.0
            utility = price_marginal_gain(of.p, trading_price, role == "SELLER") * of.q
            partner_scores.append((pid, of, context, p_eff, utility))

        partner_scores.sort(key=lambda x: x[4], reverse=True)
        top = partner_scores[: self.cfg.portfolio_k]

        best_score = float("-inf")
        best_set: set[str] = set()
        n = len(top)
        for mask in range(1 << n):
            total_q_eff = 0.0
            utility = 0.0
            risk_penalty = 0.0
            subset: set[str] = set()
            for i in range(n):
                if mask & (1 << i):
                    pid, of, _, p_eff, util = top[i]
                    subset.add(pid)
                    total_q_eff += of.q * p_eff
                    utility += util
                    if self.cfg.risk_lambda > 0:
                        risk_penalty += self.cfg.risk_lambda * of.q * (1.0 - p_eff)
            shortfall = max(0.0, need_remaining - total_q_eff)
            penalty_cost = (penalty_unit or 0.0) * shortfall
            overfill = max(0.0, total_q_eff - need_remaining)
            over_ratio = clamp(self.cfg.overfill_penalty_ratio, 0.0, 0.1)
            over_penalty = (penalty_unit or 0.0) * over_ratio * overfill
            score = utility - penalty_cost - over_penalty - risk_penalty
            if score > best_score:
                best_score = score
                best_set = subset
        return best_set

    def _record_offer_received(self, pid: str, offer: Outcome, state: Optional[SAOState]) -> None:
        offer_norm = self._normalize_offer(offer)
        if offer_norm is None:
            return
        role = "BUYER" if pid in self.awi.my_suppliers else "SELLER"
        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        of = build_offer_features(offer_norm, q_max, p_min, p_max)
        rb = round_bucket(self._round_rel_from_state(state), self.cfg.round_bucket_T)
        is_first = not self._neg_seen.get(pid, False)
        token = HistoryToken(
            speaker="OPP",
            action_type="OFFER",
            q_bucket=of.q_bucket,
            p_bin=of.p_bin,
            round_bucket=rb,
            is_counter=not is_first,
            is_first_proposal=is_first,
        )
        self._history.append(pid, self._negotiation_id(pid), role, token)
        self._neg_seen[pid] = True

    def _record_offer_sent(
        self,
        pid: str,
        offer: Outcome,
        state: Optional[SAOState],
        is_first: bool,
        is_counter: bool,
        round_rel_override: Optional[float] = None,
    ) -> None:
        offer_norm = self._normalize_offer(offer)
        if offer_norm is None:
            return
        role = "BUYER" if pid in self.awi.my_suppliers else "SELLER"
        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        of = build_offer_features(offer_norm, q_max, p_min, p_max)
        round_rel = round_rel_override if round_rel_override is not None else self._round_rel_from_state(state)
        rb = round_bucket(round_rel, self.cfg.round_bucket_T)
        token = HistoryToken(
            speaker="ME",
            action_type="OFFER",
            q_bucket=of.q_bucket,
            p_bin=of.p_bin,
            round_bucket=rb,
            is_counter=is_counter,
            is_first_proposal=is_first,
        )
        self._history.append(pid, self._negotiation_id(pid), role, token)

    def _record_accept(self, pid: str, offer: Outcome, state: Optional[SAOState], speaker: str) -> None:
        offer_norm = self._normalize_offer(offer)
        if offer_norm is None:
            return
        role = "BUYER" if pid in self.awi.my_suppliers else "SELLER"
        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        of = build_offer_features(offer_norm, q_max, p_min, p_max)
        rb = round_bucket(self._round_rel_from_state(state), self.cfg.round_bucket_T)
        token = HistoryToken(
            speaker=speaker,
            action_type="ACCEPT",
            q_bucket=of.q_bucket,
            p_bin=of.p_bin,
            round_bucket=rb,
        )
        self._history.append(pid, self._negotiation_id(pid), role, token)
        if speaker == "OPP":
            self._partner_stats.get(pid, role).record_response(accepted=True)

    def _record_end(self, pid: str, state: Optional[SAOState], speaker: str) -> None:
        role = "BUYER" if pid in self.awi.my_suppliers else "SELLER"
        rb = round_bucket(self._round_rel_from_state(state), self.cfg.round_bucket_T)
        token = HistoryToken(
            speaker=speaker,
            action_type="END",
            q_bucket=0,
            p_bin=0,
            round_bucket=rb,
        )
        self._history.append(pid, self._negotiation_id(pid), role, token)

    def _update_bou_on_reject(self, pid: str, state: Optional[SAOState]) -> None:
        last = self._last_offer_sent.get(pid)
        if not last:
            return
        role = last.role
        of_q = last.offer[QUANTITY]
        self._bou.update(
            pid=pid,
            role=role,
            round_bucket=last.round_bucket,
            p_bin=last.p_bin,
            q=of_q,
            mu=last.mu,
            strength=last.strength,
            accepted=False,
            terminal_negative=False,
        )
        self._partner_stats.get(pid, role).record_response(accepted=False)

    def _update_bou_on_accept(self, pid: str, last: SentOfferInfo) -> None:
        role = last.role
        self._bou.update(
            pid=pid,
            role=role,
            round_bucket=last.round_bucket,
            p_bin=last.p_bin,
            q=last.offer[QUANTITY],
            mu=last.mu,
            strength=last.strength,
            accepted=True,
            terminal_negative=False,
        )
        self._partner_stats.get(pid, role).record_response(accepted=True)

    def _negotiation_id(self, pid: str) -> str:
        details = self.negotiators.get(pid)
        if details and getattr(details, "nmi", None) is not None:
            nmi = details.nmi
            return str(getattr(nmi, "id", getattr(nmi, "mechanism_id", pid)))
        return pid

    def _partner_from_contract(self, contract: Contract) -> Optional[str]:
        partners = list(getattr(contract, "partners", ()) or ())
        if not partners:
            return None
        me = getattr(self, "id", None)
        if me is None and self.awi is not None:
            me = getattr(self.awi, "id", None)
        for pid in partners:
            if me is None or pid != me:
                return pid
        return partners[0]

    def _normalize_offer(self, offer: Any) -> Optional[tuple[int, int, float]]:
        if offer is None:
            return None
        if isinstance(offer, tuple) or isinstance(offer, list):
            if len(offer) <= UNIT_PRICE:
                return None
            try:
                return (int(offer[QUANTITY]), int(offer[TIME]), float(offer[UNIT_PRICE]))
            except Exception:
                return None
        if isinstance(offer, dict):
            if QUANTITY in offer and TIME in offer and UNIT_PRICE in offer:
                try:
                    return (int(offer[QUANTITY]), int(offer[TIME]), float(offer[UNIT_PRICE]))
                except Exception:
                    return None
            q = self._value_from_mapping(offer, ("quantity", "q"))
            t = self._value_from_mapping(offer, ("time", "t"))
            p = self._value_from_mapping(offer, ("unit_price", "price", "uprice", "p"))
            if q is None or t is None or p is None:
                return None
            try:
                return (int(q), int(t), float(p))
            except Exception:
                return None
        return None

    def _value_from_mapping(self, data: dict, names: tuple[str, ...]) -> Optional[Any]:
        for key in names:
            if key in data:
                return data[key]
        for key, value in data.items():
            key_name = None
            if isinstance(key, str):
                key_name = key
            elif hasattr(key, "name"):
                key_name = getattr(key, "name", None)
            if key_name:
                key_name = str(key_name).lower()
                if key_name in names:
                    return value
        return None
