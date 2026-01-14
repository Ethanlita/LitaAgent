from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
from .utils import clamp, price_marginal_gain, compute_true_profit, match_price, round_bucket


@dataclass
class SentOfferInfo:
    """2026-01-12: 增加 lcb 字段用于 Exposure Book 计算"""
    offer: tuple[int, int, float]
    role: str
    round_bucket: int
    p_bin: int
    mu: float
    strength: float
    lcb: float = 0.0  # P_sign LCB 用于 pending_expected 计算


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
        
        # =========================================================================
        # 方案 A: 全局 offer budget (2026-01-12 Reviewer 建议)
        # =========================================================================
        # 每个 step/day 的 offer budget 只计算一次，避免多波次发单累积 exposure
        # _offer_budget_total: 当天的总预算
        # _offer_budget_used: 当天已使用的预算
        self._offer_budget_total: dict[str, float] = {"BUYER": 0.0, "SELLER": float("inf")}
        self._offer_budget_used: dict[str, float] = {"BUYER": 0.0, "SELLER": 0.0}
        self._offer_budget_initialized: dict[str, bool] = {"BUYER": False, "SELLER": False}
        
        # =========================================================================
        # P0: Committed Book (2026-01-12 Reviewer 建议)
        # =========================================================================
        # 跟踪当天已签约的数量（不可撤回）
        # 用于闭环数量控制：need_live = need_init - committed - pending_expected
        self._committed_qty: dict[str, float] = {"BUYER": 0.0, "SELLER": 0.0}

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
        
        # =========================================================================
        # 方案 A: 初始化当天的 BUYER offer budget (2026-01-12)
        # =========================================================================
        # 在 first_proposals 时计算一次，后续 counter_all 共享
        # 
        # 2026-01-13 P0-2: 收紧 budget
        # 旧公式: budget = ceil(need*1.2) + 2 → 允许 overfill 最多 +4
        # 新公式: budget = target_nominal + 1 → 允许 overfill 最多 +1
        import math
        buyer_need = self._need_remaining("BUYER")
        if buyer_need > 0 and self.cfg.buyer_offer_budget_enabled:
            if self.cfg.buyer_offer_budget_use_target_plus_one:
                # P0-2 新公式: budget = target + 1
                # 计算 target (需要考虑 conditional overordering)
                should_overorder = False
                if self.cfg.conditional_overordering_enabled:
                    issues = self._issues_for_role("BUYER")
                    p_min, _ = self._price_bounds(issues)
                    spot_in = self._trading_price("BUYER")
                    spot_out = self._trading_price("SELLER")
                    shortfall_unit = (self.awi.current_shortfall_penalty or 0.0) * self._penalty_scale(False, spot_out)
                    disposal_unit = (self.awi.current_disposal_cost or 0.0) * self._penalty_scale(True, spot_in)
                    buy_price_est = spot_in if spot_in else p_min
                    threshold = buy_price_est + disposal_unit + disposal_unit * self.cfg.conditional_overordering_margin
                    should_overorder = shortfall_unit > threshold
                else:
                    should_overorder = True
                
                if should_overorder and self.cfg.overordering_ensure_plus_one:
                    target_nominal = buyer_need + max(1, math.ceil(buyer_need * self.cfg.buyer_overordering_ratio))
                elif should_overorder:
                    target_nominal = int(buyer_need * (1.0 + self.cfg.buyer_overordering_ratio))
                else:
                    target_nominal = buyer_need
                
                # P0-1 (2026-01-14): 放宽 offer_budget
                # 问题: target+1 太紧，导致 27% 的谈判没收到 offer，shortfall 上升
                # 解决: 改为 ceil(target * budget_mult) + 1
                # 例如 target=8 → budget = ceil(8*1.25)+1 = 11
                self._offer_budget_total["BUYER"] = math.ceil(target_nominal * self.cfg.buyer_offer_budget_mult_v2) + 1
            else:
                # 旧公式
                self._offer_budget_total["BUYER"] = math.ceil(buyer_need * self.cfg.buyer_offer_budget_mult) + self.cfg.buyer_offer_budget_abs
        else:
            self._offer_budget_total["BUYER"] = float("inf")
        self._offer_budget_used["BUYER"] = 0.0
        self._offer_budget_initialized["BUYER"] = True
        
        # =========================================================================
        # BUYER First Proposal Penalty-Aware (2026-01-10)
        # =========================================================================
        # 如果 shortfall_penalty 明显大于 disposal_cost，BUYER 的 first_proposal
        # 应该使用更高的价格，提高竞争力
        # 
        # 问题背景:
        #   - BUYER 默认出 p_min (最低价)
        #   - 但 SELLER 同时收到多个 BUYER 的报价
        #   - SELLER 会优先接受高价 BUYER
        #   - LOS 作为 BUYER 只有 46% 能收到 SELLER 回复
        # =========================================================================
        buyer_fp_use_pmax = False
        if self.cfg.buyer_fp_penalty_aware_enabled:
            spot_in = self._trading_price("BUYER")
            spot_out = self._trading_price("SELLER")
            shortfall_unit = (self.awi.current_shortfall_penalty or 0.0) * self._penalty_scale(False, spot_out)
            disposal_unit = (self.awi.current_disposal_cost or 0.0) * self._penalty_scale(True, spot_in)
            if disposal_unit > 0 and shortfall_unit / disposal_unit > self.cfg.buyer_fp_penalty_aware_threshold:
                buyer_fp_use_pmax = True
        
        for role, role_partners in self._split_by_role(partners).items():
            if not role_partners:
                continue
            
            # 2026-01-11 P0-修复1: 不再用 round_rel=1.0 偷价格
            # 改用 force_price 显式传递价格，round_rel 保持 0.0
            # 这样不会意外触发 panic 模式
            force_price = None
            if role == "BUYER" and buyer_fp_use_pmax:
                # 显式使用 p_max，但 round_rel 仍然是 0.0
                issues = self._issues_for_role(role)
                _, p_max = self._price_bounds(issues)
                force_price = float(p_max)
            
            responses.update(
                self._propose_for_role(
                    role,
                    role_partners,
                    is_first=True,
                    round_rel_override=0.0,  # 始终用 0.0，不触发 panic
                    force_price=force_price,
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
            # 2026-01-12: 禁用 breach 概率时 fulfill = 1
            accepted_q_eff = 0.0
            for pid in subset:
                q = offers_now[pid][QUANTITY]
                if self.cfg.disable_breach_probability:
                    fulfill = 1.0
                else:
                    fulfill = self._breach_provider.get_fulfill_prob(pid) if self._breach_provider else None
                    fulfill = fulfill if fulfill is not None else 1.0
                accepted_q_eff += q * fulfill
            
            # =========================================================================
            # P2: Unify need 口径 (2026-01-13 Reviewer 建议)
            # =========================================================================
            # 问题: counter_all 的 need_remaining 没有扣除 committed 和 pending
            #       导致给 _propose_for_role 传的 need 偏大，仍然会超发 offer
            # 
            # 解决方案:
            #   need_remaining_raw = awi.needed_xxx - accepted_q_eff (刚接受的)
            #   need_live = need_remaining_raw - committed (之前已签约的)
            #   need_remaining = need_live - pending_expected (还在等回复的)
            # =========================================================================
            need_remaining_raw = max(0.0, float(self._need_remaining(role)) - accepted_q_eff)
            
            # 扣除已签约的 (committed book)
            committed = self._get_committed_qty(role)
            need_live = max(0.0, need_remaining_raw - committed)
            
            # 扣除 pending exposure
            pending_worst, pending_expected = self._get_pending_exposure(role)
            if role == "BUYER":
                need_adj = max(0.0, need_live - pending_expected)
            else:  # SELLER 更保守
                need_adj = max(0.0, need_live - pending_worst)
            
            # P0-修复2: 使用 ceil 而非 int，避免截断导致的结构性 shortfall
            # 例如: need_remaining_eff=0.8 → int=0 导致 END，ceil=1 继续谈判
            need_remaining = max(0, math.ceil(need_adj - 1e-9))  # -eps 避免浮点误差
            remaining_partners = [pid for pid in role_partners if pid not in subset]
            for pid in role_partners:
                offer = offers_now[pid]
                if pid in subset:
                    responses[pid] = SAOResponse(ResponseType.ACCEPT_OFFER, offer)
                    self._record_accept(pid, offer, states.get(pid), speaker="ME")
                    self._accepted_by_me[pid] = True
                    self._awaiting_response[pid] = False
                    # P0: Committed Book - 立即更新已签约数量
                    if hasattr(self, '_committed_qty'):
                        self._committed_qty[role] = self._committed_qty.get(role, 0.0) + offer[QUANTITY]
            if not remaining_partners:
                continue
            # --- 1.2: Reachability constraint ---
            # P0-2 (2026-01-14): 不直接 END，改为继续发 counter
            # 问题: need_adj<=0 时直接 END 会丢失成交机会，导致 shortfall
            # 解决: 即使 need_adj<=0，也继续发送最小 q 的 counter offer
            #       只有当 committed + pending_worst >= need + slack 时才 END
            use_end_threshold_worst = self.cfg.use_end_threshold_worst
            if use_end_threshold_worst:
                # 使用更保守的 pending_worst 判断是否 END
                end_slack = self.cfg.end_threshold_slack
                should_end = (committed + pending_worst >= need_remaining_raw + end_slack)
            else:
                # 旧逻辑: need_adj <= 0 就 END
                should_end = (need_adj <= 0)
            
            if should_end:
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
                        # 2026-01-12: 计算并存储 LCB 用于 Exposure Book
                        lcb = self._bou.lcb(
                            pid=pid,
                            role=role,
                            round_bucket=context.round_bucket,
                            p_bin=of.p_bin,
                            q=of.q,
                            mu=mu,
                            strength=strength,
                            delta=self.cfg.lcb_delta_accept,
                        )
                        lcb = lcb if lcb is not None else mu
                        self._record_offer_sent(
                            pid,
                            counter_offer,
                            state,
                            is_first=False,
                            is_counter=True,
                            round_rel_override=round_rel,
                        )
                        self._last_offer_sent[pid] = self._pack_sent_offer(
                            pid, counter_offer, role, state, mu, strength, round_rel_override=round_rel, lcb=lcb
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
        
        # P0: Committed Book - 更新已签约数量
        # 判断角色：pid 在 my_suppliers 里说明我们是 BUYER
        role = "BUYER" if pid in self.awi.my_suppliers else "SELLER"
        q = agreement[QUANTITY]
        
        if self._accepted_by_me.pop(pid, False):
            # 我方接受对方 offer，已经在 counter_all 中更新了 committed_qty
            return
        
        # 对方接受我方 offer，需要更新 committed_qty
        if hasattr(self, '_committed_qty'):
            self._committed_qty[role] = self._committed_qty.get(role, 0.0) + q
        
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

    def _snap_price(self, price: float, p_min: float, p_max: float) -> float:
        """
        将价格 snap 到整数（如果价格边界是整数）。
        
        2026-01-10 修复：使用标准四舍五入（0.5 向上取整）
        
        问题背景：
          Python 的 round() 使用银行家舍入（round half to even）：
            round(16.5) = 16（不是 17，因为 16 是偶数）
          
          这在 price_range=1 的窄价差市场中会导致严重问题：
            p_min=16, p_max=17 → p_mid=16.5
            panic/让步想提价到 16.5，但被 round 回 16
            => 对手看到的仍是 p_min，让步/panic 完全失效
        
        解决方案：
          使用 math.floor(price + 0.5) 实现标准四舍五入
          16.5 → floor(17.0) = 17 ✓
        """
        import math
        price = clamp(price, p_min, p_max)
        if abs(p_min - round(p_min)) < 1e-6 and abs(p_max - round(p_max)) < 1e-6:
            # 标准四舍五入：0.5 向上取整
            snapped = int(math.floor(price + 0.5))
            snapped = max(int(round(p_min)), min(int(round(p_max)), snapped))
            return float(snapped)
        return price

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
        """
        根据角色和谈判进度计算报价。
        
        让步公式: concession = round_rel^gamma
        - BUYER: price = p_min + concession * (p_max - p_min)  # 从低价向高价让步
        - SELLER: price = p_max - concession * (p_max - p_min)  # 从高价向低价让步
        
        设计决策 (2026-01-10): gamma=0.5 而非 2.0
        原因：99.4% 的 OneShot 谈判在 round 1 结束，平均 round_rel ≈ 0.03
        - gamma=2.0 时: 0.03^2 = 0.0009 (几乎无让步)
        - gamma=0.5 时: 0.03^0.5 ≈ 0.17 (有效让步)
        详见 open_questions.md 第 7 条
        """
        if p_max <= p_min:
            return self._snap_price(p_min, p_min, p_max)
        concession = clamp(round_rel, 0.0, 1.0) ** self.cfg.price_concession_gamma
        if role == "BUYER":
            price = p_min + concession * (p_max - p_min)
        else:
            price = p_max - concession * (p_max - p_min)
        return self._snap_price(price, p_min, p_max)

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

    def _exogenous_price(self, role: str) -> Optional[float]:
        """
        获取外生合同的 **单价**（真实成本/收入基准）。
        
        注意：SCML OneShot 的 current_exogenous_*_price 是 **总价格**，不是单价！
        需要除以数量才能得到单价。
        
        OneShot 中:
        - 卖家 (Level 0): 有外生输入（采购原材料）
          单价 = current_exogenous_input_price / current_exogenous_input_quantity
        - 买家 (Level 1): 有外生输出（销售产品）
          单价 = current_exogenous_output_price / current_exogenous_output_quantity
        
        设计决策 (2026-01-10): 使用单价计算利润
        原因：总价格会导致错误的利润计算（如 120 vs 16 = -104 亏损，
              但实际单价 120/10=12 vs 16 = +4 利润）
        """
        if role == "SELLER":
            # 卖家：外生采购成本（单价）
            total_price = getattr(self.awi, "current_exogenous_input_price", None)
            quantity = getattr(self.awi, "current_exogenous_input_quantity", None)
            if total_price is None or quantity is None or quantity <= 0:
                return None
            return float(total_price) / float(quantity)
        else:
            # 买家：外生销售收入（单价）
            total_price = getattr(self.awi, "current_exogenous_output_price", None)
            quantity = getattr(self.awi, "current_exogenous_output_quantity", None)
            if total_price is None or quantity is None or quantity <= 0:
                return None
            return float(total_price) / float(quantity)

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

    # =========================================================================
    # Exposure Book (2026-01-12 Reviewer P0 建议)
    # =========================================================================
    # 
    # 问题背景:
    #   BUYER 给 N 个对手发 offer，每个 q=5，但只看到 need=10
    #   如果 3 个对手同时接受 → 实际买入 15，超买 50%
    # 
    # 解决方案:
    #   跟踪所有 pending outgoing offers (还没收到回复的)
    #   pending_worst = Σ(q_sent) for all awaiting offers
    #   pending_expected = Σ(q_sent × posterior_mean)  # P1-2: 使用后验均值
    #   need_adj = need - pending_worst (用于 select_subset)
    # =========================================================================

    def _get_committed_qty(self, role: str) -> float:
        """
        获取当天已签约的数量。
        
        P0: Committed Book - 用于数量控制闭环
        """
        if hasattr(self, '_committed_qty'):
            return self._committed_qty.get(role, 0.0)
        return 0.0

    def _get_pending_exposure(self, role: str) -> tuple[float, float]:
        """
        计算指定角色的 pending exposure。
        
        Returns:
            (pending_worst, pending_expected)
            - pending_worst: 所有 pending offers 的名义 q 之和 (假设全部被接受)
            - pending_expected: Σ(q × posterior_mean) (期望接受量)
            
        P1-2: 使用 BOU 后验均值 alpha/(alpha+beta) 而非 LCB
        原因: LCB 是保守下界，会系统性低估 pending 被接受多少
              导致 need_adj 偏大，responder 侧更倾向于再接受一批
        """
        pending_worst = 0.0
        pending_expected = 0.0
        
        for pid, awaiting in self._awaiting_response.items():
            if not awaiting:
                continue
            info = self._last_offer_sent.get(pid)
            if info is None:
                continue
            if info.role != role:
                continue
            q = info.offer[QUANTITY]
            
            # P1-2: 使用 BOU 后验均值而非 LCB
            # 尝试获取后验均值，回退到 mu
            posterior_mean = self._bou.posterior_mean(
                pid=pid,
                role=role,
                round_bucket=info.round_bucket,
                p_bin=info.p_bin,
                q=q,
                mu=info.mu,
                strength=info.strength,
            )
            p_accept = posterior_mean if posterior_mean is not None else info.mu
            
            pending_worst += q
            pending_expected += q * p_accept
        
        return pending_worst, pending_expected

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
        lcb: Optional[float] = None,  # 2026-01-12: 增加 LCB 参数用于 Exposure Book
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
            lcb=lcb if lcb is not None else mu,  # 默认用 mu
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
        force_price: Optional[float] = None,  # 2026-01-11: 显式强制价格（用于 penalty-aware）
    ) -> dict[str, Outcome | None]:
        """
        为指定角色的伙伴生成报价。
        修改点:
        - 0.1: Feasibility-first probe disable
        - 0.2: Probe q allocation (ceil(target / n_partners))
        - Q: q consistency fix (使用实际 q 计算 p_eff)
        - 2: Counter price anchoring (使用 opp_offers)
        - 3: Panic mode
        - 2026-01-11: force_price 参数，避免用 round_rel=1.0 偷价格
        """
        import math

        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        round_rel = round_rel_override if round_rel_override is not None else self._round_rel_from_state(None)
        base_price = self._price_for_role(role, p_min, p_max, round_rel)
        
        # 2026-01-11 P0-修复1: 显式强制价格，不再用 round_rel=1.0 偷价格
        # 这样可以避免 round_rel=1.0 意外触发 panic 模式
        if force_price is not None:
            base_price = force_price
        trading_price = self._trading_price(role)
        need_remaining = need_override if need_override is not None else self._need_remaining(role)
        if need_remaining <= 0 or q_max <= 0:
            return {pid: None for pid in partners}

        t_value = self._select_time(issues)
        n_partners = max(1, len(partners))
        
        # =========================================================================
        # Target 计算 (2026-01-10 更新)
        # =========================================================================
        # 灵感来源: RChan (SCML 2025 竞争对手)
        # 
        # 分析结论:
        #   Shortfall penalty 约为 Disposal cost 的 10 倍（货币量纲）
        #   因此:
        #     - BUYER 应容忍超量采购 (overbuy)，宁可多买也不要 shortfall
        #     - SELLER 保持保守，不超量 (oversell 导致高额 shortfall 惩罚)
        # 
        # RChan 参数:
        #   overordering_max_selling = 0.0 (卖家不超量)
        #   overordering_max_buying = 0.2 (买家超量 20%)
        # 
        # 我们先用 10% 做实验
        # 
        # 2026-01-10 更新: 移除 buffer 机制
        # 原因: buffer 和 overordering 功能重复，简化为只用 buyer_overordering_ratio
        # 原代码: base_target = need_remaining * (1.0 + self._buffer(round_rel))
        # =========================================================================
        if role == "BUYER":
            # BUYER: 允许超量采购，因为 disposal penalty 远低于 shortfall penalty
            # 
            # 2026-01-13 P1-1: Conditional Overordering
            # 只有当 shortfall_unit > buy_price + disposal_unit + margin 时才 +1
            # 否则超买的成本可能超过短缺的惩罚
            should_overorder = False
            if self.cfg.conditional_overordering_enabled:
                spot_in = self._trading_price("BUYER")
                spot_out = self._trading_price("SELLER")
                shortfall_unit = (self.awi.current_shortfall_penalty or 0.0) * self._penalty_scale(False, spot_out)
                disposal_unit = (self.awi.current_disposal_cost or 0.0) * self._penalty_scale(True, spot_in)
                # 使用 trading_price 作为 buy_price 估计
                buy_price_est = spot_in if spot_in else p_min
                threshold = buy_price_est + disposal_unit + disposal_unit * self.cfg.conditional_overordering_margin
                should_overorder = shortfall_unit > threshold
            else:
                should_overorder = True  # 不启用条件判断时默认超买
            
            if should_overorder and self.cfg.overordering_ensure_plus_one and need_remaining > 0:
                overorder_amount = max(1, math.ceil(need_remaining * self.cfg.buyer_overordering_ratio))
                target = need_remaining + overorder_amount
            elif should_overorder:
                target = int(need_remaining * (1.0 + self.cfg.buyer_overordering_ratio))
            else:
                target = need_remaining  # 不超买
            
            # 2026-01-11: BUYER 硬上限约束 (用于 counter offer)
            # 确保 counter 的 q 不超过 cap
            buyer_cap = math.ceil(need_remaining * self.cfg.buyer_accept_cap_mult) + self.cfg.buyer_accept_cap_abs
        else:
            # SELLER: 保守策略，不超量
            target = int(need_remaining)
            buyer_cap = float("inf")  # SELLER 不受限制

        # =========================================================================
        # Probe 阶段判断 (2026-01-13 更新：按角色分离)
        # =========================================================================
        # probe 阶段：使用名义剩余分配，避免超量
        # post-probe 阶段：使用 q_eff 逻辑，利用 BOU 估计
        # 
        # probe 天数 = max(probe_steps_min, int(n_steps * probe_steps_ratio))
        # BUYER 和 SELLER 可以有不同的 probe 配置
        n_steps = getattr(self.awi, "n_steps", 50)
        if role == "BUYER":
            probe_ratio = self.cfg.buyer_probe_steps_ratio
            probe_min = self.cfg.buyer_probe_steps_min
        else:  # SELLER
            probe_ratio = self.cfg.seller_probe_steps_ratio
            probe_min = self.cfg.seller_probe_steps_min
        probe_days = max(probe_min, int(n_steps * probe_ratio))
        in_probe_phase = self.awi.current_step < probe_days

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
            # 系统 fulfill 概率
            # 2026-01-12: 禁用 breach 概率时 fulfill = 1
            if self.cfg.disable_breach_probability:
                fulfill = 1.0
            else:
                fulfill = self._breach_provider.get_fulfill_prob(pid) if self._breach_provider else None
                fulfill = fulfill if fulfill is not None else 1.0
            p_eff = lcb_sign * fulfill
            # 用于排序的得分（使用外生价格或 trading_price）
            exo_price = self._exogenous_price(role)
            if self.cfg.use_exo_price_for_utility and exo_price is not None:
                score = compute_true_profit(base_price, exo_price, role == "SELLER") * p_eff
            else:
                score = price_marginal_gain(base_price, trading_price, role == "SELLER") * p_eff
            partner_info.append((pid, context, p_eff, score, fulfill))

        partner_info.sort(key=lambda x: x[3], reverse=True)
        
        # =========================================================================
        # 数量分配逻辑 (2026-01-10 更新)
        # =========================================================================
        # 
        # Probe 阶段 (in_probe_phase=True):
        #   - 使用"名义剩余"分配，保证 sum(q) <= target
        #   - 将数量分散给多个 partner，收集更多数据
        #   - q_per_partner = ceil(target / n_partners)
        # 
        # Post-probe 阶段 (in_probe_phase=False):
        #   - 使用"期望值"方法，q = remaining_eff / p_eff
        #   - 此时 BOU 已有足够数据，p_eff 估计更准确
        #   - remaining_eff 追踪期望接受量
        # =========================================================================
        
        offers: dict[str, Outcome | None] = {}
        
        # 2026-01-11 P1-修复3: BUYER 发出 offer 总量预算 (适用于所有阶段)
        # 防止"需求 8，却给 10 个 partner 都发 q=6"的灾难
        # 
        # 2026-01-12 方案 A: 使用全局预算而非局部预算
        # 避免多波次发单累积 exposure
        if role == "BUYER" and self.cfg.buyer_offer_budget_enabled:
            # 如果 budget 还未初始化（比如直接调用 counter_all 而没有 first_proposals）
            if not getattr(self, '_offer_budget_initialized', {}).get(role, False):
                self._offer_budget_total[role] = math.ceil(need_remaining * self.cfg.buyer_offer_budget_mult) + self.cfg.buyer_offer_budget_abs
                self._offer_budget_used[role] = 0.0
                self._offer_budget_initialized[role] = True
        # 不再使用局部 offer_budget 变量，改用每次迭代重新计算 budget_remaining
        
        # 2026-01-11 P0-修复2: probe 阶段永远走 nominal 分配
        # panic 最多影响价格，不影响 q 分配模式
        # 原因：probe 阶段的目的是"别因为概率不准而乱发大单"
        if in_probe_phase:
            # === Probe 阶段：名义剩余分配 ===
            remaining_nominal = float(target)
            q_per_partner = max(1, math.ceil(target / n_partners))
            
            for pid, context, p_eff_cand, _, fulfill in partner_info:
                # 2026-01-12 P0-2: offer budget 检查
                # 每次迭代重新计算剩余额度
                budget_remaining = self._offer_budget_total.get(role, float("inf")) - self._offer_budget_used.get(role, 0.0)
                if role == "BUYER" and self.cfg.buyer_offer_budget_enabled and budget_remaining <= 0:
                    offers[pid] = None
                    continue
                
                if remaining_nominal <= 0:
                    offers[pid] = None
                    continue
                
                # 分散给多个 partner，但不超过剩余
                q = min(q_max, max(1, min(q_per_partner, int(round(remaining_nominal)))))
                
                # 2026-01-11: BUYER 硬上限约束 (probe 阶段也需要)
                if role == "BUYER":
                    q = min(q, buyer_cap)
                    # 还要确保不超过剩余 budget
                    if self.cfg.buyer_offer_budget_enabled:
                        q = min(q, int(budget_remaining))
                    q = max(1, q)  # 至少发 1
                
                # 2026-01-12 方案 A: 更新全局预算
                if hasattr(self, '_offer_budget_used'):
                    self._offer_budget_used[role] = self._offer_budget_used.get(role, 0.0) + q
                
                # 获取该 partner 的 state（用于计算谈判内 round_rel）
                partner_state = states.get(pid) if states else None
                price = self._counter_price_for_partner(
                    role, pid, base_price, p_min, p_max, opp_offers, 
                    state=partner_state, panic_active=False
                )
                offer = (q, t_value, price)
                offers[pid] = offer
                remaining_nominal -= q
        else:
            # === Post-probe 阶段 ===
            # 
            # 2026-01-13 P0-1: 从 "1/p放量" 改为 "名义量控制 + 小颗粒分配"
            # 
            # 问题背景 (Reviewer 分析):
            #   - 原逻辑: q = remaining_eff / p_eff (期望值放量)
            #   - 当 p_eff=0.3, target=8 时: 给每个 partner 发 q=8/0.3≈27
            #   - 如果多个 partner 同时接受 → 严重 overfill
            #   - 即使加了 cap，也只是把 27 限到 6，sum 仍然很大
            # 
            # 解决方案:
            #   1. 使用名义量分配: remaining_nominal -= q (而非 -= q*p_eff)
            #   2. 限制 q 到小颗粒: q ∈ {1, 2, 3} (post_probe_max_q)
            #   3. p_eff 只用于 partner 排序（优先给高概率 partner）
            #   4. 动态 min_partners: min(n, max(4, ceil(target/2)))
            # =========================================================================
            
            # P1-2: 动态计算 min_partners
            if self.cfg.post_probe_dynamic_min_partners:
                min_partners = min(len(partner_info), max(4, math.ceil(target / 2)))
            else:
                min_partners = min(self.cfg.post_probe_min_partners, len(partner_info))
            
            # P0-1: 使用名义量分配
            if self.cfg.post_probe_use_nominal_allocation:
                # =========================================================================
                # P1-1 (2026-01-14): "先铺开再加码" 策略
                # =========================================================================
                # 问题: 按 BOU 排序后把 q 集中给前几个 partner，budget 很快用完
                #       后面的 partner 没 offer，一旦前面有 1 个没成就 shortfall
                # 
                # 解决: 两轮分配
                #   第一轮: 每个 partner 先发 q=base_q (铺开覆盖面)
                #   第二轮: 给高概率 partner 加码 (利用剩余 budget)
                # =========================================================================
                
                remaining_nominal = float(target)
                base_q = self.cfg.post_probe_spread_base_q if self.cfg.post_probe_spread_first else self.cfg.post_probe_max_q
                
                if self.cfg.post_probe_spread_first:
                    # === 第一轮: 铺开 (每个 partner 发 base_q) ===
                    first_round_offers: dict[str, tuple[int, float]] = {}  # pid -> (q, price)
                    
                    for pid, context, p_eff_cand, _, fulfill in partner_info:
                        # offer budget 检查
                        budget_remaining = self._offer_budget_total.get(role, float("inf")) - self._offer_budget_used.get(role, 0.0)
                        if role == "BUYER" and self.cfg.buyer_offer_budget_enabled and budget_remaining <= 0:
                            continue
                        
                        if remaining_nominal <= 0:
                            continue
                        
                        q = min(base_q, int(round(remaining_nominal)))
                        q = max(1, q)
                        
                        # BUYER 硬上限约束
                        if role == "BUYER":
                            q = min(q, buyer_cap)
                            if self.cfg.buyer_offer_budget_enabled:
                                q = min(q, int(budget_remaining))
                            q = max(1, q)
                        
                        # 更新全局预算
                        if hasattr(self, '_offer_budget_used'):
                            self._offer_budget_used[role] = self._offer_budget_used.get(role, 0.0) + q
                        
                        partner_state = states.get(pid) if states else None
                        price = self._counter_price_for_partner(
                            role, pid, base_price, p_min, p_max, opp_offers,
                            state=partner_state, panic_active=panic_active
                        )
                        first_round_offers[pid] = (q, price)
                        remaining_nominal -= q
                    
                    # === 第二轮: 加码 (给高概率 partner 追加) ===
                    # 按 p_eff 排序，优先给高概率的加码
                    if remaining_nominal > 0:
                        for pid, context, p_eff_cand, _, fulfill in partner_info:
                            if pid not in first_round_offers:
                                continue
                            
                            budget_remaining = self._offer_budget_total.get(role, float("inf")) - self._offer_budget_used.get(role, 0.0)
                            if role == "BUYER" and self.cfg.buyer_offer_budget_enabled and budget_remaining <= 0:
                                break
                            
                            if remaining_nominal <= 0:
                                break
                            
                            current_q, current_price = first_round_offers[pid]
                            # 最多加到 post_probe_max_q
                            max_add = self.cfg.post_probe_max_q - current_q
                            add_q = min(max_add, int(round(remaining_nominal)))
                            
                            if role == "BUYER":
                                add_q = min(add_q, buyer_cap - current_q)
                                if self.cfg.buyer_offer_budget_enabled:
                                    add_q = min(add_q, int(budget_remaining))
                            
                            add_q = max(0, add_q)
                            
                            if add_q > 0:
                                first_round_offers[pid] = (current_q + add_q, current_price)
                                remaining_nominal -= add_q
                                if hasattr(self, '_offer_budget_used'):
                                    self._offer_budget_used[role] = self._offer_budget_used.get(role, 0.0) + add_q
                    
                    # 构建最终 offers
                    for pid, _, _, _, _ in partner_info:
                        if pid in first_round_offers:
                            q, price = first_round_offers[pid]
                            offers[pid] = (q, t_value, price)
                        else:
                            offers[pid] = None
                else:
                    # 原逻辑 (不启用 spread_first)
                    q_per_partner = max(1, math.ceil(target / max(1, min_partners)))
                    q_per_partner = min(q_per_partner, self.cfg.post_probe_max_q)
                    
                    partners_assigned = 0
                    for pid, context, p_eff_cand, _, fulfill in partner_info:
                        budget_remaining = self._offer_budget_total.get(role, float("inf")) - self._offer_budget_used.get(role, 0.0)
                        if role == "BUYER" and self.cfg.buyer_offer_budget_enabled and budget_remaining <= 0:
                            offers[pid] = None
                            continue
                        
                        if remaining_nominal <= 0 and partners_assigned >= min_partners:
                            offers[pid] = None
                            continue
                        
                        q = min(self.cfg.post_probe_max_q, max(1, min(q_per_partner, int(round(remaining_nominal)))))
                        
                        if role == "BUYER":
                            q = min(q, buyer_cap)
                            if self.cfg.buyer_offer_budget_enabled:
                                q = min(q, int(budget_remaining))
                            q = max(1, q)
                        
                        if partners_assigned < min_partners and remaining_nominal <= 0:
                            q = 1
                        
                        if panic_active and remaining_nominal > 0:
                            q = min(self.cfg.post_probe_max_q, max(q, 2))
                        
                        partners_assigned += 1
                        if hasattr(self, '_offer_budget_used'):
                            self._offer_budget_used[role] = self._offer_budget_used.get(role, 0.0) + q
                        
                        partner_state = states.get(pid) if states else None
                        price = self._counter_price_for_partner(
                            role, pid, base_price, p_min, p_max, opp_offers,
                            state=partner_state, panic_active=panic_active
                        )
                        offer = (q, t_value, price)
                        offers[pid] = offer
                        remaining_nominal -= q
            else:
                # 旧逻辑: 期望值方法 (1/p放量)，保留为向后兼容
                remaining_eff = float(target)
                partners_assigned = 0
                
                for pid, context, p_eff_cand, _, fulfill in partner_info:
                    budget_remaining = self._offer_budget_total.get(role, float("inf")) - self._offer_budget_used.get(role, 0.0)
                    if role == "BUYER" and self.cfg.buyer_offer_budget_enabled and budget_remaining <= 0:
                        offers[pid] = None
                        continue
                    
                    if remaining_eff <= 0 and partners_assigned >= min_partners:
                        offers[pid] = None
                        continue
                    
                    # 旧公式: q = remaining / p_eff
                    q_raw = int(round(remaining_eff / max(p_eff_cand, 1e-6)))
                    q_max_allowed = min(q_max, self.cfg.q_candidate + self.cfg.post_probe_max_q_delta)
                    
                    if role == "BUYER":
                        q_max_allowed = min(q_max_allowed, buyer_cap)
                        if self.cfg.buyer_offer_budget_enabled:
                            q_max_allowed = min(q_max_allowed, budget_remaining)
                    
                    q = min(q_max_allowed, max(1, q_raw))
                    
                    if partners_assigned < min_partners and remaining_eff <= 0:
                        q = 1
                    
                    if panic_active:
                        q = min(q_max_allowed, max(1, need_remaining))
                    
                    partners_assigned += 1
                    if hasattr(self, '_offer_budget_used'):
                        self._offer_budget_used[role] = self._offer_budget_used.get(role, 0.0) + q
                    
                    partner_state = states.get(pid) if states else None
                    price = self._counter_price_for_partner(
                        role, pid, base_price, p_min, p_max, opp_offers,
                        state=partner_state, panic_active=panic_active
                    )
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
            # 2026-01-12: 计算并存储 LCB 用于 Exposure Book
            lcb = self._bou.lcb(
                pid=pid,
                role=role,
                round_bucket=context.round_bucket,
                p_bin=of.p_bin,
                q=of.q,
                mu=mu,
                strength=strength,
                delta=self.cfg.lcb_delta_accept,
            )
            lcb = lcb if lcb is not None else mu
            self._last_offer_sent[pid] = self._pack_sent_offer(
                pid, offer, role, state, mu, strength, round_rel_override=round_rel, lcb=lcb
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
        state: Optional[SAOState] = None,
        panic_active: bool = False,
    ) -> float:
        """
        计算针对特定 partner 的反报价价格。
        
        修改点 2: Counter price anchoring
        p_counter = (1 - η) × base_price + η × p_opp
        其中 η = round_rel^k (k = counter_anchor_eta_exp)
        
        修复 (2026-01-10 Reviewer 问题 A):
        - 原实现使用全局 round_rel (仿真进度)，导致锚定几乎不动
        - 修复后使用谈判内 round_rel，让锚定随谈判进度变化
        
        修复 (2026-01-10 Reviewer 问题 D):
        - 原实现 panic 时只改 q 不改 price
        - 修复后 panic 时直接跳到中间价 (更激进)
        """
        if opp_offers is None or pid not in opp_offers:
            return base_price
        opp_offer = opp_offers[pid]
        opp_price = opp_offer[UNIT_PRICE]

        # 问题 D 修复: Panic 模式下使用对手最优价格
        # 2026-01-10 更新: 直接出 p_min (卖家) 或 p_max (买家)
        # 
        # 原方案 (中间价) 的问题:
        #   当 price_range=1 时 (如 p_min=16, p_max=17):
        #   p_mid = 16.5 会被 rounding 压回 16
        #   => panic 根本没有把买价抬上去
        # 
        # 新方案: 直接使用对手最优价格，完全绕过 rounding 问题
        #   SELLER panic: 出 p_min (对买方最有利，最容易成交)
        #   BUYER panic:  出 p_max (对卖方最有利，最容易成交)
        if panic_active:
            if role == "BUYER":
                # 买家 panic: 出最高价，对卖方最有利
                return float(p_max)
            else:  # SELLER
                # 卖家 panic: 出最低价，对买方最有利
                return float(p_min)

        # 问题 A 修复: 使用谈判内 round_rel 而非仿真进度
        round_rel = self._round_rel_from_state(state)  # 传入具体的 state
        eta = round_rel ** self.cfg.counter_anchor_eta_exp
        price = (1.0 - eta) * base_price + eta * opp_price

        # Monotonic 约束: BUYER 价格只能升，SELLER 价格只能降
        if self.cfg.counter_monotonic and pid in self._last_price_sent:
            last_price = self._last_price_sent[pid]
            if role == "BUYER":
                price = max(price, last_price)
            else:  # SELLER
                price = min(price, last_price)

        return self._snap_price(price, p_min, p_max)

    def _should_panic(self, role: str, need_remaining: int, round_rel: float) -> bool:
        """
        判断是否应进入 panic 模式。
        触发条件: R = penalty/cost > threshold AND round_rel > threshold AND remaining > 0
        
        设计决策 (2026-01-10): panic_round_rel_threshold = 0.1 而非 0.6
        
        这个值看似很小，但实际上：
        - 99.4% 的 OneShot 谈判在 round 1 (round_rel ≈ 0.03) 结束
        - 只有约 0.6% 的谈判能达到 round_rel > 0.1
        - 因此 0.1 阈值意味着"比平均谈判长 3 倍以上才触发 panic"
        - 原阈值 0.6 在 OneShot 中几乎永远不会触发
        
        详见 open_questions.md 第 8 条
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
        price = self._counter_price_for_partner(
            role, pid, base_price, p_min, p_max, opp_offers_dict,
            state=state, panic_active=False
        )

        q = min(q_max, max(1, min(need_remaining, q_max)))
        t_value = self._select_time(issues)
        return (q, t_value, price)

    def _select_subset(self, role: str, partners: list[str], offers: dict[str, Outcome], states: dict[str, SAOState]) -> set[str]:
        """
        选择要接受的 offer 子集，使用组合优化最大化总利润。
        
        评分公式 (2026-01-10 修订):
            score = utility - penalty_cost - over_penalty
        其中:
            - utility: 使用外生价格计算的真实利润 (详见第 9 条)
            - penalty_cost: 缺货惩罚 = shortfall_unit × max(0, need - q_eff)
            - over_penalty: 超量惩罚 = disposal_unit × max(0, q_eff - need)
        
        设计决策 (2026-01-10):
        1. 使用外生价格计算真实利润，而非 trading_price 的边际收益
           - 卖家: profit = sell_price - exo_input_price
           - 买家: profit = exo_output_price - buy_price
           详见 open_questions.md 第 9 条
        
        2. 移除多余的 risk_penalty (原: risk_lambda × (1-p_eff) × q)
           - 冗余原因: q_eff = q × p_eff 已经体现风险折扣
           - 再加 risk_penalty 是双重惩罚
        
        3. 移除 over_penalty 的 10% 乘数 (原: 0.1 × disposal × overfill)
           - 冗余原因: disposal_cost 本身就是超量惩罚单位
           - 10% 乘数导致对超量风险低估
        
        详见 open_questions.md 第 10 条
        """
        need_remaining = self._need_remaining(role)
        if need_remaining <= 0:
            return set()
        
        # =========================================================================
        # P0: Committed Book + Exposure Book 闭环 (2026-01-12 Reviewer 建议)
        # =========================================================================
        # 
        # 问题: 并行谈判中，对方可能直接 ACCEPT 我方 offer
        #       导致同一 step 内 "已签约量" 不断累积
        #       但原来的 need_adj 没有扣减 committed，导致数量失控
        # 
        # 解决方案:
        #   need_live = need_remaining - committed  (扣减已签约)
        #   need_adj = need_live - pending_xxx      (再扣减 pending)
        #   cap 检查: committed + pending_worst + subset_q <= cap_total
        # 
        # 设计决策: 不对称策略
        # - BUYER: need_adj = need_live - pending_expected
        # - SELLER: need_adj = need_live - pending_worst
        # =========================================================================
        committed = self._get_committed_qty(role)
        need_live = max(0.0, need_remaining - committed)
        
        pending_worst, pending_expected = self._get_pending_exposure(role)
        if role == "BUYER":
            need_adj = max(0.0, need_live - pending_expected)
        else:  # SELLER
            need_adj = max(0.0, need_live - pending_worst)
        
        issues = self._issues_for_role(role)
        p_min, p_max = self._price_bounds(issues)
        q_max = self._q_max(issues)
        round_rel = self._round_rel_from_states(partners, states)
        trading_price = self._trading_price(role)
        spot_in = self._trading_price("BUYER")
        spot_out = self._trading_price("SELLER")
        
        # 获取外生价格（真实成本/收入基准）
        exo_price = self._exogenous_price(role)
        use_exo = self.cfg.use_exo_price_for_utility and exo_price is not None
        
        # 惩罚单位成本
        # 注意：underfill 和 overfill 的惩罚对于不同角色是不同的
        # 
        # OneShot 层级结构:
        #   Level 0 (卖家): 有外生输入（采购原材料），向 Level 1 销售
        #   Level 1 (买家): 从 Level 0 购买，有外生输出（销售产品）
        #
        # role 定义 (见 _split_by_role):
        #   role == "SELLER": LOS 是卖方（Level 0），向 Level 1 销售
        #   role == "BUYER": LOS 是买方（Level 1），从 Level 0 购买
        #
        # 惩罚逻辑:
        # - SELLER (Level 0):
        #   - underfill (卖少了) → disposal cost (原材料剩余)
        #   - overfill (卖多了) → shortfall penalty (承诺了但无法交付)
        # - BUYER (Level 1):
        #   - underfill (买少了) → shortfall penalty (无法满足外生输出需求)
        #   - overfill (买多了) → disposal cost (买多的原材料用不掉)
        shortfall_unit = (self.awi.current_shortfall_penalty or 0.0) * self._penalty_scale(False, spot_out)
        disposal_unit = (self.awi.current_disposal_cost or 0.0) * self._penalty_scale(True, spot_in)
        
        if role == "BUYER":
            underfill_penalty_unit = shortfall_unit  # 买少了 → shortfall
            overfill_penalty_unit = disposal_unit    # 买多了 → disposal
        else:  # SELLER
            underfill_penalty_unit = disposal_unit   # 卖少了 → disposal (原材料剩余)
            overfill_penalty_unit = shortfall_unit   # 卖多了 → shortfall (无法交付)
        
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
            # 2026-01-12: 禁用 breach 概率时 fulfill = 1
            if self.cfg.disable_breach_probability:
                p_eff = 1.0
            else:
                fulfill = self._breach_provider.get_fulfill_prob(pid) if self._breach_provider else None
                p_eff = fulfill if fulfill is not None else 1.0
            
            # 计算 utility: 使用真实利润或边际收益
            if use_exo:
                # 真实利润 = (价格差) × 数量
                utility = compute_true_profit(of.p, exo_price, role == "SELLER") * of.q
            else:
                # 回退到边际收益（相对于 trading_price）
                utility = price_marginal_gain(of.p, trading_price, role == "SELLER") * of.q
            
            partner_scores.append((pid, of, context, p_eff, utility))

        partner_scores.sort(key=lambda x: x[4], reverse=True)
        
        # =========================================================================
        # 候选集选择 (2026-01-10 改进: 可达性规则)
        # =========================================================================
        # 
        # 问题背景 (Reviewer 反馈):
        #   原实现: 按单体 utility 排序，硬砍到 portfolio_k=6
        #   问题: 某个 offer 单体看起来很亏（贵），但加入它能显著减少短缺惩罚
        #         这种 offer 在组合里可能是必要的，但被先按单体过滤掉了
        # 
        # 解决方案: 可达性规则
        #   1. 先按单体 utility 排序（没问题）
        #   2. 取最小前缀，使得 sum(q) >= need_remaining * 1.2
        #   3. 如果前缀太短，至少取 portfolio_k 个
        #   4. 对这个候选集做 powerset
        # =========================================================================
        
        # 使用可达性规则选择候选集
        # 2026-01-12: 使用 need_adj (扣除 pending exposure)
        target_q = need_adj * 1.2  # 目标: 覆盖 120% 的需求
        cumulative_q = 0.0
        reachability_k = 0
        
        for i, (pid, of, context, p_eff, utility) in enumerate(partner_scores):
            cumulative_q += of.q
            reachability_k = i + 1
            if cumulative_q >= target_q:
                break
        
        # 确保候选集不小于 portfolio_k，也不超过总数
        candidate_k = max(self.cfg.portfolio_k, reachability_k)
        candidate_k = min(candidate_k, len(partner_scores))
        
        top = partner_scores[:candidate_k]

        # =========================================================================
        # BUYER 硬上限约束 (2026-01-11 Reviewer 建议)
        # =========================================================================
        # 2026-01-12 方案 C: cap 纳入 pending_worst
        # 
        # 问题: 原实现 cap 没把 pending_worst 算进去
        #       可能出现: 接受了 total_q <= cap，但 pending offers 后面也被接受
        #                 导致总签约量大幅超过 cap
        # 
        # 改法: 
        #   cap_total = ceil(need_remaining * cap_mult) + cap_abs  (用原始 need)
        #   检查: pending_worst + total_q > cap_total 则跳过
        # =========================================================================
        import math
        if role == "BUYER":
            buyer_cap_total = math.ceil(need_remaining * self.cfg.buyer_accept_cap_mult) + self.cfg.buyer_accept_cap_abs
        else:
            buyer_cap_total = float("inf")  # SELLER 不受限制

        best_score = float("-inf")
        best_set: set[str] = set()
        n = len(top)
        for mask in range(1 << n):
            total_q = 0.0  # 名义数量（用于硬上限检查）
            # 2026-01-12 Reviewer P0: Responder 确定化
            # 使用名义 q 而非 q_eff = q × p_eff
            # 理由：select_subset 是 Responder（评估收到的 offer），对方已经发了 offer，
            #       如果我们接受，就是接受那个 q，不应该乘概率
            total_q_eff = 0.0  # 现在也用名义 q，不再乘 p_eff
            utility = 0.0
            subset: set[str] = set()
            
            # 收集所有选中的 offer 信息
            selected_offers = []
            for i in range(n):
                if mask & (1 << i):
                    pid, of, _, p_eff, util = top[i]
                    subset.add(pid)
                    total_q += of.q
                    total_q_eff += of.q  # 使用名义 q，不乘 p_eff
                    selected_offers.append((pid, of, p_eff, util))
            
            # BUYER 硬上限检查：committed + pending_worst + total_q > cap_total 直接跳过
            # 2026-01-12 方案 C + P0: 纳入 committed + pending_worst
            if role == "BUYER" and committed + pending_worst + total_q > buyer_cap_total:
                continue
            
            # =========================================================================
            # BUYER Score 计算 (2026-01-11 Reviewer P0 建议修正)
            # =========================================================================
            # 问题：原 buyer_marginal_utility_fix 只对超买部分的收入抹零，但成本也被抹掉了
            #       导致"超买像免费保险"，BUYER 倾向于 overfill
            # 
            # 修正：超量部分的成本必须被计入
            #   - 有用部分：利润 = (exo_out_price - buy_price) × q_useful
            #   - 超量部分：只有成本 = -buy_price × q_excess
            # 
            # 总 utility = Σ[profit_per_unit × q_useful - buy_price × q_excess]
            if role == "BUYER" and self.cfg.buyer_score_rcp:
                utility = 0.0
                cumulative_q = 0.0
                for pid, of, p_eff, util in selected_offers:
                    # 每单位的利润 = exo_out_price - buy_price
                    if of.q > 0:
                        profit_per_unit = util / of.q
                    else:
                        profit_per_unit = 0.0
                    
                    # of.p 就是 buy_price
                    buy_price = of.p
                    
                    # 这个 offer 中"有用"的数量和"超量"的数量
                    # 2026-01-12: 使用 need_adj (扣除 pending exposure)
                    q_useful = min(of.q, max(0, need_adj - cumulative_q))
                    q_excess = of.q - q_useful
                    cumulative_q += of.q
                    
                    # 有用部分计入完整利润，超量部分只计入成本（负值）
                    utility += profit_per_unit * q_useful - buy_price * q_excess
            # =========================================================================
            # SELLER Score 计算 (2026-01-12 Reviewer 建议修正)
            # =========================================================================
            # 问题：SELLER 有固定的外生采购成本，卖不够 (underfill) 时：
            #   - 原先只计入 disposal_cost
            #   - 但实际上还有已采购但卖不掉的成本 (exo_input_price × underfill)
            # 
            # 修正：卖不够的部分扣除采购成本
            #   - 有用部分：利润 = (sell_price - exo_input_price) × q_useful
            #   - 卖不够部分：损失 = -exo_input_price × q_underfill
            elif role == "SELLER" and self.cfg.seller_score_rcp:
                utility = 0.0
                cumulative_q = 0.0
                for pid, of, p_eff, util in selected_offers:
                    # 每单位的利润 = sell_price - exo_input_price
                    if of.q > 0:
                        profit_per_unit = util / of.q
                    else:
                        profit_per_unit = 0.0
                    
                    # 这个 offer 中"有用"的数量
                    q_useful = min(of.q, max(0, need_adj - cumulative_q))
                    cumulative_q += of.q
                    
                    # 只对有用部分计入利润
                    utility += profit_per_unit * q_useful
                
                # 卖不够的部分：扣除已采购但卖不掉的成本
                underfill_q = max(0, need_adj - cumulative_q)
                if underfill_q > 0 and use_exo and exo_price is not None:
                    utility -= exo_price * underfill_q
            elif role == "BUYER" and self.cfg.buyer_marginal_utility_fix:
                # 旧版本：只对有用部分计入收益（已废弃，保留为向后兼容）
                utility = 0.0
                cumulative_q = 0.0
                for pid, of, p_eff, util in selected_offers:
                    if of.q > 0:
                        profit_per_unit = util / of.q
                    else:
                        profit_per_unit = 0.0
                    # 2026-01-12: 使用 need_adj
                    q_useful_in_offer = min(of.q, max(0, need_adj - cumulative_q))
                    cumulative_q += of.q
                    utility += profit_per_unit * q_useful_in_offer
            else:
                # 原逻辑（未启用修正）
                utility = sum(util for _, _, _, util in selected_offers)
            
            # 惩罚计算
            # underfill: 接受数量不足以满足需求
            # overfill: 接受数量超过需求
            # 2026-01-12: 使用 need_adj (扣除 pending exposure 后的需求)
            underfill = max(0.0, need_adj - total_q_eff)
            overfill = max(0.0, total_q_eff - need_adj)
            penalty_cost = underfill_penalty_unit * underfill + overfill_penalty_unit * overfill
            
            score = utility - penalty_cost
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
        """更新 BOU 统计（不更新 PartnerStats，因为已在 _record_accept 中更新过）"""
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
        # 注意：不调用 partner_stats.record_response()
        # 因为 _record_accept() 已经调用过，避免双重计数
        # (2026-01-10 Reviewer 次要问题 1)

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
