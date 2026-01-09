from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scml.oneshot.common import FinancialReport


@dataclass
class CapabilityStatus:
    source: str  # "reports_at_step" / "reports_of_agent" / "none"


class CapabilityProbe:
    """探测系统 breach 信息来源。"""

    def __init__(self) -> None:
        self.status = CapabilityStatus(source="none")

    def probe(self, awi) -> CapabilityStatus:
        # 优先 reports_at_step
        if hasattr(awi, "reports_at_step"):
            try:
                _ = awi.reports_at_step(awi.current_step)
                self.status = CapabilityStatus(source="reports_at_step")
                return self.status
            except Exception:
                pass
        # 其次 reports_of_agent
        if hasattr(awi, "reports_of_agent"):
            try:
                _ = awi.reports_of_agent(awi.id)
                self.status = CapabilityStatus(source="reports_of_agent")
                return self.status
            except Exception:
                pass
        self.status = CapabilityStatus(source="none")
        return self.status


class BreachInfoProvider:
    """统一 breach 信息接口（系统优先）。"""

    def __init__(self, awi, capability: CapabilityStatus) -> None:
        self._awi = awi
        self._cap = capability

    def _find_report(self, pid: str) -> Optional[FinancialReport]:
        if self._cap.source == "reports_at_step":
            reports = self._awi.reports_at_step(self._awi.current_step) or []
            for r in reports:
                if r.agent_id == pid:
                    return r
            return None
        if self._cap.source == "reports_of_agent":
            reports = self._awi.reports_of_agent(pid) or []
            return reports[-1] if reports else None
        return None

    def get_breach_prob(self, pid: str) -> Optional[float]:
        r = self._find_report(pid)
        return None if r is None else r.breach_prob

    def get_breach_level(self, pid: str) -> Optional[float]:
        r = self._find_report(pid)
        return None if r is None else r.breach_level

    def get_fulfill_prob(self, pid: str) -> Optional[float]:
        prob = self.get_breach_prob(pid)
        if prob is not None:
            return 1.0 - prob
        level = self.get_breach_level(pid)
        if level is not None:
            return 1.0 - level
        return None
