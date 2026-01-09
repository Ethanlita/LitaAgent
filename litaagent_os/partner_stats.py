from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class PartnerStatsSnapshot:
    accept_rate_last_k: float
    counter_rate_last_k: float
    breach_rate_last_k: float
    n_deals: int
    n_breaches: int


class PartnerStats:
    def __init__(self, k: int = 32) -> None:
        self._k = k
        self._accept_seq: Deque[int] = deque(maxlen=k)
        self._counter_seq: Deque[int] = deque(maxlen=k)
        self._breach_seq: Deque[int] = deque(maxlen=k)
        self.n_deals = 0
        self.n_breaches = 0

    def record_response(self, accepted: bool) -> None:
        self._accept_seq.append(1 if accepted else 0)
        self._counter_seq.append(0 if accepted else 1)

    def record_breach(self, breached: bool) -> None:
        self._breach_seq.append(1 if breached else 0)
        if breached:
            self.n_breaches += 1
        self.n_deals += 1

    def snapshot(self) -> PartnerStatsSnapshot:
        accept_rate = sum(self._accept_seq) / len(self._accept_seq) if self._accept_seq else 0.5
        counter_rate = sum(self._counter_seq) / len(self._counter_seq) if self._counter_seq else 0.5
        breach_rate = sum(self._breach_seq) / len(self._breach_seq) if self._breach_seq else 0.5
        return PartnerStatsSnapshot(
            accept_rate_last_k=accept_rate,
            counter_rate_last_k=counter_rate,
            breach_rate_last_k=breach_rate,
            n_deals=self.n_deals,
            n_breaches=self.n_breaches,
        )


class PartnerStatsStore:
    def __init__(self, k: int = 32) -> None:
        self._k = k
        self._stats: dict[tuple[str, str], PartnerStats] = {}

    def get(self, partner_id: str, role: str) -> PartnerStats:
        key = (partner_id, role)
        if key not in self._stats:
            self._stats[key] = PartnerStats(k=self._k)
        return self._stats[key]
