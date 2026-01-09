from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Optional
from collections import deque


@dataclass
class HistoryToken:
    speaker: str  # "ME" or "OPP"
    action_type: str  # "OFFER" / "ACCEPT" / "END"
    q_bucket: int
    p_bin: int
    round_bucket: int
    is_counter: bool = False
    is_first_proposal: bool = False


class HistoryBuffer:
    def __init__(self, max_len: int = 100) -> None:
        self._tokens: Deque[HistoryToken] = deque(maxlen=max_len)

    def append(self, token: HistoryToken) -> None:
        self._tokens.append(token)

    def last_tokens(self, n: int) -> list[HistoryToken]:
        if n <= 0:
            return []
        return list(self._tokens)[-n:]

    def __len__(self) -> int:
        return len(self._tokens)


class HistoryStore:
    """按 (partner_id, negotiation_id, role) 维护 NEGOTIATION scope 历史。"""

    def __init__(self, max_len: int = 100) -> None:
        self._buffers: dict[tuple[str, str, str], HistoryBuffer] = {}
        self._max_len = max_len

    def get_buffer(self, partner_id: str, negotiation_id: str, role: str) -> HistoryBuffer:
        key = (partner_id, negotiation_id, role)
        if key not in self._buffers:
            self._buffers[key] = HistoryBuffer(max_len=self._max_len)
        return self._buffers[key]

    def append(self, partner_id: str, negotiation_id: str, role: str, token: HistoryToken) -> None:
        self.get_buffer(partner_id, negotiation_id, role).append(token)

    def clear(self) -> None:
        self._buffers.clear()
