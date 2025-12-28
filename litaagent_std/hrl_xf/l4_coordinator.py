"""L4 全局监控器 + α 协调器.

L4 由两部分组成：
1) L4GlobalMonitor：规则监控器，计算全局广播状态
2) GlobalCoordinator：自注意力网络，为每个线程输出优先级 α
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


BUCKET_RANGES = [(0, 2), (3, 7), (8, 14), (15, 40)]


def delta_to_bucket(delta: int) -> int:
    """将相对交货时间映射到桶索引."""
    for i, (lo, hi) in enumerate(BUCKET_RANGES):
        if lo <= delta <= hi:
            return i
    return 3


@dataclass
class ThreadState:
    """单个谈判线程的状态（供 L4 使用）."""
    thread_id: str
    is_buying: bool
    negotiation_step: int
    relative_time: float
    current_offer: Optional[Tuple[int, int, float]]  # (q, delta_t, price)
    history_len: int
    target_bucket: int
    goal_gap: float
    Q_safe_at_t: float
    B_remaining: float


@dataclass
class ThreadPriority:
    """L4 为单个线程计算的优先级."""
    thread_id: str
    alpha: float
    attention_weight: Optional[float] = None


@dataclass
class GlobalBroadcast:
    """L4 广播给每个 L3 线程的全局状态."""
    current_step: int
    n_steps: int
    step_progress: float
    l2_goal: np.ndarray
    goal_gap_buy: np.ndarray
    goal_gap_sell: np.ndarray
    today_bought_qty: float
    today_bought_value: float
    today_sold_qty: float
    today_sold_value: float
    B_remaining: float
    Q_safe_remaining: np.ndarray
    Q_safe_sell_remaining: np.ndarray
    n_active_buy_threads: int
    n_active_sell_threads: int
    x_static: np.ndarray
    X_temporal: np.ndarray


class L4GlobalMonitor:
    """L4 全局监控器（无可训练参数）."""

    def __init__(self, horizon: int = 40):
        self.horizon = horizon
        self._reset_daily_stats()
        self._awi = None
        self._l1_buy = None
        self._l1_sell = None
        self._l2_output = None
        self._state_dict = None

    def _reset_daily_stats(self) -> None:
        self.today_bought: List[Tuple[float, float, int]] = []
        self.today_sold: List[Tuple[float, float, int]] = []
        self._committed_buy_qty = np.zeros(self.horizon + 1, dtype=np.float32)
        self._committed_sell_qty = np.zeros(self.horizon + 1, dtype=np.float32)
        self._committed_buy_budget = 0.0

    def on_step_begin(self, awi, l1_buy, l1_sell, l2_output, state_dict) -> None:
        self._reset_daily_stats()
        self._awi = awi
        self._l1_buy = l1_buy
        self._l1_sell = l1_sell
        self._l2_output = l2_output
        self._state_dict = state_dict

    def on_contract_signed(
        self,
        quantity: int,
        unit_price: float,
        delivery_time: int,
        is_buying: bool,
    ) -> None:
        if self._awi is None:
            return
        delta_t = int(delivery_time) - int(self._awi.current_step)
        delta_t = max(0, min(delta_t, self.horizon))

        if is_buying:
            self.today_bought.append((float(quantity), float(unit_price), delta_t))
            self._committed_buy_budget += float(quantity) * float(unit_price)
            self._committed_buy_qty[delta_t] += float(quantity)
        else:
            self.today_sold.append((float(quantity), float(unit_price), delta_t))
            self._committed_sell_qty[delta_t] += float(quantity)

    def _compute_goal_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        goal = self._l2_output.goal_vector
        bought_by_bucket = np.zeros(4, dtype=np.float32)
        sold_by_bucket = np.zeros(4, dtype=np.float32)

        for qty, _, delta_t in self.today_bought:
            bought_by_bucket[delta_to_bucket(delta_t)] += float(qty)
        for qty, _, delta_t in self.today_sold:
            sold_by_bucket[delta_to_bucket(delta_t)] += float(qty)

        goal_gap_buy = np.zeros(4, dtype=np.float32)
        goal_gap_sell = np.zeros(4, dtype=np.float32)
        for i in range(4):
            goal_gap_buy[i] = max(0.0, float(goal[i * 4]) - bought_by_bucket[i])
            goal_gap_sell[i] = max(0.0, float(goal[i * 4 + 2]) - sold_by_bucket[i])

        return goal_gap_buy, goal_gap_sell

    def compute_broadcast(
        self,
        n_active_buy_threads: int = 0,
        n_active_sell_threads: int = 0,
    ) -> GlobalBroadcast:
        goal_gap_buy, goal_gap_sell = self._compute_goal_gaps()

        B_remaining = max(0.0, float(self._l1_buy.B_free) - float(self._committed_buy_budget))
        Q_safe_remaining = np.maximum(0.0, self._l1_buy.Q_safe - self._committed_buy_qty)
        Q_safe_sell_remaining = np.maximum(0.0, self._l1_sell.Q_safe_sell - self._committed_sell_qty)

        return GlobalBroadcast(
            current_step=int(self._awi.current_step),
            n_steps=int(self._awi.n_steps),
            step_progress=float(self._awi.current_step / max(1, self._awi.n_steps)),
            l2_goal=self._l2_output.goal_vector.copy(),
            goal_gap_buy=goal_gap_buy,
            goal_gap_sell=goal_gap_sell,
            today_bought_qty=float(sum(q for q, _, _ in self.today_bought)),
            today_bought_value=float(sum(q * p for q, p, _ in self.today_bought)),
            today_sold_qty=float(sum(q for q, _, _ in self.today_sold)),
            today_sold_value=float(sum(q * p for q, p, _ in self.today_sold)),
            B_remaining=B_remaining,
            Q_safe_remaining=Q_safe_remaining,
            Q_safe_sell_remaining=Q_safe_sell_remaining,
            n_active_buy_threads=int(n_active_buy_threads),
            n_active_sell_threads=int(n_active_sell_threads),
            x_static=self._state_dict.x_static.copy(),
            X_temporal=self._state_dict.X_temporal.copy(),
        )


class HeuristicAlphaGenerator:
    """启发式 α 生成器（用于离线预训练伪标签）."""

    def __init__(self, urgency_threshold: float = 0.3):
        self.urgency_threshold = urgency_threshold

    def compute_alpha(self, thread_state: ThreadState, global_broadcast: GlobalBroadcast) -> float:
        alpha = 0.0

        if thread_state.is_buying:
            denom = max(1.0, float(global_broadcast.l2_goal[0]) * 10.0)
            resource_ratio = float(thread_state.B_remaining) / denom
            if resource_ratio < self.urgency_threshold:
                alpha += 0.3 * (1.0 - resource_ratio / self.urgency_threshold)

        bucket = int(thread_state.target_bucket)
        gap = float(global_broadcast.goal_gap_buy[bucket]) if thread_state.is_buying else float(global_broadcast.goal_gap_sell[bucket])
        if gap > 0:
            alpha += 0.3 * min(1.0, gap / 10.0)

        alpha += 0.2 * float(thread_state.relative_time)

        if bucket <= 1:
            alpha += 0.2

        alpha = max(-1.0, min(1.0, alpha * 2.0 - 0.5))
        return float(alpha)


if TORCH_AVAILABLE:

    class GlobalCoordinator(nn.Module):
        """L4 全局协调器：为每个活跃线程计算优先级 α."""

        def __init__(
            self,
            d_thread: int = 64,
            d_global: int = 32,
            n_heads: int = 4,
            n_layers: int = 2,
        ):
            super().__init__()
            self.thread_encoder = nn.Sequential(
                nn.Linear(11, d_thread),
                nn.ReLU(),
                nn.Linear(d_thread, d_thread),
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(12, d_global),
                nn.ReLU(),
                nn.Linear(d_global, d_global),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_thread + d_global,
                nhead=n_heads,
                dim_feedforward=(d_thread + d_global) * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.alpha_head = nn.Sequential(
                nn.Linear(d_thread + d_global, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh(),
            )

        def forward(
            self,
            thread_states: "torch.Tensor",
            global_state: "torch.Tensor",
            thread_mask: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            B, N, _ = thread_states.shape
            thread_emb = self.thread_encoder(thread_states)
            global_emb = self.global_encoder(global_state).unsqueeze(1).expand(-1, N, -1)
            combined = torch.cat([thread_emb, global_emb], dim=-1)

            attn_mask = ~thread_mask if thread_mask is not None else None
            hidden = self.transformer(combined, src_key_padding_mask=attn_mask)
            alpha = self.alpha_head(hidden).squeeze(-1)
            return alpha

else:

    class GlobalCoordinator:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GlobalCoordinator")


class L4Layer:
    """L4 层统一接口：监控器 + 协调器."""

    def __init__(
        self,
        horizon: int = 40,
        use_neural_alpha: bool = True,
        model_path: Optional[str] = None,
    ):
        self.monitor = L4GlobalMonitor(horizon=horizon)
        self.use_neural_alpha = bool(use_neural_alpha and TORCH_AVAILABLE)
        self.heuristic_alpha = HeuristicAlphaGenerator()
        self.coordinator = None
        if self.use_neural_alpha:
            self.coordinator = GlobalCoordinator()
            if model_path:
                self._load_model(model_path)

    def compute_alphas(
        self,
        thread_states: List[ThreadState],
        broadcast: GlobalBroadcast,
    ) -> List[ThreadPriority]:
        if not thread_states:
            return []

        if not self.use_neural_alpha or self.coordinator is None:
            return [
                ThreadPriority(
                    thread_id=ts.thread_id,
                    alpha=self.heuristic_alpha.compute_alpha(ts, broadcast),
                )
                for ts in thread_states
            ]

        thread_tensor = self._encode_threads(thread_states)
        global_tensor = self._encode_global(broadcast)
        with torch.no_grad():
            alphas = self.coordinator(thread_tensor, global_tensor)
        alphas_np = alphas.squeeze(0).cpu().numpy().astype(np.float32)
        priorities: List[ThreadPriority] = []
        for i, ts in enumerate(thread_states):
            priorities.append(ThreadPriority(thread_id=ts.thread_id, alpha=float(alphas_np[i])))
        return priorities

    def _encode_threads(self, threads: List[ThreadState]) -> "torch.Tensor":
        features = []
        for t in threads:
            offer = t.current_offer
            if offer is None:
                has_offer = 0.0
                q_val, delta_val, price_val = 0.0, 0.0, 0.0
            else:
                has_offer = 1.0
                q_val, delta_val, price_val = offer
            features.append([
                0.0 if t.is_buying else 1.0,
                float(t.relative_time),
                float(t.negotiation_step) / 20.0,
                has_offer,
                float(q_val) / 10.0,
                float(price_val) / 40.0,
                float(delta_val) / 40.0,
                float(t.target_bucket) / 3.0,
                float(t.goal_gap) / 10.0,
                float(t.Q_safe_at_t) / 100.0,
                float(t.B_remaining) / 10000.0,
            ])
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _encode_global(self, gb: GlobalBroadcast) -> "torch.Tensor":
        features = [
            float(gb.step_progress),
            float(gb.B_remaining) / 10000.0,
            *[float(g) / 10.0 for g in gb.goal_gap_buy],
            *[float(g) / 10.0 for g in gb.goal_gap_sell],
            float(gb.n_active_buy_threads) / 10.0,
            float(gb.n_active_sell_threads) / 10.0,
        ]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.coordinator.load_state_dict(state_dict)
        self.coordinator.eval()


__all__ = [
    "BUCKET_RANGES",
    "delta_to_bucket",
    "ThreadState",
    "ThreadPriority",
    "GlobalBroadcast",
    "L4GlobalMonitor",
    "HeuristicAlphaGenerator",
    "GlobalCoordinator",
    "L4Layer",
]
