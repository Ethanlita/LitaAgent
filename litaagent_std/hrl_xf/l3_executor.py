"""L3 执行层 - 输出完整 AOP 动作."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .l4_coordinator import GlobalBroadcast
from .l2_manager import BUCKET_RANGES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


@dataclass
class NegotiationRound:
    """谈判轮次记录."""
    quantity: float
    price: float
    delta_t: int
    is_my_turn: bool = True


@dataclass
class SAOAction:
    """AOP 动作抽象."""
    action_type: str  # "accept" | "reject" | "end"
    offer: Optional[Tuple[int, int, float]] = None  # (q, delivery_time_abs, price)


@dataclass
class L3Input:
    """L3 执行器的输入."""
    is_buying: bool
    history: List[NegotiationRound]
    current_offer: Optional[Tuple[int, float, int]]
    negotiation_step: int
    relative_time: float
    partner_id: str
    global_broadcast: GlobalBroadcast
    alpha: float
    time_mask: np.ndarray
    Q_safe: np.ndarray
    B_free: float
    min_price: float = 0.0
    max_price: float = float("inf")


@dataclass
class L3Output:
    """L3 执行器的输出."""
    action: SAOAction
    time_probs: Optional[np.ndarray] = None
    op_probs: Optional[np.ndarray] = None
    confidence: float = 1.0


class HeuristicL3Actor:
    """启发式 L3 执行器（无需训练）."""

    def __init__(self, horizon: int = 40):
        self.horizon = horizon

    def compute(self, l3_input: L3Input) -> L3Output:
        if self._should_accept(l3_input):
            return L3Output(action=SAOAction(action_type="accept"), confidence=0.9)

        if self._should_reject(l3_input):
            return L3Output(action=SAOAction(action_type="end"), confidence=0.6)

        offer = self._generate_offer(l3_input)
        if offer is None:
            return L3Output(action=SAOAction(action_type="end"), confidence=0.5)

        return L3Output(action=SAOAction(action_type="reject", offer=offer), confidence=0.8)

    def _should_accept(self, l3_input: L3Input) -> bool:
        offer = l3_input.current_offer
        if offer is None:
            return False

        qty, price, delta_t = offer
        delta_t = int(max(0, min(delta_t, self.horizon)))
        if delta_t >= len(l3_input.time_mask) or l3_input.time_mask[delta_t] == -np.inf:
            return False

        if qty > float(l3_input.Q_safe[delta_t] if delta_t < len(l3_input.Q_safe) else 0.0):
            return False

        if l3_input.is_buying and qty * price > float(l3_input.B_free):
            return False

        goal = l3_input.global_broadcast.l2_goal
        bucket = self._delta_to_bucket(delta_t)
        if l3_input.is_buying:
            p_limit = float(goal[bucket * 4 + 1])
            return price <= p_limit * 1.1
        p_floor = float(goal[bucket * 4 + 3])
        return price >= p_floor * 0.9

    def _should_reject(self, l3_input: L3Input) -> bool:
        if l3_input.relative_time < 0.9:
            return False
        valid_times = np.where(l3_input.time_mask > -np.inf)[0]
        return len(valid_times) == 0

    def _generate_offer(self, l3_input: L3Input) -> Optional[Tuple[int, int, float]]:
        goal = l3_input.global_broadcast.l2_goal
        bucket = self._pick_bucket(l3_input, goal)
        delta_t = self._pick_delta(l3_input, bucket)
        if delta_t is None:
            return None

        gap = float(
            l3_input.global_broadcast.goal_gap_buy[bucket]
            if l3_input.is_buying
            else l3_input.global_broadcast.goal_gap_sell[bucket]
        )
        max_qty = float(l3_input.Q_safe[delta_t] if delta_t < len(l3_input.Q_safe) else 0.0)
        qty = max(1.0, min(gap if gap > 0 else max_qty, max_qty))

        p_idx = bucket * 4 + (1 if l3_input.is_buying else 3)
        base_price = float(goal[p_idx])
        concession = min(1.0, l3_input.relative_time + max(0.0, l3_input.alpha) * 0.2)
        if l3_input.is_buying:
            price = base_price * (1.0 + 0.2 * concession)
        else:
            price = base_price * (1.0 - 0.2 * concession)

        abs_time = int(l3_input.global_broadcast.current_step + delta_t)
        return (int(qty), abs_time, float(max(0.0, price)))

    def _pick_bucket(self, l3_input: L3Input, goal: np.ndarray) -> int:
        goals = goal.reshape(4, 4)
        q_idx = 0 if l3_input.is_buying else 2
        return int(np.argmax(goals[:, q_idx]))

    def _pick_delta(self, l3_input: L3Input, bucket: int) -> Optional[int]:
        dmin, dmax = BUCKET_RANGES[bucket]
        dmin = max(0, dmin)
        dmax = min(self.horizon, dmax)
        for d in range(dmin, dmax + 1):
            if l3_input.time_mask[d] != -np.inf and l3_input.Q_safe[d] > 0:
                return d
        for d in range(0, self.horizon + 1):
            if l3_input.time_mask[d] != -np.inf and l3_input.Q_safe[d] > 0:
                return d
        return None

    def _delta_to_bucket(self, delta: int) -> int:
        if delta <= 2:
            return 0
        if delta <= 7:
            return 1
        if delta <= 14:
            return 2
        return 3


if TORCH_AVAILABLE:

    class L3DecisionTransformer(nn.Module):
        """基于历史与全局上下文的 L3 网络."""

        def __init__(
            self,
            horizon: int = 40,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            max_seq_len: int = 20,
            context_dim: int = 31,
        ):
            super().__init__()
            self.horizon = horizon
            self.max_seq_len = max_seq_len
            self.history_embed = nn.Linear(4, d_model)
            self.context_embed = nn.Linear(context_dim, d_model)
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
            self.op_head = nn.Linear(d_model, 3)
            self.qty_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus(),
            )
            self.price_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
            self.time_head = nn.Linear(d_model, horizon + 1)

        def forward(
            self,
            history: "torch.Tensor",
            context: "torch.Tensor",
            time_mask: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            B, T, _ = history.shape
            if T > self.max_seq_len:
                history = history[:, -self.max_seq_len :, :]
                T = history.shape[1]

            h_tokens = self.history_embed(history)
            positions = torch.arange(T, device=h_tokens.device)
            h_tokens = h_tokens + self.pos_embed(positions)

            memory = self.context_embed(context).unsqueeze(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(h_tokens.device)
            feat = self.transformer(h_tokens, memory, tgt_mask=causal_mask)
            last_feat = feat[:, -1, :]

            op_logits = self.op_head(last_feat)
            quantity = self.qty_head(last_feat)
            price = self.price_head(last_feat)
            time_logits = self.time_head(last_feat)
            time_logits = time_logits + time_mask

            return op_logits, quantity, price, time_logits

else:

    class L3DecisionTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for L3DecisionTransformer")


class L3Actor:
    """L3 执行层统一接口."""

    def __init__(
        self,
        mode: str = "heuristic",
        horizon: int = 40,
        model_path: Optional[str] = None,
    ):
        self.mode = mode
        self.horizon = horizon

        if mode == "heuristic":
            self._impl = HeuristicL3Actor(horizon=horizon)
            self._model = None
        elif mode == "neural":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for neural mode")
            self._model = L3DecisionTransformer(horizon=horizon)
            if model_path:
                self._load_model(model_path)
            self._impl = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute(self, l3_input: L3Input) -> L3Output:
        if self.mode == "heuristic":
            return self._impl.compute(l3_input)
        return self._compute_neural(l3_input)

    def _compute_neural(self, l3_input: L3Input) -> L3Output:
        history = l3_input.history
        max_len = int(getattr(self._model, "max_seq_len", 20) or 20)
        if max_len > 0 and len(history) > max_len:
            history = history[-max_len:]

        if len(history) == 0:
            history_arr = np.zeros((1, 1, 4), dtype=np.float32)
        else:
            rows = []
            for r in history:
                delta = int(max(0, min(int(r.delta_t), self.horizon)))
                rows.append([float(r.quantity), float(r.price), float(delta), 1.0 if r.is_my_turn else 0.0])
            history_arr = np.asarray(rows, dtype=np.float32)[np.newaxis, :, :]

        context = self._build_context(l3_input)

        history_t = torch.from_numpy(history_arr).float()
        context_t = torch.from_numpy(context).float().unsqueeze(0)
        time_mask_t = torch.from_numpy(l3_input.time_mask).float().unsqueeze(0)

        with torch.no_grad():
            op_logits, q_raw, p_raw, time_logits = self._model(history_t, context_t, time_mask_t)

        op_probs = F.softmax(op_logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
        time_probs = F.softmax(time_logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
        op_id = int(op_logits.argmax(dim=-1).item())

        action = self._decode_action(op_id, q_raw.item(), p_raw.item(), time_logits.squeeze(0), l3_input)
        return L3Output(action=action, time_probs=time_probs, op_probs=op_probs, confidence=1.0)

    def _build_context(self, l3_input: L3Input) -> np.ndarray:
        gb = l3_input.global_broadcast
        min_price = float(l3_input.min_price) if np.isfinite(l3_input.min_price) else 0.0
        max_price = float(l3_input.max_price) if np.isfinite(l3_input.max_price) else 0.0
        context = np.concatenate([
            gb.l2_goal.astype(np.float32),
            gb.goal_gap_buy.astype(np.float32),
            gb.goal_gap_sell.astype(np.float32),
            np.array([
                float(gb.step_progress),
                float(l3_input.relative_time),
                float(l3_input.alpha),
                1.0 if l3_input.is_buying else 0.0,
                float(l3_input.B_free) / 10000.0,
                min_price,
                max_price,
            ], dtype=np.float32),
        ])
        return context

    def _decode_action(
        self,
        op_id: int,
        q_raw: float,
        p_raw: float,
        time_logits: "torch.Tensor",
        l3_input: L3Input,
    ) -> SAOAction:
        has_offer = l3_input.current_offer is not None
        if not has_offer and op_id == 0:
            op_id = 1

        if op_id == 0:
            return SAOAction(action_type="accept")
        if op_id == 2:
            return SAOAction(action_type="end")

        delta_t = int(time_logits.argmax(dim=-1).item())

        quantity = int(round(max(0.0, float(q_raw))))
        ratio = float(p_raw)
        if not np.isfinite(ratio):
            ratio = 0.5
        ratio = max(0.0, min(1.0, ratio))
        min_price = float(l3_input.min_price) if np.isfinite(l3_input.min_price) else 0.0
        max_price = float(l3_input.max_price) if np.isfinite(l3_input.max_price) else min_price
        if not np.isfinite(max_price) or max_price <= min_price:
            if min_price > 0.0:
                max_price = min_price * 2.0
            else:
                max_price = min_price + 1.0
        price = min_price + ratio * (max_price - min_price)

        abs_time = int(l3_input.global_broadcast.current_step + delta_t)
        offer = (int(max(0, quantity)), abs_time, float(max(0.0, price)))
        return SAOAction(action_type="reject", offer=offer)

    def _load_model(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self._model.load_state_dict(state_dict)
        self._model.eval()


__all__ = [
    "NegotiationRound",
    "SAOAction",
    "L3Input",
    "L3Output",
    "HeuristicL3Actor",
    "L3DecisionTransformer",
    "L3Actor",
]
