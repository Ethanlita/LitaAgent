"""训练脚本 - BC、AWR 预训练与 PPO 微调.

实现四阶段课程学习：
- Phase 0: Cold Start - 行为克隆 (BC)
- Phase 1: Offline RL - 优势加权回归 (AWR)
- Phase 2: Online Fine-tune - PPO + 势能奖励
- Phase 3: Self-Play - 自博弈（预留）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .data_pipeline import L4DistillSample, MacroSample, MicroSample


# ============== 配置 ==============

@dataclass
class TrainConfig:
    """训练配置."""
    
    # 数据
    data_dir: str = "./data"
    output_dir: str = "./checkpoints"
    
    # 断点续训（checkpoint 路径）
    l2_resume_path: Optional[str] = None
    l3_bc_resume_path: Optional[str] = None
    l3_awr_resume_path: Optional[str] = None
    
    # 通用
    seed: int = 42
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    # L2 训练
    l2_lr: float = 3e-4
    l2_batch_size: int = 64
    l2_epochs: int = 50
    l2_q_transform: str = "none"
    l2_q_weight: float = 1.0
    
    # L3 训练
    l3_lr: float = 1e-4
    l3_batch_size: int = 32
    l3_epochs: int = 100
    l3_max_history_len: int = 20

    # L4 蒸馏训练（监督）
    l4_lr: float = 3e-4
    l4_batch_size: int = 32
    l4_epochs: int = 50
    l4_resume_path: Optional[str] = None
    
    # PPO
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.95
    
    # AWR
    awr_beta: float = 1.0
    awr_clip: float = 20.0
    
    # 规划视界
    horizon: int = 40  # H，用于 L3Dataset 的 time_mask 长度 (H+1)
    
    # 正则化
    grad_clip: float = 0.5
    weight_decay: float = 1e-4
    
    # 保存
    save_every: int = 10
    log_every: int = 1

    # DataLoader
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 2
    dataloader_drop_last: bool = False

    # 训练日志
    progress_bar: bool = True
    batch_log_every: int = 50


# ============== 数据集 ==============

class L2Dataset(Dataset):
    """L2 宏观数据集."""
    
    def __init__(self, samples: List[MacroSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'state_static': torch.FloatTensor(s.state_static),
            'state_temporal': torch.FloatTensor(s.state_temporal),
            'x_role': torch.FloatTensor(s.x_role),  # Multi-Hot: [can_buy, can_sell]
            'goal': torch.FloatTensor(s.goal),
            'value': torch.FloatTensor([s.value if s.value else 0.0]),
        }


class L3Dataset(Dataset):
    """L3 微观数据集（AOP）。"""

    def __init__(self, samples: List[MicroSample], max_history_len: int = 20, horizon: int = 40):
        self.samples = samples
        self.max_len = max_history_len
        self.horizon = horizon

    def __len__(self):
        return len(self.samples)

    def _build_context(self, s: MicroSample) -> np.ndarray:
        step_progress = float(s.x_static[3]) if s.x_static is not None and len(s.x_static) > 3 else 0.0
        is_buying = 1.0 if s.role == 0 else 0.0
        context = np.concatenate([
            np.asarray(s.l2_goal, dtype=np.float32),
            np.asarray(s.goal_gap_buy, dtype=np.float32),
            np.asarray(s.goal_gap_sell, dtype=np.float32),
            np.array([
                step_progress,
                float(s.relative_time),
                0.0,  # α 在 BC 阶段固定为 0
                is_buying,
                float(s.B_free) / 10000.0,
            ], dtype=np.float32),
        ])
        return context

    def __getitem__(self, idx):
        s = self.samples[idx]

        history = s.history
        if len(history) > self.max_len:
            history = history[-self.max_len:]
        elif len(history) < self.max_len:
            pad = np.zeros((self.max_len - len(history), 4))
            history = np.vstack([pad, history])

        try:
            history = np.asarray(history, dtype=np.float32)
            if history.ndim == 2 and history.shape[1] >= 3:
                history[:, 2] = np.nan_to_num(
                    history[:, 2],
                    nan=0.0,
                    posinf=float(self.horizon),
                    neginf=0.0,
                )
                history[:, 2] = np.clip(history[:, 2], 0.0, float(self.horizon))
        except Exception:
            history = np.zeros((self.max_len, 4), dtype=np.float32)

        tm_len = self.horizon + 1
        time_mask = np.asarray(s.time_mask, dtype=np.float32) if s.time_mask is not None else np.zeros(tm_len, dtype=np.float32)
        if len(time_mask) > tm_len:
            time_mask = time_mask[:tm_len]
        elif len(time_mask) < tm_len:
            time_mask = np.concatenate([time_mask, np.zeros(tm_len - len(time_mask))])

        context = self._build_context(s)

        target_q = float(s.target_quantity) if s.target_quantity is not None else 0.0
        target_p = float(s.target_price) if s.target_price is not None else 0.0
        target_t = int(s.target_time) if s.target_time is not None else 0

        time_valid = 1.0
        if target_t < 0 or target_t >= len(time_mask):
            time_valid = 0.0
            target_t = 0 if len(time_mask) == 0 else max(0, min(target_t, len(time_mask) - 1))
        elif time_mask[target_t] == -np.inf:
            time_valid = 0.0

        return {
            'history': torch.FloatTensor(history),
            'context': torch.FloatTensor(context),
            'action_op': torch.LongTensor([int(s.action_op)]),
            'target_q': torch.FloatTensor([target_q]),
            'target_p': torch.FloatTensor([target_p]),
            'target_t': torch.LongTensor([target_t]),
            'time_mask': torch.FloatTensor(time_mask),
            'time_valid': torch.FloatTensor([time_valid]),
            'reward': torch.FloatTensor([s.reward if s.reward is not None else 0.0]),
        }


@dataclass
class L3TensorizedData:
    history: "torch.Tensor"
    context: "torch.Tensor"
    action_op: "torch.Tensor"
    target_q: "torch.Tensor"
    target_p: "torch.Tensor"
    target_t: "torch.Tensor"
    time_mask: "torch.Tensor"
    time_valid: "torch.Tensor"
    reward: "torch.Tensor"
    reward_valid: "torch.Tensor"

    @property
    def size(self) -> int:
        return int(self.history.shape[0])

    @classmethod
    def from_numpy(cls, arrays: Dict[str, np.ndarray]) -> "L3TensorizedData":
        return cls(
            history=torch.from_numpy(arrays["history"]),
            context=torch.from_numpy(arrays["context"]),
            action_op=torch.from_numpy(arrays["action_op"]).long(),
            target_q=torch.from_numpy(arrays["target_q"]),
            target_p=torch.from_numpy(arrays["target_p"]),
            target_t=torch.from_numpy(arrays["target_t"]).long(),
            time_mask=torch.from_numpy(arrays["time_mask"]),
            time_valid=torch.from_numpy(arrays["time_valid"]),
            reward=torch.from_numpy(arrays["reward"]),
            reward_valid=torch.from_numpy(arrays["reward_valid"]).bool(),
        )


class L3TensorDataset(Dataset):
    """预张量化的 L3 数据集，减少 __getitem__ 开销。"""

    def __init__(self, data: L3TensorizedData, indices: Optional[np.ndarray] = None):
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        if self.indices is None:
            return self.data.size
        return int(len(self.indices))

    def __getitem__(self, idx):
        if self.indices is None:
            i = idx
        else:
            i = int(self.indices[idx])
        return {
            'history': self.data.history[i],
            'context': self.data.context[i],
            'action_op': self.data.action_op[i],
            'target_q': self.data.target_q[i],
            'target_p': self.data.target_p[i],
            'target_t': self.data.target_t[i],
            'time_mask': self.data.time_mask[i],
            'time_valid': self.data.time_valid[i],
            'reward': self.data.reward[i],
        }

    def with_reward(self) -> Optional["L3TensorDataset"]:
        mask = self.data.reward_valid
        if mask.numel() == 0:
            return None
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            return None
        return L3TensorDataset(self.data, indices=indices.cpu().numpy())


class L4Dataset(Dataset):
    """L4 蒸馏数据集（变长线程集合）。"""

    def __init__(self, samples: List[L4DistillSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "global_state": torch.FloatTensor(np.asarray(s.global_state, dtype=np.float32)),
            "thread_states": torch.FloatTensor(np.asarray(s.thread_states, dtype=np.float32)),
            "teacher_alpha": torch.FloatTensor(np.asarray(s.teacher_alpha, dtype=np.float32)),
        }


def collate_l4(batch: List[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
    """将变长线程集合 padding 成固定形状，并生成 thread_mask。"""
    if not batch:
        raise ValueError("Empty batch")

    B = len(batch)
    max_k = max(int(item["thread_states"].shape[0]) for item in batch)
    d_thread = int(batch[0]["thread_states"].shape[1])
    d_global = int(batch[0]["global_state"].shape[0])

    global_state = torch.zeros((B, d_global), dtype=torch.float32)
    thread_states = torch.zeros((B, max_k, d_thread), dtype=torch.float32)
    teacher_alpha = torch.zeros((B, max_k), dtype=torch.float32)
    thread_mask = torch.zeros((B, max_k), dtype=torch.bool)

    for i, item in enumerate(batch):
        gs = item["global_state"]
        ts = item["thread_states"]
        ta = item["teacher_alpha"]
        k = int(ts.shape[0])
        global_state[i, : gs.shape[0]] = gs
        thread_states[i, :k, :] = ts
        teacher_alpha[i, :k] = ta[:k]
        thread_mask[i, :k] = True

    return {
        "global_state": global_state,
        "thread_states": thread_states,
        "teacher_alpha": teacher_alpha,
        "thread_mask": thread_mask,
    }


def _ensure_1d(x: Any, length: int, *, fill: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full((length,), fill, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size >= length:
        return arr[:length]
    pad = np.full((length - arr.size,), fill, dtype=np.float32)
    return np.concatenate([arr, pad])


def tensorize_l3_samples_chunk(
    samples: List[MicroSample],
    max_history_len: int = 20,
    horizon: int = 40,
) -> Dict[str, np.ndarray]:
    """将 L3 微观样本张量化为固定形状数组（单进程 chunk）。"""
    n = len(samples)
    tm_len = int(horizon) + 1
    history = np.zeros((n, max_history_len, 4), dtype=np.float32)
    context = np.zeros((n, 29), dtype=np.float32)
    action_op = np.zeros((n, 1), dtype=np.int64)
    target_q = np.zeros((n, 1), dtype=np.float32)
    target_p = np.zeros((n, 1), dtype=np.float32)
    target_t = np.zeros((n, 1), dtype=np.int64)
    time_mask = np.zeros((n, tm_len), dtype=np.float32)
    time_valid = np.zeros((n, 1), dtype=np.float32)
    reward = np.full((n, 1), np.nan, dtype=np.float32)
    reward_valid = np.zeros((n,), dtype=np.bool_)

    for i, s in enumerate(samples):
        hist = s.history if s is not None else None
        if hist is None:
            hist_arr = np.zeros((max_history_len, 4), dtype=np.float32)
        else:
            try:
                hist_arr = np.asarray(hist, dtype=np.float32)
                if hist_arr.ndim != 2:
                    hist_arr = np.zeros((max_history_len, 4), dtype=np.float32)
                else:
                    if hist_arr.shape[1] < 4:
                        pad_cols = 4 - hist_arr.shape[1]
                        hist_arr = np.concatenate(
                            [hist_arr, np.zeros((hist_arr.shape[0], pad_cols), dtype=np.float32)],
                            axis=1,
                        )
                    elif hist_arr.shape[1] > 4:
                        hist_arr = hist_arr[:, :4]
            except Exception:
                hist_arr = np.zeros((max_history_len, 4), dtype=np.float32)

            if hist_arr.shape[0] > max_history_len:
                hist_arr = hist_arr[-max_history_len:]
            elif hist_arr.shape[0] < max_history_len:
                pad = np.zeros((max_history_len - hist_arr.shape[0], 4), dtype=np.float32)
                hist_arr = np.vstack([pad, hist_arr])

            if hist_arr.ndim == 2 and hist_arr.shape[1] >= 3:
                hist_arr[:, 2] = np.nan_to_num(
                    hist_arr[:, 2],
                    nan=0.0,
                    posinf=float(horizon),
                    neginf=0.0,
                )
                hist_arr[:, 2] = np.clip(hist_arr[:, 2], 0.0, float(horizon))

        history[i] = hist_arr

        l2_goal = _ensure_1d(s.l2_goal if s is not None else None, 16)
        gap_buy = _ensure_1d(s.goal_gap_buy if s is not None else None, 4)
        gap_sell = _ensure_1d(s.goal_gap_sell if s is not None else None, 4)

        step_progress = 0.0
        if s is not None and s.x_static is not None and len(s.x_static) > 3:
            try:
                step_progress = float(s.x_static[3])
            except Exception:
                step_progress = 0.0
        relative_time = float(s.relative_time) if s is not None and s.relative_time is not None else 0.0
        is_buying = 1.0 if s is not None and int(s.role) == 0 else 0.0
        b_free = float(s.B_free) / 10000.0 if s is not None and s.B_free is not None else 0.0

        context[i] = np.concatenate(
            [
                l2_goal,
                gap_buy,
                gap_sell,
                np.array(
                    [step_progress, relative_time, 0.0, is_buying, b_free],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        tm = _ensure_1d(s.time_mask if s is not None else None, tm_len, fill=0.0)
        time_mask[i] = tm

        action_op[i, 0] = int(s.action_op) if s is not None else 2

        tq = float(s.target_quantity) if s is not None and s.target_quantity is not None else 0.0
        tp = float(s.target_price) if s is not None and s.target_price is not None else 0.0
        tt = int(s.target_time) if s is not None and s.target_time is not None else 0

        valid = 1.0
        if tt < 0 or tt >= tm_len:
            valid = 0.0
            tt = max(0, min(tt, tm_len - 1))
        elif tm_len > 0 and tm[tt] == -np.inf:
            valid = 0.0

        target_q[i, 0] = tq
        target_p[i, 0] = tp
        target_t[i, 0] = tt
        time_valid[i, 0] = valid

        if s is not None and s.reward is not None:
            reward[i, 0] = float(s.reward)
            reward_valid[i] = True

    return {
        "history": history,
        "context": context,
        "action_op": action_op,
        "target_q": target_q,
        "target_p": target_p,
        "target_t": target_t,
        "time_mask": time_mask,
        "time_valid": time_valid,
        "reward": reward,
        "reward_valid": reward_valid,
    }


def _dataloader_kwargs(config: TrainConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    num_workers = int(getattr(config, "dataloader_num_workers", 0) or 0)
    if num_workers > 0:
        kwargs["num_workers"] = num_workers
        if getattr(config, "dataloader_prefetch_factor", None) is not None:
            kwargs["prefetch_factor"] = int(config.dataloader_prefetch_factor)
        if getattr(config, "dataloader_persistent_workers", False):
            kwargs["persistent_workers"] = True
    if getattr(config, "dataloader_pin_memory", False):
        kwargs["pin_memory"] = True
    if getattr(config, "dataloader_drop_last", False):
        kwargs["drop_last"] = True
    return kwargs


def build_dataloader(
    dataset: "Dataset",
    *,
    batch_size: int,
    shuffle: bool,
    config: TrainConfig,
    collate_fn: Optional[Any] = None,
) -> "DataLoader":
    kwargs = _dataloader_kwargs(config)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


class _BatchProgress:
    def __init__(self, total: int, *, prefix: str, enabled: bool) -> None:
        self.total = max(0, int(total))
        self.prefix = f"{prefix} " if prefix else ""
        self.enabled = enabled
        self.width = 24
        self.current = 0
        self._done = False
        self._update_every = max(1, self.total // 100) if self.total > 0 else 1
        self._last_printed = 0
        self._start_time = time.monotonic()
        if self.enabled:
            self._render(done=self.total == 0)

    def step(self, n: int = 1) -> None:
        if not self.enabled or self._done:
            return
        self.current += int(n)
        if self.current >= self.total:
            self.current = self.total
            self._render(done=True)
            self._done = True
            return
        if self.current == 1 or (self.current - self._last_printed) >= self._update_every:
            self._render(done=False)
            self._last_printed = self.current

    def finish(self) -> None:
        if not self.enabled or self._done:
            return
        self.current = self.total
        self._render(done=True)
        self._done = True

    def _render(self, *, done: bool) -> None:
        if self.total <= 0:
            print(f"{self.prefix}[{'-' * self.width}] 0/0 elapsed --:--:-- ETA --:--:--")
            return
        filled = int(self.width * self.current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = (self.current / self.total) * 100.0
        elapsed = self._format_time(self._elapsed_seconds())
        eta = self._format_time(self._eta_seconds())
        end = "\n" if done else ""
        print(
            f"\r{self.prefix}[{bar}] {self.current}/{self.total} {percent:5.1f}% "
            f"elapsed {elapsed} ETA {eta}",
            end=end,
            flush=True,
        )

    def _elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self._start_time)

    def _eta_seconds(self) -> Optional[float]:
        if self.total <= 0:
            return None
        if self.current >= self.total:
            return 0.0
        if self.current <= 0:
            return None
        elapsed = self._elapsed_seconds()
        if elapsed <= 0:
            return None
        rate = self.current / elapsed
        if rate <= 0:
            return None
        remaining = max(0.0, float(self.total - self.current))
        return remaining / rate

    @staticmethod
    def _format_time(seconds: Optional[float]) -> str:
        if seconds is None:
            return "--:--:--"
        total = max(0, int(seconds))
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============== 行为克隆 (BC) ==============

def train_l2_bc(
    model: "nn.Module",
    samples: List[MacroSample],
    config: TrainConfig,
    resume_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """L2 行为克隆训练.
    
    Args:
        model: L2 模型
        samples: 宏观样本
        config: 训练配置
        resume_path: 断点路径（包含优化器状态）
        
    Returns:
        训练历史
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    dataset = L2Dataset(samples)
    loader = build_dataloader(dataset, batch_size=config.l2_batch_size, shuffle=True, config=config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.l2_lr,
        weight_decay=config.weight_decay
    )
    
    model.to(config.device)
    model.train()
    
    history = {'loss': [], 'mse': []}
    start_epoch = 0
    if resume_path:
        if os.path.exists(resume_path):
            start_epoch, ckpt_history = load_checkpoint(
                model, optimizer, resume_path, device=config.device
            )
            if ckpt_history:
                history = ckpt_history
        else:
            print(f"[WARN] Resume path not found: {resume_path} (start fresh)")
    
    if start_epoch >= config.l2_epochs:
        print(f"[INFO] Resume epoch {start_epoch} >= target {config.l2_epochs}, skip training")
        return history
    
    q_indices = [0, 2, 4, 6, 8, 10, 12, 14]
    p_indices = [1, 3, 5, 7, 9, 11, 13, 15]
    q_transform = (config.l2_q_transform or "none").lower()
    q_weight = float(config.l2_q_weight)
    if q_weight <= 0:
        raise ValueError("l2_q_weight 必须 > 0")

    def _apply_q_transform(x: "torch.Tensor") -> "torch.Tensor":
        if q_transform == "none":
            return x
        if q_transform == "log1p":
            return torch.log1p(torch.clamp(x, min=0.0))
        if q_transform == "sqrt":
            return torch.sqrt(torch.clamp(x, min=0.0))
        raise ValueError(f"未知的 l2_q_transform: {q_transform}")

    for epoch in range(start_epoch, config.l2_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0

        total_batches = len(loader)
        progress = _BatchProgress(
            total_batches,
            prefix=f"[L2 BC][Epoch {epoch + 1}/{config.l2_epochs}]",
            enabled=bool(config.progress_bar),
        )
        
        for step, batch in enumerate(loader, start=1):
            state_static = batch['state_static'].to(config.device)
            state_temporal = batch['state_temporal'].to(config.device)
            x_role = batch['x_role'].to(config.device)  # Multi-Hot: [can_buy, can_sell]
            goal_target = batch['goal'].to(config.device)
            
            # 前向
            goal_pred = model(state_static, state_temporal, x_role)
            
            # 如果模型返回 (mean, log_std, value)
            if isinstance(goal_pred, tuple):
                goal_pred = goal_pred[0]
            
            # 构建损失掩码：屏蔽不可谈判的分量
            # 16维 = 4桶 × (Q_buy, P_buy, Q_sell, P_sell)
            # can_buy=0 时屏蔽 Q_buy/P_buy (索引 0,1,4,5,8,9,12,13)
            # can_sell=0 时屏蔽 Q_sell/P_sell (索引 2,3,6,7,10,11,14,15)
            B = x_role.size(0)
            loss_mask = torch.ones(B, 16, device=config.device)
            
            can_buy = x_role[:, 0:1]   # (B, 1)
            can_sell = x_role[:, 1:2]  # (B, 1)
            
            # 每个桶的 Q_buy, P_buy 索引
            buy_indices = [0, 1, 4, 5, 8, 9, 12, 13]
            # 每个桶的 Q_sell, P_sell 索引
            sell_indices = [2, 3, 6, 7, 10, 11, 14, 15]
            
            for idx in buy_indices:
                loss_mask[:, idx] = can_buy.squeeze(-1)
            for idx in sell_indices:
                loss_mask[:, idx] = can_sell.squeeze(-1)
            
            # 加权 MSE 损失（只计算可谈判分量，Q 可做变换）
            err = goal_pred - goal_target
            if q_transform != "none":
                err = err.clone()
                err[:, q_indices] = _apply_q_transform(goal_pred[:, q_indices]) - _apply_q_transform(goal_target[:, q_indices])
            squared_error = err ** 2

            if q_weight != 1.0:
                weights = loss_mask.clone()
                weights[:, q_indices] = weights[:, q_indices] * q_weight
            else:
                weights = loss_mask

            masked_error = squared_error * weights
            # 归一化：除以有效分量数量（含权重）
            n_valid = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
            loss = (masked_error.sum(dim=1) / n_valid.squeeze(-1)).mean()
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            # 防止除零（极端情况：x_role=[0,0] 导致 loss_mask 全零）
            mask_sum = weights.sum()
            if mask_sum > 0:
                epoch_mse += (masked_error.sum() / mask_sum).item()
            n_batches += 1

            progress.step()
            if config.batch_log_every > 0:
                if step % config.batch_log_every == 0 or step == total_batches:
                    avg_loss = epoch_loss / max(1, n_batches)
                    avg_mse = epoch_mse / max(1, n_batches)
                    print(
                        f"[L2 BC][Epoch {epoch + 1}/{config.l2_epochs}] "
                        f"step {step}/{total_batches} loss={avg_loss:.4f} mse={avg_mse:.4f}"
                    )

        progress.finish()
        
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        
        if (epoch + 1) % config.log_every == 0:
            print(
                f"[L2 BC] Epoch {epoch + 1}/{config.l2_epochs} | "
                f"Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f}"
            )
        
        if (epoch + 1) % config.save_every == 0:
            save_model(model, config.output_dir, f"l2_bc_epoch{epoch + 1}.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                config.output_dir,
                f"l2_bc_epoch{epoch + 1}.ckpt.pt",
                history=history,
            )
    
    return history


def _resolve_l3_dataset(
    samples: Any,
    config: TrainConfig,
    *,
    require_reward: bool = False,
) -> Optional["Dataset"]:
    if isinstance(samples, L3TensorDataset):
        dataset: Dataset = samples
    elif isinstance(samples, L3TensorizedData):
        dataset = L3TensorDataset(samples)
    elif isinstance(samples, Dataset):
        dataset = samples
    else:
        if require_reward:
            samples = [s for s in samples if s.reward is not None]
            if not samples:
                return None
        dataset = L3Dataset(samples, horizon=config.horizon, max_history_len=config.l3_max_history_len)

    if require_reward and isinstance(dataset, L3TensorDataset):
        dataset = dataset.with_reward()
        if dataset is None:
            return None

    return dataset


def train_l3_bc(
    model: "nn.Module",
    samples: Any,
    config: TrainConfig,
    resume_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """L3 行为克隆训练.
    
    Args:
        model: L3 模型
        samples: 微观样本
        config: 训练配置（包含 horizon 参数）
        resume_path: 断点路径（包含优化器状态）
        
    Returns:
        训练历史
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    dataset = _resolve_l3_dataset(samples, config, require_reward=False)
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("No micro samples found for L3 training")
    loader = build_dataloader(dataset, batch_size=config.l3_batch_size, shuffle=True, config=config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.l3_lr,
        weight_decay=config.weight_decay
    )
    
    model.to(config.device)
    model.train()
    
    history = {'loss': [], 'loss_q': [], 'loss_p': [], 'loss_t': []}
    start_epoch = 0
    if resume_path:
        if os.path.exists(resume_path):
            start_epoch, ckpt_history = load_checkpoint(
                model, optimizer, resume_path, device=config.device
            )
            if ckpt_history:
                history = ckpt_history
        else:
            print(f"[WARN] Resume path not found: {resume_path} (start fresh)")
    
    if start_epoch >= config.l3_epochs:
        print(f"[INFO] Resume epoch {start_epoch} >= target {config.l3_epochs}, skip training")
        return history
    
    for epoch in range(start_epoch, config.l3_epochs):
        epoch_loss = 0.0
        epoch_loss_q = 0.0
        epoch_loss_p = 0.0
        epoch_loss_t = 0.0
        n_batches = 0
        op_correct = 0
        op_total = 0

        total_batches = len(loader)
        progress = _BatchProgress(
            total_batches,
            prefix=f"[L3 BC][Epoch {epoch + 1}/{config.l3_epochs}]",
            enabled=bool(config.progress_bar),
        )

        for step, batch in enumerate(loader, start=1):
            history_seq = batch['history'].to(config.device)
            context = batch['context'].to(config.device)
            action_op = batch['action_op'].squeeze(-1).to(config.device)
            target_q = batch['target_q'].to(config.device)
            target_p = batch['target_p'].to(config.device)
            target_t = batch['target_t'].squeeze(-1).to(config.device)
            time_mask = batch['time_mask'].to(config.device)
            time_valid = batch['time_valid'].squeeze(-1).to(config.device)

            op_logits, quantity, price, time_logits = model(
                history_seq, context, time_mask
            )

            loss_op = F.cross_entropy(op_logits, action_op)

            counter_mask = action_op == 1
            valid_mask = counter_mask & (time_valid > 0.5)
            if valid_mask.any():
                loss_q = F.mse_loss(quantity[valid_mask], target_q[valid_mask])
                loss_p = F.mse_loss(price[valid_mask], target_p[valid_mask])
                loss_t = F.cross_entropy(time_logits[valid_mask], target_t[valid_mask])
            else:
                loss_q = torch.tensor(0.0, device=config.device)
                loss_p = torch.tensor(0.0, device=config.device)
                loss_t = torch.tensor(0.0, device=config.device)

            loss = loss_op + loss_q + loss_p + loss_t
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_loss_q += loss_q.item()
            epoch_loss_p += loss_p.item()
            epoch_loss_t += loss_t.item()
            n_batches += 1

            preds = torch.argmax(op_logits, dim=1)
            op_correct += int((preds == action_op).sum().item())
            op_total += int(action_op.numel())

            progress.step()
            if config.batch_log_every > 0:
                if step % config.batch_log_every == 0 or step == total_batches:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    avg_q = epoch_loss_q / max(n_batches, 1)
                    avg_p = epoch_loss_p / max(n_batches, 1)
                    avg_t = epoch_loss_t / max(n_batches, 1)
                    acc = op_correct / max(1, op_total)
                    print(
                        f"[L3 BC][Epoch {epoch + 1}/{config.l3_epochs}] "
                        f"step {step}/{total_batches} loss={avg_loss:.4f} "
                        f"(q={avg_q:.4f}, p={avg_p:.4f}, t={avg_t:.4f}) op_acc={acc:.3f}"
                    )

        progress.finish()
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['loss_q'].append(epoch_loss_q / max(n_batches, 1))
        history['loss_p'].append(epoch_loss_p / max(n_batches, 1))
        history['loss_t'].append(epoch_loss_t / max(n_batches, 1))
        
        if (epoch + 1) % config.log_every == 0:
            acc = op_correct / max(1, op_total)
            print(
                f"[L3 BC] Epoch {epoch + 1}/{config.l3_epochs} | "
                f"Loss: {avg_loss:.4f} (q={history['loss_q'][-1]:.4f}, "
                f"p={history['loss_p'][-1]:.4f}, t={history['loss_t'][-1]:.4f}) "
                f"op_acc={acc:.3f}"
            )
        
        if (epoch + 1) % config.save_every == 0:
            save_model(model, config.output_dir, f"l3_bc_epoch{epoch + 1}.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                config.output_dir,
                f"l3_bc_epoch{epoch + 1}.ckpt.pt",
                history=history,
            )
    
    return history


# ============== 优势加权回归 (AWR) ==============

def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.98,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 GAE 优势和回报.
    
    Args:
        rewards: 奖励序列
        values: 价值估计序列
        gamma: 折扣因子
        gae_lambda: GAE lambda
        
    Returns:
        (advantages, returns)
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    
    returns = advantages + values
    
    return advantages, returns


def train_l3_awr(
    model: "nn.Module",
    samples: Any,
    config: TrainConfig,
    resume_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """L3 优势加权回归训练.
    
    Args:
        model: L3 模型
        samples: 带奖励的微观样本
        config: 训练配置（包含 horizon 参数）
        resume_path: 断点路径（包含优化器状态）
        
    Returns:
        训练历史
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    dataset = _resolve_l3_dataset(samples, config, require_reward=True)
    if dataset is None or len(dataset) == 0:
        print("[WARN] No samples with reward, falling back to BC")
        return train_l3_bc(model, samples, config, resume_path=resume_path)

    loader = build_dataloader(dataset, batch_size=config.l3_batch_size, shuffle=True, config=config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.l3_lr,
        weight_decay=config.weight_decay
    )
    
    model.to(config.device)
    model.train()
    
    history = {'loss': [], 'weighted_loss': []}
    start_epoch = 0
    if resume_path:
        if os.path.exists(resume_path):
            start_epoch, ckpt_history = load_checkpoint(
                model, optimizer, resume_path, device=config.device
            )
            if ckpt_history:
                history = ckpt_history
        else:
            print(f"[WARN] Resume path not found: {resume_path} (start fresh)")
    
    if start_epoch >= config.l3_epochs:
        print(f"[INFO] Resume epoch {start_epoch} >= target {config.l3_epochs}, skip training")
        return history
    
    for epoch in range(start_epoch, config.l3_epochs):
        epoch_loss = 0.0
        epoch_loss_q = 0.0
        epoch_loss_p = 0.0
        epoch_loss_t = 0.0
        n_batches = 0
        op_correct = 0
        op_total = 0

        total_batches = len(loader)
        progress = _BatchProgress(
            total_batches,
            prefix=f"[L3 AWR][Epoch {epoch + 1}/{config.l3_epochs}]",
            enabled=bool(config.progress_bar),
        )

        for step, batch in enumerate(loader, start=1):
            history_seq = batch['history'].to(config.device)
            context = batch['context'].to(config.device)
            action_op = batch['action_op'].squeeze(-1).to(config.device)
            target_q = batch['target_q'].to(config.device)
            target_p = batch['target_p'].to(config.device)
            target_t = batch['target_t'].squeeze(-1).to(config.device)
            rewards = batch['reward'].squeeze(-1).to(config.device)
            time_mask = batch['time_mask'].to(config.device)
            time_valid = batch['time_valid'].squeeze(-1).to(config.device)

            B = history_seq.size(0)
            advantages = rewards
            weights = torch.exp(advantages / config.awr_beta)
            weights = torch.clamp(weights, max=config.awr_clip)
            weights = weights / weights.sum() * B

            op_logits, quantity, price, time_logits = model(
                history_seq, context, time_mask
            )

            loss_op = F.cross_entropy(op_logits, action_op, reduction='none')
            loss_op = (weights * loss_op).mean()

            counter_mask = action_op == 1
            valid_mask = counter_mask & (time_valid > 0.5)
            if valid_mask.any():
                w_counter = weights[valid_mask].unsqueeze(-1)
                loss_q = (w_counter * (quantity[valid_mask] - target_q[valid_mask]) ** 2).mean()
                loss_p = (w_counter * (price[valid_mask] - target_p[valid_mask]) ** 2).mean()
                loss_t = F.cross_entropy(time_logits[valid_mask], target_t[valid_mask], reduction='none')
                loss_t = (weights[valid_mask] * loss_t).mean()
            else:
                loss_q = torch.tensor(0.0, device=config.device)
                loss_p = torch.tensor(0.0, device=config.device)
                loss_t = torch.tensor(0.0, device=config.device)

            loss = loss_op + loss_q + loss_p + loss_t
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_loss_q += loss_q.item()
            epoch_loss_p += loss_p.item()
            epoch_loss_t += loss_t.item()
            n_batches += 1

            preds = torch.argmax(op_logits, dim=1)
            op_correct += int((preds == action_op).sum().item())
            op_total += int(action_op.numel())

            progress.step()
            if config.batch_log_every > 0:
                if step % config.batch_log_every == 0 or step == total_batches:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    avg_q = epoch_loss_q / max(n_batches, 1)
                    avg_p = epoch_loss_p / max(n_batches, 1)
                    avg_t = epoch_loss_t / max(n_batches, 1)
                    acc = op_correct / max(1, op_total)
                    print(
                        f"[L3 AWR][Epoch {epoch + 1}/{config.l3_epochs}] "
                        f"step {step}/{total_batches} loss={avg_loss:.4f} "
                        f"(q={avg_q:.4f}, p={avg_p:.4f}, t={avg_t:.4f}) op_acc={acc:.3f}"
                    )

        progress.finish()
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['weighted_loss'].append(avg_loss)
        
        if (epoch + 1) % config.log_every == 0:
            acc = op_correct / max(1, op_total)
            print(
                f"[L3 AWR] Epoch {epoch + 1}/{config.l3_epochs} | "
                f"Loss: {avg_loss:.4f} (q={epoch_loss_q / max(n_batches, 1):.4f}, "
                f"p={epoch_loss_p / max(n_batches, 1):.4f}, "
                f"t={epoch_loss_t / max(n_batches, 1):.4f}) op_acc={acc:.3f}"
            )
        
        if (epoch + 1) % config.save_every == 0:
            save_model(model, config.output_dir, f"l3_awr_epoch{epoch + 1}.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                config.output_dir,
                f"l3_awr_epoch{epoch + 1}.ckpt.pt",
                history=history,
            )
    
    return history


def train_l4_distill(
    model: "nn.Module",
    samples: List[L4DistillSample],
    config: TrainConfig,
    resume_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """L4 蒸馏训练（监督：拟合启发式 soft α）。"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    dataset = L4Dataset(samples)
    loader = build_dataloader(
        dataset,
        batch_size=config.l4_batch_size,
        shuffle=True,
        config=config,
        collate_fn=collate_l4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.l4_lr,
        weight_decay=config.weight_decay,
    )

    model.to(config.device)
    model.train()

    history = {"loss": [], "ce": []}
    start_epoch = 0
    if resume_path:
        if os.path.exists(resume_path):
            start_epoch, ckpt_history = load_checkpoint(model, optimizer, resume_path, device=config.device)
            if ckpt_history:
                history = ckpt_history
        else:
            print(f"[WARN] Resume path not found: {resume_path} (start fresh)")

    if start_epoch >= config.l4_epochs:
        print(f"[INFO] Resume epoch {start_epoch} >= target {config.l4_epochs}, skip training")
        return history

    for epoch in range(start_epoch, config.l4_epochs):
        epoch_loss = 0.0
        n_batches = 0

        total_batches = len(loader)
        progress = _BatchProgress(
            total_batches,
            prefix=f"[L4 Distill][Epoch {epoch + 1}/{config.l4_epochs}]",
            enabled=bool(config.progress_bar),
        )

        for step, batch in enumerate(loader, start=1):
            global_state = batch["global_state"].to(config.device)
            thread_states = batch["thread_states"].to(config.device)
            teacher = batch["teacher_alpha"].to(config.device)
            mask = batch["thread_mask"].to(config.device)

            pred = model(thread_states, global_state, thread_mask=mask)

            diff = pred - teacher
            loss = (diff * diff * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

            progress.step()
            if config.batch_log_every > 0:
                if step % config.batch_log_every == 0 or step == total_batches:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    print(
                        f"[L4 Distill][Epoch {epoch + 1}/{config.l4_epochs}] "
                        f"step {step}/{total_batches} loss={avg_loss:.4f}"
                    )

        progress.finish()

        epoch_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(epoch_loss)
        history["ce"].append(epoch_loss)

        if (epoch + 1) % config.log_every == 0:
            print(f"[L4 Distill] Epoch {epoch+1}/{config.l4_epochs} | Loss: {epoch_loss:.6f}")

        if (epoch + 1) % config.save_every == 0:
            ckpt_name = f"l4_distill_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch + 1, config.output_dir, ckpt_name, history=history)

    return history


# ============== PPO (预留) ==============

class PPOBuffer:
    """PPO 经验缓冲区."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


def ppo_loss(
    log_probs_new: "torch.Tensor",
    log_probs_old: "torch.Tensor",
    advantages: "torch.Tensor",
    values_new: "torch.Tensor",
    returns: "torch.Tensor",
    clip_eps: float = 0.2
) -> "torch.Tensor":
    """计算 PPO 损失.
    
    Args:
        log_probs_new: 新策略对数概率
        log_probs_old: 旧策略对数概率
        advantages: 优势估计
        values_new: 新价值估计
        returns: 回报
        clip_eps: 裁剪参数
        
    Returns:
        总损失
    """
    # 策略损失
    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 价值损失
    value_loss = F.mse_loss(values_new, returns)
    
    return policy_loss + 0.5 * value_loss


# ============== 模型保存/加载 ==============

def _move_optimizer_state_to_device(optimizer: "torch.optim.Optimizer", device: str) -> None:
    """将优化器状态迁移到指定设备."""
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_model(model: "nn.Module", output_dir: str, filename: str) -> str:
    """保存模型.
    
    Args:
        model: PyTorch 模型
        output_dir: 输出目录
        filename: 文件名
        
    Returns:
        保存路径
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"[SAVE] Model saved to {path}")
    return path


def save_checkpoint(
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    output_dir: str,
    filename: str,
    history: Optional[Dict[str, List[float]]] = None,
) -> str:
    """保存断点（模型 + 优化器 + 进度）."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "history": history,
    }
    torch.save(state, path)
    print(f"[SAVE] Checkpoint saved to {path}")
    return path


def load_model(model: "nn.Module", path: str) -> "nn.Module":
    """加载模型.
    
    Args:
        model: 模型实例
        path: 检查点路径
        
    Returns:
        加载后的模型
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"[LOAD] Model loaded from {path}")
    return model


def load_checkpoint(
    model: "nn.Module",
    optimizer: Optional["torch.optim.Optimizer"],
    path: str,
    device: str = "cpu",
) -> Tuple[int, Optional[Dict[str, List[float]]]]:
    """加载断点（模型 + 优化器 + 进度）."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
        if optimizer is not None and state.get("optimizer_state") is not None:
            optimizer.load_state_dict(state["optimizer_state"])
            _move_optimizer_state_to_device(optimizer, device)
        epoch = int(state.get("epoch", 0))
        history = state.get("history")
        print(f"[LOAD] Checkpoint loaded from {path} (epoch={epoch})")
        return epoch, history
    
    # 兼容仅保存了 state_dict 的旧格式
    model.load_state_dict(state)
    print(f"[LOAD] Model state loaded from {path} (no optimizer state)")
    return 0, None


# ============== 完整训练流程 ==============

class HRLXFTrainer:
    """HRL-XF 训练器.
    
    集成 L2、L3、L4 的训练流程。
    """
    
    def __init__(
        self,
        l2_model: Optional["nn.Module"] = None,
        l3_model: Optional["nn.Module"] = None,
        l4_model: Optional["nn.Module"] = None,
        config: Optional[TrainConfig] = None
    ):
        self.l2 = l2_model
        self.l3 = l3_model
        self.l4 = l4_model
        self.config = config or TrainConfig()
    
    def train_phase0_bc(
        self,
        macro_samples: List[MacroSample],
        micro_samples: List[MicroSample]
    ) -> Dict[str, Any]:
        """Phase 0: 行为克隆冷启动."""
        results = {}
        
        if self.l2 is not None and macro_samples:
            print("=" * 50)
            print("Phase 0: L2 Behavior Cloning")
            print("=" * 50)
            results['l2_bc'] = train_l2_bc(
                self.l2,
                macro_samples,
                self.config,
                resume_path=self.config.l2_resume_path,
            )
        
        if self.l3 is not None and micro_samples:
            print("=" * 50)
            print("Phase 0: L3 Behavior Cloning")
            print("=" * 50)
            results['l3_bc'] = train_l3_bc(
                self.l3,
                micro_samples,
                self.config,
                resume_path=self.config.l3_bc_resume_path,
            )
        
        return results
    
    def train_phase1_awr(
        self,
        micro_samples: List[MicroSample]
    ) -> Dict[str, Any]:
        """Phase 1: 优势加权回归离线 RL."""
        results = {}
        
        if self.l3 is not None and micro_samples:
            print("=" * 50)
            print("Phase 1: L3 Advantage-Weighted Regression")
            print("=" * 50)
            results['l3_awr'] = train_l3_awr(
                self.l3,
                micro_samples,
                self.config,
                resume_path=self.config.l3_awr_resume_path,
            )
        
        return results

    def train_l4_distill(self, l4_samples: List[L4DistillSample]) -> Dict[str, Any]:
        """监督蒸馏 L4（拟合启发式 soft α）。"""
        results: Dict[str, Any] = {}
        if self.l4 is None or not l4_samples:
            return results
        print("=" * 50)
        print("L4 Distillation (Supervised)")
        print("=" * 50)
        results["l4_distill"] = train_l4_distill(
            self.l4,
            l4_samples,
            self.config,
            resume_path=self.config.l4_resume_path,
        )
        return results
    
    def train_phase2_ppo(self) -> Dict[str, Any]:
        """Phase 2: PPO 在线微调（预留）."""
        print("[INFO] Phase 2 PPO training requires SCML simulator integration")
        return {}
    
    def train_phase3_selfplay(self) -> Dict[str, Any]:
        """Phase 3: 自博弈（预留）."""
        print("[INFO] Phase 3 Self-Play training requires opponent pool")
        return {}
    
    def save_all(self, suffix: str = "final"):
        """保存所有模型."""
        if self.l2 is not None:
            save_model(self.l2, self.config.output_dir, f"l2_{suffix}.pt")
        if self.l3 is not None:
            save_model(self.l3, self.config.output_dir, f"l3_{suffix}.pt")
        if self.l4 is not None:
            save_model(self.l4, self.config.output_dir, f"l4_{suffix}.pt")


__all__ = [
    "TrainConfig",
    "L2Dataset",
    "L3Dataset",
    "L3TensorDataset",
    "L3TensorizedData",
    "train_l2_bc",
    "train_l3_bc",
    "train_l3_awr",
    "L4Dataset",
    "collate_l4",
    "tensorize_l3_samples_chunk",
    "build_dataloader",
    "train_l4_distill",
    "compute_advantages",
    "ppo_loss",
    "PPOBuffer",
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    "HRLXFTrainer",
]
