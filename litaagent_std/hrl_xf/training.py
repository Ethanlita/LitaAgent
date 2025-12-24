"""训练脚本 - BC、AWR 预训练与 PPO 微调.

实现四阶段课程学习：
- Phase 0: Cold Start - 行为克隆 (BC)
- Phase 1: Offline RL - 优势加权回归 (AWR)
- Phase 2: Online Fine-tune - PPO + 势能奖励
- Phase 3: Self-Play - 自博弈（预留）
"""

from __future__ import annotations

import os
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
    
    # L3 训练
    l3_lr: float = 1e-4
    l3_batch_size: int = 32
    l3_epochs: int = 100

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
    """L3 微观数据集."""
    
    def __init__(self, samples: List[MicroSample], max_history_len: int = 20, horizon: int = 40):
        self.samples = samples
        self.max_len = max_history_len
        self.horizon = horizon  # 用于动态确定 time_mask 长度
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 填充/截断历史
        history = s.history
        if len(history) > self.max_len:
            history = history[-self.max_len:]
        elif len(history) < self.max_len:
            pad = np.zeros((self.max_len - len(history), 3))
            history = np.vstack([pad, history])

        # 重要：delta_t 作为离散时间索引，必须落在 [0, horizon]，否则会导致 time_embed 越界
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
            # 极端情况下回退为全零，避免训练直接崩溃
            history = np.zeros((self.max_len, 3), dtype=np.float32)
        
        # time_mask 动态长度：使用 horizon+1，若缺失则默认全部允许 (0.0)
        tm_len = self.horizon + 1
        if s.time_mask is not None:
            time_mask = s.time_mask
            # 如果长度不匹配，截断或填充
            if len(time_mask) > tm_len:
                time_mask = time_mask[:tm_len]
            elif len(time_mask) < tm_len:
                time_mask = np.concatenate([time_mask, np.zeros(tm_len - len(time_mask))])
        else:
            time_mask = np.zeros(tm_len, dtype=np.float32)
        
        return {
            'history': torch.FloatTensor(history),
            'role': torch.LongTensor([s.role]),
            'goal': torch.FloatTensor(s.goal),
            'baseline': torch.FloatTensor(s.baseline),
            'residual': torch.FloatTensor(s.residual),
            'time_label': torch.LongTensor([s.time_label]),
            'time_mask': torch.FloatTensor(time_mask),
            'reward': torch.FloatTensor([s.reward if s.reward else 0.0]),
        }


class L4Dataset(Dataset):
    """L4 蒸馏数据集（变长线程集合）。"""

    def __init__(self, samples: List[L4DistillSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "global_feat": torch.FloatTensor(np.asarray(s.global_feat, dtype=np.float32)),
            "thread_feats": torch.FloatTensor(np.asarray(s.thread_feats, dtype=np.float32)),
            "thread_times": torch.LongTensor(np.asarray(s.thread_times, dtype=np.int64)),
            "thread_roles": torch.LongTensor(np.asarray(s.thread_roles, dtype=np.int64)),
            "teacher_weights": torch.FloatTensor(np.asarray(s.teacher_weights, dtype=np.float32)),
        }


def collate_l4(batch: List[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
    """将变长线程集合 padding 成固定形状，并生成 thread_mask。"""
    if not batch:
        raise ValueError("Empty batch")

    B = len(batch)
    max_k = max(int(item["thread_feats"].shape[0]) for item in batch)
    d_thread = int(batch[0]["thread_feats"].shape[1])
    d_global = int(batch[0]["global_feat"].shape[0])

    global_feat = torch.zeros((B, d_global), dtype=torch.float32)
    thread_feats = torch.zeros((B, max_k, d_thread), dtype=torch.float32)
    thread_times = torch.zeros((B, max_k), dtype=torch.long)
    thread_roles = torch.zeros((B, max_k), dtype=torch.long)
    teacher_weights = torch.zeros((B, max_k), dtype=torch.float32)
    thread_mask = torch.zeros((B, max_k), dtype=torch.bool)

    for i, item in enumerate(batch):
        gf = item["global_feat"]
        tf = item["thread_feats"]
        tt = item["thread_times"]
        tr = item["thread_roles"]
        tw = item["teacher_weights"]

        k = int(tf.shape[0])
        global_feat[i, : gf.shape[0]] = gf
        thread_feats[i, :k, :] = tf
        thread_times[i, :k] = tt[:k]
        thread_roles[i, :k] = tr[:k]
        teacher_weights[i, :k] = tw[:k]
        thread_mask[i, :k] = True

    # 归一化 teacher（防止数值误差导致 sum!=1）
    denom = teacher_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    teacher_weights = teacher_weights / denom

    return {
        "global_feat": global_feat,
        "thread_feats": thread_feats,
        "thread_times": thread_times,
        "thread_roles": thread_roles,
        "teacher_weights": teacher_weights,
        "thread_mask": thread_mask,
    }


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
    loader = DataLoader(dataset, batch_size=config.l2_batch_size, shuffle=True)
    
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
    
    for epoch in range(start_epoch, config.l2_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0
        
        for batch in loader:
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
            
            # 加权 MSE 损失（只计算可谈判分量）
            squared_error = (goal_pred - goal_target) ** 2  # (B, 16)
            masked_error = squared_error * loss_mask
            # 归一化：除以有效分量数量
            n_valid = loss_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            loss = (masked_error.sum(dim=1) / n_valid.squeeze(-1)).mean()
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            # 防止除零（极端情况：x_role=[0,0] 导致 loss_mask 全零）
            mask_sum = loss_mask.sum()
            if mask_sum > 0:
                epoch_mse += (masked_error.sum() / mask_sum).item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        
        if (epoch + 1) % config.log_every == 0:
            print(f"[L2 BC] Epoch {epoch + 1}/{config.l2_epochs} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f}")
        
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


def train_l3_bc(
    model: "nn.Module",
    samples: List[MicroSample],
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
    
    dataset = L3Dataset(samples, horizon=config.horizon)
    loader = DataLoader(dataset, batch_size=config.l3_batch_size, shuffle=True)
    
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
        
        for batch in loader:
            history_seq = batch['history'].to(config.device)
            role = batch['role'].squeeze(-1).to(config.device)
            goal = batch['goal'].to(config.device)
            baseline = batch['baseline'].to(config.device)
            residual_target = batch['residual'].to(config.device)
            time_target = batch['time_label'].squeeze(-1).to(config.device)
            time_mask = batch['time_mask'].to(config.device)  # 使用真实的 L1 time_mask
            
            # 前向
            delta_q, delta_p, time_logits = model(
                history_seq, goal, role, time_mask, baseline
            )
            
            # 损失
            loss_q = F.mse_loss(delta_q, residual_target[:, 0:1])
            loss_p = F.mse_loss(delta_p, residual_target[:, 1:2])
            loss_t = F.cross_entropy(time_logits, time_target)
            
            loss = loss_q + loss_p + loss_t
            
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
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['loss_q'].append(epoch_loss_q / max(n_batches, 1))
        history['loss_p'].append(epoch_loss_p / max(n_batches, 1))
        history['loss_t'].append(epoch_loss_t / max(n_batches, 1))
        
        if (epoch + 1) % config.log_every == 0:
            print(f"[L3 BC] Epoch {epoch + 1}/{config.l3_epochs} | Loss: {avg_loss:.4f}")
        
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
    samples: List[MicroSample],
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
    
    # 筛选有奖励的样本
    samples_with_reward = [s for s in samples if s.reward is not None]
    if not samples_with_reward:
        print("[WARN] No samples with reward, falling back to BC")
        return train_l3_bc(model, samples, config, resume_path=resume_path)
    
    dataset = L3Dataset(samples_with_reward, horizon=config.horizon)
    loader = DataLoader(dataset, batch_size=config.l3_batch_size, shuffle=True)
    
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
        n_batches = 0
        
        for batch in loader:
            history_seq = batch['history'].to(config.device)
            role = batch['role'].squeeze(-1).to(config.device)
            goal = batch['goal'].to(config.device)
            baseline = batch['baseline'].to(config.device)
            residual_target = batch['residual'].to(config.device)
            time_target = batch['time_label'].squeeze(-1).to(config.device)
            rewards = batch['reward'].squeeze(-1).to(config.device)
            time_mask = batch['time_mask'].to(config.device)  # 使用真实的 L1 time_mask
            
            # 计算优势权重
            # 简化版：使用 exp(reward / beta) 作为权重
            B = history_seq.size(0)
            advantages = rewards  # 简化
            weights = torch.exp(advantages / config.awr_beta)
            weights = torch.clamp(weights, max=config.awr_clip)
            weights = weights / weights.sum() * B  # 归一化
            weights_resid = weights.unsqueeze(-1)
            
            # 前向
            delta_q, delta_p, time_logits = model(
                history_seq, goal, role, time_mask, baseline
            )
            
            # 加权损失
            loss_q = (weights_resid * (delta_q - residual_target[:, 0:1]) ** 2).mean()
            loss_p = (weights_resid * (delta_p - residual_target[:, 1:2]) ** 2).mean()
            loss_t = F.cross_entropy(time_logits, time_target, reduction='none')
            loss_t = (weights * loss_t).mean()
            
            loss = loss_q + loss_p + loss_t
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['weighted_loss'].append(avg_loss)
        
        if (epoch + 1) % config.log_every == 0:
            print(f"[L3 AWR] Epoch {epoch + 1}/{config.l3_epochs} | Loss: {avg_loss:.4f}")
        
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
    loader = DataLoader(
        dataset,
        batch_size=config.l4_batch_size,
        shuffle=True,
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

        for batch in loader:
            global_feat = batch["global_feat"].to(config.device)
            thread_feats = batch["thread_feats"].to(config.device)
            thread_times = batch["thread_times"].to(config.device)
            thread_roles = batch["thread_roles"].to(config.device)
            teacher = batch["teacher_weights"].to(config.device)
            mask = batch["thread_mask"].to(config.device)

            pred, _ = model(thread_feats, thread_times, thread_roles, global_feat, thread_mask=mask)
            pred = pred.clamp_min(1e-8)

            ce = -(teacher * torch.log(pred)).sum(dim=-1)
            loss = ce.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(epoch_loss)
        history["ce"].append(epoch_loss)

        if (epoch + 1) % config.log_every == 0:
            print(f"[L4][Epoch {epoch+1}/{config.l4_epochs}] loss={epoch_loss:.6f}")

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
    "train_l2_bc",
    "train_l3_bc",
    "train_l3_awr",
    "L4Dataset",
    "collate_l4",
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
