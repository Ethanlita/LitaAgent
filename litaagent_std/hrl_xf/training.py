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

from .data_pipeline import MacroSample, MicroSample


# ============== 配置 ==============

@dataclass
class TrainConfig:
    """训练配置."""
    
    # 数据
    data_dir: str = "./data"
    output_dir: str = "./checkpoints"
    
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
    
    # PPO
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    gamma: float = 0.98
    gae_lambda: float = 0.95
    
    # AWR
    awr_beta: float = 1.0
    awr_clip: float = 20.0
    
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
            'goal': torch.FloatTensor(s.goal),
            'value': torch.FloatTensor([s.value if s.value else 0.0]),
        }


class L3Dataset(Dataset):
    """L3 微观数据集."""
    
    def __init__(self, samples: List[MicroSample], max_history_len: int = 20):
        self.samples = samples
        self.max_len = max_history_len
    
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
        
        return {
            'history': torch.FloatTensor(history),
            'role': torch.LongTensor([s.role]),
            'goal': torch.FloatTensor(s.goal),
            'baseline': torch.FloatTensor(s.baseline),
            'residual': torch.FloatTensor(s.residual),
            'time_label': torch.LongTensor([s.time_label]),
            'reward': torch.FloatTensor([s.reward if s.reward else 0.0]),
        }


# ============== 行为克隆 (BC) ==============

def train_l2_bc(
    model: "nn.Module",
    samples: List[MacroSample],
    config: TrainConfig
) -> Dict[str, List[float]]:
    """L2 行为克隆训练.
    
    Args:
        model: L2 模型
        samples: 宏观样本
        config: 训练配置
        
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
    
    for epoch in range(config.l2_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0
        
        for batch in loader:
            state_static = batch['state_static'].to(config.device)
            state_temporal = batch['state_temporal'].to(config.device)
            goal_target = batch['goal'].to(config.device)
            
            # 前向
            goal_pred = model(state_static, state_temporal)
            
            # 如果模型返回 (mean, log_std, value)
            if isinstance(goal_pred, tuple):
                goal_pred = goal_pred[0]
            
            # MSE 损失
            loss = F.mse_loss(goal_pred, goal_target)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += F.mse_loss(goal_pred, goal_target, reduction='mean').item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        history['loss'].append(avg_loss)
        history['mse'].append(avg_mse)
        
        if (epoch + 1) % config.log_every == 0:
            print(f"[L2 BC] Epoch {epoch + 1}/{config.l2_epochs} | Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f}")
        
        if (epoch + 1) % config.save_every == 0:
            save_model(model, config.output_dir, f"l2_bc_epoch{epoch + 1}.pt")
    
    return history


def train_l3_bc(
    model: "nn.Module",
    samples: List[MicroSample],
    config: TrainConfig,
    horizon: int = 40
) -> Dict[str, List[float]]:
    """L3 行为克隆训练.
    
    Args:
        model: L3 模型
        samples: 微观样本
        config: 训练配置
        horizon: 规划视界
        
    Returns:
        训练历史
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    dataset = L3Dataset(samples)
    loader = DataLoader(dataset, batch_size=config.l3_batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.l3_lr,
        weight_decay=config.weight_decay
    )
    
    model.to(config.device)
    model.train()
    
    history = {'loss': [], 'loss_q': [], 'loss_p': [], 'loss_t': []}
    
    for epoch in range(config.l3_epochs):
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
            
            # 时间掩码 (这里简化为全 1)
            B = history_seq.size(0)
            time_mask = torch.ones(B, horizon + 1, device=config.device)
            
            # 前向
            delta_q, delta_p, time_logits = model(
                history_seq, goal, role, time_mask, baseline
            )
            
            # 损失
            loss_q = F.mse_loss(delta_q, residual_target[:, 0])
            loss_p = F.mse_loss(delta_p, residual_target[:, 1])
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
    horizon: int = 40
) -> Dict[str, List[float]]:
    """L3 优势加权回归训练.
    
    Args:
        model: L3 模型
        samples: 带奖励的微观样本
        config: 训练配置
        horizon: 规划视界
        
    Returns:
        训练历史
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")
    
    # 筛选有奖励的样本
    samples_with_reward = [s for s in samples if s.reward is not None]
    if not samples_with_reward:
        print("[WARN] No samples with reward, falling back to BC")
        return train_l3_bc(model, samples, config, horizon)
    
    dataset = L3Dataset(samples_with_reward)
    loader = DataLoader(dataset, batch_size=config.l3_batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.l3_lr,
        weight_decay=config.weight_decay
    )
    
    model.to(config.device)
    model.train()
    
    history = {'loss': [], 'weighted_loss': []}
    
    for epoch in range(config.l3_epochs):
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
            
            B = history_seq.size(0)
            time_mask = torch.ones(B, horizon + 1, device=config.device)
            
            # 计算优势权重
            # 简化版：使用 exp(reward / beta) 作为权重
            advantages = rewards  # 简化
            weights = torch.exp(advantages / config.awr_beta)
            weights = torch.clamp(weights, max=config.awr_clip)
            weights = weights / weights.sum() * B  # 归一化
            
            # 前向
            delta_q, delta_p, time_logits = model(
                history_seq, goal, role, time_mask, baseline
            )
            
            # 加权损失
            loss_q = (weights * (delta_q - residual_target[:, 0]) ** 2).mean()
            loss_p = (weights * (delta_p - residual_target[:, 1]) ** 2).mean()
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
            results['l2_bc'] = train_l2_bc(self.l2, macro_samples, self.config)
        
        if self.l3 is not None and micro_samples:
            print("=" * 50)
            print("Phase 0: L3 Behavior Cloning")
            print("=" * 50)
            results['l3_bc'] = train_l3_bc(self.l3, micro_samples, self.config)
        
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
            results['l3_awr'] = train_l3_awr(self.l3, micro_samples, self.config)
        
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
    "compute_advantages",
    "ppo_loss",
    "PPOBuffer",
    "save_model",
    "load_model",
    "HRLXFTrainer",
]
