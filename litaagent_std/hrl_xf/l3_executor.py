"""L3 残差执行层 - Decision Transformer.

L3 是轮级残差执行器，基于谈判历史序列和 L2 目标，输出 (Δq, Δp, δt) 的残差/分类。

输出：
- Δq: 数量残差，[-1, 1] 经缩放
- Δp: 价格残差，[-1, 1] 经缩放
- δt: 离散交货时间分类，{0, 1, ..., H}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

# 尝试导入 PyTorch（可选依赖）
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
class L3Output:
    """L3 残差执行层的输出.
    
    Attributes:
        delta_q: 数量残差
        delta_p: 价格残差
        delta_t: 选择的交货时间（离散）
        q_final: 最终数量 = q_base + delta_q
        p_final: 最终价格 = p_base + delta_p
        t_final: 最终交货时间 = delta_t
        time_probs: 时间选择的概率分布
    """
    delta_q: float
    delta_p: float
    delta_t: int
    q_final: float
    p_final: float
    t_final: int
    time_probs: Optional[np.ndarray] = None


@dataclass
class NegotiationRound:
    """谈判轮次记录."""
    quantity: float
    price: float
    delta_t: int
    is_my_turn: bool = True


# ============== 启发式 L3（无需训练） ==============

class HeuristicL3Executor:
    """启发式 L3 执行器（无需训练）.
    
    基于规则生成残差，作为神经网络 L3 的基线。
    
    策略：
    - 买方：压低价格，接受较晚交货
    - 卖方：抬高价格，偏好较早交货
    """
    
    def __init__(
        self,
        horizon: int = 40,
        price_adjustment: float = 0.05,
        qty_adjustment: float = 0.1,
    ):
        self.horizon = horizon
        self.price_adjustment = price_adjustment
        self.qty_adjustment = qty_adjustment
    
    def compute(
        self,
        history: List[NegotiationRound],
        goal: np.ndarray,
        is_buying: bool,
        time_mask: np.ndarray,
        baseline: Tuple[float, float, int]
    ) -> L3Output:
        """计算启发式残差.
        
        Args:
            history: 谈判历史
            goal: L2 目标向量，shape (16,)
            is_buying: 当前角色
            time_mask: 时间掩码，shape (H+1,)
            baseline: 基准动作 (q_base, p_base, t_base)
            
        Returns:
            L3Output
        """
        q_base, p_base, t_base = baseline
        
        # 计算残差
        n_rounds = len(history)
        
        if is_buying:
            # 买方：随轮次推进逐步让步
            concession = min(n_rounds * 0.02, 0.1)
            delta_p = p_base * concession  # 愿意提高出价
            delta_q = -q_base * self.qty_adjustment * (1 - concession)  # 减少数量需求
        else:
            # 卖方：随轮次推进逐步让步
            concession = min(n_rounds * 0.02, 0.1)
            delta_p = -p_base * concession  # 愿意降低要价
            delta_q = q_base * self.qty_adjustment * concession  # 增加供应
        
        # 选择交货时间：基于掩码选择最优时间
        delta_t = self._select_best_time(time_mask, is_buying, t_base)
        
        # 计算最终动作
        q_final = max(0, q_base + delta_q)
        p_final = max(0, p_base + delta_p)
        t_final = delta_t
        
        # 时间概率（启发式：均匀分布在可行时间上）
        time_probs = np.where(time_mask > -np.inf, 1.0, 0.0)
        if time_probs.sum() > 0:
            time_probs = time_probs / time_probs.sum()
        
        return L3Output(
            delta_q=delta_q,
            delta_p=delta_p,
            delta_t=delta_t,
            q_final=q_final,
            p_final=p_final,
            t_final=t_final,
            time_probs=time_probs
        )
    
    def _select_best_time(
        self,
        time_mask: np.ndarray,
        is_buying: bool,
        t_base: int
    ) -> int:
        """选择最佳交货时间.
        
        买方：偏好较晚交货（资金压力小）
        卖方：偏好较早交货（快速回笼资金）
        """
        valid_times = np.where(time_mask > -np.inf)[0]
        
        if len(valid_times) == 0:
            return t_base
        
        if is_buying:
            # 买方：选择较晚的时间
            return int(valid_times[-1])
        else:
            # 卖方：选择较早的时间
            return int(valid_times[0])


# ============== PyTorch L3 网络 ==============

if TORCH_AVAILABLE:
    
    class TemporalDecisionTransformer(nn.Module):
        """L3 残差执行层 - 时序决策 Transformer.
        
        使用 Decision Transformer 架构，基于谈判历史生成残差动作。
        
        架构：
        - 嵌入层：数量、价格、时间分别嵌入后相加
        - 角色嵌入：加入角色信息
        - Goal Prompting：L2 目标作为前缀 Token
        - Transformer：GPT-2 风格的因果 Transformer
        - 输出头：数量残差、价格残差、时间分类
        
        Args:
            horizon: 规划视界 H
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer 层数
            max_seq_len: 最大序列长度
            scale_q: 数量残差缩放因子
            scale_p: 价格残差缩放因子
        """
        
        def __init__(
            self,
            horizon: int = 40,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            max_seq_len: int = 20,
            scale_q: float = 10.0,
            scale_p: float = 5.0,
        ):
            super().__init__()
            
            self.horizon = horizon
            self.d_model = d_model
            self.scale_q = scale_q
            self.scale_p = scale_p
            self.max_seq_len = max_seq_len
            
            # 嵌入层
            self.qty_embed = nn.Linear(1, d_model)
            self.price_embed = nn.Linear(1, d_model)
            self.time_embed = nn.Embedding(horizon + 1, d_model)
            self.role_embed = nn.Embedding(2, d_model)
            self.goal_embed = nn.Linear(16, d_model)  # 完整16维目标向量
            
            # 位置编码
            self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)  # +1 for goal token
            
            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # 输出头
            self.head_q = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Tanh()
            )
            self.head_p = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Tanh()
            )
            self.head_t = nn.Linear(d_model, horizon + 1)  # H+1 维
            
            # 可学习缩放因子
            self.residual_scale = nn.Parameter(torch.tensor([scale_q, scale_p]))
        
        def forward(
            self,
            history: "torch.Tensor",
            goal: "torch.Tensor",
            role: "torch.Tensor",
            time_mask: "torch.Tensor",
            baseline: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """前向传播.
            
            Args:
                history: (B, T, 3) - 谈判历史 (q, p, delta_t)
                goal: (B, 16) - L2 目标（4桶 × 4分量）
                role: (B,) - 角色索引
                time_mask: (B, H+1) - 时间掩码（0 或 -inf）
                baseline: (B, 3) - 基准动作
                
            Returns:
                delta_q: (B, 1) - 数量残差
                delta_p: (B, 1) - 价格残差
                time_logits: (B, H+1) - 时间分类 logits
            """
            B, T, _ = history.shape
            
            # 1. 嵌入历史序列
            e_q = self.qty_embed(history[:, :, 0:1])      # (B, T, d)
            e_p = self.price_embed(history[:, :, 1:2])    # (B, T, d)
            e_t = self.time_embed(history[:, :, 2].long())  # (B, T, d)
            
            tokens = e_q + e_p + e_t  # (B, T, d)
            
            # 2. 添加角色嵌入
            role_emb = self.role_embed(role)  # (B, d)
            tokens = tokens + role_emb.unsqueeze(1)  # 广播
            
            # 3. Goal Prompting (作为 prefix)
            g_token = self.goal_embed(goal).unsqueeze(1)  # (B, 1, d)
            seq = torch.cat([g_token, tokens], dim=1)  # (B, T+1, d)
            
            # 4. 位置编码
            positions = torch.arange(T + 1, device=seq.device)
            seq = seq + self.pos_embed(positions)
            
            # 5. 因果掩码
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T + 1).to(seq.device)
            
            # 6. Transformer
            feat = self.transformer(seq, mask=causal_mask)
            last_feat = feat[:, -1, :]  # (B, d)
            
            # 7. 输出头
            delta_q = self.head_q(last_feat) * self.residual_scale[0]
            delta_p = self.head_p(last_feat) * self.residual_scale[1]
            
            time_logits = self.head_t(last_feat)  # (B, H+1)
            time_logits = time_logits + time_mask  # 应用 L1 掩码
            
            return delta_q, delta_p, time_logits
        
        def get_action(
            self,
            history: "torch.Tensor",
            goal: "torch.Tensor",
            role: "torch.Tensor",
            time_mask: "torch.Tensor",
            baseline: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """获取最终动作.
            
            Returns:
                q_final: (B, 1)
                p_final: (B, 1)
                t_final: (B,)
                log_prob_t: (B,)
            """
            delta_q, delta_p, time_logits = self.forward(
                history, goal, role, time_mask, baseline
            )
            
            # 合成最终动作
            q_final = baseline[:, 0:1] + delta_q
            p_final = baseline[:, 1:2] + delta_p
            
            # 采样时间
            time_dist = torch.distributions.Categorical(logits=time_logits)
            t_final = time_dist.sample()
            
            return q_final, p_final, t_final, time_dist.log_prob(t_final)
        
        def get_deterministic_action(
            self,
            history: "torch.Tensor",
            goal: "torch.Tensor",
            role: "torch.Tensor",
            time_mask: "torch.Tensor",
            baseline: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """获取确定性动作（argmax）.
            
            Returns:
                q_final: (B, 1)
                p_final: (B, 1)
                t_final: (B,)
            """
            delta_q, delta_p, time_logits = self.forward(
                history, goal, role, time_mask, baseline
            )
            
            q_final = baseline[:, 0:1] + delta_q
            p_final = baseline[:, 1:2] + delta_p
            t_final = time_logits.argmax(dim=-1)
            
            return q_final, p_final, t_final
        
        def evaluate_actions(
            self,
            history: "torch.Tensor",
            goal: "torch.Tensor",
            role: "torch.Tensor",
            time_mask: "torch.Tensor",
            baseline: "torch.Tensor",
            actions: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """评估给定动作的 log_prob.
            
            Args:
                actions: (B, 3) - (delta_q, delta_p, t)
                
            Returns:
                log_prob: (B,)
                entropy: (B,)
            """
            delta_q, delta_p, time_logits = self.forward(
                history, goal, role, time_mask, baseline
            )
            
            # 时间的 log_prob 和熵
            time_dist = torch.distributions.Categorical(logits=time_logits)
            t_actions = actions[:, 2].long()
            log_prob_t = time_dist.log_prob(t_actions)
            entropy_t = time_dist.entropy()
            
            # 数量和价格可以用 MSE 损失（连续值）
            # 这里只返回时间的 log_prob
            return log_prob_t, entropy_t

else:
    # PyTorch 不可用时的占位类
    class TemporalDecisionTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for TemporalDecisionTransformer")


# ============== L3 包装器 ==============

class L3ResidualExecutor:
    """L3 残差执行层的统一接口.
    
    支持：
    - 启发式模式（无需训练）
    - 神经网络模式（需要 PyTorch）
    
    Args:
        mode: "heuristic" 或 "neural"
        horizon: 规划视界
        model_path: 预训练模型路径（neural 模式）
    """
    
    def __init__(
        self,
        mode: str = "heuristic",
        horizon: int = 40,
        model_path: Optional[str] = None
    ):
        self.mode = mode
        self.horizon = horizon
        
        if mode == "heuristic":
            self._impl = HeuristicL3Executor(horizon=horizon)
        elif mode == "neural":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for neural mode")
            self._model = TemporalDecisionTransformer(horizon=horizon)
            if model_path:
                self._load_model(model_path)
            self._impl = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute(
        self,
        history: List[NegotiationRound],
        goal: np.ndarray,
        is_buying: bool,
        time_mask: np.ndarray,
        baseline: Tuple[float, float, int]
    ) -> L3Output:
        """计算 L3 残差.
        
        Args:
            history: 谈判历史
            goal: L2 目标向量，shape (16,)
            is_buying: 当前角色
            time_mask: 时间掩码，shape (H+1,)
            baseline: 基准动作 (q_base, p_base, t_base)
            
        Returns:
            L3Output
        """
        if self.mode == "heuristic":
            return self._impl.compute(history, goal, is_buying, time_mask, baseline)
        else:
            return self._compute_neural(history, goal, is_buying, time_mask, baseline)
    
    def _compute_neural(
        self,
        history: List[NegotiationRound],
        goal: np.ndarray,
        is_buying: bool,
        time_mask: np.ndarray,
        baseline: Tuple[float, float, int]
    ) -> L3Output:
        """使用神经网络计算残差."""
        # 构建历史张量
        if len(history) == 0:
            # 空历史：使用零填充
            history_arr = np.zeros((1, 1, 3), dtype=np.float32)
        else:
            history_arr = np.array([
                [r.quantity, r.price, r.delta_t] for r in history
            ], dtype=np.float32)
            history_arr = history_arr[np.newaxis, :, :]  # (1, T, 3)
        
        # 转换为 Tensor
        history_t = torch.from_numpy(history_arr).float()
        goal_t = torch.from_numpy(goal).float().unsqueeze(0)
        role_t = torch.tensor([0 if is_buying else 1], dtype=torch.long)
        time_mask_t = torch.from_numpy(time_mask).float().unsqueeze(0)
        baseline_t = torch.tensor([[baseline[0], baseline[1], baseline[2]]], dtype=torch.float32)
        
        with torch.no_grad():
            q_final, p_final, t_final = self._model.get_deterministic_action(
                history_t, goal_t, role_t, time_mask_t, baseline_t
            )
            
            # 计算时间概率分布
            _, _, time_logits = self._model.forward(
                history_t, goal_t, role_t, time_mask_t, baseline_t
            )
            time_probs = F.softmax(time_logits, dim=-1).squeeze(0).numpy()
        
        q_final_val = q_final.item()
        p_final_val = p_final.item()
        t_final_val = t_final.item()
        
        return L3Output(
            delta_q=q_final_val - baseline[0],
            delta_p=p_final_val - baseline[1],
            delta_t=t_final_val,
            q_final=q_final_val,
            p_final=p_final_val,
            t_final=t_final_val,
            time_probs=time_probs
        )
    
    def _load_model(self, path: str) -> None:
        """加载预训练模型."""
        state_dict = torch.load(path, map_location='cpu')
        self._model.load_state_dict(state_dict)
        self._model.eval()


__all__ = [
    "L3Output",
    "L3ResidualExecutor",
    "HeuristicL3Executor",
    "TemporalDecisionTransformer",
    "NegotiationRound",
]
