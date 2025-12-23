"""L4 全局协调层 - 时空注意力网络.

L4 是并发协调器，处理多个谈判线程之间的资源冲突，
通过注意力权重调节各 L3 实例的激进程度。

核心机制：
- 时空注意力：考虑线程间的时间冲突
- 门控权重：控制各线程的激进程度
- 资源调制：根据全局目标调整 L3 输出
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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
class ThreadState:
    """谈判线程状态."""
    thread_id: str
    thread_feat: np.ndarray  # 线程显式特征（手工构建/可离线重建）
    target_time: int  # 意向交货时间
    role: int  # 0=Buyer, 1=Seller
    priority: float = 1.0  # 线程优先级


@dataclass
class L4Output:
    """L4 协调层的输出.
    
    Attributes:
        thread_ids: 线程 ID 列表，与 weights/modulation_factors 对齐
        weights: 各线程的权重，shape (K,)
        modulation_factors: 调制因子，shape (K,)
        conflict_scores: 冲突得分矩阵，shape (K, K)
    """
    thread_ids: Optional[List[str]]
    weights: np.ndarray
    modulation_factors: np.ndarray
    conflict_scores: Optional[np.ndarray] = None


# ============== 启发式 L4（无需训练） ==============

class HeuristicL4Coordinator:
    """启发式 L4 协调器（无需训练）.
    
    基于规则分配线程权重，作为神经网络 L4 的基线。
    
    策略：
    - 时间冲突：交货时间相近的线程需要协调
    - 优先级：紧急交易获得更高权重
    - 角色平衡：买卖双方资源均衡分配
    """
    
    def __init__(
        self,
        horizon: int = 40,
        conflict_threshold: int = 3,  # 时间差小于此值视为冲突
    ):
        self.horizon = horizon
        self.conflict_threshold = conflict_threshold
    
    def compute(
        self,
        threads: List[ThreadState],
        global_feat: np.ndarray
    ) -> L4Output:
        """计算线程权重.
        
        Args:
            threads: 活跃的谈判线程列表
            global_feat: 全局上下文特征（可选，启发式默认不使用）
            
        Returns:
            L4Output
        """
        K = len(threads)
        
        if K == 0:
            return L4Output(
                thread_ids=[],
                weights=np.array([]),
                modulation_factors=np.array([]),
                conflict_scores=None
            )
        
        # 计算冲突得分矩阵
        times = np.array([t.target_time for t in threads])
        time_diff = np.abs(times[:, np.newaxis] - times[np.newaxis, :])
        conflict_scores = np.exp(-time_diff / self.conflict_threshold)
        np.fill_diagonal(conflict_scores, 0)  # 自身不冲突
        
        # 计算每个线程的冲突度
        conflict_degree = conflict_scores.sum(axis=1)  # (K,)
        
        # 计算紧急度（时间越近越紧急）
        urgency = 1.0 / (times + 1)  # (K,)
        
        # 计算优先级得分
        priorities = np.array([t.priority for t in threads])
        
        # 综合权重：紧急度高、冲突少、优先级高 -> 权重大
        raw_weights = urgency * priorities / (1 + conflict_degree)
        
        # 归一化
        weights = raw_weights / (raw_weights.sum() + 1e-8)
        
        # 调制因子：权重高的线程更激进
        modulation_factors = 1.0 + weights
        
        return L4Output(
            thread_ids=[t.thread_id for t in threads],
            weights=weights,
            modulation_factors=modulation_factors,
            conflict_scores=conflict_scores
        )
    
    def modulate_action(
        self,
        delta_q: float,
        delta_p: float,
        modulation_factor: float
    ) -> Tuple[float, float]:
        """使用调制因子调整动作.
        
        Args:
            delta_q: 数量残差
            delta_p: 价格残差
            modulation_factor: 调制因子
            
        Returns:
            (modulated_delta_q, modulated_delta_p)
        """
        return delta_q * modulation_factor, delta_p * modulation_factor


# ============== PyTorch L4 网络 ==============

if TORCH_AVAILABLE:
    
    class GlobalCoordinator(nn.Module):
        """L4 全局协调层 - 时空注意力网络.
        
        使用多头注意力和时间偏置处理线程间的协调。
        
        架构：
        - 全局上下文编码：global_feat 作为上下文
        - 线程状态投影：thread_feat 投影
        - 时间嵌入：交货时间嵌入
        - 角色嵌入：买卖角色嵌入
        - 多头注意力：带时间偏置的注意力
        - 门控输出：生成线程权重
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            horizon: 规划视界
            thread_feat_dim: 线程特征维度
            global_feat_dim: 全局特征维度
        """
        
        def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 4,
            horizon: int = 40,
            thread_feat_dim: int = 24,
            global_feat_dim: int = 30,
        ):
            super().__init__()
            
            self.d_model = d_model
            self.horizon = horizon
            self.n_heads = n_heads
            
            # 全局上下文编码
            self.global_encoder = nn.Linear(global_feat_dim, d_model)
             
            # 线程隐状态投影
            self.thread_proj = nn.Linear(thread_feat_dim, d_model)
            
            # 时间嵌入 (H+1 维，支持 δt ∈ {0, 1, ..., H})
            self.time_embed = nn.Embedding(horizon + 1, d_model)
            
            # 角色嵌入
            self.role_embed = nn.Embedding(2, d_model)
            
            # 多头自注意力
            self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            
            # 时间偏置矩阵 (H+1 × H+1)
            self.time_bias = nn.Parameter(torch.zeros(horizon + 1, horizon + 1))
            nn.init.normal_(self.time_bias, std=0.1)
            
            # 输出门控
            self.gate = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(
            self,
            thread_feats: "torch.Tensor",
            thread_times: "torch.Tensor",
            thread_roles: "torch.Tensor",
            global_state: "torch.Tensor",
            thread_mask: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """前向传播.
            
            Args:
                thread_feats: (B, K, thread_feat_dim) - K 个线程的线程特征
                thread_times: (B, K) - 每个线程的意向交货时间
                thread_roles: (B, K) - 每个线程的角色
                global_state: (B, global_feat_dim) - 全局上下文特征
                thread_mask: (B, K) - True=有效线程，False=padding（可选）
                
            Returns:
                weights: (B, K) - 线程重要性权重
                attn_output: (B, K, d_model) - 注意力输出
            """
            B, K, _ = thread_feats.shape
            n_heads = self.mha.num_heads
            
            # 1. 投影线程状态
            h = self.thread_proj(thread_feats)  # (B, K, d)
            
            # 2. 添加时间嵌入
            # 确保时间在有效范围内
            thread_times_clamped = thread_times.clamp(0, self.horizon).long()
            t_emb = self.time_embed(thread_times_clamped)  # (B, K, d)
            h = h + t_emb
            
            # 3. 添加角色嵌入
            r_emb = self.role_embed(thread_roles.long())  # (B, K, d)
            h = h + r_emb
            
            # 4. 全局上下文作为 Query
            q = self.global_encoder(global_state).unsqueeze(1)  # (B, 1, d)
            q = q.expand(-1, K, -1)  # (B, K, d)
            
            # 5. 构建时间偏置掩码
            time_diff = thread_times.unsqueeze(-1) - thread_times.unsqueeze(-2)  # (B, K, K)
            time_diff = time_diff.abs().float()
            
            # 时间距离越近 -> 偏置越大（需要更多协调）
            attn_bias = -time_diff  # (B, K, K)
            if thread_mask is not None:
                # 将 padding key 屏蔽掉，避免其参与任何 query 的注意力计算
                pad_keys = ~thread_mask.bool()  # (B, K)
                attn_bias = attn_bias.masked_fill(pad_keys.unsqueeze(1), float("-inf"))
            
            # 扩展到多头格式
            attn_bias = attn_bias.unsqueeze(1).expand(-1, n_heads, -1, -1)
            attn_bias = attn_bias.reshape(B * n_heads, K, K)
            
            # 6. 多头注意力
            attn_output, _ = self.mha(q, h, h, attn_mask=attn_bias)
            
            # 7. 计算门控权重
            gate_values = self.gate(attn_output).squeeze(-1)  # (B, K)
            if thread_mask is not None:
                gate_values = gate_values.masked_fill(~thread_mask.bool(), float("-inf"))
            
            # 8. 归一化权重
            weights = F.softmax(gate_values, dim=-1)  # (B, K)
            
            return weights, attn_output
        
        def modulate_l3_outputs(
            self,
            delta_q: "torch.Tensor",
            delta_p: "torch.Tensor",
            weights: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """使用权重调制 L3 输出.
            
            高权重 -> 更激进（放大残差）
            低权重 -> 更保守（缩小残差）
            
            Args:
                delta_q: (B, K, 1) 或 (B, K)
                delta_p: (B, K, 1) 或 (B, K)
                weights: (B, K)
                
            Returns:
                modulated_delta_q: 调制后的数量残差
                modulated_delta_p: 调制后的价格残差
            """
            # 确保维度匹配
            if delta_q.dim() == 3:
                amplify = 1.0 + weights.unsqueeze(-1)
            else:
                amplify = 1.0 + weights
            
            modulated_q = delta_q * amplify
            modulated_p = delta_p * amplify
            
            return modulated_q, modulated_p

else:
    # PyTorch 不可用时的占位类
    class GlobalCoordinator:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GlobalCoordinator")


# ============== L4 包装器 ==============

class L4ThreadCoordinator:
    """L4 协调层的统一接口.
    
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
        model_path: Optional[str] = None,
        thread_feat_dim: int = 24,
        global_feat_dim: int = 30,
    ):
        self.mode = mode
        self.horizon = horizon
        self.thread_feat_dim = thread_feat_dim
        self.global_feat_dim = global_feat_dim
        
        if mode == "heuristic":
            self._impl = HeuristicL4Coordinator(horizon=horizon)
        elif mode == "neural":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for neural mode")
            self._model = GlobalCoordinator(
                horizon=horizon,
                thread_feat_dim=thread_feat_dim,
                global_feat_dim=global_feat_dim,
            )
            if model_path:
                self._load_model(model_path)
            self._impl = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute(
        self,
        threads: List[ThreadState],
        global_feat: np.ndarray
    ) -> L4Output:
        """计算线程权重.
        
        Args:
            threads: 活跃的谈判线程列表
            global_feat: 全局上下文特征，shape (G,)
            
        Returns:
            L4Output
        """
        if self.mode == "heuristic":
            return self._impl.compute(threads, global_feat)
        else:
            return self._compute_neural(threads, global_feat)
    
    def _compute_neural(
        self,
        threads: List[ThreadState],
        global_feat: np.ndarray
    ) -> L4Output:
        """使用神经网络计算权重."""
        K = len(threads)
        
        if K == 0:
            return L4Output(
                thread_ids=[],
                weights=np.array([]),
                modulation_factors=np.array([]),
                conflict_scores=None
            )
        
        # 构建张量
        thread_feats = np.stack([t.thread_feat for t in threads]).astype(np.float32)  # (K, feat_dim)
        times = np.array([t.target_time for t in threads])
        roles = np.array([t.role for t in threads])
        
        # 添加批次维度
        thread_feats_t = torch.from_numpy(thread_feats).float().unsqueeze(0)  # (1, K, d)
        thread_times_t = torch.from_numpy(times).long().unsqueeze(0)  # (1, K)
        thread_roles_t = torch.from_numpy(roles).long().unsqueeze(0)  # (1, K)
        global_t = torch.from_numpy(global_feat).float().unsqueeze(0)  # (1, G)
        
        with torch.no_grad():
            weights, _ = self._model.forward(
                thread_feats_t, thread_times_t, thread_roles_t, global_t
            )
        
        weights_np = weights.squeeze(0).numpy()
        modulation_factors = 1.0 + weights_np
        
        # 计算冲突得分（用于调试）
        time_diff = np.abs(times[:, np.newaxis] - times[np.newaxis, :])
        conflict_scores = np.exp(-time_diff / 3)
        np.fill_diagonal(conflict_scores, 0)
        
        return L4Output(
            thread_ids=[t.thread_id for t in threads],
            weights=weights_np,
            modulation_factors=modulation_factors,
            conflict_scores=conflict_scores
        )
    
    def modulate_action(
        self,
        delta_q: float,
        delta_p: float,
        modulation_factor: float
    ) -> Tuple[float, float]:
        """使用调制因子调整动作.
        
        Args:
            delta_q: 数量残差
            delta_p: 价格残差
            modulation_factor: 调制因子
            
        Returns:
            (modulated_delta_q, modulated_delta_p)
        """
        return delta_q * modulation_factor, delta_p * modulation_factor
    
    def _load_model(self, path: str) -> None:
        """加载预训练模型."""
        state_dict = torch.load(path, map_location='cpu')
        self._model.load_state_dict(state_dict)
        self._model.eval()


__all__ = [
    "L4Output",
    "L4ThreadCoordinator",
    "HeuristicL4Coordinator",
    "GlobalCoordinator",
    "ThreadState",
]
