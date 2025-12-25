"""L2 战略规划层 - 1D-CNN 双塔网络.

L2 是日级战略规划器，基于宏观状态生成分桶目标向量，指导 L3 的微观决策。

输出：16 维向量 = 4 桶 × 4 分量
- 桶 0 (Urgent, 0-2天): Q_buy, P_buy, Q_sell, P_sell
- 桶 1 (Short-term, 3-7天): Q_buy, P_buy, Q_sell, P_sell
- 桶 2 (Medium-term, 8-14天): Q_buy, P_buy, Q_sell, P_sell
- 桶 3 (Long-term, 15+天): Q_buy, P_buy, Q_sell, P_sell
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
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


# ============== 桶定义 ==============

BUCKET_RANGES = [
    (0, 2),    # Bucket 0: Urgent
    (3, 7),    # Bucket 1: Short-term
    (8, 14),   # Bucket 2: Medium-term
    (15, 40),  # Bucket 3: Long-term
]

N_BUCKETS = 4
BUCKET_DIM = 4  # Q_buy, P_buy, Q_sell, P_sell
OUTPUT_DIM = N_BUCKETS * BUCKET_DIM  # 16


def delta_to_bucket(delta: int) -> int:
    """将相对交货时间映射到桶索引.
    
    Args:
        delta: 相对交货时间（天数）
        
    Returns:
        桶索引 (0-3)
    """
    if delta <= 2:
        return 0  # Urgent
    elif delta <= 7:
        return 1  # Short-term
    elif delta <= 14:
        return 2  # Medium-term
    else:
        return 3  # Long-term


def bucket_to_delta_range(bucket_idx: int) -> Tuple[int, int]:
    """获取桶对应的时间范围.
    
    Args:
        bucket_idx: 桶索引 (0-3)
        
    Returns:
        (min_delta, max_delta)
    """
    if 0 <= bucket_idx < len(BUCKET_RANGES):
        return BUCKET_RANGES[bucket_idx]
    return (0, 40)


@dataclass
class L2Output:
    """L2 战略规划层的输出.
    
    Attributes:
        goal_vector: 16 维目标向量
        value: 价值估计（如果使用 Critic）
        bucket_goals: 按桶分解的目标字典
    """
    goal_vector: np.ndarray  # (16,)
    value: Optional[float] = None
    bucket_goals: Optional[Dict[int, Dict[str, float]]] = None
    
    def get_bucket_goal(self, bucket_idx: int) -> Dict[str, float]:
        """获取指定桶的目标.
        
        Args:
            bucket_idx: 桶索引 (0-3)
            
        Returns:
            {"Q_buy": ..., "P_buy": ..., "Q_sell": ..., "P_sell": ...}
        """
        offset = bucket_idx * 4
        return {
            "Q_buy": float(self.goal_vector[offset]),
            "P_buy": float(self.goal_vector[offset + 1]),
            "Q_sell": float(self.goal_vector[offset + 2]),
            "P_sell": float(self.goal_vector[offset + 3]),
        }


# ============== 启发式 L2（无需训练） ==============

class HeuristicL2Manager:
    """启发式 L2 管理器（无需训练）.
    
    ⚠️ 警告：此类仅用于开发调试和冷启动 fallback。
    生产环境中 L2-L4 层应使用 neural 模式，heuristic 模式
    的参数未经调优，可能导致次优决策。
    
    基于规则生成分桶目标，作为神经网络 L2 的基线。
    
    策略：
    - 紧急桶：满足当前生产需求
    - 短期桶：建立周转缓冲
    - 中期桶：战略储备
    - 长期桶：套利机会
    """
    
    def __init__(
        self,
        horizon: int = 40,
        base_price_margin: float = 0.1,  # 价格边际
    ):
        self.horizon = horizon
        self.base_price_margin = base_price_margin
    
    def compute(
        self,
        x_static: np.ndarray,
        X_temporal: np.ndarray,
        is_buying: bool
    ) -> L2Output:
        """计算启发式目标.
        
        Args:
            x_static: 静态特征，shape (12,)
            X_temporal: 时序特征，shape (H+1, 10)
            is_buying: 当前角色
            
        Returns:
            L2Output
        """
        goal = np.zeros(16, dtype=np.float32)
        
        # 解析静态特征
        balance_norm = x_static[0]
        inventory_raw = x_static[1]
        inventory_product = x_static[2]
        step_progress = x_static[3]
        n_lines = x_static[4] * 10  # 反归一化
        spot_price_in = x_static[6] * 50  # 反归一化
        spot_price_out = x_static[7] * 50  # 反归一化
        
        # 计算各桶目标
        for bucket_idx in range(N_BUCKETS):
            offset = bucket_idx * 4
            
            # 基础需求：基于生产能力和库存
            base_demand = max(0, n_lines - inventory_raw * 100)
            
            # 时间紧迫性权重
            urgency = 1.0 - bucket_idx * 0.2  # 越紧急越高
            
            # 买入目标
            Q_buy = base_demand * urgency / N_BUCKETS
            P_buy = spot_price_in * (1 - self.base_price_margin * urgency)
            
            # 卖出目标（基于成品库存）
            Q_sell = inventory_product * 100 * urgency / N_BUCKETS
            P_sell = spot_price_out * (1 + self.base_price_margin * urgency)
            
            # 阶段调整
            if step_progress > 0.8:
                # 后期：清库存
                Q_buy *= 0.5
                Q_sell *= 2.0
            elif step_progress < 0.2:
                # 前期：建库存
                Q_buy *= 1.5
                Q_sell *= 0.5
            
            goal[offset] = Q_buy
            goal[offset + 1] = P_buy
            goal[offset + 2] = Q_sell
            goal[offset + 3] = P_sell
        
        return L2Output(
            goal_vector=goal,
            value=None,
            bucket_goals={i: self._extract_bucket(goal, i) for i in range(N_BUCKETS)}
        )
    
    def _extract_bucket(self, goal: np.ndarray, bucket_idx: int) -> Dict[str, float]:
        """提取桶目标."""
        offset = bucket_idx * 4
        return {
            "Q_buy": float(goal[offset]),
            "P_buy": float(goal[offset + 1]),
            "Q_sell": float(goal[offset + 2]),
            "P_sell": float(goal[offset + 3]),
        }


# ============== PyTorch L2 网络 ==============

if TORCH_AVAILABLE:
    
    class HorizonManagerPPO(nn.Module):
        """L2 战略规划层 - 1D-CNN 时平规划器.
        
        使用 PPO 算法训练的 Actor-Critic 网络。
        
        架构：
        - 时序塔：1D-CNN 提取时序模式
        - 静态嵌入：线性投影静态特征
        - 角色嵌入：可学习角色向量
        - 融合层：拼接并投影
        - Actor 头：输出目标向量的均值和标准差
        - Critic 头：输出价值估计
        
        Args:
            horizon: 规划视界 H
            n_buckets: 桶数量
            d_static: 静态特征维度
            d_temporal: 时序特征通道数
            d_role: 角色嵌入维度
        """
        
        def __init__(
            self,
            horizon: int = 40,
            n_buckets: int = 4,
            d_static: int = 12,
            d_temporal: int = 10,  # 10通道：vol_in, vol_out, prod_plan, inventory_proj, capacity_free, balance_proj, price_diff_in, price_diff_out, buy_pressure, sell_pressure
            d_role: int = 16,
        ):
            super().__init__()
            
            self.horizon = horizon
            self.n_buckets = n_buckets
            self.output_dim = n_buckets * 4  # 16
            
            # 时序特征塔
            self.conv1 = nn.Conv1d(d_temporal, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
            self.pool = nn.AdaptiveMaxPool1d(1)
            
            # 静态特征嵌入
            self.static_embed = nn.Linear(d_static, 32)
            
            # 角色嵌入：使用 Linear 支持 Multi-Hot [can_buy, can_sell]
            # 例如: [0,1]=第一层（只卖）, [1,1]=中间层, [1,0]=最后层（只买）
            self.role_embed = nn.Linear(2, d_role)
            
            # 融合层
            fusion_dim = 64 + 32 + d_role
            self.fusion = nn.Linear(fusion_dim, 128)
            
            # Actor 头
            self.actor_mean = nn.Linear(128, self.output_dim)
            self.actor_log_std = nn.Linear(128, self.output_dim)
            
            # Critic 头
            self.critic = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
    def forward(
        self,
        x_static: "torch.Tensor",
        X_temporal: "torch.Tensor",
        x_role: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """前向传播.
        
        Args:
            x_static: (B, 12) - 静态特征
            X_temporal: (B, H+1, 10) - 时序特征
            x_role: (B, 2) - 角色 Multi-Hot [can_buy, can_sell]
                
            Returns:
                mean: (B, 16) - 目标向量均值
                log_std: (B, 16) - 目标向量对数标准差
                value: (B, 1) - 价值估计
            """
            # 时序塔 (需要转置为 B, C, L)
            x_t = X_temporal.permute(0, 2, 1)  # (B, 10, H)
            x_t = F.relu(self.conv1(x_t))
            x_t = F.relu(self.conv2(x_t))
            h_temp = self.pool(x_t).squeeze(-1)  # (B, 64)
            
            # 静态嵌入
            h_static = F.relu(self.static_embed(x_static))  # (B, 32)
            
            # 角色嵌入 (Linear 接受 Multi-Hot)
            h_role = F.relu(self.role_embed(x_role))  # (B, d_role)
            
            # 融合
            h = torch.cat([h_temp, h_static, h_role], dim=-1)
            h = F.relu(self.fusion(h))  # (B, 128)
            
            # Actor
            mean = self.actor_mean(h)
            log_std = self.actor_log_std(h)
            log_std = torch.clamp(log_std, -20, 2)  # 数值稳定性
            
            # Critic
            value = self.critic(h)
            
            return mean, log_std, value
        
    def sample_action(
        self,
        x_static: "torch.Tensor",
        X_temporal: "torch.Tensor",
        x_role: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """采样动作并计算 log_prob.
            
            Args:
                x_role: (B, 2) - 角色 Multi-Hot [can_buy, can_sell]
            
            Returns:
                action: (B, 16) - 采样的目标向量
                log_prob: (B,) - 对数概率
                value: (B, 1) - 价值估计
            """
            mean, log_std, value = self.forward(x_static, X_temporal, x_role)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value
        
    def evaluate_actions(
        self,
        x_static: "torch.Tensor",
        X_temporal: "torch.Tensor",
        x_role: "torch.Tensor",
        actions: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """评估给定动作的 log_prob 和熵.
            
            Args:
                x_role: (B, 2) - 角色 Multi-Hot [can_buy, can_sell]
                actions: (B, 16) - 要评估的动作
                
            Returns:
                log_prob: (B,)
                entropy: (B,)
                value: (B, 1)
            """
            mean, log_std, value = self.forward(x_static, X_temporal, x_role)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy, value
        
    def get_deterministic_action(
        self,
        x_static: "torch.Tensor",
        X_temporal: "torch.Tensor",
        x_role: "torch.Tensor"
        ) -> "torch.Tensor":
            """获取确定性动作（均值）.
            
            Args:
                x_role: (B, 2) - 角色 Multi-Hot [can_buy, can_sell]
            
            Returns:
                action: (B, 16)
            """
            mean, _, _ = self.forward(x_static, X_temporal, x_role)
            return mean

else:
    # PyTorch 不可用时的占位类
    class HorizonManagerPPO:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for HorizonManagerPPO")


# ============== L2 包装器 ==============

class L2StrategicManager:
    """L2 战略规划层的统一接口.
    
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
            self._impl = HeuristicL2Manager(horizon=horizon)
        elif mode == "neural":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for neural mode")
            self._model = HorizonManagerPPO(horizon=horizon)
            if model_path:
                self._load_model(model_path)
            self._impl = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute(
        self,
        x_static: np.ndarray,
        X_temporal: np.ndarray,
        is_buying: bool,
        awi: Optional["StdAWI"] = None
    ) -> L2Output:
        """计算 L2 目标.
        
        Args:
            x_static: 静态特征，shape (12,)
            X_temporal: 时序特征，shape (H+1, 10)
            is_buying: 当前角色（用于启发式模式）
            awi: Agent World Interface（用于神经网络模式计算 x_role）
            
        Returns:
            L2Output
        """
        if self.mode == "heuristic":
            return self._impl.compute(x_static, X_temporal, is_buying)
        else:
            return self._compute_neural(x_static, X_temporal, is_buying, awi)
    
    def _compute_neural(
        self,
        x_static: np.ndarray,
        X_temporal: np.ndarray,
        is_buying: bool,
        awi: Optional["StdAWI"] = None
    ) -> L2Output:
        """使用神经网络计算目标."""
        # 转换为 Tensor
        x_static_t = torch.from_numpy(x_static).float().unsqueeze(0)
        X_temporal_t = torch.from_numpy(X_temporal).float().unsqueeze(0)
        
        # 基于供应链位置计算 x_role (Multi-Hot: [can_buy, can_sell])
        # 第一层 (input_product=0): 采购外生 → 只能谈判销售 → [0, 1]
        # 最后层 (output_product=n_products-1): 销售外生 → 只能谈判采购 → [1, 0]
        # 中间层: 买卖都需谈判 → [1, 1]
        if awi is not None:
            input_product = getattr(awi, 'my_input_product', 0)
            output_product = getattr(awi, 'my_output_product', 1)
            n_products = getattr(awi, 'n_products', output_product + 1)
            
            is_first_level = (input_product == 0)
            is_last_level = (output_product == n_products - 1)
            
            can_buy = 0.0 if is_first_level else 1.0
            can_sell = 0.0 if is_last_level else 1.0
        else:
            # 回退：基于 is_buying 推断（不准确，但兼容旧接口）
            can_buy = 1.0
            can_sell = 1.0
        
        x_role = np.array([can_buy, can_sell], dtype=np.float32)
        x_role_t = torch.from_numpy(x_role).float().unsqueeze(0)
        
        with torch.no_grad():
            action = self._model.get_deterministic_action(
                x_static_t, X_temporal_t, x_role_t
            )
            _, _, value = self._model.forward(
                x_static_t, X_temporal_t, x_role_t
            )
        
        goal_vector = action.squeeze(0).numpy()
        value_scalar = value.item()
        
        return L2Output(
            goal_vector=goal_vector,
            value=value_scalar,
            bucket_goals={i: self._extract_bucket(goal_vector, i) for i in range(N_BUCKETS)}
        )
    
    def _extract_bucket(self, goal: np.ndarray, bucket_idx: int) -> Dict[str, float]:
        """提取桶目标."""
        offset = bucket_idx * 4
        return {
            "Q_buy": float(goal[offset]),
            "P_buy": float(goal[offset + 1]),
            "Q_sell": float(goal[offset + 2]),
            "P_sell": float(goal[offset + 3]),
        }
    
    def _load_model(self, path: str) -> None:
        """加载预训练模型."""
        state_dict = torch.load(path, map_location='cpu')
        self._model.load_state_dict(state_dict)
        self._model.eval()


__all__ = [
    "L2Output",
    "L2StrategicManager",
    "HeuristicL2Manager",
    "HorizonManagerPPO",
    "delta_to_bucket",
    "bucket_to_delta_range",
    "BUCKET_RANGES",
    "N_BUCKETS",
    "OUTPUT_DIM",
]
