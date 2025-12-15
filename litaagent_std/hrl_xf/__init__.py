"""LitaAgent-HRL (HRL-XF) 模块入口.

HRL-XF: Hybrid Residual Learner - Extended Framework, Futures Edition
针对 SCML 2025 期货市场环境的分层强化学习代理。

架构概览:
- L1 (Safety): 时序 ATP 算法，输出安全掩码 Q_safe, time_mask
- L2 (Manager): 1D-CNN 双塔网络，输出 16 维分桶目标向量
- L3 (Executor): Decision Transformer，输出残差动作 (Δq, Δp, δt)
- L4 (Coordinator): 多头注意力 + 时间偏置，输出线程权重

训练与数据:
- data_pipeline: 日志解析、状态重建、标签生成
- training: BC、AWR、PPO 训练循环
- rewards: 势能整形、复合奖励函数
"""

from .agent import LitaAgentHRL, LitaAgentHRLTracked
from .l1_safety import L1SafetyLayer, L1Output
from .state_builder import StateBuilder, StateDict
from .l2_manager import L2StrategicManager
from .l3_executor import L3ResidualExecutor
from .l4_coordinator import L4ThreadCoordinator

# 数据与训练
from .data_pipeline import (
    MacroSample,
    MicroSample,
    reconstruct_l2_goals,
    extract_l3_residuals,
    load_tournament_data,
    save_samples,
    load_samples,
)
from .training import (
    TrainConfig,
    train_l2_bc,
    train_l3_bc,
    train_l3_awr,
    HRLXFTrainer,
    save_model,
    load_model,
)
from .rewards import (
    RewardConfig,
    compute_potential,
    shaped_reward,
    compute_composite_reward,
    compute_daily_reward,
    compute_final_reward,
)

__all__ = [
    # 主代理
    "LitaAgentHRL",
    "LitaAgentHRLTracked",
    # 各层模块
    "L1SafetyLayer",
    "L1Output",
    "L2StrategicManager",
    "L3ResidualExecutor",
    "L4ThreadCoordinator",
    # 状态构建
    "StateBuilder",
    "StateDict",
    # 数据管道
    "MacroSample",
    "MicroSample",
    "reconstruct_l2_goals",
    "extract_l3_residuals",
    "load_tournament_data",
    "save_samples",
    "load_samples",
    # 训练
    "TrainConfig",
    "train_l2_bc",
    "train_l3_bc",
    "train_l3_awr",
    "HRLXFTrainer",
    "save_model",
    "load_model",
    # 奖励
    "RewardConfig",
    "compute_potential",
    "shaped_reward",
    "compute_composite_reward",
    "compute_daily_reward",
    "compute_final_reward",
]

__version__ = "2.0.0"
