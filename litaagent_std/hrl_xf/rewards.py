"""奖励函数 - 势能整形、流动性奖励、风险惩罚.

设计目标：
1. 解决跨期信用分配问题（期货合约延迟交割）
2. 鼓励成交（流动性奖励）
3. 惩罚风险行为（库存不足、资金不足）
4. 与 L2 目标对齐（内在奖励）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# ============== 势能函数 ==============

def compute_potential(
    state: Dict[str, Any],
    gamma: float = 0.98
) -> float:
    """计算状态势能 Φ(s).
    
    公式：Φ(s) = B_t + Val(I_now) + Σ γ^δ * Val(contract)
    
    Args:
        state: 状态字典
        gamma: 折扣因子
        
    Returns:
        势能值
    """
    # 当前余额
    B_t = state.get('balance', 0)
    
    # 库存价值
    I_raw = state.get('inventory_raw', 0)
    I_product = state.get('inventory_product', 0)
    P_raw = state.get('spot_price_in', 10)
    P_product = state.get('spot_price_out', 20)
    
    val_inventory = I_raw * P_raw + I_product * P_product
    
    # 未执行合约价值（时间折扣）
    val_contracts = 0.0
    current_step = state.get('current_step', 0)
    
    for contract in state.get('pending_contracts', []):
        delta = contract.get('delivery_time', 0) - current_step
        delta = max(0, delta)  # 防止负值
        
        val = contract.get('quantity', 0) * contract.get('unit_price', 0)
        
        # 买入合约为负价值（需要付款），卖出合约为正价值
        if contract.get('is_buying', True):
            val = -val
        
        val_contracts += (gamma ** delta) * val
    
    return B_t + val_inventory + val_contracts


def shaped_reward(
    r_env: float,
    s_t: Dict[str, Any],
    s_t1: Dict[str, Any],
    gamma: float = 0.98
) -> float:
    """势能整形奖励.
    
    R' = R_env + γ * Φ(s_{t+1}) - Φ(s_t)
    
    Args:
        r_env: 环境原始奖励
        s_t: 当前状态
        s_t1: 下一状态
        gamma: 折扣因子
        
    Returns:
        整形后的奖励
    """
    phi_t = compute_potential(s_t, gamma)
    phi_t1 = compute_potential(s_t1, gamma)
    
    return r_env + gamma * phi_t1 - phi_t


# ============== 复合奖励 ==============

@dataclass
class RewardConfig:
    """奖励函数配置."""
    
    gamma: float = 0.98
    lambda_liquidity: float = 0.1
    lambda_risk: float = 0.5
    lambda_intrinsic: float = 0.1
    lambda_time: float = 0.05
    
    # 风险阈值
    min_balance_ratio: float = 0.1
    min_inventory_ratio: float = -0.2  # 允许小幅负库存
    
    # 正则化
    max_reward: float = 100.0
    min_reward: float = -100.0


def compute_composite_reward(
    r_env: float,
    s_t: Dict[str, Any],
    s_t1: Dict[str, Any],
    goal: Optional[np.ndarray] = None,
    executed_qty: float = 0,
    is_buying: bool = True,
    config: Optional[RewardConfig] = None
) -> Dict[str, float]:
    """计算复合奖励.
    
    R_total = R_profit + λ1 * R_liquidity - λ2 * R_risk + λ3 * R_intrinsic
    
    Args:
        r_env: 环境原始奖励
        s_t: 当前状态
        s_t1: 下一状态
        goal: L2 目标向量 (16,)
        executed_qty: 成交数量
        is_buying: 是否为买入
        config: 奖励配置
        
    Returns:
        分项奖励字典
    """
    if config is None:
        config = RewardConfig()
    
    rewards = {}
    
    # 1. 势能利润奖励
    R_profit = shaped_reward(r_env, s_t, s_t1, config.gamma)
    rewards['profit'] = R_profit
    
    # 2. 流动性奖励（成交即给分）
    R_liquidity = 1.0 if executed_qty > 0 else 0.0
    rewards['liquidity'] = R_liquidity
    
    # 3. 风险惩罚
    R_risk = 0.0
    
    # 3.1 库存风险
    inventory_trajectory = s_t1.get('inventory_trajectory', [])
    if len(inventory_trajectory) > 0:
        min_future_inv = np.min(inventory_trajectory)
        if min_future_inv < 0:
            R_risk += np.exp(-min_future_inv) - 1
    
    # 3.2 资金风险
    balance = s_t1.get('balance', 0)
    initial_balance = s_t1.get('initial_balance', 10000)
    if balance < config.min_balance_ratio * initial_balance:
        R_risk += 1.0
    
    rewards['risk'] = R_risk
    
    # 4. 内在一致性奖励（与 L2 目标对齐）
    R_intrinsic = 0.0
    if goal is not None and len(goal) >= 4:
        # 提取相关目标
        if is_buying:
            target_qty = goal[0]  # Q_buy (第一个桶)
        else:
            target_qty = goal[2]  # Q_sell (第一个桶)
        
        # 二次惩罚偏离
        R_intrinsic = -((executed_qty - target_qty) ** 2) / (target_qty + 1)
    
    rewards['intrinsic'] = R_intrinsic
    
    # 5. 总奖励
    R_total = (
        R_profit
        + config.lambda_liquidity * R_liquidity
        - config.lambda_risk * R_risk
        + config.lambda_intrinsic * R_intrinsic
    )
    
    # 裁剪
    R_total = np.clip(R_total, config.min_reward, config.max_reward)
    rewards['total'] = R_total
    
    return rewards


# ============== 轮级奖励（谈判用） ==============

def compute_round_reward(
    offer: Dict[str, Any],
    counter_offer: Optional[Dict[str, Any]],
    deal_info: Optional[Dict[str, Any]],
    is_buyer: bool,
    config: Optional[RewardConfig] = None
) -> float:
    """计算单轮谈判奖励.
    
    Args:
        offer: 我方报价 {quantity, price, time}
        counter_offer: 对方还价（可能为 None）
        deal_info: 成交信息（可能为 None）
        is_buyer: 是否为买方
        config: 奖励配置
        
    Returns:
        轮级奖励
    """
    if config is None:
        config = RewardConfig()
    
    reward = 0.0
    
    # 成交奖励
    if deal_info is not None:
        qty = deal_info.get('quantity', 0)
        price = deal_info.get('price', 0)
        
        # 买方希望低价，卖方希望高价
        if is_buyer:
            # 低于市场价的程度
            market_price = deal_info.get('market_price', price)
            price_gain = (market_price - price) * qty / max(market_price, 1)
        else:
            market_price = deal_info.get('market_price', price)
            price_gain = (price - market_price) * qty / max(market_price, 1)
        
        reward += 1.0 + 0.5 * price_gain
    
    # 时间惩罚（鼓励早期成交）
    round_num = offer.get('round', 0)
    reward -= config.lambda_time * round_num
    
    return reward


# ============== 日级奖励（L2 用） ==============

def compute_daily_reward(
    s_t: Dict[str, Any],
    s_t1: Dict[str, Any],
    deals: List[Dict[str, Any]],
    config: Optional[RewardConfig] = None
) -> Dict[str, float]:
    """计算日级奖励.
    
    Args:
        s_t: 当天开始状态
        s_t1: 当天结束状态
        deals: 当天成交列表
        config: 奖励配置
        
    Returns:
        分项奖励
    """
    if config is None:
        config = RewardConfig()
    
    rewards = {}
    
    # 余额变化
    balance_delta = s_t1.get('balance', 0) - s_t.get('balance', 0)
    rewards['balance_delta'] = balance_delta
    
    # 成交量
    total_qty = sum(d.get('quantity', 0) for d in deals)
    rewards['volume'] = total_qty
    
    # 势能变化
    phi_delta = compute_potential(s_t1, config.gamma) - compute_potential(s_t, config.gamma)
    rewards['potential_delta'] = phi_delta
    
    # 综合奖励
    rewards['total'] = phi_delta + config.lambda_liquidity * min(total_qty, 10)
    
    return rewards


# ============== 终局奖励 ==============

def compute_final_reward(
    final_state: Dict[str, Any],
    initial_balance: float,
    config: Optional[RewardConfig] = None
) -> float:
    """计算仿真终局奖励.
    
    Args:
        final_state: 最终状态
        initial_balance: 初始余额
        config: 奖励配置
        
    Returns:
        终局奖励
    """
    if config is None:
        config = RewardConfig()
    
    final_balance = final_state.get('balance', 0)
    
    # 清算库存
    I_raw = final_state.get('inventory_raw', 0)
    I_product = final_state.get('inventory_product', 0)
    P_raw = final_state.get('spot_price_in', 10)
    P_product = final_state.get('spot_price_out', 20)
    
    liquidation_value = I_raw * P_raw + I_product * P_product
    
    # 总资产
    total_asset = final_balance + liquidation_value
    
    # 相对收益
    relative_return = (total_asset - initial_balance) / max(initial_balance, 1)
    
    return relative_return * 100  # 放大


__all__ = [
    "RewardConfig",
    "compute_potential",
    "shaped_reward",
    "compute_composite_reward",
    "compute_round_reward",
    "compute_daily_reward",
    "compute_final_reward",
]
