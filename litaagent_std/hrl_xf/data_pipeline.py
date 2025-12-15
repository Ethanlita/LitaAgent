"""数据管道 - 日志解析、状态重建、标签生成.

用于从 SCML 仿真日志中提取训练数据：
- L2 标签：日级目标向量重构
- L3 标签：轮级残差提取
- 状态重建：从日志恢复状态张量
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


# ============== 数据结构 ==============

@dataclass
class MacroSample:
    """日级样本：用于 L2 目标生成模型.
    
    Attributes:
        day: 仿真天数
        state_static: 静态状态，shape (12,)
        state_temporal: 时序状态，shape (H, 9)
        goal: 16 维目标向量
        value: 回报估计（如果有）
    """
    day: int
    state_static: np.ndarray
    state_temporal: np.ndarray
    goal: np.ndarray
    value: Optional[float] = None


@dataclass
class MicroSample:
    """轮级样本：用于 L3 残差模型.
    
    Attributes:
        negotiation_id: 谈判 ID
        history: 谈判历史，shape (T, 3)
        role: 角色 (0=Buyer, 1=Seller)
        goal: L2 目标向量，shape (16,)
        baseline: L1 基准动作 (q, p, t)
        residual: 残差 (Δq, Δp)
        time_label: 时间分类标签
        reward: 轮级奖励（如果有）
    """
    negotiation_id: str
    history: np.ndarray
    role: int
    goal: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    time_label: int
    reward: Optional[float] = None


# ============== 桶定义 ==============

BUCKET_RANGES = [(0, 2), (3, 7), (8, 14), (15, 40)]


def delta_to_bucket(delta: int) -> int:
    """将相对交货时间映射到桶索引."""
    for i, (lo, hi) in enumerate(BUCKET_RANGES):
        if lo <= delta <= hi:
            return i
    return 3  # 默认长期


# ============== 日志加载 ==============

def load_world_stats(log_dir: str) -> Dict[str, Any]:
    """加载 world_stats.json."""
    stats_path = os.path.join(log_dir, "world_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {}


def load_contracts(log_dir: str) -> List[Dict[str, Any]]:
    """加载合约日志."""
    contracts = []
    
    # 尝试多种可能的路径
    possible_paths = [
        os.path.join(log_dir, "contracts.json"),
        os.path.join(log_dir, "signed_contracts.json"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    contracts.extend(data)
                elif isinstance(data, dict):
                    for contracts_list in data.values():
                        if isinstance(contracts_list, list):
                            contracts.extend(contracts_list)
    
    return contracts


def load_negotiations_csv(log_dir: str) -> Optional["pd.DataFrame"]:
    """加载 negotiations.csv."""
    if not PANDAS_AVAILABLE:
        return None
    
    csv_files = glob.glob(os.path.join(log_dir, "**", "negotiations.csv"), recursive=True)
    
    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
    
    if not frames:
        return None
    
    return pd.concat(frames, ignore_index=True)


# ============== 状态重建 ==============

def extract_macro_state(
    day_log: Dict[str, Any],
    horizon: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """从日志重建宏观状态.
    
    Args:
        day_log: 包含当天信息的日志字典
        horizon: 规划视界
        
    Returns:
        (state_static, state_temporal)
    """
    # 静态状态 (12 维)
    state_static = np.zeros(12, dtype=np.float32)
    
    balance = day_log.get('balance', 10000)
    initial_balance = day_log.get('initial_balance', 10000)
    inventory = day_log.get('inventory', {})
    step = day_log.get('current_step', 0)
    n_steps = day_log.get('n_steps', 100)
    
    state_static[0] = balance / max(initial_balance, 1)
    state_static[1] = inventory.get('input', 0) / 100
    state_static[2] = inventory.get('output', 0) / 100
    state_static[3] = step / max(n_steps, 1)
    state_static[4] = day_log.get('n_lines', 1) / 10
    state_static[5] = day_log.get('production_cost', 1) / 50
    state_static[6] = day_log.get('spot_price_in', 10) / 50
    state_static[7] = day_log.get('spot_price_out', 20) / 50
    
    # 待执行合约
    pending = day_log.get('pending_contracts', {})
    state_static[8] = pending.get('buy_qty', 0) / 100
    state_static[9] = pending.get('sell_qty', 0) / 100
    state_static[10] = pending.get('buy_value', 0) / 10000
    state_static[11] = pending.get('sell_value', 0) / 10000
    
    # 时序状态 (H × 9)
    state_temporal = np.zeros((horizon, 9), dtype=np.float32)
    
    # 从日志中提取时序信息
    commitments = day_log.get('commitments', {})
    Q_in = np.array(commitments.get('Q_in', np.zeros(horizon)))[:horizon]
    Q_out = np.array(commitments.get('Q_out', np.zeros(horizon)))[:horizon]
    
    if len(Q_in) < horizon:
        Q_in = np.pad(Q_in, (0, horizon - len(Q_in)))
    if len(Q_out) < horizon:
        Q_out = np.pad(Q_out, (0, horizon - len(Q_out)))
    
    state_temporal[:, 0] = Q_in / 100
    state_temporal[:, 1] = Q_out / 100
    
    # 其他通道使用默认值
    n_lines = day_log.get('n_lines', 1)
    state_temporal[:, 2] = n_lines / 100  # prod_plan
    
    # 库存投影
    I_now = sum(inventory.values()) if isinstance(inventory, dict) else 0
    net_flow = Q_in - Q_out - n_lines
    I_proj = I_now + np.cumsum(net_flow)
    state_temporal[:, 3] = np.clip(I_proj / 100, -1, 2)
    
    # 自由库容
    C_total = n_lines * np.arange(horizon, 0, -1)
    C_free = C_total - I_proj
    state_temporal[:, 4] = np.clip(C_free / 100, -1, 2)
    
    return state_static, state_temporal


# ============== L2 标签重构 ==============

def reconstruct_l2_goals(
    daily_logs: List[Dict[str, Any]],
    n_buckets: int = 4
) -> List[MacroSample]:
    """从日志中反推 L2 的 16 维目标向量.
    
    策略：根据当天实际成交的合约，反推应有的目标。
    
    Args:
        daily_logs: 每天的日志列表
        n_buckets: 桶数量
        
    Returns:
        MacroSample 列表
    """
    samples = []
    
    for day_log in daily_logs:
        state_static, state_temporal = extract_macro_state(day_log)
        
        # 初始化 16 维目标
        goal = np.zeros(16, dtype=np.float32)
        
        # 按交货日期分类统计成交
        deals = day_log.get('deals', [])
        current_step = day_log.get('current_step', 0)
        
        for deal in deals:
            delta = deal.get('delivery_time', 0) - current_step
            bucket_idx = delta_to_bucket(delta)
            
            base = bucket_idx * 4
            if deal.get('is_buying', True):
                goal[base + 0] += deal.get('quantity', 0)  # Q_buy
                goal[base + 1] = max(goal[base + 1], deal.get('price', 0))  # P_buy
            else:
                goal[base + 2] += deal.get('quantity', 0)  # Q_sell
                if goal[base + 3] == 0:
                    goal[base + 3] = deal.get('price', 0)
                else:
                    goal[base + 3] = min(goal[base + 3], deal.get('price', 0))  # P_sell
        
        # 计算回报（如果有）
        value = day_log.get('daily_reward', None)
        
        samples.append(MacroSample(
            day=current_step,
            state_static=state_static,
            state_temporal=state_temporal,
            goal=goal,
            value=value
        ))
    
    return samples


# ============== L3 标签提取 ==============

def extract_l3_residuals(
    negotiation_logs: List[Dict[str, Any]],
    horizon: int = 40
) -> List[MicroSample]:
    """从谈判日志中提取 L3 残差标签.
    
    Args:
        negotiation_logs: 谈判日志列表
        horizon: 规划视界
        
    Returns:
        MicroSample 列表
    """
    samples = []
    
    for neg in negotiation_logs:
        # 谈判历史
        offer_history = neg.get('offer_history', [])
        if len(offer_history) == 0:
            continue
        
        history = np.array([
            [h.get('quantity', 0), h.get('price', 0), h.get('delta_t', 0)]
            for h in offer_history
        ], dtype=np.float32)
        
        # 角色
        role = 0 if neg.get('is_buyer', True) else 1
        
        # L2 目标（如果有）
        goal = np.array(neg.get('l2_goal', np.zeros(16)), dtype=np.float32)
        
        # L1 基准
        baseline = np.array(neg.get('baseline', [1, 10, 1]), dtype=np.float32)
        
        # 专家动作
        final_action = neg.get('final_action', {})
        expert_q = final_action.get('quantity', baseline[0])
        expert_p = final_action.get('price', baseline[1])
        expert_t = final_action.get('delta_t', int(baseline[2]))
        
        # 残差
        residual = np.array([
            expert_q - baseline[0],
            expert_p - baseline[1]
        ], dtype=np.float32)
        
        # 时间标签
        time_label = min(max(0, expert_t), horizon)
        
        # 奖励
        reward = neg.get('reward', None)
        
        samples.append(MicroSample(
            negotiation_id=neg.get('id', str(len(samples))),
            history=history,
            role=role,
            goal=goal,
            baseline=baseline,
            residual=residual,
            time_label=time_label,
            reward=reward
        ))
    
    return samples


# ============== 批量处理 ==============

def load_tournament_data(
    tournament_dir: str,
    agent_name: str = "LitaAgent"
) -> Tuple[List[MacroSample], List[MicroSample]]:
    """从锦标赛目录加载训练数据.
    
    Args:
        tournament_dir: 锦标赛日志目录
        agent_name: 要提取的代理名称
        
    Returns:
        (macro_samples, micro_samples)
    """
    macro_samples = []
    micro_samples = []
    
    # 遍历所有世界目录
    world_dirs = glob.glob(os.path.join(tournament_dir, "**", "world_*"), recursive=True)
    
    for world_dir in world_dirs:
        if not os.path.isdir(world_dir):
            continue
        
        # 加载世界统计
        stats = load_world_stats(world_dir)
        
        # 加载合约
        contracts = load_contracts(world_dir)
        
        # 按天组织数据
        daily_logs = _organize_by_day(stats, contracts, agent_name)
        
        # 提取 L2 样本
        macro_samples.extend(reconstruct_l2_goals(daily_logs))
        
        # 加载谈判日志（如果有）
        neg_logs = _load_negotiation_logs(world_dir, agent_name)
        micro_samples.extend(extract_l3_residuals(neg_logs))
    
    return macro_samples, micro_samples


def _organize_by_day(
    stats: Dict[str, Any],
    contracts: List[Dict[str, Any]],
    agent_name: str
) -> List[Dict[str, Any]]:
    """将数据按天组织."""
    n_steps = stats.get('n_steps', 100)
    daily_logs = []
    
    for step in range(n_steps):
        day_log = {
            'current_step': step,
            'n_steps': n_steps,
            'balance': stats.get('balance_history', {}).get(str(step), 10000),
            'inventory': stats.get('inventory_history', {}).get(str(step), {}),
            'deals': [],
        }
        
        # 筛选当天签署的合约
        for c in contracts:
            if c.get('signed_at', -1) == step:
                is_buying = c.get('buyer') == agent_name
                day_log['deals'].append({
                    'quantity': c.get('quantity', 0),
                    'price': c.get('unit_price', 0),
                    'delivery_time': c.get('delivery_time', step + 1),
                    'is_buying': is_buying,
                })
        
        daily_logs.append(day_log)
    
    return daily_logs


def _load_negotiation_logs(
    world_dir: str,
    agent_name: str
) -> List[Dict[str, Any]]:
    """加载谈判日志."""
    logs = []
    
    neg_file = os.path.join(world_dir, "negotiations.json")
    if os.path.exists(neg_file):
        with open(neg_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                # 筛选指定代理的谈判
                for neg in data:
                    if agent_name in str(neg.get('partners', [])):
                        logs.append(neg)
    
    return logs


# ============== 数据保存 ==============

def save_samples(
    macro_samples: List[MacroSample],
    micro_samples: List[MicroSample],
    output_dir: str
) -> None:
    """保存样本到文件.
    
    Args:
        macro_samples: L2 宏观样本
        micro_samples: L3 微观样本
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存宏观样本
    if macro_samples:
        macro_data = {
            'state_static': np.stack([s.state_static for s in macro_samples]),
            'state_temporal': np.stack([s.state_temporal for s in macro_samples]),
            'goals': np.stack([s.goal for s in macro_samples]),
            'days': np.array([s.day for s in macro_samples]),
        }
        np.savez(os.path.join(output_dir, "l2_samples.npz"), **macro_data)
    
    # 保存微观样本（变长序列需要特殊处理）
    if micro_samples:
        micro_data = {
            'roles': np.array([s.role for s in micro_samples]),
            'goals': np.stack([s.goal for s in micro_samples]),
            'baselines': np.stack([s.baseline for s in micro_samples]),
            'residuals': np.stack([s.residual for s in micro_samples]),
            'time_labels': np.array([s.time_label for s in micro_samples]),
        }
        np.savez(os.path.join(output_dir, "l3_samples.npz"), **micro_data)
        
        # 历史序列单独保存（变长）
        histories = [s.history for s in micro_samples]
        np.savez(os.path.join(output_dir, "l3_histories.npz"),
                 histories=np.array(histories, dtype=object))


def load_samples(
    data_dir: str
) -> Tuple[List[MacroSample], List[MicroSample]]:
    """从文件加载样本.
    
    Args:
        data_dir: 数据目录
        
    Returns:
        (macro_samples, micro_samples)
    """
    macro_samples = []
    micro_samples = []
    
    # 加载宏观样本
    l2_path = os.path.join(data_dir, "l2_samples.npz")
    if os.path.exists(l2_path):
        data = np.load(l2_path)
        for i in range(len(data['days'])):
            macro_samples.append(MacroSample(
                day=int(data['days'][i]),
                state_static=data['state_static'][i],
                state_temporal=data['state_temporal'][i],
                goal=data['goals'][i]
            ))
    
    # 加载微观样本
    l3_path = os.path.join(data_dir, "l3_samples.npz")
    l3_hist_path = os.path.join(data_dir, "l3_histories.npz")
    
    if os.path.exists(l3_path):
        data = np.load(l3_path)
        histories = None
        
        if os.path.exists(l3_hist_path):
            hist_data = np.load(l3_hist_path, allow_pickle=True)
            histories = hist_data['histories']
        
        for i in range(len(data['roles'])):
            history = histories[i] if histories is not None else np.zeros((1, 3))
            micro_samples.append(MicroSample(
                negotiation_id=str(i),
                history=history,
                role=int(data['roles'][i]),
                goal=data['goals'][i],
                baseline=data['baselines'][i],
                residual=data['residuals'][i],
                time_label=int(data['time_labels'][i])
            ))
    
    return macro_samples, micro_samples


__all__ = [
    "MacroSample",
    "MicroSample",
    "reconstruct_l2_goals",
    "extract_l3_residuals",
    "load_tournament_data",
    "save_samples",
    "load_samples",
    "extract_macro_state",
    "delta_to_bucket",
]
