"""数据管道 - 日志解析、状态重建、标签生成.

用于从 SCML 仿真日志中提取训练数据：
- L2 标签：日级目标向量重构
- L3 标签：轮级 AOP 提取
- 状态重建：从日志恢复状态张量
"""

from __future__ import annotations

import concurrent.futures
import glob
import itertools
import json
import math
import os
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
        state_temporal: 时序状态，shape (H+1, 10)
        x_role: L2 谈判能力 Multi-Hot 编码 (2,)
                [can_negotiate_buy, can_negotiate_sell]
                - [0, 1]: 第一层，只能谈判销售（采购外生）
                - [1, 1]: 中间层，买卖都需谈判
                - [1, 0]: 最后层，只能谈判采购（销售外生）
        goal: 16 维目标向量
        value: 回报估计（如果有）
    """
    day: int
    state_static: np.ndarray
    state_temporal: np.ndarray
    x_role: np.ndarray  # (2,) Multi-Hot: [can_buy, can_sell]
    goal: np.ndarray
    value: Optional[float] = None


@dataclass
class MicroSample:
    """轮级样本：用于 L3 AOP 训练."""
    negotiation_id: str
    history: np.ndarray  # shape (T, 4): [q, p, delta_t, is_my_turn]
    role: int  # 0=Buyer, 1=Seller
    current_offer: np.ndarray  # shape (3,): [q, p, delta_t]
    relative_time: float
    l2_goal: np.ndarray  # shape (16,)
    goal_gap_buy: np.ndarray  # shape (4,)
    goal_gap_sell: np.ndarray  # shape (4,)
    today_bought_qty: float
    today_sold_qty: float
    today_bought_by_bucket: np.ndarray  # shape (4,)
    today_sold_by_bucket: np.ndarray  # shape (4,)
    x_static: np.ndarray  # shape (12,)
    X_temporal: np.ndarray  # shape (H+1, 10)
    time_mask: np.ndarray  # shape (H+1,)
    Q_safe: np.ndarray  # shape (H+1,)
    B_free: float
    action_op: int  # 0=ACCEPT, 1=REJECT, 2=END
    target_quantity: Optional[int] = None
    target_price: Optional[float] = None
    target_time: Optional[int] = None  # delta_t
    reward: Optional[float] = None


@dataclass
class L4DistillSample:
    """用于 L4 蒸馏训练的样本（变长线程集合）。"""

    day: int
    global_state: np.ndarray  # (12,)
    thread_states: np.ndarray  # (K, 11)
    teacher_alpha: np.ndarray  # (K,)
    thread_ids: Optional[List[str]] = None


# ============== 桶定义 ==============

BUCKET_RANGES = [(0, 2), (3, 7), (8, 14), (15, 40)]


def delta_to_bucket(delta: int) -> int:
    """将相对交货时间映射到桶索引."""
    for i, (lo, hi) in enumerate(BUCKET_RANGES):
        if lo <= delta <= hi:
            return i
    return 3  # 默认长期


def delta_to_bucket_soft(delta: float, *, smooth_width: float = 1.0) -> List[Tuple[int, float]]:
    """将交货时间 delta 软分配到桶（最多两个桶非零），缓解硬分桶边界效应。

    规则：
    - 默认落在硬分桶；
    - 若 delta 靠近相邻桶边界（|delta - boundary| <= smooth_width），则按线性权重分摊到两侧桶。
    """
    if smooth_width <= 0:
        idx = delta_to_bucket(int(round(delta)))
        return [(idx, 1.0)]

    d = float(delta)
    idx = delta_to_bucket(int(round(d)))

    # 与左边界的软分配
    if idx > 0:
        left_hi = BUCKET_RANGES[idx - 1][1]
        right_lo = BUCKET_RANGES[idx][0]
        boundary = (float(left_hi) + float(right_lo)) / 2.0
        if abs(d - boundary) <= smooth_width:
            w_left = (boundary + smooth_width - d) / (2.0 * smooth_width)
            w_left = float(np.clip(w_left, 0.0, 1.0))
            w_right = 1.0 - w_left
            return [(idx - 1, w_left), (idx, w_right)]

    # 与右边界的软分配
    if idx < len(BUCKET_RANGES) - 1:
        left_hi = BUCKET_RANGES[idx][1]
        right_lo = BUCKET_RANGES[idx + 1][0]
        boundary = (float(left_hi) + float(right_lo)) / 2.0
        if abs(d - boundary) <= smooth_width:
            w_left = (boundary + smooth_width - d) / (2.0 * smooth_width)
            w_left = float(np.clip(w_left, 0.0, 1.0))
            w_right = 1.0 - w_left
            return [(idx, w_left), (idx + 1, w_right)]

    return [(idx, 1.0)]


def _weighted_median(values: List[Tuple[float, float]]) -> Optional[float]:
    """加权中位数（values=(x, w)）。"""
    if not values:
        return None
    pairs = [(float(x), float(w)) for x, w in values if w is not None and w > 0]
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    acc = 0.0
    for x, w in pairs:
        acc += w
        if acc >= 0.5 * total_w:
            return x
    return pairs[-1][0]


# ============== L1 Time Mask 离线计算 ==============

def compute_q_safe_offline(
    state_dict: Dict[str, Any],
    horizon: int = 40,
) -> np.ndarray:
    """从日志状态离线计算 L1 Q_safe（买侧，与 l1_safety.py 一致）。"""
    temporal_len = horizon + 1

    inventory = state_dict.get('inventory', {})
    n_lines = int(state_dict.get('n_lines', 1) or 1)

    commitments = state_dict.get('commitments', {})
    Q_in = np.array(commitments.get('Q_in', np.zeros(temporal_len)), dtype=np.float32)
    Q_out = np.array(commitments.get('Q_out', np.zeros(temporal_len)), dtype=np.float32)

    if len(Q_in) < temporal_len:
        Q_in = np.pad(Q_in, (0, temporal_len - len(Q_in)))
    else:
        Q_in = Q_in[:temporal_len]
    if len(Q_out) < temporal_len:
        Q_out = np.pad(Q_out, (0, temporal_len - len(Q_out)))
    else:
        Q_out = Q_out[:temporal_len]

    Q_in_h = Q_in[:horizon]
    Q_out_h = Q_out[:horizon]

    if isinstance(inventory, dict):
        I_now = float(inventory.get('input', 0) or 0.0)
    else:
        I_now = float(inventory) if inventory else 0.0

    # 计算 backlog（满载加工假设）
    backlog = np.zeros(horizon, dtype=np.float32)
    b = float(I_now)
    for k in range(horizon):
        backlog[k] = max(0.0, b)
        b = b + float(Q_in_h[k]) - float(n_lines)
        if b < 0.0:
            b = 0.0

    n_steps = int(state_dict.get('n_steps', 100) or 100)
    current_step = int(state_dict.get('current_step', 0) or 0)

    C_total = np.zeros(horizon, dtype=np.float32)
    for k in range(horizon):
        remaining_days = n_steps - (current_step + k)
        C_total[k] = n_lines * max(0, remaining_days)

    # Q_safe[δ] = max(0, C_total[δ] - backlog[δ] - Σ_{k=δ}^{H-1} Q_in[k])
    suffix_in = np.cumsum(Q_in_h[::-1])[::-1]
    Q_safe_h = C_total - backlog - suffix_in
    Q_safe_h = np.maximum(Q_safe_h, 0)

    Q_safe = np.zeros(temporal_len, dtype=np.float32)
    if horizon > 0 and len(Q_safe_h) > 0:
        Q_safe[:horizon] = Q_safe_h
        Q_safe[horizon] = Q_safe_h[-1]
    return Q_safe.astype(np.float32)


def compute_q_safe_sell_offline(
    state_dict: Dict[str, Any],
    horizon: int = 40,
) -> np.ndarray:
    """从日志状态离线计算 L1 Q_safe_sell（卖侧，与 l1_safety.py 一致）。
    
    算法分两步：
    1. 计算成品轨迹 S[k] = I_output + Σ_{j=0}^{k-1} (Q_prod[j] - Q_out[j])
    2. 使用逆向累积最小值：Q_safe_sell[δ] = min_{k>=δ} S[k]
    """
    temporal_len = horizon + 1

    inventory = state_dict.get('inventory', {})
    n_lines = int(state_dict.get('n_lines', 1) or 1)

    commitments = state_dict.get('commitments', {})
    Q_out = np.array(commitments.get('Q_out', np.zeros(temporal_len)), dtype=np.float32)

    if len(Q_out) < temporal_len:
        Q_out = np.pad(Q_out, (0, temporal_len - len(Q_out)))
    else:
        Q_out = Q_out[:temporal_len]

    Q_out_h = Q_out[:horizon]

    # 获取成品库存（output）
    if isinstance(inventory, dict):
        I_output_now = float(inventory.get('output', 0) or 0.0)
    else:
        I_output_now = 0.0

    Q_prod = np.full(horizon, n_lines, dtype=np.float32)
    
    # 步骤1: 计算成品轨迹 S[k]
    net_flow = Q_prod - Q_out_h  # 每天净产出
    
    S = np.zeros(horizon, dtype=np.float32)
    S[0] = I_output_now  # 第0天只有当前成品
    if horizon > 1:
        cumulative_net = np.cumsum(net_flow[:-1])  # 前 k-1 天的累计净产出
        S[1:] = I_output_now + cumulative_net
    
    # 步骤2: 逆向累积最小值
    reversed_S = S[::-1]
    reversed_cummin = np.minimum.accumulate(reversed_S)
    Q_safe_sell_h = reversed_cummin[::-1]
    Q_safe_sell_h = np.maximum(Q_safe_sell_h, 0)

    Q_safe_sell = np.zeros(temporal_len, dtype=np.float32)
    if horizon > 0 and len(Q_safe_sell_h) > 0:
        Q_safe_sell[:horizon] = Q_safe_sell_h
        Q_safe_sell[horizon] = Q_safe_sell_h[-1]
    return Q_safe_sell.astype(np.float32)


def compute_time_mask_offline(
    state_dict: Dict[str, Any],
    horizon: int = 40,
    min_tradable_qty: float = 1.0,
    is_buying: bool = True,
) -> np.ndarray:
    """从日志状态离线计算 L1 time_mask.
    
    严格复现 l1_safety.py 的逻辑：
    time_mask[delta] = 0.0   if Q_safe[delta] >= min_tradable_qty
    time_mask[delta] = -inf  if Q_safe[delta] < min_tradable_qty
    
    Args:
        state_dict: 日志状态字典，包含:
            - inventory: {'input': x, 'output': y} 或 sum 值
            - n_lines: 生产线数量
            - commitments: {'Q_in': [...], 'Q_out': [...]}
            - n_steps / current_step: 用于计算库容
        horizon: 规划视界
        min_tradable_qty: 最小可交易数量
        is_buying: 是否为买入角色（True=买侧用Q_safe，False=卖侧用Q_safe_sell）
        
    Returns:
        time_mask: shape (H+1,)，值为 0 或 -inf
    """
    if is_buying:
        Q_safe = compute_q_safe_offline(state_dict, horizon=horizon)
    else:
        Q_safe = compute_q_safe_sell_offline(state_dict, horizon=horizon)
    
    # ========== 生成 time_mask ==========
    time_mask = np.where(Q_safe >= min_tradable_qty, 0.0, -np.inf)
    
    return time_mask.astype(np.float32)


def _compute_b_free_from_state(state_dict: Dict[str, Any], reserve: float = 1000.0) -> float:
    """按 L1 口径离线计算可用资金（不做裁剪）。"""
    balance = float(state_dict.get('balance', 0.0))
    commitments = state_dict.get('commitments', {}) or {}
    payables = np.asarray(commitments.get('Payables', []), dtype=np.float32)
    total_payables = float(np.sum(payables)) if payables.size > 0 else 0.0
    return max(0.0, balance - reserve - total_payables)


def _aggregate_deals_by_bucket(
    deals: List[Dict[str, Any]],
    day: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """按 bucket 聚合当天成交，并计算 Q_safe/B_remaining 需要的扣减量。"""
    bought_by_bucket = np.zeros(4, dtype=np.float32)
    sold_by_bucket = np.zeros(4, dtype=np.float32)
    committed_buy_qty = np.zeros(horizon + 1, dtype=np.float32)
    committed_sell_qty = np.zeros(horizon + 1, dtype=np.float32)
    committed_buy_budget = 0.0

    for deal in deals or []:
        try:
            qty = float(deal.get('quantity', 0))
            price = float(deal.get('price', 0))
            delivery_time = int(deal.get('delivery_time', deal.get('time', 0)))
            is_buying = bool(deal.get('is_buying', False))
        except Exception:
            continue

        delta_t = max(0, min(int(delivery_time) - int(day), horizon))
        bucket = delta_to_bucket(delta_t)

        if is_buying:
            bought_by_bucket[bucket] += qty
            committed_buy_qty[delta_t] += qty
            committed_buy_budget += qty * price
        else:
            sold_by_bucket[bucket] += qty
            committed_sell_qty[delta_t] += qty

    return bought_by_bucket, sold_by_bucket, committed_buy_qty, committed_sell_qty, float(committed_buy_budget)


def build_role_embedding(is_buying: bool) -> np.ndarray:
    """构建角色 One-hot 嵌入.
    
    与 state_builder._build_role() 保持一致：
    [1, 0] = Buyer
    [0, 1] = Seller
    """
    if is_buying:
        return np.array([1.0, 0.0], dtype=np.float32)
    else:
        return np.array([0.0, 1.0], dtype=np.float32)


# ============== 日志加载 ==============

def load_world_stats(log_dir: str) -> Dict[str, Any]:
    """加载 world_stats.json."""
    stats_path = os.path.join(log_dir, "world_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
    return {}


def load_contracts(log_dir: str) -> List[Dict[str, Any]]:
    """加载合约日志（支持 JSON 和 CSV）."""
    contracts = []
    
    # 优先尝试 CSV 格式（run_default_std 输出）
    csv_path = os.path.join(log_dir, "contracts.csv")
    if PANDAS_AVAILABLE and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                contracts.append({
                    'signed_at': row.get('signed_at', row.get('sim_step', 0)),
                    'buyer': row.get('buyer', ''),
                    'seller': row.get('seller', ''),
                    'quantity': row.get('quantity', 0),
                    'unit_price': row.get('unit_price', 0),
                    'delivery_time': row.get('delivery_time', row.get('time', 0)),
                })
            return contracts
        except Exception as e:
            print(f"[WARN] Failed to load contracts.csv: {e}")
    
    # 回退到 JSON 格式
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


def load_stats_csv(log_dir: str, agent_name: str) -> Dict[str, Any]:
    """从 stats.csv 加载世界状态（run_default_std 输出格式）.
    
    Args:
        log_dir: 世界日志目录
        agent_name: 代理名称前缀（如 'LitaAgent' 或 'Pe' 表示 PenguinAgent）
        
    Returns:
        包含 n_steps, balance_history, inventory_history 等的字典
    """
    if not PANDAS_AVAILABLE:
        return {}
    
    csv_path = os.path.join(log_dir, "stats.csv")
    if not os.path.exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to load stats.csv: {e}")
        return {}
    
    n_steps = len(df)
    result = {'n_steps': n_steps, 'balance_history': {}, 'inventory_history': {}}
    
    # 查找匹配代理名称的列
    agent_cols = [c for c in df.columns if agent_name.lower() in c.lower()]
    if not agent_cols:
        # 尝试宽松匹配
        agent_cols = [c for c in df.columns if 'balance_' in c.lower()]
    
    # 提取第一个匹配的代理的数据
    balance_col = next((c for c in agent_cols if 'balance_' in c), None)
    inv_input_col = next((c for c in agent_cols if 'inventory_input_' in c), None)
    inv_output_col = next((c for c in agent_cols if 'inventory_output_' in c), None)
    
    for idx, row in df.iterrows():
        step = int(row.get('index', idx))
        
        if balance_col and balance_col in row:
            result['balance_history'][str(step)] = row[balance_col]
        
        result['inventory_history'][str(step)] = {
            'input': row.get(inv_input_col, 0) if inv_input_col else 0,
            'output': row.get(inv_output_col, 0) if inv_output_col else 0,
        }
    
    return result


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


def parse_negotiation_history(history_str: str) -> List[Dict[str, Any]]:
    """解析 negotiations.csv 中的 history 字段（JSON 字符串）.
    
    history 字段包含每轮谈判的完整状态记录。
    """
    if not history_str or history_str == '[]':
        return []
    
    try:
        # history 是一个 JSON 数组字符串
        import ast
        history = ast.literal_eval(history_str)
        return history if isinstance(history, list) else []
    except Exception:
        return []


def parse_offers_dict(offers_str: str) -> Dict[str, List]:
    """解析 negotiations.csv 中的 offers 字段.
    
    offers 字段格式: {'agent_id': [(q, t, p), ...], ...}
    """
    if not offers_str or offers_str == '{}':
        return {}
    
    try:
        import ast
        offers = ast.literal_eval(offers_str)
        return offers if isinstance(offers, dict) else {}
    except Exception:
        return {}


# ============== 状态重建 ==============

# 归一化参数（与 state_builder.py 保持一致）
DEFAULT_INITIAL_BALANCE = 10000.0
DEFAULT_MAX_PRICE = 50.0


def extract_macro_state(
    day_log: Dict[str, Any],
    horizon: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """从日志重建宏观状态（与 state_builder.py 对齐）.
    
    状态结构：
    - 静态特征 (12,)：与 StateBuilder._build_static() 一致
    - 时序状态 ((H+1), 10)：与 StateBuilder._build_temporal() 一致
    
    静态特征索引：
    | 0 | balance_norm | B_t / B_initial |
    | 1 | inventory_raw | 原材料库存 / max_inventory |
    | 2 | inventory_product | 成品库存 / max_inventory |
    | 3 | step_progress | t / T_max |
    | 4 | n_lines | 生产线数量 / 10 |
    | 5 | production_cost | 单位成本 / max_price |
    | 6 | spot_price_in | 原材料价格 / max_price |
    | 7 | spot_price_out | 成品价格 / max_price |
    | 8 | pending_buy_qty | 待买量 / max_inventory |
    | 9 | pending_sell_qty | 待卖量 / max_inventory |
    | 10 | pending_buy_value | 待买值 / initial_balance |
    | 11 | pending_sell_value | 待卖值 / initial_balance |
    
    时序通道索引：
    | 0 | vol_in | 到货量 |
    | 1 | vol_out | 发货量 |
    | 2 | prod_plan | 生产消耗 |
    | 3 | inventory_proj | 库存投影 |
    | 4 | capacity_free | 自由库容 |
    | 5 | balance_proj | 资金投影 |
    | 6 | price_diff_in | 采购侧期货溢价 |
    | 7 | price_diff_out | 销售侧期货溢价 |
    | 8 | buy_pressure | 买方压力 |
    | 9 | sell_pressure | 卖方压力 |
    
    Args:
        day_log: 包含当天信息的日志字典
        horizon: 规划视界
        
    Returns:
        (state_static, state_temporal)
    """
    temporal_len = horizon + 1
    
    # 提取日志字段
    balance = float(day_log.get('balance', 10000))
    initial_balance = float(day_log.get('initial_balance', DEFAULT_INITIAL_BALANCE))
    inventory = day_log.get('inventory', {})
    step = int(day_log.get('current_step', 0))
    n_steps = int(day_log.get('n_steps', 100))
    n_lines = int(day_log.get('n_lines', 1))
    production_cost = float(day_log.get('production_cost', 1.0))
    spot_price_in = float(day_log.get('spot_price_in', 10.0))
    spot_price_out = float(day_log.get('spot_price_out', 20.0))
    
    # 计算归一化分母（与 state_builder 一致）
    max_inventory = n_lines * n_steps  # 经济容量
    max_price = DEFAULT_MAX_PRICE
    
    # ============== 静态状态 (12 维) ==============
    state_static = np.zeros(12, dtype=np.float32)
    
    inventory_raw = float(inventory.get('input', 0))
    inventory_product = float(inventory.get('output', 0))
    
    pending = day_log.get('pending_contracts', {})
    buy_qty = float(pending.get('buy_qty', 0))
    sell_qty = float(pending.get('sell_qty', 0))
    buy_value = float(pending.get('buy_value', 0))
    sell_value = float(pending.get('sell_value', 0))
    
    state_static[0] = np.clip(balance / max(initial_balance, 1.0), 0.0, 2.0)
    state_static[1] = np.clip(inventory_raw / max(max_inventory, 1.0), 0.0, 1.0)
    state_static[2] = np.clip(inventory_product / max(max_inventory, 1.0), 0.0, 1.0)
    state_static[3] = step / max(n_steps, 1)
    state_static[4] = np.clip(n_lines / 10.0, 0.0, 1.0)
    state_static[5] = np.clip(production_cost / max(max_price, 1.0), 0.0, 1.0)
    state_static[6] = np.clip(spot_price_in / max(max_price, 1.0), 0.0, 2.0)
    state_static[7] = np.clip(spot_price_out / max(max_price, 1.0), 0.0, 2.0)
    state_static[8] = np.clip(buy_qty / max(max_inventory, 1.0), 0.0, 1.0)
    state_static[9] = np.clip(sell_qty / max(max_inventory, 1.0), 0.0, 1.0)
    state_static[10] = np.clip(buy_value / max(initial_balance, 1.0), 0.0, 1.0)
    state_static[11] = np.clip(sell_value / max(initial_balance, 1.0), 0.0, 1.0)
    
    # ============== 时序状态 ((H+1) × 10) ==============
    state_temporal = np.zeros((temporal_len, 10), dtype=np.float32)
    
    # 从日志中提取承诺量
    commitments = day_log.get('commitments', {})
    Q_in = np.array(commitments.get('Q_in', np.zeros(temporal_len)), dtype=np.float32)
    Q_out = np.array(commitments.get('Q_out', np.zeros(temporal_len)), dtype=np.float32)
    Payables = np.array(commitments.get('Payables', np.zeros(temporal_len)), dtype=np.float32)
    Receivables = np.array(commitments.get('Receivables', np.zeros(temporal_len)), dtype=np.float32)
    
    # 补齐长度
    if len(Q_in) < temporal_len:
        Q_in = np.pad(Q_in, (0, temporal_len - len(Q_in)))
    else:
        Q_in = Q_in[:temporal_len]
    if len(Q_out) < temporal_len:
        Q_out = np.pad(Q_out, (0, temporal_len - len(Q_out)))
    else:
        Q_out = Q_out[:temporal_len]
    if len(Payables) < temporal_len:
        Payables = np.pad(Payables, (0, temporal_len - len(Payables)))
    else:
        Payables = Payables[:temporal_len]
    if len(Receivables) < temporal_len:
        Receivables = np.pad(Receivables, (0, temporal_len - len(Receivables)))
    else:
        Receivables = Receivables[:temporal_len]
    
    # 生产消耗（保守估计：满负荷）
    Q_prod = np.full(temporal_len, n_lines, dtype=np.float32)
    
    # 通道 0-2: vol_in, vol_out, prod_plan
    state_temporal[:, 0] = Q_in / max(max_inventory, 1.0)
    state_temporal[:, 1] = Q_out / max(max_inventory, 1.0)
    state_temporal[:, 2] = Q_prod / max(max_inventory, 1.0)
    
    # 通道 3: inventory_proj（原材料库存投影）
    # Q_out 是成品出库，不影响原材料库存，故不参与此计算
    I_now = inventory_raw
    net_flow = Q_in - Q_prod
    I_proj = I_now + np.cumsum(net_flow)
    state_temporal[:, 3] = np.clip(I_proj / max(max_inventory, 1.0), -1.0, 2.0)
    
    # 通道 4: capacity_free（自由库容）
    # 修复：使用与在线一致的 C_total 公式
    C_total = np.zeros(temporal_len, dtype=np.float32)
    for k in range(temporal_len):
        remaining_days = n_steps - (step + k)
        C_total[k] = n_lines * max(0, remaining_days)
    C_free = C_total - I_proj
    state_temporal[:, 4] = np.clip(C_free / max(max_inventory, 1.0), -1.0, 2.0)
    
    # 通道 5: balance_proj（资金投影）
    # 现金流 = 收款 - 付款 - 生产成本
    cash_flow = Receivables - Payables - Q_prod * production_cost
    B_proj = balance + np.cumsum(cash_flow)
    state_temporal[:, 5] = np.clip(B_proj / max(initial_balance, 1.0), -1.0, 2.0)
    
    # 通道 6-9: price_diff_in/out, buy_pressure, sell_pressure
    # 优先使用 offers_snapshot_by_day（轮次衰减权重）
    # 否则回退到 tracker offers_snapshot / commitments 预计算值
    offers_snapshot = day_log.get('offers_snapshot', {'buy': [], 'sell': []})
    
    if offers_snapshot.get('buy') or offers_snapshot.get('sell'):
        # 使用 offers_snapshot 重建价格趋势与压力
        price_diff_in, price_diff_out, buy_pressure, sell_pressure = _rebuild_channels_from_offers(
            offers_snapshot=offers_snapshot,
            commitments=commitments,
            spot_price_in=spot_price_in,
            spot_price_out=spot_price_out,
            step=step,
            temporal_len=temporal_len,
            max_inventory=max_inventory,
            max_price=max_price,
            n_lines=n_lines,
            n_steps=n_steps,
        )
        state_temporal[:, 6] = price_diff_in
        state_temporal[:, 7] = price_diff_out
        state_temporal[:, 8] = buy_pressure
        state_temporal[:, 9] = sell_pressure
    else:
        # 回退: 从 commitments 中读取预计算值（如果有）
        price_diff_in = np.array(commitments.get('price_diff_in', commitments.get('price_diff', np.zeros(temporal_len))), dtype=np.float32)
        price_diff_out = np.array(commitments.get('price_diff_out', commitments.get('price_diff', np.zeros(temporal_len))), dtype=np.float32)
        buy_pressure = np.array(commitments.get('buy_pressure', np.zeros(temporal_len)), dtype=np.float32)
        sell_pressure = np.array(commitments.get('sell_pressure', np.zeros(temporal_len)), dtype=np.float32)
        
        if len(price_diff_in) >= temporal_len:
            state_temporal[:, 6] = np.clip(price_diff_in[:temporal_len] / max(max_price, 1.0), -1.0, 1.0)
        if len(price_diff_out) >= temporal_len:
            state_temporal[:, 7] = np.clip(price_diff_out[:temporal_len] / max(max_price, 1.0), -1.0, 1.0)
        if len(buy_pressure) >= temporal_len:
            state_temporal[:, 8] = buy_pressure[:temporal_len]  # 已归一化到 [0, 1]
        if len(sell_pressure) >= temporal_len:
            state_temporal[:, 9] = sell_pressure[:temporal_len]  # 已归一化到 [0, 1]
    
    return state_static, state_temporal


def _rebuild_channels_from_offers(
    offers_snapshot: Dict[str, List],
    commitments: Dict[str, Any],
    spot_price_in: float,
    spot_price_out: float,
    step: int,
    temporal_len: int,
    max_inventory: float,
    max_price: float,
    n_lines: int = 1,
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从活跃谈判快照重建价格趋势与压力（采购/销售拆分）.
    
    返回：
        price_diff_in, price_diff_out, buy_pressure, sell_pressure
    
    对齐 state_builder.py：
    - price_diff_in：使用买入谈判报价 + 采购承诺（Payables/Q_in），基于 spot_price_in
    - price_diff_out：使用卖出谈判报价 + 销售承诺（Receivables/Q_out），基于 spot_price_out
    - buy_pressure：输出市场买方需求 = Q_out + 卖出谈判报价量，以 spot_price_out 为基准价格加权
    - sell_pressure：输入市场卖方供给 = Q_in + 买入谈判报价量，以 spot_price_in 为基准价格加权
    """
    # 初始化
    price_diff_in = np.zeros(temporal_len, dtype=np.float32)
    price_diff_out = np.zeros(temporal_len, dtype=np.float32)
    buy_pressure = np.zeros(temporal_len, dtype=np.float32)
    sell_pressure = np.zeros(temporal_len, dtype=np.float32)
    
    # 已签承诺（基线）
    Q_in = np.array(commitments.get('Q_in', np.zeros(temporal_len)), dtype=np.float32)
    Q_out = np.array(commitments.get('Q_out', np.zeros(temporal_len)), dtype=np.float32)
    Payables = np.array(commitments.get('Payables', np.zeros(temporal_len)), dtype=np.float32)
    Receivables = np.array(commitments.get('Receivables', np.zeros(temporal_len)), dtype=np.float32)
    
    if len(Q_in) < temporal_len:
        Q_in = np.pad(Q_in, (0, temporal_len - len(Q_in)))
    else:
        Q_in = Q_in[:temporal_len]
    if len(Q_out) < temporal_len:
        Q_out = np.pad(Q_out, (0, temporal_len - len(Q_out)))
    else:
        Q_out = Q_out[:temporal_len]
    if len(Payables) < temporal_len:
        Payables = np.pad(Payables, (0, temporal_len - len(Payables)))
    else:
        Payables = Payables[:temporal_len]
    if len(Receivables) < temporal_len:
        Receivables = np.pad(Receivables, (0, temporal_len - len(Receivables)))
    else:
        Receivables = Receivables[:temporal_len]
    
    # 权重配置（与在线一致）
    W_SIGNED = 0.6
    W_ACTIVE = 0.3
    W_SPOT = 0.1
    
    # 经济容量 C_total[k] = n_lines * remaining_days
    C_total = np.zeros(temporal_len, dtype=np.float32)
    for k in range(temporal_len):
        remaining_days = n_steps - (step + k)
        C_total[k] = n_lines * max(0, remaining_days)
    safe_cap = np.maximum(C_total, 1.0)
    
    def _weighted_avg(values: List[Tuple[float, float]]) -> Optional[float]:
        if not values:
            return None
        total_w = sum(w for _, w in values)
        if total_w <= 0:
            return None
        return sum(p * w for p, w in values) / total_w

    # ====== 处理卖出谈判（我方卖出 -> 买方需求） -> buy_pressure、price_diff_out ======
    # buy_pressure: 买方需求强度 → 我在 output market 卖出 → 基于 Q_out + spot_price_out
    weighted_demand = Q_out.copy()
    sell_offers_by_delta: Dict[int, List[Tuple[float, float]]] = {}
    
    for offer in offers_snapshot.get('sell', []):
        try:
            delivery_time, quantity, unit_price = offer[0], offer[1], offer[2]
            weight = float(offer[3]) if len(offer) > 3 else 1.0
            delta = int(delivery_time) - step
            if 0 <= delta < temporal_len:
                price_weight = max(0.5, min(2.0, unit_price / max(spot_price_out, 1.0)))
                weighted_demand[delta] += quantity * weight * price_weight
                
                if delta not in sell_offers_by_delta:
                    sell_offers_by_delta[delta] = []
                sell_offers_by_delta[delta].append((unit_price, weight))
        except (IndexError, TypeError, ValueError):
            continue
    
    # ====== 处理买入谈判（我方买入 -> 卖方供给） -> sell_pressure、price_diff_in ======
    # sell_pressure: 卖方供给强度 → 我在 input market 买入 → 基于 Q_in + spot_price_in
    weighted_supply = Q_in.copy()
    buy_offers_by_delta: Dict[int, List[Tuple[float, float]]] = {}
    
    for offer in offers_snapshot.get('buy', []):
        try:
            delivery_time, quantity, unit_price = offer[0], offer[1], offer[2]
            weight = float(offer[3]) if len(offer) > 3 else 1.0
            delta = int(delivery_time) - step
            if 0 <= delta < temporal_len:
                price_ratio = max(spot_price_in, 1.0) / max(unit_price, 1.0)
                price_weight = max(0.5, min(2.0, price_ratio))
                weighted_supply[delta] += quantity * weight * price_weight
                
                if delta not in buy_offers_by_delta:
                    buy_offers_by_delta[delta] = []
                buy_offers_by_delta[delta].append((unit_price, weight))
        except (IndexError, TypeError, ValueError):
            continue
    
    # ====== 计算 price_diff_in / price_diff_out ======
    for k in range(temporal_len):
        # 采购侧
        vwap_in = None
        if Q_in[k] > 0:
            vwap_in = Payables[k] / max(Q_in[k], 1e-6)
        mid_active_in = _weighted_avg(buy_offers_by_delta.get(k, []))
        
        if vwap_in is not None and mid_active_in is not None:
            p_future_in = W_SIGNED * vwap_in + W_ACTIVE * mid_active_in + W_SPOT * spot_price_in
        elif vwap_in is not None:
            p_future_in = (W_SIGNED + W_ACTIVE/2) * vwap_in + (W_SPOT + W_ACTIVE/2) * spot_price_in
        elif mid_active_in is not None:
            p_future_in = (W_ACTIVE + W_SIGNED/2) * mid_active_in + (W_SPOT + W_SIGNED/2) * spot_price_in
        else:
            p_future_in = spot_price_in
        price_diff_in[k] = (p_future_in - spot_price_in) / max(max_price, 1.0)
        
        # 销售侧
        vwap_out = None
        if Q_out[k] > 0:
            vwap_out = Receivables[k] / max(Q_out[k], 1e-6)
        mid_active_out = _weighted_avg(sell_offers_by_delta.get(k, []))
        
        if vwap_out is not None and mid_active_out is not None:
            p_future_out = W_SIGNED * vwap_out + W_ACTIVE * mid_active_out + W_SPOT * spot_price_out
        elif vwap_out is not None:
            p_future_out = (W_SIGNED + W_ACTIVE/2) * vwap_out + (W_SPOT + W_ACTIVE/2) * spot_price_out
        elif mid_active_out is not None:
            p_future_out = (W_ACTIVE + W_SIGNED/2) * mid_active_out + (W_SPOT + W_SIGNED/2) * spot_price_out
        else:
            p_future_out = spot_price_out
        price_diff_out[k] = (p_future_out - spot_price_out) / max(max_price, 1.0)
    
    # ====== 压力归一化 ======
    buy_pressure = np.clip(weighted_demand / safe_cap, 0.0, 1.0)
    sell_pressure = np.clip(weighted_supply / safe_cap, 0.0, 1.0)
    
    price_diff_in = np.clip(price_diff_in, -1.0, 1.0)
    price_diff_out = np.clip(price_diff_out, -1.0, 1.0)
    
    return price_diff_in, price_diff_out, buy_pressure, sell_pressure


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
    if n_buckets != len(BUCKET_RANGES):
        raise ValueError(f"n_buckets={n_buckets} must match BUCKET_RANGES={len(BUCKET_RANGES)}")

    # v2 标签超参（保持 16 维，尽量降低稀疏/噪声）
    W_SIGNED = 0.65
    W_ACTIVE = 0.25
    W_SPOT = 0.10
    SMOOTH_WIDTH = 1.0
    GAP_ALPHA_BUY = 0.6
    GAP_BETA_SELL = 0.6
    INTENT_SCALE = 0.15  # “活跃未成交”弱意图注入系数
    BUY_FALLBACK_MARGIN = 0.95
    SELL_FALLBACK_MARGIN = 1.05

    # 将缺口分摊到桶：按桶代表交期的紧迫度（越近越大）
    rep_deltas = [0.5 * (lo + hi) for lo, hi in BUCKET_RANGES]
    rep_urg = np.array([1.0 / (d + 1.0) for d in rep_deltas], dtype=np.float32)
    rep_urg = rep_urg / max(float(rep_urg.sum()), 1e-6)

    samples: List[MacroSample] = []

    for day_log in daily_logs:
        state_static, state_temporal = extract_macro_state(day_log)

        current_step = int(day_log.get('current_step', 0) or 0)
        spot_price_in = float(day_log.get('spot_price_in', 10.0) or 10.0)
        spot_price_out = float(day_log.get('spot_price_out', 20.0) or 20.0)

        # ============== x_role（Multi-Hot：谈判能力） ==============
        my_input_product = int(day_log.get('my_input_product', 0) or 0)
        my_output_product = int(day_log.get('my_output_product', 1) or 1)
        n_products = int(day_log.get('n_products', my_output_product + 1) or (my_output_product + 1))

        is_first_level = (my_input_product == 0)
        is_last_level = (my_output_product == n_products - 1)
        can_buy = 0.0 if is_first_level else 1.0
        can_sell = 0.0 if is_last_level else 1.0
        x_role = np.array([can_buy, can_sell], dtype=np.float32)

        # ============== 统计成交（软分桶） ==============
        signed_buy_qty = np.zeros(n_buckets, dtype=np.float32)
        signed_buy_value = np.zeros(n_buckets, dtype=np.float32)
        signed_sell_qty = np.zeros(n_buckets, dtype=np.float32)
        signed_sell_value = np.zeros(n_buckets, dtype=np.float32)

        deals = day_log.get('deals', []) or []
        for deal in deals:
            try:
                qty = float(deal.get('quantity', 0) or 0)
                price = float(deal.get('price', 0) or 0)
                delivery_time = int(deal.get('delivery_time', current_step) or current_step)
            except Exception:
                continue
            if qty <= 0 or price <= 0:
                continue

            delta = float(max(0, delivery_time - current_step))
            for bidx, w in delta_to_bucket_soft(delta, smooth_width=SMOOTH_WIDTH):
                if w <= 0:
                    continue
                if deal.get('is_buying', True):
                    signed_buy_qty[bidx] += qty * w
                    signed_buy_value[bidx] += qty * price * w
                else:
                    signed_sell_qty[bidx] += qty * w
                    signed_sell_value[bidx] += qty * price * w

        # ============== 统计活跃报价（用于价格稳定 + 活跃未成交意图） ==============
        offers_snapshot = day_log.get('offers_snapshot', None) or {'buy': [], 'sell': []}

        active_buy_qty = np.zeros(n_buckets, dtype=np.float32)
        active_buy_value = np.zeros(n_buckets, dtype=np.float32)
        active_buy_prices: List[List[Tuple[float, float]]] = [[] for _ in range(n_buckets)]

        active_sell_qty = np.zeros(n_buckets, dtype=np.float32)
        active_sell_value = np.zeros(n_buckets, dtype=np.float32)
        active_sell_prices: List[List[Tuple[float, float]]] = [[] for _ in range(n_buckets)]

        def _consume_offers(
            side: str,
            qty_acc: np.ndarray,
            val_acc: np.ndarray,
            prices_acc: List[List[Tuple[float, float]]],
        ) -> None:
            for offer in offers_snapshot.get(side, []) or []:
                try:
                    delivery_time = int(offer[0])
                    qty = float(offer[1])
                    price = float(offer[2])
                    w_offer = float(offer[3]) if len(offer) > 3 else 1.0
                except Exception:
                    continue
                if qty <= 0 or price <= 0 or w_offer <= 0:
                    continue
                delta = float(max(0, delivery_time - current_step))
                for bidx, w_bucket in delta_to_bucket_soft(delta, smooth_width=SMOOTH_WIDTH):
                    # qty_acc 用于“活跃未成交”意图（不带 qty 二次加权），val/median 用于价格聚合
                    qty_acc[bidx] += float(qty) * float(w_bucket) * float(w_offer)

                    w_price = float(w_bucket) * float(w_offer) * float(max(qty, 1e-6))
                    val_acc[bidx] += float(price) * w_price
                    prices_acc[bidx].append((float(price), float(w_price)))

        _consume_offers('buy', active_buy_qty, active_buy_value, active_buy_prices)
        _consume_offers('sell', active_sell_qty, active_sell_value, active_sell_prices)

        # ============== 缺口补偿（减少“无成交全 0”稀疏） ==============
        gap_buy = float(day_log.get('needed_supplies', 0) or 0)
        gap_sell = float(day_log.get('needed_sales', 0) or 0)
        gap_buy = float(max(0.0, gap_buy))
        gap_sell = float(max(0.0, gap_sell))

        gap_buy_by_bucket = gap_buy * rep_urg
        gap_sell_by_bucket = gap_sell * rep_urg

        # ============== 组装 16 维目标（Q=成交+缺口补偿+活跃意图，P=成交VWAP+活跃报价+spot 回退） ==============
        goal = np.zeros(16, dtype=np.float32)

        total_signed_buy = float(signed_buy_qty.sum())
        total_signed_sell = float(signed_sell_qty.sum())
        total_active_buy = float(active_buy_qty.sum())
        total_active_sell = float(active_sell_qty.sum())

        for b in range(n_buckets):
            base = b * 4

            # Q 目标
            q_buy = float(signed_buy_qty[b]) + GAP_ALPHA_BUY * float(gap_buy_by_bucket[b])
            q_sell = float(signed_sell_qty[b]) + GAP_BETA_SELL * float(gap_sell_by_bucket[b])

            # “活跃未成交”弱意图：仅在该侧当天无成交但活跃时注入
            if total_signed_buy <= 0 and total_active_buy > 0:
                q_buy += INTENT_SCALE * float(active_buy_qty[b])
            if total_signed_sell <= 0 and total_active_sell > 0:
                q_sell += INTENT_SCALE * float(active_sell_qty[b])

            # 价格：成交 VWAP + 活跃报价加权均值（或中位数回退）+ spot
            deal_vwap_buy = (float(signed_buy_value[b]) / float(signed_buy_qty[b])) if signed_buy_qty[b] > 0 else None
            deal_vwap_sell = (float(signed_sell_value[b]) / float(signed_sell_qty[b])) if signed_sell_qty[b] > 0 else None

            active_avg_buy = (float(active_buy_value[b]) / float(active_buy_qty[b])) if active_buy_qty[b] > 0 else None
            active_avg_sell = (float(active_sell_value[b]) / float(active_sell_qty[b])) if active_sell_qty[b] > 0 else None

            if deal_vwap_buy is None and active_avg_buy is not None:
                active_med = _weighted_median(active_buy_prices[b])
                if active_med is not None:
                    active_avg_buy = 0.5 * float(active_avg_buy) + 0.5 * float(active_med)

            if deal_vwap_sell is None and active_avg_sell is not None:
                active_med = _weighted_median(active_sell_prices[b])
                if active_med is not None:
                    active_avg_sell = 0.5 * float(active_avg_sell) + 0.5 * float(active_med)

            # 买侧价格
            if deal_vwap_buy is not None and active_avg_buy is not None:
                p_buy = W_SIGNED * deal_vwap_buy + W_ACTIVE * active_avg_buy + W_SPOT * spot_price_in
            elif deal_vwap_buy is not None:
                p_buy = (W_SIGNED + W_ACTIVE / 2.0) * deal_vwap_buy + (W_SPOT + W_ACTIVE / 2.0) * spot_price_in
            elif active_avg_buy is not None:
                p_buy = (W_ACTIVE + W_SIGNED / 2.0) * active_avg_buy + (W_SPOT + W_SIGNED / 2.0) * spot_price_in
            else:
                p_buy = spot_price_in * BUY_FALLBACK_MARGIN

            # 卖侧价格
            if deal_vwap_sell is not None and active_avg_sell is not None:
                p_sell = W_SIGNED * deal_vwap_sell + W_ACTIVE * active_avg_sell + W_SPOT * spot_price_out
            elif deal_vwap_sell is not None:
                p_sell = (W_SIGNED + W_ACTIVE / 2.0) * deal_vwap_sell + (W_SPOT + W_ACTIVE / 2.0) * spot_price_out
            elif active_avg_sell is not None:
                p_sell = (W_ACTIVE + W_SIGNED / 2.0) * active_avg_sell + (W_SPOT + W_SIGNED / 2.0) * spot_price_out
            else:
                p_sell = spot_price_out * SELL_FALLBACK_MARGIN

            q_buy = float(max(0.0, q_buy))
            q_sell = float(max(0.0, q_sell))
            p_buy = float(max(0.0, p_buy))
            p_sell = float(max(0.0, p_sell))

            goal[base + 0] = q_buy
            goal[base + 1] = p_buy
            goal[base + 2] = q_sell
            goal[base + 3] = p_sell

        # 不可谈判侧压零（避免学到“不可谈判”的假目标）
        if can_buy <= 0:
            for b in range(n_buckets):
                base = b * 4
                goal[base + 0] = 0.0
                goal[base + 1] = 0.0
        if can_sell <= 0:
            for b in range(n_buckets):
                base = b * 4
                goal[base + 2] = 0.0
                goal[base + 3] = 0.0

        value = day_log.get('daily_reward', None)

        samples.append(MacroSample(
            day=current_step,
            state_static=state_static,
            state_temporal=state_temporal,
            x_role=x_role,
            goal=goal,
            value=value
        ))

    return samples


# ============== L3 标签提取 ==============

def _normalize_goal_vector(goal_raw: Any) -> np.ndarray:
    if isinstance(goal_raw, (list, tuple, np.ndarray)):
        goal = np.array(goal_raw, dtype=np.float32).reshape(-1)
        if goal.size == 16 and not np.any(np.isnan(goal)):
            return goal
    return np.zeros(16, dtype=np.float32)


def _parse_action_op(action_op: Any) -> Optional[int]:
    if action_op is None:
        return None
    if isinstance(action_op, (int, np.integer)):
        val = int(action_op)
        return val if val in (0, 1, 2) else None
    op = str(action_op).strip().upper()
    mapping = {"ACCEPT": 0, "REJECT": 1, "END": 2, "END_NEGOTIATION": 2}
    return mapping.get(op)


def extract_l3_samples_aop(
    negotiation_logs: List[Dict[str, Any]],
    horizon: int = 40,
    state_dict: Optional[Dict[str, Any]] = None,
    daily_states: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[MicroSample]:
    """从谈判日志中提取 L3 AOP 标签."""
    samples: List[MicroSample] = []
    day_cache: Dict[int, Dict[str, Any]] = {}

    for neg in negotiation_logs:
        offer_history = neg.get('offer_history', [])
        if not offer_history:
            continue

        day = int(neg.get('sim_step', neg.get('day', 0)) or 0)
        if day not in day_cache:
            state = None
            if daily_states is not None:
                state = daily_states.get(day)
            elif state_dict is not None:
                state = state_dict

            if state is None:
                day_cache[day] = {
                    "x_static": np.zeros(12, dtype=np.float32),
                    "X_temporal": np.zeros((horizon + 1, 10), dtype=np.float32),
                    "Q_safe_buy": np.zeros(horizon + 1, dtype=np.float32),
                    "Q_safe_sell": np.zeros(horizon + 1, dtype=np.float32),
                    "time_mask_buy": np.zeros(horizon + 1, dtype=np.float32),
                    "time_mask_sell": np.zeros(horizon + 1, dtype=np.float32),
                    "B_free": 0.0,
                    "bought_by_bucket": np.zeros(4, dtype=np.float32),
                    "sold_by_bucket": np.zeros(4, dtype=np.float32),
                    "today_bought_qty": 0.0,
                    "today_sold_qty": 0.0,
                }
            else:
                x_static, X_temporal = extract_macro_state(state, horizon=horizon)
                Q_safe_buy = compute_q_safe_offline(state, horizon=horizon)
                Q_safe_sell = compute_q_safe_sell_offline(state, horizon=horizon)
                time_mask_buy = compute_time_mask_offline(state, horizon=horizon, is_buying=True)
                time_mask_sell = compute_time_mask_offline(state, horizon=horizon, is_buying=False)
                B_free = _compute_b_free_from_state(state)
                deals = state.get('deals', [])
                bought_by_bucket, sold_by_bucket, _, _, _ = _aggregate_deals_by_bucket(
                    deals, day, horizon
                )
                day_cache[day] = {
                    "x_static": x_static,
                    "X_temporal": X_temporal,
                    "Q_safe_buy": Q_safe_buy,
                    "Q_safe_sell": Q_safe_sell,
                    "time_mask_buy": time_mask_buy,
                    "time_mask_sell": time_mask_sell,
                    "B_free": B_free,
                    "bought_by_bucket": bought_by_bucket,
                    "sold_by_bucket": sold_by_bucket,
                    "today_bought_qty": float(np.sum(bought_by_bucket)),
                    "today_sold_qty": float(np.sum(sold_by_bucket)),
                }

        cached = day_cache[day]

        history_rows = []
        current_offer = np.zeros(3, dtype=np.float32)
        for h in offer_history:
            try:
                qty = float(h.get('quantity', 0))
                price = float(h.get('price', 0))
                delta_t = int(h.get('delta_t', 0))
            except Exception:
                qty, price, delta_t = 0.0, 0.0, 0
            delta_t = max(0, min(delta_t, horizon))
            is_my_turn = 1.0 if h.get('source') == 'self' else 0.0
            history_rows.append([qty, price, float(delta_t), is_my_turn])
            current_offer = np.array([qty, price, float(delta_t)], dtype=np.float32)

        history = np.asarray(history_rows, dtype=np.float32)

        is_buyer = bool(neg.get('is_buyer', True))
        role = 0 if is_buyer else 1

        l2_goal = _normalize_goal_vector(neg.get('l2_goal'))
        goal_gap_buy = np.zeros(4, dtype=np.float32)
        goal_gap_sell = np.zeros(4, dtype=np.float32)
        for i in range(4):
            goal_gap_buy[i] = max(0.0, float(l2_goal[i * 4]) - float(cached["bought_by_bucket"][i]))
            goal_gap_sell[i] = max(0.0, float(l2_goal[i * 4 + 2]) - float(cached["sold_by_bucket"][i]))

        action_op = _parse_action_op(neg.get('action_op'))
        if action_op is None:
            continue

        action_offer = neg.get('action_offer') or neg.get('final_action') or {}
        target_quantity = None
        target_price = None
        target_time = None
        if action_op in (0, 1) and action_offer:
            try:
                target_quantity = int(action_offer.get('quantity', action_offer.get('q', 0)))
                target_price = float(action_offer.get('price', action_offer.get('unit_price', action_offer.get('p', 0.0))))
                target_time = int(action_offer.get('delta_t', action_offer.get('t', 0)))
            except Exception:
                target_quantity = None
                target_price = None
                target_time = None

        reward = neg.get('reward', None)
        if reward is None:
            reward = 1.0 if neg.get('final_status') == 'succeeded' else 0.0

        samples.append(MicroSample(
            negotiation_id=neg.get('id', str(len(samples))),
            history=history,
            role=role,
            current_offer=current_offer,
            relative_time=float(neg.get('relative_time', 0.0) or 0.0),
            l2_goal=l2_goal,
            goal_gap_buy=goal_gap_buy,
            goal_gap_sell=goal_gap_sell,
            today_bought_qty=float(cached["today_bought_qty"]),
            today_sold_qty=float(cached["today_sold_qty"]),
            today_bought_by_bucket=np.asarray(cached["bought_by_bucket"], dtype=np.float32),
            today_sold_by_bucket=np.asarray(cached["sold_by_bucket"], dtype=np.float32),
            x_static=np.asarray(cached["x_static"], dtype=np.float32),
            X_temporal=np.asarray(cached["X_temporal"], dtype=np.float32),
            time_mask=np.asarray(cached["time_mask_buy"] if is_buyer else cached["time_mask_sell"], dtype=np.float32),
            Q_safe=np.asarray(cached["Q_safe_buy"] if is_buyer else cached["Q_safe_sell"], dtype=np.float32),
            B_free=float(cached["B_free"]),
            action_op=action_op,
            target_quantity=target_quantity,
            target_price=target_price,
            target_time=target_time,
            reward=reward,
        ))

    return samples


def extract_l3_residuals(
    negotiation_logs: List[Dict[str, Any]],
    horizon: int = 40,
    state_dict: Optional[Dict[str, Any]] = None,
    daily_states: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[MicroSample]:
    """兼容旧接口：改为输出 AOP 样本."""
    return extract_l3_samples_aop(
        negotiation_logs,
        horizon=horizon,
        state_dict=state_dict,
        daily_states=daily_states,
    )


# ============== 批量处理 ==============

def _find_world_dirs(tournament_dir: str) -> List[str]:
    """查找所有世界目录和 Tracker 日志目录（支持多种命名模式）.
    
    支持的格式：
    
    1. hrl_data_runner 输出格式（Tracker JSON，推荐）：
        tournament_dir/
            tracker_logs/         # Tracker 日志目录
                agent_*.json      # 完整状态快照
    
    2. run_default_std 输出格式（CSV）：
        tournament_dir/
            <world_id>/           # 如 000020251216H...
                <world_id>/       # 嵌套目录包含实际日志
                    stats.csv
                    negotiations.csv
                    contracts.csv
    
    3. 传统 tracked runner 输出格式：
        tournament_dir/
            world_*/              # 如 world_0001
                agent_*.json      # tracker 事件日志
    """
    world_dirs: List[str] = []
    seen: set[str] = set()
    
    # 模式 1: tracker_logs 目录（hrl_data_runner 默认输出）
    # 注意：tracker_logs 目录通常包含多个 world/run 的 agent_*.json。
    # 为了避免跨 world 混合并提升并行度，这里直接把每个 agent_*.json 当作一个“world 单元”。
    # 这也与后续 _process_world_dir 对单文件的处理路径对齐。
    tracker_log_dirs = glob.glob(os.path.join(tournament_dir, "**", "tracker_logs"), recursive=True)
    for tracker_dir in tracker_log_dirs:
        if os.path.isdir(tracker_dir):
            # 检查是否包含 agent_*.json
            agent_files = glob.glob(os.path.join(tracker_dir, "agent_*.json"))
            for f in agent_files:
                if f not in seen:
                    world_dirs.append(f)
                    seen.add(f)
    
    # 模式 2: run_default_std 格式（嵌套目录，包含 stats.csv）
    for parent in os.listdir(tournament_dir):
        parent_path = os.path.join(tournament_dir, parent)
        if not os.path.isdir(parent_path):
            continue
        
        # 检查嵌套目录
        for child in os.listdir(parent_path):
            child_path = os.path.join(parent_path, child)
            if os.path.isdir(child_path):
                # 检查是否有 stats.csv 或 negotiations.csv
                if (os.path.exists(os.path.join(child_path, "stats.csv")) or
                    os.path.exists(os.path.join(child_path, "negotiations.csv"))):
                    if child_path not in seen:
                        world_dirs.append(child_path)
                        seen.add(child_path)
    
    # 模式 3: 传统 world_* 格式（JSON）
    pattern_dirs = glob.glob(os.path.join(tournament_dir, "**", "world_*"), recursive=True)
    for d in pattern_dirs:
        if os.path.isdir(d) and d not in seen:
            world_dirs.append(d)
            seen.add(d)
    
    # 模式 4: 目录根部直接包含 agent_*.json
    agent_files = glob.glob(os.path.join(tournament_dir, "agent_*.json"))
    if agent_files:
        if os.path.basename(tournament_dir).lower() == "tracker_logs":
            for f in agent_files:
                if f not in seen:
                    world_dirs.append(f)
                    seen.add(f)
        else:
            if tournament_dir not in seen:
                world_dirs.append(tournament_dir)
                seen.add(tournament_dir)
    
    return world_dirs


_L2_INFER_CACHE: Dict[Tuple[str, int], Any] = {}


def _load_l2_model_for_backfill(model_path: str, horizon: int):
    """加载 L2 模型用于 goal_hat 回填（进程内缓存）。"""
    key = (os.path.abspath(model_path), int(horizon))
    if key in _L2_INFER_CACHE:
        return _L2_INFER_CACHE[key]

    try:
        import torch  # 延迟导入，避免无 torch 环境下影响数据管道
    except Exception as exc:
        raise RuntimeError("goal_hat 回填需要 PyTorch") from exc

    from .l2_manager import HorizonManagerPPO  # 延迟导入，避免循环依赖

    model = HorizonManagerPPO(horizon=int(horizon))
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    _L2_INFER_CACHE[key] = model
    return model


def _predict_goal_hat_by_day(
    macro_samples: List[MacroSample],
    *,
    model_path: str,
) -> Dict[int, np.ndarray]:
    """用已训练 L2 对每个 day 的宏观状态预测 goal_hat（16 维）。"""
    if not macro_samples:
        return {}

    try:
        import torch  # 延迟导入
    except Exception as exc:
        raise RuntimeError("goal_hat 回填需要 PyTorch") from exc

    horizon = int(macro_samples[0].state_temporal.shape[0] - 1)
    model = _load_l2_model_for_backfill(model_path, horizon=horizon)

    goal_hat_by_day: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for s in macro_samples:
            x_static_t = torch.from_numpy(np.asarray(s.state_static, dtype=np.float32)).unsqueeze(0)
            x_temporal_t = torch.from_numpy(np.asarray(s.state_temporal, dtype=np.float32)).unsqueeze(0)
            x_role_t = torch.from_numpy(np.asarray(s.x_role, dtype=np.float32)).unsqueeze(0)

            action = model.get_deterministic_action(x_static_t, x_temporal_t, x_role_t)
            goal_hat = action.squeeze(0).cpu().numpy().astype(np.float32)

            # 与训练掩码一致：不可谈判侧压零，避免将噪声回填给 L3
            can_buy = float(s.x_role[0]) if hasattr(s.x_role, "__len__") else 1.0
            can_sell = float(s.x_role[1]) if hasattr(s.x_role, "__len__") else 1.0
            if can_buy <= 0:
                goal_hat[[0, 1, 4, 5, 8, 9, 12, 13]] = 0.0
            if can_sell <= 0:
                goal_hat[[2, 3, 6, 7, 10, 11, 14, 15]] = 0.0

            goal_hat_by_day[int(s.day)] = goal_hat

    return goal_hat_by_day


def _backfill_negotiation_goals(
    neg_logs: List[Dict[str, Any]],
    macro_samples: List[MacroSample],
    *,
    goal_backfill: str,
    l2_model_path: Optional[str],
) -> None:
    """将 L2 目标回填到谈判日志（用于生成 micro.l2_goal）。"""
    mode = (goal_backfill or "none").lower().strip()
    if mode in ("none", "off", "false", "0", ""):
        return
    if not neg_logs or not macro_samples:
        return

    if mode in ("v2", "macro"):
        by_day = {int(s.day): np.asarray(s.goal, dtype=np.float32) for s in macro_samples}
    elif mode in ("l2", "goal_hat", "model"):
        if not l2_model_path:
            raise ValueError("goal_backfill='l2' 需要提供 l2_model_path")
        by_day = _predict_goal_hat_by_day(macro_samples, model_path=l2_model_path)
    else:
        raise ValueError(f"Unknown goal_backfill mode: {goal_backfill}")

    for neg in neg_logs:
        try:
            day = int(neg.get("sim_step", neg.get("day", 0)) or 0)
        except Exception:
            day = 0
        g = by_day.get(day)
        if g is None:
            continue
        neg["l2_goal"] = np.asarray(g, dtype=np.float32).reshape(-1).tolist()


def _encode_l4_global_state(
    goal_gap_buy: np.ndarray,
    goal_gap_sell: np.ndarray,
    *,
    step_progress: float,
    B_remaining: float,
    n_buy: int,
    n_sell: int,
) -> np.ndarray:
    features = [
        float(step_progress),
        float(B_remaining) / 10000.0,
        *[float(g) / 10.0 for g in goal_gap_buy],
        *[float(g) / 10.0 for g in goal_gap_sell],
        float(n_buy) / 10.0,
        float(n_sell) / 10.0,
    ]
    return np.asarray(features, dtype=np.float32)


def _extract_l4_distill_samples_for_world(
    *,
    daily_states: Dict[int, Dict[str, Any]],
    neg_logs: List[Dict[str, Any]],
    macro_samples: List[MacroSample],
    goal_source: str,
    l2_model_path: Optional[str],
    horizon: int = 40,
) -> List[L4DistillSample]:
    """从单个 world 的日志中构建 L4 蒸馏样本（按 day 聚合线程集合）。"""
    if not daily_states or not neg_logs:
        return []

    mode = (goal_source or "v2").lower().strip()
    if mode in ("v2", "macro"):
        goal_by_day = {int(s.day): np.asarray(s.goal, dtype=np.float32) for s in macro_samples}
    elif mode in ("l2", "goal_hat", "model"):
        if not l2_model_path:
            raise ValueError("goal_source='l2' 需要提供 l2_model_path")
        goal_by_day = _predict_goal_hat_by_day(macro_samples, model_path=l2_model_path)
    else:
        raise ValueError(f"Unknown goal_source: {goal_source}")

    from collections import defaultdict
    negs_by_day: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for neg in neg_logs:
        day = int(neg.get("sim_step", neg.get("day", 0)) or 0)
        negs_by_day[day].append(neg)

    from .l4_coordinator import HeuristicAlphaGenerator, ThreadState, GlobalBroadcast

    teacher = HeuristicAlphaGenerator()
    samples: List[L4DistillSample] = []

    for day, state in daily_states.items():
        if day not in negs_by_day:
            continue

        goal = goal_by_day.get(int(day), np.zeros(16, dtype=np.float32))
        x_static, X_temporal = extract_macro_state(state, horizon=horizon)
        Q_safe_buy = compute_q_safe_offline(state, horizon=horizon)
        Q_safe_sell = compute_q_safe_sell_offline(state, horizon=horizon)
        B_free = _compute_b_free_from_state(state)

        deals = state.get('deals', [])
        bought_by_bucket, sold_by_bucket, committed_buy_qty, committed_sell_qty, committed_buy_budget = _aggregate_deals_by_bucket(
            deals, day, horizon
        )

        goal_gap_buy = np.zeros(4, dtype=np.float32)
        goal_gap_sell = np.zeros(4, dtype=np.float32)
        for i in range(4):
            goal_gap_buy[i] = max(0.0, float(goal[i * 4]) - float(bought_by_bucket[i]))
            goal_gap_sell[i] = max(0.0, float(goal[i * 4 + 2]) - float(sold_by_bucket[i]))

        today_bought_qty = float(np.sum(bought_by_bucket))
        today_sold_qty = float(np.sum(sold_by_bucket))
        today_bought_value = float(sum(float(d.get('quantity', 0)) * float(d.get('price', 0)) for d in deals if d.get('is_buying')))
        today_sold_value = float(sum(float(d.get('quantity', 0)) * float(d.get('price', 0)) for d in deals if not d.get('is_buying')))

        B_remaining = max(0.0, float(B_free) - float(committed_buy_budget))
        Q_safe_remaining = np.maximum(0.0, Q_safe_buy - committed_buy_qty)
        Q_safe_sell_remaining = np.maximum(0.0, Q_safe_sell - committed_sell_qty)

        threads = negs_by_day.get(day, [])
        n_buy = sum(1 for n in threads if n.get('offer_history') and bool(n.get('is_buyer', True)))
        n_sell = sum(1 for n in threads if n.get('offer_history') and not bool(n.get('is_buyer', True)))

        gb = GlobalBroadcast(
            current_step=int(day),
            n_steps=int(state.get('n_steps', 100) or 100),
            step_progress=float(int(day) / max(1, int(state.get('n_steps', 100) or 100))),
            l2_goal=np.asarray(goal, dtype=np.float32),
            goal_gap_buy=goal_gap_buy,
            goal_gap_sell=goal_gap_sell,
            today_bought_qty=today_bought_qty,
            today_bought_value=today_bought_value,
            today_sold_qty=today_sold_qty,
            today_sold_value=today_sold_value,
            B_remaining=B_remaining,
            Q_safe_remaining=Q_safe_remaining,
            Q_safe_sell_remaining=Q_safe_sell_remaining,
            n_active_buy_threads=int(n_buy),
            n_active_sell_threads=int(n_sell),
            x_static=x_static,
            X_temporal=X_temporal,
        )

        thread_states: List[np.ndarray] = []
        teacher_alpha: List[float] = []
        thread_ids: List[str] = []

        for neg in threads:
            history = neg.get('offer_history', [])
            if not history:
                continue

            is_buyer = bool(neg.get('is_buyer', True))
            rel_time = float(neg.get('relative_time', 0.0) or 0.0)
            rounds = [h.get('round') for h in history if h.get('round') is not None]
            negotiation_step = int(max(rounds)) if rounds else len(history)

            last = history[-1]
            try:
                q_val = float(last.get('quantity', 0))
                p_val = float(last.get('price', 0))
                delta_t = int(last.get('delta_t', 0))
            except Exception:
                q_val, p_val, delta_t = 0.0, 0.0, 0
            delta_t = max(0, min(delta_t, horizon))

            bucket = delta_to_bucket(delta_t)
            gap = float(goal_gap_buy[bucket] if is_buyer else goal_gap_sell[bucket])

            if is_buyer:
                q_safe_at_t = float(Q_safe_remaining[delta_t]) if delta_t < len(Q_safe_remaining) else 0.0
                b_remaining = float(B_remaining)
            else:
                q_safe_at_t = float(Q_safe_sell_remaining[delta_t]) if delta_t < len(Q_safe_sell_remaining) else 0.0
                b_remaining = 0.0

            ts = ThreadState(
                thread_id=str(neg.get('id', '')),
                is_buying=is_buyer,
                negotiation_step=int(negotiation_step),
                relative_time=float(rel_time),
                current_offer=(int(q_val), int(delta_t), float(p_val)),
                history_len=len(history),
                target_bucket=int(bucket),
                goal_gap=float(gap),
                Q_safe_at_t=float(q_safe_at_t),
                B_remaining=float(b_remaining),
            )
            alpha = teacher.compute_alpha(ts, gb)

            feat = np.array([
                0.0 if is_buyer else 1.0,
                float(rel_time),
                float(negotiation_step) / 20.0,
                1.0,
                float(q_val) / 10.0,
                float(p_val) / 40.0,
                float(delta_t) / 40.0,
                float(bucket) / 3.0,
                float(gap) / 10.0,
                float(q_safe_at_t) / 100.0,
                float(b_remaining) / 10000.0,
            ], dtype=np.float32)

            thread_states.append(feat)
            teacher_alpha.append(float(alpha))
            thread_ids.append(str(neg.get('id', '')))

        if not thread_states:
            continue

        global_state = _encode_l4_global_state(
            goal_gap_buy,
            goal_gap_sell,
            step_progress=float(int(day) / max(1, int(state.get('n_steps', 100) or 100))),
            B_remaining=B_remaining,
            n_buy=n_buy,
            n_sell=n_sell,
        )

        samples.append(
            L4DistillSample(
                day=int(day),
                global_state=global_state,
                thread_states=np.asarray(thread_states, dtype=np.float32),
                teacher_alpha=np.asarray(teacher_alpha, dtype=np.float32),
                thread_ids=thread_ids,
            )
        )

    return samples
def _process_world_dir_l4(
    world_dir: str,
    agent_name: str,
    strict_json_only: bool,
    goal_source: str = "l2",
    l2_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """处理单个 world 目录，提取 L4 蒸馏数据（用于多进程并行）。"""
    result: Dict[str, Any] = {
        "world_dir": world_dir,
        "l4_samples": [],
        "used_tracker": False,
        "used_csv": False,
        "skipped_csv": False,
        "has_tracker": False,
        "has_csv": False,
        "error": None,
    }
    try:
        # 模式 1：直接传入单个 Tracker JSON 文件（常见于 hrl_data_runner 的 tracker_logs 拆分）
        if (
            os.path.isfile(world_dir)
            and os.path.basename(world_dir).lower().startswith("agent_")
            and world_dir.lower().endswith(".json")
        ):
            result["has_tracker"] = True
            result["has_csv"] = False

            try:
                with open(world_dir, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                return result

            agent_aliases = _extract_agent_type_aliases(data)
            if not _agent_type_matches_any(agent_name, agent_aliases):
                return result

            daily_states = _extract_daily_states_from_tracker_data(data)
            entries = data.get("entries", []) or []
            neg_logs = _parse_tracker_entries(entries, agent_name) if entries else []
            offers_snapshot_by_day = _build_offers_snapshot_by_day(neg_logs) if neg_logs else None

            if daily_states:
                daily_logs = _daily_states_to_logs(
                    daily_states,
                    agent_name,
                    offers_snapshot_by_day=offers_snapshot_by_day,
                )
                macro_samples = reconstruct_l2_goals(daily_logs)
            else:
                macro_samples = []

            result["l4_samples"] = _extract_l4_distill_samples_for_world(
                daily_states=daily_states,
                neg_logs=neg_logs,
                macro_samples=macro_samples,
                goal_source=goal_source,
                l2_model_path=l2_model_path,
            )
            result["used_tracker"] = True
            return result

        has_tracker = len(glob.glob(os.path.join(world_dir, "agent_*.json"))) > 0
        has_csv = os.path.exists(os.path.join(world_dir, "stats.csv"))
        result["has_tracker"] = has_tracker
        result["has_csv"] = has_csv

        if has_tracker:
            # 同 load_tournament_data：tracker_logs 可能包含多个 world/run，按文件拆分，避免跨 world 混合。
            if os.path.basename(world_dir).lower() == "tracker_logs":
                l4_samples: List[L4DistillSample] = []
                for _, data in _iter_tracker_json_files(world_dir, agent_name):
                    daily_states = _extract_daily_states_from_tracker_data(data)
                    entries = data.get("entries", []) or []
                    neg_logs = _parse_tracker_entries(entries, agent_name) if entries else []
                    offers_snapshot_by_day = _build_offers_snapshot_by_day(neg_logs) if neg_logs else None

                    if daily_states:
                        daily_logs = _daily_states_to_logs(
                            daily_states,
                            agent_name,
                            offers_snapshot_by_day=offers_snapshot_by_day,
                        )
                        macro_samples = reconstruct_l2_goals(daily_logs)
                    else:
                        macro_samples = []

                    l4_samples.extend(
                        _extract_l4_distill_samples_for_world(
                            daily_states=daily_states,
                            neg_logs=neg_logs,
                            macro_samples=macro_samples,
                            goal_source=goal_source,
                            l2_model_path=l2_model_path,
                        )
                    )

                result["l4_samples"] = l4_samples
                result["used_tracker"] = True

            else:
                daily_states = _load_daily_states_from_tracker(world_dir, agent_name)
                neg_logs = _load_negotiation_logs(world_dir, agent_name)
                offers_snapshot_by_day = _build_offers_snapshot_by_day(neg_logs) if neg_logs else None

                if daily_states:
                    daily_logs = _daily_states_to_logs(
                        daily_states,
                        agent_name,
                        offers_snapshot_by_day=offers_snapshot_by_day,
                    )
                    macro_samples = reconstruct_l2_goals(daily_logs)
                else:
                    macro_samples = []

                result["l4_samples"] = _extract_l4_distill_samples_for_world(
                    daily_states=daily_states,
                    neg_logs=neg_logs,
                    macro_samples=macro_samples,
                    goal_source=goal_source,
                    l2_model_path=l2_model_path,
                )
                result["used_tracker"] = True
        elif has_csv:
            if strict_json_only:
                result["skipped_csv"] = True
                return result
            stats = load_stats_csv(world_dir, agent_name)
            contracts = load_contracts(world_dir)
            daily_logs = _organize_by_day(stats, contracts, agent_name)
            macro_samples = reconstruct_l2_goals(daily_logs)
            daily_states = {int(d["current_step"]): d for d in daily_logs}
            neg_logs = _load_negotiation_logs_csv(world_dir, agent_name)
            result["l4_samples"] = _extract_l4_distill_samples_for_world(
                daily_states=daily_states,
                neg_logs=neg_logs,
                macro_samples=macro_samples,
                goal_source=goal_source,
                l2_model_path=l2_model_path,
            )
            result["used_csv"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def load_l4_distill_data(
    tournament_dir: str,
    agent_name: str = "LitaAgent",
    strict_json_only: bool = True,
    num_workers: Optional[int] = None,
    goal_source: str = "l2",
    l2_model_path: Optional[str] = None,
) -> List[L4DistillSample]:
    """从锦标赛目录加载 L4 蒸馏训练数据。"""
    world_dirs = _find_world_dirs(tournament_dir)
    if not world_dirs:
        print(f"[WARN] No world directories found in {tournament_dir}")
        return []

    if num_workers is None:
        cpu_cnt = os.cpu_count() or 1
        num_workers = max(1, cpu_cnt - 1)
    num_workers = min(num_workers, len(world_dirs))

    l4_samples: List[L4DistillSample] = []
    errors: List[Tuple[str, str]] = []

    def _merge(result: Dict[str, Any]) -> None:
        l4_samples.extend(result.get("l4_samples", []))
        if result.get("error"):
            errors.append((result.get("world_dir", ""), result["error"]))

    if num_workers <= 1:
        for world_dir in world_dirs:
            _merge(_process_world_dir_l4(world_dir, agent_name, strict_json_only, goal_source=goal_source, l2_model_path=l2_model_path))
    else:
        print(f"[INFO] Using {num_workers} workers for L4 distill data pipeline")
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            results_iter = executor.map(
                _process_world_dir_l4,
                world_dirs,
                itertools.repeat(agent_name),
                itertools.repeat(strict_json_only),
                itertools.repeat(goal_source),
                itertools.repeat(l2_model_path),
                chunksize=1,
            )
            for r in results_iter:
                _merge(r)

    if errors:
        print(f"[WARN] {len(errors)} world(s) failed to parse for L4 distill")
        for world_dir, err in errors[:10]:
            print(f"  - {world_dir}: {err}")

    print(f"[INFO] Extracted {len(l4_samples)} L4 distill samples")
    return l4_samples

def _process_world_dir(
    world_dir: str,
    agent_name: str,
    strict_json_only: bool,
    goal_backfill: str = "none",
    l2_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """处理单个 world 目录（用于多进程并行）."""
    result: Dict[str, Any] = {
        'world_dir': world_dir,
        'macro_samples': [],
        'micro_samples': [],
        'used_tracker': False,
        'used_csv': False,
        'skipped_csv': False,
        'has_tracker': False,
        'has_csv': False,
        'error': None,
    }
    try:
        # 模式 1：直接传入单个 Tracker JSON 文件（常见于 hrl_data_runner 的 tracker_logs 拆分）
        if (
            os.path.isfile(world_dir)
            and os.path.basename(world_dir).lower().startswith("agent_")
            and world_dir.lower().endswith(".json")
        ):
            result["has_tracker"] = True
            result["has_csv"] = False

            try:
                with open(world_dir, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                return result

            agent_aliases = _extract_agent_type_aliases(data)
            if not _agent_type_matches_any(agent_name, agent_aliases):
                return result

            daily_states = _extract_daily_states_from_tracker_data(data)
            entries = data.get("entries", []) or []
            neg_logs = _parse_tracker_entries(entries, agent_name) if entries else []
            offers_snapshot_by_day = _build_offers_snapshot_by_day(neg_logs) if neg_logs else None

            if daily_states:
                daily_logs = _daily_states_to_logs(
                    daily_states,
                    agent_name,
                    offers_snapshot_by_day=offers_snapshot_by_day,
                )
                macro_samples = reconstruct_l2_goals(daily_logs)
            else:
                macro_samples = []

            _backfill_negotiation_goals(
                neg_logs,
                macro_samples,
                goal_backfill=goal_backfill,
                l2_model_path=l2_model_path,
            )

            result["macro_samples"].extend(macro_samples)
            result["micro_samples"].extend(extract_l3_residuals(neg_logs, daily_states=daily_states))
            result["used_tracker"] = True
            return result

        # 判断数据格式优先级：Tracker JSON > CSV
        # 注意：弃用 world_stats.json/negotiations.json，因为没有生成器
        # Tracker JSON 提供完整精确的状态和谈判数据
        # CSV 是降级方案，会导致精度问题（价格/压力通道为 0，缺少 aop_action 标签）
        has_tracker = len(glob.glob(os.path.join(world_dir, "agent_*.json"))) > 0
        has_csv = os.path.exists(os.path.join(world_dir, "stats.csv"))
        result['has_tracker'] = has_tracker
        result['has_csv'] = has_csv
        
        if has_tracker:
            # Tracker JSON 优先（完整精确的状态数据）
            # hrl_data_runner 的 tracker_logs 目录可能包含多个 world/run 的 agent_*.json，必须按文件拆分处理。
            if os.path.basename(world_dir).lower() == "tracker_logs":
                for _, data in _iter_tracker_json_files(world_dir, agent_name):
                    daily_states = _extract_daily_states_from_tracker_data(data)
                    entries = data.get("entries", []) or []
                    neg_logs = _parse_tracker_entries(entries, agent_name) if entries else []
                    offers_snapshot_by_day = _build_offers_snapshot_by_day(neg_logs) if neg_logs else None

                    if daily_states:
                        daily_logs = _daily_states_to_logs(
                            daily_states,
                            agent_name,
                            offers_snapshot_by_day=offers_snapshot_by_day,
                        )
                        macro_samples = reconstruct_l2_goals(daily_logs)
                    else:
                        macro_samples = []

                    _backfill_negotiation_goals(
                        neg_logs,
                        macro_samples,
                        goal_backfill=goal_backfill,
                        l2_model_path=l2_model_path,
                    )

                    result["macro_samples"].extend(macro_samples)
                    result["micro_samples"].extend(extract_l3_residuals(neg_logs, daily_states=daily_states))

                result["used_tracker"] = True

            else:
                daily_states = _load_daily_states_from_tracker(world_dir, agent_name)
                neg_logs = _load_negotiation_logs(world_dir, agent_name)
                offers_snapshot_by_day = _build_offers_snapshot_by_day(neg_logs) if neg_logs else None

                if daily_states:
                    daily_logs = _daily_states_to_logs(
                        daily_states,
                        agent_name,
                        offers_snapshot_by_day=offers_snapshot_by_day,
                    )
                    result["macro_samples"].extend(reconstruct_l2_goals(daily_logs))

                _backfill_negotiation_goals(
                    neg_logs,
                    result["macro_samples"],
                    goal_backfill=goal_backfill,
                    l2_model_path=l2_model_path,
                )

                result["micro_samples"].extend(extract_l3_residuals(neg_logs, daily_states=daily_states))
                result["used_tracker"] = True
        elif has_csv:
            if strict_json_only:
                result['skipped_csv'] = True
                return result
            
            stats = load_stats_csv(world_dir, agent_name)
            contracts = load_contracts(world_dir)
            daily_logs = _organize_by_day(stats, contracts, agent_name)
            result['macro_samples'].extend(reconstruct_l2_goals(daily_logs))
            
            daily_states = {}
            for dlog in daily_logs:
                step = dlog['current_step']
                daily_states[step] = {
                    'balance': dlog.get('balance', 10000),
                    'inventory': dlog.get('inventory', {}),
                    'n_steps': dlog.get('n_steps', 100),
                    'current_step': step,
                    'n_lines': dlog.get('n_lines', 1),
                    'spot_price_in': dlog.get('spot_price_in', 10.0),
                    'spot_price_out': dlog.get('spot_price_out', 20.0),
                    'commitments': dlog.get('commitments', {}),
                }
            
            neg_logs = _load_negotiation_logs_csv(world_dir, agent_name)

            _backfill_negotiation_goals(
                neg_logs,
                result['macro_samples'],
                goal_backfill=goal_backfill,
                l2_model_path=l2_model_path,
            )

            result['micro_samples'].extend(extract_l3_residuals(neg_logs, daily_states=daily_states))
            result['used_csv'] = True
    except Exception as exc:
        result['error'] = f"{type(exc).__name__}: {exc}"
    return result


def load_tournament_data(
    tournament_dir: str,
    agent_name: str = "LitaAgent",
    strict_json_only: bool = True,
    num_workers: Optional[int] = None,
    goal_backfill: str = "none",
    l2_model_path: Optional[str] = None,
) -> Tuple[List[MacroSample], List[MicroSample]]:
    """从锦标赛目录加载训练数据.
    
    数据格式：
    - **Tracker JSON 格式**（推荐）：包含完整状态与 aop_action 标签，
      buy_pressure/sell_pressure、price_diff_in/out 等字段，适合完整训练。
    - **CSV 格式**（已弃用）：缺少 aop_action/快照通道，无法保证标签正确，
      仅在 strict_json_only=False 时作为降级方案。
    
    Args:
        tournament_dir: 锦标赛日志目录
        agent_name: 要提取的代理名称（用于筛选相关记录）
        strict_json_only: 是否仅使用 Tracker JSON 格式（默认 True）
            - True: 跳过所有 CSV 目录，确保训练数据质量
            - False: 允许 CSV 降级（仅用于调试）
        num_workers: 并行进程数（默认自动，<=1 则串行）
        goal_backfill: micro.l2_goal 的回填来源（默认 "none"）
            - "none": 不回填，依赖日志自带 l2_goal（通常为空/旧版本为 0）
            - "v2": 用 reconstruct_l2_goals() 的 v2 标签按 day 回填到谈判日志
            - "l2": 用已训练 L2 模型预测 goal_hat 后按 day 回填到谈判日志（推荐）
        l2_model_path: 当 goal_backfill="l2" 时使用的 L2 权重路径（torch state_dict 或 checkpoint）
        
    Returns:
        (macro_samples, micro_samples)
    """
    macro_samples: List[MacroSample] = []
    micro_samples: List[MicroSample] = []
    
    # 查找所有世界目录
    world_dirs = _find_world_dirs(tournament_dir)
    
    if not world_dirs:
        print(f"[WARN] No world directories found in {tournament_dir}")
        return macro_samples, micro_samples
    
    print(f"[INFO] Found {len(world_dirs)} world directories")
    
    csv_count = 0
    csv_skipped_count = 0  # 严格模式下跳过的 CSV 目录数
    tracker_count = 0
    # P2 修复: 删除未使用的 json_count 变量（旧的 JSON 格式已弃用）
    csv_worlds_used: List[str] = []
    errors: List[Tuple[str, str]] = []
    
    if num_workers is None:
        cpu_cnt = os.cpu_count() or 1
        num_workers = max(1, cpu_cnt - 1)
    num_workers = min(num_workers, len(world_dirs))
    
    def _merge_result(result: Dict[str, Any]) -> None:
        nonlocal csv_count, csv_skipped_count, tracker_count
        macro_samples.extend(result.get('macro_samples', []))
        micro_samples.extend(result.get('micro_samples', []))
        if result.get('used_tracker'):
            tracker_count += 1
        if result.get('used_csv'):
            csv_count += 1
            csv_worlds_used.append(result.get('world_dir', ''))
        if result.get('skipped_csv'):
            csv_skipped_count += 1
        if result.get('error'):
            errors.append((result.get('world_dir', ''), result['error']))
    
    if num_workers <= 1:
        for world_dir in world_dirs:
            result = _process_world_dir(
                world_dir,
                agent_name,
                strict_json_only,
                goal_backfill=goal_backfill,
                l2_model_path=l2_model_path,
            )
            _merge_result(result)
    else:
        print(f"[INFO] Using {num_workers} workers for data pipeline")
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx
        ) as executor:
            results_iter = executor.map(
                _process_world_dir,
                world_dirs,
                itertools.repeat(agent_name),
                itertools.repeat(strict_json_only),
                itertools.repeat(goal_backfill),
                itertools.repeat(l2_model_path),
                chunksize=1,
            )
            for result in results_iter:
                _merge_result(result)
    
    # 输出格式统计
    # P2 修复: 删除旧的 JSON 格式统计（已弃用）
    if tracker_count > 0:
        print(f"[INFO] Data sources: {tracker_count} Tracker JSON (complete)")
    if csv_count > 0:
        for world_dir in csv_worlds_used:
            print(f"[WARN] Using deprecated CSV format for {world_dir}. "
                  "CSV data has quality issues (channels 6-9 = 0, missing aop_action). "
                  "For accurate training, use Tracker JSON format with hrl_data_runner.")
        if tracker_count == 0:
            print(f"[WARN] Using CSV format only ({csv_count} worlds). "
                  "Training data quality issues:\n"
                  "  - Channels 6-9 (price_diff_in/out, buy_pressure, sell_pressure) = 0\n"
                  "  - Missing aop_action labels (ACCEPT/REJECT/END)\n"
                  "  For accurate training, use Tracker JSON format.")
        else:
            print(f"[INFO] Data sources: {csv_count} CSV (lossy, fallback only)")
    
    # 严格模式下跳过 CSV 的统计
    if csv_skipped_count > 0:
        print(f"[INFO] Skipped {csv_skipped_count} CSV-only directories (strict_json_only=True)")
        if tracker_count == 0 and len(macro_samples) == 0:
            print(f"[ERROR] No Tracker JSON data found! All {csv_skipped_count} directories contain only CSV format.\n"
                  f"  Options:\n"
                  f"    1. Use hrl_data_runner to generate Tracker JSON data\n"
                  f"    2. Set strict_json_only=False to use CSV (not recommended for training)")
    
    # 检查 Tracker JSON 存在但样本为 0 的情况（可能是 agent_name 不匹配）
    if tracker_count > 0 and len(macro_samples) == 0 and len(micro_samples) == 0:
        print(f"[WARN] Found {tracker_count} Tracker JSON directories but extracted 0 samples!\n"
              f"  Possible causes:\n"
              f"    1. agent_name='{agent_name}' does not match any agent_type in the logs\n"
              f"    2. Tracker files are empty or corrupted\n"
              f"  Try: Check tracker JSON agent_type field and adjust agent_name")
    
    if errors:
        print(f"[WARN] {len(errors)} world(s) failed to parse")
        for world_dir, err in errors:
            print(f"  - {world_dir}: {err}")
    
    print(f"[INFO] Extracted {len(macro_samples)} macro samples, {len(micro_samples)} micro samples")
    return macro_samples, micro_samples


def _normalize_agent_name(agent_name: str) -> str:
    name = agent_name.lower().strip()
    if name.endswith("tracked"):
        name = name[:-7]
    return name


def _agent_type_matches(agent_name: str, agent_type: str) -> bool:
    if not agent_name:
        return True
    if not agent_type:
        return False
    name = _normalize_agent_name(agent_name)
    atype = _normalize_agent_name(agent_type)
    if not name or not atype:
        return False
    return atype == name or atype.startswith(name) or name in atype


def _agent_type_matches_any(agent_name: str, agent_types: List[str]) -> bool:
    if not agent_name:
        return True
    for agent_type in agent_types:
        if _agent_type_matches(agent_name, agent_type):
            return True
    return False


def _extract_agent_type_aliases(tracker_data: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    
    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            aliases.append(value)
    
    for key in (
        "agent_type",
        "agent_type_full",
        "agent_type_raw",
        "agent_type_raw_full",
        "agent_type_raw_qualname",
        "agent_type_base",
        "agent_type_base_full",
        "agent_type_base_qualname",
    ):
        _add(tracker_data.get(key))
    
    entries = tracker_data.get("entries", [])
    for entry in entries:
        _add(entry.get("agent_type"))
        if entry.get("event") == "agent_initialized":
            data = entry.get("data", {})
            for key in (
                "agent_type_full",
                "agent_type_raw",
                "agent_type_raw_full",
                "agent_type_raw_qualname",
                "agent_type_base",
                "agent_type_base_full",
                "agent_type_base_qualname",
                "agent_display_name",
                "agent_registry_name",
                "agent_short_name",
                "agent_name",
            ):
                _add(data.get(key))
            break
    
    # 去重并保持顺序
    seen = set()
    unique_aliases: List[str] = []
    for alias in aliases:
        if alias not in seen:
            unique_aliases.append(alias)
            seen.add(alias)
    
    return unique_aliases


def _load_daily_states_from_tracker(
    world_dir: str,
    agent_name: str
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """从 Tracker JSON 加载每日状态.
    
    Tracker 的 daily_status 事件包含 HRL-XF 训练所需的完整状态字段。
    同时提取 contract_signed 事件构建每天的 deals。
    
    P1 修复: 使用 agent_id 进行隔离，防止跨 agent 数据混合。
    
    Args:
        world_dir: 世界目录
        agent_name: 代理名称（用于过滤）
        
    Returns:
        agent_daily_states: {agent_id: {step: state_dict}} 嵌套字典
    """
    # P1 修复: 按 agent_id 隔离，防止跨 agent 数据混合
    agent_daily_states: Dict[str, Dict[int, Dict[str, Any]]] = {}
    agent_contract_events: Dict[str, List] = {}  # 收集每个 agent 的 contract_signed 事件
    
    # 查找匹配的 tracker 文件
    tracker_files = glob.glob(os.path.join(world_dir, "agent_*.json"))
    
    for tracker_file in tracker_files:
        try:
            with open(tracker_file, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        agent_aliases = _extract_agent_type_aliases(data)
        if not _agent_type_matches_any(agent_name, agent_aliases):
            continue

        # P1 修复: 从日志提取 agent_id 用于隔离
        # 格式: agent_<AgentType>@<world_id>.json 或 agent_<AgentType>_<index>.json
        filename = os.path.basename(tracker_file)
        agent_id = data.get("agent_id") or filename.replace('agent_', '').replace('.json', '')
        
        # 初始化该 agent 的存储
        if agent_id not in agent_daily_states:
            agent_daily_states[agent_id] = {}
            agent_contract_events[agent_id] = []
        
        daily_states = agent_daily_states[agent_id]
        contract_events = agent_contract_events[agent_id]
        
        entries = data.get('entries', [])
        
        for entry in entries:
            category = entry.get('category', '')
            event_type = entry.get('event', '')
            
            if event_type == 'daily_status':
                day = entry.get('day', 0)
                entry_data = entry.get('data', {})
                
                # 提取 HRL-XF 训练所需的完整状态
                daily_states[day] = {
                    # 基本状态
                    'balance': float(entry_data.get('balance', 10000)),
                    'initial_balance': float(entry_data.get('initial_balance', 10000)),
                    'inventory': {
                        'input': int(entry_data.get('inventory_input', 0)),
                        'output': int(entry_data.get('inventory_output', 0)),
                    },
                    # 时间参数
                    'n_steps': int(entry_data.get('n_steps', 100)),
                    'current_step': day,
                    'n_lines': int(entry_data.get('n_lines', 1)),
                    # 价格信息
                    'spot_price_in': float(entry_data.get('spot_price_in', 10.0)),
                    'spot_price_out': float(entry_data.get('spot_price_out', 20.0)),
                    'trading_prices': entry_data.get('trading_prices', []),
                    # 产品索引
                    'my_input_product': int(entry_data.get('my_input_product', 0)),
                    'my_output_product': int(entry_data.get('my_output_product', 1)),
                    # P0 修复: 添加 n_products 用于正确计算 x_role
                    'n_products': int(entry_data.get('n_products', entry_data.get('my_output_product', 1) + 1)),
                    # 合同承诺
                    'commitments': entry_data.get('commitments', {}),
                    # 成本参数
                    'production_cost': float(entry_data.get('production_cost', 1.0)),
                    'disposal_cost': float(entry_data.get('disposal_cost', 0)),
                    'shortfall_penalty': float(entry_data.get('shortfall_penalty', 0)),
                    'storage_cost': float(entry_data.get('storage_cost', 0)),
                    # 需求信息
                    'needed_supplies': int(entry_data.get('needed_supplies', 0)),
                    'needed_sales': int(entry_data.get('needed_sales', 0)),
                    'total_supplies': int(entry_data.get('total_supplies', 0)),
                    'total_sales': int(entry_data.get('total_sales', 0)),
                    # HRL-XF 6-9 通道: 活跃谈判快照
                    'offers_snapshot': entry_data.get('offers_snapshot', {'buy': [], 'sell': []}),
                    # deals 初始化为空列表，后续会填充
                    'deals': [],
                }
            
            elif event_type == 'contract_signed' or (category == 'contract' and event_type == 'signed'):
                # 收集合约签署事件（兼容 tracker: contract/signed）
                entry_data = entry.get('data', {})
                role = entry_data.get('role', None)
                if role in ('seller', 'buyer'):
                    is_seller = (role == 'seller')
                else:
                    is_seller = entry_data.get('is_seller', False)
                contract_events.append({
                    'day': entry.get('day', 0),
                    'quantity': int(entry_data.get('quantity', 0)),
                    'price': float(entry_data.get('price', 0)),
                    'delivery_time': int(entry_data.get('delivery_day', entry_data.get('delivery_time', entry_data.get('time', 0)))),
                    'is_seller': is_seller,
                })
    
    # P1 修复: 将合约事件分配到对应 agent 的对应天
    for agent_id, contract_events in agent_contract_events.items():
        if agent_id not in agent_daily_states:
            continue
        daily_states = agent_daily_states[agent_id]
        for contract in contract_events:
            day = contract['day']
            if day in daily_states:
                daily_states[day]['deals'].append({
                    'quantity': contract['quantity'],
                    'price': contract['price'],
                    'delivery_time': contract['delivery_time'],
                    'is_buying': not contract['is_seller'],  # is_seller 的反向
                })
    
    # P1 修复: 返回扁平化的 daily_states（合并所有 agent）
    # 每个 agent 的数据独立处理后合并，不会跨 agent 覆盖
    merged_daily_states: Dict[int, Dict[str, Any]] = {}
    for agent_id, daily_states in agent_daily_states.items():
        for day, state in daily_states.items():
            # 如果同一天有多个 agent 的数据，创建新条目（使用 agent_id 区分）
            # 但由于我们通过 agent_name 过滤，通常每个 world 只有一个匹配的 agent
            if day not in merged_daily_states:
                merged_daily_states[day] = state
            else:
                # 如果已存在，说明可能有多个同名 agent，追加 deals
                merged_daily_states[day]['deals'].extend(state.get('deals', []))
    
    return merged_daily_states


def _iter_tracker_json_files(
    world_dir: str,
    agent_name: str,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """遍历目录下匹配的 Tracker JSON 文件并返回 (path, json_data)。"""
    tracker_files = glob.glob(os.path.join(world_dir, "agent_*.json"))
    for tracker_file in tracker_files:
        try:
            with open(tracker_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        agent_aliases = _extract_agent_type_aliases(data)
        if not _agent_type_matches_any(agent_name, agent_aliases):
            continue

        yield tracker_file, data


def _extract_daily_states_from_tracker_data(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """从单个 Tracker JSON（单次仿真运行）中提取 daily_states（按 day）。

    注意：hrl_data_runner 的 tracker_logs 目录可能包含多个 world/run 的 `agent_*.json`。
    对这种目录必须按文件（run）拆分处理，不能把多个文件合并到同一个 daily_states，
    否则会发生跨 world 的 day 覆盖与数据混合。
    """
    daily_states: Dict[int, Dict[str, Any]] = {}
    contract_events: List[Dict[str, Any]] = []

    entries = data.get("entries", []) or []
    for entry in entries:
        category = entry.get("category", "")
        event_type = entry.get("event", "")

        if event_type == "daily_status":
            day = int(entry.get("day", 0) or 0)
            entry_data = entry.get("data", {}) or {}

            daily_states[day] = {
                "balance": float(entry_data.get("balance", 10000)),
                "initial_balance": float(entry_data.get("initial_balance", 10000)),
                "inventory": {
                    "input": int(entry_data.get("inventory_input", 0)),
                    "output": int(entry_data.get("inventory_output", 0)),
                },
                "n_steps": int(entry_data.get("n_steps", 100)),
                "current_step": day,
                "n_lines": int(entry_data.get("n_lines", 1)),
                "spot_price_in": float(entry_data.get("spot_price_in", 10.0)),
                "spot_price_out": float(entry_data.get("spot_price_out", 20.0)),
                "trading_prices": entry_data.get("trading_prices", []),
                "my_input_product": int(entry_data.get("my_input_product", 0)),
                "my_output_product": int(entry_data.get("my_output_product", 1)),
                "n_products": int(entry_data.get("n_products", entry_data.get("my_output_product", 1) + 1)),
                "commitments": entry_data.get("commitments", {}),
                "production_cost": float(entry_data.get("production_cost", 1.0)),
                "disposal_cost": float(entry_data.get("disposal_cost", 0)),
                "shortfall_penalty": float(entry_data.get("shortfall_penalty", 0)),
                "storage_cost": float(entry_data.get("storage_cost", 0)),
                "needed_supplies": int(entry_data.get("needed_supplies", 0)),
                "needed_sales": int(entry_data.get("needed_sales", 0)),
                "total_supplies": int(entry_data.get("total_supplies", 0)),
                "total_sales": int(entry_data.get("total_sales", 0)),
                "offers_snapshot": entry_data.get("offers_snapshot", {"buy": [], "sell": []}),
                "deals": [],
            }

        elif event_type == "contract_signed" or (category == "contract" and event_type == "signed"):
            entry_data = entry.get("data", {}) or {}
            role = entry_data.get("role", None)
            if role in ("seller", "buyer"):
                is_seller = (role == "seller")
            else:
                is_seller = bool(entry_data.get("is_seller", False))
            contract_events.append(
                {
                    "day": int(entry.get("day", 0) or 0),
                    "quantity": int(entry_data.get("quantity", 0) or 0),
                    "price": float(entry_data.get("price", 0) or 0),
                    "delivery_time": int(
                        entry_data.get(
                            "delivery_day",
                            entry_data.get("delivery_time", entry_data.get("time", 0)),
                        )
                        or 0
                    ),
                    "is_seller": is_seller,
                }
            )

    for contract in contract_events:
        day = int(contract["day"])
        if day not in daily_states:
            continue
        daily_states[day]["deals"].append(
            {
                "quantity": contract["quantity"],
                "price": contract["price"],
                "delivery_time": contract["delivery_time"],
                "is_buying": not contract["is_seller"],
            }
        )

    return daily_states


def _daily_states_to_logs(
    daily_states: Dict[int, Dict[str, Any]],
    agent_name: str,
    offers_snapshot_by_day: Optional[Dict[int, Dict[str, List[List[float]]]]] = None,
) -> List[Dict[str, Any]]:
    """将 daily_states 转换为 daily_logs 格式，用于 reconstruct_l2_goals.
    
    Args:
        daily_states: 从 Tracker 加载的每日状态
        agent_name: 代理名称
        offers_snapshot_by_day: 基于谈判日志重建的报价快照（按天）
        
    Returns:
        daily_logs: 每天的日志列表
    """
    if not daily_states:
        return []
    
    # 按天排序
    days = sorted(daily_states.keys())
    daily_logs = []
    
    for day in days:
        state = daily_states[day]
        
        # 从 commitments 重建 pending_contracts 统计
        commitments = state.get('commitments', {})
        Q_in = commitments.get('Q_in', [])
        Q_out = commitments.get('Q_out', [])
        Payables = commitments.get('Payables', [])
        Receivables = commitments.get('Receivables', [])
        
        pending_buy_qty = sum(Q_in) if Q_in else 0
        pending_sell_qty = sum(Q_out) if Q_out else 0
        pending_buy_value = sum(Payables) if Payables else 0
        pending_sell_value = sum(Receivables) if Receivables else 0
        
        offers_snapshot = state.get('offers_snapshot', {'buy': [], 'sell': []})
        if offers_snapshot_by_day and day in offers_snapshot_by_day:
            offers_snapshot = offers_snapshot_by_day[day]

        day_log = {
            'current_step': day,
            'n_steps': state.get('n_steps', 100),
            'balance': state.get('balance', 10000),
            'initial_balance': state.get('initial_balance', 10000),
            'inventory': state.get('inventory', {}),
            'n_lines': state.get('n_lines', 1),
            'production_cost': state.get('production_cost', 1.0),
            'spot_price_in': state.get('spot_price_in', 10.0),
            'spot_price_out': state.get('spot_price_out', 20.0),
            # 需求/缺口信号（用于 L2 v2 标签缺口补偿）
            'needed_supplies': state.get('needed_supplies', 0),
            'needed_sales': state.get('needed_sales', 0),
            'total_supplies': state.get('total_supplies', 0),
            'total_sales': state.get('total_sales', 0),
            'commitments': commitments,
            'pending_contracts': {
                'buy_qty': pending_buy_qty,
                'sell_qty': pending_sell_qty,
                'buy_value': pending_buy_value,
                'sell_value': pending_sell_value,
            },
            # HRL-XF 6-9 通道: 活跃谈判快照
            'offers_snapshot': offers_snapshot,
            # 产品索引（用于 x_role 计算）
            'my_input_product': state.get('my_input_product', 0),
            'my_output_product': state.get('my_output_product', 1),
            'n_products': state.get('n_products', state.get('my_output_product', 1) + 1),
            # 当天签署的合约（从 Tracker contract_signed 事件重建）
            'deals': state.get('deals', []),
        }
        
        daily_logs.append(day_log)
    
    return daily_logs


def _organize_by_day(
    stats: Dict[str, Any],
    contracts: List[Dict[str, Any]],
    agent_name: str,
    horizon: int = 40
) -> List[Dict[str, Any]]:
    """将数据按天组织，提取完整状态供 extract_macro_state 使用.
    
    关键字段（与 extract_macro_state 对齐）：
    - balance, inventory: 当天快照
    - n_lines, production_cost: 代理配置
    - spot_price_in, spot_price_out: 市场价格
    - commitments: {'Q_in', 'Q_out', 'Payables', 'Receivables'}
    - pending_contracts: {'buy_qty', 'sell_qty', 'buy_value', 'sell_value'}
    - deals: 当天签署的合约
    """
    n_steps = stats.get('n_steps', 100)
    
    # 从 stats 中提取全局配置
    n_lines = int(stats.get('n_lines', 1))
    production_cost = float(stats.get('production_cost', 1.0))
    initial_balance = float(stats.get('initial_balance', 10000.0))
    
    # 修复：从 stats 获取产品索引（与在线 awi.my_input_product 一致）
    # 如果没有记录，使用默认值（第一层级代理：input=0, output=1）
    my_input_product = int(stats.get('my_input_product', 0))
    my_output_product = int(stats.get('my_output_product', 1))
    
    # 市场价格（可能是每天变化的，或者是固定的）
    catalog_prices = stats.get('catalog_prices', {})
    trading_prices = stats.get('trading_prices', {})
    
    def get_spot_price(product_id: int, default: float, step: int = 0) -> float:
        """获取某个产品的现货价格."""
        # 优先使用 trading_prices（每天更新）
        if trading_prices:
            if isinstance(trading_prices, dict):
                step_prices = trading_prices.get(str(step), trading_prices)
                if isinstance(step_prices, dict):
                    return float(step_prices.get(str(product_id), default))
                elif hasattr(step_prices, '__getitem__'):
                    try:
                        return float(step_prices[product_id])
                    except (IndexError, TypeError):
                        pass
            elif hasattr(trading_prices, '__getitem__'):
                try:
                    return float(trading_prices[product_id])
                except (IndexError, TypeError):
                    pass
        # 回退到 catalog_prices
        if catalog_prices:
            if isinstance(catalog_prices, dict):
                return float(catalog_prices.get(str(product_id), default))
            elif hasattr(catalog_prices, '__getitem__'):
                try:
                    return float(catalog_prices[product_id])
                except (IndexError, TypeError):
                    pass
        return default
    
    daily_logs = []
    temporal_len = horizon + 1
    
    for step in range(n_steps):
        # 基本状态
        balance = stats.get('balance_history', {}).get(str(step), initial_balance)
        inventory = stats.get('inventory_history', {}).get(str(step), {})
        
        # 市场价格 - 使用正确的产品索引
        spot_price_in = get_spot_price(my_input_product, 10.0, step)
        spot_price_out = get_spot_price(my_output_product, 20.0, step)
        
        # 从合约中计算承诺量（commitments）
        Q_in = np.zeros(temporal_len, dtype=np.float32)
        Q_out = np.zeros(temporal_len, dtype=np.float32)
        Payables = np.zeros(temporal_len, dtype=np.float32)
        Receivables = np.zeros(temporal_len, dtype=np.float32)
        
        # pending_contracts 统计
        pending_buy_qty = 0.0
        pending_sell_qty = 0.0
        pending_buy_value = 0.0
        pending_sell_value = 0.0
        
        for c in contracts:
            signed_at = c.get('signed_at', -1)
            delivery_time = c.get('delivery_time', 0)
            quantity = float(c.get('quantity', 0))
            unit_price = float(c.get('unit_price', 0))
            
            # 修复：先判断合约是否与目标代理相关
            buyer = c.get('buyer', '')
            seller = c.get('seller', '')
            
            # 使用包含匹配（与其他地方一致）
            is_my_buy = agent_name.lower() in buyer.lower() if buyer else False
            is_my_sell = agent_name.lower() in seller.lower() if seller else False
            
            # 如果既不是买方也不是卖方，跳过此合约
            if not is_my_buy and not is_my_sell:
                continue
            
            # 计算相对于当天的交货偏移
            delta = delivery_time - step
            
            if signed_at <= step and delivery_time > step:
                # 这个合约在当天之前签署，还未交货
                if 0 <= delta < temporal_len:
                    if is_my_buy:
                        Q_in[delta] += quantity
                        Payables[delta] += quantity * unit_price
                        pending_buy_qty += quantity
                        pending_buy_value += quantity * unit_price
                    elif is_my_sell:
                        Q_out[delta] += quantity
                        Receivables[delta] += quantity * unit_price
                        pending_sell_qty += quantity
                        pending_sell_value += quantity * unit_price
        
        # 当天签署的合约（deals）
        deals = []
        for c in contracts:
            if c.get('signed_at', -1) == step:
                buyer = c.get('buyer', '')
                seller = c.get('seller', '')
                
                # 使用包含匹配判断角色
                is_my_buy = agent_name.lower() in buyer.lower() if buyer else False
                is_my_sell = agent_name.lower() in seller.lower() if seller else False
                
                # 如果既不是买方也不是卖方，跳过
                if not is_my_buy and not is_my_sell:
                    continue
                
                deals.append({
                    'quantity': c.get('quantity', 0),
                    'price': c.get('unit_price', 0),
                    'delivery_time': c.get('delivery_time', step + 1),
                    'is_buying': is_my_buy,
                })
        
        day_log = {
            'current_step': step,
            'n_steps': n_steps,
            'balance': float(balance),
            'initial_balance': initial_balance,
            'inventory': inventory,
            'n_lines': n_lines,
            'production_cost': production_cost,
            'spot_price_in': spot_price_in,
            'spot_price_out': spot_price_out,
            'commitments': {
                'Q_in': Q_in.tolist(),
                'Q_out': Q_out.tolist(),
                'Payables': Payables.tolist(),
                'Receivables': Receivables.tolist(),
            },
            'pending_contracts': {
                'buy_qty': pending_buy_qty,
                'sell_qty': pending_sell_qty,
                'buy_value': pending_buy_value,
                'sell_value': pending_sell_value,
            },
            'deals': deals,
        }
        
        daily_logs.append(day_log)
    
    return daily_logs


def _parse_tracker_entries(entries: List[Dict[str, Any]], agent_name: str) -> List[Dict[str, Any]]:
    """从 tracker 事件日志中解析谈判记录.
    
    Tracker 输出的是分散的事件日志，需要按谈判聚合：
    - negotiation:started - 谈判开始
    - negotiation:offer_received - 收到报价
    - negotiation:offer_made - 发出报价
    - negotiation:accept - 接受报价
    - negotiation:reject - 拒绝报价
    - negotiation:success - 谈判成功
    - negotiation:failure - 谈判失败
    
    Args:
        entries: tracker 的 entries 列表
        agent_name: 代理名称
        
    Returns:
        按谈判聚合的日志列表，格式与 extract_l3_samples_aop 期望一致
    """
    from collections import defaultdict

    negotiations: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'partner': '',
        'day': 0,
        'current_step': 0,
        'is_buyer': True,
        'offer_history': [],
        'final_action': {},
        'final_status': 'unknown',
        'action_op': None,
        'action_offer': None,
        'relative_time': None,
        'l2_goal': None,
    })

    neg_counter: Dict[str, int] = defaultdict(int)

    def _get_partner(payload: Dict[str, Any]) -> str:
        return (
            payload.get('partner')
            or payload.get('negotiator_id')
            or payload.get('counterparty', '')
        )

    def _make_key(partner: str, day: int, mechanism_id: str, event: str) -> str:
        if mechanism_id:
            return f"{partner}_{day}_{mechanism_id}"
        base_key = f"{partner}_{day}"
        if event == 'started':
            neg_counter[base_key] += 1
        counter = neg_counter[base_key] if neg_counter[base_key] > 0 else 1
        return f"{partner}_{day}_{counter}"

    for entry in entries:
        event = entry.get('event', '')
        data = entry.get('data', {}) or {}
        day = int(entry.get('day', 0) or 0)

        if event not in (
            'started',
            'offer_received',
            'offer_made',
            'accept',
            'success',
            'failure',
            'aop_action',
        ):
            continue

        partner = _get_partner(data)
        if not partner:
            continue

        mechanism_id = data.get('mechanism_id', '')
        neg_key = _make_key(partner, day, mechanism_id, event)
        neg = negotiations[neg_key]
        neg['partner'] = partner
        neg['current_step'] = day
        neg['day'] = day

        if event == 'started':
            role = data.get('role', 'buyer')
            if role in ('buyer', 'seller'):
                neg['is_buyer'] = (role == 'buyer')
            if 'l2_goal' in data:
                neg['l2_goal'] = data.get('l2_goal')

        elif event == 'offer_received':
            offer = data.get('offer', {})
            if offer:
                abs_delivery = offer.get('delivery_day', offer.get('time', offer.get('t', offer.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                neg['offer_history'].append({
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    'delta_t': relative_delta_t,
                    'round': offer.get('round'),
                    'source': 'opponent',
                })

        elif event == 'offer_made':
            offer = data.get('offer', {})
            if offer:
                abs_delivery = offer.get('delivery_day', offer.get('time', offer.get('t', offer.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                neg['offer_history'].append({
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    'delta_t': relative_delta_t,
                    'round': offer.get('round'),
                    'source': 'self',
                })

        elif event == 'accept':
            offer = data.get('offer', {})
            if offer:
                abs_delivery = offer.get('delivery_day', offer.get('time', offer.get('t', offer.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                neg['final_action'] = {
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    'delta_t': relative_delta_t,
                }
            neg['final_status'] = 'succeeded'

        elif event == 'success':
            agreement = data.get('agreement', {})
            if agreement:
                abs_delivery = agreement.get('delivery_day', agreement.get('time', agreement.get('t', agreement.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                neg['final_action'] = {
                    'quantity': agreement.get('quantity', agreement.get('q', 0)),
                    'price': agreement.get('unit_price', agreement.get('price', agreement.get('p', 0))),
                    'delta_t': relative_delta_t,
                }
            neg['final_status'] = 'succeeded'

        elif event == 'failure':
            neg['final_status'] = 'failed'

        elif event == 'aop_action':
            role = data.get('role', None)
            if role in ('buyer', 'seller'):
                neg['is_buyer'] = (role == 'buyer')
            if data.get('relative_time') is not None:
                neg['relative_time'] = data.get('relative_time')

            neg['action_op'] = data.get('action_op')
            offer = data.get('offer', None)
            if offer:
                abs_delivery = offer.get('delivery_day', offer.get('time', offer.get('t', offer.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                neg['action_offer'] = {
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    'delta_t': relative_delta_t,
                }

    result = []
    for neg_key, neg in negotiations.items():
        if len(neg['offer_history']) == 0:
            continue
        result.append({
            'id': neg_key,
            'partner': neg['partner'],
            'is_buyer': neg['is_buyer'],
            'sim_step': neg['day'],
            'offer_history': neg['offer_history'],
            'final_action': neg['final_action'],
            'final_status': neg['final_status'],
            'action_op': neg.get('action_op'),
            'action_offer': neg.get('action_offer'),
            'relative_time': neg.get('relative_time'),
            'l2_goal': neg.get('l2_goal'),
        })

    return result


def _build_offers_snapshot_by_day(
    negotiation_logs: List[Dict[str, Any]]
) -> Dict[int, Dict[str, List[List[float]]]]:
    """从谈判日志构建按天的报价快照（用于 6-9 通道回填）。"""
    round_decay = 0.3
    qty_scale = 10.0
    offers_by_day: Dict[int, Dict[str, List[List[float]]]] = {}
    for neg in negotiation_logs:
        day = int(neg.get('sim_step', neg.get('day', 0)))
        history = neg.get('offer_history', [])
        if not history:
            continue
        rounds = [h.get('round') for h in history if h.get('round') is not None]
        max_round = max(rounds) if rounds else (len(history) - 1)
        side = 'buy' if neg.get('is_buyer', True) else 'sell'
        day_entry = offers_by_day.setdefault(day, {'buy': [], 'sell': []})
        for idx, offer in enumerate(history):
            try:
                quantity = float(offer.get('quantity', 0))
                price = float(offer.get('price', 0))
                delta_t = int(offer.get('delta_t', 0))
            except (TypeError, ValueError):
                continue
            if quantity <= 0 or price <= 0:
                continue
            round_val = offer.get('round')
            if round_val is None:
                round_val = idx
            age = max(0, max_round - int(round_val))
            w_round = math.exp(-round_decay * age)
            w_qty = math.sqrt(quantity / (quantity + qty_scale))
            weight = w_round * w_qty
            if weight <= 0:
                continue
            delivery_time = day + delta_t
            day_entry[side].append([delivery_time, quantity, price, float(weight)])
    return offers_by_day


def _load_negotiation_logs(
    world_dir: str,
    agent_name: str
) -> List[Dict[str, Any]]:
    """加载谈判日志（JSON 格式）.
    
    支持两种 JSON 格式：
    1. negotiations.json - 直接的谈判列表
    2. agent_*.json - tracker 输出的事件日志（需要聚合）
    """
    logs = []
    
    # 格式 1: negotiations.json（直接的谈判列表）
    neg_file = os.path.join(world_dir, "negotiations.json")
    if os.path.exists(neg_file):
        with open(neg_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                for neg in data:
                    if agent_name.lower() in str(neg.get('partners', [])).lower():
                        logs.append(neg)
        if logs:
            return logs
    
    # 格式 2: agent_*.json（tracker 事件日志）
    import glob as glob_module
    agent_files = glob_module.glob(os.path.join(world_dir, "agent_*.json"))
    
    for agent_file in agent_files:
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            agent_aliases = _extract_agent_type_aliases(data)
            if not _agent_type_matches_any(agent_name, agent_aliases):
                continue

            entries = data.get('entries', [])
            if entries:
                parsed = _parse_tracker_entries(entries, agent_name)
                logs.extend(parsed)
        except Exception as e:
            print(f"[WARN] Failed to parse {agent_file}: {e}")
    
    return logs


def _load_negotiation_logs_csv(
    world_dir: str,
    agent_name: str
) -> List[Dict[str, Any]]:
    """从 negotiations.csv 加载谈判日志.
    
    CSV 格式字段说明：
    - id: 谈判 ID
    - partners: 参与方 ID 数组（字符串格式）
    - is_buy: 是否是买方发起
    - buyer / seller: 买卖方 ID
    - sim_step: 仿真步（天）
    - final_status: 最终状态 (succeeded/failed)
    - agreement: 最终协议 (quantity, time, unit_price) 元组
    - history: 完整的轮级历史记录（JSON 数组字符串）
    - offers: 每个参与者的出价序列（JSON 对象字符串）
    """
    if not PANDAS_AVAILABLE:
        return []
    
    csv_path = os.path.join(world_dir, "negotiations.csv")
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Failed to load negotiations.csv: {e}")
        return []
    
    logs = []
    
    for _, row in df.iterrows():
        # 筛选包含目标代理的谈判
        partners_str = str(row.get('partners', ''))
        if agent_name.lower() not in partners_str.lower():
            continue
        
        # 解析 offers 字段获取出价历史
        offers_dict = parse_offers_dict(str(row.get('offers', '{}')))
        
        # 查找目标代理的出价序列
        agent_offers = []
        agent_id = None
        for partner_id, offer_list in offers_dict.items():
            if agent_name.lower() in partner_id.lower():
                agent_offers = offer_list
                agent_id = partner_id
                break
        
        # 解析 history 字段
        history_data = parse_negotiation_history(str(row.get('history', '[]')))
        
        # 获取 sim_step 用于转换绝对时间为 delta_t
        sim_step = int(row.get('sim_step', 0))
        
        # 构建 offer_history（每轮的出价记录）
        # 注意：SCML 的 current_offer = (quantity, time, unit_price)
        # 其中 time 是绝对交货时间，需要转换为 delta_t = time - sim_step
        offer_history = []
        for round_data in history_data:
            current_offer = round_data.get('current_offer')
            if current_offer and isinstance(current_offer, (list, tuple)) and len(current_offer) >= 3:
                absolute_time = current_offer[1]
                delta_t = absolute_time - sim_step  # 修复：转换为相对时间
                offer_history.append({
                    'quantity': current_offer[0],
                    'delta_t': max(0, delta_t),  # 确保非负
                    'price': current_offer[2],
                })
        
        # 解析最终协议
        agreement = None
        agreement_str = str(row.get('agreement', 'None'))
        if agreement_str and agreement_str != 'None':
            try:
                import ast
                agreement = ast.literal_eval(agreement_str)
            except Exception:
                pass
        
        # 确定角色
        buyer = str(row.get('buyer', ''))
        is_buyer = agent_name.lower() in buyer.lower()
        
        # 获取最终动作（如果成功则使用协议，否则使用最后出价）
        # 修复：将绝对时间转换为 delta_t
        final_action = {}
        if agreement and isinstance(agreement, (list, tuple)) and len(agreement) >= 3:
            absolute_time = agreement[1]
            delta_t = max(0, absolute_time - sim_step)
            final_action = {
                'quantity': agreement[0],
                'delta_t': delta_t,
                'price': agreement[2],
            }
        elif agent_offers:
            last_offer = agent_offers[-1]
            if isinstance(last_offer, (list, tuple)) and len(last_offer) >= 3:
                absolute_time = last_offer[1]
                delta_t = max(0, absolute_time - sim_step)
                final_action = {
                    'quantity': last_offer[0],
                    'delta_t': delta_t,
                    'price': last_offer[2],
                }
        
        # CSV 仅能区分是否成交，无法还原完整 AOP 行为
        action_op = "ACCEPT" if row.get("final_status") == "succeeded" else "END"
        action_offer = final_action if action_op != "END" else None
        log_entry = {
            'id': row.get('id', ''),
            'partners': partners_str,
            'is_buyer': is_buyer,
            'sim_step': row.get('sim_step', 0),
            'final_status': row.get('final_status', 'unknown'),
            'offer_history': offer_history,
            'final_action': final_action,
            'action_op': action_op,
            'action_offer': action_offer,
            'l2_goal': np.zeros(16).tolist(),  # 需要从宏观数据中填充
            'reward': 1.0 if row.get('final_status') == 'succeeded' else 0.0,
        }
        
        logs.append(log_entry)
    
    return logs


# ============== 数据保存 ==============

def save_samples(
    macro_samples: List[MacroSample],
    micro_samples: List[MicroSample],
    output_dir: str,
    horizon: int = 40,
) -> None:
    """保存样本到文件.
    
    Args:
        macro_samples: L2 宏观样本
        micro_samples: L3 微观样本
        output_dir: 输出目录
        horizon: 规划视界 H，用于确定 time_mask 默认长度 (H+1)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存宏观样本
    if macro_samples:
        macro_data = {
            'state_static': np.stack([s.state_static for s in macro_samples]),
            'state_temporal': np.stack([s.state_temporal for s in macro_samples]),
            'x_roles': np.stack([s.x_role for s in macro_samples]),  # Multi-Hot (2,) 谈判能力
            'goals': np.stack([s.goal for s in macro_samples]),
            'days': np.array([s.day for s in macro_samples]),
        }
        np.savez(os.path.join(output_dir, "l2_samples.npz"), **macro_data)
    
    # 保存微观样本（变长序列需要特殊处理）
    if micro_samples:
        # 检查 time_mask 是否存在，统一处理为固定形状
        # 遍历所有样本找最大 time_mask 长度（应为 H+1）
        max_tm_len = horizon + 1  # 使用传入的 horizon 参数
        for s in micro_samples:
            if s.time_mask is not None:
                max_tm_len = max(max_tm_len, len(s.time_mask))
        
        # 构建 time_masks / Q_safe 数组，缺失的填充 0.0
        time_masks = []
        q_safe_masks = []
        for s in micro_samples:
            if s.time_mask is not None:
                tm = s.time_mask
                # 如果长度不足，填充 0.0
                if len(tm) < max_tm_len:
                    tm = np.concatenate([tm, np.zeros(max_tm_len - len(tm))])
                time_masks.append(tm)
            else:
                time_masks.append(np.zeros(max_tm_len, dtype=np.float32))

            qs = s.Q_safe if s.Q_safe is not None else np.zeros(max_tm_len, dtype=np.float32)
            if len(qs) < max_tm_len:
                qs = np.concatenate([qs, np.zeros(max_tm_len - len(qs))])
            q_safe_masks.append(qs)
        
        micro_data = {
            'negotiation_ids': np.array([s.negotiation_id for s in micro_samples], dtype=object),
            'roles': np.array([s.role for s in micro_samples]),
            'current_offers': np.stack([
                np.asarray(s.current_offer if s.current_offer is not None else np.zeros(3, dtype=np.float32), dtype=np.float32)
                for s in micro_samples
            ]),
            'relative_times': np.array([float(s.relative_time) for s in micro_samples], dtype=np.float32),
            'l2_goals': np.stack([np.asarray(s.l2_goal, dtype=np.float32) for s in micro_samples]),
            'goal_gap_buy': np.stack([np.asarray(s.goal_gap_buy, dtype=np.float32) for s in micro_samples]),
            'goal_gap_sell': np.stack([np.asarray(s.goal_gap_sell, dtype=np.float32) for s in micro_samples]),
            'today_bought_qty': np.array([float(s.today_bought_qty) for s in micro_samples], dtype=np.float32),
            'today_sold_qty': np.array([float(s.today_sold_qty) for s in micro_samples], dtype=np.float32),
            'today_bought_by_bucket': np.stack([np.asarray(s.today_bought_by_bucket, dtype=np.float32) for s in micro_samples]),
            'today_sold_by_bucket': np.stack([np.asarray(s.today_sold_by_bucket, dtype=np.float32) for s in micro_samples]),
            'x_static': np.stack([np.asarray(s.x_static, dtype=np.float32) for s in micro_samples]),
            'X_temporal': np.stack([np.asarray(s.X_temporal, dtype=np.float32) for s in micro_samples]),
            'time_masks': np.stack(time_masks),
            'Q_safe': np.stack(q_safe_masks),
            'B_free': np.array([float(s.B_free) for s in micro_samples], dtype=np.float32),
            'action_ops': np.array([int(s.action_op) for s in micro_samples], dtype=np.int64),
            'target_q': np.array([
                float(s.target_quantity) if s.target_quantity is not None else np.nan
                for s in micro_samples
            ], dtype=np.float32),
            'target_p': np.array([
                float(s.target_price) if s.target_price is not None else np.nan
                for s in micro_samples
            ], dtype=np.float32),
            'target_t': np.array([
                int(s.target_time) if s.target_time is not None else -1
                for s in micro_samples
            ], dtype=np.int64),
            'rewards': np.array([
                float(s.reward) if s.reward is not None else 0.0
                for s in micro_samples
            ], dtype=np.float32),
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
        # 兼容旧格式（无 x_roles 字段）
        has_x_roles = 'x_roles' in data
        for i in range(len(data['days'])):
            if has_x_roles:
                x_role = data['x_roles'][i]
            else:
                # 回退：假设平衡角色
                x_role = np.array([0.5, 0.5], dtype=np.float32)
            macro_samples.append(MacroSample(
                day=int(data['days'][i]),
                state_static=data['state_static'][i],
                state_temporal=data['state_temporal'][i],
                x_role=x_role,
                goal=data['goals'][i]
            ))
    
    # 加载微观样本
    l3_path = os.path.join(data_dir, "l3_samples.npz")
    l3_hist_path = os.path.join(data_dir, "l3_histories.npz")
    
    if os.path.exists(l3_path):
        data = np.load(l3_path)
        histories = None
        time_masks = data.get('time_masks', None)
        q_safe_masks = data.get('Q_safe', None)
        neg_ids = data.get('negotiation_ids', None)
        current_offers = data.get('current_offers', None)
        relative_times = data.get('relative_times', None)
        l2_goals = data.get('l2_goals', data.get('goals', None))
        goal_gap_buy = data.get('goal_gap_buy', None)
        goal_gap_sell = data.get('goal_gap_sell', None)
        today_bought_qty = data.get('today_bought_qty', None)
        today_sold_qty = data.get('today_sold_qty', None)
        today_bought_by_bucket = data.get('today_bought_by_bucket', None)
        today_sold_by_bucket = data.get('today_sold_by_bucket', None)
        x_static = data.get('x_static', None)
        X_temporal = data.get('X_temporal', None)
        B_free = data.get('B_free', None)
        action_ops = data.get('action_ops', None)
        target_q = data.get('target_q', None)
        target_p = data.get('target_p', None)
        target_t = data.get('target_t', None)
        rewards = data.get('rewards', None)

        if os.path.exists(l3_hist_path):
            hist_data = np.load(l3_hist_path, allow_pickle=True)
            histories = hist_data['histories']

        roles = data.get('roles', None)
        count = int(len(roles)) if roles is not None else (len(histories) if histories is not None else 0)
        for i in range(count):
            history = histories[i] if histories is not None else np.zeros((1, 4))
            role = int(roles[i]) if roles is not None else 0

            tm_len = 41
            if time_masks is not None:
                tm_len = len(time_masks[i]) if time_masks.ndim > 1 else len(time_masks)
            elif q_safe_masks is not None:
                tm_len = len(q_safe_masks[i]) if q_safe_masks.ndim > 1 else len(q_safe_masks)
            elif X_temporal is not None and len(X_temporal) > i:
                tm_len = int(X_temporal[i].shape[0])

            time_mask = time_masks[i] if time_masks is not None else np.zeros(tm_len, dtype=np.float32)
            q_safe = q_safe_masks[i] if q_safe_masks is not None else np.zeros(tm_len, dtype=np.float32)

            tq = target_q[i] if target_q is not None else np.nan
            tp = target_p[i] if target_p is not None else np.nan
            tt = target_t[i] if target_t is not None else -1

            micro_samples.append(MicroSample(
                negotiation_id=str(neg_ids[i]) if neg_ids is not None else str(i),
                history=history,
                role=role,
                current_offer=(current_offers[i] if current_offers is not None else np.zeros(3, dtype=np.float32)),
                relative_time=float(relative_times[i]) if relative_times is not None else 0.0,
                l2_goal=(l2_goals[i] if l2_goals is not None else np.zeros(16, dtype=np.float32)),
                goal_gap_buy=(goal_gap_buy[i] if goal_gap_buy is not None else np.zeros(4, dtype=np.float32)),
                goal_gap_sell=(goal_gap_sell[i] if goal_gap_sell is not None else np.zeros(4, dtype=np.float32)),
                today_bought_qty=float(today_bought_qty[i]) if today_bought_qty is not None else 0.0,
                today_sold_qty=float(today_sold_qty[i]) if today_sold_qty is not None else 0.0,
                today_bought_by_bucket=(today_bought_by_bucket[i] if today_bought_by_bucket is not None else np.zeros(4, dtype=np.float32)),
                today_sold_by_bucket=(today_sold_by_bucket[i] if today_sold_by_bucket is not None else np.zeros(4, dtype=np.float32)),
                x_static=(x_static[i] if x_static is not None else np.zeros(12, dtype=np.float32)),
                X_temporal=(X_temporal[i] if X_temporal is not None else np.zeros((tm_len, 10), dtype=np.float32)),
                time_mask=time_mask,
                Q_safe=q_safe,
                B_free=float(B_free[i]) if B_free is not None else 0.0,
                action_op=int(action_ops[i]) if action_ops is not None else 2,
                target_quantity=None if np.isnan(tq) else int(tq),
                target_price=None if np.isnan(tp) else float(tp),
                target_time=None if int(tt) < 0 else int(tt),
                reward=float(rewards[i]) if rewards is not None else 0.0,
            ))
    
    return macro_samples, micro_samples


def save_l4_samples(
    l4_samples: List[L4DistillSample],
    output_dir: str,
) -> None:
    """保存 L4 蒸馏样本到文件（变长 K 使用 object 数组）。"""
    os.makedirs(output_dir, exist_ok=True)
    if not l4_samples:
        return

    l4_data = {
        "days": np.array([s.day for s in l4_samples], dtype=np.int32),
        "global_states": np.stack([np.asarray(s.global_state, dtype=np.float32) for s in l4_samples]),
        "thread_states": np.array([np.asarray(s.thread_states, dtype=np.float32) for s in l4_samples], dtype=object),
        "teacher_alpha": np.array([np.asarray(s.teacher_alpha, dtype=np.float32) for s in l4_samples], dtype=object),
        "thread_ids": np.array([s.thread_ids for s in l4_samples], dtype=object),
    }
    np.savez(os.path.join(output_dir, "l4_samples.npz"), **l4_data)


def load_l4_samples(data_dir: str) -> List[L4DistillSample]:
    """从文件加载 L4 蒸馏样本。"""
    path = os.path.join(data_dir, "l4_samples.npz")
    if not os.path.exists(path):
        return []

    data = np.load(path, allow_pickle=True)
    days = data.get("days", np.array([], dtype=np.int32))
    global_states = data.get("global_states", data.get("global_state", None))
    thread_states = data.get("thread_states", None)
    teacher_alpha = data.get("teacher_alpha", None)
    thread_ids = data.get("thread_ids", None)

    if global_states is None:
        global_states = data.get("global_feats", None)
    if thread_states is None:
        thread_states = data.get("thread_feats", None)
    if teacher_alpha is None:
        teacher_alpha = data.get("teacher_weights", None)

    samples: List[L4DistillSample] = []
    for i in range(len(days)):
        gs = global_states[i] if global_states is not None else np.zeros((12,), dtype=np.float32)
        ts = thread_states[i] if thread_states is not None else np.zeros((1, 11), dtype=np.float32)
        ta = teacher_alpha[i] if teacher_alpha is not None else np.full((ts.shape[0],), 0.0, dtype=np.float32)
        ids = thread_ids[i] if thread_ids is not None else None

        samples.append(
            L4DistillSample(
                day=int(days[i]),
                global_state=np.asarray(gs, dtype=np.float32),
                thread_states=np.asarray(ts, dtype=np.float32),
                teacher_alpha=np.asarray(ta, dtype=np.float32),
                thread_ids=list(ids) if ids is not None else None,
            )
        )

    return samples


__all__ = [
    "MacroSample",
    "MicroSample",
    "L4DistillSample",
    "reconstruct_l2_goals",
    "extract_l3_samples_aop",
    "extract_l3_residuals",
    "load_tournament_data",
    "load_l4_distill_data",
    "save_samples",
    "load_samples",
    "save_l4_samples",
    "load_l4_samples",
    "extract_macro_state",
    "delta_to_bucket",
    "delta_to_bucket_soft",
    "compute_q_safe_offline",
    "compute_time_mask_offline",
    "build_role_embedding",
]
