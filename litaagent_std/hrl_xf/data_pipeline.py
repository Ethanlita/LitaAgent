"""数据管道 - 日志解析、状态重建、标签生成.

用于从 SCML 仿真日志中提取训练数据：
- L2 标签：日级目标向量重构
- L3 标签：轮级残差提取
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
    """轮级样本：用于 L3 残差模型.
    
    Attributes:
        negotiation_id: 谈判 ID
        history: 谈判历史，shape (T, 3)
        role: 角色索引 (0=Buyer, 1=Seller)
        x_role: L3 谈判角色 One-hot 编码 (2,)
                [is_buyer, is_seller] - 当前这个具体谈判的方向
                - [1, 0]: Buyer，当前谈判中我方是买家
                - [0, 1]: Seller，当前谈判中我方是卖家
                注意：与 L2 的 x_role (Multi-Hot 谈判能力) 含义不同
        goal: L2 目标向量，shape (16,)
        baseline: L1 基准动作 (q, p, t)
        residual: 残差 (Δq, Δp)
        time_label: 时间分类标签
        reward: 轮级奖励（如果有）
    
    关于 baseline 的重要说明（方案 E）：
    -----------------------------------------
    baseline 必须与推理时 L1SafetyLayer._compute_baseline() 的输出一致。
    
    当前实现使用 compute_l1_baseline_offline() 函数从日志状态重建 baseline：
    - 买入: q_base = min(Q_safe[δ]/2, B_free/price), p_base = market_price * 0.95
    - 卖出: q_base = inventory/2, p_base = market_price * 1.05
    
    这与 L1SafetyLayer._compute_baseline() 的计算逻辑一致，确保训练/推理一致性。
    
    回退逻辑（如果状态数据不足）：
    - JSON/Tracker 格式：使用 l1_baseline 字段（如果有）
    - CSV 格式：使用专家第一轮出价作为近似值（会产生警告）
    """
    negotiation_id: str
    history: np.ndarray
    role: int
    x_role: np.ndarray  # One-hot: [1,0]=Buyer, [0,1]=Seller
    goal: np.ndarray
    baseline: np.ndarray
    residual: np.ndarray
    time_label: int
    time_mask: Optional[np.ndarray] = None  # shape (H+1,)，0 或 -inf。旧数据可能为 None
    reward: Optional[float] = None


# ============== 桶定义 ==============

BUCKET_RANGES = [(0, 2), (3, 7), (8, 14), (15, 40)]


def delta_to_bucket(delta: int) -> int:
    """将相对交货时间映射到桶索引."""
    for i, (lo, hi) in enumerate(BUCKET_RANGES):
        if lo <= delta <= hi:
            return i
    return 3  # 默认长期


# ============== L1 Time Mask 离线计算 ==============

def compute_time_mask_offline(
    state_dict: Dict[str, Any],
    horizon: int = 40,
    min_tradable_qty: float = 1.0,
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
        
    Returns:
        time_mask: shape (H+1,)，值为 0 或 -inf
    """
    temporal_len = horizon + 1
    
    # ========== 1. 提取状态 ==========
    inventory = state_dict.get('inventory', {})
    n_lines = int(state_dict.get('n_lines', 1))
    
    # ========== 2. 提取承诺量 ==========
    commitments = state_dict.get('commitments', {})
    Q_in = np.array(commitments.get('Q_in', np.zeros(temporal_len)), dtype=np.float32)
    Q_out = np.array(commitments.get('Q_out', np.zeros(temporal_len)), dtype=np.float32)
    
    # 补齐长度
    if len(Q_in) < temporal_len:
        Q_in = np.pad(Q_in, (0, temporal_len - len(Q_in)))
    else:
        Q_in = Q_in[:temporal_len]
    if len(Q_out) < temporal_len:
        Q_out = np.pad(Q_out, (0, temporal_len - len(Q_out)))
    else:
        Q_out = Q_out[:temporal_len]
    
    # 截断为 horizon（用于轨迹计算）
    Q_in_h = Q_in[:horizon]
    Q_out_h = Q_out[:horizon]
    
    # ========== 3. 计算库存轨迹 L ==========
    if isinstance(inventory, dict):
        I_now = float(inventory.get('input', 0))
    else:
        I_now = float(inventory) if inventory else 0.0
    
    # 生产消耗（保守：假设满负荷）
    Q_prod = np.full(horizon, n_lines, dtype=np.float32)
    
    # 库存轨迹：I[k+1] = I[k] + Q_in[k] - Q_out[k] - Q_prod[k]
    net_flow = Q_in_h - Q_out_h - Q_prod
    L = I_now + np.cumsum(net_flow)
    
    # ========== 4. 计算库容 C_total ==========
    n_steps = int(state_dict.get('n_steps', 100))
    current_step = int(state_dict.get('current_step', 0))
    
    C_total = np.zeros(horizon, dtype=np.float32)
    for k in range(horizon):
        remaining_days = n_steps - (current_step + k)
        C_total[k] = n_lines * max(0, remaining_days)
    
    # ========== 5. 计算 Q_safe（逆向累积最小值） ==========
    raw_free = C_total - L
    reversed_free = raw_free[::-1]
    reversed_cummin = np.minimum.accumulate(reversed_free)
    Q_safe_h = reversed_cummin[::-1]
    Q_safe_h = np.maximum(Q_safe_h, 0)
    
    # 扩展为 H+1 维
    Q_safe = np.zeros(temporal_len, dtype=np.float32)
    Q_safe[:horizon] = Q_safe_h
    Q_safe[horizon] = Q_safe_h[-1] if len(Q_safe_h) > 0 else 0.0
    
    # ========== 6. 生成 time_mask ==========
    time_mask = np.where(Q_safe >= min_tradable_qty, 0.0, -np.inf)
    
    return time_mask.astype(np.float32)


def fix_invalid_time_label(
    time_label: int,
    time_mask: np.ndarray,
    t_base: int = 1,
) -> int:
    """修正落在被掩码位置的 time_label.
    
    策略：
    1. 如果 time_label 可行，直接返回
    2. 否则投影到最近的可行 δ
    3. 如果没有可行 δ，使用 t_base（如果合法）或 0
    
    Args:
        time_label: 原始时间标签
        time_mask: shape (H+1,)，0 或 -inf
        t_base: 基准时间
        
    Returns:
        修正后的 time_label
    """
    # 如果 time_mask 为 None 或当前位置可行，直接返回
    if time_mask is None:
        return time_label
    
    if time_label < len(time_mask) and time_mask[time_label] > -np.inf:
        return time_label
    
    # 找到所有可行的 delta
    valid_deltas = np.where(time_mask > -np.inf)[0]
    
    if len(valid_deltas) > 0:
        # 投影到最近的可行 delta
        distances = np.abs(valid_deltas - time_label)
        return int(valid_deltas[np.argmin(distances)])
    
    # 没有可行 delta，尝试使用 t_base
    if t_base < len(time_mask) and time_mask[t_base] > -np.inf:
        return t_base
    
    # 最后回退为 0
    return 0


# ============== L1 Baseline 离线计算 ==============

def compute_l1_baseline_offline(
    state_dict: Dict[str, Any],
    is_buying: bool,
    horizon: int = 40,
    min_tradable_qty: float = 1.0,
    reserve: float = 1000.0
) -> Tuple[float, float, int]:
    """从日志状态离线计算 L1 baseline.
    
    此函数复现 L1SafetyLayer._compute_baseline() 的计算逻辑，
    但使用日志中重建的状态而非实时 AWI。
    
    算法核心（必须与 l1_safety.py 完全一致）：
    
    1. Q_safe 计算（仅买入）：
       Q_safe[δ] = min_{k=δ}^{H-1} (C_total[k] - L[k])
       使用逆向累积最小值高效计算。
       
    2. B_free 计算：
       B_free = balance - reserve - Σ Payables  (标量，而非数组)
       
    3. 基准动作：
       - 买入: q_base = min(Q_safe[δ_best]/2, B_free/price)
               p_base = market_price * 0.95
       - 卖出: q_base = inventory/2, p_base = market_price * 1.05
       - t_base = 最早可行的交货日 δ
    
    Args:
        state_dict: 日志状态字典，包含:
            - balance: 当前余额
            - inventory: {'input': x, 'output': y} 或 sum 值
            - n_lines: 生产线数量
            - spot_price_in / catalog_prices: 原材料市场价
            - spot_price_out / catalog_prices: 成品市场价
            - commitments: {'Q_in': [...], 'Q_out': [...], 'Payables': [...]}
        is_buying: 是否为买入角色
        horizon: 规划视界
        min_tradable_qty: 最小可交易数量
        reserve: 安全储备金
        
    Returns:
        (q_base, p_base, t_base): 基准动作三元组
    """
    temporal_len = horizon + 1  # 支持 δ ∈ {0, 1, ..., H}
    
    # ========== 1. 提取状态 ==========
    balance = float(state_dict.get('balance', 10000))
    inventory = state_dict.get('inventory', {})
    n_lines = int(state_dict.get('n_lines', 1))
    
    # 市场价格提取（优先使用 catalog_prices，回退 spot_price_*）
    catalog_prices = state_dict.get('catalog_prices', {})
    trading_prices = state_dict.get('trading_prices', {})
    
    def get_price(product_id: int, default: float) -> float:
        """从价格字典安全获取价格."""
        for prices in [trading_prices, catalog_prices]:
            if prices:
                if isinstance(prices, dict):
                    if str(product_id) in prices:
                        return float(prices[str(product_id)])
                    if product_id in prices:
                        return float(prices[product_id])
                elif hasattr(prices, '__getitem__'):
                    try:
                        if product_id < len(prices):
                            return float(prices[product_id])
                    except (IndexError, TypeError):
                        pass
        return default
    
    spot_price_in = float(state_dict.get('spot_price_in', get_price(0, 10.0)))
    spot_price_out = float(state_dict.get('spot_price_out', get_price(1, 20.0)))
    
    # ========== 2. 提取承诺量 ==========
    commitments = state_dict.get('commitments', {})
    Q_in = np.array(commitments.get('Q_in', np.zeros(temporal_len)), dtype=np.float32)
    Q_out = np.array(commitments.get('Q_out', np.zeros(temporal_len)), dtype=np.float32)
    Payables = np.array(commitments.get('Payables', np.zeros(temporal_len)), dtype=np.float32)
    
    # 补齐长度到 temporal_len
    if len(Q_in) < temporal_len:
        Q_in = np.pad(Q_in, (0, temporal_len - len(Q_in)))
    if len(Q_out) < temporal_len:
        Q_out = np.pad(Q_out, (0, temporal_len - len(Q_out)))
    if len(Payables) < temporal_len:
        Payables = np.pad(Payables, (0, temporal_len - len(Payables)))
    
    # 截断为 horizon（用于轨迹计算）
    Q_in_h = Q_in[:horizon]
    Q_out_h = Q_out[:horizon]
    Payables_h = Payables[:horizon]
    
    # ========== 3. 计算库存轨迹 L ==========
    # 重要：使用原材料库存（input），与 l1_safety.py 保持一致
    # 因为采购入库的是原材料，库容约束针对原材料存储空间
    if isinstance(inventory, dict):
        I_now = float(inventory.get('input', 0))
    else:
        # 如果是单个数值，假设就是原材料库存
        I_now = float(inventory) if inventory else 0.0
    
    # 生产消耗（保守：假设满负荷）
    Q_prod = np.full(horizon, n_lines, dtype=np.float32)
    
    # 库存轨迹：I[k+1] = I[k] + Q_in[k] - Q_out[k] - Q_prod[k]
    net_flow = Q_in_h - Q_out_h - Q_prod
    L = I_now + np.cumsum(net_flow)
    
    # ========== 4. 计算库容 C_total ==========
    # 修复：与在线版本一致，使用 n_lines * (n_steps - (current_step + k))
    # 如果 state_dict 包含 n_steps/current_step，使用精确计算
    # 否则回退到 n_lines × horizon（保守估计）
    n_steps = int(state_dict.get('n_steps', 100))
    current_step = int(state_dict.get('current_step', 0))
    
    C_total = np.zeros(horizon, dtype=np.float32)
    for k in range(horizon):
        remaining_days = n_steps - (current_step + k)
        C_total[k] = n_lines * max(0, remaining_days)
    
    # ========== 5. 计算 Q_safe（关键：使用逆向累积最小值） ==========
    # Q_safe[δ] = min_{k=δ}^{H-1} (C_total[k] - L[k])
    raw_free = C_total - L  # shape (horizon,)
    
    # 逆向累积最小值
    reversed_free = raw_free[::-1]
    reversed_cummin = np.minimum.accumulate(reversed_free)
    Q_safe_h = reversed_cummin[::-1]
    
    # 非负约束
    Q_safe_h = np.maximum(Q_safe_h, 0)
    
    # 扩展为 H+1 维，支持 δ ∈ {0, 1, ..., H}
    Q_safe = np.zeros(temporal_len, dtype=np.float32)
    Q_safe[:horizon] = Q_safe_h
    Q_safe[horizon] = Q_safe_h[-1] if len(Q_safe_h) > 0 else 0.0
    
    # ========== 6. 计算 B_free（标量，而非数组） ==========
    # B_free = balance - reserve - Σ Payables
    total_payables = np.sum(Payables_h)
    B_free = max(0.0, balance - reserve - total_payables)
    
    # ========== 7. 计算基准动作 ==========
    if is_buying:
        market_price = spot_price_in
        
        # 找到最早可行的交货日
        best_delta = 1  # 默认明天
        for delta in range(temporal_len):
            if Q_safe[delta] >= min_tradable_qty:
                best_delta = delta
                break
        
        # 基准数量：安全量的一半（与在线一致）
        q_base = min(Q_safe[best_delta] / 2, B_free / max(market_price, 1.0))
        q_base = max(1.0, q_base)  # 至少 1 单位
        
        # 基准价格：略低于市场价（买方希望便宜）
        p_base = market_price * 0.95
        
    else:
        market_price = spot_price_out
        
        # 卖出使用成品库存（output）
        if isinstance(inventory, dict):
            available = float(inventory.get('output', 0))
        else:
            available = float(inventory) if inventory else 0.0
        
        best_delta = 1
        q_base = max(1.0, available / 2)
        
        # 基准价格：略高于市场价（卖方希望贵）
        p_base = market_price * 1.05
    
    return float(q_base), float(p_base), int(best_delta)


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
    
    # 通道 3: inventory_proj（库存投影）
    I_now = inventory_raw
    net_flow = Q_in - Q_out - Q_prod
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
    # 优先从 offers_snapshot 重建（Tracker 完整数据）
    # 否则回退到 commitments 中的预计算值（如有）
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
        
        # 计算 x_role: Multi-Hot 编码表示代理的"谈判能力"
        # x_role = [can_negotiate_buy, can_negotiate_sell]
        # 
        # SCML 2025 Standard 规则：
        # - 第一层 (is_first_level): 采购来自外生合约(SELLER)，只能谈判销售 → [0, 1]
        # - 最后一层 (is_last_level): 销售来自外生合约(BUYER)，只能谈判采购 → [1, 0]
        # - 中间层: 买卖都需要谈判 → [1, 1]
        # - 单层世界 (既是第一层又是最后层): 理论上无谈判 → [0, 0]
        #
        my_input_product = day_log.get('my_input_product', 0)
        my_output_product = day_log.get('my_output_product', 1)
        n_products = day_log.get('n_products', my_output_product + 1)
        
        is_first_level = (my_input_product == 0)
        is_last_level = (my_output_product == n_products - 1)
        
        # Multi-Hot 编码：[can_negotiate_buy, can_negotiate_sell]
        can_buy = 0.0 if is_first_level else 1.0   # 第一层不能谈判采购
        can_sell = 0.0 if is_last_level else 1.0  # 最后层不能谈判销售
        x_role = np.array([can_buy, can_sell], dtype=np.float32)
        
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

def extract_l3_residuals(
    negotiation_logs: List[Dict[str, Any]],
    horizon: int = 40,
    state_dict: Optional[Dict[str, Any]] = None,
    daily_states: Optional[Dict[int, Dict[str, Any]]] = None
) -> List[MicroSample]:
    """从谈判日志中提取 L3 残差标签.
    
    L1 Baseline 计算优先级：
    1. 如果日志包含 'l1_baseline' 字段（tracker 记录），直接使用
    2. 如果提供 daily_states 或 state_dict，调用 compute_l1_baseline_offline() 计算
    3. 回退：使用日志中的 baseline 字段（近似值，会产生警告）
    4. 最后回退：默认值（会产生警告）
    
    Args:
        negotiation_logs: 谈判日志列表
        horizon: 规划视界
        state_dict: 单一状态字典（用于所有谈判，向后兼容）
        daily_states: 按天索引的状态字典 {day: state_dict}，优先于 state_dict
        
    Returns:
        MicroSample 列表
    """
    samples = []
    baseline_warnings_shown = 0
    time_label_fixes = 0  # 统计 time_label 被修正的次数
    invalid_time_mask_count = 0  # time_mask 全部不可行的次数
    max_warnings = 3  # 最多显示警告次数
    baseline_stats = {'tracker': 0, 'computed': 0, 'log_fallback': 0, 'default': 0}
    
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
        is_buyer = neg.get('is_buyer', True)
        role = 0 if is_buyer else 1
        x_role = build_role_embedding(is_buyer)
        
        # L2 目标（如果有）
        goal_raw = neg.get('l2_goal', None)
        if isinstance(goal_raw, (list, tuple, np.ndarray)):
            goal = np.array(goal_raw, dtype=np.float32).reshape(-1)
            if goal.size != 16 or np.any(np.isnan(goal)):
                goal = np.zeros(16, dtype=np.float32)
        else:
            goal = np.zeros(16, dtype=np.float32)
        
        # L1 基准动作 - 优先级处理
        baseline = None
        baseline_source = 'unknown'
        
        # 优先级 1: tracker 记录的 l1_baseline
        lb = neg.get('l1_baseline')
        if lb is not None:
            lb_arr = np.array(lb, dtype=np.float32).flatten()
            if lb_arr.size >= 3 and not np.any(np.isnan(lb_arr[:3])):
                baseline = lb_arr[:3]
                baseline_source = 'tracker'
        
        # 优先级 2: 从状态离线计算（按天索引优先）
        if baseline is None and (daily_states is not None or state_dict is not None):
            # 获取当天的状态
            # 修复：统一使用 'sim_step' 或 'day' 键（兼容两种格式）
            day = neg.get('sim_step', neg.get('day', 0))
            if daily_states is not None and day in daily_states:
                current_state = daily_states[day]
            elif state_dict is not None:
                current_state = state_dict
            else:
                current_state = None
            
            if current_state is not None:
                q_base, p_base, t_base = compute_l1_baseline_offline(
                    current_state, is_buying=is_buyer, horizon=horizon
                )
                baseline = np.array([q_base, p_base, t_base], dtype=np.float32)
                baseline_source = 'computed'
        
        # 优先级 3: 使用日志中的 baseline 字段（可能是近似值）
        if baseline is None and 'baseline' in neg:
            bl_data = neg['baseline']
            # 确保 baseline 是正确的形状
            if isinstance(bl_data, (list, tuple)) and len(bl_data) >= 3:
                baseline = np.array(bl_data[:3], dtype=np.float32)
                baseline_source = 'log_fallback'
            elif isinstance(bl_data, np.ndarray) and bl_data.ndim >= 1 and len(bl_data) >= 3:
                baseline = np.array(bl_data[:3], dtype=np.float32)
                baseline_source = 'log_fallback'
        
        # 最后回退: 默认值
        if baseline is None or (isinstance(baseline, np.ndarray) and np.any(np.isnan(baseline))):
            baseline = np.array([1.0, 10.0, 1], dtype=np.float32)
            baseline_source = 'default'
        
        # 确保 baseline 是 1D 数组且有正确维度
        baseline = np.atleast_1d(baseline).flatten()
        if len(baseline) < 3:
            baseline = np.concatenate([baseline, np.array([1.0, 10.0, 1])[:3-len(baseline)]])
        
        # 统计 baseline 来源
        baseline_stats[baseline_source] = baseline_stats.get(baseline_source, 0) + 1
        
        # 警告：非理想 baseline 来源
        if baseline_source in ('log_fallback', 'default'):
            baseline_warnings_shown += 1
            if baseline_warnings_shown <= max_warnings:
                print(f"[WARN] L3 sample {neg.get('id', '?')}: baseline from {baseline_source}, "
                      "may not match L1SafetyLayer output. Consider using tracked runner with l1_baseline field.")
            elif baseline_warnings_shown == max_warnings + 1:
                print(f"[WARN] ... (suppressing further baseline warnings)")
        
        # 专家动作
        final_action = neg.get('final_action', {})
        expert_q = final_action.get('quantity', baseline[0])
        expert_p = final_action.get('price', baseline[1])
        expert_t = final_action.get('delta_t', int(baseline[2]))
        
        # 残差 = 专家动作 - L1 基准
        residual = np.array([
            expert_q - baseline[0],
            expert_p - baseline[1]
        ], dtype=np.float32)
        
        # 时间标签
        time_label = min(max(0, expert_t), horizon)
        
        # ========== 计算 time_mask ==========
        # 从状态重建 time_mask，确保训练/推理分布一致
        time_mask = None
        t_base = int(baseline[2]) if len(baseline) > 2 else 1
        if daily_states is not None or state_dict is not None:
            day = neg.get('sim_step', neg.get('day', 0))
            if daily_states is not None and day in daily_states:
                current_state = daily_states[day]
            elif state_dict is not None:
                current_state = state_dict
            else:
                current_state = None
            
            if current_state is not None:
                try:
                    time_mask = compute_time_mask_offline(current_state, horizon=horizon)
                except Exception:
                    time_mask = None  # 回退：无法计算时使用 None
        
        if time_mask is not None and np.isneginf(time_mask).all():
            # 若无可交易日期，至少保留 t_base 以避免全 -inf
            invalid_time_mask_count += 1
            safe_time_mask = np.full_like(time_mask, -np.inf, dtype=np.float32)
            if 0 <= t_base < len(safe_time_mask):
                safe_time_mask[t_base] = 0.0
            else:
                safe_time_mask[0] = 0.0
            time_mask = safe_time_mask
        
        # 修正非法 time_label（落在被 mask 的位置）
        original_time_label = time_label
        time_label = fix_invalid_time_label(time_label, time_mask, t_base)
        
        if time_label != original_time_label:
            time_label_fixes += 1
        
        # 奖励
        reward = neg.get('reward', None)
        
        samples.append(MicroSample(
            negotiation_id=neg.get('id', str(len(samples))),
            history=history,
            role=role,
            x_role=x_role,
            goal=goal,
            baseline=baseline,
            residual=residual,
            time_label=time_label,
            time_mask=time_mask,
            reward=reward
        ))
    
    if baseline_warnings_shown > 0:
        print(f"[INFO] Total samples with approximate baseline: {baseline_warnings_shown}/{len(samples)}")
    
    if time_label_fixes > 0:
        print(f"[INFO] Time labels fixed (projected to valid delta): {time_label_fixes}/{len(samples)}")
    
    if invalid_time_mask_count > 0:
        print(f"[WARN] time_mask all invalid in {invalid_time_mask_count} samples; forced t_base to be valid")
    
    return samples


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
    world_dirs = []
    
    # 模式 1: tracker_logs 目录（hrl_data_runner 默认输出）
    # 递归搜索所有 tracker_logs 目录
    tracker_log_dirs = glob.glob(os.path.join(tournament_dir, "**", "tracker_logs"), recursive=True)
    for tracker_dir in tracker_log_dirs:
        if os.path.isdir(tracker_dir):
            # 检查是否包含 agent_*.json
            agent_files = glob.glob(os.path.join(tracker_dir, "agent_*.json"))
            if agent_files and tracker_dir not in world_dirs:
                world_dirs.append(tracker_dir)
    
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
                    if child_path not in world_dirs:
                        world_dirs.append(child_path)
    
    # 模式 3: 传统 world_* 格式（JSON）
    pattern_dirs = glob.glob(os.path.join(tournament_dir, "**", "world_*"), recursive=True)
    for d in pattern_dirs:
        if os.path.isdir(d) and d not in world_dirs:
            world_dirs.append(d)
    
    # 模式 4: 目录根部直接包含 agent_*.json
    agent_files = glob.glob(os.path.join(tournament_dir, "agent_*.json"))
    if agent_files and tournament_dir not in world_dirs:
        world_dirs.append(tournament_dir)
    
    return world_dirs


def _process_world_dir(
    world_dir: str,
    agent_name: str,
    strict_json_only: bool,
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
        # 判断数据格式优先级：Tracker JSON > CSV
        # 注意：弃用 world_stats.json/negotiations.json，因为没有生成器
        # Tracker JSON 提供完整精确的状态和谈判数据
        # CSV 是降级方案，会导致精度问题（价格/压力通道为 0，baseline 不精确）
        has_tracker = len(glob.glob(os.path.join(world_dir, "agent_*.json"))) > 0
        has_csv = os.path.exists(os.path.join(world_dir, "stats.csv"))
        result['has_tracker'] = has_tracker
        result['has_csv'] = has_csv
        
        if has_tracker:
            # Tracker JSON 优先（完整精确的状态数据）
            daily_states = _load_daily_states_from_tracker(world_dir, agent_name)
            neg_logs = _load_negotiation_logs(world_dir, agent_name)
            offers_snapshot_by_day = _build_offers_snapshot_by_day(neg_logs) if neg_logs else None
            
            if daily_states:
                daily_logs = _daily_states_to_logs(
                    daily_states,
                    agent_name,
                    offers_snapshot_by_day=offers_snapshot_by_day,
                )
                result['macro_samples'].extend(reconstruct_l2_goals(daily_logs))
            
            result['micro_samples'].extend(extract_l3_residuals(neg_logs, daily_states=daily_states))
            result['used_tracker'] = True
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
) -> Tuple[List[MacroSample], List[MicroSample]]:
    """从锦标赛目录加载训练数据.
    
    数据格式：
    - **Tracker JSON 格式**（推荐）：包含完整的 L1 baseline、
      buy_pressure/sell_pressure、price_diff_in/out 等字段，适合完整训练。
    - **CSV 格式**（已弃用）：基础字段，部分状态通道无法精确重建，
      仅在 strict_json_only=False 时作为降级方案。
    
    Args:
        tournament_dir: 锦标赛日志目录
        agent_name: 要提取的代理名称（用于筛选相关记录）
        strict_json_only: 是否仅使用 Tracker JSON 格式（默认 True）
            - True: 跳过所有 CSV 目录，确保训练数据质量
            - False: 允许 CSV 降级（仅用于调试）
        num_workers: 并行进程数（默认自动，<=1 则串行）
        
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
            result = _process_world_dir(world_dir, agent_name, strict_json_only)
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
                  "⚠️ CSV data has quality issues (channels 6-9 = 0, approximate baseline). "
                  "For accurate training, use Tracker JSON format with hrl_data_runner.")
        if tracker_count == 0:
            print(f"[WARN] Using CSV format only ({csv_count} worlds). "
                  "⚠️ Training data quality issues:\n"
                  "  - Channels 6-9 (price_diff_in/out, buy_pressure, sell_pressure) = 0\n"
                  "  - L1 baseline uses approximate values (expert first offer)\n"
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
        if offers_snapshot_by_day:
            has_offers = offers_snapshot and (offers_snapshot.get('buy') or offers_snapshot.get('sell'))
            if not has_offers and day in offers_snapshot_by_day:
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
        按谈判聚合的日志列表，格式与 extract_l3_residuals 期望一致
    """
    from collections import defaultdict
    
    # P1 修复: 使用 partner+day+thread_id 区分同天多次谈判
    # thread_id 从 negotiation:started 事件的 mechanism_id 或递增计数器获取
    negotiations: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'partner': '',
        'day': 0,
        'current_step': 0,  # P0 修复: 记录当前步骤用于 delta_t 转换
        'is_buyer': True,
        'offer_history': [],
        'final_action': {},
        'final_status': 'unknown',
        'baseline': None,  # L1 基准（如果 tracker 有记录）
        'l2_goal': None,   # L2 目标（如果 tracker 有记录）
    })
    
    # P1 修复: 跟踪每个 partner+day 的谈判计数，用于区分同天多次谈判
    neg_counter: Dict[str, int] = defaultdict(int)
    
    for entry in entries:
        category = entry.get('category', '')
        event = entry.get('event', '')
        data = entry.get('data', {})
        day = entry.get('day', 0)
        
        if category != 'negotiation':
            continue
        
        partner = data.get('partner', '')
        if not partner:
            continue
        
        # P1 修复: 使用 partner+day+mechanism_id 或 partner+day+counter
        mechanism_id = data.get('mechanism_id', '')
        if mechanism_id:
            neg_key = f"{partner}_{day}_{mechanism_id}"
        else:
            # 回退: 对于 started 事件递增计数器
            base_key = f"{partner}_{day}"
            if event == 'started':
                neg_counter[base_key] += 1
            counter = neg_counter[base_key] if neg_counter[base_key] > 0 else 1
            neg_key = f"{partner}_{day}_{counter}"
        
        neg = negotiations[neg_key]
        neg['partner'] = partner
        neg['current_step'] = day  # 记录当前步骤
        neg['day'] = day
        
        if event == 'started':
            role = data.get('role', 'buyer')
            neg['is_buyer'] = (role == 'buyer')
            # 如果 tracker 记录了 L1 baseline，写入 'l1_baseline' 字段（优先级 1）
            if 'l1_baseline' in data:
                neg['l1_baseline'] = data['l1_baseline']
        
        elif event == 'offer_received':
            offer = data.get('offer', {})
            if offer:
                # P0 修复: 将绝对 delivery_day 转换为相对 delta_t
                abs_delivery = offer.get('delivery_day', offer.get('time', offer.get('t', offer.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)  # 相对于当前天的剩余天数
                
                neg['offer_history'].append({
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    # 修复：Tracker 用 unit_price，也支持 price/p
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    # P0 修复: 使用相对 delta_t 而非绝对 delivery_day
                    'delta_t': relative_delta_t,
                    'round': offer.get('round'),
                    'source': 'opponent',
                })
        
        elif event == 'offer_made':
            offer = data.get('offer', {})
            if offer:
                # P0 修复: 将绝对 delivery_day 转换为相对 delta_t
                abs_delivery = offer.get('delivery_day', offer.get('time', offer.get('t', offer.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                
                neg['offer_history'].append({
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    'delta_t': relative_delta_t,  # P0 修复: 相对 delta_t
                    'round': offer.get('round'),
                    'source': 'self',
                })
                # 最后一次发出的报价作为专家动作
                neg['final_action'] = {
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    'delta_t': relative_delta_t,  # P0 修复: 相对 delta_t
                }
        
        elif event == 'accept':
            offer = data.get('offer', {})
            if offer:
                # P0 修复: 将绝对 delivery_day 转换为相对 delta_t
                abs_delivery = offer.get('delivery_day', offer.get('time', offer.get('t', offer.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                
                # 接受的报价是最终协议
                neg['final_action'] = {
                    'quantity': offer.get('quantity', offer.get('q', 0)),
                    'price': offer.get('unit_price', offer.get('price', offer.get('p', 0))),
                    'delta_t': relative_delta_t,  # P0 修复: 相对 delta_t
                }
            neg['final_status'] = 'succeeded'
        
        elif event == 'success':
            agreement = data.get('agreement', {})
            if agreement:
                # P0 修复: 将绝对 delivery_day 转换为相对 delta_t
                abs_delivery = agreement.get('delivery_day', agreement.get('time', agreement.get('t', agreement.get('delta_t', day + 1))))
                relative_delta_t = max(0, abs_delivery - day)
                
                neg['final_action'] = {
                    'quantity': agreement.get('quantity', agreement.get('q', 0)),
                    'price': agreement.get('unit_price', agreement.get('price', agreement.get('p', 0))),
                    'delta_t': relative_delta_t,  # P0 修复: 相对 delta_t
                }
            neg['final_status'] = 'succeeded'
        
        elif event == 'failure':
            neg['final_status'] = 'failed'
    
    # 转换为列表格式
    result = []
    for neg_key, neg in negotiations.items():
        # 跳过没有历史的谈判
        if len(neg['offer_history']) == 0:
            continue
        
        # 如果没有记录的 baseline，使用第一次发出的报价作为近似
        if neg['baseline'] is None:
            for offer in neg['offer_history']:
                if offer.get('source') == 'self':
                    neg['baseline'] = [
                        offer['quantity'],
                        offer['price'],
                        offer['delta_t']
                    ]
                    break
            if neg['baseline'] is None:
                neg['baseline'] = [1, 10, 1]  # 默认值
        
        result.append({
            'id': neg_key,
            'partner': neg['partner'],
            'is_buyer': neg['is_buyer'],
            'sim_step': neg['day'],
            'offer_history': neg['offer_history'],
            'final_action': neg['final_action'],
            'final_status': neg['final_status'],
            'baseline': neg['baseline'],
            # 修复：输出 l1_baseline 字段（如果存在）
            'l1_baseline': neg.get('l1_baseline'),
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
        
        # 计算基准动作
        # ⚠️ CSV 格式限制：使用专家第一轮出价作为 baseline 的近似值。
        # 这不是真正的 L1 安全层输出，仅适用于调试和验证数据管道。
        # 正确的 L3 残差训练需要 JSON 格式提供的 l1_baseline 字段，
        # 该字段由 L1SafetyLayer.compute() 在谈判开始时计算并记录。
        baseline = [1, 10, 1]  # 默认 [quantity, price, delta_t]
        if agent_offers and len(agent_offers) > 0:
            first_offer = agent_offers[0]
            if isinstance(first_offer, (list, tuple)) and len(first_offer) >= 3:
                absolute_time = first_offer[1]
                delta_t = max(0, absolute_time - sim_step)
                baseline = [first_offer[0], first_offer[2], delta_t]  # q, p, delta_t
        
        log_entry = {
            'id': row.get('id', ''),
            'partners': partners_str,
            'is_buyer': is_buyer,
            'sim_step': row.get('sim_step', 0),
            'final_status': row.get('final_status', 'unknown'),
            'offer_history': offer_history,
            'final_action': final_action,
            'baseline': baseline,
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
        
        # 构建 time_masks 数组，缺失的填充 0.0（表示全部允许）
        time_masks = []
        for s in micro_samples:
            if s.time_mask is not None:
                tm = s.time_mask
                # 如果长度不足，填充 0.0
                if len(tm) < max_tm_len:
                    tm = np.concatenate([tm, np.zeros(max_tm_len - len(tm))])
                time_masks.append(tm)
            else:
                time_masks.append(np.zeros(max_tm_len, dtype=np.float32))
        
        micro_data = {
            'roles': np.array([s.role for s in micro_samples]),
            'x_roles': np.stack([s.x_role for s in micro_samples]),  # One-hot (2,) 谈判角色
            'goals': np.stack([s.goal for s in micro_samples]),
            'baselines': np.stack([s.baseline for s in micro_samples]),
            'residuals': np.stack([s.residual for s in micro_samples]),
            'time_labels': np.array([s.time_label for s in micro_samples]),
            'time_masks': np.stack(time_masks),  # 新增：L1 安全约束掩码
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
        x_roles = None
        time_masks = None
        
        if os.path.exists(l3_hist_path):
            hist_data = np.load(l3_hist_path, allow_pickle=True)
            histories = hist_data['histories']
        
        # 兼容旧格式（无 x_roles 字段）
        if 'x_roles' in data:
            x_roles = data['x_roles']
        
        # 兼容旧格式（无 time_masks 字段）
        if 'time_masks' in data:
            time_masks = data['time_masks']
        
        for i in range(len(data['roles'])):
            history = histories[i] if histories is not None else np.zeros((1, 3))
            role = int(data['roles'][i])
            x_role = x_roles[i] if x_roles is not None else build_role_embedding(role == 0)
            time_mask = time_masks[i] if time_masks is not None else None
            
            micro_samples.append(MicroSample(
                negotiation_id=str(i),
                history=history,
                role=role,
                x_role=x_role,
                goal=data['goals'][i],
                baseline=data['baselines'][i],
                residual=data['residuals'][i],
                time_label=int(data['time_labels'][i]),
                time_mask=time_mask,
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
    "compute_l1_baseline_offline",
    "build_role_embedding",
]
