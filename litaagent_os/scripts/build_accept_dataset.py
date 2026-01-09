#!/usr/bin/env python3
"""
Accept Dataset Builder for LitaAgent-OS

从 OneShot Tournament World Logs 构建 accept_dataset.parquet。

样本单位：offer_sent（每个 agent 发出的每个 offer）
标签：y_accept = 1 如果该 offer 被对方接受（与 agreement 匹配）

用法:
    python build_accept_dataset.py --log-dirs <dir1> <dir2> ... --output accept_dataset.parquet
    python build_accept_dataset.py --tournament-dir <tournament> --output accept_dataset.parquet
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class OfferSample:
    """一条 offer_sent 样本"""
    # 键
    world_id: str
    negotiation_id: str
    sim_step: int
    proposer_id: str
    partner_id: str
    offer_index: int  # 该 proposer 在本次谈判中发出的第几个 offer

    # Offer
    q: int
    p: float
    t: int

    # Context
    role: str  # "BUYER" or "SELLER"
    round_rel: float
    n_lines: int
    price_min: float
    price_max: float
    need_remaining: int
    trading_price: Optional[float]

    # Label
    y_accept: int  # 1=accepted, 0=rejected


# ============================================================
# 解析工具
# ============================================================

def safe_literal_eval(value: Any, default: Any = None) -> Any:
    """安全解析字符串为 Python 对象"""
    if pd.isna(value):
        return default
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def parse_issues(issues_str: str) -> Dict[str, Tuple[float, float]]:
    """
    解析 issues 字符串，例如:
    "['quantity: (1, 10)', 'time: (0, 0)', 'unit_price: (15, 16)']"
    返回 {'quantity': (1, 10), 'time': (0, 0), 'unit_price': (15, 16)}
    """
    result = {}
    issues_list = safe_literal_eval(issues_str, default=[])
    for item in issues_list:
        if not isinstance(item, str):
            continue
        match = re.match(r"(\w+):\s*\(([^,]+),\s*([^)]+)\)", item)
        if match:
            name = match.group(1)
            lo = float(match.group(2))
            hi = float(match.group(3))
            result[name] = (lo, hi)
    return result


def offers_match(offer1: Tuple, offer2: Tuple, price_tol: float = 0.01) -> bool:
    """检查两个 offer 是否匹配（考虑浮点容忍）"""
    if offer1 is None or offer2 is None:
        return False
    if len(offer1) < 3 or len(offer2) < 3:
        return False
    # offer 格式: (q, t, p)
    q1, t1, p1 = int(offer1[0]), int(offer1[1]), float(offer1[2])
    q2, t2, p2 = int(offer2[0]), int(offer2[1]), float(offer2[2])
    return q1 == q2 and t1 == t2 and abs(p1 - p2) < price_tol


def get_responder(proposer: str, partners: List[str]) -> Optional[str]:
    """获取 responder（对方）"""
    if len(partners) != 2:
        return None
    for p in partners:
        if p != proposer:
            return p
    return None


# ============================================================
# 核心逻辑：从单个谈判提取样本
# ============================================================

def extract_samples_from_negotiation(
    neg_row: pd.Series,
    negs_row: Optional[pd.Series],
    world_id: str,
) -> List[OfferSample]:
    """
    从一条谈判记录提取所有 offer_sent 样本。
    
    关键逻辑：
    1. 从 offers 字段获取每个 agent 发出的所有 offer
    2. 对每个 offer，判断是否被接受（与 agreement 匹配）
    3. 只有当 offer 有明确响应时才生成样本（接受或被 counter）
    """
    samples = []
    
    neg_id = str(neg_row["id"])
    partners = safe_literal_eval(neg_row.get("partners"), default=[])
    if len(partners) != 2:
        return samples
    
    buyer = str(neg_row.get("buyer", ""))
    seller = str(neg_row.get("seller", ""))
    failed = bool(neg_row.get("failed", True))
    agreement = safe_literal_eval(neg_row.get("agreement"), default=None)
    offers_dict = safe_literal_eval(neg_row.get("offers"), default={})
    history = safe_literal_eval(neg_row.get("history"), default=[])
    sim_step = int(neg_row.get("sim_step", 0))
    
    # 解析 issues
    issues = parse_issues(str(neg_row.get("issues", "")))
    price_bounds = issues.get("unit_price", (0, 100))
    q_bounds = issues.get("quantity", (1, 10))
    n_lines = int(q_bounds[1])  # 近似
    price_min, price_max = float(price_bounds[0]), float(price_bounds[1])
    
    # 从 negs.csv 获取上下文
    trading_price = None
    need_map = {}  # agent_id -> need_remaining
    if negs_row is not None:
        trading_price = negs_row.get("trading_price")
        if pd.notna(trading_price):
            trading_price = float(trading_price)
        else:
            trading_price = None
        
        # 解析 need (注意 agent0/agent1 对应关系)
        agent_time0 = str(negs_row.get("agent_time0", ""))
        agent_time1 = str(negs_row.get("agent_time1", ""))
        
        # needed_sales/needed_supplies：根据 role 选择
        # 如果 agent 是 seller，用 needed_sales
        # 如果 agent 是 buyer，用 needed_supplies
        for agent_time, suffix in [(agent_time0, "0"), (agent_time1, "1")]:
            if agent_time:
                # 判断这个 agent 的 role
                if agent_time == seller:
                    # 卖家需要 sales
                    need_val = negs_row.get(f"needed_sales{suffix}")
                else:
                    # 买家需要 supplies
                    need_val = negs_row.get(f"needed_supplies{suffix}")
                if pd.notna(need_val):
                    need_map[agent_time] = int(need_val)
    
    # 从 history 重建 offer 序列（带时间顺序）
    # history 是一个 step 列表，每个 step 有 new_offers
    offer_sequence: List[Tuple[str, Tuple, float]] = []  # (proposer, offer, relative_time)
    
    for step_data in history:
        if not isinstance(step_data, dict):
            continue
        new_offers = step_data.get("new_offers", [])
        rel_time = step_data.get("relative_time", 0.0)
        if rel_time is None:
            rel_time = 0.0
        
        for item in new_offers:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            proposer, offer = item[0], item[1]
            if offer is not None and len(offer) >= 3:
                offer_sequence.append((str(proposer), tuple(offer), float(rel_time)))
    
    # 如果 history 中没有 offer，尝试从 offers_dict 重建（无时间信息）
    if not offer_sequence:
        for agent_id, agent_offers in offers_dict.items():
            for offer in agent_offers:
                if offer is not None and len(offer) >= 3:
                    offer_sequence.append((str(agent_id), tuple(offer), 0.0))
    
    if not offer_sequence:
        return samples
    
    # 判断每个 offer 的 y_accept
    # 规则：
    # 1. 只有最后一个与 agreement 匹配的 offer 才标记 y_accept=1
    #    （因为可能有交叉报价或同一方多次发出相同 offer）
    # 2. 如果 offer 后有对方的 counter offer -> y_accept = 0
    # 3. 如果是最后一个 offer 且没有 agreement（END/timeout）-> 不生成样本
    
    # 找出最后一个与 agreement 匹配的 offer 的索引
    last_agreement_match_idx = None
    if agreement is not None and not failed:
        for i in range(len(offer_sequence) - 1, -1, -1):
            _, offer, _ = offer_sequence[i]
            if offers_match(offer, agreement):
                last_agreement_match_idx = i
                break
    
    proposer_offer_index: Dict[str, int] = {}  # 每个 proposer 的 offer 计数
    
    for i, (proposer, offer, rel_time) in enumerate(offer_sequence):
        responder = get_responder(proposer, partners)
        if not responder:
            continue
        
        # 确定 proposer 的 role
        if proposer == buyer:
            role = "BUYER"
        elif proposer == seller:
            role = "SELLER"
        else:
            # 尝试从 partners 推断
            role = "BUYER" if proposer in [buyer] else "SELLER"
        
        # 计算 offer index
        if proposer not in proposer_offer_index:
            proposer_offer_index[proposer] = 0
        offer_idx = proposer_offer_index[proposer]
        proposer_offer_index[proposer] += 1
        
        # 判断 y_accept
        y_accept = None
        
        # 检查是否是最后一个与 agreement 匹配的 offer
        if i == last_agreement_match_idx:
            y_accept = 1
        else:
            # 检查后续是否有对方的 offer（counter）
            has_counter = False
            for j in range(i + 1, len(offer_sequence)):
                future_proposer, _, _ = offer_sequence[j]
                if future_proposer == responder:
                    has_counter = True
                    break
            
            if has_counter:
                y_accept = 0
            elif i == len(offer_sequence) - 1:
                # 最后一个 offer，没有 agreement
                if failed:
                    # END/timeout -> 不生成样本
                    y_accept = None
                else:
                    # 有 agreement 但不是这个 offer -> 理论上不应该发生
                    y_accept = None
        
        if y_accept is None:
            continue
        
        # 获取 need
        need_remaining = need_map.get(proposer, n_lines)
        
        # 创建样本
        sample = OfferSample(
            world_id=world_id,
            negotiation_id=neg_id,
            sim_step=sim_step,
            proposer_id=proposer,
            partner_id=responder,
            offer_index=offer_idx,
            q=int(offer[0]),
            p=float(offer[2]),  # offer 格式是 (q, t, p)
            t=int(offer[1]),
            role=role,
            round_rel=rel_time,
            n_lines=n_lines,
            price_min=price_min,
            price_max=price_max,
            need_remaining=need_remaining,
            trading_price=trading_price,
            y_accept=y_accept,
        )
        samples.append(sample)
    
    return samples


# ============================================================
# World 级别处理
# ============================================================

def process_world_dir(world_dir: Path) -> List[OfferSample]:
    """处理单个 World 目录"""
    samples = []
    
    negotiations_path = world_dir / "negotiations.csv"
    negs_path = world_dir / "negs.csv"
    
    if not negotiations_path.exists():
        print(f"  [SKIP] negotiations.csv not found in {world_dir}")
        return samples
    
    neg_df = pd.read_csv(negotiations_path)
    
    negs_df = None
    negs_map = {}
    if negs_path.exists():
        negs_df = pd.read_csv(negs_path)
        # 建立 name -> row 的映射
        for _, row in negs_df.iterrows():
            negs_map[str(row.get("name", ""))] = row
    
    # 获取 world_id
    world_id = world_dir.name
    
    for _, neg_row in neg_df.iterrows():
        neg_id = str(neg_row["id"])
        negs_row = negs_map.get(neg_id)
        
        try:
            extracted = extract_samples_from_negotiation(neg_row, negs_row, world_id)
            samples.extend(extracted)
        except Exception as e:
            print(f"  [WARN] Error processing negotiation {neg_id}: {e}")
            continue
    
    return samples


def find_world_dirs(base_dir: Path) -> List[Path]:
    """递归查找所有包含 negotiations.csv 的目录"""
    world_dirs = []
    
    for path in base_dir.rglob("negotiations.csv"):
        world_dirs.append(path.parent)
    
    return world_dirs


# ============================================================
# 主函数
# ============================================================

def build_dataset(
    log_dirs: List[Path],
    output_path: Path,
    max_worlds: Optional[int] = None,
) -> pd.DataFrame:
    """
    从多个日志目录构建数据集
    """
    all_samples: List[OfferSample] = []
    
    # 收集所有 world 目录
    world_dirs: List[Path] = []
    for log_dir in log_dirs:
        if not log_dir.exists():
            print(f"[WARN] Directory not found: {log_dir}")
            continue
        found = find_world_dirs(log_dir)
        print(f"Found {len(found)} world directories in {log_dir}")
        world_dirs.extend(found)
    
    if max_worlds:
        world_dirs = world_dirs[:max_worlds]
    
    print(f"Processing {len(world_dirs)} world directories...")
    
    for i, world_dir in enumerate(world_dirs):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(world_dirs)}] {world_dir.name[:50]}...")
        
        samples = process_world_dir(world_dir)
        all_samples.extend(samples)
    
    print(f"Total samples: {len(all_samples)}")
    
    if not all_samples:
        print("[ERROR] No samples extracted!")
        return pd.DataFrame()
    
    # 转换为 DataFrame
    df = pd.DataFrame([vars(s) for s in all_samples])
    
    # 添加衍生特征（与 features.py 对齐）
    df["q_norm"] = df["q"] / df["n_lines"].clip(lower=1)
    df["p_bin"] = ((df["p"] - df["price_min"]) / (df["price_max"] - df["price_min"] + 1e-6) > 0.5).astype(int)
    df["p_norm"] = (df["p"] - df["price_min"]) / (df["price_max"] - df["price_min"] + 1e-6)
    df["need_norm"] = df["need_remaining"] / df["n_lines"].clip(lower=1)
    df["role_is_seller"] = (df["role"] == "SELLER").astype(int)
    
    # 保存
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  y_accept=1: {(df['y_accept'] == 1).sum()} ({(df['y_accept'] == 1).mean()*100:.1f}%)")
    print(f"  y_accept=0: {(df['y_accept'] == 0).sum()} ({(df['y_accept'] == 0).mean()*100:.1f}%)")
    
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build accept dataset from OneShot tournament logs"
    )
    parser.add_argument(
        "--log-dirs",
        nargs="+",
        type=Path,
        help="Log directories to process",
    )
    parser.add_argument(
        "--tournament-dir",
        type=Path,
        help="Tournament directory (will search for world logs inside)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("accept_dataset.parquet"),
        help="Output file path",
    )
    parser.add_argument(
        "--max-worlds",
        type=int,
        default=None,
        help="Maximum number of worlds to process (for testing)",
    )
    
    args = parser.parse_args()
    
    log_dirs = []
    if args.log_dirs:
        log_dirs.extend(args.log_dirs)
    if args.tournament_dir:
        log_dirs.append(args.tournament_dir)
    
    if not log_dirs:
        print("[ERROR] No log directories specified. Use --log-dirs or --tournament-dir")
        return 1
    
    df = build_dataset(log_dirs, args.output, args.max_worlds)
    
    if df.empty:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
