#!/usr/bin/env python
"""
LitaAgentOS Tracker 日志分析工具
================================

分析 LitaAgentOS 在 OneShot 比赛中作为 BUYER/SELLER 时的表现：
- Shortfall: 签约量 < 需求量 (会产生 shortfall_penalty)
- Exact: 签约量 = 需求量 (最优)
- Overfull: 签约量 > 需求量 (会产生 disposal_cost)

使用方法:
    python analyze_shortfall.py [tournament_dir1] [tournament_dir2] ...
    
    如果不提供参数，将分析默认的比赛目录。

Tracker 日志结构说明:
- agent_initialized: 包含 level (0=SELLER, 1=BUYER)
- daily_status: 包含 exo_input_qty, exo_output_qty, current_step
- signed: 包含 delivery_day, quantity

OneShot 角色理解:
- Level 0 (SELLER): 从外部获得 exo_input_qty，需要通过谈判卖出
- Level 1 (BUYER): 需要通过谈判买入，然后卖给外部 exo_output_qty
"""

import json
import glob
import sys
import os
from collections import defaultdict


def analyze_daily_balance(log_dir, label=None):
    """
    分析指定目录下的 LOS tracker 日志。
    
    Args:
        log_dir: tracker_logs 目录的路径
        label: 显示标签，默认使用目录名
    
    Returns:
        dict: 包含 seller_stats 和 buyer_stats 的统计结果
    """
    if label is None:
        label = os.path.basename(os.path.dirname(log_dir))
    
    logs = glob.glob(f'{log_dir}/agent_*LOS*.json')
    
    if not logs:
        print(f'警告: 在 {log_dir} 中未找到 LOS 日志文件')
        return None
    
    seller_stats = {'shortfall': 0, 'exact': 0, 'overfull': 0, 'total_days': 0, 'scores': []}
    buyer_stats = {'shortfall': 0, 'exact': 0, 'overfull': 0, 'total_days': 0, 'scores': []}
    
    for f in logs:
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        level = None
        for e in data.get('entries', []):
            if e.get('event') == 'agent_initialized':
                level = e.get('data', {}).get('level')
                break
        
        if level is None:
            continue
        
        stats = seller_stats if level == 0 else buyer_stats
        
        # 分析 signed 合同按交付日统计
        signed_by_day = defaultdict(int)
        for e in data.get('entries', []):
            if e.get('event') == 'signed':
                d = e.get('data', {})
                day = d.get('delivery_day')
                q = d.get('quantity', 0)
                if day is not None:
                    signed_by_day[day] += q
        
        # 分析 daily_status 获取需求
        # OneShot 中:
        # - Level 0 (SELLER): 从外部获得 exo_input_qty，需要通过谈判卖出
        # - Level 1 (BUYER): 需要通过谈判买入，然后卖给外部 exo_output_qty
        demand_by_day = {}
        for e in data.get('entries', []):
            if e.get('event') == 'daily_status':
                d = e.get('data', {})
                day = d.get('current_step')
                if level == 0:
                    # SELLER: 需要卖出的量 = 外部输入量
                    demand = d.get('exo_input_qty', 0)
                else:
                    # BUYER: 需要买入的量 = 外部输出需求
                    demand = d.get('exo_output_qty', 0)
                if day is not None and demand is not None:
                    demand_by_day[day] = demand
        
        # 比较每天的签约量和需求量
        all_days = set(signed_by_day.keys()) | set(demand_by_day.keys())
        for day in all_days:
            signed = signed_by_day.get(day, 0)
            need = demand_by_day.get(day, 0)
            if need == 0:
                continue  # 跳过没有需求的天
            
            stats['total_days'] += 1
            if signed < need:
                stats['shortfall'] += 1
            elif signed > need:
                stats['overfull'] += 1
            else:
                stats['exact'] += 1
    
    print(f'=== {label} ===')
    print(f'SELLER (Level 0):')
    total = max(1, seller_stats["total_days"])
    print(f'  Total days with demand: {seller_stats["total_days"]}')
    print(f'  Shortfall days: {seller_stats["shortfall"]} ({100*seller_stats["shortfall"]/total:.1f}%)')
    print(f'  Exact days: {seller_stats["exact"]} ({100*seller_stats["exact"]/total:.1f}%)')
    print(f'  Overfull days: {seller_stats["overfull"]} ({100*seller_stats["overfull"]/total:.1f}%)')
    print()
    print(f'BUYER (Level 1):')
    total = max(1, buyer_stats["total_days"])
    print(f'  Total days with demand: {buyer_stats["total_days"]}')
    print(f'  Shortfall days: {buyer_stats["shortfall"]} ({100*buyer_stats["shortfall"]/total:.1f}%)')
    print(f'  Exact days: {buyer_stats["exact"]} ({100*buyer_stats["exact"]/total:.1f}%)')
    print(f'  Overfull days: {buyer_stats["overfull"]} ({100*buyer_stats["overfull"]/total:.1f}%)')
    print()
    
    return {'seller': seller_stats, 'buyer': buyer_stats, 'label': label}


def find_latest_tournaments(base_dir='d:/SCML_initial/tournament_history', n=2):
    """查找最近的 n 个 oneshot 比赛目录"""
    import re
    pattern = re.compile(r'^\d{8}_\d{6}_oneshot$')
    dirs = []
    for name in os.listdir(base_dir):
        if pattern.match(name):
            tracker_dir = os.path.join(base_dir, name, 'tracker_logs')
            if os.path.isdir(tracker_dir):
                dirs.append((name, tracker_dir))
    dirs.sort(reverse=True)
    return dirs[:n]


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 使用命令行参数指定目录
        for path in sys.argv[1:]:
            if os.path.isdir(path):
                tracker_dir = os.path.join(path, 'tracker_logs') if 'tracker_logs' not in path else path
                if os.path.isdir(tracker_dir):
                    analyze_daily_balance(tracker_dir)
                else:
                    print(f'错误: 找不到 tracker_logs 目录: {tracker_dir}')
            else:
                print(f'错误: 目录不存在: {path}')
    else:
        # 默认分析最近的两个比赛
        print('未指定目录，分析最近的两个 OneShot 比赛...\n')
        tournaments = find_latest_tournaments()
        for name, tracker_dir in tournaments:
            analyze_daily_balance(tracker_dir, name)
