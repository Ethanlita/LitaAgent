#!/usr/bin/env python
"""分析 BUYER 表现随时间的变化趋势"""

import json
import glob
import sys
from collections import defaultdict
from pathlib import Path


def analyze_time_trend(log_dir: str, window_size: int = 10):
    """按时间窗口分析表现趋势"""
    logs = glob.glob(f'{log_dir}/agent_*LOS*.json')
    
    # 按天统计 BUYER
    day_stats = defaultdict(lambda: {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0})
    
    for f in logs:
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        level = None
        for e in data.get('entries', []):
            if e.get('event') == 'agent_initialized':
                level = e.get('data', {}).get('level')
                break
        
        if level != 1:  # BUYER only
            continue
        
        # 统计 signed
        signed_by_day = defaultdict(int)
        for e in data.get('entries', []):
            if e.get('event') == 'signed':
                d = e.get('data', {})
                day = d.get('delivery_day')
                q = d.get('quantity', 0)
                if day is not None:
                    signed_by_day[day] += q
        
        # 统计 demand
        demand_by_day = {}
        for e in data.get('entries', []):
            if e.get('event') == 'daily_status':
                d = e.get('data', {})
                day = d.get('current_step')
                demand = d.get('exo_output_qty', 0)
                if day is not None and demand is not None:
                    demand_by_day[day] = demand
        
        # 统计
        for day in demand_by_day:
            need = demand_by_day[day]
            if need <= 0:
                continue
            
            signed_qty = signed_by_day.get(day, 0)
            stats = day_stats[day]
            
            stats['need'] += need
            if signed_qty < need:
                stats['shortfall'] += 1
            elif signed_qty == need:
                stats['exact'] += 1
            else:
                stats['overfull'] += 1
                stats['overfill'] += (signed_qty - need)
    
    return day_stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_time_trend.py <tournament_dir> [window_size]")
        sys.exit(1)
    
    tourney_dir = Path(sys.argv[1])
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    log_dir = tourney_dir / 'tracker_logs'
    
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        sys.exit(1)
    
    day_stats = analyze_time_trend(str(log_dir), window_size)
    
    if not day_stats:
        print("No data found")
        return
    
    max_day = max(day_stats.keys())
    print(f"分析 {tourney_dir.name}, 共 {max_day + 1} 天, 窗口大小={window_size}")
    print()
    
    # 按窗口汇总
    print(f"{'Window':<15} {'Days':<6} {'Shortfall%':<12} {'Exact%':<10} {'Overfull%':<12} {'Overfill/Need':<15}")
    print("-" * 75)
    
    window_results = []
    for start in range(0, max_day + 1, window_size):
        end = min(start + window_size, max_day + 1)
        
        shortfall = exact = overfull = overfill = need = 0
        for day in range(start, end):
            if day in day_stats:
                stats = day_stats[day]
                shortfall += stats['shortfall']
                exact += stats['exact']
                overfull += stats['overfull']
                overfill += stats['overfill']
                need += stats['need']
        
        total = shortfall + exact + overfull
        if total == 0:
            continue
        
        shortfall_pct = 100 * shortfall / total
        exact_pct = 100 * exact / total
        overfull_pct = 100 * overfull / total
        overfill_ratio = 100 * overfill / need if need > 0 else 0
        
        window_label = f"Day {start}-{end-1}"
        print(f"{window_label:<15} {total:<6} {shortfall_pct:>10.1f}%  {exact_pct:>8.1f}%  {overfull_pct:>10.1f}%  {overfill_ratio:>12.1f}%")
        
        window_results.append({
            'window': window_label,
            'start': start,
            'exact_pct': exact_pct,
            'overfill_ratio': overfill_ratio
        })
    
    print()
    
    # 分析趋势
    if len(window_results) >= 3:
        print("=== 趋势分析 ===")
        probe_exact = window_results[0]['exact_pct']
        first_post = window_results[1]['exact_pct'] if len(window_results) > 1 else 0
        last_post = window_results[-1]['exact_pct']
        
        probe_overfill = window_results[0]['overfill_ratio']
        first_post_overfill = window_results[1]['overfill_ratio'] if len(window_results) > 1 else 0
        last_post_overfill = window_results[-1]['overfill_ratio']
        
        print(f"Exact率:     Probe={probe_exact:.1f}%  → 第一个Post窗口={first_post:.1f}%  → 最后窗口={last_post:.1f}%")
        print(f"Overfill比:  Probe={probe_overfill:.1f}%  → 第一个Post窗口={first_post_overfill:.1f}%  → 最后窗口={last_post_overfill:.1f}%")
        
        # 计算 post-probe 期间的趋势（排除 probe 阶段）
        if len(window_results) > 2:
            post_results = window_results[1:]  # 排除 probe
            exact_trend = post_results[-1]['exact_pct'] - post_results[0]['exact_pct']
            overfill_trend = post_results[-1]['overfill_ratio'] - post_results[0]['overfill_ratio']
            
            print()
            print(f"Post-probe 期间趋势:")
            print(f"  Exact率变化:    {exact_trend:+.1f}% ({'改善' if exact_trend > 0 else '恶化'})")
            print(f"  Overfill比变化: {overfill_trend:+.1f}% ({'改善' if overfill_trend < 0 else '恶化'})")


if __name__ == '__main__':
    main()
