#!/usr/bin/env python
"""
详细分析多次比赛中 BUYER/SELLER 在 probe/post-probe 阶段的表现
"""

import json
import glob
import sys
from collections import defaultdict
from pathlib import Path


def analyze_tournament(log_dir: str, probe_days: int = 10):
    """分析单个比赛"""
    logs = glob.glob(f'{log_dir}/agent_*LOS*.json')
    
    results = {
        'BUYER': {
            'probe': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'signed': 0},
            'post': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'signed': 0},
        },
        'SELLER': {
            'probe': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'signed': 0},
            'post': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'signed': 0},
        },
    }
    
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
        
        role = 'SELLER' if level == 0 else 'BUYER'
        
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
                if level == 0:
                    demand = d.get('exo_input_qty', 0)
                else:
                    demand = d.get('exo_output_qty', 0)
                if day is not None and demand is not None:
                    demand_by_day[day] = demand
        
        # 分阶段统计
        for day in demand_by_day:
            need = demand_by_day[day]
            if need <= 0:
                continue
            
            signed_qty = signed_by_day.get(day, 0)
            phase = 'probe' if day < probe_days else 'post'
            stats = results[role][phase]
            
            stats['need'] += need
            stats['signed'] += signed_qty
            if signed_qty < need:
                stats['shortfall'] += 1
            elif signed_qty == need:
                stats['exact'] += 1
            else:
                stats['overfull'] += 1
                stats['overfill'] += (signed_qty - need)
    
    return results


def print_comparison(tournaments: dict):
    """打印多次比赛的对比"""
    
    for role in ['BUYER', 'SELLER']:
        print(f"\n{'='*80}")
        print(f"  {role} 表现对比")
        print(f"{'='*80}")
        
        for phase in ['probe', 'post']:
            phase_name = 'Probe (day 0-9)' if phase == 'probe' else 'Post-probe (day 10+)'
            print(f"\n--- {phase_name} ---")
            print(f"{'比赛':<30} {'Days':<8} {'Shortfall%':<12} {'Exact%':<10} {'Overfull%':<12} {'Overfill/Need%':<15} {'Signed/Need%':<15}")
            print("-" * 110)
            
            for name, data in tournaments.items():
                stats = data[role][phase]
                total = stats['shortfall'] + stats['exact'] + stats['overfull']
                if total == 0:
                    continue
                
                shortfall_pct = 100 * stats['shortfall'] / total
                exact_pct = 100 * stats['exact'] / total
                overfull_pct = 100 * stats['overfull'] / total
                overfill_ratio = 100 * stats['overfill'] / stats['need'] if stats['need'] > 0 else 0
                signed_ratio = 100 * stats['signed'] / stats['need'] if stats['need'] > 0 else 0
                
                print(f"{name:<30} {total:<8} {shortfall_pct:>10.1f}%  {exact_pct:>8.1f}%  {overfull_pct:>10.1f}%  {overfill_ratio:>13.1f}%  {signed_ratio:>13.1f}%")
        
        # 计算变化趋势
        names = list(tournaments.keys())
        if len(names) >= 2:
            print(f"\n--- 变化趋势 (最新 vs 最早) ---")
            first_name, last_name = names[0], names[-1]
            
            for phase in ['probe', 'post']:
                first = tournaments[first_name][role][phase]
                last = tournaments[last_name][role][phase]
                
                first_total = first['shortfall'] + first['exact'] + first['overfull']
                last_total = last['shortfall'] + last['exact'] + last['overfull']
                
                if first_total == 0 or last_total == 0:
                    continue
                
                phase_name = 'Probe' if phase == 'probe' else 'Post'
                
                first_exact = 100 * first['exact'] / first_total
                last_exact = 100 * last['exact'] / last_total
                exact_diff = last_exact - first_exact
                
                first_overfill = 100 * first['overfill'] / first['need'] if first['need'] > 0 else 0
                last_overfill = 100 * last['overfill'] / last['need'] if last['need'] > 0 else 0
                overfill_diff = last_overfill - first_overfill
                
                first_shortfall = 100 * first['shortfall'] / first_total
                last_shortfall = 100 * last['shortfall'] / last_total
                shortfall_diff = last_shortfall - first_shortfall
                
                print(f"  {phase_name:<12} Exact: {first_exact:.1f}% → {last_exact:.1f}% ({exact_diff:+.1f}%)  "
                      f"Overfill/Need: {first_overfill:.1f}% → {last_overfill:.1f}% ({overfill_diff:+.1f}%)  "
                      f"Shortfall: {first_shortfall:.1f}% → {last_shortfall:.1f}% ({shortfall_diff:+.1f}%)")


def main():
    # 要分析的比赛列表 (110 worlds 50 steps)
    tourneys = [
        ('20260111_172250', '172250'),
        ('20260111_233038', '233038'),
        ('20260112_001335', '001335'),
        ('20260112_005306', '005306'),
    ]
    
    tournaments = {}
    
    for tourney_id, label in tourneys:
        log_dir = Path(f'tournament_history/{tourney_id}_oneshot/tracker_logs')
        if not log_dir.exists():
            print(f"Warning: {log_dir} does not exist, skipping")
            continue
        
        name = f"{tourney_id[-6:]} {label}"
        tournaments[name] = analyze_tournament(str(log_dir))
    
    if not tournaments:
        print("No tournaments found to analyze")
        return
    
    print_comparison(tournaments)
    
    # 打印总分对比
    print(f"\n{'='*80}")
    print("  总分对比 (110 worlds, 50 steps)")
    print(f"{'='*80}")
    
    scores = {
        '143815 修改前(probe_only)': {'score': 1.0296, 'rank': 9},
        '172250 大幅修改后': {'score': 1.0296, 'rank': 9},
        '233038 +rcp+ceil': {'score': 1.0051, 'rank': 10},
    }
    
    print(f"{'比赛':<30} {'Score':<10} {'Rank':<8}")
    print("-" * 50)
    for name, data in scores.items():
        print(f"{name:<30} {data['score']:<10.4f} {data['rank']:<8}")


if __name__ == '__main__':
    main()
