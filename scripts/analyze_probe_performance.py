#!/usr/bin/env python
"""
分析 LOS 和 COS 在 Probe vs Post-probe 阶段的表现
用法: python scripts/analyze_probe_performance.py [tournament_ids...]
"""

import json
import glob
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys


def analyze_tournament(log_dir, n_steps, agent_pattern):
    """分析单个比赛中指定 agent 的表现"""
    logs = glob.glob(f'{log_dir}/agent_*{agent_pattern}*.json')
    
    # probe 阶段: 10% 但至少 15 天
    probe_days = max(15, int(n_steps * 0.1))
    
    results = {
        'BUYER': {
            'probe': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'shortfall_qty': 0},
            'post': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'shortfall_qty': 0},
        },
        'SELLER': {
            'probe': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'shortfall_qty': 0},
            'post': {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'need': 0, 'shortfall_qty': 0},
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
        
        signed_by_day = defaultdict(int)
        for e in data.get('entries', []):
            if e.get('event') == 'signed':
                d = e.get('data', {})
                day = d.get('delivery_day')
                q = d.get('quantity', 0)
                if day is not None:
                    signed_by_day[day] += q
        
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
        
        for day in demand_by_day:
            need = demand_by_day[day]
            if need <= 0:
                continue
            
            signed_qty = signed_by_day.get(day, 0)
            phase = 'probe' if day < probe_days else 'post'
            stats = results[role][phase]
            
            stats['need'] += need
            if signed_qty < need:
                stats['shortfall'] += 1
                stats['shortfall_qty'] += (need - signed_qty)
            elif signed_qty == need:
                stats['exact'] += 1
            else:
                stats['overfull'] += 1
                stats['overfill'] += (signed_qty - need)
    
    return results, probe_days


def get_tournament_config(tourney_dir):
    """获取比赛配置"""
    params_file = Path(tourney_dir) / 'params.json'
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = json.load(f)
        return params.get('n_steps', 50)
    return 50


def print_agent_analysis(tourneys, agent_name, agent_pattern):
    """打印单个 agent 的分析结果"""
    print(f'\n{"="*140}')
    print(f'{agent_name}: Probe vs Post-probe Performance')
    print(f'{"="*140}')
    
    for role in ['BUYER', 'SELLER']:
        print(f'\n{"="*60}')
        print(f'  {role}')
        print(f'{"="*60}')
        
        for phase_name, phase_key in [('Probe', 'probe'), ('Post-probe', 'post')]:
            print(f'\n--- {phase_name} ---')
            print(f"{'Tournament':<26} {'Steps':<6} {'ProbeDays':<10} {'Days':<6} {'Shortfall%':<11} {'Exact%':<9} {'Overfull%':<11} {'Overfill/Need%':<15} {'Shortfall/Need%':<15}")
            print('-'*120)
            
            for tourney, n_steps in tourneys:
                log_dir = Path(f'tournament_history/{tourney}/tracker_logs')
                if not log_dir.exists():
                    print(f'{tourney}: NOT FOUND')
                    continue
                
                data, probe_days = analyze_tournament(str(log_dir), n_steps, agent_pattern)
                stats = data[role][phase_key]
                total = stats['shortfall'] + stats['exact'] + stats['overfull']
                if total == 0:
                    continue
                
                shortfall_pct = 100 * stats['shortfall'] / total
                exact_pct = 100 * stats['exact'] / total
                overfull_pct = 100 * stats['overfull'] / total
                overfill_ratio = 100 * stats['overfill'] / stats['need'] if stats['need'] > 0 else 0
                shortfall_ratio = 100 * stats['shortfall_qty'] / stats['need'] if stats['need'] > 0 else 0
                
                probe_info = f'0-{probe_days-1}' if phase_key == 'probe' else f'{probe_days}+'
                print(f'{tourney:<26} {n_steps:<6} {probe_info:<10} {total:<6} {shortfall_pct:>9.1f}%  {exact_pct:>7.1f}%  {overfull_pct:>9.1f}%  {overfill_ratio:>13.1f}%  {shortfall_ratio:>13.1f}%')


def main():
    # 默认比赛列表
    default_tourneys = [
        '20260111_172250_oneshot',
        '20260111_233038_oneshot',
        '20260112_001335_oneshot',
        '20260112_005306_oneshot',
    ]
    
    # 从命令行参数获取比赛列表
    if len(sys.argv) > 1:
        tourney_ids = sys.argv[1:]
    else:
        tourney_ids = default_tourneys
    
    # 获取每个比赛的配置
    tourneys = []
    for tourney_id in tourney_ids:
        tourney_dir = Path(f'tournament_history/{tourney_id}')
        if tourney_dir.exists():
            n_steps = get_tournament_config(tourney_dir)
            tourneys.append((tourney_id, n_steps))
        else:
            print(f'Warning: {tourney_id} not found')
    
    if not tourneys:
        print('No tournaments found')
        return
    
    # 分析 LOS
    print_agent_analysis(tourneys, 'LitaAgentOS (LOS)', 'LOS')
    
    # 分析 COS (文件名是 Ca 而不是 CautiousOneShot)
    print_agent_analysis(tourneys, 'CautiousOneShotAgent (COS)', 'Ca_at')
    
    # Score comparison
    print('\n' + '='*140)
    print('Score Summary')
    print('='*140)
    print(f"{'Tournament':<26} {'Steps':<6} {'LOS Mean':<10} {'LOS Std':<10} {'LOS Rank':<10} {'COS Mean':<10} {'COS Rank':<10}")
    print('-'*100)
    
    for tourney, n_steps in tourneys:
        try:
            df = pd.read_csv(f'tournament_history/{tourney}/scores.csv')
            los = df[df['agent_type'].str.contains('LitaAgentOSTracked')]['score']
            cos = df[df['agent_type'].str.contains('CautiousOneShotAgentTracked')]['score']
            
            agg = df.groupby('agent_type')['score'].mean().sort_values(ascending=False)
            los_rank = list(agg.index).index([x for x in agg.index if 'LitaAgentOSTracked' in x][0]) + 1
            cos_rank = list(agg.index).index([x for x in agg.index if 'CautiousOneShotAgentTracked' in x][0]) + 1
            
            print(f'{tourney:<26} {n_steps:<6} {los.mean():<10.4f} {los.std():<10.4f} {los_rank:<10} {cos.mean():<10.4f} {cos_rank:<10}')
        except Exception as e:
            print(f'{tourney}: {e}')


if __name__ == '__main__':
    main()
