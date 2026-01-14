#!/usr/bin/env python
"""对比不同 Agent 的超量表现"""

import json
import glob
from collections import defaultdict
import sys

def analyze_agent(log_dir, pattern, name):
    logs = glob.glob(f'{log_dir}/agent_*{pattern}*.json')
    
    seller_stats = {'overfill': 0, 'need': 0, 'shortfall': 0, 'exact': 0, 'overfull': 0, 'total': 0}
    buyer_stats = {'overfill': 0, 'need': 0, 'shortfall': 0, 'exact': 0, 'overfull': 0, 'total': 0}
    
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
            if need > 0:
                signed_qty = signed_by_day.get(day, 0)
                stats['need'] += need
                stats['total'] += 1
                if signed_qty < need:
                    stats['shortfall'] += 1
                elif signed_qty == need:
                    stats['exact'] += 1
                else:
                    stats['overfull'] += 1
                    stats['overfill'] += (signed_qty - need)
    
    print(f'=== {name} ===')
    for role, stats in [('SELLER', seller_stats), ('BUYER', buyer_stats)]:
        if stats['total'] == 0:
            continue
        shortfall_pct = 100 * stats['shortfall'] / stats['total']
        exact_pct = 100 * stats['exact'] / stats['total']
        overfull_pct = 100 * stats['overfull'] / stats['total']
        print(f'{role}:')
        print(f'  Shortfall: {stats["shortfall"]} ({shortfall_pct:.1f}%)')
        print(f'  Exact:     {stats["exact"]} ({exact_pct:.1f}%)')
        print(f'  Overfull:  {stats["overfull"]} ({overfull_pct:.1f}%)')
        if stats['need'] > 0:
            overfill_ratio = 100 * stats['overfill'] / stats['need']
            print(f'  Total overfill/need: {stats["overfill"]}/{stats["need"]} = {overfill_ratio:.1f}%')
    print()


if __name__ == '__main__':
    tourney = sys.argv[1] if len(sys.argv) > 1 else 'tournament_history/20260111_143815_oneshot'
    log_dir = f'{tourney}/tracker_logs'
    
    print(f'Analyzing: {tourney}')
    print()
    
    analyze_agent(log_dir, 'LOS', 'LitaAgentOS')
    analyze_agent(log_dir, 'Ca', 'CautiousOneShotAgent')
