#!/usr/bin/env python
"""分析 probe 阶段 vs post-probe 阶段的表现差异"""

import json
import glob
import sys
from collections import defaultdict
from pathlib import Path


def analyze_phases(log_dir: str, probe_days: int = 10):
    """分析两个阶段的表现"""
    logs = glob.glob(f'{log_dir}/agent_*LOS*.json')
    
    # BUYER 统计
    buyer_probe = {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'shortfall_qty': 0, 'need': 0, 'penalty_cost': 0.0, 'disposal_cost': 0.0}
    buyer_post = {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'shortfall_qty': 0, 'need': 0, 'penalty_cost': 0.0, 'disposal_cost': 0.0}
    
    # SELLER 统计
    seller_probe = {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'shortfall_qty': 0, 'need': 0, 'penalty_cost': 0.0, 'disposal_cost': 0.0}
    seller_post = {'shortfall': 0, 'exact': 0, 'overfull': 0, 'overfill': 0, 'shortfall_qty': 0, 'need': 0, 'penalty_cost': 0.0, 'disposal_cost': 0.0}
    
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
        
        # 统计 signed
        signed_by_day = defaultdict(int)
        for e in data.get('entries', []):
            if e.get('event') == 'signed':
                d = e.get('data', {})
                day = d.get('delivery_day')
                q = d.get('quantity', 0)
                if day is not None:
                    signed_by_day[day] += q
        
        # 统计 demand 和每天的成本参数
        demand_by_day = {}
        cost_params_by_day = {}  # {day: (shortfall_penalty, disposal_cost)}
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
                    cost_params_by_day[day] = (
                        d.get('shortfall_penalty', 0.0),
                        d.get('disposal_cost', 0.0)
                    )
        
        # 分阶段统计
        for day in demand_by_day:
            need = demand_by_day[day]
            if need <= 0:
                continue
            
            signed_qty = signed_by_day.get(day, 0)
            is_probe = day < probe_days
            
            if level == 0:  # SELLER
                stats = seller_probe if is_probe else seller_post
            else:  # BUYER
                stats = buyer_probe if is_probe else buyer_post
            
            stats['need'] += need
            shortfall_penalty_rate, disposal_cost_rate = cost_params_by_day.get(day, (0.0, 0.0))
            if signed_qty < need:
                stats['shortfall'] += 1
                sf_qty = need - signed_qty
                stats['shortfall_qty'] += sf_qty
                stats['penalty_cost'] += sf_qty * shortfall_penalty_rate
            elif signed_qty == need:
                stats['exact'] += 1
            else:
                stats['overfull'] += 1
                of_qty = signed_qty - need
                stats['overfill'] += of_qty
                stats['disposal_cost'] += of_qty * disposal_cost_rate
    
    return buyer_probe, buyer_post, seller_probe, seller_post


def print_stats(name, probe, post, probe_days):
    probe_total = probe['shortfall'] + probe['exact'] + probe['overfull']
    post_total = post['shortfall'] + post['exact'] + post['overfull']
    
    if probe_total == 0 or post_total == 0:
        print(f'=== {name}: 数据不足 ===')
        return
    
    print(f'=== {name} ===')
    print(f'                  Probe (day 0-{probe_days-1})    Post-probe (day {probe_days}+)')
    print(f'  Days:           {probe_total:5d}               {post_total:5d}')
    print(f'  Shortfall:      {probe["shortfall"]:5d} ({100*probe["shortfall"]/probe_total:5.1f}%)     {post["shortfall"]:5d} ({100*post["shortfall"]/post_total:5.1f}%)')
    print(f'  Exact:          {probe["exact"]:5d} ({100*probe["exact"]/probe_total:5.1f}%)     {post["exact"]:5d} ({100*post["exact"]/post_total:5.1f}%)')
    print(f'  Overfull:       {probe["overfull"]:5d} ({100*probe["overfull"]/probe_total:5.1f}%)     {post["overfull"]:5d} ({100*post["overfull"]/post_total:5.1f}%)')
    
    probe_sf_ratio = 100*probe['shortfall_qty']/probe['need'] if probe['need'] else 0
    post_sf_ratio = 100*post['shortfall_qty']/post['need'] if post['need'] else 0
    print(f'  Shortfall/Need: {probe_sf_ratio:5.1f}%               {post_sf_ratio:5.1f}%')
    
    probe_of_ratio = 100*probe['overfill']/probe['need'] if probe['need'] else 0
    post_of_ratio = 100*post['overfill']/post['need'] if post['need'] else 0
    print(f'  Overfill/Need:  {probe_of_ratio:5.1f}%               {post_of_ratio:5.1f}%')
    print(f'  Penalty Cost:   {probe["penalty_cost"]:8.1f}             {post["penalty_cost"]:8.1f}')
    print(f'  Disposal Cost:  {probe["disposal_cost"]:8.1f}             {post["disposal_cost"]:8.1f}')
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_probe_vs_postprobe.py <tournament_dir> [probe_days]")
        sys.exit(1)
    
    tourney_dir = Path(sys.argv[1])
    probe_days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    log_dir = tourney_dir / 'tracker_logs'
    
    if not log_dir.exists():
        print(f"Error: {log_dir} does not exist")
        sys.exit(1)
    
    print(f"分析 {tourney_dir.name}, probe_days={probe_days}")
    print()
    
    buyer_probe, buyer_post, seller_probe, seller_post = analyze_phases(str(log_dir), probe_days)
    
    print_stats('BUYER', buyer_probe, buyer_post, probe_days)
    print_stats('SELLER', seller_probe, seller_post, probe_days)
    
    # 汇总对比
    print("=== 阶段表现对比总结 ===")
    buyer_probe_total = buyer_probe['shortfall'] + buyer_probe['exact'] + buyer_probe['overfull']
    buyer_post_total = buyer_post['shortfall'] + buyer_post['exact'] + buyer_post['overfull']
    
    if buyer_probe_total > 0 and buyer_post_total > 0:
        probe_good = buyer_probe['exact'] / buyer_probe_total
        post_good = buyer_post['exact'] / buyer_post_total
        probe_bad = (buyer_probe['shortfall'] + buyer_probe['overfull']) / buyer_probe_total
        post_bad = (buyer_post['shortfall'] + buyer_post['overfull']) / buyer_post_total
        
        print(f"BUYER Exact率:  Probe={100*probe_good:.1f}%  Post-probe={100*post_good:.1f}%  差异={100*(probe_good-post_good):+.1f}%")
        print(f"BUYER 问题率:   Probe={100*probe_bad:.1f}%  Post-probe={100*post_bad:.1f}%  差异={100*(probe_bad-post_bad):+.1f}%")


if __name__ == '__main__':
    main()
