#!/usr/bin/env python3
"""Quick analysis of probe vs post-probe performance.

In OneShot:
- Level 0 agents: Have exo_input (buy from system), sell to Level 1
- Level 1 agents: Buy from Level 0, have exo_output (sell to system)

For BUYER role: need to buy = exo_output_qty (what they must deliver)
For SELLER role: need to sell = exo_input_qty (what they received)
"""
import json
from pathlib import Path
from collections import defaultdict

tracker_dir = Path(r'd:\SCML_initial\tournament_history\20260112_125825_oneshot\tracker_logs')
probe_days = 15

# Collect stats by agent type and phase (separate BUYER and SELLER)
los_buyer = {'probe': defaultdict(int), 'postprobe': defaultdict(int)}
los_seller = {'probe': defaultdict(int), 'postprobe': defaultdict(int)}
cos_buyer = {'probe': defaultdict(int), 'postprobe': defaultdict(int)}
cos_seller = {'probe': defaultdict(int), 'postprobe': defaultdict(int)}

def process_file(f, buyer_stats, seller_stats):
    with open(f) as fp:
        data = json.load(fp)
    
    entries = data.get('entries', [])
    
    # Track per-day need from daily_status
    for e in entries:
        day = e.get('day', 0)
        event = e.get('event', '')
        d = e.get('data', {})
        
        phase = 'probe' if day < probe_days else 'postprobe'
        
        if event == 'daily_status':
            # In OneShot:
            # - SELLER (Level 0): has exo_input, needs to sell it → need = exo_input_qty
            # - BUYER (Level 1): has exo_output, needs to buy input → need = exo_output_qty
            exo_in = d.get('exo_input_qty', 0)
            exo_out = d.get('exo_output_qty', 0)
            
            # The level determines the role
            level = d.get('my_output_product', 1) - 1  # 0 for Level 0, 1 for Level 1
            
            if exo_in > 0:  # Level 0: SELLER
                seller_stats[phase]['need'] += exo_in
            if exo_out > 0:  # Level 1: BUYER
                buyer_stats[phase]['need'] += exo_out
            
            buyer_stats[phase]['days'] += 1
            seller_stats[phase]['days'] += 1
        
        if event == 'signed':
            role = d.get('role', '')
            qty = d.get('quantity', 0)
            if role == 'buyer':
                buyer_stats[phase]['signed'] += qty
            elif role == 'seller':
                seller_stats[phase]['signed'] += qty

for f in tracker_dir.glob('agent_*LOS*.json'):
    process_file(f, los_buyer, los_seller)

for f in tracker_dir.glob('agent_*Ca*.json'):
    process_file(f, cos_buyer, cos_seller)

def calc_metrics(stats):
    need = stats['need']
    signed = stats['signed']
    if need == 0:
        return 0, 0, 0, need, signed
    exact = min(signed, need)
    overfill = max(0, signed - need)
    shortfall = max(0, need - signed)
    return exact/need*100, overfill/need*100, shortfall/need*100, need, signed

def print_stats(name, buyer_stats, seller_stats):
    print(f'=== {name} ===')
    print('  [BUYER role - Level 1 agents buying input]')
    e, o, s, n, sg = calc_metrics(buyer_stats['probe'])
    print(f'    Probe:      Need={n:5d}, Signed={sg:5d} | Exact={e:5.1f}%, Overfill={o:5.1f}%, Shortfall={s:5.1f}%')
    e, o, s, n, sg = calc_metrics(buyer_stats['postprobe'])
    print(f'    Post-Probe: Need={n:5d}, Signed={sg:5d} | Exact={e:5.1f}%, Overfill={o:5.1f}%, Shortfall={s:5.1f}%')
    
    print('  [SELLER role - Level 0 agents selling output]')
    e, o, s, n, sg = calc_metrics(seller_stats['probe'])
    print(f'    Probe:      Need={n:5d}, Signed={sg:5d} | Exact={e:5.1f}%, Overfill={o:5.1f}%, Shortfall={s:5.1f}%')
    e, o, s, n, sg = calc_metrics(seller_stats['postprobe'])
    print(f'    Post-Probe: Need={n:5d}, Signed={sg:5d} | Exact={e:5.1f}%, Overfill={o:5.1f}%, Shortfall={s:5.1f}%')

print_stats('LOS (LitaAgentOS)', los_buyer, los_seller)
print()
print_stats('COS (CautiousOneShotAgent)', cos_buyer, cos_seller)
