#!/usr/bin/env python3
"""分析谈判的 round 分布"""
import json
from pathlib import Path
from collections import Counter, defaultdict

tracker_dir = Path('tournament_history/20260110_032957_oneshot/tracker_logs')
files = list(tracker_dir.glob('agent_*LOS*.json'))
print(f'Found {len(files)} LOS tracker files')

# 先看一下数据结构
sample_file = files[0] if files else None
if sample_file:
    data = json.load(open(sample_file))
    entries = data.get('entries', [])
    negs = [e for e in entries if e.get('category') == 'negotiation']
    print(f'\nSample file: {sample_file.name}')
    print(f'Total negotiation entries: {len(negs)}')
    
    # 看每种事件类型的数据字段
    events_by_type = defaultdict(list)
    for e in negs:
        events_by_type[e.get('event')].append(e)
    
    print('\nEvent types and sample data keys:')
    for event_type, events in events_by_type.items():
        if events:
            sample_data = events[0].get('data', {})
            print(f'  {event_type}: {list(sample_data.keys())[:10]}')
            # 看看有没有 round 相关的字段
            for k in sample_data.keys():
                if 'round' in k.lower() or 'step' in k.lower() or 'time' in k.lower():
                    print(f'    -> {k} = {sample_data[k]}')

# 收集所有 offer_made 和 offer_received 的 round_rel
print('\n--- Analyzing round_rel from offers ---')
all_round_rels = []
success_round_rels = []
failure_round_rels = []

for f in files[:100]:
    data = json.load(open(f))
    entries = data.get('entries', [])
    negs = [e for e in entries if e.get('category') == 'negotiation']
    
    # 按 mechanism_id 分组
    negs_by_mech = defaultdict(list)
    for e in negs:
        mech_id = e.get('data', {}).get('mechanism_id')
        if mech_id:
            negs_by_mech[mech_id].append(e)
    
    for mech_id, neg_entries in negs_by_mech.items():
        # 检查是否成功
        is_success = any(e.get('event') == 'success' for e in neg_entries)
        
        # 收集 round_rel
        for e in neg_entries:
            rr = e.get('data', {}).get('round_rel')
            if rr is not None:
                all_round_rels.append(rr)
                if is_success:
                    success_round_rels.append(rr)
                else:
                    failure_round_rels.append(rr)

print(f'Total round_rel samples: {len(all_round_rels)}')
if all_round_rels:
    print(f'  Min: {min(all_round_rels):.3f}')
    print(f'  Max: {max(all_round_rels):.3f}')
    print(f'  Avg: {sum(all_round_rels)/len(all_round_rels):.3f}')

if success_round_rels:
    print(f'\nSuccess negotiations (final round_rel):')
    # 只看每个谈判的最后一个 round_rel
    print(f'  Avg round_rel at success: {sum(success_round_rels)/len(success_round_rels):.3f}')
    print(f'  Max round_rel at success: {max(success_round_rels):.3f}')

# 分桶统计
print('\n--- round_rel distribution ---')
buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(buckets) - 1):
    low, high = buckets[i], buckets[i+1]
    count = sum(1 for rr in all_round_rels if low <= rr < high)
    pct = count / len(all_round_rels) * 100 if all_round_rels else 0
    bar = '█' * int(pct / 2)
    print(f'  [{low:.1f}, {high:.1f}): {count:5d} ({pct:5.1f}%) {bar}')

print(f'\nTotal negotiations analyzed: {len(neg_rounds)}')

# 统计
if neg_rounds:
    print(f'\nMin round: {min(neg_rounds)}')
    print(f'Max round: {max(neg_rounds)}')
    print(f'Avg round: {sum(neg_rounds)/len(neg_rounds):.1f}')
    print(f'Median round: {sorted(neg_rounds)[len(neg_rounds)//2]}')
