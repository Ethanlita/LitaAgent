#!/usr/bin/env python3
"""分析谈判的 round 分布 - 从 offer 序列推断"""
import json
from pathlib import Path
from collections import Counter, defaultdict

tracker_dir = Path('tournament_history/20260110_032957_oneshot/tracker_logs')
files = list(tracker_dir.glob('agent_*LOS*.json'))
print(f'Found {len(files)} LOS tracker files')

# 统计每个谈判的 offer 交换次数
offers_per_neg = []  # 每个谈判的 offer 数量
success_offers = []  # 成功谈判的 offer 数量
failure_offers = []  # 失败谈判的 offer 数量

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
        # 统计 offer 数量
        offers = [e for e in neg_entries if e.get('event') in ['offer_made', 'offer_received']]
        n_offers = len(offers)
        
        # 检查是否成功
        is_success = any(e.get('event') == 'success' for e in neg_entries)
        
        offers_per_neg.append(n_offers)
        if is_success:
            success_offers.append(n_offers)
        else:
            failure_offers.append(n_offers)

print(f'\nTotal negotiations: {len(offers_per_neg)}')
print(f'Success: {len(success_offers)}, Failure: {len(failure_offers)}')

# Offer 数量分布
print('\n--- Offers per negotiation distribution ---')
offer_counts = Counter(offers_per_neg)
for n in sorted(offer_counts.keys())[:15]:
    pct = offer_counts[n] / len(offers_per_neg) * 100
    bar = '█' * int(pct / 2)
    print(f'  {n:2d} offers: {offer_counts[n]:4d} ({pct:5.1f}%) {bar}')

# 成功 vs 失败
if success_offers:
    print(f'\nSuccess negotiations:')
    print(f'  Avg offers: {sum(success_offers)/len(success_offers):.1f}')
    print(f'  Most common: {Counter(success_offers).most_common(3)}')

if failure_offers:
    print(f'\nFailure negotiations:')
    print(f'  Avg offers: {sum(failure_offers)/len(failure_offers):.1f}')
    print(f'  Most common: {Counter(failure_offers).most_common(3)}')

# 推算 round (假设每个 round 有 2 个 offer: 1 sent + 1 received)
print('\n--- Estimated rounds (offers / 2) ---')
rounds = [n // 2 for n in offers_per_neg]
round_counts = Counter(rounds)
for r in sorted(round_counts.keys())[:10]:
    pct = round_counts[r] / len(rounds) * 100
    bar = '█' * int(pct / 2)
    print(f'  Round {r:2d}: {round_counts[r]:4d} ({pct:5.1f}%) {bar}')

# 累积分布
print('\n--- Cumulative distribution (negotiations ending by round X) ---')
total = len(rounds)
cumsum = 0
for r in sorted(set(rounds))[:10]:
    cumsum += round_counts[r]
    pct = cumsum / total * 100
    print(f'  By round {r:2d}: {pct:5.1f}%')

# 结论
if rounds:
    print(f'\n=== SUMMARY ===')
    print(f'Most negotiations end by round: {sorted(rounds)[int(len(rounds)*0.9)]} (90th percentile)')
    print(f'Average rounds: {sum(rounds)/len(rounds):.1f}')
    print(f'If n_rounds=20, round_rel at avg: {sum(rounds)/len(rounds)/20:.2f}')
