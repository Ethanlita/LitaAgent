#!/usr/bin/env python3
"""分析 LOS 拒绝 offer 的原因"""
import json
from pathlib import Path
from collections import Counter, defaultdict

tracker_dir = Path('tournament_history/20260110_032957_oneshot/tracker_logs')
files = list(tracker_dir.glob('agent_*LOS*.json'))
print(f'Found {len(files)} LOS tracker files')

if not files:
    exit()

# 收集所有谈判数据
all_negs = []
all_daily = []
for f in files[:20]:  # 看前20个文件
    data = json.load(open(f))
    entries = data.get('entries', [])
    negs = [e for e in entries if e.get('category') == 'negotiation']
    daily = [e for e in entries if e.get('event') == 'daily_status']
    all_negs.extend(negs)
    all_daily.extend(daily)

# 统计事件类型
events = Counter(e.get('event') for e in all_negs)
print(f'\nNegotiation events: {dict(events)}')

# 查看 offer_made 和 offer_received 的价格
offers_made = [e for e in all_negs if e.get('event') == 'offer_made']
offers_received = [e for e in all_negs if e.get('event') == 'offer_received']
print(f'\nOffers made by LOS: {len(offers_made)}')
print(f'Offers received by LOS: {len(offers_received)}')

# 查看拒绝情况
rejected = [e for e in all_negs if e.get('event') == 'rejected']
print(f'Rejections: {len(rejected)}')

# 查看成功和失败
success = [e for e in all_negs if e.get('event') == 'success']
failed = [e for e in all_negs if e.get('event') == 'failed']
print(f'Success: {len(success)}')
print(f'Failed: {len(failed)}')

# 查看 daily_status 中的 trading_price
daily = [e for e in data.get('entries', []) if e.get('event') == 'daily_status']
if daily:
    d = daily[0]['data']
    print(f'\nTrading prices: {d.get("trading_prices")}')
    print(f'spot_price_in: {d.get("spot_price_in")}')
    print(f'spot_price_out: {d.get("spot_price_out")}')
    print(f'Level (my_input_product): {d.get("my_input_product")}')
    print(f'shortfall_penalty: {d.get("shortfall_penalty")}')
    print(f'disposal_cost: {d.get("disposal_cost")}')

# 看具体的 offer 价格
print('\n--- Sample offers received ---')
for e in offers_received[:5]:
    d = e.get('data', {})
    offer = d.get('offer', {})
    print(f"  role={d.get('role')}, partner={d.get('partner')}, "
          f"q={offer.get('quantity')}, p={offer.get('unit_price')}")

# 分析 failure 原因
print('\n--- Failure analysis ---')
failures = [e for e in all_negs if e.get('event') == 'failure']
print(f'Total failures: {len(failures)}')

# 查看 needed_supplies 和 needed_sales 的情况
print('\n--- Needed supplies/sales at each day ---')
needs_by_day = defaultdict(list)
for e in all_daily:
    day = e.get('day', 0)
    d = e.get('data', {})
    needs_by_day[day].append({
        'needed_supplies': d.get('needed_supplies', 0),
        'needed_sales': d.get('needed_sales', 0),
        'total_supplies': d.get('total_supplies', 0),
        'total_sales': d.get('total_sales', 0),
        'exo_input_qty': d.get('exo_input_qty', 0),
        'exo_output_qty': d.get('exo_output_qty', 0),
    })

for day in sorted(needs_by_day.keys())[:5]:
    records = needs_by_day[day]
    avg_needed_sales = sum(r['needed_sales'] for r in records) / len(records)
    avg_needed_supplies = sum(r['needed_supplies'] for r in records) / len(records)
    avg_total_sales = sum(r['total_sales'] for r in records) / len(records)
    avg_exo_input = sum(r['exo_input_qty'] for r in records) / len(records)
    print(f"Day {day}: needed_sales={avg_needed_sales:.1f}, needed_supplies={avg_needed_supplies:.1f}, "
          f"total_sales={avg_total_sales:.1f}, exo_input={avg_exo_input:.1f}")

# 查看 accept 事件中的价格分布
print('\n--- Accept price distribution ---')
accepts = [e for e in all_negs if e.get('event') == 'accept']
accept_prices = []
for e in accepts:
    offer = e.get('data', {}).get('offer', {})
    p = offer.get('unit_price')
    if p is not None:
        accept_prices.append(p)
if accept_prices:
    print(f"Accept prices: min={min(accept_prices)}, max={max(accept_prices)}, avg={sum(accept_prices)/len(accept_prices):.1f}")

# 查看 offer_received 但没有 accept 的情况
print('\n--- Offer received vs Accept ratio ---')
print(f"Offers received: {len(offers_received)}")
print(f"Accepts: {len(accepts)}")
print(f"Accept rate: {len(accepts)/max(1,len(offers_received))*100:.1f}%")
