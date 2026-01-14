"""
调试 first_proposals 逻辑
找出为什么 probe 阶段给每个 partner 报了 need 而不是 q_per_partner
"""
import json
from pathlib import Path
import math

# 找一个比赛日志
log_dir = Path('tournament_history/20260110_212745_oneshot/tracker_logs')
los_files = sorted(log_dir.glob('agent_*LOS*.json'))

print(f"找到 {len(los_files)} 个 LOS 日志文件")

# 分析第一个 BUYER (level=1) 的日志
for f in los_files[:20]:  # 只看前20个
    with open(f, encoding='utf-8') as fp:
        data = json.load(fp)
    
    level = None
    n_steps = 50
    for e in data.get('entries', []):
        if e.get('event') == 'agent_initialized':
            level = e.get('data', {}).get('level')
            n_steps = e.get('data', {}).get('n_steps', 50)
            break
    
    if level != 1:  # 只看 BUYER
        continue
    
    probe_days = max(10, int(n_steps * 0.1))
    
    print(f"\n{'='*80}")
    print(f"文件: {f.name}")
    print(f"level: {level}, n_steps: {n_steps}, probe_days: {probe_days}")
    
    # 收集 daily_status 和 offer_made 事件
    for day in range(min(3, n_steps)):  # 只看前3天
        day_offers = []
        exo_output = 0
        n_partners = 0
        
        for e in data.get('entries', []):
            event = e.get('event')
            d = e.get('data', {})
            
            if event == 'daily_status' and d.get('current_step') == day:
                exo_output = d.get('exo_output_qty', 0)
                
            if event == 'offer_made':
                offer_day = d.get('t_value', -1)
                if offer_day == day and d.get('role') == 'BUYER':
                    is_first = d.get('is_first', False)
                    if is_first:
                        day_offers.append({
                            'partner': d.get('partner'),
                            'qty': d.get('quantity'),
                            'price': d.get('price'),
                            'round_rel': d.get('round_rel')
                        })
        
        if day_offers:
            n_partners = len(set(o['partner'] for o in day_offers))
            
            # 计算理论值
            need = exo_output
            target = need + max(1, math.ceil(need * 0.1))  # overordering_ensure_plus_one
            q_per_partner_expected = max(1, math.ceil(target / n_partners)) if n_partners > 0 else 0
            
            print(f"\n  Day {day} (PROBE={day < probe_days}):")
            print(f"    need={need}, target={target}, n_partners={n_partners}")
            print(f"    理论 q_per_partner = ceil({target}/{n_partners}) = {q_per_partner_expected}")
            print(f"    实际报价:")
            for o in day_offers[:5]:  # 只显示前5个
                print(f"      {o['partner']}: qty={o['qty']}, round_rel={o['round_rel']}")
            
            actual_qty = day_offers[0]['qty'] if day_offers else 0
            if actual_qty != q_per_partner_expected:
                print(f"    ⚠️ 不匹配! 期望 q={q_per_partner_expected}, 实际 q={actual_qty}")
            else:
                print(f"    ✓ 匹配!")
    
    break  # 只分析一个文件
