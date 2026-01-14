"""
复盘 LOS BUYER 超量购买的具体案例
选取 probe 阶段和 post-probe 阶段各一个超量严重的日子进行详细分析
"""
import json
from pathlib import Path
from collections import defaultdict

log_dir = Path('tournament_history/20260110_212745_oneshot/tracker_logs')
los_files = sorted(log_dir.glob('agent_*LOS*.json'))

print(f"找到 {len(los_files)} 个 LOS 日志文件")

# 找到超量购买严重的 BUYER 案例
probe_examples = []
post_probe_examples = []

for f in los_files:
    with open(f) as fp:
        data = json.load(fp)
    
    level = None
    n_steps = 50
    for e in data.get('entries', []):
        if e.get('event') == 'agent_initialized':
            level = e.get('data', {}).get('level')
            n_steps = e.get('data', {}).get('n_steps', 50)
            break
    
    if level != 1:
        continue
    
    probe_days = max(10, int(n_steps * 0.1))
    entries = data.get('entries', [])
    daily_data = defaultdict(lambda: {'exo_output': 0, 'signed_qty': 0, 'contracts': []})
    
    for e in entries:
        event = e.get('event')
        d = e.get('data', {})
        
        if event == 'daily_status':
            day = d.get('current_step')
            daily_data[day]['exo_output'] = d.get('exo_output_qty', 0)
        
        elif event == 'signed':
            day = d.get('delivery_day')
            qty = d.get('quantity', 0)
            partner = d.get('partner', '')
            price = d.get('price', 0)
            daily_data[day]['signed_qty'] += qty
            daily_data[day]['contracts'].append({
                'partner': partner,
                'qty': qty,
                'price': price
            })
    
    for day, dd in daily_data.items():
        if dd['exo_output'] <= 0:
            continue
        ratio = dd['signed_qty'] / dd['exo_output']
        if ratio >= 2.5:
            example = (f.name, day, ratio, data, dd)
            if day < probe_days:
                probe_examples.append(example)
            else:
                post_probe_examples.append(example)

probe_examples.sort(key=lambda x: x[2], reverse=True)
post_probe_examples.sort(key=lambda x: x[2], reverse=True)

print(f"\n找到 {len(probe_examples)} 个 probe 阶段超量案例")
print(f"找到 {len(post_probe_examples)} 个 post-probe 阶段超量案例")


def detailed_day_replay(filename, target_day, ratio, data, daily_summary):
    """
    详细复盘某一天的决策过程
    重点分析: 
    1. first_proposals 发出了什么
    2. 对手的响应是什么
    3. _select_subset 选择了哪些来 Accept
    """
    entries = data.get('entries', [])
    
    # 提取基本信息
    init_info = {}
    for e in entries:
        if e.get('event') == 'agent_initialized':
            init_info = e.get('data', {})
            break
    
    n_steps = init_info.get('n_steps', 50)
    n_lines = init_info.get('n_lines', 10)
    probe_days = max(10, int(n_steps * 0.1))
    phase = "PROBE" if target_day < probe_days else "POST-PROBE"
    
    # 提取当天的 daily_status
    daily_status = {}
    for e in entries:
        if e.get('event') == 'daily_status':
            d = e.get('data', {})
            if d.get('current_step') == target_day:
                daily_status = d
                break
    
    # 收集当天的所有谈判事件 (按 mechanism_id 分组)
    negotiations = defaultdict(lambda: {
        'partner': '',
        'role': '',
        'started': None,
        'offers_made': [],
        'offers_received': [],
        'aop_actions': [],
        'outcome': None,  # 'signed', 'failure'
        'signed_info': None
    })
    
    for e in entries:
        event = e.get('event')
        d = e.get('data', {})
        
        # 根据 delivery_day 过滤当天的事件
        # started 事件可能没有 delivery_day，用 mechanism_id 追踪
        
        if event == 'started':
            mech_id = d.get('mechanism_id', '')
            negotiations[mech_id]['partner'] = d.get('partner', '')
            negotiations[mech_id]['role'] = d.get('role', '')
            negotiations[mech_id]['started'] = d
            # issues 可能是字符串或字典
        
        elif event == 'offer_made':
            mech_id = d.get('mechanism_id', '')
            offer = d.get('offer', {})
            delivery_day = offer.get('delivery_day', -1)
            if delivery_day == target_day:
                negotiations[mech_id]['offers_made'].append(d)
        
        elif event == 'offer_received':
            mech_id = d.get('mechanism_id', '')
            offer = d.get('offer', {})
            delivery_day = offer.get('delivery_day', -1)
            if delivery_day == target_day:
                negotiations[mech_id]['offers_received'].append(d)
        
        elif event == 'aop_action':
            mech_id = d.get('mechanism_id', '')
            # aop_action 需要检查是否关联到当天
            negotiations[mech_id]['aop_actions'].append(d)
        
        elif event == 'signed':
            delivery_day = d.get('delivery_day', -1)
            if delivery_day == target_day:
                partner = d.get('partner', '')
                # 找到对应的 mechanism_id
                for mech_id, neg in negotiations.items():
                    if neg['partner'] == partner:
                        neg['outcome'] = 'signed'
                        neg['signed_info'] = d
                        break
        
        elif event == 'success':
            partner = d.get('partner', '')
            agreement = d.get('agreement', {})
            if agreement.get('time') == target_day:
                for mech_id, neg in negotiations.items():
                    if neg['partner'] == partner and neg['outcome'] is None:
                        neg['outcome'] = 'success'
                        break
    
    # 筛选出当天相关的谈判
    day_negotiations = {}
    for mech_id, neg in negotiations.items():
        # 检查是否有当天的 offer_made 或 signed
        if neg['offers_made'] or neg.get('signed_info'):
            day_negotiations[mech_id] = neg
    
    # 打印复盘报告
    print(f"\n{'='*100}")
    print(f"📅 Day {target_day} 完整复盘 [{phase}]")
    print(f"   文件: {filename}")
    print(f"{'='*100}")
    
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 🔹 基本信息                                                                                 │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    print(f"   n_steps: {n_steps}, n_lines: {n_lines}")
    print(f"   阶段: {phase} (probe_days = {probe_days})")
    print(f"   exo_output_qty (外生需求): {daily_status.get('exo_output_qty', '?')}")
    print(f"   exo_output_price (外生售价): {daily_status.get('exo_output_price', '?')}")
    print(f"   needed_supplies: {daily_status.get('needed_supplies', '?')}")
    
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 🔹 最终结果                                                                                 │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    print(f"   签约总数量: {daily_summary['signed_qty']}")
    print(f"   外生需求量: {daily_summary['exo_output']}")
    print(f"   超量比例: {ratio:.2f}x")
    print(f"   签约合同数: {len(daily_summary['contracts'])}")
    for i, c in enumerate(daily_summary['contracts']):
        print(f"     [{i+1}] {c['partner']:25s} qty={c['qty']:2d} price={c['price']}")
    
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 🔹 谈判过程详情                                                                             │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # 按 partner 整理
    partner_negs = {}
    for mech_id, neg in day_negotiations.items():
        partner = neg['partner']
        if partner:
            partner_negs[partner] = neg
    
    # 直接从 entries 提取更完整的信息
    # 按 partner 收集该天的 offer_made
    offers_by_partner = defaultdict(list)
    signed_by_partner = {}
    
    for e in entries:
        event = e.get('event')
        d = e.get('data', {})
        
        if event == 'offer_made':
            offer = d.get('offer', {})
            delivery_day = offer.get('delivery_day', -1)
            if delivery_day == target_day:
                partner = d.get('partner', '')
                offers_by_partner[partner].append({
                    'quantity': offer.get('quantity'),
                    'unit_price': offer.get('unit_price'),
                    'round': offer.get('round', 0),
                    'reason': d.get('reason', '')
                })
        
        elif event == 'signed':
            delivery_day = d.get('delivery_day', -1)
            if delivery_day == target_day:
                partner = d.get('partner', '')
                signed_by_partner[partner] = {
                    'quantity': d.get('quantity'),
                    'price': d.get('price'),
                }
    
    print(f"\n   LOS 作为 BUYER，需要从 Level 0 (SELLER) 购买")
    print(f"   当天与 {len(offers_by_partner)} 个 partner 有谈判")
    
    print(f"\n   📤 LOS 发出的 first_proposal:")
    first_proposals = []
    for partner, offers in offers_by_partner.items():
        # 找 first_proposal
        for o in offers:
            if o['reason'] == 'first_proposal':
                first_proposals.append((partner, o))
                break
    
    if first_proposals:
        for partner, o in sorted(first_proposals, key=lambda x: x[0]):
            signed = "✅ SIGNED" if partner in signed_by_partner else ""
            print(f"      → {partner:25s} qty={o['quantity']:2d} price={o['unit_price']:5.1f} {signed}")
    else:
        print(f"      (无 first_proposal 记录)")
    
    print(f"\n   ✅ 最终签约的 partner:")
    for partner, info in sorted(signed_by_partner.items()):
        print(f"      → {partner:25s} qty={info['quantity']:2d} price={info['price']:5.1f}")
    
    # 分析: 为什么这么多 partner 都签约了?
    print(f"\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 🔍 关键问题分析                                                                             │")
    print(f"└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    need = daily_summary['exo_output']
    signed_qty = daily_summary['signed_qty']
    n_partners_signed = len(signed_by_partner)
    n_partners_negotiated = len(offers_by_partner)
    
    print(f"""
   问题: 为什么 LOS 与 {n_partners_signed} 个 partner 都签约了?
   
   📊 数据:
      - 外生需求 (exo_output_qty): {need}
      - 签约总量: {signed_qty}
      - 超量: {signed_qty - need} ({((signed_qty/need - 1) * 100):.0f}%)
      - 谈判 partner 数: {n_partners_negotiated}
      - 签约 partner 数: {n_partners_signed}
   
   🔍 可能的原因:
""")
    
    # 检查是否每个 partner 都签约了
    if n_partners_signed == n_partners_negotiated:
        print(f"      1. ❌ 所有谈判的 partner 都签约了 ({n_partners_signed}/{n_partners_negotiated})")
        print(f"         → _select_subset 没有过滤任何 partner")
    else:
        print(f"      1. ⚠️ 部分 partner 签约 ({n_partners_signed}/{n_partners_negotiated})")
    
    # 检查每个签约的数量
    avg_qty = signed_qty / n_partners_signed if n_partners_signed > 0 else 0
    print(f"      2. 平均每个 partner 签约数量: {avg_qty:.1f}")
    print(f"         → 如果 LOS 给每个 partner 都报了 need={need}，被全部接受就会超量 {n_partners_signed}x")
    
    # 检查价格
    if first_proposals:
        prices = [o['unit_price'] for _, o in first_proposals if o['unit_price']]
        if prices:
            print(f"      3. first_proposal 价格: {min(prices):.1f} - {max(prices):.1f}")
            print(f"         → 作为 BUYER，LOS 应该报低价 (p_min) 开始让步")
    
    print(f"""
   💡 根本原因推断:
      
      在 {phase} 阶段，LOS 的报价策略是:
""")
    
    if phase == "PROBE":
        print(f"""      - probe 阶段给每个 partner 分散报价
      - 目标: 收集 BOU 数据
      - 问题: 当多个 partner 都接受时，_select_subset 选择了全部
      
      🎯 _select_subset 逻辑分析:
         - need_remaining = {need}
         - 如果 8 个 partner 都报了 qty=6 并被接受
         - 评分公式: score = utility - penalty
         - utility 是累加的，选更多 → utility ↑
         - penalty 只有在超量时才出现
         - 但 disposal_penalty << shortfall_penalty
         - 所以优化器选择"全都要"
""")
    else:
        print(f"""      - post-probe 阶段使用 q = remaining / p_eff 计算
      - 问题: 如果 p_eff 估计过低，q 会过大
      - 或者: 如果多个 partner 同时接受，_select_subset 选择太多
      
      🎯 可能的问题:
         1. BOU 估计的 p_eff 不准确
         2. _select_subset 没有严格限制总量
         3. post_probe_min_partners = 3 导致至少发 3 单
""")
    
    return init_info, daily_status


# 执行复盘
if probe_examples:
    print("\n" + "="*100)
    print("🔬 PROBE 阶段案例详细复盘")
    print("="*100)
    example = probe_examples[0]
    detailed_day_replay(*example)

if post_probe_examples:
    print("\n" + "="*100)
    print("🔬 POST-PROBE 阶段案例详细复盘")
    print("="*100)
    example = post_probe_examples[0]
    detailed_day_replay(*example)


# 额外分析: 签约是在 first_proposal 还是 counter 阶段发生的?
print("\n" + "="*100)
print("🔬 深度分析: 签约发生在哪个阶段?")
print("="*100)

def analyze_signing_phase(filename, target_day, ratio, data, daily_summary):
    """分析签约是在 first_proposal 回合还是后续 counter 回合发生的"""
    entries = data.get('entries', [])
    
    # 收集该天的 aop_action 事件
    # aop_action 记录了 LOS 对每个 partner 的响应
    aop_by_partner = defaultdict(list)
    
    for e in entries:
        if e.get('event') == 'aop_action':
            d = e.get('data', {})
            partner = d.get('partner', '')
            sim_step = d.get('sim_step', -1)
            if sim_step == target_day:
                aop_by_partner[partner].append({
                    'round': d.get('round', 0),
                    'action_op': d.get('action_op', ''),
                    'response_type': d.get('response_type', ''),
                })
    
    # 收集 success 事件
    success_by_partner = {}
    for e in entries:
        if e.get('event') == 'success':
            d = e.get('data', {})
            agreement = d.get('agreement', {})
            if agreement.get('time') == target_day:
                partner = d.get('partner', '')
                success_by_partner[partner] = agreement
    
    print(f"\n📅 Day {target_day} 签约阶段分析")
    print(f"   签约 partner 数: {len(success_by_partner)}")
    
    print(f"\n   📋 各 partner 的 aop_action 记录:")
    for partner in sorted(aop_by_partner.keys()):
        actions = aop_by_partner[partner]
        signed = "✅" if partner in success_by_partner else "❌"
        print(f"\n      {partner} {signed}")
        for a in actions:
            print(f"         Round {a['round']}: {a['action_op']} ({a['response_type']})")

if probe_examples:
    example = probe_examples[0]
    analyze_signing_phase(*example)


# 关键问题: LOS 的 first_proposal 是不是直接被对方 Accept 了?
print("\n" + "="*100)
print("🔬 核心问题: 对方是直接 Accept 还是需要多轮谈判?")
print("="*100)

def analyze_negotiation_rounds(filename, target_day, ratio, data, daily_summary):
    """分析每个签约需要经过多少轮谈判"""
    entries = data.get('entries', [])
    
    # 收集该天每个 partner 的 offer_made 事件
    offers_by_partner = defaultdict(list)
    
    for e in entries:
        if e.get('event') == 'offer_made':
            d = e.get('data', {})
            offer = d.get('offer', {})
            if offer.get('delivery_day') == target_day:
                partner = d.get('partner', '')
                offers_by_partner[partner].append({
                    'round': offer.get('round', 0),
                    'quantity': offer.get('quantity'),
                    'price': offer.get('unit_price'),
                    'reason': d.get('reason', '')
                })
    
    # 收集签约信息
    signed_partners = set()
    for e in entries:
        if e.get('event') == 'signed':
            d = e.get('data', {})
            if d.get('delivery_day') == target_day:
                signed_partners.add(d.get('partner', ''))
    
    print(f"\n📅 Day {target_day} 谈判轮数分析")
    
    first_round_accepts = 0
    multi_round = 0
    
    for partner in sorted(offers_by_partner.keys()):
        offers = offers_by_partner[partner]
        signed = partner in signed_partners
        n_rounds = len(offers)
        
        if signed:
            if n_rounds == 1:
                first_round_accepts += 1
            else:
                multi_round += 1
        
        status = "✅ SIGNED" if signed else "❌ NOT SIGNED"
        print(f"\n      {partner:20s} {status}")
        for o in offers:
            print(f"         Round {o['round']}: qty={o['quantity']} price={o['price']} ({o['reason']})")
    
    print(f"\n   📊 统计:")
    print(f"      First round accept (对方直接接受): {first_round_accepts}")
    print(f"      Multi-round (需要多轮谈判): {multi_round}")
    
    if first_round_accepts > 0:
        print(f"\n   💡 发现: {first_round_accepts} 个 partner 在第一轮就接受了 LOS 的 offer")
        print(f"      这说明问题不在 counter_all，而是在 first_proposals 阶段!")
        print(f"      LOS 给每个 partner 都报了完整的 need 数量，而不是分散报价")

if probe_examples:
    example = probe_examples[0]
    analyze_negotiation_rounds(*example)
