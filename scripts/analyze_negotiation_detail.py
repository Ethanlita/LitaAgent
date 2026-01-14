"""分析 LOS BUYER 的详细谈判记录"""
import json
from pathlib import Path
from collections import defaultdict, Counter

log_dir = Path('tournament_history/20260110_212745_oneshot/tracker_logs')
los_files = sorted(log_dir.glob('agent_*LOS*.json'))

print(f'找到 {len(los_files)} 个 LOS 日志文件')

# 分析所有 BUYER 的统计模式
buyer_stats = {
    'total_days': 0,
    'multi_contract_days': 0,
    'single_contract_days': 0,
    'contracts_per_day': [],
    'qty_ratio': [],
    'multi_partner_days': 0,  # 不同 partner 签约的天数
    'same_partner_multi': 0,  # 同一个 partner 多份合同
}

for f in los_files:
    with open(f) as fp:
        data = json.load(fp)
    
    level = None
    for e in data.get('entries', []):
        if e.get('event') == 'agent_initialized':
            level = e.get('data', {}).get('level')
            break
    
    if level != 1:
        continue
    
    entries = data.get('entries', [])
    daily_data = {}
    
    for e in entries:
        event = e.get('event')
        d = e.get('data', {})
        
        if event == 'daily_status':
            day = d.get('current_step')
            if day not in daily_data:
                daily_data[day] = {'exo_output': 0, 'signed_qty': 0, 'num_contracts': 0, 'partners': []}
            daily_data[day]['exo_output'] = d.get('exo_output_qty', 0)
        
        elif event == 'signed':
            day = d.get('delivery_day')
            qty = d.get('quantity', 0)
            partner = d.get('partner', '')
            if day not in daily_data:
                daily_data[day] = {'exo_output': 0, 'signed_qty': 0, 'num_contracts': 0, 'partners': []}
            daily_data[day]['signed_qty'] += qty
            daily_data[day]['num_contracts'] += 1
            daily_data[day]['partners'].append(partner)
    
    for day, dd in daily_data.items():
        if dd['exo_output'] > 0:
            buyer_stats['total_days'] += 1
            buyer_stats['contracts_per_day'].append(dd['num_contracts'])
            buyer_stats['qty_ratio'].append(dd['signed_qty'] / dd['exo_output'])
            
            unique_partners = set(dd['partners'])
            if len(unique_partners) > 1:
                buyer_stats['multi_partner_days'] += 1
            if len(dd['partners']) > len(unique_partners):
                buyer_stats['same_partner_multi'] += 1
            
            if dd['num_contracts'] > 1:
                buyer_stats['multi_contract_days'] += 1
            else:
                buyer_stats['single_contract_days'] += 1

print(f"\n{'='*80}")
print("📊 LOS BUYER 合同签署模式分析")
print(f"{'='*80}")

print(f"\n总天数: {buyer_stats['total_days']}")
print(f"单合同天数: {buyer_stats['single_contract_days']} ({100*buyer_stats['single_contract_days']/buyer_stats['total_days']:.1f}%)")
print(f"多合同天数: {buyer_stats['multi_contract_days']} ({100*buyer_stats['multi_contract_days']/buyer_stats['total_days']:.1f}%)")
print(f"  ↳ 来自多个不同 partner: {buyer_stats['multi_partner_days']} 天")
print(f"  ↳ 同一个 partner 多份合同: {buyer_stats['same_partner_multi']} 天")

# 合同数量分布
contract_dist = Counter(buyer_stats['contracts_per_day'])
print(f"\n每天合同数量分布:")
for n, count in sorted(contract_dist.items()):
    print(f"  {n} 个合同: {count} 天 ({100*count/buyer_stats['total_days']:.1f}%)")

# 购买量/需求量 比率分布
print(f"\n购买量/需求量 比率分布:")
ratios = buyer_stats['qty_ratio']
bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.01), (1.01, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 10.0)]
for lo, hi in bins:
    count = sum(1 for r in ratios if lo <= r < hi)
    pct = 100 * count / len(ratios)
    label = f"{lo:.1f}-{hi:.1f}"
    bar = '█' * int(pct / 2)
    print(f"  {label:>8}: {count:4d} ({pct:5.1f}%) {bar}")

avg_ratio = sum(ratios) / len(ratios)
print(f"\n平均购买量/需求量比率: {avg_ratio:.2f}x")

print(f"\n{'='*80}")
print("🔍 关键发现与根本原因分析")
print(f"{'='*80}")

print("""
📌 核心问题: LOS BUYER 每天签约 2.34x 需求量

📊 数据支持:
   - 93.4% 的天数签了 2 个以上的合同
   - 购买量超过需求量 2 倍以上的天数: {:.1f}%
   - 平均购买量/需求量比率: {:.2f}x

🔍 根本原因分析:

   1. 【Multiple Accept 问题】
      - LOS 发送 first_proposal 给多个 partner
      - 每个 partner 同时回复 Accept
      - LOS 在 counter_all 中使用 _select_subset 选择子集
      - 但如果多个 partner 都是"好价格"，会全部接受
      
   2. 【Overordering 策略】
      - buyer_overordering_ratio = 0.1 (10%)
      - overordering_ensure_plus_one = True (确保至少+1)
      - 设计目的: 宁可多买也不要 shortfall
      - 副作用: 导致 target > need，触发更多接受
      
   3. 【Probe 阶段分散报价】
      - probe 阶段 (前 10 天) 给每个 partner 都发报价
      - post_probe_min_partners = 3 (至少给 3 个 partner 发单)
      - 当 3 个 partner 同时 Accept → 购买量 = 3x 需求量
      
   4. 【Subset Selection 不够严格】
      - _select_subset 选择最优子集接受
      - 但在 BOU 数据不足时，可能高估 p_eff
      - 导致选择过多 partner 来接受

🎯 理论目标 vs 实际表现:
   - 理论: target = need * 1.1 (超量 10%)
   - 实际: bought = need * 2.34 (超量 134%)
   
   这表明 subset selection 未能控制总接受量！
""".format(
    100 * sum(1 for r in ratios if r >= 2.0) / len(ratios),
    avg_ratio
))

# 进一步分析: 看看 counter_all 的子集选择是否有问题
print(f"\n{'='*80}")
print("🔬 详细分析: 每天签约的 partner 数量")
print(f"{'='*80}")

# 统计每天有多少个独立 partner
partner_per_day = defaultdict(list)
for f in los_files:
    with open(f) as fp:
        data = json.load(fp)
    
    level = None
    for e in data.get('entries', []):
        if e.get('event') == 'agent_initialized':
            level = e.get('data', {}).get('level')
            break
    
    if level != 1:
        continue
    
    entries = data.get('entries', [])
    daily_partners = defaultdict(set)
    
    for e in entries:
        if e.get('event') == 'signed':
            d = e.get('data', {})
            day = d.get('delivery_day')
            partner = d.get('partner', '')
            daily_partners[day].add(partner)
    
    for day, partners in daily_partners.items():
        partner_per_day[len(partners)].append(day)

print(f"\n每天独立签约 partner 数量分布:")
total_counted = sum(len(v) for v in partner_per_day.values())
for n_partners in sorted(partner_per_day.keys()):
    count = len(partner_per_day[n_partners])
    pct = 100 * count / total_counted
    bar = '█' * int(pct / 2)
    print(f"  {n_partners} 个 partner: {count:4d} 天 ({pct:5.1f}%) {bar}")

print(f"\n{'='*80}")
print("💡 解决方案建议")
print(f"{'='*80}")

print("""
根据以上分析，LOS BUYER 超量购买的根本原因是:

   📍 问题定位: _select_subset 的组合优化逻辑

   当多个 partner 的 offer 都有正 utility 时:
   - utility 累加 (每多选一个 offer → utility ↑)
   - penalty_cost 只有在超量时才出现
   - 但 disposal_unit << shortfall_unit (约 1/10)
   
   => 优化器倾向于 "宁可多买 10 个，也不要少买 1 个"
   => 结果是选择几乎所有 offer

🔧 建议修复方案:

   1. 【方案 A: 强制限制接受数量】
      在 _select_subset 中加入硬约束:
      total_q <= need_remaining × max_overfill_ratio
      
   2. 【方案 B: 增加超量惩罚】
      目前 overfill_penalty = disposal_unit × overfill
      可改为 overfill_penalty = disposal_unit × overfill × overfill_multiplier
      让超量惩罚与短缺惩罚更均衡
      
   3. 【方案 C: 使用贪心而非穷举】
      当前: 穷举 2^n 个子集
      改为: 贪心选择，直到 q_eff >= need × (1 + buffer)
      
   4. 【方案 D: 考虑边际收益递减】
      第 i 个 offer 的边际贡献 = utility_i - marginal_penalty
      当 sum(q) > need 时，边际贡献急剧下降
""")
