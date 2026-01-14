"""
验证 Reviewer 关于 _select_subset 效用函数问题的分析

核心问题：
- utility 对每个 offer 用的是 (exo_output_price - buy_price) * q
- 这对于"满足需求"的部分是正确的
- 但对于"超过需求"的部分，这些额外单位卖不出去，实际收益是负的

示例验证:
- need = 8
- 8 个 partner 每个给 q=6 的 offer
- exo_output_price = 90, buy_price = 17
- disposal_cost = 3.3 (假设)
"""

# 模拟 _select_subset 的逻辑
need_remaining = 8
exo_output_price = 90
buy_price = 17
disposal_unit = 3.3  # disposal cost per unit

# 8 个 offer，每个 q=6
offers = [(f"partner_{i}", 6, buy_price) for i in range(8)]

# 当前代码的计算方式
print("=" * 80)
print("当前 _select_subset 的效用计算")
print("=" * 80)

def current_utility_calc(selected_offers):
    """当前代码的效用计算方式"""
    total_q = sum(q for _, q, _ in selected_offers)
    
    # 每个 offer 的 utility = (exo_output_price - buy_price) * q
    utility = sum((exo_output_price - buy_price) * q for _, q, _ in selected_offers)
    
    # 惩罚
    underfill = max(0, need_remaining - total_q)
    overfill = max(0, total_q - need_remaining)
    
    # underfill_penalty_unit = shortfall_unit (假设 33)
    # overfill_penalty_unit = disposal_unit (3.3)
    shortfall_unit = 33.0
    penalty_cost = shortfall_unit * underfill + disposal_unit * overfill
    
    score = utility - penalty_cost
    return {
        'total_q': total_q,
        'utility': utility,
        'underfill': underfill,
        'overfill': overfill,
        'penalty_cost': penalty_cost,
        'score': score
    }

def correct_utility_calc(selected_offers):
    """修正后的效用计算方式 - 边际收益"""
    total_q = sum(q for _, q, _ in selected_offers)
    
    # 有用部分（满足需求）
    q_use = min(total_q, need_remaining)
    # 额外部分（超过需求）
    q_extra = max(0, total_q - need_remaining)
    
    # 有用部分的收益 = (exo_output_price - buy_price) * q_use
    useful_profit = (exo_output_price - buy_price) * q_use
    
    # 额外部分：没有收入，只有采购成本和处理成本
    # 额外部分的"损失" = buy_price * q_extra + disposal_unit * q_extra
    # 但由于我们分别计算，这里额外部分的 utility = 0
    extra_utility = 0  # 不应该计入正收益
    
    utility = useful_profit + extra_utility
    
    # 惩罚 (保持不变)
    underfill = max(0, need_remaining - total_q)
    overfill = max(0, total_q - need_remaining)
    shortfall_unit = 33.0
    penalty_cost = shortfall_unit * underfill + disposal_unit * overfill
    
    # 额外采购成本 = buy_price * q_extra
    extra_cost = buy_price * q_extra
    
    score = utility - penalty_cost - extra_cost
    return {
        'total_q': total_q,
        'q_use': q_use,
        'q_extra': q_extra,
        'useful_profit': useful_profit,
        'extra_utility': extra_utility,
        'utility': utility,
        'underfill': underfill,
        'overfill': overfill,
        'penalty_cost': penalty_cost,
        'extra_cost': extra_cost,
        'score': score
    }

# 比较不同选择的得分
print("\n对比：选择 2 个 offer (q=12) vs 选择 7 个 offer (q=42)")
print("-" * 80)

# 选择 2 个 offer (刚好满足需求 + 一点超量)
selected_2 = offers[:2]  # 2 * 6 = 12
result_2_current = current_utility_calc(selected_2)
result_2_correct = correct_utility_calc(selected_2)

print(f"\n选择 2 个 offer (total_q = 12, need = 8):")
print(f"  当前逻辑:")
print(f"    utility = {result_2_current['utility']:.1f} (所有 12 单位都按 90-17=73 计算)")
print(f"    overfill = {result_2_current['overfill']:.1f}")
print(f"    penalty = {result_2_current['penalty_cost']:.1f}")
print(f"    score = {result_2_current['score']:.1f}")
print(f"  修正逻辑:")
print(f"    q_use = {result_2_correct['q_use']}, q_extra = {result_2_correct['q_extra']}")
print(f"    useful_profit = {result_2_correct['useful_profit']:.1f}")
print(f"    extra_cost = {result_2_correct['extra_cost']:.1f} (额外采购成本)")
print(f"    penalty = {result_2_correct['penalty_cost']:.1f}")
print(f"    score = {result_2_correct['score']:.1f}")

# 选择 7 个 offer (严重超量)
selected_7 = offers[:7]  # 7 * 6 = 42
result_7_current = current_utility_calc(selected_7)
result_7_correct = correct_utility_calc(selected_7)

print(f"\n选择 7 个 offer (total_q = 42, need = 8):")
print(f"  当前逻辑:")
print(f"    utility = {result_7_current['utility']:.1f} (所有 42 单位都按 73 计算!)")
print(f"    overfill = {result_7_current['overfill']:.1f}")
print(f"    penalty = {result_7_current['penalty_cost']:.1f} (只有 disposal)")
print(f"    score = {result_7_current['score']:.1f}")
print(f"  修正逻辑:")
print(f"    q_use = {result_7_correct['q_use']}, q_extra = {result_7_correct['q_extra']}")
print(f"    useful_profit = {result_7_correct['useful_profit']:.1f}")
print(f"    extra_cost = {result_7_correct['extra_cost']:.1f}")
print(f"    penalty = {result_7_correct['penalty_cost']:.1f}")
print(f"    score = {result_7_correct['score']:.1f}")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print(f"当前逻辑: 选 7 个 ({result_7_current['score']:.1f}) > 选 2 个 ({result_2_current['score']:.1f})")
print(f"  → 会选择 7 个 offer，导致 8 倍超量!")
print(f"修正逻辑: 选 2 个 ({result_2_correct['score']:.1f}) > 选 7 个 ({result_7_correct['score']:.1f})")
print(f"  → 会选择 2 个 offer，合理超量")

print("\n" + "=" * 80)
print("Reviewer 分析验证结果：完全正确!")
print("=" * 80)
print("""
问题根源:
  当前 utility 对每个 offer 用的是 (exo_output_price - buy_price) * q
  对于超过 need 的部分，这些额外单位卖不出去，但仍被计入正收益
  
  overfill_penalty = disposal_unit * overfill 只覆盖了处理成本
  没有覆盖:
    1. 额外采购成本 (buy_price * q_extra)
    2. 额外部分没有收入这个事实
    
修复方案:
  1. BUYER 硬上限约束: cap = ceil(need * 1.3) + 1
  2. 边际收益计算: 超过 need 的部分 utility = 0
  3. 大单 counter: 对手给的 q > cap 时不 ACCEPT，而是 COUNTER
""")
