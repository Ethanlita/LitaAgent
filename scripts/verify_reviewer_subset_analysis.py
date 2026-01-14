#!/usr/bin/env python
"""
验证 Reviewer 对 BUYER _select_subset 的两个核心分析：

1. buyer_marginal_utility_fix 只对超买部分的收入抹零，但成本也被抹掉了
   - 原问题：profit_per_unit = util / q，其中 util = (exo_out_price - buy_price) * q
   - 对超买部分：只计入 q_useful 的 profit，超出部分 profit=0
   - 但这意味着超出部分的 buy_price 成本也被抹掉了！
   - 实际上应该是：超出部分 revenue=0，但 cost=buy_price*q_extra 仍然存在

2. int(need_remaining_eff) 会把 0.8 截断为 0，导致结构性 shortfall

让我们用具体数字来验证。
"""

import math


def simulate_current_implementation(
    offers: list[tuple[int, int, float]],  # (q, price, p_eff)
    need: int,
    exo_out_price: int,
    underfill_penalty_unit: float,
    overfill_penalty_unit: float,
) -> dict:
    """
    模拟当前的 _select_subset BUYER 逻辑
    offers: [(q, buy_price, p_eff), ...]
    """
    # 计算每个 offer 的 utility（当前实现）
    # util = (exo_out_price - buy_price) * q
    selected_offers = []
    for q, buy_price, p_eff in offers:
        util = (exo_out_price - buy_price) * q
        selected_offers.append((q, buy_price, p_eff, util))
    
    # buyer_marginal_utility_fix 逻辑
    utility = 0.0
    cumulative_q = 0.0
    need_remaining = need
    
    for q, buy_price, p_eff, util in selected_offers:
        if q > 0:
            profit_per_unit = util / q  # = exo_out_price - buy_price
        else:
            profit_per_unit = 0.0
        
        # 有用的数量
        q_useful = min(q, max(0, need_remaining - cumulative_q))
        cumulative_q += q
        
        # 只对有用部分计入正收益
        utility += profit_per_unit * q_useful
    
    # 总 q_eff
    total_q = sum(q for q, _, _, _ in selected_offers)
    total_q_eff = sum(q * p_eff for q, _, p_eff, _ in selected_offers)
    
    # penalty
    underfill = max(0.0, need - total_q_eff)
    overfill = max(0.0, total_q_eff - need)
    penalty_cost = underfill_penalty_unit * underfill + overfill_penalty_unit * overfill
    
    score = utility - penalty_cost
    
    return {
        "utility": utility,
        "penalty_cost": penalty_cost,
        "score": score,
        "total_q": total_q,
        "total_q_eff": total_q_eff,
        "underfill": underfill,
        "overfill": overfill,
    }


def simulate_reviewer_proposed(
    offers: list[tuple[int, int, float]],
    need: int,
    exo_out_price: int,
    underfill_penalty_unit: float,
    overfill_penalty_unit: float,
) -> dict:
    """
    Reviewer 建议的实现：
    Score = R - C - P
    R = exo_out_price * min(Q_eff, N)  # 收入只来自满足需求的部分
    C = Σ q_i * p_i                     # 采购成本（签了就算）
    P = underfill_penalty + overfill_penalty
    """
    total_q = sum(q for q, _, _ in offers)
    total_q_eff = sum(q * p_eff for q, _, p_eff in offers)
    total_cost = sum(q * buy_price for q, buy_price, _ in offers)
    
    # Revenue 只来自满足需求的部分
    revenue = exo_out_price * min(total_q_eff, need)
    
    # penalty
    underfill = max(0.0, need - total_q_eff)
    overfill = max(0.0, total_q_eff - need)
    penalty_cost = underfill_penalty_unit * underfill + overfill_penalty_unit * overfill
    
    score = revenue - total_cost - penalty_cost
    
    return {
        "revenue": revenue,
        "cost": total_cost,
        "penalty_cost": penalty_cost,
        "score": score,
        "total_q": total_q,
        "total_q_eff": total_q_eff,
        "underfill": underfill,
        "overfill": overfill,
    }


def main():
    print("=" * 80)
    print("验证 Reviewer 分析：BUYER _select_subset 的两个核心问题")
    print("=" * 80)
    
    # 场景参数
    need = 10
    exo_out_price = 20  # 卖给下游的价格
    underfill_penalty = 2.0
    overfill_penalty = 1.0  # disposal cost 较小
    
    print(f"\n场景参数: need={need}, exo_out_price={exo_out_price}")
    print(f"         underfill_penalty={underfill_penalty}, overfill_penalty={overfill_penalty}")
    
    # 场景 1：刚好满足需求（Exact）
    print("\n" + "-" * 80)
    print("场景 1：Exact (q=10)")
    offers_exact = [(10, 15, 0.9)]  # q=10, buy_price=15, p_eff=0.9
    current = simulate_current_implementation(offers_exact, need, exo_out_price, underfill_penalty, overfill_penalty)
    proposed = simulate_reviewer_proposed(offers_exact, need, exo_out_price, underfill_penalty, overfill_penalty)
    print(f"当前实现:    utility={current['utility']:.1f}, penalty={current['penalty_cost']:.1f}, score={current['score']:.1f}")
    print(f"Reviewer建议: revenue={proposed['revenue']:.1f}, cost={proposed['cost']:.1f}, penalty={proposed['penalty_cost']:.1f}, score={proposed['score']:.1f}")
    
    # 场景 2：超买（Overfull）- 核心测试
    print("\n" + "-" * 80)
    print("场景 2：Overfull (q=15, 超买 5)")
    offers_over = [(15, 15, 0.9)]  # q=15, buy_price=15, p_eff=0.9 → q_eff=13.5
    current = simulate_current_implementation(offers_over, need, exo_out_price, underfill_penalty, overfill_penalty)
    proposed = simulate_reviewer_proposed(offers_over, need, exo_out_price, underfill_penalty, overfill_penalty)
    
    print(f"q=15, q_eff={15*0.9:.1f}, need=10")
    print(f"当前实现:    utility={current['utility']:.1f} (只计入10个的利润)")
    print(f"           penalty={current['penalty_cost']:.1f} (overfill={current['overfill']:.1f})")
    print(f"           score={current['score']:.1f}")
    print(f"           ⚠️ 问题：超买的5个单位，成本 15*5=75 被忽略了！")
    print()
    print(f"Reviewer建议: revenue={proposed['revenue']:.1f} (只有10个的收入)")
    print(f"             cost={proposed['cost']:.1f} (15个全部的采购成本)")
    print(f"             penalty={proposed['penalty_cost']:.1f}")
    print(f"             score={proposed['score']:.1f}")
    print(f"             ✓ 超买的5个单位成本被正确计入")
    
    # 场景 3：对比 Exact vs Overfull 的决策
    print("\n" + "-" * 80)
    print("场景 3：决策对比 - Exact(q=10) vs Overfull(q=15)")
    print("        哪个 subset 会被选择？")
    
    offers_exact = [(10, 15, 0.9)]
    offers_over = [(15, 15, 0.9)]
    
    current_exact = simulate_current_implementation(offers_exact, need, exo_out_price, underfill_penalty, overfill_penalty)
    current_over = simulate_current_implementation(offers_over, need, exo_out_price, underfill_penalty, overfill_penalty)
    
    proposed_exact = simulate_reviewer_proposed(offers_exact, need, exo_out_price, underfill_penalty, overfill_penalty)
    proposed_over = simulate_reviewer_proposed(offers_over, need, exo_out_price, underfill_penalty, overfill_penalty)
    
    print(f"\n当前实现:")
    print(f"  Exact(q=10):    score={current_exact['score']:.1f}")
    print(f"  Overfull(q=15): score={current_over['score']:.1f}")
    print(f"  决策: 选择 {'Exact' if current_exact['score'] > current_over['score'] else 'Overfull'} ← ", end="")
    if current_over['score'] > current_exact['score']:
        print("❌ 错误！超买更优是因为成本被忽略")
    else:
        print("✓ 正确")
    
    print(f"\nReviewer建议:")
    print(f"  Exact(q=10):    score={proposed_exact['score']:.1f}")
    print(f"  Overfull(q=15): score={proposed_over['score']:.1f}")
    print(f"  决策: 选择 {'Exact' if proposed_exact['score'] > proposed_over['score'] else 'Overfull'} ← ", end="")
    if proposed_exact['score'] > proposed_over['score']:
        print("✓ 正确！Exact 更优")
    else:
        print("需要调整参数")
    
    # 问题 2：int() 截断问题
    print("\n" + "=" * 80)
    print("问题 2：int(need_remaining_eff) 截断问题")
    print("=" * 80)
    
    need_remaining_eff_values = [0.1, 0.5, 0.8, 0.99, 1.0, 1.5]
    print("\nneed_remaining_eff → int() → ceil()")
    for v in need_remaining_eff_values:
        int_v = int(v)
        ceil_v = math.ceil(v - 0.001)  # 小 epsilon 避免浮点误差
        print(f"  {v:.2f} → int={int_v} → ceil={ceil_v}", end="")
        if int_v == 0 and ceil_v > 0:
            print(" ← ⚠️ int截断会导致不必要的END!")
        elif int_v < ceil_v:
            print(" ← 少1个单位的修正机会")
        else:
            print()
    
    print("\n结论：当 need_remaining_eff 在 (0, 1) 区间时，int() 会返回 0，")
    print("      导致立即 END 谈判，失去获取最后 1 个单位的机会 → 结构性 shortfall")


if __name__ == "__main__":
    main()
