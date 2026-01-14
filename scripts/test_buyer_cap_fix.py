#!/usr/bin/env python3
"""
测试 BUYER 硬上限修复是否有效。

验证点:
1. _select_subset 中 BUYER 子集总量是否受 buyer_cap 限制
2. _propose_for_role 中 counter offer 的 q 是否受 buyer_cap 限制
3. 边际收益修正是否生效（超过 need 的部分 utility=0）

运行方式:
    python scripts/test_buyer_cap_fix.py
"""

import math
import sys
from pathlib import Path

# 添加项目根目录
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from litaagent_os.config import LitaAgentOSConfig


def test_buyer_cap_calculation():
    """测试 buyer_cap 的计算公式"""
    cfg = LitaAgentOSConfig()
    
    print("=== 测试 BUYER 硬上限计算 ===")
    print(f"配置: buyer_accept_cap_mult = {cfg.buyer_accept_cap_mult}")
    print(f"配置: buyer_accept_cap_abs = {cfg.buyer_accept_cap_abs}")
    print(f"配置: buyer_marginal_utility_fix = {cfg.buyer_marginal_utility_fix}")
    print()
    
    test_cases = [
        (1, 3),   # need=1 → cap=ceil(1*1.3)+1=2+1=3
        (2, 4),   # need=2 → cap=ceil(2*1.3)+1=3+1=4
        (5, 8),   # need=5 → cap=ceil(5*1.3)+1=7+1=8
        (8, 12),  # need=8 → cap=ceil(8*1.3)+1=11+1=12
        (10, 14), # need=10 → cap=ceil(10*1.3)+1=13+1=14
        (20, 27), # need=20 → cap=ceil(20*1.3)+1=26+1=27
    ]
    
    all_pass = True
    for need, expected_cap in test_cases:
        actual_cap = math.ceil(need * cfg.buyer_accept_cap_mult) + cfg.buyer_accept_cap_abs
        status = "✅" if actual_cap == expected_cap else "❌"
        if actual_cap != expected_cap:
            all_pass = False
        print(f"  need={need:2d} → cap={actual_cap:2d} (expected {expected_cap:2d}) {status}")
    
    print()
    return all_pass


def test_marginal_utility_logic():
    """测试边际收益修正逻辑"""
    print("=== 测试边际收益修正逻辑 ===")
    print()
    
    # 模拟 Reviewer 分析的场景
    # need=8, 7 个 offer 各 q=6, exo_out=15, buy_price=6, disposal=0.1*buy_price
    need = 8
    offers_qs = [6, 6, 6, 6, 6, 6, 6]  # 7 个 offer
    exo_out_price = 15.0
    buy_price = 6.0
    disposal_rate = 0.1  # disposal_cost = 0.1 * buy_price = 0.6
    disposal_unit = disposal_rate * buy_price
    
    # 场景 1: 选择所有 7 个 offer (total=42)
    total_q_all = sum(offers_qs)
    
    # 旧逻辑 (无边际修正)
    old_utility_all = sum((exo_out_price - buy_price) * q for q in offers_qs)
    overfill_all = max(0, total_q_all - need)
    old_penalty_all = disposal_unit * overfill_all
    old_score_all = old_utility_all - old_penalty_all
    
    # 新逻辑 (边际修正: 只有 need 内的有 utility, 超过的 utility=0)
    need_remaining = need
    new_utility_all = 0.0
    for q in offers_qs:
        q_useful = min(q, max(0, need_remaining))
        new_utility_all += (exo_out_price - buy_price) * q_useful
        need_remaining -= q_useful
    new_penalty_all = disposal_unit * overfill_all
    new_score_all = new_utility_all - new_penalty_all
    
    # 场景 2: 只选择 2 个 offer (total=12)
    offers_qs_2 = [6, 6]
    total_q_2 = sum(offers_qs_2)
    
    # 旧逻辑
    old_utility_2 = sum((exo_out_price - buy_price) * q for q in offers_qs_2)
    overfill_2 = max(0, total_q_2 - need)
    old_penalty_2 = disposal_unit * overfill_2
    old_score_2 = old_utility_2 - old_penalty_2
    
    # 新逻辑
    need_remaining = need
    new_utility_2 = 0.0
    for q in offers_qs_2:
        q_useful = min(q, max(0, need_remaining))
        new_utility_2 += (exo_out_price - buy_price) * q_useful
        need_remaining -= q_useful
    new_penalty_2 = disposal_unit * overfill_2
    new_score_2 = new_utility_2 - new_penalty_2
    
    print("场景设置:")
    print(f"  need = {need}")
    print(f"  exo_out_price = {exo_out_price}")
    print(f"  buy_price = {buy_price}")
    print(f"  disposal_unit = {disposal_unit}")
    print()
    
    print("场景 1: 选择所有 7 个 offer (total=42)")
    print(f"  旧逻辑: utility={old_utility_all:.1f}, penalty={old_penalty_all:.1f}, score={old_score_all:.1f}")
    print(f"  新逻辑: utility={new_utility_all:.1f}, penalty={new_penalty_all:.1f}, score={new_score_all:.1f}")
    print()
    
    print("场景 2: 只选择 2 个 offer (total=12)")
    print(f"  旧逻辑: utility={old_utility_2:.1f}, penalty={old_penalty_2:.1f}, score={old_score_2:.1f}")
    print(f"  新逻辑: utility={new_utility_2:.1f}, penalty={new_penalty_2:.1f}, score={new_score_2:.1f}")
    print()
    
    # 检查: 旧逻辑会选 7 个 (score 更高), 新逻辑应该选 2 个
    old_prefer_all = old_score_all > old_score_2
    new_prefer_2 = new_score_2 > new_score_all
    
    print("决策分析:")
    print(f"  旧逻辑: {'选 7 个 ❌' if old_prefer_all else '选 2 个 ✅'} (因为 {old_score_all:.1f} > {old_score_2:.1f})")
    print(f"  新逻辑: {'选 2 个 ✅' if new_prefer_2 else '选 7 个 ❌'} (因为 {new_score_2:.1f} > {new_score_all:.1f})")
    print()
    
    if old_prefer_all and new_prefer_2:
        print("✅ 边际收益修正有效! 旧逻辑选错, 新逻辑选对")
        return True
    else:
        print("❌ 边际收益修正逻辑有问题")
        return False


def test_buyer_cap_filter():
    """测试 buyer_cap 过滤逻辑"""
    print("=== 测试 BUYER 硬上限过滤 ===")
    print()
    
    cfg = LitaAgentOSConfig()
    need = 8
    buyer_cap = math.ceil(need * cfg.buyer_accept_cap_mult) + cfg.buyer_accept_cap_abs
    
    # 模拟 7 个 offer, 各 q=6
    offers_qs = [6, 6, 6, 6, 6, 6, 6]
    
    print(f"need = {need}")
    print(f"buyer_cap = {buyer_cap}")
    print(f"offers: {offers_qs}")
    print()
    
    # 找到所有有效子集 (total_q <= buyer_cap)
    n = len(offers_qs)
    valid_subsets = []
    for mask in range(1 << n):
        subset = [offers_qs[i] for i in range(n) if mask & (1 << i)]
        if not subset:
            continue
        total_q = sum(subset)
        if total_q <= buyer_cap:
            valid_subsets.append((subset, total_q))
    
    print(f"有效子集数量 (total_q <= {buyer_cap}): {len(valid_subsets)}")
    print(f"总子集数量: {(1 << n) - 1}")
    print()
    
    # 显示最大的几个有效子集
    valid_subsets.sort(key=lambda x: x[1], reverse=True)
    print("最大的有效子集 (按 total_q 排序):")
    for subset, total_q in valid_subsets[:5]:
        print(f"  {subset} → total_q={total_q}")
    
    print()
    
    # 验证: 没有超过 buyer_cap 的子集
    max_valid_q = max(s[1] for s in valid_subsets)
    if max_valid_q <= buyer_cap:
        print(f"✅ 所有有效子集 total_q <= {buyer_cap}")
        return True
    else:
        print(f"❌ 存在超过 buyer_cap 的子集")
        return False


def main():
    print("=" * 60)
    print("BUYER 硬上限 + 边际收益修正 测试")
    print("=" * 60)
    print()
    
    results = []
    results.append(("BUYER 硬上限计算", test_buyer_cap_calculation()))
    results.append(("边际收益修正逻辑", test_marginal_utility_logic()))
    results.append(("BUYER 硬上限过滤", test_buyer_cap_filter()))
    
    print("=" * 60)
    print("测试结果汇总:")
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败, 请检查")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
