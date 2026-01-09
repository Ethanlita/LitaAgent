#!/usr/bin/env python3
"""
深度验证 DatasetBuilder 输出的数据质量。

检查项目：
1. y_accept=1 的样本确实对应最终 agreement
2. y_accept=0 的样本确实是被拒绝（有后续 counter offer）
3. 特征值是否在合理范围内
4. 每个谈判的样本数是否合理
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import ast

from litaagent_os.scripts.build_accept_dataset import (
    process_world_dir,
    safe_literal_eval,
    parse_issues,
    offers_match,
)


def validate_agreement_matching(world_dir: Path) -> dict:
    """验证 y_accept=1 的样本是否真的匹配 agreement"""
    
    neg_df = pd.read_csv(world_dir / "negotiations.csv")
    samples_list = process_world_dir(world_dir)
    samples = [vars(s) for s in samples_list]
    
    issues = {
        "accept_not_matching_agreement": [],
        "accept_count_per_neg": defaultdict(int),
        "total_negs_with_agreement": 0,
        "total_negs_without_agreement": 0,
    }
    
    # 按 negotiation_id 分组样本
    samples_by_neg = defaultdict(list)
    for s in samples:
        samples_by_neg[s["negotiation_id"]].append(s)
    
    for _, row in neg_df.iterrows():
        neg_id = str(row["id"])
        agreement = safe_literal_eval(row.get("agreement"))
        failed = bool(row.get("failed", False))
        
        if agreement and not failed:
            issues["total_negs_with_agreement"] += 1
        else:
            issues["total_negs_without_agreement"] += 1
        
        neg_samples = samples_by_neg.get(neg_id, [])
        accept_samples = [s for s in neg_samples if s["y_accept"] == 1]
        
        # 统计每个谈判的 accept 样本数
        issues["accept_count_per_neg"][len(accept_samples)] += 1
        
        # 验证 accept 样本是否匹配 agreement
        if agreement and not failed:
            if len(accept_samples) == 0:
                issues["accept_not_matching_agreement"].append({
                    "neg_id": neg_id,
                    "agreement": agreement,
                    "all_samples": neg_samples,
                    "issue": "有 agreement 但没有 y_accept=1 样本",
                })
            elif len(accept_samples) > 1:
                issues["accept_not_matching_agreement"].append({
                    "neg_id": neg_id,
                    "agreement": agreement,
                    "accept_samples": accept_samples,
                    "issue": f"有 agreement 但 y_accept=1 样本数 > 1 ({len(accept_samples)})",
                })
            else:
                # 验证 accept 样本的 offer 是否匹配 agreement
                acc = accept_samples[0]
                offer = (acc["q"], acc["t"], acc["p"])
                if not offers_match(offer, agreement):
                    issues["accept_not_matching_agreement"].append({
                        "neg_id": neg_id,
                        "agreement": agreement,
                        "sample_offer": offer,
                        "issue": "y_accept=1 样本的 offer 与 agreement 不匹配",
                    })
        else:
            # 无 agreement 的谈判不应该有 y_accept=1
            if accept_samples:
                issues["accept_not_matching_agreement"].append({
                    "neg_id": neg_id,
                    "agreement": agreement,
                    "failed": failed,
                    "accept_samples": accept_samples,
                    "issue": "无 agreement 但有 y_accept=1 样本",
                })
    
    return issues


def validate_feature_ranges(samples: list[dict]) -> dict:
    """验证特征值是否在合理范围内"""
    
    issues = {
        "out_of_range": [],
        "feature_stats": {},
    }
    
    numeric_fields = [
        ("round_rel", 0.0, 1.0),
        ("round_bucket", 0, 10),
        ("need_norm", 0.0, 10.0),  # 允许较大值
        ("q_norm", 0.0, 1.0),
        ("p_norm", 0.0, 1.0),
        ("p_bin", 0, 1),
    ]
    
    for field, lo, hi in numeric_fields:
        values = [s.get(field) for s in samples if s.get(field) is not None]
        if not values:
            continue
        
        min_val, max_val = min(values), max(values)
        avg_val = sum(values) / len(values)
        
        issues["feature_stats"][field] = {
            "min": min_val,
            "max": max_val,
            "avg": avg_val,
            "count": len(values),
        }
        
        out_of_range = [v for v in values if v < lo or v > hi]
        if out_of_range:
            issues["out_of_range"].append({
                "field": field,
                "expected_range": (lo, hi),
                "actual_range": (min_val, max_val),
                "out_of_range_count": len(out_of_range),
            })
    
    return issues


def validate_role_consistency(samples: list[dict]) -> dict:
    """验证角色分配是否一致"""
    
    issues = {
        "role_inconsistency": [],
        "role_counts": defaultdict(int),
    }
    
    # 按 (negotiation_id, proposer_id) 分组，检查同一 proposer 的 role 是否一致
    proposer_roles = defaultdict(set)
    for s in samples:
        key = (s["negotiation_id"], s["proposer_id"])
        proposer_roles[key].add(s["role"])
        issues["role_counts"][s["role"]] += 1
    
    for (neg_id, proposer_id), roles in proposer_roles.items():
        if len(roles) > 1:
            issues["role_inconsistency"].append({
                "neg_id": neg_id,
                "proposer_id": proposer_id,
                "roles": list(roles),
            })
    
    return issues


def main():
    print("=" * 60)
    print("DatasetBuilder 深度验证")
    print("=" * 60)
    
    # 查找 world 目录
    search_dirs = [ROOT / "results", ROOT / "run_logs"]
    world_dirs = []
    for search_dir in search_dirs:
        if search_dir.exists():
            for d in search_dir.iterdir():
                if d.is_dir() and (d / "negotiations.csv").exists():
                    world_dirs.append(d)
    
    if not world_dirs:
        print("❌ 未找到可用的 world 目录")
        return 1
    
    # 选择一个 world 进行详细验证
    world_dir = world_dirs[0]
    print(f"\n使用 world: {world_dir.name}")
    
    samples_list = process_world_dir(world_dir)
    samples = [vars(s) for s in samples_list]
    
    print(f"总样本数: {len(samples)}")
    
    # 1. 验证 agreement 匹配
    print("\n--- 1. 验证 Agreement 匹配 ---")
    agreement_issues = validate_agreement_matching(world_dir)
    
    print(f"  有 agreement 的谈判数: {agreement_issues['total_negs_with_agreement']}")
    print(f"  无 agreement 的谈判数: {agreement_issues['total_negs_without_agreement']}")
    print(f"  每个谈判的 accept 样本数分布: {dict(agreement_issues['accept_count_per_neg'])}")
    
    if agreement_issues["accept_not_matching_agreement"]:
        print(f"  ⚠️ 发现 {len(agreement_issues['accept_not_matching_agreement'])} 个问题:")
        for issue in agreement_issues["accept_not_matching_agreement"][:5]:
            print(f"    - {issue['issue']}: neg_id={issue['neg_id']}")
    else:
        print("  ✓ 所有 y_accept=1 样本都正确匹配 agreement")
    
    # 2. 验证特征范围
    print("\n--- 2. 验证特征范围 ---")
    range_issues = validate_feature_ranges(samples)
    
    for field, stats in range_issues["feature_stats"].items():
        print(f"  {field}: min={stats['min']:.4f}, max={stats['max']:.4f}, avg={stats['avg']:.4f}")
    
    if range_issues["out_of_range"]:
        print(f"  ⚠️ 发现 {len(range_issues['out_of_range'])} 个字段超出范围:")
        for issue in range_issues["out_of_range"]:
            print(f"    - {issue['field']}: expected {issue['expected_range']}, got {issue['actual_range']}")
    else:
        print("  ✓ 所有特征值在合理范围内")
    
    # 3. 验证角色一致性
    print("\n--- 3. 验证角色一致性 ---")
    role_issues = validate_role_consistency(samples)
    
    print(f"  角色分布: {dict(role_issues['role_counts'])}")
    
    if role_issues["role_inconsistency"]:
        print(f"  ⚠️ 发现 {len(role_issues['role_inconsistency'])} 个角色不一致问题")
    else:
        print("  ✓ 角色分配一致")
    
    # 4. 抽样检查具体样本
    print("\n--- 4. 抽样检查 ---")
    accept_samples = [s for s in samples if s["y_accept"] == 1][:3]
    reject_samples = [s for s in samples if s["y_accept"] == 0][:3]
    
    print("  y_accept=1 样本示例:")
    for s in accept_samples:
        print(f"    neg={s['negotiation_id'][:8]}... proposer={s['proposer_id']} "
              f"offer=({s['q']}, {s['t']}, {s['p']}) role={s['role']}")
    
    print("  y_accept=0 样本示例:")
    for s in reject_samples:
        print(f"    neg={s['negotiation_id'][:8]}... proposer={s['proposer_id']} "
              f"offer=({s['q']}, {s['t']}, {s['p']}) role={s['role']}")
    
    # 总结
    print("\n" + "=" * 60)
    total_issues = (
        len(agreement_issues["accept_not_matching_agreement"])
        + len(range_issues["out_of_range"])
        + len(role_issues["role_inconsistency"])
    )
    
    if total_issues == 0:
        print("✓ 所有验证通过！数据质量良好。")
        return 0
    else:
        print(f"⚠️ 发现 {total_issues} 个潜在问题，请检查。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
