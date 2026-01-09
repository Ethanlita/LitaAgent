#!/usr/bin/env python3
"""
测试 build_accept_dataset.py 的正确性

验证：
1. 样本提取是否正确
2. y_accept 标签是否正确
3. 特征是否合理
"""

import sys
from pathlib import Path

# 添加 litaagent_os 到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from litaagent_os.scripts.build_accept_dataset import (
    safe_literal_eval,
    parse_issues,
    offers_match,
    extract_samples_from_negotiation,
    process_world_dir,
)


def test_safe_literal_eval():
    """测试 safe_literal_eval"""
    print("Testing safe_literal_eval...")
    
    # 基本测试
    assert safe_literal_eval("{'a': 1}") == {'a': 1}
    assert safe_literal_eval("[(1, 2, 3)]") == [(1, 2, 3)]
    assert safe_literal_eval(None, default=[]) == []
    assert safe_literal_eval("invalid{", default="fallback") == "fallback"
    
    print("  ✓ safe_literal_eval passed")


def test_parse_issues():
    """测试 parse_issues"""
    print("Testing parse_issues...")
    
    issues_str = "['quantity: (1, 10)', 'time: (0, 0)', 'unit_price: (15, 16)']"
    parsed = parse_issues(issues_str)
    
    assert "quantity" in parsed
    assert "time" in parsed
    assert "unit_price" in parsed
    
    assert parsed["quantity"] == (1, 10)
    assert parsed["time"] == (0, 0)
    assert parsed["unit_price"] == (15, 16)
    
    print("  ✓ parse_issues passed")


def test_offers_match():
    """测试 offers_match"""
    print("Testing offers_match...")
    
    # 完全匹配
    assert offers_match((3, 0, 27), (3, 0, 27)) == True
    
    # 浮点容忍
    assert offers_match((3, 0, 27.0), (3, 0, 27.001)) == True
    
    # 不匹配
    assert offers_match((3, 0, 27), (3, 0, 28)) == False
    assert offers_match((2, 0, 27), (3, 0, 27)) == False
    assert offers_match((3, 1, 27), (3, 0, 27)) == False
    
    # None 处理
    assert offers_match(None, (3, 0, 27)) == False
    assert offers_match((3, 0, 27), None) == False
    
    print("  ✓ offers_match passed")


def test_extract_samples_basic():
    """测试基本的样本提取"""
    print("Testing extract_samples_from_negotiation (basic)...")
    
    # 构造一个简单的谈判记录
    neg_row = pd.Series({
        "id": "test-neg-001",
        "partners": "['AgentA@0', 'AgentB@1']",
        "buyer": "AgentA@0",
        "seller": "AgentB@1",
        "failed": False,
        "agreement": "(3, 0, 27)",
        "offers": "{'AgentA@0': [(1, 0, 26)], 'AgentB@1': [(3, 0, 27)]}",
        "history": """[
            {"new_offers": [("AgentA@0", (1, 0, 26))], "relative_time": 0.2},
            {"new_offers": [("AgentB@1", (3, 0, 27))], "relative_time": 0.4, "agreement": (3, 0, 27)}
        ]""",
        "issues": "['quantity: (1, 10)', 'time: (0, 0)', 'unit_price: (20, 30)']",
        "sim_step": 5,
    })
    
    negs_row = pd.Series({
        "name": "test-neg-001",
        "trading_price": 25.0,
        "agent_time0": "AgentA@0",
        "agent_time1": "AgentB@1",
        "needed_supplies0": 5,
        "needed_sales1": 5,
    })
    
    samples = extract_samples_from_negotiation(neg_row, negs_row, "test-world")
    
    print(f"  Extracted {len(samples)} samples")
    for s in samples:
        print(f"    - {s.proposer_id}: ({s.q}, {s.t}, {s.p}) y_accept={s.y_accept}")
    
    # 验证
    assert len(samples) == 2, f"Expected 2 samples, got {len(samples)}"
    
    # 第一个 offer (AgentA@0: 1,0,26) 应该 y_accept=0（因为有后续 counter）
    sample_a = [s for s in samples if s.proposer_id == "AgentA@0"][0]
    assert sample_a.y_accept == 0, f"Expected y_accept=0 for counter, got {sample_a.y_accept}"
    assert sample_a.q == 1
    assert sample_a.p == 26
    
    # 第二个 offer (AgentB@1: 3,0,27) 应该 y_accept=1（与 agreement 匹配）
    sample_b = [s for s in samples if s.proposer_id == "AgentB@1"][0]
    assert sample_b.y_accept == 1, f"Expected y_accept=1 for accepted, got {sample_b.y_accept}"
    assert sample_b.q == 3
    assert sample_b.p == 27
    
    print("  ✓ extract_samples_from_negotiation (basic) passed")


def test_extract_samples_failed_negotiation():
    """测试失败谈判的样本提取"""
    print("Testing extract_samples_from_negotiation (failed)...")
    
    # 构造一个失败的谈判（双方来回 counter，最后没有达成协议）
    neg_row = pd.Series({
        "id": "test-neg-002",
        "partners": "['AgentA@0', 'AgentB@1']",
        "buyer": "AgentA@0",
        "seller": "AgentB@1",
        "failed": True,
        "agreement": "None",
        "offers": "{'AgentA@0': [(1, 0, 20), (2, 0, 22)], 'AgentB@1': [(1, 0, 25)]}",
        "history": """[
            {"new_offers": [("AgentA@0", (1, 0, 20))], "relative_time": 0.2},
            {"new_offers": [("AgentB@1", (1, 0, 25))], "relative_time": 0.4},
            {"new_offers": [("AgentA@0", (2, 0, 22))], "relative_time": 0.6}
        ]""",
        "issues": "['quantity: (1, 10)', 'time: (0, 0)', 'unit_price: (15, 30)']",
        "sim_step": 3,
    })
    
    samples = extract_samples_from_negotiation(neg_row, None, "test-world")
    
    print(f"  Extracted {len(samples)} samples")
    for s in samples:
        print(f"    - {s.proposer_id}: ({s.q}, {s.t}, {s.p}) y_accept={s.y_accept}")
    
    # 验证：
    # - AgentA@0 第一个 offer (1,0,20)：有后续 counter -> y_accept=0
    # - AgentB@1 offer (1,0,25)：有后续 counter -> y_accept=0
    # - AgentA@0 第二个 offer (2,0,22)：最后一个，没有 agreement -> 不生成样本
    
    assert len(samples) == 2, f"Expected 2 samples, got {len(samples)}"
    
    for s in samples:
        assert s.y_accept == 0, f"Expected all y_accept=0 for countered offers"
    
    print("  ✓ extract_samples_from_negotiation (failed) passed")


def test_real_data():
    """测试真实数据（如果可用）"""
    print("Testing with real data...")
    
    # 查找第一个可用的 world 目录
    run_logs = Path(__file__).parent.parent.parent / "run_logs"
    if not run_logs.exists():
        print("  [SKIP] run_logs directory not found")
        return
    
    world_dirs = list(run_logs.rglob("negotiations.csv"))
    if not world_dirs:
        print("  [SKIP] No negotiations.csv found")
        return
    
    world_dir = world_dirs[0].parent
    print(f"  Using world: {world_dir.name}")
    
    samples = process_world_dir(world_dir)
    print(f"  Extracted {len(samples)} samples")
    
    if samples:
        # 统计
        accept_count = sum(1 for s in samples if s.y_accept == 1)
        reject_count = sum(1 for s in samples if s.y_accept == 0)
        print(f"  y_accept=1: {accept_count} ({accept_count/len(samples)*100:.1f}%)")
        print(f"  y_accept=0: {reject_count} ({reject_count/len(samples)*100:.1f}%)")
        
        # 打印几个样例
        print("  Sample examples:")
        for s in samples[:3]:
            print(f"    - {s.proposer_id[:15]} ({s.role}): q={s.q}, p={s.p:.1f}, y={s.y_accept}")
    
    print("  ✓ Real data test passed")


def main():
    print("=" * 60)
    print("DatasetBuilder Test Suite")
    print("=" * 60)
    
    test_safe_literal_eval()
    test_parse_issues()
    test_offers_match()
    test_extract_samples_basic()
    test_extract_samples_failed_negotiation()
    test_real_data()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
