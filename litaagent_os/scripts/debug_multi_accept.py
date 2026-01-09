#!/usr/bin/env python3
"""
深入分析 y_accept=1 样本数 > 1 的问题
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import ast

from litaagent_os.scripts.build_accept_dataset import (
    safe_literal_eval,
    offers_match,
    extract_samples_from_negotiation,
)


def analyze_multi_accept_issue():
    world_dir = ROOT / "run_logs" / "oneshot_syslog_probe_20260108_164620"
    
    neg_df = pd.read_csv(world_dir / "negotiations.csv")
    negs_df = pd.read_csv(world_dir / "negs.csv")
    negs_map = {str(row.get("name", "")): row for _, row in negs_df.iterrows()}
    
    # 找出有问题的谈判
    problem_neg_ids = [
        "721be7c1-71fd-4882-8983-633ef6b03568",
        "08573c34-75c4-49ab-a837-1cac7d936e4a",
    ]
    
    for neg_id in problem_neg_ids:
        neg_row = neg_df[neg_df["id"] == neg_id].iloc[0]
        negs_row = negs_map.get(neg_id)
        
        print(f"\n{'='*60}")
        print(f"Negotiation: {neg_id}")
        print(f"{'='*60}")
        
        partners = safe_literal_eval(neg_row.get("partners"), default=[])
        buyer = str(neg_row.get("buyer", ""))
        seller = str(neg_row.get("seller", ""))
        agreement = safe_literal_eval(neg_row.get("agreement"), default=None)
        offers_dict = safe_literal_eval(neg_row.get("offers"), default={})
        history = safe_literal_eval(neg_row.get("history"), default=[])
        
        print(f"Partners: {partners}")
        print(f"Buyer: {buyer}")
        print(f"Seller: {seller}")
        print(f"Agreement: {agreement}")
        
        print(f"\nOffers dict:")
        for agent, agent_offers in offers_dict.items():
            print(f"  {agent}: {agent_offers}")
        
        print(f"\nHistory (offer sequence):")
        for i, step_data in enumerate(history):
            if not isinstance(step_data, dict):
                continue
            new_offers = step_data.get("new_offers", [])
            rel_time = step_data.get("relative_time", 0.0)
            if new_offers:
                for item in new_offers:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        proposer, offer = item[0], item[1]
                        match_agreement = offers_match(offer, agreement) if offer and agreement else False
                        print(f"  Step {i}: proposer={proposer}, offer={offer}, "
                              f"match_agreement={match_agreement}, rel_time={rel_time}")
        
        # 提取样本并分析
        samples = extract_samples_from_negotiation(neg_row, negs_row, "test_world")
        accept_samples = [s for s in samples if s.y_accept == 1]
        
        print(f"\n提取的 y_accept=1 样本:")
        for s in accept_samples:
            print(f"  proposer={s.proposer_id}, offer=({s.q}, {s.t}, {s.p}), offer_index={s.offer_index}")


if __name__ == "__main__":
    analyze_multi_accept_issue()
