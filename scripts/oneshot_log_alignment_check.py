#!/usr/bin/env python3
"""
Check alignment between reconstructed REJECT/ACCEPT/END events and world logs.

This script reconstructs response events from negotiation step logs and
compares them with negotiation-level agreement status and actions.csv states.
"""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_literal(value: Any, default: Any = None) -> Any:
    if pd.isna(value):
        return default
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def is_agreement(value: Any) -> bool:
    if pd.isna(value):
        return False
    s = str(value).strip()
    return s not in ("", "None", "nan", "NaN")


def extract_offers(neg_step_df: pd.DataFrame) -> List[Dict[str, Any]]:
    offers: List[Dict[str, Any]] = []
    last_current_offer = None

    for _, row in neg_step_df.iterrows():
        round_rel = row.get("relative_time")
        step = row.get("step")

        new_offers = parse_literal(row.get("new_offers"), default=[])
        if new_offers:
            for item in new_offers:
                try:
                    proposer, offer = item
                except Exception:
                    continue
                if offer is None:
                    continue
                offers.append(
                    {
                        "proposer": proposer,
                        "offer": offer,
                        "round_rel": round_rel,
                        "step": step,
                    }
                )
            continue

        current_offer = parse_literal(row.get("current_offer"), default=None)
        current_proposer = row.get("current_proposer")
        if current_offer is not None and current_offer != last_current_offer and current_proposer:
            offers.append(
                {
                    "proposer": current_proposer,
                    "offer": current_offer,
                    "round_rel": round_rel,
                    "step": step,
                }
            )
            last_current_offer = current_offer

    return offers


def reconstruct_events(
    log_dir: Path, neg_df: pd.DataFrame, neg_id: str
) -> Optional[Dict[str, Any]]:
    neg_path = log_dir / "negotiations" / f"{neg_id}.csv"
    if not neg_path.exists():
        return None

    neg_step_df = pd.read_csv(neg_path)
    offers = extract_offers(neg_step_df)

    neg_row = neg_df.loc[neg_df["id"] == neg_id]
    if neg_row.empty:
        return None
    neg_row = neg_row.iloc[0]

    partners = parse_literal(neg_row.get("partners"), default=[])
    agreement = neg_row.get("agreement")

    events: List[Dict[str, Any]] = []
    prev_offer = None
    for offer in offers:
        if prev_offer is None:
            events.append(
                {
                    "type": "REJECT",
                    "responder": offer["proposer"],
                    "partner": None,
                    "responded_offer": None,
                    "counter_offer": offer["offer"],
                    "is_first_proposal": True,
                    "step": offer["step"],
                }
            )
        else:
            events.append(
                {
                    "type": "REJECT",
                    "responder": offer["proposer"],
                    "partner": prev_offer["proposer"],
                    "responded_offer": prev_offer["offer"],
                    "counter_offer": offer["offer"],
                    "is_first_proposal": False,
                    "step": offer["step"],
                }
            )
        prev_offer = offer

    if is_agreement(agreement):
        last_prop = prev_offer["proposer"] if prev_offer else None
        other = None
        if partners and last_prop in partners and len(partners) == 2:
            other = partners[0] if partners[1] == last_prop else partners[1]
        events.append(
            {
                "type": "ACCEPT",
                "responder": other,
                "partner": last_prop,
                "responded_offer": parse_literal(agreement, default=agreement),
                "counter_offer": None,
                "is_first_proposal": False,
                "step": prev_offer["step"] if prev_offer else None,
            }
        )
    else:
        events.append(
            {
                "type": "END",
                "responder": None,
                "partner": None,
                "responded_offer": None,
                "counter_offer": None,
                "is_first_proposal": False,
                "step": None,
            }
        )

    return {
        "neg_id": neg_id,
        "partners": partners,
        "agreement": agreement,
        "events": events,
    }


def sample_negotiations(
    neg_df: pd.DataFrame, n: int, seed: int
) -> List[str]:
    if n <= 0:
        return []

    rnd = random.Random(seed)
    with_agreement = neg_df[neg_df["agreement"].apply(is_agreement)]["id"].tolist()
    without_agreement = neg_df[~neg_df["agreement"].apply(is_agreement)]["id"].tolist()

    selected: List[str] = []
    if with_agreement:
        selected.append(rnd.choice(with_agreement))
    if n > 1 and without_agreement:
        choice = rnd.choice(without_agreement)
        if choice not in selected:
            selected.append(choice)

    remaining = n - len(selected)
    if remaining <= 0:
        return selected

    pool = [x for x in neg_df["id"].tolist() if x not in selected]
    if pool:
        selected.extend(rnd.sample(pool, min(remaining, len(pool))))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check REJECT/ACCEPT/END reconstruction from OneShot world logs."
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="World log directory (contains actions.csv, negotiations.csv, negs.csv).",
    )
    parser.add_argument(
        "--neg-id",
        action="append",
        default=[],
        help="Negotiation UUID to inspect (can be repeated).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of negotiations to sample if --neg-id is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--print-events",
        action="store_true",
        help="Print reconstructed event list for each negotiation.",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    required = ["actions.csv", "negotiations.csv", "negs.csv"]
    missing = [name for name in required if not (log_dir / name).exists()]
    if missing:
        print(f"Missing required files in {log_dir}: {missing}")
        return 1
    if not (log_dir / "negotiations").exists():
        print(f"Missing negotiations folder in {log_dir}")
        return 1

    neg_df = pd.read_csv(log_dir / "negotiations.csv")
    negs_df = pd.read_csv(log_dir / "negs.csv")
    actions_df = pd.read_csv(log_dir / "actions.csv")

    neg_ids = args.neg_id
    if not neg_ids:
        neg_ids = sample_negotiations(neg_df, args.sample, args.seed)

    if not neg_ids:
        print("No negotiations found to inspect.")
        return 1

    for neg_id in neg_ids:
        rec = reconstruct_events(log_dir, neg_df, neg_id)
        if not rec:
            print(f"---\nnegotiation_id: {neg_id}\nERROR: missing negotiation log")
            continue

        negs_row = negs_df.loc[negs_df["name"] == neg_id]
        neg_int_id = int(negs_row.iloc[0]["id"]) if not negs_row.empty else None
        action_states: List[str] = []
        if neg_int_id is not None:
            act = actions_df[actions_df["neg_id"] == neg_int_id]
            if not act.empty:
                action_states = act["state"].unique().tolist()

        counts = {"REJECT": 0, "ACCEPT": 0, "END": 0}
        for ev in rec["events"]:
            counts[ev["type"]] += 1

        last_event = rec["events"][-1] if rec["events"] else None
        agreement_present = is_agreement(rec["agreement"])

        print("---")
        print(f"negotiation_id: {rec['neg_id']}")
        print(f"partners: {rec['partners']}")
        print(f"agreement: {rec['agreement']}")
        print(f"event_counts: {counts}")
        print(f"last_event: {last_event}")
        print(f"actions.csv states: {action_states}")

        if agreement_present and counts["ACCEPT"] != 1:
            print("WARNING: agreement present but ACCEPT count != 1")
        if not agreement_present and counts["END"] != 1:
            print("WARNING: no agreement but END count != 1")
        if agreement_present and "agreement" not in action_states:
            print("NOTE: actions.csv does not include agreement state for this negotiation")

        if args.print_events:
            for i, ev in enumerate(rec["events"], 1):
                print(f"{i}. {ev}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
