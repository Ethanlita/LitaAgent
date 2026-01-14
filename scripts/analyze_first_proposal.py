"""分析 LOS 的 first_proposal 行为，检查是否超出外生输入/输出"""
import json
from pathlib import Path
from collections import defaultdict

# 找到最新的比赛
tournament_dirs = sorted(Path(r"D:\SCML_initial\tournament_history").glob("20260110_*_oneshot"))
if not tournament_dirs:
    print("No tournaments found")
    exit(1)

tracker_dir = tournament_dirs[-1] / "tracker_logs"
print(f"Analyzing: {tournament_dirs[-1].name}")

los_files = list(tracker_dir.glob("agent_*LOS*.json"))
print(f"Found {len(los_files)} LOS agent files")

# 同时需要从 world 的 contracts.csv 获取外生合同信息
stage_dir = tournament_dirs[-1] / f"{tournament_dirs[-1].name}-stage-0001"

for f in los_files[:5]:
    print(f"\n{'='*60}")
    print(f"File: {f.name}")
    
    data = json.loads(f.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    
    # 找到 agent_initialized 获取 level 和 world_id
    init = next((e for e in entries if e["event"] == "agent_initialized"), None)
    if init:
        level = init["data"].get("level", "?")
        n_lines = init["data"].get("n_lines", "?")
        print(f"Level: {level}, n_lines: {n_lines}")
    
    # 按 step 分组分析 first_proposal
    step_data = defaultdict(lambda: {"first_proposals": []})
    
    for e in entries:
        step = e.get("day", 0)
        event = e.get("event", "")
        d = e.get("data", {})
        
        if event == "offer_made":
            reason = d.get("reason", "")
            offer = d.get("offer", {})
            q = offer.get("quantity")
            p = offer.get("unit_price")
            partner = d.get("partner", "?")[:15]
            role = d.get("role", "?")
            
            if reason == "first_proposal":
                step_data[step]["first_proposals"].append({
                    "q": q, "p": p, "partner": partner, "role": role
                })
    
    # 分析每个 step
    print("\nFirst Proposal Analysis:")
    for step in sorted(step_data.keys())[:10]:
        sd = step_data[step]
        fps = sd["first_proposals"]
        
        if not fps:
            continue
        
        total_q = sum(fp["q"] or 0 for fp in fps)
        n_offers = len(fps)
        role = fps[0]["role"] if fps else "?"
        
        print(f"  Step {step} [{role.upper()}]: {n_offers} first_proposals, total_q={total_q}")
        for fp in fps[:5]:
            print(f"    → Q={fp['q']}, P={fp['p']}, to={fp['partner']}")
