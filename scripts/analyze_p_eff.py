"""分析 LOS 的 accept model 预测值 (mu) 和 p_eff"""
import json
from pathlib import Path

# 找到最新的比赛
tournament_dirs = sorted(Path(r"D:\SCML_initial\tournament_history").glob("20260110_*_oneshot"))
if not tournament_dirs:
    print("No tournaments found")
    exit(1)

tracker_dir = tournament_dirs[-1] / "tracker_logs"
print(f"Analyzing: {tournament_dirs[-1].name}")

los_files = list(tracker_dir.glob("agent_*LOS*.json"))
print(f"Found {len(los_files)} LOS agent files")

for f in los_files[:2]:
    print(f"\n{'='*60}")
    print(f"File: {f.name}")
    
    data = json.loads(f.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    
    # 找 aop_action 事件（包含 mu, strength 等信息）
    aop_actions = [e for e in entries if e["event"] == "aop_action"]
    print(f"Found {len(aop_actions)} aop_action events")
    
    # 看看第一个的结构
    if aop_actions:
        first_aop = aop_actions[0]
        print(f"\nFirst aop_action (day {first_aop.get('day')}):")
        print(json.dumps(first_aop.get("data", {}), indent=2))
    
    # 看看 offer_made 事件中有没有更多信息
    offer_made = [e for e in entries if e["event"] == "offer_made" and e.get("data", {}).get("reason") == "first_proposal"]
    print(f"\nFound {len(offer_made)} first_proposal events")
    
    if offer_made:
        first_offer = offer_made[0]
        print(f"\nFirst offer_made (day {first_offer.get('day')}):")
        print(json.dumps(first_offer.get("data", {}), indent=2))
