#!/usr/bin/env python3
"""检查 tracker 数据"""
import json
from pathlib import Path

# 加载新比赛的 tracker 数据
tracker_dir = Path('tournament_history/20260110_025715_oneshot/tracker_logs')
agent_files = list(tracker_dir.glob('agent_*LOS*.json'))
print(f'Found {len(agent_files)} LitaAgentOS tracker files')

if agent_files:
    with open(agent_files[0]) as f:
        data = json.load(f)
    entries = [e for e in data.get('entries', []) if e.get('event') == 'daily_status']
    print(f'daily_status entries: {len(entries)}')
    for e in entries[:5]:
        d = e['data']
        print(f"Day {e['day']}: exo_in={d.get('exo_input_qty')}, needed_sales={d.get('needed_sales')}, balance={d.get('balance'):.0f}")
