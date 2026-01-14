#!/usr/bin/env python
"""分析 SELLER 的详细行为"""

import json
import glob
from collections import Counter

log_dir = 'tournament_history/20260111_231959_oneshot/tracker_logs'
logs = glob.glob(f'{log_dir}/agent_*LOS*.json')

# 看一个 SELLER 的所有事件
for f in logs:
    with open(f, 'r') as fp:
        data = json.load(fp)
    
    level = None
    for e in data.get('entries', []):
        if e.get('event') == 'agent_initialized':
            level = e.get('data', {}).get('level')
            break
    
    if level != 0:  # 只看 SELLER
        continue
    
    print(f'File: {f}')
    print(f'Level: {level}')
    events = [e.get('event') for e in data.get('entries', [])]
    print(f'Events: {Counter(events)}')
    
    # 看前30个事件
    for e in data.get('entries', [])[:30]:
        print(f"  {e.get('event')}: {str(e.get('data', {}))[:100]}")
    break
