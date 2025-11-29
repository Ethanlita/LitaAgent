import json
import glob
import os

log_dir = 'tournament_history/20251128_130949_oneshot'
tracker_dir = f'{log_dir}/tracker_logs'

files = glob.glob(f'{tracker_dir}/*.json')
print(f'Tracker logs found: {len(files)}')

if files:
    with open(files[0]) as f:
        data = json.load(f)
    
    print(f'\nSample file: {os.path.basename(files[0])}')
    print(f'Keys: {list(data.keys())}')
    print(f'Agent type: {data.get("agent_type", "N/A")}')
    print(f'Agent ID: {data.get("agent_id", "N/A")}')
    print(f'World ID: {data.get("world_id", "N/A")}')
    
    events = data.get('events', [])
    print(f'\nTotal events: {len(events)}')
    
    # 统计事件类型
    event_types = {}
    for e in events:
        et = e.get('event', 'unknown')
        event_types[et] = event_types.get(et, 0) + 1
    
    print('\nEvent types:')
    for et, count in sorted(event_types.items()):
        print(f'  {et}: {count}')
    
    # 检查是否有 offer_made 事件的详细内容
    offer_events = [e for e in events if e.get('event') == 'offer_made']
    if offer_events:
        print('\nSample offer_made event:')
        print(json.dumps(offer_events[0], indent=2, ensure_ascii=False))
    else:
        print('\nNo offer_made events found!')
        
    # 检查 negotiation 相关事件
    neg_events = [e for e in events if 'negotiation' in e.get('event', '').lower() or 'offer' in e.get('event', '').lower()]
    print(f'\nNegotiation-related events: {len(neg_events)}')
    if neg_events:
        print('Sample:')
        print(json.dumps(neg_events[0], indent=2, ensure_ascii=False))
