#!/usr/bin/env python3
import json

f=open(r'd:\SCML_initial\tournament_history\20260112_125825_oneshot\tracker_logs\agent_00LOS_at_0#W4d66faa0_world_000420260112H125836601794JGIh_L-M-R-S-An_4d66faa0ddd9.json')
d=json.load(f)
entries = d.get('entries', [])
# Find daily_status for first 5 days
daily = [e for e in entries if e.get('event') == 'daily_status']
for ds in daily[:10]:
    day = ds['day']
    data = ds['data']
    print(f"Day {day}: needed_supplies={data.get('needed_supplies')}, needed_sales={data.get('needed_sales')}, total_supplies={data.get('total_supplies')}, total_sales={data.get('total_sales')}, exo_input={data.get('exo_input_qty')}, exo_output={data.get('exo_output_qty')}")
