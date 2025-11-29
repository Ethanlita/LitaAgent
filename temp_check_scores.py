import csv
f = open('C:/Users/11985/negmas/tournaments/20251128H160240847759Kqg-stage-0001/scores.csv', 'r', encoding='utf-8')
reader = csv.DictReader(f)
rows = list(reader)
f.close()
print('Total score entries:', len(rows))
print('Columns:', list(rows[0].keys()))
print('\nFirst 10 entries:')
for r in rows[:10]:
    world = r['world'][:50]
    agent = r['agent_type'].split('.')[-1]
    score = r['score']
    print(f'  {world}... | {agent} | {score}')

# 分析相同配置但不同 run 的情况
print('\n\n=== 分析同一配置的不同 run ===')
from collections import defaultdict
config_runs = defaultdict(list)
for r in rows:
    # 提取配置名 (去掉最后的 .XX run index)
    world = r['world']
    parts = world.rsplit('.', 1)
    if len(parts) == 2:
        config_name = parts[0]
        run_idx = parts[1]
        config_runs[config_name].append({
            'run': run_idx,
            'agent': r['agent_type'].split('.')[-1],
            'score': float(r['score'])
        })

# 找一个有多个 run 的例子
for config, runs in list(config_runs.items())[:1]:
    print(f'\n配置: {config[-60:]}')
    print(f'  共 {len(runs)} 条记录')
    # 按 agent 分组
    agent_scores = defaultdict(list)
    for run in runs:
        agent_scores[run['agent']].append((run['run'], run['score']))
    
    for agent, scores in agent_scores.items():
        print(f'  {agent}:')
        for run_idx, score in sorted(scores):
            print(f'    Run {run_idx}: {score:.4f}')
