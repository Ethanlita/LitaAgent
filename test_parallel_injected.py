"""测试静态定义的 Tracked Agent 在并行模式下是否工作"""
import os
import sys
import shutil
import glob

sys.path.insert(0, '.')


def main():
    log_dir = './test_parallel_tracker'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    tracker_log_dir = os.path.abspath(f'{log_dir}/tracker_logs')
    os.makedirs(tracker_log_dir, exist_ok=True)

    # 设置环境变量（必须在导入前设置）
    os.environ['SCML_TRACKER_LOG_DIR'] = tracker_log_dir

    from litaagent_std.litaagent_y import LitaAgentYTracked
    from litaagent_std.litaagent_p import LitaAgentP
    from scml.utils import anac2024_oneshot

    print(f'LitaAgentYTracked: {LitaAgentYTracked}')
    print(f'SCML_TRACKER_LOG_DIR: {os.environ.get("SCML_TRACKER_LOG_DIR")}')

    print('\nRunning PARALLEL tournament...')
    results = anac2024_oneshot(
        competitors=[LitaAgentYTracked, LitaAgentP],
        n_configs=2,
        n_runs_per_world=1,
        n_steps=10,
        print_exceptions=True,
        verbose=False,
        parallelism='parallel',
    )

    print('Tournament completed.')

    import json
    files = glob.glob(f'{tracker_log_dir}/agent_*.json')
    print(f'Tracker files: {len(files)}')
    for f in files[:10]:
        data = json.load(open(f))
        stats = data['stats']
        om = stats.get('offers_made', 0)
        aid = data.get('agent_id', '?')
        print(f'  {aid}: offers_made={om}')


if __name__ == '__main__':
    main()
