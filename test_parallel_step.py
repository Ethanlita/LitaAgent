"""测试并行模式下 step 方法是否被调用"""
import os
import sys
import shutil
import glob

sys.path.insert(0, '.')

log_dir = './test_parallel_tracker'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(f'{log_dir}/tracker_logs', exist_ok=True)

from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_p import LitaAgentP
from scml.utils import anac2024_oneshot

# 创建一个测试类，直接在 step 中写文件
class TestLitaAgentY(LitaAgentY):
    def step(self):
        super().step()
        try:
            # 在每一步都写文件
            current_step = self.awi.current_step
            n_steps = self.awi.n_steps
            agent_id = self.id.replace('@', '_at_')
            
            # 使用 world_id
            world_id = 'unknown'
            if hasattr(self.awi, '_world') and self.awi._world:
                world_id = getattr(self.awi._world, 'id', 'unknown')
            safe_world_id = str(world_id).replace('/', '_').replace('\\', '_')[:30]
            
            # 写标记文件
            marker_file = f'./test_parallel_tracker/step_{agent_id}_{safe_world_id}_{current_step}.txt'
            with open(marker_file, 'w') as f:
                f.write(f'agent={self.id}, step={current_step}/{n_steps}, world={world_id}')
            
            if current_step >= n_steps - 1:
                last_step_file = f'./test_parallel_tracker/LAST_STEP_{agent_id}_{safe_world_id}.txt'
                with open(last_step_file, 'w') as f:
                    f.write(f'LAST STEP: agent={self.id}, world={world_id}')
        except Exception as e:
            with open(f'./test_parallel_tracker/error_{self.id.replace("@", "_at_")}.txt', 'w') as f:
                import traceback
                f.write(str(e) + '\n' + traceback.format_exc())


if __name__ == '__main__':
    print('Running PARALLEL tournament with TestLitaAgentY...')
    results = anac2024_oneshot(
        competitors=[TestLitaAgentY, LitaAgentP],
        n_configs=2,
        n_runs_per_world=1,
        n_steps=5,
        print_exceptions=True,
        verbose=False,
        parallelism='parallel',
    )
    
    print('Tournament completed.')
    
    # 检查标记文件
    step_files = glob.glob('./test_parallel_tracker/step_*.txt')
    last_files = glob.glob('./test_parallel_tracker/LAST_STEP_*.txt')
    error_files = glob.glob('./test_parallel_tracker/error_*.txt')
    
    print(f'Step files: {len(step_files)}')
    print(f'Last step files: {len(last_files)}')
    print(f'Error files: {len(error_files)}')
    
    for f in last_files[:3]:
        with open(f) as fh:
            print(f'  {os.path.basename(f)}: {fh.read()}')
    
    for f in error_files[:3]:
        with open(f) as fh:
            print(f'  ERROR {os.path.basename(f)}: {fh.read()[:200]}')
