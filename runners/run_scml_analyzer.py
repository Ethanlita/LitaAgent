"""
SCML Analyzer - 运行比赛并分析结果

这是 scml_analyzer 框架的入口脚本。
功能:
1. 运行SCML比赛（支持多种预设）
2. 自动收集日志（NegMAS日志 + Agent打点数据）
3. 比赛后分析并生成报告
4. 启动可视化界面查看结果

Usage:
    python run_scml_analyzer.py                  # 快速测试
    python run_scml_analyzer.py --mode standard  # 标准比赛
    python run_scml_analyzer.py --mode full      # 完整比赛
    python run_scml_analyzer.py --visualize ./tournament_logs/xxx  # 可视化已有数据
"""

import os
import sys
import argparse
import warnings
import builtins
import io
import json
import csv

# Suppress noisy warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 检查是否是静默模式 (在导入之前检查)
_SILENT_MODE = '--silent' in sys.argv

# 如果是静默模式，在导入前就禁用 tqdm
if _SILENT_MODE:
    # 禁用 tqdm 进度条
    os.environ['TQDM_DISABLE'] = '1'
    
    # 替换 tqdm 为无操作版本
    class FakeTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def close(self):
            pass
        def set_description(self, *args, **kwargs):
            pass
        def set_postfix(self, *args, **kwargs):
            pass
        def write(self, *args, **kwargs):
            pass
        @staticmethod
        def tqdm(iterable=None, *args, **kwargs):
            return FakeTqdm(iterable, *args, **kwargs)
    
    # 预先注入 fake tqdm
    sys.modules['tqdm'] = FakeTqdm()
    sys.modules['tqdm.auto'] = FakeTqdm()
    sys.modules['tqdm.std'] = FakeTqdm()

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 静音 print 输出
# ============================================================

class SilentPrinter:
    """静默打印器 - 将 print 输出重定向到日志"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._original_print = builtins.print
        self._log_entries = []
    
    def __call__(self, *args, **kwargs):
        """拦截 print 调用"""
        if not self.enabled:
            return self._original_print(*args, **kwargs)
        
        # 将输出存储到日志
        message = " ".join(str(arg) for arg in args)
        self._log_entries.append(message)
        
        # 只显示重要信息（包含特定标记的）
        important_markers = ['✓', '✗', '==', '--', 'Tournament', 'Phase', 
                            'SUMMARY', 'Winner', 'Rankings', 'Complete',
                            'Loading', 'Loaded', 'Duration', 'Worlds']
        if any(marker in message for marker in important_markers):
            self._original_print(*args, **kwargs)
    
    def get_logs(self) -> list:
        return self._log_entries.copy()
    
    def restore(self):
        builtins.print = self._original_print


class OutputSuppressor:
    """完全静默输出 - 包括 stderr (tqdm进度条等)"""
    
    def __init__(self):
        import io
        self._original_stderr = sys.stderr
        self._original_stdout = sys.stdout
        self._devnull = io.StringIO()
        self._enabled = False
    
    def suppress(self, suppress_stderr: bool = True, suppress_stdout: bool = False):
        """开始静默"""
        self._enabled = True
        if suppress_stderr:
            sys.stderr = self._devnull
        if suppress_stdout:
            sys.stdout = self._devnull
    
    def restore(self):
        """恢复输出"""
        if self._enabled:
            sys.stderr = self._original_stderr
            sys.stdout = self._original_stdout
            self._enabled = False


def load_agents(verbose: bool = True):
    """Load all available agents."""
    agents = []
    
    if verbose:
        print("Loading agents...")
    
    # LitaAgents (local)
    lita_agents = [
        ('litaagent_std.litaagent_y', 'LitaAgentY'),
        ('litaagent_std.litaagent_yr', 'LitaAgentYR'),
        ('litaagent_std.litaagent_n', 'LitaAgentN'),
    ]
    
    for module, classname in lita_agents:
        try:
            mod = __import__(module, fromlist=[classname])
            cls = getattr(mod, classname)
            agents.append(cls)
            if verbose:
                print(f"  ✓ {classname}")
        except Exception as e:
            if verbose:
                print(f"  ✗ {classname}: {e}")
    
    # Top agents from scml-agents
    top_agents = [
        ('scml_agents.scml2024.standard.team_penguin.p_agent', 'PenguinAgent'),
        ('scml_agents.scml2024.standard.team_atsunaga.as0', 'AS0'),
    ]
    
    for module, classname in top_agents:
        try:
            mod = __import__(module, fromlist=[classname])
            cls = getattr(mod, classname)
            agents.append(cls)
            if verbose:
                print(f"  ✓ {classname}")
        except Exception as e:
            if verbose:
                print(f"  ✗ {classname}: {e}")
    
    # 注入 Tracker 到 LitaAgents
    try:
        from litaagent_std.tracker_mixin import inject_tracker_to_agents
        agents = inject_tracker_to_agents(agents)
        if verbose:
            print("  ✓ Tracker 已注入到所有 Agent")
    except Exception as e:
        if verbose:
            print(f"  ⚠ Tracker 注入失败: {e}")
    
    return agents


def run_tournament_and_analyze(agents, mode='quick'):
    """
    Run tournament and analyze results.
    
    Args:
        agents: List of agent classes
        mode: 'quick', 'standard', or 'full'
    """
    from scml_analyzer import Tournament, TournamentConfig
    from scml_analyzer.log_parser import LogParser
    from scml_analyzer.report import ReportGenerator
    
    # Create tournament with appropriate config
    if mode == 'quick':
        config = TournamentConfig.quick_test()
    elif mode == 'standard':
        config = TournamentConfig.standard()
    elif mode == 'full':
        config = TournamentConfig.full()
    else:
        config = TournamentConfig.quick_test()
    
    config.name = f"scml_analysis_{mode}"
    
    print("\n" + "=" * 60)
    print("Phase 1: Running Tournament")
    print("=" * 60)
    
    tournament = Tournament(agents=agents, config=config)
    results = tournament.run(verbose=True)
    
    print("\n" + "=" * 60)
    print("Phase 2: Analyzing Results")
    print("=" * 60)
    
    # Analyze NegMAS logs
    print("\nAnalyzing NegMAS logs...")
    
    log_dir = results.log_dir
    world_dirs = [d for d in os.listdir(log_dir) if d.startswith("world_")]
    
    # Build proper rankings from agents.json with short_type
    agent_scores = {}  # agent_type -> list of scores
    
    for world_dir in world_dirs:
        world_path = os.path.join(log_dir, world_dir)
        agents_json_path = os.path.join(world_path, "agents.json")
        stats_csv_path = os.path.join(world_path, "stats.csv.csv")  # SCML uses this name
        
        try:
            # Load agent info
            with open(agents_json_path, 'r', encoding='utf-8') as f:
                agents_info = json.load(f)
            
            # Load scores from stats.csv
            # Format: score_AGENTID, balance_AGENTID
            agent_final_scores = {}
            if os.path.exists(stats_csv_path):
                with open(stats_csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]  # Last step has final scores
                        for key, value in last_row.items():
                            if key.startswith('score_'):
                                agent_id = key[6:]  # Remove 'score_' prefix
                                try:
                                    agent_final_scores[agent_id] = float(value)
                                except (ValueError, TypeError):
                                    pass
            
            # Map short_type to scores
            for agent_id, info in agents_info.items():
                short_type = info.get('short_type', 'Unknown')
                if short_type in ('System', 'BUYER', 'SELLER'):
                    continue  # Skip system agents
                
                # Find score for this agent
                score = agent_final_scores.get(agent_id)
                
                if score is not None:
                    if short_type not in agent_scores:
                        agent_scores[short_type] = []
                    agent_scores[short_type].append(score)
        except Exception as e:
            print(f"  Warning: Could not load {world_dir}/agents.json: {e}")
    
    # Calculate proper rankings
    import pandas as pd
    proper_rankings = []
    for agent_type, scores in sorted(agent_scores.items()):
        if scores:
            proper_rankings.append({
                'agent_type': agent_type,
                'mean': sum(scores) / len(scores),
                'std': pd.Series(scores).std() if len(scores) > 1 else 0,
                'min': min(scores),
                'max': max(scores),
                'count': len(scores),
            })
    
    # Sort by mean descending
    proper_rankings.sort(key=lambda x: x['mean'], reverse=True)
    
    analysis_summary = {
        "tournament_name": results.name,
        "winner": proper_rankings[0]['agent_type'] if proper_rankings else results.winner,
        "rankings": proper_rankings if proper_rankings else (results.rankings.to_dict(orient='records') if not results.rankings.empty else []),
        "worlds_analyzed": [],
        "agent_stats": {},
    }
    
    # Aggregate stats per agent type
    agent_contracts = {}
    agent_negotiations = {}
    
    for world_dir in world_dirs[:5]:  # Analyze first 5 worlds for summary
        world_path = os.path.join(log_dir, world_dir)
        
        # Try to parse logs
        try:
            parser = LogParser(exclude_system_agents=True)
            data = parser.parse_directory(world_path)
            
            analysis_summary["worlds_analyzed"].append({
                "world": world_dir,
                "n_contracts": len(data.contracts),
                "n_negotiations": len(data.negotiations),
            })
            
            # Aggregate by agent type
            for agent_id, agent_data in data.agents.items():
                # Extract agent type from ID (e.g., "LitaAgentY@0" -> "LitaAgentY")
                agent_type = agent_id.split("@")[0] if "@" in agent_id else agent_id
                
                if agent_type not in agent_contracts:
                    agent_contracts[agent_type] = 0
                    agent_negotiations[agent_type] = 0
                
                agent_contracts[agent_type] += len(agent_data.all_contracts)
                agent_negotiations[agent_type] += len(agent_data.negotiations)
                
        except Exception as e:
            print(f"  Warning: Could not parse {world_dir}: {e}")
    
    # Check for tracker logs
    tracker_dir = os.path.join(log_dir, "tracker_logs")
    if os.path.exists(tracker_dir):
        tracker_files = [f for f in os.listdir(tracker_dir) if f.startswith('agent_') and f.endswith('.json')]
        print(f"\nFound {len(tracker_files)} tracker log files")
        analysis_summary["tracker_files"] = len(tracker_files)
    
    print("\n" + "=" * 60)
    print("Phase 3: Generating Report")
    print("=" * 60)
    
    # Save analysis summary
    summary_path = os.path.join(log_dir, "analysis_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis saved to: {summary_path}")
    
    # Print summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Tournament: {results.name}")
    print(f"Duration: {results.duration_seconds:.1f}s")
    print(f"Worlds completed: {results.n_worlds_completed}")
    print(f"Worlds failed: {results.n_worlds_failed}")
    print(f"\nWinner: {results.winner}")
    print(f"\nFull results at: {log_dir}")
    
    return results, analysis_summary


def visualize_data(log_dir: str = None, launch_browser: bool = True, port: int = 8080):
    """
    可视化已有的比赛数据
    
    Args:
        log_dir: 比赛日志目录 (可选，如果不提供则显示 tournament_history 列表)
        launch_browser: 是否自动打开浏览器
        port: HTTP服务器端口
    """
    from scml_analyzer.visualizer import start_server
    
    print("=" * 60)
    print("SCML Analyzer - 数据可视化")
    print("=" * 60)
    
    if log_dir:
        # 如果提供了 log_dir，先导入到 tournament_history
        try:
            from scml_analyzer.history import import_tournament
            print(f"\n导入数据: {log_dir}")
            tournament_id = import_tournament(log_dir, copy_mode=False)
            if tournament_id:
                print(f"✓ 数据已导入: {tournament_id}")
        except Exception as e:
            print(f"⚠ 导入失败: {e}")
    
    # 启动无参数可视化服务器
    print(f"\n启动可视化服务器...")
    start_server(port=port, open_browser=launch_browser)


def main():
    parser = argparse.ArgumentParser(description='SCML Analyzer - 运行比赛并分析')
    parser.add_argument('--mode', choices=['quick', 'standard', 'full'], 
                        default='quick', help='比赛规模')
    parser.add_argument('--visualize', type=str, default=None,
                        help='可视化已有数据（提供日志目录路径）')
    parser.add_argument('--no-browser', action='store_true',
                        help='不自动打开浏览器')
    parser.add_argument('--silent', action='store_true',
                        help='静默模式，减少Agent的print输出')
    parser.add_argument('--port', type=int, default=8080,
                        help='可视化服务器端口')
    parser.add_argument('--auto-visualize', action='store_true',
                        help='比赛完成后自动启动可视化界面')
    args = parser.parse_args()
    
    # 如果只是可视化
    if args.visualize:
        visualize_data(args.visualize, launch_browser=not args.no_browser, port=args.port)
        return
    
    print("=" * 60)
    print("SCML Analyzer v0.3.0")
    print("=" * 60)
    
    # 启用静默模式
    silent_printer = None
    output_suppressor = None
    if args.silent:
        silent_printer = SilentPrinter(enabled=True)
        output_suppressor = OutputSuppressor()
        builtins.print = silent_printer
        output_suppressor.suppress(suppress_stderr=True)  # 抑制 tqdm 进度条
        print("\n静默模式已启用 - Agent输出和进度条将被过滤")
    
    print("\nLoading agents...")
    agents = load_agents(verbose=True)
    
    if len(agents) < 2:
        print("\nError: Need at least 2 agents to run a tournament!")
        sys.exit(1)
    
    print(f"\nLoaded {len(agents)} agents")
    
    try:
        # Run tournament and analyze
        results, analysis = run_tournament_and_analyze(agents, args.mode)
        
        print("\n" + "=" * 60)
        print("Complete!")
        print("=" * 60)
        
        # 获取日志目录
        log_dir = results.log_dir
        
        # 自动启动可视化或显示提示
        if args.auto_visualize:
            print("\n" + "=" * 60)
            print("自动启动可视化界面...")
            print("=" * 60)
            # 恢复输出以便可视化正常显示
            if silent_printer:
                silent_printer.restore()
                silent_printer = None
            if output_suppressor:
                output_suppressor.restore()
                output_suppressor = None
            
            # 先导入数据到 tournament_history
            try:
                from scml_analyzer.history import import_tournament
                tournament_id = import_tournament(log_dir, copy_mode=False)
                if tournament_id:
                    print(f"✓ 数据已导入: {tournament_id}")
            except Exception as e:
                print(f"⚠ 导入失败: {e}")
            
            # 启动无参数可视化
            visualize_data(log_dir=None, launch_browser=True, port=args.port)
        else:
            print(f"\n提示: 使用以下命令启动可视化界面:")
            print(f"  python run_scml_analyzer.py --visualize \"{log_dir}\"")
            print(f"  或直接运行: python -c \"from scml_analyzer.visualizer import start_server; start_server()\"")
        
    finally:
        # 恢复 print
        if silent_printer:
            silent_printer.restore()
        if output_suppressor:
            output_suppressor.restore()


if __name__ == "__main__":
    main()
