"""
SCML Analyzer 可视化服务器

提供 Web 界面查看比赛数据分析结果。
使用简单的 Flask 服务器 + 静态 HTML/JS。

Usage:
    from scml_analyzer.visualizer import start_server
    start_server(data_dir="./tournament_logs/xxx")
    
    # 或者命令行
    python -m scml_analyzer.visualizer --data ./tournament_logs/xxx
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser
import urllib.parse


class VisualizerData:
    """可视化数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._tournament_results: Optional[Dict] = None
        self._agent_data: Dict[str, Dict] = {}
        self._world_data: Dict[str, Dict] = {}
        # 自动加载数据
        self.load_all()
    
    def load_all(self):
        """加载所有数据"""
        self._load_tournament_results()
        self._load_agent_data()
        self._load_world_data()
    
    def _load_tournament_results(self):
        """加载比赛结果"""
        results_file = self.data_dir / "tournament_results.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                self._tournament_results = json.load(f)
    
    def _load_agent_data(self):
        """加载 Agent 数据"""
        tracker_dir = self.data_dir / "tracker_logs"
        if tracker_dir.exists():
            for file in tracker_dir.glob("agent_*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    agent_id = data.get("agent_id", file.stem)
                    self._agent_data[agent_id] = data
    
    def _load_world_data(self):
        """加载 World 数据"""
        for world_dir in self.data_dir.glob("world_*"):
            if world_dir.is_dir():
                world_id = world_dir.name
                self._world_data[world_id] = {
                    "path": str(world_dir),
                    "contracts": self._load_csv(world_dir / "contracts.csv"),
                    "negotiations": self._load_csv(world_dir / "negotiations.csv"),
                }
    
    def _load_csv(self, path: Path) -> List[Dict]:
        """加载 CSV 文件"""
        if not path.exists():
            return []
        
        import csv
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def get_summary(self) -> Dict:
        """获取摘要数据"""
        return {
            "tournament": self._tournament_results or {},
            "n_agents": len(self._agent_data),
            "n_worlds": len(self._world_data),
            "agent_types": list(set(
                d.get("agent_type", "unknown") 
                for d in self._agent_data.values()
            )),
        }
    
    def get_rankings(self) -> List[Dict]:
        """获取排名数据"""
        if self._tournament_results and "rankings" in self._tournament_results:
            return self._tournament_results["rankings"]
        return []
    
    def get_agent_stats(self, agent_id: str) -> Dict:
        """获取 Agent 统计数据"""
        return self._agent_data.get(agent_id, {})
    
    def get_all_agents(self) -> List[str]:
        """获取所有 Agent ID"""
        return list(self._agent_data.keys())
    
    def get_time_series(self, agent_id: str, metric: str) -> List[tuple]:
        """获取时间序列数据"""
        agent_data = self._agent_data.get(agent_id, {})
        time_series = agent_data.get("time_series", {})
        return time_series.get(metric, [])
    
    def to_json(self) -> str:
        """导出为 JSON"""
        return json.dumps({
            "summary": self.get_summary(),
            "rankings": self.get_rankings(),
            "agents": self._agent_data,
            "worlds": {k: {"path": v["path"]} for k, v in self._world_data.items()},
        }, ensure_ascii=False, indent=2)


def generate_html_report(data: VisualizerData) -> str:
    """生成 HTML 报告"""
    
    summary = data.get_summary()
    rankings = data.get_rankings()
    
    # Rankings 表格
    rankings_rows = ""
    for i, r in enumerate(rankings):
        rankings_rows += f"""
        <tr>
            <td>{i + 1}</td>
            <td>{r.get('agent_type', 'N/A')}</td>
            <td>{r.get('mean', 0):.4f}</td>
            <td>{r.get('std', 0):.4f}</td>
            <td>{r.get('min', 0):.4f}</td>
            <td>{r.get('max', 0):.4f}</td>
            <td>{r.get('count', 0)}</td>
        </tr>
        """
    
    # Agent 列表
    agent_options = ""
    for agent_id in data.get_all_agents():
        agent_options += f'<option value="{agent_id}">{agent_id}</option>\n'
    
    # Agent 统计卡片
    agent_stats_json = json.dumps({
        agent_id: data.get_agent_stats(agent_id).get("stats", {})
        for agent_id in data.get_all_agents()
    })
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCML Analyzer - 数据可视化</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        header p {{
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 25px;
            margin-bottom: 25px;
        }}
        .card h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .stat-box .label {{
            opacity: 0.9;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .rank-1 {{ background: linear-gradient(90deg, #ffd70020, transparent); }}
        .rank-2 {{ background: linear-gradient(90deg, #c0c0c020, transparent); }}
        .rank-3 {{ background: linear-gradient(90deg, #cd7f3220, transparent); }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin-top: 20px;
        }}
        select {{
            padding: 10px 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 1em;
            margin-right: 10px;
            cursor: pointer;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        .winner-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #ffd700, #ffb700);
            color: #333;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .agent-stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .agent-stat {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .agent-stat .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        .agent-stat .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        footer {{
            text-align: center;
            color: white;
            opacity: 0.8;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 SCML Analyzer</h1>
            <p>比赛数据可视化分析报告</p>
        </header>
        
        <!-- 摘要统计 -->
        <div class="card">
            <h2>📊 比赛概览</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{summary.get('tournament', {}).get('n_worlds_completed', 0)}</div>
                    <div class="label">完成的世界</div>
                </div>
                <div class="stat-box">
                    <div class="value">{summary.get('n_agents', 0)}</div>
                    <div class="label">参与的 Agent</div>
                </div>
                <div class="stat-box">
                    <div class="value">{len(summary.get('agent_types', []))}</div>
                    <div class="label">Agent 类型</div>
                </div>
                <div class="stat-box">
                    <div class="value">{summary.get('tournament', {}).get('duration_seconds', 0):.1f}s</div>
                    <div class="label">总耗时</div>
                </div>
            </div>
            <p><strong>🏆 冠军:</strong> 
                <span class="winner-badge">{summary.get('tournament', {}).get('winner', 'N/A')}</span>
            </p>
        </div>
        
        <!-- 排名表 -->
        <div class="card">
            <h2>🥇 Agent 排名</h2>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>Agent 类型</th>
                        <th>平均分</th>
                        <th>标准差</th>
                        <th>最低分</th>
                        <th>最高分</th>
                        <th>场次</th>
                    </tr>
                </thead>
                <tbody>
                    {rankings_rows}
                </tbody>
            </table>
        </div>
        
        <!-- 得分分布图 -->
        <div class="card">
            <h2>📈 得分分布</h2>
            <div class="chart-container">
                <canvas id="scoreChart"></canvas>
            </div>
        </div>
        
        <!-- Agent 详情 -->
        <div class="card">
            <h2>🤖 Agent 详细统计</h2>
            <div class="controls">
                <select id="agentSelect" onchange="updateAgentStats()">
                    <option value="">选择 Agent...</option>
                    {agent_options}
                </select>
            </div>
            <div id="agentStatsContainer" class="agent-stats-grid">
                <p style="color: #666;">请选择一个 Agent 查看详细统计</p>
            </div>
        </div>
        
        <!-- 时间序列图 -->
        <div class="card">
            <h2>📉 时间序列分析</h2>
            <div class="controls">
                <select id="metricSelect" onchange="updateTimeSeriesChart()">
                    <option value="balance">余额</option>
                    <option value="raw_material">原材料</option>
                    <option value="product">产品</option>
                </select>
            </div>
            <div class="chart-container">
                <canvas id="timeSeriesChart"></canvas>
            </div>
        </div>
        
        <footer>
            <p>Generated by SCML Analyzer v0.2.0</p>
        </footer>
    </div>
    
    <script>
        // 数据
        const agentStats = {agent_stats_json};
        const rankings = {json.dumps(rankings)};
        
        // 得分分布图
        const scoreCtx = document.getElementById('scoreChart').getContext('2d');
        new Chart(scoreCtx, {{
            type: 'bar',
            data: {{
                labels: rankings.map(r => r.agent_type),
                datasets: [{{
                    label: '平均分',
                    data: rankings.map(r => r.mean),
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }}, {{
                    label: '标准差',
                    data: rankings.map(r => r.std),
                    backgroundColor: 'rgba(118, 75, 162, 0.5)',
                    borderColor: 'rgba(118, 75, 162, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Agent 统计更新
        function updateAgentStats() {{
            const agentId = document.getElementById('agentSelect').value;
            const container = document.getElementById('agentStatsContainer');
            
            if (!agentId || !agentStats[agentId]) {{
                container.innerHTML = '<p style="color: #666;">请选择一个 Agent 查看详细统计</p>';
                return;
            }}
            
            const stats = agentStats[agentId];
            let html = '';
            
            const statLabels = {{
                'negotiations_started': '协商发起',
                'negotiations_success': '协商成功',
                'negotiations_failed': '协商失败',
                'contracts_signed': '签署合同',
                'contracts_breached': '违约合同',
                'offers_made': '发出报价',
                'offers_accepted': '接受报价',
                'offers_rejected': '拒绝报价',
                'production_scheduled': '计划生产',
                'production_executed': '实际生产'
            }};
            
            for (const [key, label] of Object.entries(statLabels)) {{
                const value = stats[key] || 0;
                html += `
                    <div class="agent-stat">
                        <div class="value">${{value}}</div>
                        <div class="label">${{label}}</div>
                    </div>
                `;
            }}
            
            container.innerHTML = html;
        }}
        
        // 时间序列图
        let timeSeriesChart = null;
        
        function updateTimeSeriesChart() {{
            const metric = document.getElementById('metricSelect').value;
            const ctx = document.getElementById('timeSeriesChart').getContext('2d');
            
            if (timeSeriesChart) {{
                timeSeriesChart.destroy();
            }}
            
            // 这里需要真实的时间序列数据
            // 目前使用模拟数据演示
            const labels = Array.from({{length: 30}}, (_, i) => `Day ${{i + 1}}`);
            const datasets = [];
            
            let colorIndex = 0;
            const colors = [
                'rgba(102, 126, 234, 0.8)',
                'rgba(118, 75, 162, 0.8)',
                'rgba(234, 102, 126, 0.8)',
                'rgba(126, 234, 102, 0.8)',
                'rgba(234, 206, 102, 0.8)',
            ];
            
            for (const agentId of Object.keys(agentStats).slice(0, 5)) {{
                datasets.push({{
                    label: agentId.split('@')[0],
                    data: labels.map(() => Math.random() * 1000 + 500),
                    borderColor: colors[colorIndex % colors.length],
                    fill: false,
                    tension: 0.1
                }});
                colorIndex++;
            }}
            
            timeSeriesChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: false
                        }}
                    }}
                }}
            }});
        }}
        
        // 初始化
        updateTimeSeriesChart();
    </script>
</body>
</html>
"""
    return html


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP 请求处理器"""
    
    data: VisualizerData = None
    data_dir: str = None
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html' or self.path == '/analysis_report.html':
            # 生成并返回 HTML 报告
            html = generate_html_report(self.data)
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        elif self.path == '/api/data':
            # 返回 JSON 数据
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.data.to_json().encode('utf-8'))
        elif self.path.endswith('.json') or self.path.endswith('.csv'):
            # 提供数据文件
            try:
                file_path = Path(self.data_dir) / self.path.lstrip('/')
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    content_type = 'application/json' if self.path.endswith('.json') else 'text/csv'
                    self.send_response(200)
                    self.send_header('Content-type', f'{content_type}; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                else:
                    self.send_error(404, "File not found")
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "File not found")
    
    def log_message(self, format, *args):
        # 静默日志
        pass


def start_server(data_dir: str, port: int = 8080, open_browser: bool = True):
    """
    启动可视化服务器
    
    Args:
        data_dir: 数据目录路径
        port: 服务器端口
        open_browser: 是否自动打开浏览器
    """
    # 加载数据
    data = VisualizerData(data_dir)
    data.load_all()
    
    # 配置处理器
    VisualizerHandler.data = data
    VisualizerHandler.data_dir = data_dir
    
    # 启动服务器
    server = HTTPServer(('localhost', port), VisualizerHandler)
    
    url = f"http://localhost:{port}"
    print(f"🌐 可视化服务器已启动: {url}")
    print(f"📁 数据目录: {data_dir}")
    print("按 Ctrl+C 停止服务器")
    
    if open_browser:
        webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.shutdown()


def generate_static_report(data_dir: str, output_file: str = "report.html"):
    """
    生成静态 HTML 报告文件
    
    Args:
        data_dir: 数据目录路径
        output_file: 输出文件路径
    """
    data = VisualizerData(data_dir)
    data.load_all()
    
    html = generate_html_report(data)
    
    output_path = Path(data_dir) / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📄 报告已生成: {output_path}")
    return str(output_path)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SCML Analyzer 可视化服务器')
    parser.add_argument('--data', '-d', required=True, help='数据目录路径')
    parser.add_argument('--port', '-p', type=int, default=8080, help='服务器端口')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--static', action='store_true', help='生成静态报告而非启动服务器')
    
    args = parser.parse_args()
    
    if args.static:
        generate_static_report(args.data)
    else:
        start_server(args.data, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
