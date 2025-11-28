"""
SCML Analyzer 可视化服务器

提供 Web 界面查看比赛数据分析结果。
**设计原则**: 完全独立，从 negmas tournament 目录自动提取所有数据，不依赖任何 runner。

Usage:
    # 命令行 - 只需要 negmas tournament 目录路径
    python -m scml_analyzer.visualizer --data C:\\Users\\xxx\\negmas\\tournaments\\xxx-stage-0001
    
    # Python API
    from scml_analyzer.visualizer import start_server
    start_server("C:\\Users\\xxx\\negmas\\tournaments\\xxx-stage-0001")

详细设计文档请参考: scml_analyzer/DESIGN.md
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser
import urllib.parse


def _extract_short_name(full_name: str) -> str:
    """从完整类型名提取简短名称
    
    Examples:
        "scml.oneshot.sysagents.DefaultOneShotAdapter:litaagent_std.litaagent_y.LitaAgentY"
        -> "LitaAgentY"
        
        "litaagent_std.litaagent_y.LitaAgentY"
        -> "LitaAgentY"
    """
    # 处理 Adapter 包装的情况
    if ":" in full_name:
        full_name = full_name.split(":")[-1]
    # 取最后一个点后的部分
    return full_name.split(".")[-1]


class VisualizerData:
    """
    从 negmas tournament 目录自动加载所有数据
    
    设计原则:
    - 不依赖任何 runner 传递的数据
    - 所有数据都从 negmas 生成的 CSV/JSON 文件中提取
    - 支持 negmas tournament 目录作为唯一输入
    """
    
    def __init__(self, tournament_dir: str):
        """
        Args:
            tournament_dir: negmas tournament 目录路径
                           (例如 C:\\Users\\xxx\\negmas\\tournaments\\xxx-stage-0001)
        """
        self.tournament_dir = Path(tournament_dir)
        
        # negmas 数据
        self._params: Dict = {}
        self._total_scores: List[Dict] = []
        self._winners: List[Dict] = []
        self._world_stats: List[Dict] = []
        self._score_stats: List[Dict] = []
        self._scores: List[Dict] = []
        
        # Tracker 数据
        self._tracker_data: Dict[str, Dict] = {}  # agent_id -> tracker export data
        self._tracker_summary: Dict = {}
        
        # 自动加载数据
        self.load_all()
    
    def load_all(self):
        """加载所有 negmas 数据文件"""
        self._load_params()
        self._load_total_scores()
        self._load_winners()
        self._load_world_stats()
        self._load_score_stats()
        self._load_scores()
        self._load_tracker_data()
    
    def _load_csv(self, filename: str) -> List[Dict]:
        """加载 CSV 文件"""
        path = self.tournament_dir / filename
        if not path.exists():
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception:
            return []
    
    def _load_json(self, filename: str) -> Dict:
        """加载 JSON 文件"""
        path = self.tournament_dir / filename
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _load_params(self):
        """加载 params.json"""
        self._params = self._load_json("params.json")
    
    def _load_total_scores(self):
        """加载 total_scores.csv"""
        self._total_scores = self._load_csv("total_scores.csv")
    
    def _load_winners(self):
        """加载 winners.csv"""
        self._winners = self._load_csv("winners.csv")
    
    def _load_world_stats(self):
        """加载 world_stats.csv"""
        self._world_stats = self._load_csv("world_stats.csv")
    
    def _load_score_stats(self):
        """加载 score_stats.csv"""
        self._score_stats = self._load_csv("score_stats.csv")
    
    def _load_scores(self):
        """加载 scores.csv（每个 world 每个 agent 的分数）"""
        self._scores = self._load_csv("scores.csv")
    
    def _load_tracker_data(self):
        """加载 Tracker 日志数据"""
        # 尝试多个可能的 tracker logs 位置
        tracker_dirs = [
            self.tournament_dir / "tracker_logs",
            self.tournament_dir.parent / "tracker_logs",  # tournament_history 结构
        ]
        
        tracker_dir = None
        for td in tracker_dirs:
            if td.exists() and td.is_dir():
                tracker_dir = td
                break
        
        if not tracker_dir:
            return
        
        # 加载 tracker_summary.json
        summary_path = tracker_dir / "tracker_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    self._tracker_summary = json.load(f)
            except Exception:
                pass
        
        # 加载所有 agent_*.json 文件
        for agent_file in tracker_dir.glob("agent_*.json"):
            try:
                with open(agent_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    agent_id = data.get("agent_id", agent_file.stem)
                    self._tracker_data[agent_id] = data
            except Exception:
                pass
    
    def get_tracker_stats_by_type(self, agent_type: str) -> Dict:
        """获取某个 Agent 类型的汇总 Tracker 统计"""
        stats = {
            "negotiations_started": 0,
            "negotiations_success": 0,
            "negotiations_failed": 0,
            "contracts_signed": 0,
            "contracts_breached": 0,
            "offers_made": 0,
            "offers_accepted": 0,
            "offers_rejected": 0,
            "production_scheduled": 0,
            "production_executed": 0,
        }
        
        count = 0
        for agent_id, data in self._tracker_data.items():
            if _extract_short_name(data.get("agent_type", "")) == agent_type:
                agent_stats = data.get("stats", {})
                for key in stats:
                    stats[key] += agent_stats.get(key, 0)
                count += 1
        
        return stats
    
    def get_tracker_entries_by_type(self, agent_type: str, category: str = None, limit: int = 1000) -> List[Dict]:
        """获取某个 Agent 类型的 Tracker 条目
        
        Args:
            agent_type: Agent 类型名称
            category: 过滤的类别（如 "negotiation", "contract", "inventory"）
            limit: 返回的最大条目数
        """
        entries = []
        for agent_id, data in self._tracker_data.items():
            if _extract_short_name(data.get("agent_type", "")) == agent_type:
                for entry in data.get("entries", []):
                    if category is None or entry.get("category") == category:
                        entry["agent_id"] = agent_id
                        entries.append(entry)
        
        # 按天和时间戳排序
        entries.sort(key=lambda e: (e.get("day", 0), e.get("timestamp", "")))
        return entries[:limit]
    
    def get_tracker_time_series(self, agent_type: str) -> Dict[str, List]:
        """获取某个 Agent 类型的时间序列数据（汇总）"""
        # 按天汇总所有同类型 agent 的数据
        series = {
            "raw_material": {},  # day -> [values]
            "product": {},
            "balance": {},
        }
        
        for agent_id, data in self._tracker_data.items():
            if _extract_short_name(data.get("agent_type", "")) == agent_type:
                ts = data.get("time_series", {})
                for key in series:
                    for day, value in ts.get(key, []):
                        if day not in series[key]:
                            series[key][day] = []
                        series[key][day].append(value)
        
        # 计算平均值
        result = {}
        for key in series:
            days = sorted(series[key].keys())
            result[key] = [(d, sum(series[key][d]) / len(series[key][d])) for d in days]
        
        return result
    
    def get_negotiation_details(self, agent_type: str, limit: int = 100) -> List[Dict]:
        """获取协商详情（按协商分组）"""
        # 获取所有协商相关条目
        entries = self.get_tracker_entries_by_type(agent_type, category="negotiation", limit=10000)
        
        # 按 partner + day 分组
        negotiations = {}
        for e in entries:
            partner = e.get("data", {}).get("partner", "unknown")
            day = e.get("day", 0)
            key = f"{e.get('agent_id')}_{partner}_{day}"
            
            if key not in negotiations:
                negotiations[key] = {
                    "agent_id": e.get("agent_id"),
                    "partner": partner,
                    "day": day,
                    "events": [],
                    "result": "ongoing",
                }
            
            negotiations[key]["events"].append({
                "event": e.get("event"),
                "data": e.get("data"),
                "timestamp": e.get("timestamp"),
            })
            
            # 确定结果
            if e.get("event") == "success":
                negotiations[key]["result"] = "success"
            elif e.get("event") == "failure":
                negotiations[key]["result"] = "failure"
        
        # 转换为列表并排序
        result = list(negotiations.values())
        result.sort(key=lambda n: (n["day"], n["agent_id"], n["partner"]))
        return result[:limit]
    
    def get_daily_status(self, agent_type: str) -> List[Dict]:
        """获取每日状态数据"""
        entries = self.get_tracker_entries_by_type(agent_type, limit=10000)
        daily_status = []
        
        for e in entries:
            if e.get("category") == "custom" and e.get("event") == "daily_status":
                status = {
                    "agent_id": e.get("agent_id"),
                    "day": e.get("day"),
                    **e.get("data", {})
                }
                daily_status.append(status)
        
        daily_status.sort(key=lambda s: (s.get("day", 0), s.get("agent_id", "")))
        return daily_status

    def get_summary(self) -> Dict:
        """获取比赛概览"""
        # 计算总耗时
        total_duration = 0.0
        for w in self._world_stats:
            try:
                total_duration += float(w.get("execution_time", 0))
            except (ValueError, TypeError):
                pass
        
        # 提取冠军名称
        winner_name = "N/A"
        winner_score = 0.0
        if self._winners:
            winner_name = _extract_short_name(self._winners[0].get("agent_type", "N/A"))
            try:
                winner_score = float(self._winners[0].get("score", 0))
            except (ValueError, TypeError):
                pass
        
        # 提取参赛者列表
        competitors = self._params.get("competitors", [])
        agent_types = [_extract_short_name(c) for c in competitors]
        
        return {
            "tournament": {
                "name": self._params.get("name", "Unknown"),
                "track": "oneshot" if self._params.get("oneshot_world") else "std",
                "n_configs": self._params.get("n_configs", 0),
                "n_runs_per_world": self._params.get("n_runs_per_world", 1),
                "n_steps": self._params.get("n_steps", 0),
                "n_worlds": self._params.get("n_worlds", 0),
                "n_worlds_completed": len(self._world_stats),
                "duration_seconds": total_duration,
                "winner": winner_name,
                "winner_score": winner_score,
                "parallelism": self._params.get("parallelism", "unknown"),
            },
            "n_agents": len(competitors),
            "n_worlds": len(self._world_stats),
            "agent_types": agent_types,
        }
    
    def get_rankings(self) -> List[Dict]:
        """获取排名数据（合并 total_scores 和 score_stats）"""
        rankings = []
        
        # 从 total_scores 构建基础排名
        for i, row in enumerate(self._total_scores):
            agent_type = _extract_short_name(row.get("agent_type", "Unknown"))
            try:
                score = float(row.get("score", 0))
            except (ValueError, TypeError):
                score = 0.0
            
            rankings.append({
                "rank": i + 1,
                "agent_type": agent_type,
                "score": score,
                "mean": score,  # 默认使用 total score 作为 mean
                "std": 0.0,
                "min": score,
                "max": score,
                "count": 0,
            })
        
        # 从 score_stats 补充统计数据
        stats_by_type = {}
        for row in self._score_stats:
            agent_type = _extract_short_name(row.get("agent_type", ""))
            stats_by_type[agent_type] = row
        
        for r in rankings:
            stats = stats_by_type.get(r["agent_type"], {})
            try:
                r["mean"] = float(stats.get("mean", r["mean"]))
                r["std"] = float(stats.get("std", 0))
                r["min"] = float(stats.get("min", r["min"]))
                r["max"] = float(stats.get("max", r["max"]))
                r["count"] = int(float(stats.get("count", 0)))
            except (ValueError, TypeError):
                pass
        
        return rankings
    
    def get_score_distribution(self, agent_type: str) -> List[float]:
        """获取某个 Agent 类型的分数分布"""
        scores = []
        for row in self._scores:
            row_agent_type = _extract_short_name(row.get("agent_type", ""))
            if row_agent_type == agent_type:
                try:
                    scores.append(float(row.get("score", 0)))
                except (ValueError, TypeError):
                    pass
        return scores
    
    def get_all_agents(self) -> List[str]:
        """获取所有 Agent 类型"""
        return [_extract_short_name(c) for c in self._params.get("competitors", [])]
    
    def get_agent_stats(self, agent_type: str) -> Dict:
        """获取某个 Agent 类型的统计数据（合并 score_stats 和 tracker 数据）"""
        result = {}
        
        # 从 score_stats 提取分数统计
        for row in self._score_stats:
            if _extract_short_name(row.get("agent_type", "")) == agent_type:
                result = {
                    "mean": float(row.get("mean", 0)),
                    "std": float(row.get("std", 0)),
                    "min": float(row.get("min", 0)),
                    "max": float(row.get("max", 0)),
                    "count": int(float(row.get("count", 0))),
                }
                break
        
        # 添加 Tracker 统计
        tracker_stats = self.get_tracker_stats_by_type(agent_type)
        result.update(tracker_stats)
        
        return {"stats": result}
    
    def get_world_stats(self) -> List[Dict]:
        """获取所有 world 的统计数据"""
        return self._world_stats
    
    def to_json(self) -> str:
        """导出为 JSON"""
        return json.dumps({
            "summary": self.get_summary(),
            "rankings": self.get_rankings(),
            "world_stats": self._world_stats[:100],  # 限制大小
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
    
    # Tournament path for API calls
    tournament_path_encoded = urllib.parse.quote(str(data.tournament_dir), safe='')
    
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
        .back-btn {{
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            text-decoration: none;
            display: inline-block;
            margin-bottom: 20px;
        }}
        .back-btn:hover {{
            background: rgba(255,255,255,0.3);
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← 返回比赛列表</a>
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
        
        <!-- 协商详情 -->
        <div class="card">
            <h2>🤝 协商详情</h2>
            <div class="controls">
                <select id="negotiationAgentSelect" onchange="loadNegotiationDetails()">
                    <option value="">选择 Agent...</option>
                    {agent_options}
                </select>
                <span id="negotiationCount" style="margin-left: 15px; color: #666;"></span>
            </div>
            <div id="negotiationContainer" style="max-height: 500px; overflow-y: auto;">
                <p style="color: #666;">请选择一个 Agent 查看协商详情</p>
            </div>
        </div>
        
        <!-- 每日状态 -->
        <div class="card">
            <h2>📅 每日状态</h2>
            <div class="controls">
                <select id="dailyAgentSelect" onchange="loadDailyStatus()">
                    <option value="">选择 Agent...</option>
                    {agent_options}
                </select>
            </div>
            <div id="dailyStatusContainer" style="max-height: 500px; overflow-y: auto;">
                <p style="color: #666;">请选择一个 Agent 查看每日状态</p>
            </div>
            <div class="chart-container" style="margin-top: 20px;">
                <canvas id="dailyChart"></canvas>
            </div>
        </div>
        
        <footer>
            <p>Generated by SCML Analyzer v0.3.0</p>
        </footer>
    </div>
    
    <script>
        // 数据
        const agentStats = {agent_stats_json};
        const rankings = {json.dumps(rankings)};
        const tournamentPath = "{tournament_path_encoded}";
        
        // API 请求辅助函数
        function apiUrl(endpoint) {{
            // 如果有 tournament path，添加为查询参数
            if (tournamentPath) {{
                return `${{endpoint}}?path=${{tournamentPath}}`;
            }}
            return endpoint;
        }}
        
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
            
            // 分数统计 (来自 score_stats.csv)
            const scoreLabels = {{
                'mean': '平均分',
                'std': '标准差',
                'min': '最低分',
                'max': '最高分',
                'count': '参赛场次'
            }};
            
            for (const [key, label] of Object.entries(scoreLabels)) {{
                const value = stats[key];
                if (value !== undefined) {{
                    const displayValue = key === 'count' ? value : value.toFixed(4);
                    html += `
                        <div class="agent-stat">
                            <div class="value">${{displayValue}}</div>
                            <div class="label">${{label}}</div>
                        </div>
                    `;
                }}
            }}
            
            // 如果有 tracker 数据中的其他统计
            const trackerLabels = {{
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
            
            let hasTrackerData = false;
            for (const key of Object.keys(trackerLabels)) {{
                if (stats[key] !== undefined && stats[key] > 0) {{
                    hasTrackerData = true;
                    break;
                }}
            }}
            
            if (hasTrackerData) {{
                html += '<div style="grid-column: 1/-1; border-top: 1px solid #eee; margin-top: 15px; padding-top: 15px;"><strong>Tracker 数据</strong></div>';
                for (const [key, label] of Object.entries(trackerLabels)) {{
                    const value = stats[key] || 0;
                    if (value > 0) {{
                        html += `
                            <div class="agent-stat">
                                <div class="value">${{value}}</div>
                                <div class="label">${{label}}</div>
                            </div>
                        `;
                    }}
                }}
            }}
            
            container.innerHTML = html || '<p style="color: #666;">暂无详细统计数据</p>';
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
        
        // 加载协商详情
        async function loadNegotiationDetails() {{
            const agentType = document.getElementById('negotiationAgentSelect').value;
            const container = document.getElementById('negotiationContainer');
            const countSpan = document.getElementById('negotiationCount');
            
            if (!agentType) {{
                container.innerHTML = '<p style="color: #666;">请选择一个 Agent 查看协商详情</p>';
                countSpan.textContent = '';
                return;
            }}
            
            container.innerHTML = '<p style="color: #666;">加载中...</p>';
            
            try {{
                const response = await fetch(apiUrl(`/api/negotiations/${{encodeURIComponent(agentType)}}`));
                const negotiations = await response.json();
                
                countSpan.textContent = `共 ${{negotiations.length}} 次协商`;
                
                if (negotiations.length === 0) {{
                    container.innerHTML = '<p style="color: #666;">暂无协商数据（需要 Tracker 日志）</p>';
                    return;
                }}
                
                // 统计信息
                const successCount = negotiations.filter(n => n.result === 'success').length;
                const failCount = negotiations.filter(n => n.result === 'failure').length;
                const hasOffers = negotiations.some(n => n.events.some(e => e.event === 'offer_made' || e.event === 'offer_received'));
                
                let html = `<div style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px;">
                    <strong>统计：</strong> 
                    <span style="color: #28a745;">✓ 成功 ${{successCount}}</span> | 
                    <span style="color: #dc3545;">✗ 失败 ${{failCount}}</span> | 
                    成功率 ${{(successCount / negotiations.length * 100).toFixed(1)}}%
                    ${{hasOffers ? '' : '<br><small style="color: #999;">⚠️ 旧版 Tracker 未记录出价过程，运行新比赛可获得完整数据</small>'}}
                </div>`;
                
                html += '<table style="width:100%; font-size: 0.85em;"><thead><tr>' +
                    '<th>Day</th><th>Partner</th><th>结果</th><th>事件数</th><th>最终协议/报价详情</th>' +
                    '</tr></thead><tbody>';
                
                for (const neg of negotiations.slice(0, 100)) {{
                    const resultClass = neg.result === 'success' ? 'color: #28a745;' : 
                                       neg.result === 'failure' ? 'color: #dc3545;' : 'color: #ffc107;';
                    const resultText = neg.result === 'success' ? '✓ 成功' : 
                                      neg.result === 'failure' ? '✗ 失败' : '⋯ 进行中';
                    
                    // 提取详情
                    let detailsHtml = '';
                    for (const event of neg.events) {{
                        const data = event.data || {{}};
                        if (event.event === 'success') {{
                            const agreement = data.agreement || {{}};
                            detailsHtml += `<div style="font-size: 0.85em; color: #28a745; font-weight: bold;">` +
                                `协议: Q=${{agreement.quantity || 'N/A'}}, P=${{agreement.price || 'N/A'}}</div>`;
                        }} else if (event.event === 'offer_received') {{
                            detailsHtml += `<div style="font-size: 0.8em; color: #666;">` +
                                `← R${{data.round || '?'}}: Q=${{data.quantity || 'N/A'}}, P=${{data.unit_price || 'N/A'}}, D=${{data.delivery_day || 'N/A'}}</div>`;
                        }} else if (event.event === 'offer_made') {{
                            detailsHtml += `<div style="font-size: 0.8em; color: #007bff;">` +
                                `→ R${{data.round || '?'}}: Q=${{data.quantity || 'N/A'}}, P=${{data.unit_price || 'N/A'}}, D=${{data.delivery_day || 'N/A'}}</div>`;
                        }} else if (event.event === 'started') {{
                            detailsHtml += `<div style="font-size: 0.8em; color: #17a2b8;">开始协商</div>`;
                        }}
                    }}
                    if (!detailsHtml && neg.result === 'failure') {{
                        detailsHtml = '<span style="color: #999;">无协议达成</span>';
                    }}
                    
                    html += `<tr>
                        <td>${{neg.day}}</td>
                        <td style="font-size: 0.8em;">${{neg.partner.substring(0, 20)}}</td>
                        <td style="${{resultClass}}">${{resultText}}</td>
                        <td>${{neg.events.length}}</td>
                        <td>${{detailsHtml || 'N/A'}}</td>
                    </tr>`;
                }}
                
                html += '</tbody></table>';
                if (negotiations.length > 100) {{
                    html += `<p style="color: #999; text-align: center; margin-top: 10px;">显示前 100 条，共 ${{negotiations.length}} 条</p>`;
                }}
                container.innerHTML = html;
            }} catch (error) {{
                container.innerHTML = `<p style="color: #dc3545;">加载失败: ${{error.message}}</p>`;
            }}
        }}
        
        // 每日状态图表
        let dailyChart = null;
        
        // 加载每日状态
        async function loadDailyStatus() {{
            const agentType = document.getElementById('dailyAgentSelect').value;
            const container = document.getElementById('dailyStatusContainer');
            
            if (!agentType) {{
                container.innerHTML = '<p style="color: #666;">请选择一个 Agent 查看每日状态</p>';
                if (dailyChart) {{ dailyChart.destroy(); dailyChart = null; }}
                return;
            }}
            
            container.innerHTML = '<p style="color: #666;">加载中...</p>';
            
            try {{
                const response = await fetch(apiUrl(`/api/daily_status/${{encodeURIComponent(agentType)}}`));
                const dailyStatus = await response.json();
                
                if (dailyStatus.length === 0) {{
                    container.innerHTML = '<p style="color: #666;">暂无每日状态数据（需要 Tracker 日志）</p>';
                    if (dailyChart) {{ dailyChart.destroy(); dailyChart = null; }}
                    return;
                }}
                
                // 按天汇总数据 - 包含所有字段
                const dayData = {{}};
                for (const status of dailyStatus) {{
                    const day = status.day;
                    if (!dayData[day]) {{
                        dayData[day] = {{ 
                            count: 0, 
                            balance: 0, 
                            score: 0, 
                            disposal_cost: 0, 
                            shortfall_penalty: 0, 
                            storage_cost: 0,
                            exo_input_qty: 0,
                            exo_input_price: 0,
                            exo_output_qty: 0,
                            exo_output_price: 0,
                            needed_supplies: 0,
                            needed_sales: 0,
                            total_supplies: 0,
                            total_sales: 0,
                            n_lines: 0,
                        }};
                    }}
                    dayData[day].count++;
                    dayData[day].balance += status.balance || 0;
                    dayData[day].score += status.score || 0;
                    dayData[day].disposal_cost += status.disposal_cost || 0;
                    dayData[day].shortfall_penalty += status.shortfall_penalty || 0;
                    dayData[day].storage_cost += status.storage_cost || 0;
                    dayData[day].exo_input_qty += status.exo_input_qty || 0;
                    dayData[day].exo_input_price += status.exo_input_price || 0;
                    dayData[day].exo_output_qty += status.exo_output_qty || 0;
                    dayData[day].exo_output_price += status.exo_output_price || 0;
                    dayData[day].needed_supplies += status.needed_supplies || 0;
                    dayData[day].needed_sales += status.needed_sales || 0;
                    dayData[day].total_supplies += status.total_supplies || 0;
                    dayData[day].total_sales += status.total_sales || 0;
                    dayData[day].n_lines += status.n_lines || 0;
                }}
                
                // 表格 - 显示所有字段
                const days = Object.keys(dayData).sort((a, b) => parseInt(a) - parseInt(b));
                let html = `
                <div style="overflow-x: auto;">
                <table style="width:100%; font-size: 0.75em; white-space: nowrap;">
                <thead><tr>
                    <th>Day</th>
                    <th>Agents</th>
                    <th>平均分</th>
                    <th>平均余额</th>
                    <th>外生输入量</th>
                    <th>外生输入价</th>
                    <th>外生输出量</th>
                    <th>外生输出价</th>
                    <th>需求采购</th>
                    <th>需求销售</th>
                    <th>已签采购</th>
                    <th>已签销售</th>
                    <th>处置成本</th>
                    <th>短缺惩罚</th>
                    <th>存储成本</th>
                    <th>产线数</th>
                </tr></thead><tbody>`;
                
                for (const day of days.slice(0, 50)) {{
                    const d = dayData[day];
                    const c = d.count;
                    html += `<tr>
                        <td>${{day}}</td>
                        <td>${{c}}</td>
                        <td>${{(d.score / c).toFixed(4)}}</td>
                        <td>${{(d.balance / c).toFixed(0)}}</td>
                        <td>${{(d.exo_input_qty / c).toFixed(1)}}</td>
                        <td>${{(d.exo_input_price / c).toFixed(0)}}</td>
                        <td>${{(d.exo_output_qty / c).toFixed(1)}}</td>
                        <td>${{(d.exo_output_price / c).toFixed(0)}}</td>
                        <td>${{(d.needed_supplies / c).toFixed(1)}}</td>
                        <td>${{(d.needed_sales / c).toFixed(1)}}</td>
                        <td>${{(d.total_supplies / c).toFixed(1)}}</td>
                        <td>${{(d.total_sales / c).toFixed(1)}}</td>
                        <td>${{(d.disposal_cost / c).toFixed(3)}}</td>
                        <td>${{(d.shortfall_penalty / c).toFixed(3)}}</td>
                        <td>${{(d.storage_cost / c).toFixed(3)}}</td>
                        <td>${{(d.n_lines / c).toFixed(0)}}</td>
                    </tr>`;
                }}
                
                html += '</tbody></table></div>';
                if (days.length > 50) {{
                    html += `<p style="color: #999; text-align: center; margin-top: 10px;">显示前 50 天</p>`;
                }}
                container.innerHTML = html;
                
                // 绘制图表
                const ctx = document.getElementById('dailyChart').getContext('2d');
                if (dailyChart) {{ dailyChart.destroy(); }}
                
                dailyChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: days,
                        datasets: [{{
                            label: '平均分数',
                            data: days.map(d => dayData[d].score / dayData[d].count),
                            borderColor: 'rgba(102, 126, 234, 1)',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.1,
                            yAxisID: 'y'
                        }}, {{
                            label: '平均余额',
                            data: days.map(d => dayData[d].balance / dayData[d].count),
                            borderColor: 'rgba(118, 75, 162, 1)',
                            backgroundColor: 'rgba(118, 75, 162, 0.1)',
                            fill: false,
                            tension: 0.1,
                            yAxisID: 'y1'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{
                            mode: 'index',
                            intersect: false,
                        }},
                        scales: {{
                            y: {{
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: {{ display: true, text: '分数' }}
                            }},
                            y1: {{
                                type: 'linear',
                                display: true,
                                position: 'right',
                                title: {{ display: true, text: '余额' }},
                                grid: {{ drawOnChartArea: false }}
                            }}
                        }}
                    }}
                }});
            }} catch (error) {{
                container.innerHTML = `<p style="color: #dc3545;">加载失败: ${{error.message}}</p>`;
            }}
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
        elif self.path.startswith('/api/negotiations/'):
            # 返回协商详情 /api/negotiations/{agent_type}
            agent_type = urllib.parse.unquote(self.path.split('/')[-1])
            negotiations = self.data.get_negotiation_details(agent_type)
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(negotiations, ensure_ascii=False).encode('utf-8'))
        elif self.path.startswith('/api/daily_status/'):
            # 返回每日状态 /api/daily_status/{agent_type}
            agent_type = urllib.parse.unquote(self.path.split('/')[-1])
            daily_status = self.data.get_daily_status(agent_type)
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(daily_status, ensure_ascii=False).encode('utf-8'))
        elif self.path.startswith('/api/time_series/'):
            # 返回时间序列 /api/time_series/{agent_type}
            agent_type = urllib.parse.unquote(self.path.split('/')[-1])
            time_series = self.data.get_tracker_time_series(agent_type)
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(time_series, ensure_ascii=False).encode('utf-8'))
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
