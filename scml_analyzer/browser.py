"""
SCML Analyzer 比赛浏览器

提供一个 Web 界面让用户选择并查看历史比赛数据。

Usage:
    # 命令行 - 优先从 tournament_history 读取
    python -m scml_analyzer.browser
    
    # 指定扫描 negmas 目录（直接扫描原始数据）
    python -m scml_analyzer.browser --mode negmas
    
    # 导入所有比赛到 tournament_history
    python -m scml_analyzer.browser --import-all

数据源：
    - history 模式（默认）: 从 project/tournament_history/ 读取已导入的比赛
    - negmas 模式: 直接扫描 ~/negmas/tournaments/
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser
from datetime import datetime
from urllib.parse import quote, unquote

from .visualizer import VisualizerData, _extract_short_name
from .history import (
    get_history_dir, 
    list_tournaments as list_history_tournaments,
    scan_and_import_all,
    auto_import_tournament,
)


def get_default_tournaments_dir() -> Path:
    """获取默认的 negmas tournaments 目录"""
    # Windows: C:\Users\xxx\negmas\tournaments
    # Linux/Mac: ~/negmas/tournaments
    home = Path.home()
    return home / "negmas" / "tournaments"


def scan_tournaments(tournaments_dir: Path) -> List[Dict]:
    """扫描所有可用的比赛"""
    tournaments = []
    
    if not tournaments_dir.exists():
        return tournaments
    
    for item in tournaments_dir.iterdir():
        if not item.is_dir():
            continue
        
        # 检查是否是有效的 tournament 目录
        params_file = item / "params.json"
        if not params_file.exists():
            continue
        
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            
            # 加载 winners 和 world_stats
            winners = []
            winners_file = item / "winners.csv"
            if winners_file.exists():
                import csv
                with open(winners_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    winners = list(reader)
            
            # 统计完成的 world 数
            world_stats_file = item / "world_stats.csv"
            n_completed = 0
            total_duration = 0.0
            if world_stats_file.exists():
                import csv
                with open(world_stats_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        n_completed += 1
                        try:
                            total_duration += float(row.get("execution_time", 0))
                        except (ValueError, TypeError):
                            pass
            
            # 提取比赛信息
            tournament_info = {
                "path": str(item),
                "name": params.get("name", item.name),
                "track": "oneshot" if params.get("oneshot_world") else "std",
                "n_configs": params.get("n_configs", 0),
                "n_steps": params.get("n_steps", 0),
                "n_worlds": params.get("n_worlds", 0),
                "n_completed": n_completed,
                "duration_seconds": total_duration,
                "parallelism": params.get("parallelism", "unknown"),
                "competitors": [_extract_short_name(c) for c in params.get("competitors", [])],
                "winner": _extract_short_name(winners[0]["agent_type"]) if winners else "N/A",
                "winner_score": float(winners[0]["score"]) if winners else 0,
                # 从目录名提取时间戳
                "timestamp": _extract_timestamp(item.name),
            }
            
            tournaments.append(tournament_info)
            
        except Exception as e:
            # 跳过无效的目录
            continue
    
    # 按时间戳排序（最新的在前）
    tournaments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return tournaments


def _extract_timestamp(name: str) -> str:
    """从 tournament 目录名提取时间戳"""
    # 目录名格式: 20251128H130949613919Kqg-stage-0001
    # 提取: 2025-11-28 13:09:49
    try:
        if len(name) >= 14:
            date_part = name[:8]  # 20251128
            time_part = name[9:15]  # 130949
            return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
    except:
        pass
    return ""


def get_tournaments_from_history() -> List[Dict]:
    """从 tournament_history 目录获取比赛列表"""
    history_tournaments = list_history_tournaments()
    
    # 转换为浏览器需要的格式
    tournaments = []
    for t in history_tournaments:
        results = t.get("results", {})
        settings = t.get("settings", {})
        tournaments.append({
            "path": t.get("path", ""),
            "name": t.get("id", ""),
            "track": t.get("track", "unknown"),
            "n_configs": settings.get("n_configs", 0),
            "n_steps": settings.get("n_steps", 0),
            "n_worlds": settings.get("n_worlds", 0),
            "n_completed": results.get("n_completed", 0),
            "duration_seconds": results.get("total_duration_seconds", 0),
            "parallelism": settings.get("parallelism", "unknown"),
            "competitors": t.get("competitors", []),
            "winner": results.get("winner", "N/A") or "N/A",
            "winner_score": results.get("winner_score", 0) or 0,
            "timestamp": t.get("timestamp", ""),
            "has_tracker": (Path(t.get("path", "")) / "tracker_logs").exists() if t.get("path") else False,
        })
    
    return tournaments


def generate_browser_html(tournaments: List[Dict], data_source: str, source_path: str) -> str:
    """生成比赛浏览器 HTML
    
    Args:
        tournaments: 比赛列表
        data_source: 数据源类型 ("history" 或 "negmas")
        source_path: 数据源路径
    """
    
    # 生成比赛列表
    tournament_rows = ""
    for t in tournaments:
        status = "✅" if t["n_completed"] == t["n_worlds"] else f"⚠️ {t['n_completed']}/{t['n_worlds']}"
        duration_str = f"{t['duration_seconds']:.1f}s" if t['duration_seconds'] > 0 else "-"
        
        # 正确编码路径用于 URL
        encoded_path = quote(t['path'], safe='')
        
        # 检查是否有 tracker 数据
        tracker_badge = "📊" if t.get("has_tracker") else ""
        
        tournament_rows += f"""
        <tr onclick="window.location.href='/view?path={encoded_path}'" style="cursor: pointer;">
            <td>{t['timestamp']}</td>
            <td><span class="track-badge track-{t['track']}">{t['track'].upper()}</span></td>
            <td>{', '.join(t['competitors'][:4])}{'...' if len(t['competitors']) > 4 else ''}</td>
            <td>{t['n_configs']}</td>
            <td>{t['n_steps']}</td>
            <td>{status}</td>
            <td>{duration_str}</td>
            <td><span class="winner-badge">{t['winner']}</span> ({t['winner_score']:.3f}) {tracker_badge}</td>
        </tr>
        """
    
    # 数据源信息
    source_info = {
        "history": ("📁 数据源: tournament_history (已整合)", "success"),
        "negmas": ("📁 数据源: negmas/tournaments (原始数据)", "warning"),
    }
    source_text, source_class = source_info.get(data_source, ("📁 数据源: 未知", ""))
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCML Analyzer - 比赛浏览器</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: white;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        header p {{
            opacity: 0.8;
        }}
        .info-bar {{
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .info-bar .path {{
            font-family: monospace;
            background: rgba(0,0,0,0.3);
            padding: 5px 10px;
            border-radius: 4px;
        }}
        .card {{
            background: rgba(255,255,255,0.95);
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 25px;
            margin-bottom: 25px;
            color: #333;
        }}
        .card h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
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
            background: #f0f4ff;
        }}
        .track-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .track-oneshot {{
            background: #e3f2fd;
            color: #1565c0;
        }}
        .track-std {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}
        .winner-badge {{
            background: linear-gradient(135deg, #ffd700, #ffb347);
            color: #333;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }}
        .empty-state h3 {{
            margin-bottom: 10px;
        }}
        footer {{
            text-align: center;
            opacity: 0.7;
            margin-top: 30px;
            padding: 20px;
        }}
        .refresh-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
        }}
        .refresh-btn:hover {{
            background: #5a6fd6;
        }}
        .import-btn {{
            background: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin-left: 10px;
        }}
        .import-btn:hover {{
            background: #218838;
        }}
        .source-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }}
        .source-history {{
            background: #d4edda;
            color: #155724;
        }}
        .source-negmas {{
            background: #fff3cd;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 SCML Analyzer</h1>
            <p>比赛数据浏览器</p>
        </header>
        
        <div class="info-bar">
            <div>
                <span class="source-badge source-{data_source}">{source_text}</span>
                <span class="path" style="margin-left: 10px;">{source_path}</span>
            </div>
            <div>
                <strong>找到 {len(tournaments)} 场比赛</strong>
                <button class="refresh-btn" onclick="location.reload()">🔄 刷新</button>
                <button class="import-btn" onclick="window.location.href='/import-all'">📥 导入全部</button>
            </div>
        </div>
        
        <div class="card">
            <h2>📋 比赛列表</h2>
            {"<div class='empty-state'><h3>暂无比赛数据</h3><p>运行比赛后使用导入全部按钮导入数据</p></div>" if not tournaments else f'''
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>赛道</th>
                        <th>参赛者</th>
                        <th>配置数</th>
                        <th>步数</th>
                        <th>完成状态</th>
                        <th>耗时</th>
                        <th>冠军</th>
                    </tr>
                </thead>
                <tbody>
                    {tournament_rows}
                </tbody>
            </table>
            '''}
        </div>
        
        <footer>
            <p>SCML Analyzer v0.2.0 | 点击任意比赛行查看详情 | 📊 = 含 Tracker 数据</p>
        </footer>
    </div>
</body>
</html>
"""
    return html


class BrowserHandler(SimpleHTTPRequestHandler):
    """比赛浏览器 HTTP 处理器"""
    
    data_source: str = "history"  # "history" 或 "negmas"
    source_path: str = ""
    tournaments_dir: Optional[Path] = None  # 仅 negmas 模式使用
    tournaments: List[Dict] = []
    
    def _refresh_tournaments(self):
        """刷新比赛列表"""
        if BrowserHandler.data_source == "history":
            BrowserHandler.tournaments = get_tournaments_from_history()
        elif BrowserHandler.tournaments_dir:
            BrowserHandler.tournaments = scan_tournaments(BrowserHandler.tournaments_dir)
    
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        
        parsed = urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "":
            # 主页 - 比赛列表
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            
            # 重新扫描（支持刷新）
            self._refresh_tournaments()
            html = generate_browser_html(
                BrowserHandler.tournaments, 
                BrowserHandler.data_source,
                BrowserHandler.source_path
            )
            self.wfile.write(html.encode('utf-8'))
            
        elif parsed.path == "/import-all":
            # 导入所有比赛
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            
            try:
                imported = scan_and_import_all()
                msg = f"成功导入 {len(imported)} 场比赛" if imported else "没有新的比赛需要导入"
                html = f"""<html><body style="font-family: sans-serif; padding: 40px; text-align: center;">
                <h1>✅ {msg}</h1>
                <p><a href="/">返回列表</a></p>
                </body></html>"""
            except Exception as e:
                html = f"""<html><body style="font-family: sans-serif; padding: 40px; text-align: center;">
                <h1>❌ 导入失败</h1>
                <p>{e}</p>
                <p><a href="/">返回列表</a></p>
                </body></html>"""
            
            self.wfile.write(html.encode('utf-8'))
            
        elif parsed.path == "/view":
            # 查看特定比赛
            params = parse_qs(parsed.query)
            tournament_path_encoded = params.get("path", [None])[0]
            
            if tournament_path_encoded:
                # URL 解码路径
                tournament_path = unquote(tournament_path_encoded)
                
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                
                try:
                    from .visualizer import VisualizerData, generate_html_report
                    data = VisualizerData(tournament_path)
                    html = generate_html_report(data)
                    self.wfile.write(html.encode('utf-8'))
                except Exception as e:
                    import traceback
                    error_html = f"""<html><body>
                    <h1>Error loading tournament</h1>
                    <p><strong>Path:</strong> {tournament_path}</p>
                    <p><strong>Error:</strong> {e}</p>
                    <pre>{traceback.format_exc()}</pre>
                    <p><a href="/">返回列表</a></p>
                    </body></html>"""
                    self.wfile.write(error_html.encode('utf-8'))
            else:
                self.send_error(400, "Missing path parameter")
        
        elif parsed.path.startswith("/api/"):
            # API 端点 - 需要 tournament_path 参数
            params = parse_qs(parsed.query)
            tournament_path_encoded = params.get("path", [None])[0]
            
            if not tournament_path_encoded:
                self.send_response(400)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing path parameter"}).encode('utf-8'))
                return
            
            tournament_path = unquote(tournament_path_encoded)
            
            try:
                from .visualizer import VisualizerData
                data = VisualizerData(tournament_path)
                
                # 处理不同的 API 端点
                if parsed.path.startswith("/api/negotiations/"):
                    agent_type = unquote(parsed.path.split("/")[-1])
                    result = data.get_negotiation_details(agent_type)
                elif parsed.path.startswith("/api/daily_status/"):
                    agent_type = unquote(parsed.path.split("/")[-1])
                    result = data.get_daily_status(agent_type)
                elif parsed.path.startswith("/api/time_series/"):
                    agent_type = unquote(parsed.path.split("/")[-1])
                    result = data.get_tracker_time_series(agent_type)
                elif parsed.path == "/api/data":
                    result = json.loads(data.to_json())
                else:
                    self.send_response(404)
                    self.send_header("Content-type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Unknown API endpoint"}).encode('utf-8'))
                    return
                
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                import traceback
                self.send_response(500)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }, ensure_ascii=False).encode('utf-8'))
        
        else:
            self.send_error(404, "Not Found")
    
    def log_message(self, format, *args):
        """禁止打印每个请求"""
        pass


def start_browser(
    mode: str = "history",
    tournaments_dir: Optional[str] = None, 
    port: int = 8080, 
    open_browser: bool = True
):
    """
    启动比赛浏览器服务器
    
    Args:
        mode: 数据源模式 ("history" 或 "negmas")
        tournaments_dir: negmas tournaments 目录路径（仅 negmas 模式使用）
        port: 服务器端口
        open_browser: 是否自动打开浏览器
    """
    BrowserHandler.data_source = mode
    
    if mode == "history":
        # 从 tournament_history 读取
        history_path = get_history_dir()
        BrowserHandler.source_path = str(history_path)
        BrowserHandler.tournaments = get_tournaments_from_history()
        print(f"📁 数据源: tournament_history")
        print(f"📂 路径: {history_path}")
    else:
        # 直接扫描 negmas tournaments
        tournaments_path: Path
        if tournaments_dir is None:
            tournaments_path = get_default_tournaments_dir()
        else:
            tournaments_path = Path(tournaments_dir)
        
        BrowserHandler.tournaments_dir = tournaments_path
        BrowserHandler.source_path = str(tournaments_path)
        BrowserHandler.tournaments = scan_tournaments(tournaments_path)
        print(f"📁 数据源: negmas/tournaments (原始数据)")
        print(f"📂 路径: {tournaments_path}")
    
    server = HTTPServer(("localhost", port), BrowserHandler)
    
    print(f"🌐 比赛浏览器已启动: http://localhost:{port}")
    print(f"📊 找到 {len(BrowserHandler.tournaments)} 场比赛")
    print("按 Ctrl+C 停止服务器")
    
    if open_browser:
        webbrowser.open(f"http://localhost:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="SCML 比赛数据浏览器"
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="history",
        choices=["history", "negmas"],
        help="数据源模式: history (从 tournament_history 读取, 默认) 或 negmas (直接扫描原始数据)"
    )
    parser.add_argument(
        "--tournaments-dir", "-d", type=str, default=None,
        help="negmas tournaments 目录路径 (仅 negmas 模式有效)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="服务器端口 (默认: 8080)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="不自动打开浏览器"
    )
    parser.add_argument(
        "--import-all", action="store_true",
        help="导入所有比赛后启动浏览器"
    )
    
    args = parser.parse_args()
    
    # 如果指定了 --import-all，先导入所有比赛
    if args.import_all:
        print("📥 正在导入所有比赛...")
        imported = scan_and_import_all()
        print(f"✅ 导入完成，共 {len(imported)} 场新比赛")
    
    start_browser(
        mode=args.mode,
        tournaments_dir=args.tournaments_dir,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
