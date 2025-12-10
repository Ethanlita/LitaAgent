"""
SCML Analyzer 可视化服务器

提供 Web 界面查看比赛数据分析结果。
**设计原则**: 
- 完全独立，自动从 tournament_history 目录读取比赛数据
- 不需要任何参数，自动发现比赛列表
- 用户可以从列表选择查看具体比赛

Usage:
    # 命令行 - 不需要任何参数！
    python -m scml_analyzer.visualizer
    
    # Python API - 不需要任何参数！
    from scml_analyzer.visualizer import start_server
    start_server()

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

# 导入 history 模块
from . import history


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
    short = full_name.split(".")[-1]
    # 去掉 Tracked 后缀，避免可视化侧命名不一致
    if short.endswith("Tracked"):
        short = short[:-7] or short
    return short


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

    # ========== 单 World 模式相关方法 ==========
    
    def get_available_worlds_from_tracker(self) -> List[Dict]:
        """从 Tracker 数据中获取所有可用的 World 列表
        
        Returns:
            [{"world_id": "...", "agents": [{"agent_id": "00LY@0", "agent_type": "LitaAgentY", "level": 0}, ...]}]
        """
        from collections import defaultdict
        
        worlds = defaultdict(list)
        
        for agent_id, data in self._tracker_data.items():
            world_id = data.get("world_id", "unknown")
            agent_type = _extract_short_name(data.get("agent_type", "Unknown"))
            
            # 解析 level（格式: 00LY@0 -> level 0）
            level = None
            if "@" in agent_id:
                try:
                    level = int(agent_id.split("@")[-1])
                except ValueError:
                    pass
            
            worlds[world_id].append({
                "agent_id": agent_id,
                "agent_type": agent_type,
                "level": level,
                "display_name": f"{agent_id} ({agent_type})" if level is None else f"{agent_id} ({agent_type}) L{level}",
            })
        
        result = []
        for world_id, agents in worlds.items():
            # 按 level 排序
            agents.sort(key=lambda a: (a["level"] if a["level"] is not None else 999, a["agent_id"]))
            result.append({
                "world_id": world_id,
                "agents": agents,
                "agent_count": len(agents),
            })
        
        result.sort(key=lambda w: w["world_id"])
        return result
    
    def get_agent_instances(self) -> List[Dict]:
        """获取所有 Agent 实例的列表（包含 ID、类型、层级、World）
        
        Returns:
            [{"agent_id": "00LY@0", "agent_type": "LitaAgentY", "level": 0, "world_id": "...", "display_name": "00LY@0 (LitaAgentY)"}]
        """
        instances = []
        
        for agent_id, data in self._tracker_data.items():
            agent_type = _extract_short_name(data.get("agent_type", "Unknown"))
            world_id = data.get("world_id", "unknown")
            
            # 解析 level
            level = None
            if "@" in agent_id:
                try:
                    level = int(agent_id.split("@")[-1])
                except ValueError:
                    pass
            
            instances.append({
                "agent_id": agent_id,
                "agent_type": agent_type,
                "level": level,
                "world_id": world_id,
                "display_name": f"{agent_id} ({agent_type})",
            })
        
        # 按 agent_type, world_id, level 排序
        instances.sort(key=lambda a: (a["agent_type"], a["world_id"], a["level"] if a["level"] is not None else 999))
        return instances
    
    def get_single_agent_data(self, agent_id: str) -> Dict:
        """获取单个 Agent 实例的完整数据
        
        Args:
            agent_id: Agent 实例 ID（如 "00LY@0"）
            
        Returns:
            完整的 tracker 数据，包括 entries, time_series, stats 等
        """
        data = self._tracker_data.get(agent_id, {})
        if not data:
            # 尝试匹配部分 ID（文件名可能被截断）
            for aid, adata in self._tracker_data.items():
                if agent_id in aid or aid in agent_id:
                    data = adata
                    break
        
        if not data:
            return {}
        
        agent_type = _extract_short_name(data.get("agent_type", "Unknown"))
        
        return {
            "agent_id": data.get("agent_id", agent_id),
            "agent_type": agent_type,
            "world_id": data.get("world_id", "unknown"),
            "stats": data.get("stats", {}),
            "time_series": data.get("time_series", {}),
            "entries": data.get("entries", []),
            "entry_count": len(data.get("entries", [])),
        }
    
    def get_single_world_data(self, world_id: str) -> Dict:
        """获取单个 World 的完整数据
        
        Args:
            world_id: World ID
            
        Returns:
            {
                "world_id": "...",
                "agents": [...],
                "aggregated_stats": {...},
                "all_entries": [...],
                "time_series_by_agent": {...},
            }
        """
        agents = []
        all_entries = []
        time_series_by_agent = {}
        aggregated_stats = {
            "negotiations_started": 0,
            "negotiations_success": 0,
            "negotiations_failed": 0,
            "contracts_signed": 0,
            "offers_made": 0,
            "offers_accepted": 0,
            "offers_rejected": 0,
        }
        
        for agent_id, data in self._tracker_data.items():
            if data.get("world_id") != world_id:
                continue
            
            agent_type = _extract_short_name(data.get("agent_type", "Unknown"))
            level = None
            if "@" in agent_id:
                try:
                    level = int(agent_id.split("@")[-1])
                except ValueError:
                    pass
            
            agents.append({
                "agent_id": agent_id,
                "agent_type": agent_type,
                "level": level,
                "stats": data.get("stats", {}),
            })
            
            # 聚合统计
            for key in aggregated_stats:
                aggregated_stats[key] += data.get("stats", {}).get(key, 0)
            
            # 添加 entries（附加 agent_id）
            for entry in data.get("entries", []):
                entry_copy = dict(entry)
                entry_copy["agent_id"] = agent_id
                entry_copy["agent_type"] = agent_type
                all_entries.append(entry_copy)
            
            # 时间序列
            time_series_by_agent[agent_id] = data.get("time_series", {})
        
        # 按天和时间戳排序 entries
        all_entries.sort(key=lambda e: (e.get("day", 0), e.get("timestamp", "")))
        
        # 按 level 排序 agents
        agents.sort(key=lambda a: (a["level"] if a["level"] is not None else 999, a["agent_id"]))
        
        return {
            "world_id": world_id,
            "agents": agents,
            "agent_count": len(agents),
            "aggregated_stats": aggregated_stats,
            "all_entries": all_entries,
            "entry_count": len(all_entries),
            "time_series_by_agent": time_series_by_agent,
        }
    
    def get_single_world_negotiations(self, world_id: str) -> List[Dict]:
        """获取单个 World 的所有协商详情
        
        Args:
            world_id: World ID
            
        Returns:
            协商列表，包含所有参与者的出价记录
        """
        negotiations = {}
        
        for agent_id, data in self._tracker_data.items():
            if data.get("world_id") != world_id:
                continue
            
            agent_type = _extract_short_name(data.get("agent_type", "Unknown"))
            
            for entry in data.get("entries", []):
                if entry.get("category") != "negotiation":
                    continue
                
                partner = entry.get("data", {}).get("partner", "unknown")
                day = entry.get("day", 0)
                
                # 创建双向 key（确保同一协商只记录一次）
                key = tuple(sorted([agent_id, partner])) + (day,)
                
                if key not in negotiations:
                    negotiations[key] = {
                        "participants": list(sorted([agent_id, partner])),
                        "day": day,
                        "events": [],
                        "result": "ongoing",
                    }
                
                negotiations[key]["events"].append({
                    "from_agent": agent_id,
                    "from_type": agent_type,
                    "event": entry.get("event"),
                    "data": entry.get("data"),
                    "timestamp": entry.get("timestamp"),
                })
                
                # 确定结果
                if entry.get("event") == "success":
                    negotiations[key]["result"] = "success"
                elif entry.get("event") == "failure":
                    negotiations[key]["result"] = "failure"
        
        # 转换为列表并按天排序
        result = list(negotiations.values())
        
        # 对每个协商的 events 按时间戳排序
        for neg in result:
            neg["events"].sort(key=lambda e: e.get("timestamp", ""))
        
        result.sort(key=lambda n: (n["day"], str(n["participants"])))
        return result

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
    
    def get_world_list(self) -> List[Dict]:
        """获取所有 world 的列表（包含配置和 run 信息）
        
        返回格式:
        [
            {
                "world_name": "000020251128H..._LitaAgentY-LitaAgentYR.00",
                "config_id": "000020251128H..._LitaAgentY-LitaAgentYR",
                "run_index": 0,
                "agents": ["LitaAgentY", "LitaAgentYR", ...],
                "execution_time": 3.5,
                "n_steps": 20,
            }
        ]
        """
        worlds = []
        for w in self._world_stats:
            world_name = w.get("name", "")
            # 解析 world 名称: {config}.{run_index}
            parts = world_name.rsplit(".", 1)
            if len(parts) == 2:
                config_id, run_index_str = parts
                try:
                    run_index = int(run_index_str)
                except ValueError:
                    run_index = 0
            else:
                config_id, run_index = world_name, 0
            
            # 从配置名中提取 agent 列表
            # 格式: 00002025..._{Agent1}-{Agent2}-{Agent3}xxx
            agent_part = config_id.split("_", 1)[-1] if "_" in config_id else config_id
            # 简化提取 agent 名称
            agents = []
            for part in agent_part.replace("-", " ").split():
                if part.startswith("LitaAgent") or part.startswith("M") or len(part) > 3:
                    # 尝试识别 agent 名称
                    if "LitaAgent" in part:
                        agents.append(part)
            
            worlds.append({
                "world_name": world_name,
                "config_id": config_id,
                "run_index": run_index,
                "agents": agents,
                "execution_time": float(w.get("execution_time", 0)),
                "n_steps": int(w.get("executed_n_steps", w.get("planned_n_steps", 0))),
                "n_contracts": int(w.get("n_contracts_executed", 0)),
                "n_negotiations": int(w.get("n_negs_registered", 0)),
            })
        
        return worlds
    
    def get_scores_by_world(self, world_name: str = None) -> List[Dict]:
        """获取指定 world 或所有 world 的分数详情
        
        Args:
            world_name: World 名称，如果为 None 返回所有
            
        Returns:
            [{"world": "...", "agent_type": "LitaAgentY", "score": 0.95, "level": 0}, ...]
        """
        scores = []
        for row in self._scores:
            world = row.get("world", "")
            if world_name is None or world == world_name:
                # 解析 world 名称获取 config_id 和 run_index
                parts = world.rsplit(".", 1)
                if len(parts) == 2:
                    config_id, run_index = parts
                    try:
                        run_index = int(run_index)
                    except ValueError:
                        run_index = 0
                else:
                    config_id, run_index = world, 0
                
                # 从 agent_id 提取 level (格式: 00Li@0 -> level 0)
                agent_id = row.get("agent_id", "")
                level = None
                if "@" in agent_id:
                    try:
                        level = int(agent_id.split("@")[-1])
                    except ValueError:
                        level = None
                
                scores.append({
                    "world": world,
                    "config_id": config_id,
                    "run_index": run_index,
                    "agent_type": _extract_short_name(row.get("agent_type", "")),
                    "agent_id": agent_id,
                    "agent_name": row.get("agent_name", ""),
                    "level": level,
                    "score": float(row.get("score", 0)),
                    "run_id": row.get("run_id", ""),
                })
        return scores
    
    def get_config_list(self) -> List[Dict]:
        """获取所有配置的列表（汇总同一配置的多个 run）
        
        Returns:
            [
                {
                    "config_id": "000020251128H..._LitaAgentY-LitaAgentYR",
                    "run_count": 3,
                    "runs": [0, 1, 2],
                    "avg_score": 0.85,
                    "avg_score_by_agent": {"LitaAgentY": 0.85, ...},
                }
            ]
        """
        from collections import defaultdict
        
        configs = defaultdict(lambda: {
            "runs": [],
            "scores_by_agent": defaultdict(list),
            "all_scores": [],
        })
        
        for row in self._scores:
            world_name = row.get("world", "")
            parts = world_name.rsplit(".", 1)
            if len(parts) == 2:
                config_id, run_index_str = parts
                try:
                    run_index = int(run_index_str)
                except ValueError:
                    run_index = 0
            else:
                config_id, run_index = world_name, 0
            
            agent_type = _extract_short_name(row.get("agent_type", ""))
            score = float(row.get("score", 0))
            
            if run_index not in configs[config_id]["runs"]:
                configs[config_id]["runs"].append(run_index)
            configs[config_id]["scores_by_agent"][agent_type].append(score)
            configs[config_id]["all_scores"].append(score)
        
        result = []
        for config_id, data in configs.items():
            avg_scores = {}
            for agent, scores in data["scores_by_agent"].items():
                avg_scores[agent] = sum(scores) / len(scores) if scores else 0
            
            overall_avg = sum(data["all_scores"]) / len(data["all_scores"]) if data["all_scores"] else 0
            
            result.append({
                "config_id": config_id,
                "run_count": len(data["runs"]),
                "runs": sorted(data["runs"]),
                "avg_score": overall_avg,
                "avg_score_by_agent": avg_scores,
            })
        
        return result
    
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
    
    # 比赛配置信息 JSON（用于前端展示）
    tournament_config_json = json.dumps({
        "name": summary.get("tournament", {}).get("name", "Unknown"),
        "track": summary.get("tournament", {}).get("track", "std"),
        "n_configs": summary.get("tournament", {}).get("n_configs", 0),
        "n_runs_per_world": summary.get("tournament", {}).get("n_runs_per_world", 1),
        "n_steps": summary.get("tournament", {}).get("n_steps", 0),
        "n_worlds_completed": summary.get("tournament", {}).get("n_worlds_completed", 0),
        "duration_seconds": summary.get("tournament", {}).get("duration_seconds", 0),
        "winner": summary.get("tournament", {}).get("winner", "N/A"),
        "winner_score": summary.get("tournament", {}).get("winner_score", 0),
        "parallelism": summary.get("tournament", {}).get("parallelism", "unknown"),
        "agents": summary.get("agent_types", []),
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
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .config-item .key {{
            color: #666;
            font-size: 0.9em;
        }}
        .config-item .value {{
            font-weight: 600;
            color: #333;
        }}
        .agent-position {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .level-0 {{ background: #e3f2fd; color: #1565c0; }}
        .level-1 {{ background: #f3e5f5; color: #7b1fa2; }}
        .level-2 {{ background: #e8f5e9; color: #2e7d32; }}
        .neg-detail {{
            border: 1px solid #eee;
            border-radius: 8px;
            margin: 10px 0;
            overflow: hidden;
        }}
        .neg-detail-header {{
            padding: 12px 15px;
            background: #f8f9fa;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .neg-detail-header:hover {{
            background: #f0f0f0;
        }}
        .neg-detail-body {{
            padding: 15px;
            border-top: 1px solid #eee;
            display: none;
        }}
        .neg-detail-body.open {{
            display: block;
        }}
        .neg-round {{
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px dashed #eee;
        }}
        .neg-round:last-child {{
            border-bottom: none;
        }}
        .filter-bar {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 15px;
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
            <p style="margin-top: 15px;"><strong>🏆 冠军:</strong> 
                <span class="winner-badge">{summary.get('tournament', {}).get('winner', 'N/A')}</span>
                <span style="margin-left: 10px; color: #666;">得分: {summary.get('tournament', {}).get('winner_score', 0):.4f}</span>
            </p>
            
            <!-- 比赛配置详情 -->
            <div class="config-grid">
                <div class="config-item">
                    <span class="key">Track</span>
                    <span class="value">{summary.get('tournament', {}).get('track', 'std').upper()}</span>
                </div>
                <div class="config-item">
                    <span class="key">n_steps</span>
                    <span class="value">{summary.get('tournament', {}).get('n_steps', 'N/A')}</span>
                </div>
                <div class="config-item">
                    <span class="key">n_configs</span>
                    <span class="value">{summary.get('tournament', {}).get('n_configs', 'N/A')}</span>
                </div>
                <div class="config-item">
                    <span class="key">n_runs_per_world</span>
                    <span class="value">{summary.get('tournament', {}).get('n_runs_per_world', 'N/A')}</span>
                </div>
                <div class="config-item">
                    <span class="key">并行度</span>
                    <span class="value">{summary.get('tournament', {}).get('parallelism', 'N/A')}</span>
                </div>
            </div>
            
            <!-- 参赛 Agent 列表 -->
            <div style="margin-top: 20px;">
                <h4 style="margin-bottom: 10px; color: #333;">📋 参赛 Agent</h4>
                <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                    {"".join(f'<span class="agent-position level-{i % 3}">{agent}</span>' for i, agent in enumerate(summary.get("agent_types", [])))}
                </div>
            </div>
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
        
        <!-- World/Run 详细得分 -->
        <div class="card">
            <h2>🌍 World/Run 详细得分</h2>
            <p style="color: #666; margin-bottom: 15px;">
                查看每个配置和每次运行的详细得分。每个 Config 可能有多次 Run（标记为 .00, .01, .02 等）。
            </p>
            <div class="controls" style="display: flex; gap: 15px; flex-wrap: wrap; align-items: center;">
                <select id="configSelect" onchange="loadConfigRuns()">
                    <option value="">选择配置 (Config)...</option>
                </select>
                <select id="worldSelect" onchange="loadWorldScores()" style="min-width: 400px;">
                    <option value="">选择具体运行 (Run)...</option>
                </select>
                <button onclick="loadAllWorlds()" style="padding: 10px 15px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer;">
                    显示所有
                </button>
            </div>
            <div id="worldScoresContainer" style="max-height: 500px; overflow-y: auto; margin-top: 15px;">
                <p style="color: #666;">点击"显示所有"查看所有 world 的得分，或选择特定配置/运行</p>
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
            <div class="filter-bar">
                <div>
                    <label>Agent:</label>
                    <select id="negotiationAgentSelect" onchange="loadNegotiationDetails()">
                        <option value="">选择 Agent...</option>
                        {agent_options}
                    </select>
                </div>
                <div>
                    <label>时间范围:</label>
                    <input type="number" id="negDayFrom" placeholder="起始Day" min="0" style="width: 80px; padding: 8px;">
                    <span>-</span>
                    <input type="number" id="negDayTo" placeholder="结束Day" min="0" style="width: 80px; padding: 8px;">
                </div>
                <div>
                    <label>对手:</label>
                    <select id="negPartnerFilter">
                        <option value="">所有对手</option>
                    </select>
                </div>
                <div>
                    <label>结果:</label>
                    <select id="negResultFilter">
                        <option value="">全部</option>
                        <option value="success">成功</option>
                        <option value="failure">失败</option>
                    </select>
                </div>
                <button onclick="applyNegotiationFilters()" style="padding: 8px 15px; background: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer;">
                    应用筛选
                </button>
                <span id="negotiationCount" style="margin-left: 15px; color: #666;"></span>
            </div>
            <div id="negotiationContainer" style="max-height: 600px; overflow-y: auto;">
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
        
        <!-- ========== 单 World 分析模式 ========== -->
        <div class="card" style="border: 2px solid #667eea;">
            <h2>🔬 单 World 深度分析</h2>
            <p style="color: #666; margin-bottom: 15px;">
                选择特定的 World（一次完整模拟）深入分析所有 Agent 的行为和协商过程。
                每个 World 包含固定的 Agent 组合，可以追踪完整的交互轨迹。
            </p>
            
            <div class="filter-bar" style="background: #e8f4fd;">
                <div>
                    <label><strong>选择 World:</strong></label>
                    <select id="singleWorldSelect" onchange="loadSingleWorldData()" style="min-width: 300px;">
                        <option value="">选择一个 World...</option>
                    </select>
                </div>
                <div>
                    <label><strong>或选择 Agent 实例:</strong></label>
                    <select id="singleAgentSelect" onchange="loadSingleAgentData()" style="min-width: 250px;">
                        <option value="">选择 Agent 实例...</option>
                    </select>
                </div>
            </div>
            
            <!-- World 概览 -->
            <div id="singleWorldOverview" style="display: none; margin-top: 20px;">
                <h3 style="color: #667eea; margin-bottom: 10px;">📊 World 概览</h3>
                <div id="singleWorldStats" class="stats-grid" style="margin-bottom: 20px;"></div>
                
                <h4 style="margin-bottom: 10px;">🤖 参与的 Agent</h4>
                <div id="singleWorldAgents" style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;"></div>
            </div>
            
            <!-- 单 World 时间序列 -->
            <div id="singleWorldTimeSeries" style="display: none; margin-top: 20px;">
                <h3 style="color: #667eea; margin-bottom: 10px;">📈 时间序列对比</h3>
                <div class="controls">
                    <select id="singleWorldMetric" onchange="updateSingleWorldChart()">
                        <option value="balance">余额 (Balance)</option>
                        <option value="raw_material">原材料库存</option>
                        <option value="product">产品库存</option>
                    </select>
                </div>
                <div class="chart-container">
                    <canvas id="singleWorldChart"></canvas>
                </div>
            </div>
            
            <!-- 单 World 协商详情 -->
            <div id="singleWorldNegotiations" style="display: none; margin-top: 20px;">
                <h3 style="color: #667eea; margin-bottom: 10px;">🤝 协商过程详情</h3>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                    显示该 World 中所有协商的完整出价记录，可以追踪双方的报价轨迹。
                </p>
                <div class="controls" style="margin-bottom: 10px;">
                    <label>结果筛选:</label>
                    <select id="singleWorldNegFilter" onchange="filterSingleWorldNegotiations()">
                        <option value="">全部</option>
                        <option value="success">成功</option>
                        <option value="failure">失败</option>
                    </select>
                    <span id="singleWorldNegCount" style="margin-left: 15px; color: #666;"></span>
                </div>
                <div id="singleWorldNegContainer" style="max-height: 600px; overflow-y: auto;"></div>
            </div>
            
            <!-- 单 Agent 详情 -->
            <div id="singleAgentDetails" style="display: none; margin-top: 20px;">
                <h3 style="color: #667eea; margin-bottom: 10px;">👤 Agent 实例详情</h3>
                <div id="singleAgentInfo"></div>
                <div id="singleAgentEntries" style="max-height: 400px; overflow-y: auto; margin-top: 15px;"></div>
            </div>
        </div>
        
        <footer>
            <p>Generated by SCML Analyzer v0.4.0</p>
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
        
        // ========== World/Run 相关函数 ==========
        let allWorlds = [];
        let allConfigs = [];
        
        // 页面加载时初始化 World/Config 列表
        async function initWorldData() {{
            try {{
                // 加载所有 worlds
                const worldsResp = await fetch(apiUrl('/api/worlds'));
                allWorlds = await worldsResp.json();
                
                // 加载 configs
                const configsResp = await fetch(apiUrl('/api/configs'));
                allConfigs = await configsResp.json();
                
                // 填充 Config 下拉框
                const configSelect = document.getElementById('configSelect');
                configSelect.innerHTML = '<option value="">选择配置 (Config)...</option>';
                for (const config of allConfigs) {{
                    configSelect.innerHTML += `<option value="${{config.config_id}}">${{config.config_id}} (平均分: ${{config.avg_score.toFixed(4)}}, ${{config.run_count}} 次运行)</option>`;
                }}
            }} catch (error) {{
                console.error('初始化 World 数据失败:', error);
            }}
        }}
        
        // 当选择 Config 时，加载对应的 Runs
        function loadConfigRuns() {{
            const configId = document.getElementById('configSelect').value;
            const worldSelect = document.getElementById('worldSelect');
            
            worldSelect.innerHTML = '<option value="">选择具体运行 (Run)...</option>';
            
            if (!configId) return;
            
            // 筛选属于该 config 的 worlds
            const configWorlds = allWorlds.filter(w => w.config_id === configId);
            for (const world of configWorlds) {{
                worldSelect.innerHTML += `<option value="${{world.world_name}}">Run ${{world.run_index}} - ${{world.world_name.substring(0, 50)}}...</option>`;
            }}
        }}
        
        // 加载指定 World 的得分
        async function loadWorldScores() {{
            const worldName = document.getElementById('worldSelect').value;
            const container = document.getElementById('worldScoresContainer');
            
            if (!worldName) {{
                container.innerHTML = '<p style="color: #666;">请选择一个运行查看详细得分</p>';
                return;
            }}
            
            container.innerHTML = '<p style="color: #666;">加载中...</p>';
            
            try {{
                const response = await fetch(apiUrl(`/api/world_scores`) + `&world=${{encodeURIComponent(worldName)}}`);
                const scores = await response.json();
                
                if (scores.length === 0) {{
                    container.innerHTML = '<p style="color: #666;">暂无得分数据</p>';
                    return;
                }}
                
                renderScoresTable(scores, `World: ${{worldName}}`);
            }} catch (error) {{
                container.innerHTML = `<p style="color: #dc3545;">加载失败: ${{error.message}}</p>`;
            }}
        }}
        
        // 加载所有 World 的得分
        async function loadAllWorlds() {{
            const container = document.getElementById('worldScoresContainer');
            container.innerHTML = '<p style="color: #666;">加载中...</p>';
            
            try {{
                const response = await fetch(apiUrl('/api/world_scores'));
                const scores = await response.json();
                
                if (scores.length === 0) {{
                    container.innerHTML = '<p style="color: #666;">暂无得分数据</p>';
                    return;
                }}
                
                renderScoresTable(scores, `所有 World 得分 (共 ${{scores.length}} 条)`);
            }} catch (error) {{
                container.innerHTML = `<p style="color: #dc3545;">加载失败: ${{error.message}}</p>`;
            }}
        }}
        
        // 渲染得分表格
        function renderScoresTable(scores, title) {{
            const container = document.getElementById('worldScoresContainer');
            
            // 按 world 分组统计
            const worldGroups = {{}};
            for (const s of scores) {{
                if (!worldGroups[s.world]) {{
                    worldGroups[s.world] = [];
                }}
                worldGroups[s.world].push(s);
            }}
            
            let html = `<h4 style="margin-bottom: 10px;">${{title}}</h4>`;
            html += `<p style="color: #666; margin-bottom: 10px;">共 ${{Object.keys(worldGroups).length}} 个 World，${{scores.length}} 条得分记录</p>`;
            
            html += '<table style="width:100%; font-size: 0.85em;"><thead><tr>' +
                '<th>World</th><th>Agent</th><th>得分</th><th>Config</th><th>Run</th>' +
                '</tr></thead><tbody>';
            
            // 对 scores 按 world 和得分排序
            scores.sort((a, b) => {{
                if (a.world !== b.world) return a.world.localeCompare(b.world);
                return (b.score || 0) - (a.score || 0);
            }});
            
            let currentWorld = null;
            for (const s of scores.slice(0, 500)) {{
                const worldDisplay = s.world.substring(0, 30) + '...';
                const isNewWorld = s.world !== currentWorld;
                currentWorld = s.world;
                
                const scoreColor = s.score > 0 ? 'color: #28a745;' : s.score < 0 ? 'color: #dc3545;' : '';
                
                html += `<tr style="${{isNewWorld ? 'border-top: 2px solid #667eea;' : ''}}">
                    <td style="font-size: 0.75em;">${{isNewWorld ? worldDisplay : ''}}</td>
                    <td>${{s.agent_type || s.name || 'N/A'}}</td>
                    <td style="${{scoreColor}}">${{(s.score || 0).toFixed(4)}}</td>
                    <td style="font-size: 0.75em;">${{s.config_id || 'N/A'}}</td>
                    <td>${{s.run_index !== undefined ? '.' + s.run_index.toString().padStart(2, '0') : 'N/A'}}</td>
                </tr>`;
            }}
            
            html += '</tbody></table>';
            if (scores.length > 500) {{
                html += `<p style="color: #999; text-align: center; margin-top: 10px;">显示前 500 条，共 ${{scores.length}} 条</p>`;
            }}
            
            container.innerHTML = html;
        }}
        
        // 页面加载时初始化
        document.addEventListener('DOMContentLoaded', initWorldData);
        // ========== World/Run 相关函数结束 ==========
        
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
        let allNegotiations = [];  // 存储所有协商数据用于筛选
        
        async function loadNegotiationDetails() {{
            const agentType = document.getElementById('negotiationAgentSelect').value;
            const container = document.getElementById('negotiationContainer');
            const countSpan = document.getElementById('negotiationCount');
            
            if (!agentType) {{
                container.innerHTML = '<p style="color: #666;">请选择一个 Agent 查看协商详情</p>';
                countSpan.textContent = '';
                allNegotiations = [];
                return;
            }}
            
            container.innerHTML = '<p style="color: #666;">加载中...</p>';
            
            try {{
                const response = await fetch(apiUrl(`/api/negotiations/${{encodeURIComponent(agentType)}}`));
                allNegotiations = await response.json();
                
                // 填充对手筛选下拉框
                const partners = [...new Set(allNegotiations.map(n => n.partner))];
                const partnerSelect = document.getElementById('negPartnerFilter');
                partnerSelect.innerHTML = '<option value="">所有对手</option>';
                for (const partner of partners.slice(0, 50)) {{
                    const shortPartner = partner.substring(0, 25);
                    partnerSelect.innerHTML += `<option value="${{partner}}">${{shortPartner}}</option>`;
                }}
                
                renderNegotiations(allNegotiations);
            }} catch (error) {{
                container.innerHTML = `<p style="color: #dc3545;">加载失败: ${{error.message}}</p>`;
            }}
        }}
        
        function applyNegotiationFilters() {{
            const dayFrom = parseInt(document.getElementById('negDayFrom').value) || 0;
            const dayTo = parseInt(document.getElementById('negDayTo').value) || 999;
            const partner = document.getElementById('negPartnerFilter').value;
            const result = document.getElementById('negResultFilter').value;
            
            let filtered = allNegotiations.filter(n => {{
                if (n.day < dayFrom || n.day > dayTo) return false;
                if (partner && n.partner !== partner) return false;
                if (result && n.result !== result) return false;
                return true;
            }});
            
            renderNegotiations(filtered);
        }}
        
        function renderNegotiations(negotiations) {{
            const container = document.getElementById('negotiationContainer');
            const countSpan = document.getElementById('negotiationCount');
            
            countSpan.textContent = `显示 ${{negotiations.length}} 条 / 共 ${{allNegotiations.length}} 次协商`;
            
            if (negotiations.length === 0) {{
                container.innerHTML = '<p style="color: #666;">暂无协商数据（需要 Tracker 日志）</p>';
                return;
            }}
            
            // 统计信息
            const successCount = negotiations.filter(n => n.result === 'success').length;
            const failCount = negotiations.filter(n => n.result === 'failure').length;
            const hasOffers = negotiations.some(n => n.events && n.events.some(e => e.event === 'offer_made' || e.event === 'offer_received'));
            
            let html = `<div style="margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 8px;">
                <strong>统计：</strong> 
                <span style="color: #28a745;">✓ 成功 ${{successCount}}</span> | 
                <span style="color: #dc3545;">✗ 失败 ${{failCount}}</span> | 
                成功率 ${{negotiations.length > 0 ? (successCount / negotiations.length * 100).toFixed(1) : 0}}%
                ${{hasOffers ? '' : '<br><small style="color: #999;">⚠️ 旧版 Tracker 未记录出价过程，运行新比赛可获得完整数据</small>'}}
            </div>`;
            
            // 协商卡片列表（可展开）
            for (let i = 0; i < Math.min(negotiations.length, 100); i++) {{
                const neg = negotiations[i];
                const resultClass = neg.result === 'success' ? 'background: #d4edda;' : 
                                   neg.result === 'failure' ? 'background: #f8d7da;' : 'background: #fff3cd;';
                const resultText = neg.result === 'success' ? '✓ 成功' : 
                                  neg.result === 'failure' ? '✗ 失败' : '⋯ 进行中';
                const resultColor = neg.result === 'success' ? '#28a745' : 
                                   neg.result === 'failure' ? '#dc3545' : '#ffc107';
                
                // 提取最终协议
                let agreement = null;
                let rounds = [];
                if (neg.events) {{
                    for (const event of neg.events) {{
                        const data = event.data || {{}};
                        if (event.event === 'success') {{
                            agreement = data.agreement || {{}};
                        }} else if (event.event === 'offer_received' || event.event === 'offer_made') {{
                            rounds.push({{
                                type: event.event === 'offer_received' ? 'received' : 'made',
                                round: data.round || '?',
                                quantity: data.quantity,
                                unit_price: data.unit_price,
                                delivery_day: data.delivery_day
                            }});
                        }}
                    }}
                }}
                
                const agreementText = agreement ? 
                    `Q=${{agreement.quantity || 'N/A'}}, P=${{agreement.price || 'N/A'}}` : 
                    (neg.result === 'failure' ? '无协议' : '-');
                
                html += `<div class="neg-detail">
                    <div class="neg-detail-header" style="${{resultClass}}" onclick="toggleNegDetail(${{i}})">
                        <div>
                            <strong>Day ${{neg.day}}</strong> | 
                            <span style="font-size: 0.9em;">对手: ${{neg.partner.substring(0, 25)}}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="color: ${{resultColor}}; font-weight: bold;">${{resultText}}</span>
                            <span style="font-size: 0.85em; color: #666;">${{agreementText}}</span>
                            <span style="font-size: 1.2em;">▼</span>
                        </div>
                    </div>
                    <div class="neg-detail-body" id="neg-body-${{i}}">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div>
                                <h4 style="margin-bottom: 10px; color: #333;">📋 协商背景</h4>
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; font-size: 0.9em;">
                                    <div><strong>Day:</strong> ${{neg.day}}</div>
                                    <div><strong>对手:</strong> ${{neg.partner}}</div>
                                    <div><strong>World:</strong> ${{neg.world || 'N/A'}}</div>
                                    <div><strong>事件数:</strong> ${{neg.events ? neg.events.length : 0}}</div>
                                </div>
                            </div>
                            <div>
                                <h4 style="margin-bottom: 10px; color: #333;">📊 结果</h4>
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 6px; font-size: 0.9em;">
                                    <div><strong>状态:</strong> <span style="color: ${{resultColor}};">${{resultText}}</span></div>
                                    ${{agreement ? `
                                    <div><strong>数量:</strong> ${{agreement.quantity || 'N/A'}}</div>
                                    <div><strong>单价:</strong> ${{agreement.price || 'N/A'}}</div>
                                    <div><strong>交货日:</strong> ${{agreement.time || 'N/A'}}</div>
                                    ` : '<div style="color: #999;">无协议达成</div>'}}
                                </div>
                            </div>
                        </div>
                        ${{rounds.length > 0 ? `
                        <div style="margin-top: 15px;">
                            <h4 style="margin-bottom: 10px; color: #333;">🔄 谈判过程 (${{rounds.length}} 轮)</h4>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 6px;">
                                ${{rounds.map(r => `
                                <div class="neg-round">
                                    <span style="width: 30px; text-align: center; font-weight: bold;">R${{r.round}}</span>
                                    <span style="width: 80px; color: ${{r.type === 'received' ? '#28a745' : '#007bff'}};">
                                        ${{r.type === 'received' ? '← 收到' : '→ 发出'}}
                                    </span>
                                    <span style="flex: 1;">
                                        Q=${{r.quantity || 'N/A'}}, P=${{r.unit_price || 'N/A'}}, D=${{r.delivery_day || 'N/A'}}
                                    </span>
                                </div>
                                `).join('')}}
                            </div>
                        </div>
                        ` : '<p style="color: #999; margin-top: 10px; font-size: 0.9em;">⚠️ 无详细出价记录（旧版 Tracker）</p>'}}
                    </div>
                </div>`;
            }}
            
            if (negotiations.length > 100) {{
                html += `<p style="color: #999; text-align: center; margin-top: 10px;">显示前 100 条，共 ${{negotiations.length}} 条</p>`;
            }}
            container.innerHTML = html;
        }}
        
        function toggleNegDetail(index) {{
            const body = document.getElementById(`neg-body-${{index}}`);
            body.classList.toggle('open');
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
        
        // ========== 单 World 分析模式 JavaScript ==========
        let singleWorldData = null;
        let singleWorldNegotiations = [];
        let singleWorldChart = null;
        let trackerWorlds = [];
        let agentInstances = [];
        
        // 初始化单 World 模式数据
        async function initSingleWorldMode() {{
            try {{
                // 加载 tracker worlds
                const worldsResp = await fetch(apiUrl('/api/tracker_worlds'));
                trackerWorlds = await worldsResp.json();
                
                // 加载 agent instances
                const instancesResp = await fetch(apiUrl('/api/agent_instances'));
                agentInstances = await instancesResp.json();
                
                // 填充 World 下拉框
                const worldSelect = document.getElementById('singleWorldSelect');
                worldSelect.innerHTML = '<option value="">选择一个 World...</option>';
                for (const world of trackerWorlds) {{
                    const agentList = world.agents.map(a => a.agent_type).join(', ');
                    const label = world.world_id === 'unknown' ? 
                        `[未命名] ${{world.agent_count}} 个 Agent: ${{agentList.substring(0, 50)}}...` :
                        `${{world.world_id.substring(0, 40)}}... (${{world.agent_count}} agents)`;
                    worldSelect.innerHTML += `<option value="${{world.world_id}}">${{label}}</option>`;
                }}
                
                // 填充 Agent Instance 下拉框
                const agentSelect = document.getElementById('singleAgentSelect');
                agentSelect.innerHTML = '<option value="">选择 Agent 实例...</option>';
                for (const inst of agentInstances) {{
                    agentSelect.innerHTML += `<option value="${{inst.agent_id}}">${{inst.display_name}}</option>`;
                }}
            }} catch (error) {{
                console.error('初始化单 World 模式失败:', error);
            }}
        }}
        
        // 加载单个 World 数据
        async function loadSingleWorldData() {{
            const worldId = document.getElementById('singleWorldSelect').value;
            const overview = document.getElementById('singleWorldOverview');
            const timeSeries = document.getElementById('singleWorldTimeSeries');
            const negotiations = document.getElementById('singleWorldNegotiations');
            const agentDetails = document.getElementById('singleAgentDetails');
            
            // 清除 agent 选择
            document.getElementById('singleAgentSelect').value = '';
            agentDetails.style.display = 'none';
            
            if (!worldId) {{
                overview.style.display = 'none';
                timeSeries.style.display = 'none';
                negotiations.style.display = 'none';
                return;
            }}
            
            try {{
                // 加载 world 数据
                const resp = await fetch(apiUrl(`/api/single_world`) + `&world_id=${{encodeURIComponent(worldId)}}`);
                singleWorldData = await resp.json();
                
                // 显示概览
                overview.style.display = 'block';
                
                // 统计信息
                const stats = singleWorldData.aggregated_stats || {{}};
                document.getElementById('singleWorldStats').innerHTML = `
                    <div class="stat-box">
                        <div class="value">${{singleWorldData.agent_count || 0}}</div>
                        <div class="label">Agent 数量</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">${{singleWorldData.entry_count || 0}}</div>
                        <div class="label">事件记录数</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">${{stats.negotiations_success || 0}}</div>
                        <div class="label">成功协商</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">${{stats.negotiations_failed || 0}}</div>
                        <div class="label">失败协商</div>
                    </div>
                    <div class="stat-box">
                        <div class="value">${{stats.offers_made || 0}}</div>
                        <div class="label">发出报价</div>
                    </div>
                `;
                
                // Agent 列表
                const agentsHtml = (singleWorldData.agents || []).map(a => `
                    <span class="agent-position level-${{a.level || 0}}" style="cursor: pointer;" 
                          onclick="selectAgentInstance('${{a.agent_id}}')">
                        ${{a.agent_id}} (${{a.agent_type}})
                        ${{a.level !== null ? ' L' + a.level : ''}}
                    </span>
                `).join('');
                document.getElementById('singleWorldAgents').innerHTML = agentsHtml;
                
                // 时间序列
                timeSeries.style.display = 'block';
                updateSingleWorldChart();
                
                // 加载协商详情
                const negResp = await fetch(apiUrl(`/api/single_world_negotiations`) + `&world_id=${{encodeURIComponent(worldId)}}`);
                singleWorldNegotiations = await negResp.json();
                negotiations.style.display = 'block';
                renderSingleWorldNegotiations(singleWorldNegotiations);
                
            }} catch (error) {{
                console.error('加载单 World 数据失败:', error);
                overview.innerHTML = `<p style="color: #dc3545;">加载失败: ${{error.message}}</p>`;
            }}
        }}
        
        // 更新单 World 时间序列图表
        function updateSingleWorldChart() {{
            if (!singleWorldData || !singleWorldData.time_series_by_agent) return;
            
            const metric = document.getElementById('singleWorldMetric').value;
            const ctx = document.getElementById('singleWorldChart').getContext('2d');
            
            if (singleWorldChart) {{ singleWorldChart.destroy(); }}
            
            const colors = [
                'rgba(102, 126, 234, 0.8)',
                'rgba(118, 75, 162, 0.8)',
                'rgba(234, 102, 126, 0.8)',
                'rgba(126, 234, 102, 0.8)',
                'rgba(234, 206, 102, 0.8)',
                'rgba(102, 234, 206, 0.8)',
            ];
            
            const datasets = [];
            let allDays = new Set();
            let colorIndex = 0;
            
            for (const [agentId, ts] of Object.entries(singleWorldData.time_series_by_agent)) {{
                const data = ts[metric] || [];
                for (const [day, value] of data) {{
                    allDays.add(day);
                }}
                
                const dataMap = Object.fromEntries(data);
                datasets.push({{
                    label: agentId,
                    data: [...allDays].sort((a,b) => a-b).map(d => dataMap[d] || null),
                    borderColor: colors[colorIndex % colors.length],
                    backgroundColor: colors[colorIndex % colors.length].replace('0.8', '0.1'),
                    fill: false,
                    tension: 0.1,
                    spanGaps: true,
                }});
                colorIndex++;
            }}
            
            const labels = [...allDays].sort((a, b) => a - b);
            
            singleWorldChart = new Chart(ctx, {{
                type: 'line',
                data: {{ labels, datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: `${{metric}} 随时间变化 (单 World 模式)`
                        }}
                    }},
                    scales: {{
                        y: {{ beginAtZero: false }}
                    }}
                }}
            }});
        }}
        
        // 渲染单 World 协商
        function renderSingleWorldNegotiations(negotiations) {{
            const container = document.getElementById('singleWorldNegContainer');
            const countSpan = document.getElementById('singleWorldNegCount');
            
            const successCount = negotiations.filter(n => n.result === 'success').length;
            const failCount = negotiations.filter(n => n.result === 'failure').length;
            
            countSpan.textContent = `共 ${{negotiations.length}} 次协商，成功 ${{successCount}}，失败 ${{failCount}}`;
            
            if (negotiations.length === 0) {{
                container.innerHTML = '<p style="color: #666;">暂无协商数据</p>';
                return;
            }}
            
            let html = '';
            for (let i = 0; i < Math.min(negotiations.length, 100); i++) {{
                const neg = negotiations[i];
                const resultClass = neg.result === 'success' ? 'background: #d4edda;' : 
                                   neg.result === 'failure' ? 'background: #f8d7da;' : 'background: #fff3cd;';
                const resultText = neg.result === 'success' ? '✓ 成功' : 
                                  neg.result === 'failure' ? '✗ 失败' : '⋯ 进行中';
                const resultColor = neg.result === 'success' ? '#28a745' : 
                                   neg.result === 'failure' ? '#dc3545' : '#ffc107';
                
                const participants = neg.participants.join(' ↔ ');
                
                // 构建出价历史
                let offersHtml = '';
                if (neg.events && neg.events.length > 0) {{
                    for (const event of neg.events) {{
                        const data = event.data || {{}};
                        const offer = data.offer || data.agreement || {{}};
                        const eventType = event.event;
                        const fromAgent = event.from_agent || 'unknown';
                        
                        let eventLabel = eventType;
                        let eventColor = '#666';
                        if (eventType === 'offer_made') {{ eventLabel = '→ 发出报价'; eventColor = '#007bff'; }}
                        else if (eventType === 'offer_received') {{ eventLabel = '← 收到报价'; eventColor = '#28a745'; }}
                        else if (eventType === 'success') {{ eventLabel = '✓ 达成协议'; eventColor = '#28a745'; }}
                        else if (eventType === 'failure') {{ eventLabel = '✗ 协商失败'; eventColor = '#dc3545'; }}
                        else if (eventType === 'started') {{ eventLabel = '● 开始协商'; eventColor = '#17a2b8'; }}
                        
                        const offerText = offer.quantity !== undefined ? 
                            `Q=${{offer.quantity}}, P=${{offer.unit_price || offer.price || 'N/A'}}, D=${{offer.delivery_day || offer.time || 'N/A'}}` :
                            '';
                        
                        offersHtml += `<div class="neg-round">
                            <span style="width: 120px; font-size: 0.8em; color: #999;">${{fromAgent.substring(0, 12)}}</span>
                            <span style="width: 100px; color: ${{eventColor}}; font-size: 0.85em;">${{eventLabel}}</span>
                            <span style="flex: 1; font-size: 0.85em;">${{offerText}}</span>
                        </div>`;
                    }}
                }}
                
                html += `<div class="neg-detail">
                    <div class="neg-detail-header" style="${{resultClass}}" onclick="toggleSingleWorldNeg(${{i}})">
                        <div>
                            <strong>Day ${{neg.day}}</strong> | 
                            <span style="font-size: 0.9em;">${{participants}}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="color: ${{resultColor}}; font-weight: bold;">${{resultText}}</span>
                            <span style="font-size: 0.85em; color: #666;">${{neg.events ? neg.events.length : 0}} 事件</span>
                            <span style="font-size: 1.2em;">▼</span>
                        </div>
                    </div>
                    <div class="neg-detail-body" id="sw-neg-body-${{i}}">
                        <h4 style="margin-bottom: 10px; color: #333;">📝 交互记录</h4>
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 6px;">
                            ${{offersHtml || '<p style="color: #999;">无详细记录</p>'}}
                        </div>
                    </div>
                </div>`;
            }}
            
            if (negotiations.length > 100) {{
                html += `<p style="color: #999; text-align: center; margin-top: 10px;">显示前 100 条，共 ${{negotiations.length}} 条</p>`;
            }}
            
            container.innerHTML = html;
        }}
        
        function toggleSingleWorldNeg(index) {{
            const body = document.getElementById(`sw-neg-body-${{index}}`);
            body.classList.toggle('open');
        }}
        
        function filterSingleWorldNegotiations() {{
            const filter = document.getElementById('singleWorldNegFilter').value;
            let filtered = singleWorldNegotiations;
            if (filter) {{
                filtered = singleWorldNegotiations.filter(n => n.result === filter);
            }}
            renderSingleWorldNegotiations(filtered);
        }}
        
        // 选择并加载 Agent 实例
        function selectAgentInstance(agentId) {{
            document.getElementById('singleAgentSelect').value = agentId;
            loadSingleAgentData();
        }}
        
        // 加载单个 Agent 实例数据
        async function loadSingleAgentData() {{
            const agentId = document.getElementById('singleAgentSelect').value;
            const detailsDiv = document.getElementById('singleAgentDetails');
            
            if (!agentId) {{
                detailsDiv.style.display = 'none';
                return;
            }}
            
            try {{
                const resp = await fetch(apiUrl(`/api/single_agent`) + `&agent_id=${{encodeURIComponent(agentId)}}`);
                const data = await resp.json();
                
                detailsDiv.style.display = 'block';
                
                // Agent 信息
                const stats = data.stats || {{}};
                document.getElementById('singleAgentInfo').innerHTML = `
                    <div class="config-grid">
                        <div class="config-item"><span class="key">Agent ID</span><span class="value">${{data.agent_id || agentId}}</span></div>
                        <div class="config-item"><span class="key">类型</span><span class="value">${{data.agent_type || 'Unknown'}}</span></div>
                        <div class="config-item"><span class="key">World ID</span><span class="value" style="font-size: 0.7em;">${{(data.world_id || 'unknown').substring(0, 30)}}...</span></div>
                        <div class="config-item"><span class="key">事件数</span><span class="value">${{data.entry_count || 0}}</span></div>
                        <div class="config-item"><span class="key">成功协商</span><span class="value">${{stats.negotiations_success || 0}}</span></div>
                        <div class="config-item"><span class="key">失败协商</span><span class="value">${{stats.negotiations_failed || 0}}</span></div>
                        <div class="config-item"><span class="key">发出报价</span><span class="value">${{stats.offers_made || 0}}</span></div>
                        <div class="config-item"><span class="key">接受报价</span><span class="value">${{stats.offers_accepted || 0}}</span></div>
                    </div>
                `;
                
                // 事件列表
                const entries = data.entries || [];
                let entriesHtml = `<h4 style="margin-bottom: 10px;">📜 事件记录 (共 ${{entries.length}} 条)</h4>`;
                entriesHtml += '<table style="width: 100%; font-size: 0.8em;"><thead><tr><th>Day</th><th>类别</th><th>事件</th><th>详情</th></tr></thead><tbody>';
                
                for (const entry of entries.slice(0, 100)) {{
                    const dataStr = JSON.stringify(entry.data || {{}}).substring(0, 100);
                    entriesHtml += `<tr>
                        <td>${{entry.day}}</td>
                        <td>${{entry.category || 'N/A'}}</td>
                        <td>${{entry.event || 'N/A'}}</td>
                        <td style="font-size: 0.75em; color: #666;">${{dataStr}}...</td>
                    </tr>`;
                }}
                
                entriesHtml += '</tbody></table>';
                if (entries.length > 100) {{
                    entriesHtml += `<p style="color: #999; text-align: center;">显示前 100 条</p>`;
                }}
                
                document.getElementById('singleAgentEntries').innerHTML = entriesHtml;
                
            }} catch (error) {{
                console.error('加载 Agent 数据失败:', error);
                detailsDiv.innerHTML = `<p style="color: #dc3545;">加载失败: ${{error.message}}</p>`;
            }}
        }}
        
        // 初始化
        updateTimeSeriesChart();
        initSingleWorldMode();
    </script>
</body>
</html>
"""
    return html


def generate_tournament_list_html(tournaments: List[Dict]) -> str:
    """生成比赛列表页面 HTML"""
    
    # 如果没有比赛
    if not tournaments:
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCML Analyzer - 比赛列表</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .empty-state {
            background: white;
            padding: 60px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .empty-state h1 { color: #333; margin-bottom: 20px; }
        .empty-state p { color: #666; margin-bottom: 30px; }
        .empty-state code {
            background: #f5f5f5;
            padding: 10px 20px;
            border-radius: 8px;
            display: block;
            font-size: 14px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="empty-state">
        <h1>🏆 SCML Analyzer</h1>
        <p>暂无比赛数据。请先运行比赛或导入比赛数据。</p>
        <p>导入命令：</p>
        <code>python -m scml_analyzer.history import-all</code>
    </div>
</body>
</html>
"""
    
    # 生成比赛列表行
    tournament_rows = ""
    for t in tournaments:
        results = t.get("results", {})
        settings = t.get("settings", {})
        track = t.get("track", "unknown").upper()
        track_class = "oneshot" if track == "ONESHOT" else "std"
        
        winner = results.get("winner", "N/A") or "N/A"
        winner_score = results.get("winner_score", 0) or 0
        n_completed = results.get("n_completed", 0) or 0
        duration = results.get("total_duration_seconds", 0) or 0
        
        timestamp = t.get("timestamp", "")
        tournament_id = t.get("id", "unknown")
        n_competitors = t.get("n_competitors", 0) or 0
        
        tournament_rows += f"""
        <tr onclick="window.location='/tournament/{tournament_id}'" style="cursor: pointer;">
            <td>{timestamp}</td>
            <td><span class="track-badge {track_class}">{track}</span></td>
            <td>{n_competitors}</td>
            <td>{n_completed}</td>
            <td><strong>{winner}</strong></td>
            <td>{winner_score:.4f}</td>
            <td>{duration:.1f}s</td>
        </tr>
        """
    
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCML Analyzer - 比赛列表</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header p {{ opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }}
        .card h2 {{
            color: #333;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #667eea;
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #333; }}
        tr:hover {{ background: #f5f5f5; }}
        .track-badge {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .track-badge.oneshot {{ background: #e3f2fd; color: #1976d2; }}
        .track-badge.std {{ background: #f3e5f5; color: #7b1fa2; }}
        .stats-bar {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}
        .stat {{ text-align: center; }}
        .stat .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .stat .label {{ color: #666; font-size: 0.9em; }}
        footer {{
            text-align: center;
            color: white;
            opacity: 0.8;
            margin-top: 30px;
            padding: 20px;
        }}
        .import-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin-right: 10px;
        }}
        .import-btn:hover {{ background: #5a6fd6; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 SCML Analyzer</h1>
            <p>比赛历史数据可视化分析平台</p>
        </header>
        
        <div class="card">
            <h2>📋 比赛列表</h2>
            
            <div class="stats-bar">
                <div class="stat">
                    <div class="value">{len(tournaments)}</div>
                    <div class="label">总比赛数</div>
                </div>
                <div class="stat">
                    <div class="value">{sum(1 for t in tournaments if t.get('track') == 'oneshot')}</div>
                    <div class="label">OneShot 比赛</div>
                </div>
                <div class="stat">
                    <div class="value">{sum(1 for t in tournaments if t.get('track') == 'std')}</div>
                    <div class="label">Standard 比赛</div>
                </div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>赛道</th>
                        <th>参赛者</th>
                        <th>完成场次</th>
                        <th>🏆 冠军</th>
                        <th>冠军得分</th>
                        <th>耗时</th>
                    </tr>
                </thead>
                <tbody>
                    {tournament_rows}
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>点击任意比赛查看详细分析 | SCML Analyzer v0.4.0</p>
        </footer>
    </div>
</body>
</html>
"""


class VisualizerHandler(SimpleHTTPRequestHandler):
    """HTTP 请求处理器 - 支持比赛列表和详情页"""
    
    # 当前加载的比赛数据
    current_data: VisualizerData = None
    current_tournament_id: str = None
    
    def _parse_path(self):
        """解析 URL 路径，提取 tournament_id"""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        return path, query
    
    def _load_tournament_data(self, tournament_id: str) -> Optional[VisualizerData]:
        """加载指定比赛的数据"""
        tournament = history.get_tournament(tournament_id)
        if not tournament:
            return None
        
        data = VisualizerData(tournament["path"])
        data.load_all()
        return data
    
    def do_GET(self):
        path, query = self._parse_path()
        
        # 首页 - 比赛列表
        if path == '/' or path == '/index.html':
            tournaments = history.list_tournaments()
            html = generate_tournament_list_html(tournaments)
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            return
        
        # 比赛详情页 /tournament/{id}
        if path.startswith('/tournament/'):
            tournament_id = path.split('/')[-1]
            data = self._load_tournament_data(tournament_id)
            
            if not data:
                self.send_error(404, f"Tournament not found: {tournament_id}")
                return
            
            html = generate_html_report(data)
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            return
        
        # API: 协商详情 /api/negotiations/{agent_type}?tournament={id}
        if path.startswith('/api/negotiations/'):
            agent_type = urllib.parse.unquote(path.split('/')[-1].split('?')[0])
            tournament_id = query.get('tournament', [None])[0]
            
            # 从 query 中提取 path 参数（兼容旧模式）
            if not tournament_id:
                path_param = query.get('path', [None])[0]
                if path_param:
                    data = VisualizerData(urllib.parse.unquote(path_param))
                    data.load_all()
                    negotiations = data.get_negotiation_details(agent_type)
                    self._send_json(negotiations)
                    return
            
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
                if data:
                    negotiations = data.get_negotiation_details(agent_type)
                    self._send_json(negotiations)
                    return
            
            self._send_json([])
            return
        
        # API: 每日状态 /api/daily_status/{agent_type}?tournament={id}
        if path.startswith('/api/daily_status/'):
            agent_type = urllib.parse.unquote(path.split('/')[-1].split('?')[0])
            tournament_id = query.get('tournament', [None])[0]
            
            # 从 query 中提取 path 参数（兼容旧模式）
            if not tournament_id:
                path_param = query.get('path', [None])[0]
                if path_param:
                    data = VisualizerData(urllib.parse.unquote(path_param))
                    data.load_all()
                    daily_status = data.get_daily_status(agent_type)
                    self._send_json(daily_status)
                    return
            
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
                if data:
                    daily_status = data.get_daily_status(agent_type)
                    self._send_json(daily_status)
                    return
            
            self._send_json([])
            return
        
        # API: 时间序列 /api/time_series/{agent_type}?tournament={id} 或 ?path={path}
        if path.startswith('/api/time_series/'):
            agent_type = urllib.parse.unquote(path.split('/')[-1].split('?')[0])
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                time_series = data.get_tracker_time_series(agent_type)
                self._send_json(time_series)
                return
            
            self._send_json({})
            return
        
        # API: World 列表 /api/worlds?tournament={id} 或 /api/worlds?path={path}
        if path == '/api/worlds':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                worlds = data.get_world_list()
                self._send_json(worlds)
                return
            self._send_json([])
            return
        
        # API: Config 列表（按配置分组） /api/configs?tournament={id} 或 /api/configs?path={path}
        if path == '/api/configs':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                configs = data.get_config_list()
                self._send_json(configs)
                return
            self._send_json([])
            return
        
        # API: 指定 World 的分数 /api/world_scores?tournament={id}&world={world_name}
        if path == '/api/world_scores':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            world_name = query.get('world', [None])[0]
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                scores = data.get_scores_by_world(world_name)
                self._send_json(scores)
                return
            self._send_json([])
            return
        
        # API: 比赛列表
        if path == '/api/tournaments':
            tournaments = history.list_tournaments()
            self._send_json(tournaments)
            return
        
        # ========== 单 World 模式 API ==========
        
        # API: 从 Tracker 获取 World 列表（用于单 World 模式）
        if path == '/api/tracker_worlds':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                worlds = data.get_available_worlds_from_tracker()
                self._send_json(worlds)
                return
            self._send_json([])
            return
        
        # API: Agent 实例列表
        if path == '/api/agent_instances':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                instances = data.get_agent_instances()
                self._send_json(instances)
                return
            self._send_json([])
            return
        
        # API: 单个 Agent 实例数据
        if path == '/api/single_agent':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            agent_id = query.get('agent_id', [None])[0]
            
            if not agent_id:
                self._send_json({"error": "agent_id required"})
                return
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                agent_data = data.get_single_agent_data(urllib.parse.unquote(agent_id))
                self._send_json(agent_data)
                return
            self._send_json({})
            return
        
        # API: 单个 World 数据
        if path == '/api/single_world':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            world_id = query.get('world_id', [None])[0]
            
            if not world_id:
                self._send_json({"error": "world_id required"})
                return
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                world_data = data.get_single_world_data(urllib.parse.unquote(world_id))
                self._send_json(world_data)
                return
            self._send_json({})
            return
        
        # API: 单个 World 的协商详情
        if path == '/api/single_world_negotiations':
            tournament_id = query.get('tournament', [None])[0]
            path_param = query.get('path', [None])[0]
            world_id = query.get('world_id', [None])[0]
            
            if not world_id:
                self._send_json({"error": "world_id required"})
                return
            
            data = None
            if tournament_id:
                data = self._load_tournament_data(tournament_id)
            elif path_param:
                data = VisualizerData(urllib.parse.unquote(path_param))
                data.load_all()
            
            if data:
                negotiations = data.get_single_world_negotiations(urllib.parse.unquote(world_id))
                self._send_json(negotiations)
                return
            self._send_json([])
            return
        
        self.send_error(404, "Not found")
    
    def _send_json(self, data):
        """发送 JSON 响应"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def log_message(self, format, *args):
        # 静默日志
        pass


def start_server(port: int = 8080, open_browser: bool = True, host: str = "0.0.0.0"):
    """
    启动可视化服务器 - 无参数模式
    
    自动从 tournament_history 目录读取比赛数据。
    不需要指定任何数据目录！
    
    Args:
        port: 服务器端口
        open_browser: 是否自动打开浏览器
        host: 监听地址（默认 0.0.0.0 以便远程访问）
    """
    # 确保 tournament_history 目录存在
    history_dir = history.get_history_dir()
    history_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取比赛列表
    tournaments = history.list_tournaments()
    
    # 启动服务器
    server = HTTPServer((host, port), VisualizerHandler)
    
    url = f"http://{host if host!='0.0.0.0' else '0.0.0.0'}:{port}"
    print(f"🌐 可视化服务器已启动: {url}")
    print(f"📁 数据目录: {history_dir}")
    print(f"📊 已导入比赛: {len(tournaments)} 场")
    print("按 Ctrl+C 停止服务器")
    
    if open_browser:
        webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.shutdown()


def generate_static_report(tournament_id: str, output_file: str = "report.html"):
    """
    生成静态 HTML 报告文件
    
    Args:
        tournament_id: 比赛 ID
        output_file: 输出文件名
    """
    tournament = history.get_tournament(tournament_id)
    if not tournament:
        print(f"❌ 找不到比赛: {tournament_id}")
        return None
    
    data = VisualizerData(tournament["path"])
    data.load_all()
    
    html = generate_html_report(data)
    
    output_path = Path(tournament["path"]) / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📄 报告已生成: {output_path}")
    return str(output_path)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='SCML Analyzer 可视化服务器 - 无参数启动！'
    )
    parser.add_argument('--port', '-p', type=int, default=8080, help='服务器端口')
    parser.add_argument('--host', '-H', type=str, default='0.0.0.0', help='监听地址（默认 0.0.0.0）')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--static', type=str, metavar='TOURNAMENT_ID',
                       help='生成静态报告（需要指定比赛 ID）')
    
    args = parser.parse_args()
    
    if args.static:
        generate_static_report(args.static)
    else:
        start_server(args.port, not args.no_browser, host=args.host)


if __name__ == "__main__":
    main()
