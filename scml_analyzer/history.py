"""
SCML Tournament History Manager

管理比赛历史数据，将 negmas 和 tracker 的数据整合到统一的目录结构中。

设计原则：
- 比赛完成后，自动将数据复制/移动到项目的 tournament_history 目录
- 每场比赛有独立的子目录，包含所有相关数据
- 支持从任意 negmas tournament 目录导入

目录结构：
    tournament_history/
    ├── 20251128_130949_oneshot/
    │   ├── tournament_info.json    # 比赛元信息
    │   ├── params.json             # negmas 参数（复制）
    │   ├── total_scores.csv        # 排名（复制）
    │   ├── winners.csv             # 冠军（复制）
    │   ├── world_stats.csv         # world 统计（复制）
    │   ├── score_stats.csv         # 分数统计（复制）
    │   ├── scores.csv              # 详细分数（复制）
    │   └── tracker_logs/           # tracker 数据
    │       ├── agent_xxx.json
    │       └── tracker_summary.json
    └── 20251128_125624_std/
        └── ...
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import csv


def get_history_dir() -> Path:
    """获取 tournament_history 目录路径"""
    # 假设在项目根目录下
    # 尝试找到项目根目录
    current = Path.cwd()
    
    # 检查是否在项目目录中
    if (current / "runners").exists() or (current / "scml_analyzer").exists():
        return current / "tournament_history"
    
    # 否则使用当前目录
    return current / "tournament_history"


def generate_tournament_id(negmas_name: str, track: str) -> str:
    """从 negmas tournament 名称生成简洁的 ID
    
    Args:
        negmas_name: 如 "20251128H130949613919Kqg-stage-0001"
        track: "oneshot" 或 "std"
    
    Returns:
        如 "20251128_130949_oneshot"
    """
    try:
        # 提取日期时间部分
        date_part = negmas_name[:8]  # 20251128
        time_part = negmas_name[9:15]  # 130949
        return f"{date_part}_{time_part}_{track}"
    except:
        # 如果解析失败，使用时间戳
        return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{track}"


def import_tournament(
    negmas_dir: str,
    tracker_dir: Optional[str] = None,
    history_dir: Optional[str] = None,
    copy_mode: bool = True,  # True=复制, False=移动
) -> str:
    """
    将 negmas tournament 数据导入到 tournament_history
    
    Args:
        negmas_dir: negmas tournament 目录路径
        tracker_dir: tracker 日志目录路径（可选）
        history_dir: 目标 history 目录（可选，默认为项目下的 tournament_history）
        copy_mode: True 复制文件，False 移动文件
    
    Returns:
        导入后的目录路径
    """
    negmas_path = Path(negmas_dir)
    
    if not negmas_path.exists():
        raise ValueError(f"negmas 目录不存在: {negmas_dir}")
    
    # 加载 params.json 获取信息
    params_file = negmas_path / "params.json"
    if not params_file.exists():
        raise ValueError(f"无效的 tournament 目录，缺少 params.json: {negmas_dir}")
    
    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    # 确定赛道类型
    track = "oneshot" if params.get("oneshot_world") else "std"
    
    # 生成目录 ID
    tournament_id = generate_tournament_id(params.get("name", "unknown"), track)
    
    # 确定目标目录
    history_path: Path
    if history_dir is None:
        history_path = get_history_dir()
    else:
        history_path = Path(history_dir)
    
    target_dir = history_path / tournament_id
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 需要复制的文件列表
    files_to_copy = [
        "params.json",
        "total_scores.csv",
        "winners.csv",
        "world_stats.csv",
        "score_stats.csv",
        "scores.csv",
        "agent_stats.csv",
        "type_stats.csv",
    ]
    
    # 复制/移动文件
    operation = shutil.copy2 if copy_mode else shutil.move
    
    for filename in files_to_copy:
        src = negmas_path / filename
        if src.exists():
            dst = target_dir / filename
            operation(str(src), str(dst))
    
    # 复制 tracker 数据
    if tracker_dir:
        tracker_path = Path(tracker_dir)
        if tracker_path.exists():
            target_tracker = target_dir / "tracker_logs"
            if target_tracker.exists():
                shutil.rmtree(target_tracker)
            if copy_mode:
                shutil.copytree(str(tracker_path), str(target_tracker))
            else:
                shutil.move(str(tracker_path), str(target_tracker))
    
    # 创建 tournament_info.json（元信息）
    info = create_tournament_info(target_dir, params, negmas_dir)
    info_file = target_dir / "tournament_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    return str(target_dir)


def create_tournament_info(target_dir: Path, params: Dict, source_dir: str) -> Dict:
    """创建比赛元信息"""
    
    # 从 world_stats.csv 计算统计
    n_completed = 0
    total_duration = 0.0
    world_stats_file = target_dir / "world_stats.csv"
    if world_stats_file.exists():
        with open(world_stats_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_completed += 1
                try:
                    total_duration += float(row.get("execution_time", 0))
                except (ValueError, TypeError):
                    pass
    
    # 从 winners.csv 获取冠军
    winner = None
    winner_score = None
    winners_file = target_dir / "winners.csv"
    if winners_file.exists():
        with open(winners_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                winner = _extract_short_name(row.get("agent_type", ""))
                try:
                    winner_score = float(row.get("score", 0))
                except:
                    pass
                break
    
    # 提取参赛者
    competitors = [_extract_short_name(c) for c in params.get("competitors", [])]
    
    info = {
        "id": target_dir.name,
        "source_dir": source_dir,
        "imported_at": datetime.now().isoformat(),
        
        # 比赛类型
        "track": "oneshot" if params.get("oneshot_world") else "std",
        
        # 比赛设置
        "settings": {
            "n_configs": params.get("n_configs"),
            "n_runs_per_world": params.get("n_runs_per_world", 1),
            "n_steps": params.get("n_steps"),
            "n_worlds": params.get("n_worlds"),
            "n_processes": params.get("n_processes"),
            "parallelism": params.get("parallelism"),
            "world_generator": params.get("world_generator_name"),
            "score_calculator": params.get("score_calculator_name"),
            
            # OneShot 特定设置
            "publish_exogenous_summary": params.get("publish_exogenous_summary"),
            "publish_trading_prices": params.get("publish_trading_prices"),
            
            # 其他设置
            "min_factories_per_level": params.get("min_factories_per_level"),
            "n_agents_per_competitor": params.get("n_agents_per_competitor"),
            "n_competitors_per_world": params.get("n_competitors_per_world"),
        },
        
        # 参赛者
        "competitors": competitors,
        "n_competitors": len(competitors),
        
        # 结果
        "results": {
            "n_completed": n_completed,
            "total_duration_seconds": total_duration,
            "winner": winner,
            "winner_score": winner_score,
        },
        
        # 时间戳（从目录名提取）
        "timestamp": _extract_timestamp_from_id(target_dir.name),
    }
    
    return info


def _extract_short_name(full_name: str) -> str:
    """从完整类型名提取简短名称"""
    if ":" in full_name:
        full_name = full_name.split(":")[-1]
    return full_name.split(".")[-1]


def _extract_timestamp_from_id(tournament_id: str) -> str:
    """从 tournament ID 提取时间戳"""
    # ID 格式: 20251128_130949_oneshot
    try:
        parts = tournament_id.split("_")
        if len(parts) >= 2:
            date_part = parts[0]  # 20251128
            time_part = parts[1]  # 130949
            return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
    except:
        pass
    return ""


def list_tournaments(history_dir: Optional[str] = None) -> List[Dict]:
    """列出所有已导入的比赛"""
    history_path: Path
    if history_dir is None:
        history_path = get_history_dir()
    else:
        history_path = Path(history_dir)
    
    tournaments = []
    
    if not history_path.exists():
        return tournaments
    
    for item in history_path.iterdir():
        if not item.is_dir():
            continue
        
        info_file = item / "tournament_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                info["path"] = str(item)
                tournaments.append(info)
            except:
                continue
    
    # 按时间戳排序（最新的在前）
    tournaments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return tournaments


def get_tournament(tournament_id: str, history_dir: Optional[str] = None) -> Optional[Dict]:
    """获取特定比赛的信息"""
    history_path: Path
    if history_dir is None:
        history_path = get_history_dir()
    else:
        history_path = Path(history_dir)
    
    tournament_path = history_path / tournament_id
    info_file = tournament_path / "tournament_info.json"
    
    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        info["path"] = str(tournament_path)
        return info
    
    return None


def find_matching_tracker_dir(negmas_dir: str, results_base: Optional[str] = None) -> Optional[str]:
    """
    根据 negmas tournament 目录自动查找对应的 tracker 日志目录
    
    匹配逻辑：从 negmas 目录名提取时间戳，在 results 目录中查找匹配的目录
    
    Args:
        negmas_dir: negmas tournament 目录路径
        results_base: results 基础目录（默认为项目下的 results）
    
    Returns:
        匹配的 tracker_logs 目录路径，或 None
    """
    negmas_path = Path(negmas_dir)
    negmas_name = negmas_path.name  # 如 "20251128H130949613919Kqg-stage-0001"
    
    # 提取时间戳 (YYYYMMDD_HHMMSS)
    try:
        date_part = negmas_name[:8]  # 20251128
        time_part = negmas_name[9:15]  # 130949
        timestamp_pattern = f"{date_part}_{time_part}"  # 20251128_130949
    except:
        return None
    
    # 确定 results 目录
    results_path: Path
    if results_base is None:
        # 尝试找到项目目录
        current = Path.cwd()
        if (current / "results").exists():
            results_path = current / "results"
        else:
            return None
    else:
        results_path = Path(results_base)
    
    if not results_path.exists():
        return None
    
    # 在 results 目录中查找匹配的目录
    for item in results_path.iterdir():
        if not item.is_dir():
            continue
        
        # 检查目录名是否包含时间戳
        # 格式如: oneshot_quick_20251128_130949 或 std_quick_20251128_130949
        if timestamp_pattern in item.name:
            tracker_logs = item / "tracker_logs"
            if tracker_logs.exists():
                return str(tracker_logs)
    
    return None


def auto_import_tournament(
    negmas_dir: str,
    results_base: Optional[str] = None,
    history_dir: Optional[str] = None,
) -> str:
    """
    自动导入比赛数据，自动匹配 negmas 和 tracker 数据
    
    Args:
        negmas_dir: negmas tournament 目录路径
        results_base: results 基础目录（用于查找 tracker 数据）
        history_dir: 目标 history 目录
    
    Returns:
        导入后的目录路径
    """
    # 自动查找匹配的 tracker 目录
    tracker_dir = find_matching_tracker_dir(negmas_dir, results_base)
    
    # 执行导入
    return import_tournament(
        negmas_dir=negmas_dir,
        tracker_dir=tracker_dir,
        history_dir=history_dir,
        copy_mode=True,  # 保留原始数据
    )


def scan_and_import_all(
    negmas_tournaments_dir: Optional[str] = None,
    results_base: Optional[str] = None,
    history_dir: Optional[str] = None,
    force_reimport: bool = False,
) -> List[str]:
    """
    扫描 negmas tournaments 目录，导入所有比赛
    
    Args:
        negmas_tournaments_dir: negmas tournaments 目录（默认 ~/negmas/tournaments）
        results_base: results 目录
        history_dir: 目标 history 目录
        force_reimport: 是否强制重新导入已存在的比赛
    
    Returns:
        导入的目录列表
    """
    # 确定 negmas tournaments 目录
    negmas_path: Path
    if negmas_tournaments_dir is None:
        negmas_path = Path.home() / "negmas" / "tournaments"
    else:
        negmas_path = Path(negmas_tournaments_dir)
    
    if not negmas_path.exists():
        return []
    
    # 确定 history 目录
    history_path: Path
    if history_dir is None:
        history_path = get_history_dir()
    else:
        history_path = Path(history_dir)
    
    # 获取已导入的比赛 ID
    existing_ids = set()
    if not force_reimport and history_path.exists():
        for item in history_path.iterdir():
            if item.is_dir():
                existing_ids.add(item.name)
    
    imported = []
    
    for item in negmas_path.iterdir():
        if not item.is_dir():
            continue
        
        params_file = item / "params.json"
        if not params_file.exists():
            continue
        
        # 检查是否已导入
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            track = "oneshot" if params.get("oneshot_world") else "std"
            tournament_id = generate_tournament_id(params.get("name", "unknown"), track)
            
            if tournament_id in existing_ids:
                continue
            
            # 执行导入
            target = auto_import_tournament(
                negmas_dir=str(item),
                results_base=results_base,
                history_dir=history_dir,
            )
            imported.append(target)
            print(f"✅ 已导入: {tournament_id}")
            
        except Exception as e:
            print(f"⚠️ 导入失败 {item.name}: {e}")
            continue
    
    return imported


def get_rankings_from_history(tournament_path: str) -> List[Dict]:
    """从 history 目录读取排名数据"""
    path = Path(tournament_path)
    
    rankings = []
    
    # 读取 total_scores.csv
    scores_file = path / "total_scores.csv"
    if scores_file.exists():
        with open(scores_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                agent_type = row.get("agent_type", "")
                rankings.append({
                    "agent": _extract_short_name(agent_type),
                    "agent_type": agent_type,
                    "score": float(row.get("score", 0)),
                    "count": int(row.get("count", 0)),
                    "mean": float(row.get("mean", 0)),
                    "std": float(row.get("std", 0)),
                    "min": float(row.get("min", 0)),
                    "max": float(row.get("max", 0)),
                })
    
    # 按得分排序
    rankings.sort(key=lambda x: x["score"], reverse=True)
    
    return rankings


# CLI 入口
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SCML Tournament History Manager")
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # import 命令
    import_parser = subparsers.add_parser("import", help="导入单个比赛")
    import_parser.add_argument("negmas_dir", help="negmas tournament 目录")
    import_parser.add_argument("--tracker-dir", help="tracker 日志目录（可选，自动匹配）")
    import_parser.add_argument("--history-dir", help="目标 history 目录")
    
    # import-all 命令
    import_all_parser = subparsers.add_parser("import-all", help="导入所有比赛")
    import_all_parser.add_argument("--negmas-dir", help="negmas tournaments 目录")
    import_all_parser.add_argument("--results-dir", help="results 目录")
    import_all_parser.add_argument("--history-dir", help="目标 history 目录")
    import_all_parser.add_argument("--force", action="store_true", help="强制重新导入")
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="列出所有已导入的比赛")
    list_parser.add_argument("--history-dir", help="history 目录")
    
    args = parser.parse_args()
    
    if args.command == "import":
        tracker_dir = args.tracker_dir
        if tracker_dir is None:
            tracker_dir = find_matching_tracker_dir(args.negmas_dir)
            if tracker_dir:
                print(f"🔍 自动匹配 tracker: {tracker_dir}")
        
        result = import_tournament(
            negmas_dir=args.negmas_dir,
            tracker_dir=tracker_dir,
            history_dir=args.history_dir,
        )
        print(f"✅ 已导入到: {result}")
        
    elif args.command == "import-all":
        imported = scan_and_import_all(
            negmas_tournaments_dir=args.negmas_dir,
            results_base=args.results_dir,
            history_dir=args.history_dir,
            force_reimport=args.force,
        )
        print(f"\n共导入 {len(imported)} 场比赛")
        
    elif args.command == "list":
        tournaments = list_tournaments(args.history_dir)
        if not tournaments:
            print("暂无导入的比赛")
        else:
            print(f"共 {len(tournaments)} 场比赛:\n")
            for t in tournaments:
                print(f"  {t['timestamp']} | {t['track'].upper():7} | "
                      f"{t['results']['winner'] or 'N/A':15} | "
                      f"{t['results']['n_completed']} worlds | "
                      f"{t['id']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
