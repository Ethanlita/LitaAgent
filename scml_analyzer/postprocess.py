"""
SCML 比赛后处理模块

提供比赛完成后的数据处理功能：
- 将 negmas 数据和 tracker 日志移动到 tournament_history
- 启动 Visualizer 服务器

设计原则：
- 移动（而非复制）日志文件，避免多次比赛记录混淆
- Visualizer 不接受参数，自动从 tournament_history 读取
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from . import history
from .auto_tracker import TrackerManager


def postprocess_tournament(
    output_dir: str,
    start_visualizer: bool = True,
    visualizer_port: int = 8080,
) -> Optional[str]:
    """
    比赛后处理：保存 Tracker 数据，导入到 tournament_history，启动 Visualizer
    
    流程：
    1. 保存 Tracker 数据到 output_dir/tracker_logs/
    2. 查找最新的 negmas tournament 目录
    3. 将 negmas 数据 + tracker 日志移动到 tournament_history/
    4. 清理 output_dir 中的 tracker_logs（已移动）
    5. 启动 Visualizer（无参数）
    
    Args:
        output_dir: runner 的输出目录
        start_visualizer: 是否启动 Visualizer 服务器
        visualizer_port: Visualizer 服务器端口
    
    Returns:
        导入后的 tournament_history 路径，失败返回 None
    """
    print("\n" + "=" * 60)
    print("📦 比赛后处理")
    print("=" * 60)
    
    # 1. 保存 Tracker 数据
    print("\n💾 保存 Tracker 数据...")
    tracker_log_dir = os.path.join(output_dir, "tracker_logs")
    TrackerManager.save_all(tracker_log_dir)
    
    # 统计保存的文件
    if os.path.exists(tracker_log_dir):
        tracker_files = list(Path(tracker_log_dir).glob("agent_*.json"))
        print(f"  ✅ 已保存 {len(tracker_files)} 个 Agent 的追踪数据")
    else:
        print("  ⚠️ 没有 Tracker 数据")
    
    # 2. 查找最新的 negmas tournament 目录
    print("\n🔍 查找 negmas 比赛数据...")
    imported_path = None
    
    try:
        negmas_tournaments_dir = Path.home() / "negmas" / "tournaments"
        if negmas_tournaments_dir.exists():
            # 找到最新创建的目录
            tournament_dirs = [
                d for d in negmas_tournaments_dir.iterdir() 
                if d.is_dir() and (d / "params.json").exists()
            ]
            if tournament_dirs:
                # 按修改时间排序，取最新的
                latest_dir = max(tournament_dirs, key=lambda d: d.stat().st_mtime)
                print(f"  找到: {latest_dir.name}")
                
                # 3. 移动数据到 tournament_history（使用 move 模式）
                print("\n📂 移动数据到 tournament_history...")
                imported_path = history.import_tournament(
                    negmas_dir=str(latest_dir),
                    tracker_dir=tracker_log_dir if os.path.exists(tracker_log_dir) else None,
                    copy_mode=False,  # 移动而非复制！
                )
                print(f"  ✅ 已导入到: {imported_path}")
                
                # 4. 清理 output_dir 中的 tracker_logs（已移动）
                if os.path.exists(tracker_log_dir):
                    try:
                        shutil.rmtree(tracker_log_dir)
                        print(f"  🗑️ 已清理临时目录: {tracker_log_dir}")
                    except Exception as e:
                        print(f"  ⚠️ 清理临时目录失败: {e}")
            else:
                print("  ⚠️ 未找到 negmas 比赛数据")
        else:
            print(f"  ⚠️ negmas tournaments 目录不存在: {negmas_tournaments_dir}")
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 启动 Visualizer
    if start_visualizer:
        print("\n🌐 启动可视化服务器...")
        try:
            from .visualizer import start_server
            # 不传参数！Visualizer 自动从 tournament_history 读取
            start_server(port=visualizer_port, open_browser=True)
        except ImportError:
            print("  ⚠️ 无法导入 scml_analyzer.visualizer")
        except KeyboardInterrupt:
            print("\n👋 服务器已停止")
        except Exception as e:
            print(f"  ⚠️ 启动服务器失败: {e}")
    else:
        print("\n📌 提示: 使用以下命令启动可视化服务器:")
        print("  python -m scml_analyzer.visualizer")
    
    return imported_path


def import_existing_tournament(
    negmas_dir: str,
    tracker_dir: Optional[str] = None,
    move: bool = True,
) -> Optional[str]:
    """
    手动导入已有的比赛数据
    
    Args:
        negmas_dir: negmas tournament 目录路径
        tracker_dir: tracker 日志目录（可选）
        move: True 移动文件，False 复制文件
    
    Returns:
        导入后的路径
    """
    return history.import_tournament(
        negmas_dir=negmas_dir,
        tracker_dir=tracker_dir,
        copy_mode=not move,
    )
