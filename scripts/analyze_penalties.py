#!/usr/bin/env python
"""
分析 disposal penalty 和 shortfall penalty 的货币量纲对比
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_world(world_path: Path):
    """分析单个 world 的 penalty 情况"""
    # 找到 params.json
    params_file = world_path / "params.json"
    if not params_file.exists():
        return None
    
    with open(params_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    
    # 获取每个 agent 的信息
    # 需要从 info.json 获取详细信息
    info_file = world_path / "info.json"
    if not info_file.exists():
        return None
    
    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    return {
        "params": params,
        "info": info
    }


def main():
    # 从 tournament 目录读取 world 数据
    tournament_dir = Path("D:/SCML_initial/tournament_history/20260110_171712_oneshot")
    
    # 遍历所有 world 目录（假设在某个子目录下）
    # 先检查目录结构
    print(f"检查目录: {tournament_dir}")
    for item in tournament_dir.iterdir():
        print(f"  - {item.name}")
    
    # 尝试读取 scores.json 获取 world 信息
    scores_file = tournament_dir / "scores.json"
    if scores_file.exists():
        with open(scores_file, "r", encoding="utf-8") as f:
            scores = json.load(f)
        print(f"\n分数数据:")
        for k, v in scores.items():
            print(f"  {k}: {v}")
    
    # 读取 agent 日志文件，分析 penalty
    # 从根目录的 agent_*.json 文件读取
    root_dir = Path("D:/SCML_initial")
    agent_files = list(root_dir.glob("agent_*LOS*.json"))
    
    print(f"\n找到 {len(agent_files)} 个 LOS agent 日志文件")
    
    # 统计 disposal 和 shortfall
    seller_stats = {"disposal": [], "shortfall": [], "disposal_cost": [], "shortfall_penalty": []}
    buyer_stats = {"disposal": [], "shortfall": [], "disposal_cost": [], "shortfall_penalty": []}
    
    for agent_file in agent_files[:10]:  # 只分析前 10 个
        print(f"\n分析: {agent_file.name}")
        with open(agent_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                print(f"  无法解析")
                continue
        
        # 检查数据结构
        if isinstance(data, dict):
            keys = list(data.keys())[:20]
            print(f"  Keys (前20): {keys}")
            
            # 尝试找到相关字段
            if "awi" in data:
                awi = data["awi"]
                if isinstance(awi, dict):
                    print(f"  AWI keys: {list(awi.keys())[:10]}")
            
            if "history" in data:
                history = data["history"]
                if isinstance(history, list) and len(history) > 0:
                    first_step = history[0]
                    if isinstance(first_step, dict):
                        print(f"  History[0] keys: {list(first_step.keys())[:10]}")
        elif isinstance(data, list):
            print(f"  数据是列表，长度: {len(data)}")
            if len(data) > 0 and isinstance(data[0], dict):
                print(f"  data[0] keys: {list(data[0].keys())[:20]}")

if __name__ == "__main__":
    main()
