"""运行标准赛道锦标赛并产生日志，用于 HRL 训练数据采集。

特性：
- 参赛名单包含：所有 LitaAgent（已存在的 tracked 版本，除 HRL 外）、PenguinAgent、
  SCML 2025 Standard 前 5 名（scml-agents），以及内置 RandomAgent。
- 启用 scml_analyzer Tracker 记录所有 LitaAgent 行为。
- 启用 log_negotiations/log_ufuns，输出到 tournament_history。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Type

import scml_agents
from negmas.tournaments import run_tournament
from scml.std.agents import RandomAgent, SyncAgent, DecayAgent
from scml.scml2024 import SCML2024StdWorld

# LitaAgent tracked 版本（排除 HRL）
from litaagent_std.litaagent_y import LitaAgentYTracked
from litaagent_std.litaagent_p import LitaAgentPTracked
from litaagent_std.litaagent_yr import LitaAgentYRTracked
from litaagent_std.litaagent_ys import LitaAgentYSTracked
from litaagent_std.litaagent_n import LitaAgentNTracked
from litaagent_std.litaagent_cir import LitaAgentCIRTracked
from litaagent_std.litaagent_cirs import LitaAgentCIRSTracked

try:
    from scml_analyzer.auto_tracker import TrackerConfig

    _TRACKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRACKER_AVAILABLE = False
    TrackerConfig = None


def _get_penguin_agent() -> List[Type]:
    """尝试加载 PenguinAgent（2024 标准赛道冠军）。"""
    penguins = []
    try:
        winners = scml_agents.get_agents(version=2024, track="std", winners_only=True, as_class=True)
        for cls in winners:
            if "Penguin" in cls.__name__:
                penguins.append(cls)
                break
    except Exception as exc:  # pragma: no cover - 容错
        print(f"[WARN] 加载 PenguinAgent 失败: {exc}")
    return penguins


def _get_top5_std2025() -> List[Type]:
    """加载 SCML 2025 Standard 前 5 代理（若不支持 top_n，则退化为 winners_only）。"""
    try:
        return scml_agents.get_agents(version=2025, track="std", top_n=5, as_class=True)
    except TypeError:
        try:
            return scml_agents.get_agents(version=2025, track="std", winners_only=True, as_class=True)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] 加载 2025 top5 失败: {exc}")
            return []


def build_competitors() -> List[Type]:
    competitors: List[Type] = []
    competitors.extend(
        [
            LitaAgentYTracked,
            LitaAgentPTracked,
            LitaAgentYRTracked,
            LitaAgentYSTracked,
            LitaAgentNTracked,
            LitaAgentCIRTracked,
            LitaAgentCIRSTracked,
        ]
    )
    competitors.extend(_get_penguin_agent())
    competitors.extend(_get_top5_std2025())
    competitors.extend([RandomAgent, SyncAgent, DecayAgent])

    # 去重保持顺序
    seen = set()
    unique = []
    for cls in competitors:
        if cls in seen:
            continue
        seen.add(cls)
        unique.append(cls)
    return unique


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.abspath(os.path.join("tournament_history", f"hrl_data_{timestamp}_std"))
    os.makedirs(save_path, exist_ok=True)

    # 配置 Tracker
    if _TRACKER_AVAILABLE:
        tracker_dir = os.path.join(save_path, "tracker_logs")
        os.makedirs(tracker_dir, exist_ok=True)
        TrackerConfig.configure(log_dir=tracker_dir, enabled=True)
        os.environ["SCML_TRACKER_LOG_DIR"] = tracker_dir
        print(f"[INFO] Tracker enabled, log dir: {tracker_dir}")
    else:
        print("[WARN] scml_analyzer 未安装，Tracker 功能不可用。")

    competitors = build_competitors()
    print(f"[INFO] 参赛代理数: {len(competitors)}")
    print([c.__name__ for c in competitors])

    results = run_tournament(
        name=f"LitaHRLData_{timestamp}",
        competitors=competitors,
        n_competitors=min(10, len(competitors)),
        n_runs_per_world=3,
        n_steps=95,  # 接近标准赛完整长度
        world_class=SCML2024StdWorld,
        log_negotiations=True,
        log_ufuns=True,
        save_path=save_path,
        verbosity=1,
        compact=False,
    )
    print(f"[INFO] 锦标赛完成，日志保存在 {results.log_path}")


if __name__ == "__main__":
    main()
