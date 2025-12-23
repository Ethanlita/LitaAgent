"""运行标准赛道锦标赛并产生日志，用于 HRL 训练数据采集。

特性：
- 参赛名单包含：所有 LitaAgent（动态创建 Tracked 版本，除 HRL 外）、
  SCML 2025 Standard 前 5 名（scml-agents）和 SCML 2024 Standard 前 5 名、
  RandomStdAgent/SyncRandomStdAgent。
- 启用 scml_analyzer Tracker 记录所有 LitaAgent 行为（包含 HRL-XF 完整字段）。
- 默认启用 log_negotiations/log_ufuns（可用 --no-csv 关闭大部分 CSV 以减轻 I/O）。
- 使用 loky 执行器避免并行死锁问题。
- 结束后自动归集数据，不启动浏览器。
- 支持后台运行并将输出重定向到日志文件。

安装：
    cd /path/to/LitaAgent
    pip install -e .

用法：
    # 默认运行（后台模式，输出到日志文件）
    python -m runners.hrl_data_runner
    
    # 前台运行（输出到终端）
    python -m runners.hrl_data_runner --foreground
    
    # 自定义规模
    python -m runners.hrl_data_runner --configs 3 --runs 1

    # 关闭大部分 CSV（仍保留最小 stats/params 等）
    python -m runners.hrl_data_runner --no-csv
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Type

# 重要：在导入 SCML 之前启用 loky 执行器，避免并行死锁
from runners.loky_patch import enable_loky_executor
enable_loky_executor()

from negmas.helpers.inout import load
from negmas.helpers.numeric import truncated_mean
from negmas.tournaments.tournaments import (
    ASSIGNED_CONFIGS_JSON_FILE,
    ASSIGNED_CONFIGS_PICKLE_FILE,
    RESULTS_FILE,
    evaluate_tournament,
    run_tournament,
)
import scml_agents
from scml.utils import anac2024_std, anac2024_std_world_generator, balance_calculator_std
from scml.std.agents import RandomStdAgent, SyncRandomStdAgent

# LitaAgent 基类（不使用硬编码的 *Tracked 版本，改用动态创建）
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_p import LitaAgentP
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_cir import LitaAgentCIR
from litaagent_std.tracker_mixin import create_tracked_agent

# LitaAgent 基类列表（用于动态创建 Tracked 版本）
# 注意：CIRS 和 YS 版本的基类与 CIR/YR 同名，暂时只使用主版本
LITA_AGENT_BASES = [
    LitaAgentY,
    LitaAgentP,
    LitaAgentYR,
    LitaAgentN,
    LitaAgentCIR,
]

# 明确指定的 Top 代理（优先于 scml_agents.get_agents）
EXPLICIT_STD_TOP2024: List[Tuple[str, str]] = [
    ("scml_agents.scml2024.standard.team_penguin.penguinagent", "PenguinAgent"),
    ("scml_agents.scml2024.standard.team_miyajima_std.cautious", "CautiousStdAgent"),
    ("scml_agents.scml2024.standard.team_181.dogagent", "DogAgent"),
    ("scml_agents.scml2024.standard.team_178.ax", "AX"),
    ("scml_agents.scml2024.standard.teamyuzuru.quick_decision_agent", "QuickDecisionAgent"),
]

EXPLICIT_STD_TOP2025: List[Tuple[str, str]] = [
    ("scml_agents.scml2025.standard.team_atsunaga.as0", "AS0"),
    ("scml_agents.scml2024.standard.team_penguin.penguinagent", "PenguinAgent"),
    ("scml_agents.scml2025.standard.team_253.master_sota_agent", "XenoSotaAgent"),
    ("scml_agents.scml2025.standard.team_254.ultra", "UltraSuperMiracleSoraFinalAgentZ"),
    ("scml_agents.scml2025.standard.team_280.price_trend", "PriceTrendStdAgent"),
]

try:
    from scml_analyzer.auto_tracker import TrackerConfig

    _TRACKER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRACKER_AVAILABLE = False
    TrackerConfig = None


def _filter_legacy_agents(agents: List[Type]) -> List[Type]:
    filtered: List[Type] = []
    for cls in agents:
        module = getattr(cls, "__module__", "")
        if "scml2020" in module.lower():
            continue
        filtered.append(cls)
    return filtered


def _is_oneshot_track_agent(cls: Type) -> bool:
    module_name = getattr(cls, "__module__", "")
    module_lower = module_name.lower()
    if "oneshot" in module_lower:
        return True
    
    module = sys.modules.get(module_name)
    doc = ""
    if module and getattr(module, "__doc__", None):
        doc = module.__doc__ or ""
    if not doc and getattr(cls, "__doc__", None):
        doc = cls.__doc__ or ""
    
    doc_l = doc.lower()
    oneshot_marker = ("oneshot track" in doc_l) or ("one-shot track" in doc_l)
    std_marker = "standard track" in doc_l
    if oneshot_marker:
        return True
    if std_marker:
        return False
    
    # 尝试读取文件头部注释（仅做一次性判断）
    module_path = getattr(module, "__file__", None) if module else None
    if module_path and os.path.exists(module_path):
        try:
            with open(module_path, "r", encoding="utf-8") as f:
                head = "\n".join([next(f) for _ in range(12)])
            head_l = head.lower()
            if "oneshot track" in head_l or "one-shot track" in head_l:
                return True
            if "standard track" in head_l:
                return False
        except Exception:
            pass

    # 模块路径含 standard 时，默认视为 Standard 代理（避免误杀 StdTrack 中的 OneShot 基类实现）
    if "standard" in module_lower:
        return False

    # 兜底：无法判断时不当作 OneShot
    return False


def _filter_oneshot_track_agents(agents: List[Type]) -> List[Type]:
    filtered: List[Type] = []
    removed: List[str] = []
    for cls in agents:
        if _is_oneshot_track_agent(cls):
            removed.append(cls.__name__)
            continue
        filtered.append(cls)
    if removed:
        print(f"[WARN] 已排除 OneShot 代理: {sorted(removed)}")
    return filtered


def _load_explicit_agents(entries: List[Tuple[str, str]]) -> List[Type]:
    agents: List[Type] = []
    missing: List[str] = []
    for module_path, class_name in entries:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            agents.append(cls)
        except Exception:
            missing.append(f"{module_path}.{class_name}")
    if missing:
        print(f"[WARN] 未能导入指定 Top 代理: {missing}")
    return agents


def _merge_unique_agents(primary: List[Type], fallback: List[Type], max_top: int) -> List[Type]:
    unique: List[Type] = []
    seen = set()
    
    def _add(cls: Type) -> None:
        key = (cls.__module__, cls.__name__)
        if key in seen:
            return
        seen.add(key)
        unique.append(cls)
    
    for cls in primary:
        _add(cls)
    if len(unique) < max_top:
        for cls in fallback:
            _add(cls)
            if len(unique) >= max_top:
                break
    return unique[:max_top]


def _is_penguin_agent(cls: Type) -> bool:
    name = getattr(cls, "__name__", "") or ""
    module = getattr(cls, "__module__", "") or ""
    if name == "PenguinAgent":
        return True
    module_l = module.lower()
    name_l = name.lower()
    return ("team_penguin" in module_l) and ("penguin" in name_l)


def _maybe_track_agent(
    cls: Type,
    tracker_log_dir: str,
    track_only_penguin: bool,
) -> Type:
    if track_only_penguin and (not _is_penguin_agent(cls)):
        return cls
    tracked_cls = create_tracked_agent(cls, log_dir=tracker_log_dir)
    if tracked_cls is cls or tracked_cls.__name__ == cls.__name__:
        raise RuntimeError(f"无法为 {cls.__name__} 创建动态 Tracked 版本")
    return tracked_cls


def _get_top5_std2025(
    tracker_log_dir: str,
    max_top: int = 5,
    track_only_penguin: bool = False,
) -> List[Type]:
    """加载 SCML 2025 Standard 前 5 代理，并用 Tracker 包装。"""
    explicit = _load_explicit_agents(EXPLICIT_STD_TOP2025)
    explicit = _filter_legacy_agents(explicit)
    explicit = _filter_oneshot_track_agents(explicit)
    
    fallback: List[Type] = []
    if len(explicit) < max_top:
        try:
            fallback = list(scml_agents.get_agents(version=2025, track="std", top_only=max_top, as_class=True))
        except TypeError:
            try:
                fallback = list(scml_agents.get_agents(version=2025, track="std", winners_only=True, as_class=True))
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] 加载 2025 top5 失败: {exc}")
                fallback = []
        fallback = _filter_legacy_agents(fallback)
        fallback = _filter_oneshot_track_agents(fallback)
    
    agents = _merge_unique_agents(explicit, fallback, max_top)
    if len(agents) < max_top:
        print(f"[WARN] 2025 Std Top 代理不足: {len(agents)}/{max_top}")
    # 用 Tracker 包装所有代理
    wrapped_agents: List[Type] = []
    for cls in agents:
        wrapped_agents.append(_maybe_track_agent(cls, tracker_log_dir, track_only_penguin))
    return wrapped_agents


def _get_top5_std2024(
    tracker_log_dir: str,
    max_top: int = 5,
    track_only_penguin: bool = False,
) -> List[Type]:
    """加载 SCML 2024 Standard 前 5 代理，并用 Tracker 包装。"""
    explicit = _load_explicit_agents(EXPLICIT_STD_TOP2024)
    explicit = _filter_legacy_agents(explicit)
    explicit = _filter_oneshot_track_agents(explicit)
    
    fallback: List[Type] = []
    if len(explicit) < max_top:
        try:
            fallback = list(scml_agents.get_agents(version=2024, track="std", top_only=max_top, as_class=True))
        except TypeError:
            try:
                fallback = list(scml_agents.get_agents(version=2024, track="std", winners_only=True, as_class=True))
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] 加载 2024 top5 失败: {exc}")
                fallback = []
        fallback = _filter_legacy_agents(fallback)
        fallback = _filter_oneshot_track_agents(fallback)
    
    agents = _merge_unique_agents(explicit, fallback, max_top)
    if len(agents) < max_top:
        print(f"[WARN] 2024 Std Top 代理不足: {len(agents)}/{max_top}")
    wrapped_agents: List[Type] = []
    for cls in agents:
        wrapped_agents.append(_maybe_track_agent(cls, tracker_log_dir, track_only_penguin))
    return wrapped_agents


def build_competitors(
    tracker_log_dir: str,
    max_top_2025: int = 5,
    max_top_2024: int = 5,
    track_only_penguin: bool = False,
) -> Tuple[List[Type], List[str], List[str]]:
    """构建参赛代理列表，所有 LitaAgent 使用动态创建的 Tracked 版本。
    
    Args:
        tracker_log_dir: Tracker 日志目录路径
        
    Returns:
        (competitors, lita_names, external_names)
    """
    if not _TRACKER_AVAILABLE:
        raise RuntimeError("必须安装 scml_analyzer 以启用全量 Tracker")

    competitors: List[Type] = []
    lita_agents: List[Type] = []
    
    # 动态创建 LitaAgent Tracked 版本（包含完整 HRL-XF 字段）
    for base_cls in LITA_AGENT_BASES:
        wrapped_cls = _maybe_track_agent(base_cls, tracker_log_dir, track_only_penguin)
        competitors.append(wrapped_cls)
        lita_agents.append(wrapped_cls)
        if wrapped_cls is base_cls:
            print(f"[INFO] 使用未追踪版本: {base_cls.__name__}")
        else:
            print(f"[INFO] 动态创建 Tracked 版本: {wrapped_cls.__name__}")
    
    top_agents_2025 = _get_top5_std2025(
        tracker_log_dir,
        max_top=max_top_2025,
        track_only_penguin=track_only_penguin,
    )
    top_agents_2024 = _get_top5_std2024(
        tracker_log_dir,
        max_top=max_top_2024,
        track_only_penguin=track_only_penguin,
    )
    lita_base_names = {c.__name__ for c in LITA_AGENT_BASES}
    top_agents = [
        cls for cls in list(top_agents_2025) + list(top_agents_2024)
        if cls.__name__ not in lita_base_names
    ]
    competitors.extend(top_agents)

    # 若启用“仅追踪 PenguinAgent”，但由于 max_top=0 等原因未包含 Penguin，则强制加入
    if track_only_penguin:
        have_penguin = any("penguinagent" in (getattr(c, "__name__", "") or "").lower() for c in competitors)
        if not have_penguin:
            forced = _load_explicit_agents([
                ("scml_agents.scml2024.standard.team_penguin.penguinagent", "PenguinAgent"),
            ])
            forced = _filter_legacy_agents(forced)
            forced = _filter_oneshot_track_agents(forced)
            if forced:
                penguin_cls = forced[0]
                competitors.append(_maybe_track_agent(penguin_cls, tracker_log_dir, track_only_penguin))
                print("[INFO] 已强制加入 PenguinAgent（track_only_penguin=True）")
            else:
                print("[WARN] track_only_penguin=True 但未能导入 PenguinAgent，可能导致 tracker_logs 为空")
    
    # 若启用“仅追踪 PenguinAgent”，但参赛名单里没有 Penguin，则强制加入
    if track_only_penguin:
        have_penguin = any("penguinagent" in (getattr(c, "__name__", "") or "").lower() for c in competitors)
        if not have_penguin:
            forced = _load_explicit_agents([
                ("scml_agents.scml2024.standard.team_penguin.penguinagent", "PenguinAgent"),
            ])
            forced = _filter_legacy_agents(forced)
            forced = _filter_oneshot_track_agents(forced)
            if forced:
                penguin_cls = forced[0]
                competitors.append(_maybe_track_agent(penguin_cls, tracker_log_dir, track_only_penguin))
                print("[INFO] 已强制加入 PenguinAgent（track_only_penguin=True）")
            else:
                print("[WARN] track_only_penguin=True 但未能导入 PenguinAgent，可能导致 tracker_logs 为空")

    # 内置基线代理（Random/SyncRandom）：默认也追踪；track_only_penguin 下不追踪
    extra_agents = [RandomStdAgent, SyncRandomStdAgent]
    for cls in extra_agents:
        competitors.append(_maybe_track_agent(cls, tracker_log_dir, track_only_penguin))

    # 去重保持顺序
    seen = set()
    unique = []
    for cls in competitors:
        key = (cls.__module__, cls.__name__)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cls)
    
    base_names = [c.__name__ for c in LITA_AGENT_BASES]
    def _is_lita(name: str) -> bool:
        return any(name.startswith(base) for base in base_names)
    
    lita_names = [c.__name__ for c in unique if _is_lita(c.__name__)]
    external_names = [c.__name__ for c in unique if not _is_lita(c.__name__)]
    return unique, lita_names, external_names


def _estimate_competitor_sets(
    n_competitors: int,
    n_per_world: int,
    round_robin: bool,
) -> int:
    if n_per_world >= n_competitors:
        return 1
    if round_robin:
        return math.comb(n_competitors, n_per_world)
    return math.ceil(n_competitors / n_per_world)


def _strip_adapter_prefix(agent_type: str) -> str:
    if not isinstance(agent_type, str):
        return agent_type
    if "DefaultOneShotAdapter" in agent_type and ":" in agent_type:
        return agent_type.split(":", 1)[1]
    if "DefaultStdAdapter" in agent_type and ":" in agent_type:
        return agent_type.split(":", 1)[1]
    return agent_type


def _patch_score_calculator() -> None:
    import scml.utils as scml_utils

    if getattr(scml_utils, "_litaagent_score_patch", False):
        return

    original = scml_utils.balance_calculator_std

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            if result is not None and getattr(result, "types", None):
                result.types = [_strip_adapter_prefix(t) for t in result.types]
        except Exception:
            pass
        return result

    scml_utils.balance_calculator_std = wrapped
    scml_utils._litaagent_score_patch = True


def _calc_max_worlds_per_config(
    target_worlds: int,
    n_configs: int,
    n_runs: int,
    n_competitors: int,
    n_per_world: int,
    round_robin: bool,
) -> int:
    n_sets = _estimate_competitor_sets(n_competitors, n_per_world, round_robin)
    denom = max(1, n_configs * n_runs * n_sets)
    return max(1, math.ceil(target_worlds / denom))


def _has_existing_tournament(tournament_dir: Path) -> bool:
    return any(
        (tournament_dir / fname).exists()
        for fname in (
            ASSIGNED_CONFIGS_PICKLE_FILE,
            ASSIGNED_CONFIGS_JSON_FILE,
            "assigned_configs",
        )
    )


def _load_assignments(tournament_dir: Path):
    for fname in (
        ASSIGNED_CONFIGS_PICKLE_FILE,
        ASSIGNED_CONFIGS_JSON_FILE,
        "assigned_configs",
    ):
        fpath = tournament_dir / fname
        if not fpath.exists():
            continue
        try:
            data = load(fpath)
            if data:
                return data
        except Exception:
            continue
    return []


def _summarize_progress(tournament_dir: Path) -> Tuple[int, int]:
    assignments = _load_assignments(tournament_dir)
    if not assignments:
        return 0, 0
    total = len(assignments)
    done = 0
    for config_set in assignments:
        if not config_set:
            continue
        dir_name = config_set[0].get("__dir_name")
        if not dir_name:
            continue
        run_root = Path(dir_name).parent
        if (run_root / RESULTS_FILE).exists():
            done += 1
    return done, total


def _find_existing_tournament_root(base_dir: Path) -> Path | None:
    if _has_existing_tournament(base_dir):
        return base_dir
    stage_candidate = base_dir.parent / f"{base_dir.name}-stage-0001"
    if _has_existing_tournament(stage_candidate):
        return stage_candidate
    for p in base_dir.parent.glob(f"{base_dir.name}-stage-*"):
        if _has_existing_tournament(p):
            return p
    return None


def _resolve_tracker_dir(base_dir: Path, tournament_root: Path) -> Path:
    base_tracker = base_dir / "tracker_logs"
    if base_tracker.exists() or base_dir == tournament_root:
        return base_tracker
    return tournament_root / "tracker_logs"


def main():
    """主函数：解析参数并运行锦标赛。"""
    parser = argparse.ArgumentParser(
        description="HRL 训练数据采集 Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--configs", type=int, default=20, help="World 配置数量 (default: 20)")
    parser.add_argument("--runs", type=int, default=2, help="每配置运行次数 (default: 2)")
    parser.add_argument("--max-top-2025", type=int, default=5, help="2025 Top Agents 数量上限")
    parser.add_argument("--max-top-2024", type=int, default=5, help="2024 Top Agents 数量上限")
    parser.add_argument("--n-competitors-per-world", type=int, default=None, help="每个 world 参赛者数量（默认使用全部参赛者）")
    parser.add_argument("--max-worlds-per-config", type=int, default=None, help="限制每个配置的最大 world 数")
    parser.add_argument("--target-worlds", type=int, default=None, help="目标总 world 数（自动折算为 max_worlds_per_config）")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认自动生成）")
    parser.add_argument(
        "--resumable",
        "--resume",
        action="store_true",
        help="启用断点续跑（复用 --output-dir；若目录内已存在配置将自动续跑）",
    )
    parser.add_argument("--foreground", action="store_true", help="前台运行（输出到终端而非日志文件）")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式")
    parser.add_argument("--parallelism", type=str, default="loky", help="并行模式 (parallel/serial/dask/loky)")
    parser.add_argument(
        "--round-robin",
        dest="round_robin",
        action="store_true",
        help="启用 round-robin（保留官方全组合，默认开启）",
    )
    parser.add_argument(
        "--no-round-robin",
        dest="round_robin",
        action="store_false",
        help="禁用 round-robin（仅采样少量随机组合，运行更快）",
    )
    parser.set_defaults(round_robin=True)
    parser.add_argument("--steps", type=int, default=None, help="固定 n_steps（小规模快速验证用）")
    parser.add_argument(
        "--track-only-penguin",
        action="store_true",
        help="仅追踪 PenguinAgent（其它参赛者不写 Tracker JSON，节省磁盘/解析开销）",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="尽量关闭 negmas CSV 输出（仍会保留少量必要文件，如 stats/params）",
    )
    parser.add_argument("--no-auto-collect", action="store_true", help="禁用自动归集")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        save_path = Path(args.output_dir).resolve()
    else:
        save_path = Path("tournament_history") / f"hrl_data_{timestamp}_std"
        save_path = save_path.resolve()
    existing_root = None
    if args.resumable:
        existing_root = _find_existing_tournament_root(save_path)
        if existing_root is None and save_path.exists():
            if any(save_path.iterdir()):
                raise RuntimeError(
                    f"{save_path} 已存在但未发现配置文件，无法续跑。请更换 --output-dir 或手动清理。"
                )
    tournament_root = existing_root or save_path
    created_now = existing_root is None
    
    # 日志文件路径
    log_file = save_path / "tournament_run.log"
    
    # 如果非前台模式，重定向 stdout/stderr 到日志文件
    if not args.foreground:
        save_path.mkdir(parents=True, exist_ok=True)
        print("[INFO] 比赛将在后台运行")
        print(f"[INFO] 输出目录: {save_path}")
        print(f"[INFO] 日志文件: {log_file}")
        print(f"[INFO] 查看进度: tail -f {log_file}")
        
        # 重定向输出到日志文件
        log_mode = "a" if args.resumable and log_file.exists() else "w"
        log_handle = open(log_file, log_mode, buffering=1, encoding="utf-8")
        sys.stdout = log_handle
        sys.stderr = log_handle
        print(f"[INFO] 锦标赛开始于 {timestamp}")
        print(f"[INFO] 配置: configs={args.configs}, runs={args.runs}")
        if args.resumable:
            print(f"[INFO] resumable=True, 目标目录: {save_path}")
            if existing_root:
                print(f"[INFO] 已发现可续跑目录: {existing_root}")

    # 配置 Tracker（必须启用）
    if not _TRACKER_AVAILABLE:
        raise RuntimeError("必须安装 scml_analyzer 以启用全量 Tracker")
    tracker_dir = _resolve_tracker_dir(save_path, Path(tournament_root))
    tracker_dir.mkdir(parents=True, exist_ok=True)
    TrackerConfig.configure(log_dir=str(tracker_dir), enabled=True)
    os.environ["SCML_TRACKER_LOG_DIR"] = str(tracker_dir)
    print(f"[INFO] Tracker enabled, log dir: {tracker_dir}")
    if args.track_only_penguin:
        print("[INFO] Tracker 过滤模式：仅追踪 PenguinAgent")

    competitors, lita_names, external_names = build_competitors(
        str(tracker_dir),
        max_top_2025=args.max_top_2025,
        max_top_2024=args.max_top_2024,
        track_only_penguin=args.track_only_penguin,
    )
    _patch_score_calculator()
    n_per_world = args.n_competitors_per_world
    if n_per_world is not None:
        if not args.round_robin and len(competitors) % n_per_world != 0:
            raise RuntimeError(
                f"n_competitors_per_world={n_per_world} 不能整除参赛数量 {len(competitors)}，"
                f"请调整或启用 --round-robin"
            )
        if (
            args.max_worlds_per_config is None
            and args.target_worlds is None
        ):
            args.max_worlds_per_config = n_per_world
    if args.target_worlds and args.max_worlds_per_config is None:
        if n_per_world is None:
            raise RuntimeError("使用 --target-worlds 时必须指定 --n-competitors-per-world")
        args.max_worlds_per_config = _calc_max_worlds_per_config(
            args.target_worlds,
            args.configs,
            args.runs,
            len(competitors),
            n_per_world,
            args.round_robin,
        )
        if (
            args.round_robin
            and n_per_world is not None
            and args.max_worlds_per_config < n_per_world
        ):
            args.max_worlds_per_config = n_per_world
            print(f"[WARN] round_robin 下 max_worlds_per_config 需 >= {n_per_world}，已自动提升")
    parallelism_label = args.parallelism
    parallelism = args.parallelism
    if args.parallelism.startswith("loky"):
        os.environ["SCML_PARALLELISM"] = args.parallelism
        parallelism_label = f"{args.parallelism} (via SCML_PARALLELISM)"
        parallelism = "parallel"

    print("\n" + "=" * 60)
    print("🎯 SCML 2025 Standard 数据采集 Runner")
    print("=" * 60)
    print(f"📋 参赛代理: {len(competitors)} 个")
    print(f"   LitaAgent: {lita_names}")
    print(f"   外部 Agent: {external_names}")
    print(f"📊 配置: n_configs={args.configs}, n_runs={args.runs}")
    if args.max_worlds_per_config is not None and n_per_world is not None:
        n_sets = _estimate_competitor_sets(len(competitors), n_per_world, args.round_robin)
        approx_worlds = args.configs * args.runs * args.max_worlds_per_config * n_sets
        print(f"🧮 约束: max_worlds_per_config={args.max_worlds_per_config} (≈ {approx_worlds} worlds)")
    print(
        f"🔧 选项: tracker=True, visualizer=False, auto_collect={not args.no_auto_collect}, "
        f"round_robin={args.round_robin}, no_csv={args.no_csv}"
    )
    print(f"⚙️  并行: {parallelism_label}")
    print("=" * 60 + "\n")

    # 使用 anac2024_std 运行标准赛，强制保留日志以便 HRL 数据采集。
    tournament_kwargs = {}
    if args.steps is not None:
        tournament_kwargs["n_steps"] = args.steps
    if args.no_csv:
        # 尽量关闭 negmas 侧的 CSV 输出（保留最小必要文件）
        tournament_kwargs.update(
            {
                "log_ufuns": False,
                "log_negotiations": False,
                "save_signed_contracts": True,
                "save_cancelled_contracts": False,
                "save_negotiations": False,
                "save_resolved_breaches": False,
                "save_unresolved_breaches": False,
                "saved_details_level": 0,
                "log_stats_every": 0,
            }
        )

    if args.resumable:
        if created_now:
            if save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 生成可续跑配置: {save_path}")
            configs_path = anac2024_std(
                competitors=competitors,
                n_configs=args.configs,
                n_runs_per_world=args.runs,
                n_competitors_per_world=n_per_world,
                max_worlds_per_config=args.max_worlds_per_config,
                tournament_path=str(save_path.parent),
                forced_logs_fraction=1.0,
                parallelism=parallelism,
                round_robin=args.round_robin,
                name=save_path.name,
                verbose=not args.quiet,
                compact=False,
                configs_only=True,
                print_exceptions=True,
                **tournament_kwargs,
            )
            try:
                if configs_path is not None:
                    configs_path = Path(configs_path)
                    tournament_root = configs_path.parent
            except Exception:
                pass
        done, total = _summarize_progress(Path(tournament_root))
        if total:
            print(f"[INFO] 进度: {done}/{total} world 已完成 ({done/total:.1%})")
        print(f"[INFO] 启动/恢复比赛: {tournament_root}")
        run_tournament(
            tournament_path=str(tournament_root),
            world_generator=anac2024_std_world_generator,
            score_calculator=balance_calculator_std,
            parallelism=parallelism,
            verbose=not args.quiet,
            compact=False,
            print_exceptions=True,
        )
        print("[INFO] 汇总结果")
        results = evaluate_tournament(
            tournament_path=str(tournament_root),
            metric=truncated_mean,
            verbose=not args.quiet,
            recursive=True,
        )
    else:
        results = anac2024_std(
            competitors=competitors,
            n_configs=args.configs,
            n_runs_per_world=args.runs,
            n_competitors_per_world=n_per_world,
            max_worlds_per_config=args.max_worlds_per_config,
            tournament_path=str(save_path),
            forced_logs_fraction=1.0,
            parallelism=parallelism,
            round_robin=args.round_robin,
            name=f"LitaHRLData_{timestamp}",
            verbose=not args.quiet,
            compact=False,
            print_exceptions=True,
            **tournament_kwargs,
        )
    
    print(f"[INFO] 锦标赛完成，日志保存在 {save_path}")
    if args.resumable and Path(tournament_root) != save_path:
        print(f"[INFO] 比赛目录: {tournament_root}")

    if not args.no_auto_collect:
        try:
            from scml_analyzer.postprocess import postprocess_tournament
            print("[INFO] 自动归集日志...")
            postprocess_tournament(
                output_dir=str(save_path),
                start_visualizer=False,
                visualizer_port=None,
            )
        except ImportError:
            print("[WARN] scml_analyzer.postprocess 不可用，跳过自动归集")
        except Exception as exc:
            print(f"[WARN] 自动归集失败: {exc}")
    
    # 如果重定向了输出，恢复并关闭
    if not args.foreground:
        log_handle.flush()
        log_handle.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"[INFO] 锦标赛完成！结果保存在: {save_path}")


if __name__ == "__main__":
    main()
