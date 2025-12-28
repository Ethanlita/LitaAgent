"""运行标准赛道锦标赛并产生日志，用于 HRL 训练数据采集。

特性：
- 参赛名单包含：所有 LitaAgent（动态创建 Tracked 版本，除 HRL 外）、
  SCML 2025 Standard 前 5 名（scml-agents）和 SCML 2024 Standard 前 5 名、
  RandomStdAgent/SyncRandomStdAgent。
- 启用 scml_analyzer Tracker 记录所有 LitaAgent 行为（包含 HRL-XF 完整字段）。
- 默认 forced_logs_fraction=0.1（可用 --forced-logs-fraction 调整强制日志比例）。
- 默认启用 log_negotiations/log_ufuns（可用 --no-csv 关闭大部分 CSV 以减轻 I/O）。
- 使用 loky 执行器避免并行死锁问题。
- 结束后自动归集数据，不启动浏览器。
- 可续跑场次结束后自动清理 resumable 中间数据以节省磁盘。
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

    # 调整强制日志比例（默认 0.1）
    python -m runners.hrl_data_runner --forced-logs-fraction 0.05
"""

from __future__ import annotations

import argparse
import copy
import importlib
import itertools
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Type

# 重要：在导入 SCML 之前启用 loky 执行器，避免并行死锁
from runners.loky_patch import enable_loky_executor
enable_loky_executor()

from negmas.helpers import get_full_type_name, shortest_unique_names, unique_name
from negmas.helpers.inout import dump, load
from negmas.helpers.numeric import truncated_mean
from negmas.tournaments.tournaments import (
    ASSIGNED_CONFIGS_JSON_FILE,
    ASSIGNED_CONFIGS_PICKLE_FILE,
    PARAMS_FILE,
    RESULTS_FILE,
    _hash,
    _divide_into_sets,
    _run_id,
    evaluate_tournament,
    run_tournament,
    to_file,
)
import scml_agents
from scml.utils import (
    DefaultAgentsOneShot,
    anac2024_config_generator_std,
    anac2024_std,
    anac2024_std_world_generator,
    anac_assigner_std,
    balance_calculator_std,
)
from scml.std.agents import RandomStdAgent, SyncRandomStdAgent

# LitaAgent 基类（不使用硬编码的 *Tracked 版本，改用动态创建）
from litaagent_std.litaagent_y import LitaAgentY
from litaagent_std.litaagent_p import LitaAgentP
from litaagent_std.litaagent_yr import LitaAgentYR
from litaagent_std.litaagent_n import LitaAgentN
from litaagent_std.litaagent_h import LitaAgentH
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
    LitaAgentH,
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


def _is_penguin_type_name(type_name: str) -> bool:
    if not type_name:
        return False
    name_l = type_name.lower()
    return ("penguinagent" in name_l) or ("team_penguin" in name_l)


def _build_penguin_only_competitor_sets(
    competitor_info: List[Tuple[str, dict]],
    n_competitors_per_world: int,
    round_robin: bool,
):
    penguin_info = None
    for info in competitor_info:
        if _is_penguin_type_name(info[0]):
            penguin_info = info
            break
    if penguin_info is None:
        return None

    if n_competitors_per_world <= 1:
        return [(penguin_info,)]

    others = [info for info in competitor_info if info is not penguin_info]
    if n_competitors_per_world - 1 > len(others):
        raise RuntimeError(
            f"n_competitors_per_world={n_competitors_per_world} 超过可用参赛者数量"
        )

    if round_robin:
        return (
            (penguin_info,) + comb
            for comb in itertools.combinations(others, n_competitors_per_world - 1)
        )

    comp_ind = list(range(len(competitor_info)))
    random.shuffle(comp_ind)
    competitor_sets = _divide_into_sets(comp_ind, n_competitors_per_world)
    competitor_sets = [[competitor_info[i] for i in lst] for lst in competitor_sets]
    return [
        cset
        for cset in competitor_sets
        if any(_is_penguin_type_name(info[0]) for info in cset)
    ]


def _create_penguin_only_tournament(
    tournament_path: Path,
    *,
    competitors: List[Type],
    competitor_params: List[dict] | None,
    n_competitors_per_world: int,
    round_robin: bool,
    n_configs: int,
    max_worlds_per_config: int | None,
    n_runs_per_world: int,
    config_generator,
    config_assigner,
    world_generator,
    score_calculator,
    total_timeout: int | None,
    parallelism: str,
    scheduler_ip: str | None,
    scheduler_port: str | None,
    non_competitors,
    non_competitor_params,
    dynamic_non_competitors,
    dynamic_non_competitor_params,
    exclude_competitors_from_reassignment: bool,
    verbose: bool,
    compact: bool,
    forced_logs_fraction: float,
    **kwargs,
) -> Path:
    tournament_path = tournament_path.resolve()
    if tournament_path.exists():
        raise ValueError(f"tournament path {tournament_path} exists. You cannot create two tournaments in the same place")
    tournament_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Results of Tournament {tournament_path.name} will be saved to {tournament_path}")

    if competitor_params is None:
        competitor_params = [dict() for _ in range(len(competitors))]
    competitors = [get_full_type_name(_) for _ in competitors]
    non_competitors = (
        None
        if non_competitors is None
        else tuple(get_full_type_name(_) for _ in non_competitors)
    )

    params = dict(
        competitors=competitors,
        competitor_params=competitor_params,
        non_competitors=non_competitors,
        non_competitor_params=non_competitor_params,
        n_agents_per_competitor=1,
        tournament_path=str(tournament_path),
        total_timeout=total_timeout,
        parallelism=parallelism,
        scheduler_ip=scheduler_ip,
        scheduler_port=scheduler_port,
        name=tournament_path.name,
        n_configs=n_configs,
        n_world_per_config=max_worlds_per_config,
        n_runs_per_world=n_runs_per_world,
        n_worlds=None,
        compact=compact,
        n_competitors_per_world=n_competitors_per_world,
        penguin_only=True,
    )
    params.update(kwargs)
    dump(params, tournament_path / PARAMS_FILE)

    configs = [
        config_generator(
            n_competitors=n_competitors_per_world,
            n_agents_per_competitor=1,
            agent_names_reveal_type=False,
            non_competitors=non_competitors,
            non_competitor_params=non_competitor_params,
            compact=compact,
            **kwargs,
        )
        for _ in range(n_configs)
    ]
    for i, cs in enumerate(configs):
        for c in cs:
            name_ = c["world_params"].get("name", "")
            c["config_id"] = f"{i:04d}" + (
                name_ if name_ else unique_name("", add_time=False, sep="", rand_digits=2)
            )
            c["world_params"]["name"] = c["config_id"]
    to_file(configs, tournament_path / "base_configs")
    if verbose:
        print(
            f"Will run {len(configs)}  different base world configurations ({parallelism})",
            flush=True,
        )

    competitor_info = list(zip(competitors, competitor_params))
    competitor_sets = _build_penguin_only_competitor_sets(
        competitor_info, n_competitors_per_world, round_robin
    )
    if competitor_sets is None:
        print("[WARN] 未找到 PenguinAgent，退回原有组合逻辑")
        if round_robin:
            competitor_sets = itertools.combinations(
                competitor_info, n_competitors_per_world
            )
        else:
            comp_ind = list(range(len(competitor_info)))
            random.shuffle(comp_ind)
            competitor_sets = _divide_into_sets(comp_ind, n_competitors_per_world)
            competitor_sets = [
                [competitor_info[_] for _ in lst] for lst in competitor_sets
            ]

    assigned = []
    for effective_competitor_infos in competitor_sets:
        effective_competitors = [_[0] for _ in effective_competitor_infos]
        effective_params = [_[1] for _ in effective_competitor_infos]
        effective_names = [
            a + _hash(b)[:4] if b else a for a, b in effective_competitor_infos
        ]
        effective_names = shortest_unique_names(effective_names, max_compression=True)
        if verbose:
            print(
                f"Running {'|'.join(effective_competitors)} together ({'|'.join(effective_names)})"
            )
        myconfigs = copy.deepcopy(configs)
        for conf in myconfigs:
            for c in conf:
                c["world_params"]["name"] += (
                    "_"
                    + "-".join(effective_names)
                    + unique_name("", add_time=False, rand_digits=3, sep=".")
                )
        this_assigned = list(
            itertools.chain(
                *(
                    config_assigner(
                        config=c,
                        max_n_worlds=max_worlds_per_config,
                        n_agents_per_competitor=1,
                        competitors=effective_competitors,
                        params=effective_params,
                        dynamic_non_competitors=dynamic_non_competitors,
                        dynamic_non_competitor_params=dynamic_non_competitor_params,
                        exclude_competitors_from_reassignment=exclude_competitors_from_reassignment,
                    )
                    for c in myconfigs
                )
            )
        )
        for i, config_set in enumerate(this_assigned):
            for c in config_set:
                c["world_params"]["name"] += f".{i:02d}"
        assigned += this_assigned

    for config_set in assigned:
        run_id = _run_id(config_set)
        for c in config_set:
            c["world_params"].update(
                {
                    "log_folder": str(
                        (tournament_path / run_id / c["world_params"]["name"]).absolute()
                    ),
                    "log_to_file": not compact,
                }
            )

    score_calculator_name = (
        get_full_type_name(score_calculator)
        if not isinstance(score_calculator, str)
        else score_calculator
    )
    world_generator_name = (
        get_full_type_name(world_generator)
        if not isinstance(world_generator, str)
        else world_generator
    )
    params["n_worlds"] = len(assigned) * n_runs_per_world
    params["world_generator_name"] = world_generator_name
    params["score_calculator_name"] = score_calculator_name
    dump(params, tournament_path / PARAMS_FILE)

    if verbose:
        print(
            f"Will run {len(assigned)}  different agent assignments ({parallelism})",
            flush=True,
        )

    if n_runs_per_world > 1:
        n_before_duplication = len(assigned)
        all_assigned = []
        for r in range(n_runs_per_world):
            for a_ in assigned:
                all_assigned.append([])
                for w_ in a_:
                    cpy = copy.deepcopy(w_)
                    cpy["world_params"]["name"] += f"_{r+1}"
                    if cpy["world_params"]["log_folder"]:
                        cpy["world_params"]["log_folder"] += f"_{r+1}"
                    all_assigned[-1].append(cpy)
        assigned = all_assigned
        assert n_before_duplication * n_runs_per_world == len(assigned), (
            f"Got {len(assigned)} assigned worlds for {n_before_duplication} "
            f"initial set with {n_runs_per_world} runs/world"
        )

    for config_set in assigned:
        run_id = _run_id(config_set)
        for config in config_set:
            dir_name = tournament_path / run_id / config["world_params"]["name"]
            config.update(
                {
                    "log_file_name": str(dir_name / "log.txt"),
                    "__dir_name": str(dir_name),
                }
            )
            config["world_params"].update(
                {"log_file_name": "log.txt", "log_folder": str(dir_name)}
            )

    if forced_logs_fraction > 1e-5:
        n_logged = max(1, int(len(assigned) * forced_logs_fraction))
        for cs in assigned[:n_logged]:
            run_id = _run_id(cs)
            for _ in cs:
                for subkey in ("world_params",):
                    if subkey not in _.keys():
                        continue
                    _[subkey].update(
                        dict(
                            compact=False,
                            save_negotiations=True,
                            log_to_file=True,
                            no_logs=False,
                        )
                    )
                    if _[subkey].get("log_folder", None) is None:
                        _[subkey].update(
                            dict(
                                log_folder=str(
                                    (tournament_path / run_id / _[subkey]["name"]).absolute()
                                )
                            )
                        )

                _.update(
                    dict(
                        compact=False,
                        save_negotiations=True,
                        log_to_file=True,
                        no_logs=False,
                    )
                )
                if _.get("log_folder", None) is None:
                    _.update(
                        dict(
                            log_folder=str(
                                (tournament_path / run_id / _["world_params"]["name"]).absolute()
                            )
                        )
                    )

    saved_configs = []
    for cs in assigned:
        for _ in cs:
            saved_configs.append(
                {
                    k: copy.copy(v)
                    if k != "competitors"
                    else [
                        get_full_type_name(c) if not isinstance(c, str) else c
                        for c in v
                    ]
                    for k, v in _.items()
                }
            )

    config_path = tournament_path / "configs"
    config_path.mkdir(exist_ok=True, parents=True)
    for i, conf in enumerate(saved_configs):
        f_name = config_path / f"{i:06}"
        to_file(conf, f_name)

    to_file(assigned, tournament_path / "assigned_configs")
    dump(assigned, tournament_path / ASSIGNED_CONFIGS_PICKLE_FILE)

    return tournament_path


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


def _cleanup_resumable_data(save_path: Path, tournament_root: Path) -> None:
    stage_pattern = f"{save_path.name}-stage-*"
    candidates: List[Path] = []
    for p in save_path.parent.glob(stage_pattern):
        candidates.append(p)
    if tournament_root.exists() and tournament_root != save_path and tournament_root not in candidates:
        candidates.append(tournament_root)

    for p in candidates:
        if p.exists():
            print(f"[INFO] 清理 resumable 数据目录: {p}")
            shutil.rmtree(p, ignore_errors=True)

    for fname in (
        ASSIGNED_CONFIGS_PICKLE_FILE,
        ASSIGNED_CONFIGS_JSON_FILE,
        "assigned_configs",
    ):
        fpath = save_path / fname
        if not fpath.exists():
            continue
        if fpath.is_dir():
            print(f"[INFO] 清理 resumable 文件夹: {fpath}")
            shutil.rmtree(fpath, ignore_errors=True)
        else:
            print(f"[INFO] 清理 resumable 文件: {fpath}")
            try:
                fpath.unlink()
            except Exception:
                pass


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
    parser.add_argument(
        "--n-competitors-per-world",
        type=int,
        choices=[2, 3, 4],
        default=None,
        help="每个 world 参赛者数量（2/3/4；不指定则按默认随机 2~4）",
    )
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
        "--forced-logs-fraction",
        type=float,
        default=0.1,
        help="强制保留详细日志的 world 比例 (default: 0.1)",
    )
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
    penguin_stage_root = None
    if args.track_only_penguin and created_now:
        if args.resumable:
            penguin_stage_root = save_path.parent / f"{save_path.name}-stage-0001"
        else:
            penguin_stage_root = save_path / f"LitaHRLData_{timestamp}-stage-0001"
    
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
    print("SCML 2025 Standard 数据采集 Runner")
    print("=" * 60)
    print(f"参赛代理: {len(competitors)} 个")
    print(f"   LitaAgent: {lita_names}")
    print(f"   外部 Agent: {external_names}")
    print(f"配置: n_configs={args.configs}, n_runs={args.runs}")
    if args.max_worlds_per_config is not None and n_per_world is not None:
        n_sets = _estimate_competitor_sets(len(competitors), n_per_world, args.round_robin)
        approx_worlds = args.configs * args.runs * args.max_worlds_per_config * n_sets
        print(f"约束: max_worlds_per_config={args.max_worlds_per_config} (≈ {approx_worlds} worlds)")
    print(
        f"选项: tracker=True, visualizer=False, auto_collect={not args.no_auto_collect}, "
        f"round_robin={args.round_robin}, no_csv={args.no_csv}, "
        f"forced_logs_fraction={args.forced_logs_fraction}"
    )
    print(f"并行: {parallelism_label}")
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

    if args.track_only_penguin:
        effective_n_per_world = n_per_world
        if effective_n_per_world is None:
            effective_n_per_world = random.randint(2, min(4, len(competitors)))
            print(
                f"[INFO] track_only_penguin=True，自动选择 n_competitors_per_world={effective_n_per_world}"
            )

        penguin_kwargs = dict(tournament_kwargs)
        penguin_kwargs.update(
            {
                "std_world": True,
                "publish_exogenous_summary": True,
                "publish_trading_prices": True,
            }
        )
        penguin_kwargs.pop("n_competitors_per_world", None)

        non_competitors = DefaultAgentsOneShot
        non_competitor_params = [dict() for _ in non_competitors]

        if args.resumable:
            if created_now:
                if save_path.exists():
                    save_path.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] 生成可续跑配置: {save_path}")
                if penguin_stage_root is None:
                    penguin_stage_root = save_path.parent / f"{save_path.name}-stage-0001"
                tournament_root = _create_penguin_only_tournament(
                    penguin_stage_root,
                    competitors=competitors,
                    competitor_params=None,
                    n_competitors_per_world=effective_n_per_world,
                    round_robin=args.round_robin,
                    n_configs=args.configs,
                    max_worlds_per_config=args.max_worlds_per_config,
                    n_runs_per_world=args.runs,
                    config_generator=anac2024_config_generator_std,
                    config_assigner=anac_assigner_std,
                    world_generator=anac2024_std_world_generator,
                    score_calculator=balance_calculator_std,
                    total_timeout=None,
                    parallelism=parallelism,
                    scheduler_ip=None,
                    scheduler_port=None,
                    non_competitors=non_competitors,
                    non_competitor_params=non_competitor_params,
                    dynamic_non_competitors=None,
                    dynamic_non_competitor_params=None,
                    exclude_competitors_from_reassignment=False,
                    verbose=not args.quiet,
                    compact=False,
                    forced_logs_fraction=args.forced_logs_fraction,
                    **penguin_kwargs,
                )
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
            if penguin_stage_root is None:
                penguin_stage_root = save_path / f"LitaHRLData_{timestamp}-stage-0001"
            tournament_root = _create_penguin_only_tournament(
                penguin_stage_root,
                competitors=competitors,
                competitor_params=None,
                n_competitors_per_world=effective_n_per_world,
                round_robin=args.round_robin,
                n_configs=args.configs,
                max_worlds_per_config=args.max_worlds_per_config,
                n_runs_per_world=args.runs,
                config_generator=anac2024_config_generator_std,
                config_assigner=anac_assigner_std,
                world_generator=anac2024_std_world_generator,
                score_calculator=balance_calculator_std,
                total_timeout=None,
                parallelism=parallelism,
                scheduler_ip=None,
                scheduler_port=None,
                non_competitors=non_competitors,
                non_competitor_params=non_competitor_params,
                dynamic_non_competitors=None,
                dynamic_non_competitor_params=None,
                exclude_competitors_from_reassignment=False,
                verbose=not args.quiet,
                compact=False,
                forced_logs_fraction=args.forced_logs_fraction,
                **penguin_kwargs,
            )
            print(f"[INFO] 启动比赛: {tournament_root}")
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
                    forced_logs_fraction=args.forced_logs_fraction,
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
                forced_logs_fraction=args.forced_logs_fraction,
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

    if args.resumable:
        try:
            _cleanup_resumable_data(save_path, Path(tournament_root))
        except Exception as exc:
            print(f"[WARN] 清理 resumable 数据失败: {exc}")
    
    # 如果重定向了输出，恢复并关闭
    if not args.foreground:
        log_handle.flush()
        log_handle.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"[INFO] 锦标赛完成！结果保存在: {save_path}")


if __name__ == "__main__":
    main()
