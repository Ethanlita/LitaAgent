"""
AutoTracker - 自动数据记录器

Agent 只需继承 TrackedAgent mixin，所有关键操作会自动记录。
无需手动调用，全自动采集数据。

Usage:
    from scml_analyzer.auto_tracker import TrackedAgent
    
    class MyAgent(TrackedAgent, StdSyncAgent):
        pass  # 自动记录所有数据
    
    # 如果需要自定义记录
    class MyAgent(TrackedAgent, StdSyncAgent):
        def respond_to_negotiation_request(self, ...):
            result = super().respond_to_negotiation_request(...)
            self.log("custom_event", reason="my custom reason")
            return result
"""

import os
import json
import threading
import functools
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from contextlib import contextmanager
import traceback


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# Global Configuration
# ============================================================

@dataclass
class TrackerConfig:
    """全局配置"""
    enabled: bool = True
    log_dir: Optional[str] = None
    console_echo: bool = False  # 是否同时输出到控制台（调试用）
    log_level: str = "INFO"  # DEBUG, INFO, WARN, ERROR
    
    # 自动记录选项
    auto_log_negotiations: bool = True
    auto_log_contracts: bool = True
    auto_log_inventory: bool = True
    auto_log_production: bool = True
    auto_log_decisions: bool = True


# 全局单例
_tracker_config_instance: Optional[TrackerConfig] = None


def get_tracker_config() -> TrackerConfig:
    """获取全局 TrackerConfig 实例"""
    global _tracker_config_instance
    if _tracker_config_instance is None:
        _tracker_config_instance = TrackerConfig()
    return _tracker_config_instance


def configure_tracker_config(**kwargs) -> TrackerConfig:
    """配置 Tracker"""
    config = get_tracker_config()
    for key, value in kwargs.items():
        if hasattr(config, key) and not key.startswith('_'):
            setattr(config, key, value)
    
    # 创建日志目录
    if config.log_dir:
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    return config


# 给 TrackerConfig 类添加便捷方法
TrackerConfig.get = staticmethod(get_tracker_config)
TrackerConfig.configure = staticmethod(configure_tracker_config)


# ============================================================
# Log Entry
# ============================================================

@dataclass
class LogEntry:
    """单条日志记录"""
    timestamp: str
    agent_id: str
    agent_type: str
    world_id: str
    day: int
    category: str  # negotiation, contract, inventory, production, decision, error, custom
    event: str     # 具体事件名
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, cls=NumpyJSONEncoder)


# ============================================================
# Agent Logger (每个 Agent 一个实例)
# ============================================================

class AgentLogger:
    """
    单个 Agent 的日志记录器
    自动收集该 Agent 的所有活动数据
    """
    
    def __init__(self, agent_id: str, agent_type: str, world_id: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.world_id = world_id
        self._entries: List[LogEntry] = []
        self._current_day: int = 0
        self._lock = threading.Lock()
        
        # 统计数据
        self._stats = {
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
        
        # 时间序列数据
        self._time_series: Dict[str, List[tuple]] = defaultdict(list)

    def __getstate__(self):
        """Pickle support: exclude lock"""
        state = self.__dict__.copy()
        if '_lock' in state:
            del state['_lock']
        return state

    def __setstate__(self, state):
        """Pickle support: restore lock"""
        self.__dict__.update(state)
        self._lock = threading.Lock()
    
    def set_day(self, day: int):
        """设置当前天数"""
        self._current_day = day
    
    def _create_entry(self, category: str, event: str, data: Dict[str, Any]) -> LogEntry:
        """创建日志条目"""
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            world_id=self.world_id,
            day=self._current_day,
            category=category,
            event=event,
            data=data,
        )
    
    def _log(self, category: str, event: str, data: Dict[str, Any]):
        """内部记录方法"""
        config = TrackerConfig.get()
        if not config.enabled:
            return
        
        entry = self._create_entry(category, event, data)
        
        with self._lock:
            self._entries.append(entry)
        
        # 控制台回显（调试用）
        if config.console_echo:
            print(f"[{self.agent_id}] {category}:{event} day={self._current_day} {data}")
    
    # ========== 协商相关 ==========
    
    def negotiation_started(
        self,
        partner: str,
        issues: Dict,
        is_seller: bool,
        mechanism_id: Optional[str] = None,
        negotiator_id: Optional[str] = None,
    ):
        """协商开始"""
        self._stats["negotiations_started"] += 1
        min_price = None
        max_price = None
        if isinstance(issues, dict):
            if "min_price" in issues:
                min_price = issues.get("min_price")
            if "max_price" in issues:
                max_price = issues.get("max_price")
            if min_price is None and isinstance(issues.get("issues"), dict):
                nested = issues.get("issues", {})
                if "min_price" in nested:
                    min_price = nested.get("min_price")
                if "max_price" in nested:
                    max_price = nested.get("max_price")
        data = {
            "partner": partner,
            "issues": str(issues),
            "role": "seller" if is_seller else "buyer",
        }
        if min_price is not None:
            try:
                data["min_price"] = float(min_price)
            except Exception:
                pass
        if max_price is not None:
            try:
                data["max_price"] = float(max_price)
            except Exception:
                pass
        if mechanism_id:
            data["mechanism_id"] = mechanism_id
        if negotiator_id:
            data["negotiator_id"] = negotiator_id
        self._log("negotiation", "started", data)
    
    def negotiation_offer_received(
        self,
        partner: str,
        offer: Dict,
        mechanism_id: Optional[str] = None,
        negotiator_id: Optional[str] = None,
    ):
        """收到报价"""
        data = {"partner": partner, "offer": offer}
        if mechanism_id:
            data["mechanism_id"] = mechanism_id
        if negotiator_id:
            data["negotiator_id"] = negotiator_id
        self._log("negotiation", "offer_received", data)
    
    def negotiation_offer_made(
        self,
        partner: str,
        offer: Dict,
        reason: str = "",
        mechanism_id: Optional[str] = None,
        negotiator_id: Optional[str] = None,
    ):
        """发出报价"""
        self._stats["offers_made"] += 1
        data = {"partner": partner, "offer": offer, "reason": reason}
        if mechanism_id:
            data["mechanism_id"] = mechanism_id
        if negotiator_id:
            data["negotiator_id"] = negotiator_id
        self._log("negotiation", "offer_made", data)
    
    def negotiation_accept(
        self,
        partner: str,
        offer: Dict,
        reason: str = "",
        mechanism_id: Optional[str] = None,
        negotiator_id: Optional[str] = None,
    ):
        """接受报价"""
        self._stats["offers_accepted"] += 1
        data = {"partner": partner, "offer": offer, "reason": reason}
        if mechanism_id:
            data["mechanism_id"] = mechanism_id
        if negotiator_id:
            data["negotiator_id"] = negotiator_id
        self._log("negotiation", "accept", data)
    
    def negotiation_reject(
        self,
        partner: str,
        offer: Dict,
        reason: str = "",
        mechanism_id: Optional[str] = None,
        negotiator_id: Optional[str] = None,
    ):
        """拒绝报价"""
        self._stats["offers_rejected"] += 1
        data = {"partner": partner, "offer": offer, "reason": reason}
        if mechanism_id:
            data["mechanism_id"] = mechanism_id
        if negotiator_id:
            data["negotiator_id"] = negotiator_id
        self._log("negotiation", "reject", data)
    
    def negotiation_success(
        self,
        partner: str,
        agreement: Dict,
        mechanism_id: Optional[str] = None,
        negotiator_id: Optional[str] = None,
    ):
        """协商成功"""
        self._stats["negotiations_success"] += 1
        data = {"partner": partner, "agreement": agreement}
        if mechanism_id:
            data["mechanism_id"] = mechanism_id
        if negotiator_id:
            data["negotiator_id"] = negotiator_id
        self._log("negotiation", "success", data)
    
    def negotiation_failure(
        self,
        partner: str,
        reason: str = "",
        mechanism_id: Optional[str] = None,
        negotiator_id: Optional[str] = None,
    ):
        """协商失败"""
        self._stats["negotiations_failed"] += 1
        data = {"partner": partner, "reason": reason}
        if mechanism_id:
            data["mechanism_id"] = mechanism_id
        if negotiator_id:
            data["negotiator_id"] = negotiator_id
        self._log("negotiation", "failure", data)
    
    # ========== 合同相关 ==========
    
    def contract_signed(self, contract_id: str, partner: str, 
                        quantity: int, price: float, delivery_day: int,
                        is_seller: bool):
        """合同签署"""
        self._stats["contracts_signed"] += 1
        self._log("contract", "signed", {
            "contract_id": contract_id,
            "partner": partner,
            "quantity": quantity,
            "price": price,
            "delivery_day": delivery_day,
            "role": "seller" if is_seller else "buyer",
        })
    
    def contract_executed(self, contract_id: str, quantity: int):
        """合同执行"""
        self._log("contract", "executed", {
            "contract_id": contract_id,
            "quantity": quantity,
        })
    
    def contract_breached(self, contract_id: str, reason: str = ""):
        """合同违约"""
        self._stats["contracts_breached"] += 1
        self._log("contract", "breached", {
            "contract_id": contract_id,
            "reason": reason,
        })
    
    # ========== 库存相关 ==========
    
    def inventory_state(self, raw_material: int, product: int, 
                        balance: float, **extra):
        """库存状态"""
        data = {
            "raw_material": raw_material,
            "product": product,
            "balance": balance,
            **extra,
        }
        self._log("inventory", "state", data)
        
        # 记录时间序列
        self._time_series["raw_material"].append((self._current_day, raw_material))
        self._time_series["product"].append((self._current_day, product))
        self._time_series["balance"].append((self._current_day, balance))
    
    # ========== 生产相关 ==========
    
    def production_scheduled(self, quantity: int, day: int):
        """生产计划"""
        self._stats["production_scheduled"] += quantity
        self._log("production", "scheduled", {
            "quantity": quantity,
            "day": day,
        })
    
    def production_executed(self, quantity: int):
        """生产执行"""
        self._stats["production_executed"] += quantity
        self._log("production", "executed", {
            "quantity": quantity,
        })
    
    def production_failed(self, quantity: int, reason: str):
        """生产失败"""
        self._log("production", "failed", {
            "quantity": quantity,
            "reason": reason,
        })
    
    # ========== 决策相关 ==========
    
    def decision(self, name: str, result: Any, reason: str = "", **context):
        """记录决策"""
        self._log("decision", name, {
            "result": str(result),
            "reason": reason,
            **context,
        })
    
    # ========== 通用方法 ==========
    
    def custom(self, event: str, **data):
        """自定义日志"""
        self._log("custom", event, data)
    
    def error(self, event: str, message: str, **data):
        """错误日志"""
        self._log("error", event, {"message": message, **data})
    
    def debug(self, event: str, **data):
        """调试日志"""
        config = TrackerConfig.get()
        if config.log_level == "DEBUG":
            self._log("debug", event, data)
    
    # ========== 数据导出 ==========
    
    def get_entries(self) -> List[Dict]:
        """获取所有日志条目"""
        return [e.to_dict() for e in self._entries]
    
    def get_stats(self) -> Dict:
        """获取统计数据"""
        return self._stats.copy()
    
    def get_time_series(self) -> Dict[str, List[tuple]]:
        """获取时间序列数据"""
        return dict(self._time_series)
    
    def export(self) -> Dict:
        """导出所有数据"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "world_id": self.world_id,
            "stats": self.get_stats(),
            "time_series": self.get_time_series(),
            "entries": self.get_entries(),
        }
    
    def save(self, path: Optional[str] = None):
        """保存到文件"""
        config = TrackerConfig.get()
        
        if path is None:
            if config.log_dir is None:
                return
            safe_id = self.agent_id.replace("@", "_at_").replace("/", "_")
            path = os.path.join(config.log_dir, f"agent_{safe_id}.json")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.export(), f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)


# ============================================================
# Global Tracker Manager
# ============================================================

class TrackerManager:
    """
    全局 Tracker 管理器
    管理所有 Agent 的 Logger 实例
    """
    
    _loggers: Dict[str, AgentLogger] = {}
    _world_id: str = "unknown"
    _lock = threading.Lock()
    _pid = os.getpid()
    
    @classmethod
    def _get_lock(cls):
        """获取进程安全的锁"""
        if os.getpid() != cls._pid:
            cls._lock = threading.Lock()
            cls._pid = os.getpid()
            # 在新进程中（fork模式），清空继承的 loggers
            cls._loggers = {} 
        return cls._lock
    
    @classmethod
    def configure(cls, log_dir: Optional[str] = None, **kwargs):
        """配置 Tracker"""
        TrackerConfig.configure(log_dir=log_dir, **kwargs)
        with cls._get_lock():
            cls._loggers.clear()
    
    @classmethod
    def set_world(cls, world_id: str):
        """设置当前 World ID"""
        cls._world_id = world_id
    
    @classmethod
    def get_logger(cls, agent_id: str, agent_type: str) -> AgentLogger:
        """获取或创建 Agent 的 Logger"""
        with cls._get_lock():
            if agent_id not in cls._loggers:
                cls._loggers[agent_id] = AgentLogger(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    world_id=cls._world_id,
                )
            return cls._loggers[agent_id]
    
    @classmethod
    def get_all_loggers(cls) -> Dict[str, AgentLogger]:
        """获取所有 Logger"""
        return cls._loggers.copy()
    
    @classmethod
    def save_all(cls, output_dir: Optional[str] = None):
        """保存所有 Agent 数据"""
        config = TrackerConfig.get()
        output_dir = output_dir or config.log_dir
        
        if not output_dir:
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存每个 Agent 的数据
        for agent_id, logger in cls._loggers.items():
            safe_id = agent_id.replace("@", "_at_").replace("/", "_")
            agent_path = os.path.join(output_dir, f"agent_{safe_id}.json")
            logger.save(agent_path)
        
        # 保存汇总数据
        summary = {
            "world_id": cls._world_id,
            "agents": {
                agent_id: {
                    "type": logger.agent_type,
                    "stats": logger.get_stats(),
                }
                for agent_id, logger in cls._loggers.items()
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        summary_path = os.path.join(output_dir, "tracker_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
    
    @classmethod
    def export_all(cls) -> Dict:
        """导出所有数据"""
        return {
            "world_id": cls._world_id,
            "agents": {
                agent_id: logger.export()
                for agent_id, logger in cls._loggers.items()
            },
        }
    
    @classmethod
    def rebuild_summary(cls, log_dir: str) -> Optional[str]:
        """
        从日志目录重建 summary 文件
        适用于并行执行后，主进程没有收集到数据的情况
        """
        path = Path(log_dir)
        if not path.exists():
            return None
            
        agent_files = list(path.glob("agent_*.json"))
        if not agent_files:
            return None
            
        summary = {
            "world_id": "rebuilt_from_files",
            "agents": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        for file_path in agent_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    agent_id = data.get("agent_id", "unknown")
                    summary["agents"][agent_id] = {
                        "type": data.get("agent_type", "unknown"),
                        "stats": data.get("stats", {}),
                    }
                    # 尝试获取 world_id
                    if summary["world_id"] == "rebuilt_from_files":
                        summary["world_id"] = data.get("world_id", "unknown")
            except Exception:
                pass
                
        summary_path = os.path.join(log_dir, "tracker_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            
        return summary_path

    @classmethod
    def cleanup(cls):
        """清理所有 Logger"""
        cls.save_all()
        cls._loggers.clear()


# ============================================================
# TrackedAgent Mixin
# ============================================================

class TrackedAgent:
    """
    Agent 跟踪 Mixin
    
    继承此类后，Agent 的关键操作会自动被记录。
    
    Usage:
        class MyAgent(TrackedAgent, StdSyncAgent):
            pass
    """
    
    _tracker_logger: Optional[AgentLogger] = None
    
    @property
    def logger(self) -> AgentLogger:
        """获取当前 Agent 的 Logger"""
        if self._tracker_logger is None:
            # 延迟初始化
            agent_id = getattr(self, 'id', 'unknown')
            agent_type = type(self).__name__
            self._tracker_logger = TrackerManager.get_logger(agent_id, agent_type)
        return self._tracker_logger
    
    def log(self, event: str, **data):
        """快捷日志方法"""
        self.logger.custom(event, **data)
    
    # ========== 自动记录的方法覆写 ==========
    
    def init(self):
        """初始化时自动记录"""
        super().init()
        self.logger.custom("agent_initialized", awi_info={
            "n_steps": getattr(self.awi, 'n_steps', None),
            "n_lines": getattr(self.awi, 'n_lines', None),
            "level": getattr(self.awi, 'level', None),
        })
    
    def before_step(self):
        """每步开始时自动记录"""
        super().before_step()
        self.logger.set_day(self.awi.current_step)
        
        # 自动记录库存状态
        config = TrackerConfig.get()
        if config.auto_log_inventory:
            try:
                state = self.awi.state
                self.logger.inventory_state(
                    raw_material=state.inventory[self.awi.my_input_product],
                    product=state.inventory[self.awi.my_output_product],
                    balance=state.balance,
                )
            except Exception:
                pass
    
    def step(self):
        """执行步骤"""
        result = super().step()
        
        # 在最后一步自动保存日志
        # 这对于并行执行至关重要，因为进程结束时内存中的日志会丢失
        if hasattr(self, 'awi') and hasattr(self.awi, 'n_steps'):
            if self.awi.current_step == self.awi.n_steps - 1:
                self.logger.save()
                
        return result
    
    def on_negotiation_success(self, contract, mechanism):
        """协商成功时自动记录"""
        super().on_negotiation_success(contract, mechanism)
        
        config = TrackerConfig.get()
        if config.auto_log_contracts:
            try:
                partner = [p for p in contract.partners if p != self.id][0]
                agreement = contract.agreement
                is_seller = self.awi.is_first_level is False  # 简化判断
                
                self.logger.contract_signed(
                    contract_id=contract.id,
                    partner=partner,
                    quantity=agreement.get("quantity", 0),
                    price=agreement.get("unit_price", 0),
                    delivery_day=agreement.get("time", 0),
                    is_seller=is_seller,
                )
            except Exception:
                pass
    
    def on_negotiation_failure(self, partners, annotation, mechanism, state):
        """协商失败时自动记录"""
        super().on_negotiation_failure(partners, annotation, mechanism, state)
        
        config = TrackerConfig.get()
        if config.auto_log_negotiations:
            try:
                partner = partners[0] if partners else "unknown"
                self.logger.negotiation_failure(partner)
            except Exception:
                pass


# ============================================================
# 便捷函数
# ============================================================

def configure_tracker(log_dir: Optional[str] = None, **kwargs):
    """配置全局 Tracker"""
    TrackerManager.configure(log_dir=log_dir, **kwargs)


def get_logger(agent_id: str, agent_type: str = "unknown") -> AgentLogger:
    """获取 Agent Logger"""
    return TrackerManager.get_logger(agent_id, agent_type)


def save_all_logs(output_dir: Optional[str] = None):
    """保存所有日志"""
    TrackerManager.save_all(output_dir)


def export_all_data() -> Dict:
    """导出所有数据"""
    return TrackerManager.export_all()
