"""
LitaAgent 日志系统

将所有 print 输出重定向到 Tracker，避免控制台输出。
提供简单的日志接口供 LitaAgent 各模块使用。
"""

import os
import functools
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager


# ============================================================
# 全局日志记录器
# ============================================================

class LitaLogger:
    """
    LitaAgent 专用日志记录器
    
    用法:
        from litaagent_std.lita_logger import logger
        
        # 代替 print
        logger.info("message", agent_id="xxx")
        logger.debug("message", data={...})
        logger.decision("accept_offer", reason="price ok", context={...})
    """
    
    _instance: Optional['LitaLogger'] = None
    _enabled: bool = True
    _console_fallback: bool = False  # 如果 Tracker 不可用，是否回退到控制台
    _file_log: bool = True
    _log_file: Optional[str] = None
    _entries: list = []
    
    @classmethod
    def get(cls) -> 'LitaLogger':
        if cls._instance is None:
            cls._instance = LitaLogger()
        return cls._instance
    
    @classmethod
    def configure(cls, 
                  enabled: bool = True, 
                  console_fallback: bool = False,
                  log_file: Optional[str] = None):
        """配置日志器"""
        logger = cls.get()
        logger._enabled = enabled
        logger._console_fallback = console_fallback
        logger._log_file = log_file
        
        if log_file:
            logger._file_log = True
    
    def _write(self, level: str, category: str, message: str, 
               agent_id: str = "", day: int = -1, **data):
        """写入日志"""
        if not self._enabled:
            return
        
        entry = {
            "level": level,
            "category": category,
            "message": message,
            "agent_id": agent_id,
            "day": day,
            "data": data,
        }
        
        self._entries.append(entry)
        
        # 尝试写入 Tracker
        try:
            from scml_analyzer.auto_tracker import TrackerManager, TrackerConfig
            config = TrackerConfig.get()
            
            if config.enabled and agent_id:
                logger = TrackerManager.get_logger(agent_id, "LitaAgent")
                logger.set_day(day)
                logger.custom(category, message=message, **data)
                return  # 成功写入 Tracker，不需要回退
        except Exception:
            pass
        
        # 写入文件
        if self._file_log and self._log_file:
            try:
                with open(self._log_file, 'a', encoding='utf-8') as f:
                    import json
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                return
            except Exception:
                pass
        
        # 回退到控制台
        if self._console_fallback:
            prefix = f"[{level}]"
            if agent_id:
                prefix += f" [{agent_id}]"
            if day >= 0:
                prefix += f" Day {day}"
            print(f"{prefix} {category}: {message}")
    
    # ========== 日志级别方法 ==========
    
    def debug(self, message: str, agent_id: str = "", day: int = -1, **data):
        """调试日志"""
        self._write("DEBUG", "debug", message, agent_id, day, **data)
    
    def info(self, message: str, agent_id: str = "", day: int = -1, **data):
        """信息日志"""
        self._write("INFO", "info", message, agent_id, day, **data)
    
    def warn(self, message: str, agent_id: str = "", day: int = -1, **data):
        """警告日志"""
        self._write("WARN", "warning", message, agent_id, day, **data)
    
    def error(self, message: str, agent_id: str = "", day: int = -1, **data):
        """错误日志"""
        self._write("ERROR", "error", message, agent_id, day, **data)
    
    # ========== 业务相关日志 ==========
    
    def negotiation(self, event: str, agent_id: str = "", day: int = -1, **data):
        """协商日志"""
        self._write("INFO", f"negotiation_{event}", "", agent_id, day, **data)
    
    def decision(self, decision_name: str, result: str = "", reason: str = "",
                 agent_id: str = "", day: int = -1, **context):
        """决策日志"""
        self._write("INFO", f"decision_{decision_name}", reason, agent_id, day, 
                    result=result, **context)
    
    def contract(self, event: str, agent_id: str = "", day: int = -1, **data):
        """合同日志"""
        self._write("INFO", f"contract_{event}", "", agent_id, day, **data)
    
    def inventory(self, agent_id: str = "", day: int = -1, **state):
        """库存日志"""
        self._write("INFO", "inventory_state", "", agent_id, day, **state)
    
    def production(self, event: str, agent_id: str = "", day: int = -1, **data):
        """生产日志"""
        self._write("INFO", f"production_{event}", "", agent_id, day, **data)
    
    def concession(self, partner: str, direction: str, 
                   old_price: float, new_price: float, reason: str = "",
                   agent_id: str = "", day: int = -1, **context):
        """让步日志"""
        self._write("INFO", "concession", reason, agent_id, day,
                    partner=partner, direction=direction,
                    old_price=old_price, new_price=new_price, **context)
    
    def offer(self, action: str, partner: str, 
              quantity: int, price: float, delivery: int,
              reason: str = "", agent_id: str = "", day: int = -1, **context):
        """报价日志"""
        self._write("INFO", f"offer_{action}", reason, agent_id, day,
                    partner=partner, quantity=quantity, 
                    price=price, delivery=delivery, **context)
    
    # ========== 数据导出 ==========
    
    def get_entries(self) -> list:
        """获取所有日志条目"""
        return self._entries.copy()
    
    def clear(self):
        """清空日志"""
        self._entries.clear()
    
    def save(self, path: str):
        """保存日志到文件"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._entries, f, indent=2, ensure_ascii=False)


# ============================================================
# 便捷接口
# ============================================================

# 全局 logger 实例
logger = LitaLogger.get()


def log_info(message: str, agent_id: str = "", day: int = -1, **data):
    """便捷信息日志"""
    logger.info(message, agent_id, day, **data)


def log_debug(message: str, agent_id: str = "", day: int = -1, **data):
    """便捷调试日志"""
    logger.debug(message, agent_id, day, **data)


def log_decision(decision: str, result: str, reason: str = "", 
                 agent_id: str = "", day: int = -1, **context):
    """便捷决策日志"""
    logger.decision(decision, result, reason, agent_id, day, **context)


def log_offer(action: str, partner: str, qty: int, price: float, delivery: int,
              reason: str = "", agent_id: str = "", day: int = -1, **context):
    """便捷报价日志"""
    logger.offer(action, partner, qty, price, delivery, reason, agent_id, day, **context)


# ============================================================
# 静音上下文管理器
# ============================================================

@contextmanager
def silent_mode():
    """
    静音模式上下文管理器
    临时禁用所有日志输出
    """
    old_enabled = logger._enabled
    logger._enabled = False
    try:
        yield
    finally:
        logger._enabled = old_enabled


@contextmanager  
def console_mode():
    """
    控制台模式上下文管理器
    临时启用控制台回退输出
    """
    old_fallback = logger._console_fallback
    logger._console_fallback = True
    try:
        yield
    finally:
        logger._console_fallback = old_fallback


# ============================================================
# Print 替换装饰器
# ============================================================

def no_print(func: Callable) -> Callable:
    """
    装饰器：禁止函数内的 print 输出
    将 print 重定向到 logger
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import builtins
        original_print = builtins.print
        
        def silent_print(*pargs, **pkwargs):
            # 将 print 内容转换为日志
            message = " ".join(str(arg) for arg in pargs)
            logger.debug(message)
        
        builtins.print = silent_print
        try:
            return func(*args, **kwargs)
        finally:
            builtins.print = original_print
    
    return wrapper
