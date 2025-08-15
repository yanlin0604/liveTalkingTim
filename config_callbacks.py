#!/usr/bin/env python3
"""
配置变化回调处理器
处理配置参数变化时的实时更新逻辑
"""

from logger import logger
from dynamic_config import register_config_callback, get_all_config

def setup_config_callbacks(opt, nerfreals):
    """设置配置变化回调"""
    
    def _coerce_like(old_val, new_val):
        """将 new_val 尽量转换成与 old_val 相同的类型，失败则原样返回"""
        try:
            if isinstance(old_val, bool):
                # 支持多种布尔表示
                if isinstance(new_val, str):
                    return new_val.strip().lower() in ("1", "true", "yes", "on")
                return bool(new_val)
            if isinstance(old_val, int) and not isinstance(old_val, bool):
                return int(new_val)
            if isinstance(old_val, float):
                return float(new_val)
        except Exception:
            pass
        return new_val

    def on_any_change(key, old_value, new_value):
        """通用配置变化回调：监控与即时生效分发"""
        try:
            logger.info(f"配置变更: {key} = {old_value} -> {new_value}")

            # 0) 尝试将 opt 中同名字段也更新（若存在则按原类型做转换）
            if hasattr(opt, key):
                try:
                    casted = _coerce_like(getattr(opt, key), new_value)
                    setattr(opt, key, casted)
                except Exception:
                    setattr(opt, key, new_value)

            # 1) 通用广播：实例若实现 on_config_change 或 update_config，则全部分发
            for _, nerfreal in nerfreals.items():
                if not nerfreal:
                    continue
                if hasattr(nerfreal, 'on_config_change'):
                    try:
                        nerfreal.on_config_change(key, old_value, new_value)
                    except Exception as be:
                        logger.warning(f"实例 on_config_change 处理失败: key={key}, err={be}")
                elif hasattr(nerfreal, 'update_config'):
                    try:
                        nerfreal.update_config(key, new_value)
                    except Exception as be:
                        logger.warning(f"实例 update_config 处理失败: key={key}, err={be}")

        except Exception as e:
            logger.error(f"通用配置变化处理失败: {key}, 错误: {e}")
    
    

    
    # 为所有配置参数（含嵌套）注册通用监控回调
    def _flatten_keys(d, prefix=""):
        for k, v in d.items():
            if k.startswith('//') or k == '_descriptions':
                continue
            path = f"{prefix}.{k}" if prefix else k
            yield path
            if isinstance(v, dict):
                # 继续向下展开
                yield from _flatten_keys(v, path)

    try:
        current_cfg = get_all_config() or {}
        for key_path in _flatten_keys(current_cfg):
            try:
                register_config_callback(key_path, on_any_change)
            except Exception as e:
                logger.warning(f"注册通用回调失败: {key_path}, 错误: {e}")
        logger.info("已为所有配置参数注册通用监控回调")
    except Exception as e:
        logger.error(f"展开并注册通用配置回调失败: {e}")
    
    logger.info("配置变化回调已设置完成")