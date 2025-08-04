#!/usr/bin/env python3
"""
动态配置管理器
支持运行时热更新配置，无需重启服务
"""

import json
import os
import time
import threading
from typing import Dict, Any, Callable
from pathlib import Path
from logger import logger

class DynamicConfig:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.callbacks: Dict[str, list] = {}  # 参数变化回调
        self.last_modified = 0
        self.lock = threading.Lock()
        self.monitoring = False
        
        # 加载初始配置
        self.load_config()
        
    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_file):
                logger.warning(f"配置文件不存在: {self.config_file}")
                return False
                
            with open(self.config_file, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
            
            # 过滤注释
            new_config = {k: v for k, v in new_config.items() 
                         if not k.startswith('//') and k != '_descriptions'}
            
            with self.lock:
                old_config = self.config.copy()
                self.config = new_config
                self.last_modified = os.path.getmtime(self.config_file)
                
                # 检查变化并触发回调
                self._trigger_callbacks(old_config, new_config)
                
            logger.info(f"配置已加载: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return False
    
    def get(self, key: str, default=None):
        """获取配置值"""
        with self.lock:
            return self.config.get(key, default)
    
    def set(self, key: str, value: Any, save: bool = True):
        """设置配置值"""
        with self.lock:
            old_value = self.config.get(key)
            self.config[key] = value
            
            if save:
                self.save_config()
            
            # 触发回调
            if key in self.callbacks and old_value != value:
                for callback in self.callbacks[key]:
                    try:
                        callback(key, old_value, value)
                    except Exception as e:
                        logger.error(f"配置回调执行失败 {key}: {e}")
    
    def set_nested(self, key_path: str, value: Any, save: bool = True):
        """设置嵌套配置值（支持点号分隔的键路径）"""
        with self.lock:
            keys = key_path.split('.')
            current = self.config
            
            # 遍历到最后一个键的父级
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # 获取旧值用于回调
            old_value = current.get(keys[-1])
            
            # 设置最终值
            current[keys[-1]] = value
            
            if save:
                self.save_config()
            
            # 触发回调（使用完整路径作为键）
            if key_path in self.callbacks and old_value != value:
                for callback in self.callbacks[key_path]:
                    try:
                        callback(key_path, old_value, value)
                    except Exception as e:
                        logger.error(f"配置回调执行失败 {key_path}: {e}")
    
    def save_config(self):
        """保存配置到文件"""
        try:
            # 添加时间戳注释
            config_with_comments = {
                "// 配置文件": "Dynamic Configuration File",
                "// 最后更新": time.strftime("%Y-%m-%d %H:%M:%S"),
                **self.config
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_with_comments, f, indent=2, ensure_ascii=False)
                
            logger.info(f"配置已保存: {self.config_file}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def register_callback(self, key: str, callback: Callable):
        """注册配置变化回调"""
        if key not in self.callbacks:
            self.callbacks[key] = []
        self.callbacks[key].append(callback)
        logger.info(f"已注册配置回调: {key}")
    
    def _trigger_callbacks(self, old_config: Dict, new_config: Dict):
        """触发配置变化回调"""
        for key in set(old_config.keys()) | set(new_config.keys()):
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value and key in self.callbacks:
                for callback in self.callbacks[key]:
                    try:
                        callback(key, old_value, new_value)
                    except Exception as e:
                        logger.error(f"配置回调执行失败 {key}: {e}")
    
    def start_monitoring(self, interval: float = 1.0):
        """开始监控配置文件变化"""
        if self.monitoring:
            return
            
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                try:
                    if os.path.exists(self.config_file):
                        current_modified = os.path.getmtime(self.config_file)
                        if current_modified > self.last_modified:
                            logger.info("检测到配置文件变化，重新加载...")
                            self.load_config()
                except Exception as e:
                    logger.error(f"监控配置文件失败: {e}")
                
                time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        logger.info(f"开始监控配置文件: {self.config_file}")
    
    def stop_monitoring(self):
        """停止监控配置文件"""
        self.monitoring = False
        logger.info("停止监控配置文件")
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self.lock:
            return self.config.copy()
    
    def update_batch(self, updates: Dict[str, Any], save: bool = True):
        """批量更新配置"""
        with self.lock:
            old_config = self.config.copy()
            self.config.update(updates)
            
            if save:
                self.save_config()
            
            # 触发回调
            self._trigger_callbacks(old_config, self.config)

# 全局配置实例
dynamic_config = DynamicConfig()

# 便捷函数
def get_config(key: str, default=None):
    """获取配置值"""
    return dynamic_config.get(key, default)

def set_config(key: str, value: Any, save: bool = True):
    """设置配置值"""
    dynamic_config.set(key, value, save)

def set_nested_config(key_path: str, value: Any, save: bool = True):
    """设置嵌套配置值（支持点号分隔的键路径）"""
    dynamic_config.set_nested(key_path, value, save)

def register_config_callback(key: str, callback: Callable):
    """注册配置变化回调"""
    dynamic_config.register_callback(key, callback)

def start_config_monitoring(interval: float = 1.0):
    """开始监控配置文件"""
    dynamic_config.start_monitoring(interval)

def get_all_config() -> Dict[str, Any]:
    """获取所有配置"""
    return dynamic_config.get_all()