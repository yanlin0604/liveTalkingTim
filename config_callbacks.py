#!/usr/bin/env python3
"""
配置变化回调处理器
处理配置参数变化时的实时更新逻辑
"""

from logger import logger
from dynamic_config import register_config_callback

def setup_config_callbacks(opt, nerfreals):
    """设置配置变化回调"""
    
    def on_batch_size_change(key, old_value, new_value):
        """批次大小变化回调"""
        try:
            opt.batch_size = int(new_value)
            logger.info(f"批次大小已更新: {old_value} -> {new_value}")
            
            # 通知所有活跃的数字人实例
            for sessionid, nerfreal in nerfreals.items():
                if nerfreal and hasattr(nerfreal, 'update_batch_size'):
                    nerfreal.update_batch_size(int(new_value))
                    
        except Exception as e:
            logger.error(f"更新批次大小失败: {e}")
    
    def on_tts_change(key, old_value, new_value):
        """TTS配置变化回调"""
        try:
            setattr(opt, key, new_value)
            logger.info(f"TTS配置已更新: {key} = {old_value} -> {new_value}")
            
            # 通知所有活跃的数字人实例更新TTS配置
            for sessionid, nerfreal in nerfreals.items():
                if nerfreal and hasattr(nerfreal, 'update_tts_config'):
                    nerfreal.update_tts_config(key, new_value)
                    
        except Exception as e:
            logger.error(f"更新TTS配置失败: {e}")
    
    def on_llm_change(key, old_value, new_value):
        """LLM配置变化回调"""
        try:
            setattr(opt, key, new_value)
            logger.info(f"LLM配置已更新: {key} = {old_value} -> {new_value}")
            
            # 更新LLM配置
            if key == 'llm_system_prompt':
                # 重新初始化LLM系统提示词
                from llm import update_system_prompt
                if 'update_system_prompt' in globals():
                    update_system_prompt(new_value)
                    
        except Exception as e:
            logger.error(f"更新LLM配置失败: {e}")
    
    def on_color_matching_change(key, old_value, new_value):
        """颜色匹配配置变化回调"""
        try:
            if key == 'enable_color_matching':
                opt.enable_color_matching = bool(new_value)
            elif key == 'color_matching_strength':
                opt.color_matching_strength = float(new_value)
                
            logger.info(f"颜色匹配配置已更新: {key} = {old_value} -> {new_value}")
            
            # 通知所有活跃的数字人实例
            for sessionid, nerfreal in nerfreals.items():
                if nerfreal and hasattr(nerfreal, 'update_color_matching'):
                    nerfreal.update_color_matching(
                        opt.enable_color_matching, 
                        opt.color_matching_strength
                    )
                    
        except Exception as e:
            logger.error(f"更新颜色匹配配置失败: {e}")
    
    def on_custom_silent_change(key, old_value, new_value):
        """自定义静默动作相关配置变化回调"""
        try:
            if key == 'custom_silent_audiotype':
                opt.custom_silent_audiotype = str(new_value or "")
                logger.info(f"静默动作类型已更新: {old_value or '未指定'} -> {opt.custom_silent_audiotype or '未指定'}")
                # 推送到所有实例：运行时立即重新加载自定义动作
                for sessionid, nerfreal in nerfreals.items():
                    if nerfreal and hasattr(nerfreal, 'set_custom_silent_audiotype'):
                        nerfreal.set_custom_silent_audiotype(opt.custom_silent_audiotype)
            elif key == 'use_custom_silent':
                # 允许布尔/字符串输入
                val = bool(new_value)
                opt.use_custom_silent = val
                logger.info(f"静默自定义动作开关更新: {old_value} -> {val}")
                for sessionid, nerfreal in nerfreals.items():
                    if nerfreal and hasattr(nerfreal, 'set_use_custom_silent'):
                        nerfreal.set_use_custom_silent(val)
        except Exception as e:
            logger.error(f"更新自定义静默动作配置失败: {e}")

    def on_avatar_change(key, old_value, new_value):
        """数字人ID变化回调"""
        try:
            opt.avatar_id = str(new_value)
            logger.info(f"数字人ID已更新: {old_value} -> {new_value}")
            logger.warning("数字人ID变化需要重新创建会话才能生效")
            
        except Exception as e:
            logger.error(f"更新数字人ID失败: {e}")
    
    def on_restart_required_change(key, old_value, new_value):
        """需要重启的配置变化回调"""
        try:
            setattr(opt, key, new_value)
            logger.warning(f"配置 {key} 已更新: {old_value} -> {new_value}")
            logger.warning("此配置需要重启服务才能生效")
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
    
    # 注册回调
    register_config_callback('batch_size', on_batch_size_change)
    
    # TTS相关回调
    register_config_callback('tts', on_tts_change)
    register_config_callback('REF_FILE', on_tts_change)
    register_config_callback('REF_TEXT', on_tts_change)
    register_config_callback('TTS_SERVER', on_tts_change)
    
    # LLM相关回调
    register_config_callback('llm_provider', on_llm_change)
    register_config_callback('llm_model', on_llm_change)
    register_config_callback('llm_system_prompt', on_llm_change)
    register_config_callback('ollama_host', on_llm_change)
    
    # 颜色匹配回调
    register_config_callback('enable_color_matching', on_color_matching_change)
    register_config_callback('color_matching_strength', on_color_matching_change)
    
    # 自定义静默动作回调
    register_config_callback('custom_silent_audiotype', on_custom_silent_change)
    register_config_callback('use_custom_silent', on_custom_silent_change)

    # 数字人相关回调
    register_config_callback('avatar_id', on_avatar_change)
    
    # 需要重启的配置
    register_config_callback('model', on_restart_required_change)
    register_config_callback('transport', on_restart_required_change)
    register_config_callback('listenport', on_restart_required_change)
    register_config_callback('max_session', on_restart_required_change)
    register_config_callback('model_path', on_restart_required_change)
    
    logger.info("配置变化回调已设置完成")