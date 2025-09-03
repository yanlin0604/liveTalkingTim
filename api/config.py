"""
配置管理相关API接口
"""
import json
import os
from aiohttp import web
from dynamic_config import dynamic_config, get_config, set_config, set_nested_config
from logger import logger


class ConfigAPI:
    """配置管理API接口类"""
    
    async def get_config_api(self, request):
        """
        获取当前配置接口
        
        功能：返回当前所有配置参数
        
        ---
        tags:
          - Config
        summary: 获取当前配置
        description: 返回当前所有配置参数
        produces:
          - application/json
        responses:
          200:
            description: 获取成功
            schema:
              type: object
              description: 所有配置参数的JSON对象
        """
        try:
            config = dynamic_config.get_all()
            return web.Response(
                content_type="application/json",
                text=json.dumps(config, ensure_ascii=False),
            )
        except Exception as e:
            logger.exception('获取配置失败:')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"error": str(e)}, ensure_ascii=False),
                status=500
            )

    async def update_config_api(self, request):
        """
        更新配置接口
        
        功能：动态更新配置参数，支持单个和批量更新，并自动保存到文件
        方法：POST
        参数格式1（单个配置）：
            - key: 配置参数名（支持点号分隔的嵌套路径，如 "streaming_quality.target_fps"）
            - value: 新的配置值
        参数格式2（批量配置）：
            - configs: 配置字典，格式为 {"key1": "value1", "key2": "value2", ...}
        返回：
            - success: 是否成功
            - message: 状态消息
            - updated_count: 更新的配置项数量（批量模式）
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"success": False, "message": "请求体为空"}, ensure_ascii=False),
                    status=400
                )
            
            params = await request.json()
            
            def update_nested_config(key_path, value):
                """更新嵌套配置的辅助函数"""
                set_nested_config(key_path, value, save=False)
            
            # 检查是否为批量更新模式
            if 'configs' in params:
                # 批量更新模式
                configs = params.get('configs')
                if not isinstance(configs, dict):
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps({"success": False, "message": "configs参数必须是字典格式"}, ensure_ascii=False),
                        status=400
                    )
                
                if not configs:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps({"success": False, "message": "configs字典不能为空"}, ensure_ascii=False),
                        status=400
                    )
                
                # 批量更新配置
                updated_count = 0
                for key, value in configs.items():
                    if key:  # 确保key不为空
                        if '.' in key:
                            # 嵌套配置更新
                            update_nested_config(key, value)
                        else:
                            # 普通配置更新
                            set_config(key, value, save=False)
                        updated_count += 1
                
                # 统一保存到文件
                dynamic_config.save_config()
                
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        "success": True, 
                        "message": f"批量更新了 {updated_count} 个配置项并已保存",
                        "updated_count": updated_count
                    }, ensure_ascii=False),
                )
            
            else:
                # 单个更新模式（保持向后兼容）
                key = params.get('key')
                value = params.get('value')
                
                if not key:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps({"success": False, "message": "缺少参数key或configs"}, ensure_ascii=False),
                        status=400
                    )
                
                # 检查是否为嵌套配置更新
                if '.' in key:
                    # 嵌套配置更新
                    set_nested_config(key, value, save=True)
                else:
                    # 普通配置更新
                    set_config(key, value, save=True)
                
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"success": True, "message": f"配置 {key} 已更新并保存"}, ensure_ascii=False),
                )
                
        except json.JSONDecodeError as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": False, "message": f"JSON格式错误: {str(e)}"}, ensure_ascii=False),
                status=400
            )
        except Exception as e:
            logger.exception('更新配置失败:')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": False, "message": str(e)}, ensure_ascii=False),
                status=500
            )

    async def reset_config_api(self, request):
        """
        重置配置接口
        
        功能：重置为默认配置
        方法：POST
        返回：
            - success: 是否成功
            - message: 状态消息
        """
        try:
            # 重新加载配置文件
            dynamic_config.load_config()
            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": True, "message": "配置已重置"}, ensure_ascii=False),
            )
        except Exception as e:
            logger.exception('重置配置失败:')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": False, "message": str(e)}, ensure_ascii=False),
                status=500
            )

    async def get_config_for_frontend(self, request):
        """
        获取前端需要的配置信息接口

        功能：返回前端页面需要的配置信息
        方法：GET
        返回：
            - code: 状态码（0=成功）
            - data: 配置信息，包含：
                * use_custom_silent: 是否默认开启自定义动作

        使用场景：
            - 前端页面初始化时获取配置
            - 动态更新前端界面状态
        """
        try:
            # 从配置文件中读取设置
            config_path = "config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                use_custom_silent = config_data.get('use_custom_silent', True)
                custom_silent_audiotype = config_data.get('custom_silent_audiotype', "")
                multi_action_mode = config_data.get('multi_action_mode', 'single')
                multi_action_list = config_data.get('multi_action_list', [])
                multi_action_interval = config_data.get('multi_action_interval', 100)
                multi_action_switch_policy = config_data.get('multi_action_switch_policy', 'interval')
            else:
                use_custom_silent = True  # 默认值
                custom_silent_audiotype = ""  # 默认值
                multi_action_mode = 'single'  # 默认值
                multi_action_list = []  # 默认值
                multi_action_interval = 100  # 默认值
                multi_action_switch_policy = 'interval'  # 默认值
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": 0,
                    "data": {
                        "use_custom_silent": use_custom_silent,
                        "custom_silent_audiotype": custom_silent_audiotype,
                        "multi_action_mode": multi_action_mode,
                        "multi_action_list": multi_action_list,
                        "multi_action_interval": multi_action_interval,
                        "multi_action_switch_policy": multi_action_switch_policy
                    }
                }),
            )
            
        except Exception as e:
            logger.exception('获取前端配置失败:')
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": str(e)
                }),
            ) 