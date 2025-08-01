"""
配置管理相关API接口
"""
import json
import os
from aiohttp import web
from dynamic_config import dynamic_config, get_config, set_config
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
        
        功能：动态更新单个配置参数
        方法：POST
        参数：
            - key: 配置参数名
            - value: 新的配置值
        返回：
            - success: 是否成功
            - message: 状态消息
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
            key = params.get('key')
            value = params.get('value')
            
            if not key:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({"success": False, "message": "缺少参数key"}, ensure_ascii=False),
                    status=400
                )
            
            # 更新配置
            set_config(key, value, save=True)
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": True, "message": f"配置 {key} 已更新"}, ensure_ascii=False),
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

    async def save_config_api(self, request):
        """
        保存配置到文件接口
        
        功能：将当前配置保存到配置文件
        方法：POST
        返回：
            - success: 是否成功
            - message: 状态消息
        """
        try:
            dynamic_config.save_config()
            return web.Response(
                content_type="application/json",
                text=json.dumps({"success": True, "message": "配置已保存"}, ensure_ascii=False),
            )
        except Exception as e:
            logger.exception('保存配置失败:')
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
            else:
                use_custom_silent = True  # 默认值
                custom_silent_audiotype = ""  # 默认值
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": 0,
                    "data": {
                        "use_custom_silent": use_custom_silent,
                        "custom_silent_audiotype": custom_silent_audiotype
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