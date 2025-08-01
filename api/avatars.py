"""
头像和动作管理相关API接口
"""
import json
import os
import glob
from aiohttp import web
from logger import logger


class AvatarsAPI:
    """头像和动作管理API接口类"""
    
    async def get_actions(self, request):
        """
        获取可用动作列表接口

        功能：扫描data/customvideo目录，返回所有可用的动作名称
        方法：GET
        返回：
            - code: 状态码（0=成功）
            - data: 动作列表，每个动作包含：
                * name: 动作目录名称
                * description: 动作描述（来自custom_config.json）
                * created_time: 创建时间

        使用场景：
            - 前端动作选择界面
            - 动态加载可用动作
            - 动作管理功能
        """
        try:
            custom_config_path = "data/custom_config.json"
            actions_list = []
            
            # 检查custom_config.json是否存在
            if not os.path.exists(custom_config_path):
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        "code": -1,
                        "msg": f"Custom config file not found: {custom_config_path}"
                    }),
                )
            
            # 读取custom_config.json
            with open(custom_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 构建动作列表
            for item in config_data:
                action_info = {
                    "name": item.get('audiotype', ''),
                    "description": item.get('description', ''),
                    "created_time": item.get('created_time', ''),
                    "imgpath": item.get('imgpath', ''),
                    "audiopath": item.get('audiopath', '')
                }
                actions_list.append(action_info)
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": 0,
                    "data": actions_list
                }),
            )
            
        except Exception as e:
            logger.exception('获取动作列表失败:')
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": str(e)
                }),
            )

    async def get_avatars(self, request):
        """
        获取可用头像列表接口

        功能：扫描data/avatars目录，返回所有可用的头像名称和预览图
        
        ---
        tags:
          - Avatars
        summary: 获取头像列表
        description: 扫描data/avatars目录，返回所有可用的头像名称和预览图
        produces:
          - application/json
        responses:
          200:
            description: 获取成功
            schema:
              type: object
              properties:
                code:
                  type: integer
                  description: 状态码（0=成功）
                data:
                  type: array
                  items:
                    type: object
                    properties:
                      name:
                        type: string
                        description: 头像目录名称
                      preview_image:
                        type: string
                        description: 预览图片路径
        """
        try:
            avatars_dir = "data/avatars"
            avatars_list = []
            
            # 检查avatars目录是否存在
            if not os.path.exists(avatars_dir):
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        "code": -1,
                        "msg": f"Avatars directory not found: {avatars_dir}"
                    }),
                )
            
            # 扫描avatars目录下的所有子目录
            for item in os.listdir(avatars_dir):
                item_path = os.path.join(avatars_dir, item)
                
                # 只处理目录，跳过文件
                if not os.path.isdir(item_path):
                    continue
                    
                # 跳过特殊目录
                if item.startswith('.') or item == '__MACOSX':
                    continue
                
                # 检查是否有full_imgs目录
                full_imgs_path = os.path.join(item_path, "full_imgs")
                if not os.path.exists(full_imgs_path):
                    continue
                
                # 查找full_imgs目录中的第一张图片作为预览图
                image_files = glob.glob(os.path.join(full_imgs_path, "*.png"))
                if not image_files:
                    # 如果没有PNG文件，尝试其他图片格式
                    image_files = glob.glob(os.path.join(full_imgs_path, "*.jpg")) + \
                                 glob.glob(os.path.join(full_imgs_path, "*.jpeg"))
                
                if image_files:
                    # 按文件名排序，选择第一张图片
                    image_files.sort()
                    preview_image = image_files[0]
                    
                    # 转换为相对路径
                    preview_image = os.path.relpath(preview_image, ".")
                    
                    avatars_list.append({
                        "name": item,
                        "preview_image": preview_image
                    })
            
            # 按名称排序
            avatars_list.sort(key=lambda x: x["name"])
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": 0,
                    "data": avatars_list
                }, ensure_ascii=False),
            )
            
        except Exception as e:
            logger.exception('获取头像列表失败:')
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": str(e)
                }, ensure_ascii=False),
                status=500
            ) 