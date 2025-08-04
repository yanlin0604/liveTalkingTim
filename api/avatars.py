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

        功能：扫描data/customvideo目录，返回所有可用的动作名称和预览图
        
        ---
        tags:
          - Actions
        summary: 获取动作列表
        description: 扫描data/customvideo目录，返回所有可用的动作名称和预览图
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
                        description: 动作目录名称
                      preview_image:
                        type: string
                        description: 预览图片路径
        """
        try:
            customvideo_dir = "data/customvideo"
            actions_list = []
            
            # 检查customvideo目录是否存在
            if not os.path.exists(customvideo_dir):
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        "code": -1,
                        "msg": f"Custom video directory not found: {customvideo_dir}"
                    }),
                )
            
            # 扫描customvideo目录下的所有子目录
            for item in os.listdir(customvideo_dir):
                item_path = os.path.join(customvideo_dir, item)
                
                # 只处理目录，跳过文件
                if not os.path.isdir(item_path):
                    continue
                    
                # 跳过特殊目录
                if item.startswith('.') or item == '__MACOSX':
                    continue
                
                # 查找目录中的图片文件作为预览图
                image_files = glob.glob(os.path.join(item_path, "*.png"))
                if not image_files:
                    # 如果没有PNG文件，尝试其他图片格式
                    image_files = glob.glob(os.path.join(item_path, "*.jpg")) + \
                                 glob.glob(os.path.join(item_path, "*.jpeg"))
                
                if image_files:
                    # 按文件名排序，选择第一张图片
                    image_files.sort()
                    preview_image = image_files[0]
                    
                    # 转换为相对路径
                    preview_image = os.path.relpath(preview_image, ".")
                    
                    actions_list.append({
                        "name": item,
                        "preview_image": preview_image
                    })
            
            # 按名称排序
            actions_list.sort(key=lambda x: x["name"])
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": 0,
                    "data": actions_list
                }, ensure_ascii=False),
            )
            
        except Exception as e:
            logger.exception('获取动作列表失败:')
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": str(e)
                }, ensure_ascii=False),
                status=500
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