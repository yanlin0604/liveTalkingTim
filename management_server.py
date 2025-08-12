###############################################################################
#  Copyright (C) 2025 unimed
#  email: zengyanlin99@gmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

"""
独立的管理服务器
提供配置管理、头像管理、鉴权、视频训练和TTS试听接口
与主数字人服务分离，确保在主服务停止时仍能进行管理操作
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional
import threading
import argparse

from aiohttp import web
import aiohttp_cors

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from swagger import create_swagger_docs
except ImportError as e:
    print(f"Warning: Could not import swagger module: {e}")
    print("Swagger documentation will be disabled.")
    # 创建一个空的swagger函数作为fallback
    def create_swagger_docs(app):
        print("Swagger documentation is disabled due to import error.")
        return app

from api.config import ConfigAPI
from api.avatars import AvatarsAPI
from api.auth import AuthAPI
from api.training import TrainingAPI, TrainingTask
from api.tts import TTSAPI
from api.service import ServiceAPI
from api.audio import AudioAPI
from dynamic_config import dynamic_config, start_config_monitoring, get_config, set_config
from config_callbacks import setup_config_callbacks
from logger import logger

# 训练任务管理
training_tasks: Dict[str, TrainingTask] = {}  # task_id -> TrainingTask object
training_tasks_lock = threading.Lock()
TRAINING_TASKS_FILE = "data/training_tasks.json"  # 训练任务数据文件

def load_training_tasks_from_file():
    """从文件加载训练任务数据"""
    try:
        # 确保data目录存在
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        if Path(TRAINING_TASKS_FILE).exists():
            with open(TRAINING_TASKS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                with training_tasks_lock:
                    for task_id, task_data in data.items():
                        # 重建TrainingTask对象
                        task = TrainingTask(
                            task_id=task_data['task_id'],
                            video_name=task_data['video_name'],
                            video_url=task_data.get('video_url'),
                            train_type=task_data['train_type'],
                            force_retrain=task_data.get('force_retrain', False)
                        )
                        # 恢复其他属性
                        task.status = task_data.get('status', 'pending')
                        task.progress = task_data.get('progress', 0)
                        task.message = task_data.get('message', '任务已创建')
                        task.error = task_data.get('error')
                        task.start_time = task_data.get('start_time', time.time())
                        task.end_time = task_data.get('end_time')
                        task.video_path = Path(task_data['video_path']) if task_data.get('video_path') else None
                        task.is_url_video = task_data.get('is_url_video', False)
                        
                        training_tasks[task_id] = task
                
                logger.info(f"从文件加载了 {len(data)} 个训练任务")
        else:
            logger.info("训练任务文件不存在，将创建新文件")
            
    except Exception as e:
        logger.error(f"加载训练任务文件失败: {e}")

def save_training_tasks_to_file():
    """保存训练任务数据到文件"""
    try:
        # 确保data目录存在
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        with training_tasks_lock:
            # 转换为可序列化的字典
            data = {}
            for task_id, task in training_tasks.items():
                data[task_id] = {
                    'task_id': task.task_id,
                    'video_name': task.video_name,
                    'video_url': task.video_url,
                    'train_type': task.train_type,
                    'force_retrain': task.force_retrain,
                    'status': task.status,
                    'progress': task.progress,
                    'message': task.message,
                    'error': task.error,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'video_path': str(task.video_path) if task.video_path else None,
                    'is_url_video': task.is_url_video
                }
        
        # 保存到文件
        with open(TRAINING_TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存了 {len(data)} 个训练任务到文件")
        
    except Exception as e:
        logger.error(f"保存训练任务文件失败: {e}")

def cleanup_old_completed_tasks(max_age_days: int = 7):
    """清理过期的已完成任务"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        with training_tasks_lock:
            tasks_to_remove = []
            for task_id, task in training_tasks.items():
                # 清理已完成且超过指定天数的任务
                if (task.status in ["completed", "failed", "cancelled"] and 
                    task.end_time and 
                    (current_time - task.end_time) > max_age_seconds):
                    tasks_to_remove.append(task_id)
            
            # 删除过期任务
            for task_id in tasks_to_remove:
                del training_tasks[task_id]
                logger.info(f"清理过期任务: {task_id}")
            
            if tasks_to_remove:
                save_training_tasks_to_file()
                logger.info(f"清理了 {len(tasks_to_remove)} 个过期任务")
                
    except Exception as e:
        logger.error(f"清理过期任务失败: {e}")

def schedule_cleanup():
    """定期清理过期任务"""
    while True:
        try:
            time.sleep(3600)  # 每小时执行一次
            cleanup_old_completed_tasks()
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")

def safe_get_training_task(task_id: str) -> Optional[TrainingTask]:
    """安全获取训练任务"""
    with training_tasks_lock:
        return training_tasks.get(task_id)

def safe_set_training_task(task_id: str, task: TrainingTask):
    """安全设置训练任务"""
    with training_tasks_lock:
        training_tasks[task_id] = task
        # 保存到文件
        save_training_tasks_to_file()

def safe_del_training_task(task_id: str) -> bool:
    """安全删除训练任务"""
    with training_tasks_lock:
        if task_id in training_tasks:
            del training_tasks[task_id]
            # 保存到文件
            save_training_tasks_to_file()
            return True
        return False

def generate_task_id() -> str:
    """生成任务ID"""
    return str(uuid.uuid4())

def update_task_progress(task_id: str, status: str, progress: int, message: str, error: str = None):
    """更新任务进度"""
    task = safe_get_training_task(task_id)
    if task:
        task.status = status
        task.progress = progress
        task.message = message
        if error:
            task.error = error
        if status in ["completed", "failed"]:
            task.end_time = time.time()
        logger.info(f"任务 {task_id} 进度更新: {status} - {progress}% - {message}")
        # 保存到文件
        save_training_tasks_to_file()

async def create_management_app(config_file: str = 'config.json', port: int = 8011):
    """创建管理服务器应用"""
    
    # 初始化动态配置系统
    dynamic_config.config_file = config_file
    
    # 启动配置文件监控
    start_config_monitoring(interval=2.0)
    
    # 设置配置变化回调（这里传入None，因为管理服务器不需要nerfreals）
    setup_config_callbacks(None, None)
    
    logger.info("管理服务器动态配置系统已启动")
    
    # 程序启动时加载训练任务数据
    load_training_tasks_from_file()
    
    # 启动清理线程
    cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
    cleanup_thread.start()
    
    app = web.Application(client_max_size=1024**2*100)
    
    # 初始化API模块
    config_api = ConfigAPI()
    avatars_api = AvatarsAPI()
    auth_api = AuthAPI()
    training_api = TrainingAPI(training_tasks, training_tasks_lock, auth_api)
    tts_api = TTSAPI()
    service_api = ServiceAPI()
    audio_api = AudioAPI()
    
    # 头像管理接口
    app.router.add_get("/get_avatars", avatars_api.get_avatars)  # 获取可用头像列表
    app.router.add_get("/get_actions", avatars_api.get_actions)  # 获取可用动作列表
    app.router.add_get("/get_config_for_frontend", config_api.get_config_for_frontend)  # 获取前端配置
    
    # 配置管理接口
    app.router.add_get("/get_config", config_api.get_config_api)  # 获取当前配置
    app.router.add_post("/update_config", config_api.update_config_api)  # 更新配置参数
    app.router.add_post("/reset_config", config_api.reset_config_api)  # 重置配置
    
    # 鉴权接口
    app.router.add_post("/auth/token", auth_api.get_token_api)  # 获取访问token
    app.router.add_post("/auth/revoke", auth_api.revoke_token_api)  # 撤销访问token
    app.router.add_post("/auth/verify", auth_api.verify_token_api)  # 验证访问token
    
    # 视频训练接口（需要认证）
    app.router.add_post("/train_video", training_api.train_video_api)  # 根据视频名称单独训练头像或动作
    app.router.add_get("/training/progress/{task_id}", training_api.get_training_progress)  # 获取训练任务进度
    app.router.add_get("/training/tasks", training_api.list_training_tasks)  # 获取所有训练任务列表
    app.router.add_post("/training/cancel/{task_id}", training_api.cancel_training_task)  # 取消训练任务
    app.router.add_delete("/training/delete/{task_id}", training_api.delete_training_task_api)  # 删除训练任务
    
    # TTS试听接口
    app.router.add_post("/preview_tts", tts_api.preview_tts)  # TTS试听接口
    
    # 主数字人服务管理接口
    app.router.add_get("/get_status", service_api.get_status)  # 查询主数字人服务状态接口
    app.router.add_post("/start_service", service_api.start_service)  # 启动主数字人服务接口
    app.router.add_post("/stop_service", service_api.stop_service)  # 停止主数字人服务接口
    
    # 音频管理接口
    app.router.add_post("/audio/upload", audio_api.upload_file)         # 上传本地音频文件
    app.router.add_post("/audio/upload_url", audio_api.upload_url)      # 通过远程URL保存音频
    app.router.add_get("/audio/list", audio_api.list_audios)            # 列出音频记录
    app.router.add_delete("/audio/{id}", audio_api.delete_audio)        # 根据ID删除音频及索引
    
    # 添加Swagger文档
    create_swagger_docs(app)
    
    # 添加静态文件服务
    app.router.add_static('/', path='web')
    app.router.add_static('/data', path='data')
    
    # 配置CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    # 为所有路由配置CORS
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LiveTalking 管理服务器')
    parser.add_argument('--port', type=int, default=8011, help='管理服务器端口 (默认: 8011)')
    parser.add_argument('--config_file', type=str, default='config.json', help='配置文件路径 (默认: config.json)')
    
    args = parser.parse_args()
    
    logger.info(f"启动LiveTalking管理服务器，端口: {args.port}")
    logger.info(f"配置文件: {args.config_file}")
    
    app = await create_management_app(args.config_file, args.port)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', args.port)
    await site.start()
    
    logger.info(f'管理服务器已启动: http://localhost:{args.port}')
    logger.info(f'API文档地址: http://localhost:{args.port}/swagger')
    logger.info('按 Ctrl+C 停止服务器')
    
    try:
        await asyncio.Future()  # 保持服务器运行
    except KeyboardInterrupt:
        logger.info("正在关闭管理服务器...")
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("管理服务器已停止")
    except Exception as e:
        logger.error(f"管理服务器启动失败: {e}")
        sys.exit(1) 