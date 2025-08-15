"""
训练相关API接口
"""
import json
import shutil
import asyncio
import time
import uuid
import urllib.parse
import tempfile
import random
from datetime import datetime
from pathlib import Path
from aiohttp import web
from logger import logger
# from .auth import require_auth  # 暂时注释掉，因为我们直接在方法中实现认证


class TrainingTask:
    """训练任务类"""
    def __init__(self, task_id: str, video_name: str, video_url, 
                 train_type: str, force_retrain: bool):
        self.task_id = task_id
        self.video_name = video_name
        self.video_url = video_url
        self.train_type = train_type
        self.force_retrain = force_retrain
        self.status = "pending"  # pending, processing, training, completed, failed
        self.progress = 0  # 0-100
        self.message = "任务已创建"
        self.error = None
        self.start_time = time.time()
        self.end_time = None
        self.video_path = None
        self.is_url_video = False


class TrainingAPI:
    """训练相关API接口类"""
    
    def __init__(self, training_tasks_dict, training_tasks_lock, auth_api=None):
        self.training_tasks = training_tasks_dict
        self.training_tasks_lock = training_tasks_lock
        self.auth_api = auth_api
    
    def generate_task_id(self) -> str:
        """生成任务ID"""
        return str(uuid.uuid4())
    
    def generate_unique_video_name(self, base_name: str, train_type: str = 'avatar', task_id: str = None) -> str:
        """生成唯一的视频名称，格式为: base_name_日期_task_id_类型"""
        # 使用线程锁确保并发安全
        with self.training_tasks_lock:
            # 获取当前日期
            current_date = datetime.now().strftime("%Y%m%d")
            
            # 如果没有提供task_id，生成一个
            if not task_id:
                task_id = self.generate_task_id()
            
            # 根据训练类型生成最终名称，剔除前缀
            if train_type == 'avatar':
                # 头像训练
                clean_base_name = base_name
                unique_name = f"{clean_base_name}_{current_date}_{task_id}_avatar"
            else:  # action
                # 动作训练：剔除 action_ 前缀
                clean_base_name = base_name
                unique_name = f"{clean_base_name}_{current_date}_{task_id}"
            
            # 立即将生成的名称添加到内存中，避免并发冲突
            temp_task = TrainingTask(
                task_id=f"temp_{len(self.training_tasks)}",
                video_name=unique_name,
                video_url=None,
                train_type=train_type,
                force_retrain=False
            )
            self.training_tasks[temp_task.task_id] = temp_task
            
            return unique_name
    
    def safe_get_training_task(self, task_id: str):
        """安全获取训练任务"""
        with self.training_tasks_lock:
            task = self.training_tasks.get(task_id)
            if task:
                logger.debug(f"成功获取训练任务: {task_id}")
            else:
                logger.debug(f"训练任务不存在: {task_id}")
            return task
    
    def safe_set_training_task(self, task_id: str, task: TrainingTask):
        """安全设置训练任务"""
        logger.info(f"开始安全设置训练任务: {task_id}")
        with self.training_tasks_lock:
            self.training_tasks[task_id] = task
            logger.info(f"训练任务已添加到内存: {task_id}")
        
        # 保存到文件（在锁外调用，避免死锁）
        logger.info(f"开始保存训练任务到文件: {task_id}")
        self.save_training_tasks_to_file()
        logger.info(f"训练任务设置完成: {task_id}")
    
    def save_training_tasks_to_file(self):
        """保存训练任务数据到文件"""
        try:
            # 确保data目录存在
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            with self.training_tasks_lock:
                # 转换为可序列化的字典
                data = {}
                for task_id, task in self.training_tasks.items():
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
            with open("data/training_tasks.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存了 {len(data)} 个训练任务到文件")
            
        except Exception as e:
            logger.error(f"保存训练任务文件失败: {e}")
    
    async def _verify_auth(self, request):
        """验证认证token"""
        # 从请求头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return web.json_response({
                "success": False,
                "error": "缺少认证信息",
                "message": "请在请求头中提供 Authorization: Bearer <token>"
            }, status=401)
        
        # 检查Authorization格式
        if not auth_header.startswith('Bearer '):
            return web.json_response({
                "success": False,
                "error": "认证格式错误",
                "message": "Authorization格式应为: Bearer <token>"
            }, status=401)
        
        # 提取token
        token = auth_header[7:]  # 去掉 "Bearer " 前缀
        
        # 验证token
        client_uuid = self.auth_api.verify_token(token)
        if not client_uuid:
            return web.json_response({
                "success": False,
                "error": "token无效",
                "message": "token无效或已过期，请重新获取"
            }, status=401)
        
        # 将client_uuid添加到请求中，供后续使用
        request['client_uuid'] = client_uuid
        return None
    
    def update_task_progress(self, task_id: str, status: str, progress: int, message: str, error: str = None):
        """更新任务进度"""
        task = self.safe_get_training_task(task_id)
        if task:
            with self.training_tasks_lock:
                task.status = status
                task.progress = progress
                task.message = message
                if error:
                    task.error = error
                if status in ["completed", "failed"]:
                    task.end_time = time.time()
            
            logger.info(f"任务 {task_id} 进度更新: {status} - {progress}% - {message}")
            # 保存到文件（在锁外调用，避免死锁）
            self.save_training_tasks_to_file()
    
    async def train_video_api(self, request):
        """
        根据视频名称或URL单独训练头像或动作 - 异步版本
        
        ---
        tags:
          - Training
        summary: 训练视频
        description: 根据视频名称或URL单独训练头像或动作（需要认证）
        consumes:
          - application/json
        produces:
          - application/json
        security:
          - Bearer: []
        parameters:
          - in: header
            name: Authorization
            required: true
            type: string
            description: Bearer token for authentication
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                task_id:
                  type: string
                  description: 任务ID，如果不提供则自动生成。如果提供的task_id已存在且状态为completed/failed/cancelled，将允许重新训练
                video_name:
                  type: string
                  description: 视频文件名（不含扩展名），当提供video_url时，此名称将作为记录的文件名
                video_url:
                  type: string
                  description: 视频URL地址，如果同时提供video_name，将保留video_name作为记录的文件名
                type:
                  type: string
                  enum: [avatar, action]
                  description: 训练类型（avatar=头像训练，action=动作训练）
                  default: avatar
                force_retrain:
                  type: boolean
                  description: 是否强制重新训练
                  default: false
        responses:
          200:
            description: 任务创建成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                message:
                  type: string
                  description: 状态消息
                task_id:
                  type: string
                  description: 任务ID
                status:
                  type: string
                  description: 任务状态
          401:
            description: 认证失败
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        logger.info("=== 开始处理 train_video API 请求 ===")
        
        # 验证token
        if self.auth_api:
            logger.info("开始验证认证token")
            auth_result = await self._verify_auth(request)
            if auth_result:
                logger.warning("认证验证失败")
                return auth_result
            logger.info("认证验证成功")
        
        try:
            # 检查请求体是否为空
            logger.info("开始解析请求体")
            body = await request.text()
            if not body.strip():
                logger.error("请求体为空")
                return web.json_response({
                    "success": False,
                    "error": "请求体为空",
                    "message": "请提供有效的JSON数据"
                }, status=200)
            
            logger.info(f"请求体内容长度: {len(body)} 字符")
            data = await request.json()
            logger.info(f"JSON解析成功，请求参数: {data}")
            
            task_id = data.get('task_id')  # 获取传入的任务ID
            video_name = data.get('video_name')
            video_url = data.get('video_url')  # 新增：支持视频URL
            train_type = data.get('type', 'avatar')  # avatar 或 action
            force_retrain = data.get('force_retrain', False)  # 是否强制重新训练
            
            logger.info(f"解析参数完成 - task_id: {task_id}, video_name: {video_name}, video_url: {video_url}, train_type: {train_type}, force_retrain: {force_retrain}")
            
            # 参数验证
            logger.info("开始参数验证")
            if not video_name and not video_url:
                logger.error("缺少必要参数: video_name 和 video_url 都为空")
                return web.json_response({
                    "success": False,
                    "error": "缺少必要参数",
                    "message": "请提供 video_name 或 video_url 参数"
                }, status=200)
            
            if train_type not in ['avatar', 'action']:
                logger.error(f"无效的训练类型: {train_type}")
                return web.json_response({
                    "success": False,
                    "error": "无效的训练类型",
                    "message": "训练类型必须是 'avatar' 或 'action'"
                }, status=200)
            
            logger.info("参数验证通过")
            
            # 生成任务ID - 优先使用传入的task_id，如果没有则自动生成
            logger.info("开始处理任务ID")
            if not task_id:
                task_id = self.generate_task_id()
                logger.info(f"自动生成任务ID: {task_id}")
            else:
                logger.info(f"使用传入的任务ID: {task_id}")
                # 检查传入的task_id是否已存在
                existing_task = self.safe_get_training_task(task_id)
                if existing_task:
                    logger.info(f"任务ID {task_id} 已存在，当前状态: {existing_task.status}")
                    # 如果任务已完成或失败，允许重新训练
                    if existing_task.status in ["completed", "failed", "cancelled"]:
                        logger.info(f"任务 {task_id} 状态为 {existing_task.status}，允许重新训练")
                        # 删除旧任务，创建新任务
                        with self.training_tasks_lock:
                            del self.training_tasks[task_id]
                        logger.info(f"已删除旧任务 {task_id}")
                    else:
                        # 如果任务正在处理中，不允许重新训练
                        logger.warning(f"任务 {task_id} 正在处理中，状态: {existing_task.status}")
                        return web.json_response({
                            "success": False,
                            "error": "任务ID正在使用中",
                            "message": f"任务正在处理中（状态：{existing_task.status}），请等待完成"
                        }, status=200)
            
            # 生成唯一的视频名称
            logger.info("开始生成唯一的视频名称")
            if video_name:
                # 如果提供了video_name，生成唯一名称
                unique_video_name = self.generate_unique_video_name(video_name, train_type, task_id)
                logger.info(f"原始视频名称: {video_name}, 生成的唯一名称: {unique_video_name}")
            else:
                # 如果没有提供video_name，使用默认名称
                unique_video_name = self.generate_unique_video_name("video", train_type, task_id)
                logger.info(f"未提供视频名称，使用默认名称: {unique_video_name}")
            
            # 创建训练任务
            logger.info("开始创建训练任务对象")
            task = TrainingTask(
                task_id=task_id,
                video_name=unique_video_name,
                video_url=video_url,
                train_type=train_type,
                force_retrain=force_retrain
            )
            logger.info(f"训练任务对象创建成功: {task_id}")
            
            # 保存任务
            logger.info("开始保存训练任务")
            self.safe_set_training_task(task_id, task)
            logger.info(f"训练任务已保存到内存和文件: {task_id}")
            
            logger.info(f"创建训练任务成功: {task_id}, 视频名称={unique_video_name}, URL={video_url or '本地文件'}, 类型={train_type}")
            
            # 启动异步训练任务，添加异常处理
            logger.info("开始启动异步训练任务")
            try:
                # 使用asyncio.create_task并添加异常处理
                training_task = asyncio.create_task(self.execute_training_task(task_id))
                logger.info(f"异步训练任务已创建: {task_id}")
                
                # 添加异常处理回调
                def handle_exception(task):
                    try:
                        task.result()
                    except Exception as e:
                        logger.error(f"训练任务 {task_id} 执行失败: {e}")
                        # 更新任务状态为失败
                        self.update_task_progress(task_id, "failed", 0, f"训练任务执行失败: {str(e)}", str(e))
                
                training_task.add_done_callback(handle_exception)
                logger.info(f"异常处理回调已添加: {task_id}")
                
            except Exception as e:
                logger.error(f"启动训练任务失败: {e}")
                self.update_task_progress(task_id, "failed", 0, f"启动训练任务失败: {str(e)}", str(e))
                return web.json_response({
                    "success": False,
                    "error": "启动训练任务失败",
                    "message": f"无法启动训练任务: {str(e)}"
                }, status=200)
            
            logger.info(f"=== train_video API 请求处理完成，任务ID: {task_id} ===")
            return web.json_response({
                "success": True,
                "message": "训练任务已创建",
                "task_id": task_id,
                "status": "pending"
            })
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return web.json_response({
                "success": False,
                "error": "JSON格式错误",
                "message": f"无效的JSON格式: {str(e)}"
            }, status=200)
        except Exception as e:
            logger.error(f"创建训练任务失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "创建训练任务时发生错误"
            }, status=200)
    
    async def execute_training_task(self, task_id: str):
        """异步执行训练任务"""
        logger.info(f"=== 开始执行训练任务: {task_id} ===")
        
        task = self.safe_get_training_task(task_id)
        if not task:
            logger.error(f"任务 {task_id} 不存在，无法执行")
            return
        
        logger.info(f"获取到训练任务: {task_id}, 视频名称: {task.video_name}, 训练类型: {task.train_type}")
        
        try:
            logger.info(f"任务 {task_id}: 开始处理训练任务")
            self.update_task_progress(task_id, "pending", 0, "开始处理训练任务")
            
            video_path = None
            is_url_video = False
            
            # 处理URL视频
            if task.video_url:
                logger.info(f"任务 {task_id}: 检测到URL视频，开始处理")
                try:
                    logger.info(f"任务 {task_id}: 更新任务状态为处理中")
                    self.update_task_progress(task_id, "processing", 10, "正在处理视频文件")
                    
                    # 验证URL格式
                    logger.info(f"任务 {task_id}: 开始验证URL格式: {task.video_url}")
                    parsed_url = urllib.parse.urlparse(task.video_url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        logger.error(f"任务 {task_id}: URL格式无效: {task.video_url}")
                        raise Exception("无效的URL格式")
                    logger.info(f"任务 {task_id}: URL格式验证通过")
                    
                    # 创建临时目录
                    logger.info(f"任务 {task_id}: 创建临时目录")
                    temp_dir = Path(tempfile.gettempdir()) / "Unimed_videos"
                    temp_dir.mkdir(exist_ok=True)
                    logger.info(f"任务 {task_id}: 临时目录创建成功: {temp_dir}")
                    
                    # 如果video_name为空或为"unknown"，则从URL中提取文件名
                    if not task.video_name or task.video_name == "unknown":
                        logger.info(f"任务 {task_id}: video_name为空，从URL提取文件名")
                        url_path = parsed_url.path
                        video_name = Path(url_path).stem
                        if not video_name:
                            video_name = f"video_{int(time.time())}"
                        task.video_name = video_name
                        logger.info(f"任务 {task_id}: 从URL提取的文件名: {video_name}")
                    else:
                        logger.info(f"任务 {task_id}: 使用传入的video_name: {task.video_name}")
                    
                    # 确定文件扩展名
                    logger.info(f"任务 {task_id}: 确定文件扩展名")
                    url_path = parsed_url.path.lower()
                    if '.mp4' in url_path:
                        ext = '.mp4'
                    elif '.avi' in url_path:
                        ext = '.avi'
                    elif '.mov' in url_path:
                        ext = '.mov'
                    elif '.mkv' in url_path:
                        ext = '.mkv'
                    elif '.flv' in url_path:
                        ext = '.flv'
                    elif '.wmv' in url_path:
                        ext = '.wmv'
                    else:
                        ext = '.mp4'  # 默认扩展名
                    logger.info(f"任务 {task_id}: 确定的文件扩展名: {ext}")
                    
                    # 构建本地文件路径
                    local_filename = f"{task.video_name}{ext}"
                    video_path = temp_dir / local_filename
                    task.video_path = video_path
                    task.is_url_video = True
                    is_url_video = True
                    logger.info(f"任务 {task_id}: 本地文件路径: {video_path}")
                    
                    # 下载视频文件
                    logger.info(f"任务 {task_id}: 开始下载视频文件")
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        logger.info(f"任务 {task_id}: 创建HTTP会话")
                        async with session.get(task.video_url) as response:
                            logger.info(f"任务 {task_id}: HTTP响应状态码: {response.status}")
                            if response.status != 200:
                                logger.error(f"任务 {task_id}: 下载失败，HTTP状态码: {response.status}")
                                raise Exception(f"无法下载视频，HTTP状态码: {response.status}")
                            
                            # 检查文件大小
                            content_length = response.headers.get('content-length')
                            if content_length:
                                file_size = int(content_length)
                                logger.info(f"任务 {task_id}: 文件大小: {file_size / 1024 / 1024:.1f}MB")
                                if file_size > 500 * 1024 * 1024:  # 500MB限制
                                    logger.error(f"任务 {task_id}: 文件过大: {file_size / 1024 / 1024:.1f}MB")
                                    raise Exception(f"视频文件过大 ({file_size / 1024 / 1024:.1f}MB)，最大支持500MB")
                            
                            # 写入文件
                            logger.info(f"任务 {task_id}: 开始写入文件")
                            total_size = 0
                            with open(video_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                                    total_size += len(chunk)
                                    # 更新下载进度
                                    if content_length:
                                        download_progress = min(30, int(10 + (total_size / int(content_length)) * 20))
                                        self.update_task_progress(task_id, "processing", download_progress, f"正在获取视频文件 ({total_size / 1024 / 1024:.1f}MB)")
                            
                            logger.info(f"任务 {task_id}: 文件下载完成，总大小: {total_size / 1024 / 1024:.1f}MB")
                    
                    logger.info(f"任务 {task_id}: 视频文件获取完成，开始训练")
                    self.update_task_progress(task_id, "training", 30, "视频文件获取完成，开始训练")
                    
                except Exception as e:
                    logger.error(f"任务 {task_id}: 获取视频文件失败: {e}")
                    self.update_task_progress(task_id, "failed", 0, f"获取视频文件失败: {str(e)}", str(e))
                    return
            
            # 处理本地视频文件
            else:
                logger.info(f"任务 {task_id}: 检测到本地视频文件，开始查找")
                self.update_task_progress(task_id, "training", 10, "正在查找本地视频文件")
                
                # 查找视频文件
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                logger.info(f"任务 {task_id}: 支持的视频扩展名: {video_extensions}")
                
                # 根据训练类型确定扫描目录
                if task.train_type == 'avatar':
                    scan_dirs = ['videos', 'data/videos', 'uploads']
                else:  # action
                    scan_dirs = ['action_videos', 'data/action_videos', 'uploads']
                logger.info(f"任务 {task_id}: 扫描目录: {scan_dirs}")
                
                for scan_dir in scan_dirs:
                    scan_path = Path(scan_dir)
                    logger.info(f"任务 {task_id}: 检查目录: {scan_path}")
                    if scan_path.exists():
                        logger.info(f"任务 {task_id}: 目录存在: {scan_path}")
                        for ext in video_extensions:
                            potential_path = scan_path / f"{task.video_name}{ext}"
                            logger.info(f"任务 {task_id}: 检查文件: {potential_path}")
                            if potential_path.exists():
                                video_path = potential_path
                                logger.info(f"任务 {task_id}: 找到视频文件: {video_path}")
                                break
                            # 也检查大写扩展名
                            potential_path = scan_path / f"{task.video_name}{ext.upper()}"
                            logger.info(f"任务 {task_id}: 检查文件: {potential_path}")
                            if potential_path.exists():
                                video_path = potential_path
                                logger.info(f"任务 {task_id}: 找到视频文件: {video_path}")
                                break
                        if video_path:
                            break
                    else:
                        logger.info(f"任务 {task_id}: 目录不存在: {scan_path}")
                
                if not video_path:
                    logger.error(f"任务 {task_id}: 在目录 {', '.join(scan_dirs)} 中未找到视频文件: {task.video_name}")
                    self.update_task_progress(task_id, "failed", 0, f"在目录 {', '.join(scan_dirs)} 中未找到视频文件: {task.video_name}", "视频文件未找到")
                    return
                
                task.video_path = video_path
                logger.info(f"任务 {task_id}: 找到视频文件，开始训练: {video_path}")
                self.update_task_progress(task_id, "training", 20, "找到视频文件，开始训练")
            
            # 执行训练 - 使用线程池避免阻塞事件循环
            logger.info(f"任务 {task_id}: 开始初始化训练环境")
            self.update_task_progress(task_id, "training", 40, "正在初始化训练环境")
            
            # 获取事件循环
            loop = asyncio.get_event_loop()
            logger.info(f"任务 {task_id}: 获取事件循环成功")
            
            if task.train_type == 'avatar':
                logger.info(f"任务 {task_id}: 开始头像训练")
                # 在线程池中执行头像训练
                success = await loop.run_in_executor(
                    None, 
                    self._train_avatar_sync, 
                    task_id, 
                    video_path, 
                    is_url_video, 
                    task.force_retrain
                )
                logger.info(f"任务 {task_id}: 头像训练完成，结果: {success}")
            else:  # action
                logger.info(f"任务 {task_id}: 开始动作训练")
                # 在线程池中执行动作训练
                success = await loop.run_in_executor(
                    None, 
                    self._train_action_sync, 
                    task_id, 
                    video_path, 
                    is_url_video, 
                    task.force_retrain
                )
                logger.info(f"任务 {task_id}: 动作训练完成，结果: {success}")
            
            if success:
                logger.info(f"任务 {task_id}: 训练成功，开始清理临时文件")
                self.update_task_progress(task_id, "training", 90, "训练完成，正在清理临时文件")
                
                # 清理临时文件（如果是URL视频）
                if is_url_video and video_path and video_path.exists():
                    try:
                        logger.info(f"任务 {task_id}: 清理临时文件: {video_path}")
                        video_path.unlink()
                        logger.info(f"任务 {task_id}: 临时文件清理成功")
                    except Exception as e:
                        logger.warning(f"任务 {task_id}: 清理临时文件失败: {e}")
                
                logger.info(f"任务 {task_id}: 训练任务完成")
                self.update_task_progress(task_id, "completed", 100, f"视频 {task.video_name} {task.train_type}训练成功")
            else:
                logger.error(f"任务 {task_id}: 训练失败")
                # 清理临时文件（如果训练失败）
                if is_url_video and video_path and video_path.exists():
                    try:
                        logger.info(f"任务 {task_id}: 训练失败，清理临时文件: {video_path}")
                        video_path.unlink()
                        logger.info(f"任务 {task_id}: 临时文件清理成功")
                    except Exception as e:
                        logger.warning(f"任务 {task_id}: 清理临时文件失败: {e}")
                
                logger.error(f"任务 {task_id}: 训练失败，更新任务状态")
                self.update_task_progress(task_id, "failed", 0, f"视频 {task.video_name} {task.train_type}训练失败，请检查日志", "训练失败")
                
        except Exception as e:
            logger.error(f"任务 {task_id}: 训练过程中发生异常: {e}")
            # 清理临时文件（如果发生异常）
            if hasattr(task, 'video_path') and task.video_path and task.video_path.exists():
                try:
                    logger.info(f"任务 {task_id}: 发生异常，清理临时文件: {task.video_path}")
                    task.video_path.unlink()
                    logger.info(f"任务 {task_id}: 临时文件清理成功")
                except Exception as cleanup_error:
                    logger.warning(f"任务 {task_id}: 清理临时文件失败: {cleanup_error}")
            
            logger.error(f"任务 {task_id}: 更新任务状态为失败")
            self.update_task_progress(task_id, "failed", 0, f"训练过程中发生错误: {str(e)}", str(e))
        
        logger.info(f"=== 训练任务执行完成: {task_id} ===")
    
    def _train_avatar_sync(self, task_id: str, video_path: Path, is_url_video: bool, force_retrain: bool) -> bool:
        """同步执行头像训练（在线程池中运行）"""
        logger.info(f"任务 {task_id}: === 开始头像训练 ===")
        logger.info(f"任务 {task_id}: 视频路径: {video_path}")
        logger.info(f"任务 {task_id}: 是否URL视频: {is_url_video}")
        logger.info(f"任务 {task_id}: 强制重新训练: {force_retrain}")
        
        try:
            logger.info(f"任务 {task_id}: 导入VideoScanner模块")
            from video_scanner import VideoScanner
            
            logger.info(f"任务 {task_id}: 创建VideoScanner实例")
            scanner = VideoScanner(
                scan_directory=str(video_path.parent),
                avatar_base_dir="data/avatars",
                config_file="video_scanner_config.json" if Path("video_scanner_config.json").exists() else None
            )
            logger.info(f"任务 {task_id}: VideoScanner实例创建成功")
            
            # 检查是否已经训练过（仅对本地文件）
            if not is_url_video and not force_retrain:
                logger.info(f"任务 {task_id}: 检查是否已经训练过")
                if scanner.is_trained(video_path):
                    logger.warning(f"任务 {task_id}: 视频 {video_path.stem} 已经训练过")
                    self.update_task_progress(task_id, "failed", 0, f"视频 {video_path.stem} 已经训练过，如需重新训练请设置 force_retrain=true", "视频已训练过")
                    return False
                logger.info(f"任务 {task_id}: 视频未训练过，可以继续训练")
            else:
                logger.info(f"任务 {task_id}: 跳过训练检查（URL视频或强制重新训练）")
            
            logger.info(f"任务 {task_id}: 开始执行头像训练")
            self.update_task_progress(task_id, "training", 60, "正在训练头像模型")
            success = scanner.train_video(video_path)
            logger.info(f"任务 {task_id}: 头像训练执行完成，结果: {success}")
            return success
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 头像训练失败: {e}")
            self.update_task_progress(task_id, "failed", 0, f"头像训练失败: {str(e)}", str(e))
            return False
    
    def _train_action_sync(self, task_id: str, video_path: Path, is_url_video: bool, force_retrain: bool) -> bool:
        """同步执行动作训练（在线程池中运行）"""
        logger.info(f"任务 {task_id}: === 开始动作训练 ===")
        logger.info(f"任务 {task_id}: 视频路径: {video_path}")
        logger.info(f"任务 {task_id}: 是否URL视频: {is_url_video}")
        logger.info(f"任务 {task_id}: 强制重新训练: {force_retrain}")
        
        try:
            logger.info(f"任务 {task_id}: 导入ActionScanner模块")
            from action_scanner import ActionScanner
            
            logger.info(f"任务 {task_id}: 创建ActionScanner实例")
            scanner = ActionScanner(
                scan_directory=str(video_path.parent),
                action_base_dir="data/customvideo",
                config_file="action_scanner_config.json" if Path("action_scanner_config.json").exists() else None
            )
            logger.info(f"任务 {task_id}: ActionScanner实例创建成功")
            
            # 检查是否已经处理过（仅对本地文件）
            if not is_url_video and not force_retrain:
                logger.info(f"任务 {task_id}: 检查是否已经处理过")
                if scanner.is_processed(video_path):
                    logger.warning(f"任务 {task_id}: 视频 {video_path.stem} 已经处理过")
                    self.update_task_progress(task_id, "failed", 0, f"视频 {video_path.stem} 已经处理过，如需重新处理请设置 force_retrain=true", "视频已处理过")
                    return False
                logger.info(f"任务 {task_id}: 视频未处理过，可以继续处理")
            else:
                logger.info(f"任务 {task_id}: 跳过处理检查（URL视频或强制重新训练）")
            
            logger.info(f"任务 {task_id}: 开始执行动作处理")
            self.update_task_progress(task_id, "training", 60, "正在处理动作编排")
            success = scanner.process_video(video_path)
            logger.info(f"任务 {task_id}: 动作处理执行完成，结果: {success}")
            return success
            
        except Exception as e:
            logger.error(f"任务 {task_id}: 动作训练失败: {e}")
            self.update_task_progress(task_id, "failed", 0, f"动作训练失败: {str(e)}", str(e))
            return False
    
    async def get_training_progress(self, request):
        """
        获取训练任务进度
        
        ---
        tags:
          - Training
        summary: 获取训练进度
        description: 获取指定训练任务的进度信息（需要认证）
        produces:
          - application/json
        security:
          - Bearer: []
        parameters:
          - in: header
            name: Authorization
            required: true
            type: string
            description: Bearer token for authentication
          - in: path
            name: task_id
            required: true
            type: string
            description: 任务ID
        responses:
          200:
            description: 获取成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                task_id:
                  type: string
                  description: 任务ID
                video_name:
                  type: string
                  description: 视频名称
                video_url:
                  type: string
                  description: 视频URL
                train_type:
                  type: string
                  description: 训练类型
                status:
                  type: string
                  description: 任务状态
                progress:
                  type: integer
                  description: 进度百分比
                message:
                  type: string
                  description: 状态消息
                error:
                  type: string
                  description: 错误信息
                start_time:
                  type: number
                  description: 开始时间
                end_time:
                  type: number
                  description: 结束时间
                duration:
                  type: number
                  description: 运行时长
                is_url_video:
                  type: boolean
                  description: 是否为URL视频
          401:
            description: 认证失败
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        logger.info("=== 开始处理获取训练进度请求 ===")
        
        # 验证token
        if self.auth_api:
            logger.info("开始验证认证token")
            auth_result = await self._verify_auth(request)
            if auth_result:
                logger.warning("认证验证失败")
                return auth_result
            logger.info("认证验证成功")
        
        try:
            task_id = request.match_info.get('task_id')
            logger.info(f"请求查询任务进度: {task_id}")
            
            if not task_id:
                logger.error("缺少任务ID参数")
                return web.json_response({
                    "success": False,
                    "error": "缺少任务ID",
                    "message": "请提供任务ID"
                }, status=200)
            
            task = self.safe_get_training_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return web.json_response({
                    "success": False,
                    "error": "任务不存在",
                    "message": f"任务ID {task_id} 不存在"
                }, status=200)
            
            logger.info(f"找到任务: {task_id}, 状态: {task.status}, 进度: {task.progress}%")
            
            # 计算运行时间
            duration = time.time() - task.start_time
            if task.end_time:
                duration = task.end_time - task.start_time
            
            logger.info(f"任务 {task_id} 运行时长: {duration:.2f}秒")
            
            return web.json_response({
                "success": True,
                "task_id": task.task_id,
                "video_name": task.video_name,
                "video_url": task.video_url,
                "train_type": task.train_type,
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
                "error": task.error,
                "start_time": task.start_time,
                "end_time": task.end_time,
                "duration": duration,
                "is_url_video": task.is_url_video
            })
            
        except Exception as e:
            logger.error(f"获取训练进度失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "获取训练进度时发生错误"
            }, status=200)
    
    async def list_training_tasks(self, request):
        """
        获取所有训练任务列表
        
        ---
        tags:
          - Training
        summary: 获取训练任务列表
        description: 获取所有训练任务列表（需要认证）
        produces:
          - application/json
        security:
          - Bearer: []
        parameters:
          - in: header
            name: Authorization
            required: true
            type: string
            description: Bearer token for authentication
        responses:
          200:
            description: 获取成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                tasks:
                  type: array
                  items:
                    type: object
                    properties:
                      task_id:
                        type: string
                        description: 任务ID
                      video_name:
                        type: string
                        description: 视频名称
                      video_url:
                        type: string
                        description: 视频URL
                      train_type:
                        type: string
                        description: 训练类型
                      status:
                        type: string
                        description: 任务状态
                      progress:
                        type: integer
                        description: 进度百分比
                      message:
                        type: string
                        description: 状态消息
                      start_time:
                        type: number
                        description: 开始时间
                      duration:
                        type: number
                        description: 运行时长
                      is_url_video:
                        type: boolean
                        description: 是否为URL视频
                total:
                  type: integer
                  description: 任务总数
          401:
            description: 认证失败
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        logger.info("=== 开始处理获取训练任务列表请求 ===")
        
        # 验证token
        if self.auth_api:
            logger.info("开始验证认证token")
            auth_result = await self._verify_auth(request)
            if auth_result:
                logger.warning("认证验证失败")
                return auth_result
            logger.info("认证验证成功")
        
        try:
            logger.info("开始获取训练任务列表")
            with self.training_tasks_lock:
                tasks = []
                logger.info(f"当前共有 {len(self.training_tasks)} 个训练任务")
                
                for task_id, task in self.training_tasks.items():
                    duration = time.time() - task.start_time
                    if task.end_time:
                        duration = task.end_time - task.start_time
                    
                    tasks.append({
                        "task_id": task.task_id,
                        "video_name": task.video_name,
                        "video_url": task.video_url,
                        "train_type": task.train_type,
                        "status": task.status,
                        "progress": task.progress,
                        "message": task.message,
                        "start_time": task.start_time,
                        "duration": duration,
                        "is_url_video": task.is_url_video
                    })
                
                # 按开始时间倒序排列
                tasks.sort(key=lambda x: x["start_time"], reverse=True)
                logger.info(f"返回 {len(tasks)} 个训练任务")
                
                return web.json_response({
                    "success": True,
                    "tasks": tasks,
                    "total": len(tasks)
                })
            
        except Exception as e:
            logger.error(f"获取训练任务列表失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "获取训练任务列表时发生错误"
            }, status=200)
    
    async def cancel_training_task(self, request):
        """
        取消训练任务
        
        ---
        tags:
          - Training
        summary: 取消训练任务
        description: 取消指定的训练任务（需要认证）
        produces:
          - application/json
        security:
          - Bearer: []
        parameters:
          - in: header
            name: Authorization
            required: true
            type: string
            description: Bearer token for authentication
          - in: path
            name: task_id
            required: true
            type: string
            description: 任务ID
        responses:
          200:
            description: 取消成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                message:
                  type: string
                  description: 状态消息
                task_id:
                  type: string
                  description: 任务ID
          401:
            description: 认证失败
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        logger.info("=== 开始处理取消训练任务请求 ===")
        
        # 验证token
        if self.auth_api:
            logger.info("开始验证认证token")
            auth_result = await self._verify_auth(request)
            if auth_result:
                logger.warning("认证验证失败")
                return auth_result
            logger.info("认证验证成功")
        
        try:
            task_id = request.match_info.get('task_id')
            logger.info(f"请求取消任务: {task_id}")
            
            if not task_id:
                logger.error("缺少任务ID参数")
                return web.json_response({
                    "success": False,
                    "error": "缺少任务ID",
                    "message": "请提供任务ID"
                }, status=200)
            
            task = self.safe_get_training_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return web.json_response({
                    "success": False,
                    "error": "任务不存在",
                    "message": f"任务ID {task_id} 不存在"
                }, status=200)
            
            logger.info(f"找到任务: {task_id}, 当前状态: {task.status}")
            
            if task.status in ["completed", "failed", "cancelled"]:
                logger.warning(f"任务 {task_id} 状态为 {task.status}，无法取消")
                return web.json_response({
                    "success": False,
                    "error": "任务无法取消",
                    "message": f"任务状态为 {task.status}，无法取消"
                }, status=400)
            
            # 更新任务状态为取消
            logger.info(f"更新任务 {task_id} 状态为取消")
            self.update_task_progress(task_id, "cancelled", task.progress, "任务已取消")
            
            # 清理临时文件
            if hasattr(task, 'video_path') and task.video_path and task.video_path.exists():
                try:
                    logger.info(f"取消任务，清理临时文件: {task.video_path}")
                    task.video_path.unlink()
                    logger.info(f"临时文件清理成功: {task.video_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
            
            logger.info(f"任务 {task_id} 取消成功")
            return web.json_response({
                "success": True,
                "message": "任务已取消",
                "task_id": task_id
            })
            
        except Exception as e:
            logger.error(f"取消训练任务失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "取消训练任务时发生错误"
            }, status=200) 
    
    async def delete_training_task_api(self, request):
        """
        删除训练任务
        
        ---
        tags:
          - Training
        summary: 删除训练任务
        description: 删除指定的训练任务（需要认证，仅限 avatar 或 action 类型；正在训练中的任务不可删除）
        produces:
          - application/json
        security:
          - Bearer: []
        parameters:
          - in: header
            name: Authorization
            required: true
            type: string
            description: Bearer token for authentication
          - in: path
            name: task_id
            required: true
            type: string
            description: 任务ID
        responses:
          200:
            description: 删除成功
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                message:
                  type: string
                  description: 状态消息
                task_id:
                  type: string
                  description: 任务ID
          401:
            description: 认证失败
            schema:
              type: object
              properties:
                success:
                  type: boolean
                  description: 是否成功
                error:
                  type: string
                  description: 错误信息
                message:
                  type: string
                  description: 错误描述
        """
        logger.info("=== 开始处理删除训练任务请求 ===")
        
        # 验证token
        if self.auth_api:
            logger.info("开始验证认证token")
            auth_result = await self._verify_auth(request)
            if auth_result:
                logger.warning("认证验证失败")
                return auth_result
            logger.info("认证验证成功")
        
        try:
            task_id = request.match_info.get('task_id')
            logger.info(f"请求删除任务: {task_id}")
            
            if not task_id:
                logger.error("缺少任务ID参数")
                return web.json_response({
                    "success": False,
                    "error": "缺少任务ID",
                    "message": "请提供任务ID"
                }, status=200)
            
            task = self.safe_get_training_task(task_id)
            if not task:
                logger.warning(f"任务不存在: {task_id}")
                return web.json_response({
                    "success": False,
                    "error": "任务不存在",
                    "message": f"任务ID {task_id} 不存在"
                }, status=200)
            
            # 仅允许删除 avatar 或 action 类型
            if task.train_type not in ["avatar", "action"]:
                logger.warning(f"任务 {task_id} 的类型 {task.train_type} 不允许删除")
                return web.json_response({
                    "success": False,
                    "error": "类型不允许删除",
                    "message": f"仅支持删除 avatar 或 action 类型的任务"
                }, status=400)
            
            # 正在训练中的任务不允许删除
            if task.status in ["processing", "training"]:
                logger.warning(f"任务 {task_id} 正在{task.status}，不可删除")
                return web.json_response({
                    "success": False,
                    "error": "任务正在进行中",
                    "message": f"任务状态为 {task.status}，不可删除，请先取消"
                }, status=400)
            
            # 在删除任务记录前，尝试清理相关文件/目录
            try:
                # 删除下载的视频临时文件（如果存在）
                if hasattr(task, 'video_path') and task.video_path and Path(task.video_path).exists():
                    try:
                        logger.info(f"删除任务临时视频文件: {task.video_path}")
                        Path(task.video_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"删除临时视频文件失败: {e}")

                # 删除训练输出目录
                if task.train_type == "avatar":
                    target_dir = Path("data/avatars") / task.video_name
                    if target_dir.exists() and target_dir.is_dir():
                        logger.info(f"删除头像训练目录: {target_dir}")
                        shutil.rmtree(target_dir, ignore_errors=True)
                elif task.train_type == "action":
                    target_dir = Path("data/customvideo") / task.video_name
                    if target_dir.exists() and target_dir.is_dir():
                        logger.info(f"删除动作训练目录: {target_dir}")
                        shutil.rmtree(target_dir, ignore_errors=True)
                    # 同步更新 custom_config.json，移除对应配置
                    try:
                        custom_config_path = Path("data/custom_config.json")
                        if custom_config_path.exists():
                            with open(custom_config_path, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                            original_len = len(config_data) if isinstance(config_data, list) else 0
                            if isinstance(config_data, list):
                                filtered = [item for item in config_data if item.get('audiotype') != task.video_name]
                                if len(filtered) != original_len:
                                    with open(custom_config_path, 'w', encoding='utf-8') as f:
                                        json.dump(filtered, f, ensure_ascii=False, indent=2)
                                    logger.info(f"已从 custom_config.json 移除 audiotype={task.video_name} 的配置")
                    except Exception as e:
                        logger.warning(f"更新 custom_config.json 失败: {e}")
            except Exception as e:
                logger.warning(f"清理任务相关文件时发生错误: {e}")

            # 检查并自动删除关联的 action 任务（task_id + "_action"）
            try:
                # 仅当原始 task_id 未以 "_action" 结尾时才尝试关联删除
                if task_id and not task_id.endswith("_action"):
                    related_task_id = f"{task_id}_action"
                    # 直接从内存字典获取，避免封装方法未命中
                    related_task = self.training_tasks.get(related_task_id)
                    # 兼容对象或字典结构
                    related_type = getattr(related_task, 'train_type', None) if related_task is not None else None
                    if related_type is None and isinstance(related_task, dict):
                        related_type = related_task.get('train_type')
                    if related_task and related_type == 'action':
                        logger.info(f"检测到关联的 action 任务，将尝试自动删除: {related_task_id}")
                        # 正在训练中的任务不允许删除
                        related_status = getattr(related_task, 'status', None)
                        if related_status is None and isinstance(related_task, dict):
                            related_status = related_task.get('status')
                        if related_status in ["processing", "training"]:
                            logger.warning(f"关联任务 {related_task_id} 正在{related_status}，跳过自动删除")
                        else:
                            # 关联任务文件清理（与 action 类型一致）
                            try:
                                related_video_path = getattr(related_task, 'video_path', None)
                                if related_video_path is None and isinstance(related_task, dict):
                                    related_video_path = related_task.get('video_path')
                                if related_video_path and Path(related_video_path).exists():
                                    try:
                                        logger.info(f"删除关联任务临时视频文件: {related_video_path}")
                                        Path(related_video_path).unlink(missing_ok=True)
                                    except Exception as e:
                                        logger.warning(f"删除关联任务临时视频文件失败: {e}")

                                related_video_name = getattr(related_task, 'video_name', None)
                                if related_video_name is None and isinstance(related_task, dict):
                                    related_video_name = related_task.get('video_name')
                                target_dir = Path("data/customvideo") / str(related_video_name)
                                if target_dir.exists() and target_dir.is_dir():
                                    logger.info(f"删除关联动作训练目录: {target_dir}")
                                    shutil.rmtree(target_dir, ignore_errors=True)

                                # 同步更新 custom_config.json，移除对应配置
                                try:
                                    custom_config_path = Path("data/custom_config.json")
                                    if custom_config_path.exists():
                                        with open(custom_config_path, 'r', encoding='utf-8') as f:
                                            config_data = json.load(f)
                                        original_len = len(config_data) if isinstance(config_data, list) else 0
                                        if isinstance(config_data, list):
                                            filtered = [item for item in config_data if item.get('audiotype') != related_video_name]
                                            if len(filtered) != original_len:
                                                with open(custom_config_path, 'w', encoding='utf-8') as f:
                                                    json.dump(filtered, f, ensure_ascii=False, indent=2)
                                                logger.info(f"已从 custom_config.json 移除 audiotype={related_video_name} 的配置（关联任务）")
                                except Exception as e:
                                    logger.warning(f"更新 custom_config.json（关联任务）失败: {e}")
                            except Exception as e:
                                logger.warning(f"清理关联 action 任务相关文件时发生错误: {e}")

                            # 删除关联任务（线程安全）
                            with self.training_tasks_lock:
                                if related_task_id in self.training_tasks:
                                    del self.training_tasks[related_task_id]
                                    logger.info(f"关联任务 {related_task_id} 删除成功")
                            # 持久化保存
                            self.save_training_tasks_to_file()
                    else:
                        # Fallback：遍历查找可能的关联 action 任务
                        try:
                            logger.info(f"未直接命中 {related_task_id}，开始遍历查找关联 action 任务")
                            candidates = []
                            with self.training_tasks_lock:
                                for tk, tv in self.training_tasks.items():
                                    # 统一取字段
                                    tv_type = getattr(tv, 'train_type', None)
                                    if tv_type is None and isinstance(tv, dict):
                                        tv_type = tv.get('train_type')
                                    if tv_type != 'action':
                                        continue
                                    tv_tid = getattr(tv, 'task_id', None)
                                    if tv_tid is None and isinstance(tv, dict):
                                        tv_tid = tv.get('task_id')
                                    tv_vn = getattr(tv, 'video_name', None)
                                    if tv_vn is None and isinstance(tv, dict):
                                        tv_vn = tv.get('video_name')
                                    # 匹配条件：task_id 完全匹配，或 video_name 含主 id 且以 _action 结尾
                                    if tk == related_task_id or tv_tid == related_task_id or (str(task_id) in str(tv_vn) and str(tv_vn).endswith('_action')):
                                        candidates.append((tk, tv))

                            if not candidates:
                                logger.info(f"未找到任何可删除的关联 action 任务，related_task_id={related_task_id}")
                            else:
                                logger.info(f"找到 {len(candidates)} 个关联 action 任务候选，将尝试删除")
                                for del_id, del_task in candidates:
                                    del_status = getattr(del_task, 'status', None)
                                    if del_status is None and isinstance(del_task, dict):
                                        del_status = del_task.get('status')
                                    if del_status in ["processing", "training"]:
                                        logger.warning(f"关联任务 {del_id} 正在{del_status}，跳过")
                                        continue

                                    # 清理文件
                                    try:
                                        del_video_path = getattr(del_task, 'video_path', None)
                                        if del_video_path is None and isinstance(del_task, dict):
                                            del_video_path = del_task.get('video_path')
                                        if del_video_path and Path(del_video_path).exists():
                                            try:
                                                logger.info(f"删除关联任务临时视频文件: {del_video_path}")
                                                Path(del_video_path).unlink(missing_ok=True)
                                            except Exception as e:
                                                logger.warning(f"删除关联任务临时视频文件失败: {e}")

                                        del_video_name = getattr(del_task, 'video_name', None)
                                        if del_video_name is None and isinstance(del_task, dict):
                                            del_video_name = del_task.get('video_name')
                                        tdir = Path("data/customvideo") / str(del_video_name)
                                        if tdir.exists() and tdir.is_dir():
                                            logger.info(f"删除关联动作训练目录: {tdir}")
                                            shutil.rmtree(tdir, ignore_errors=True)

                                        # 更新 custom_config.json
                                        try:
                                            custom_config_path = Path("data/custom_config.json")
                                            if custom_config_path.exists():
                                                with open(custom_config_path, 'r', encoding='utf-8') as f:
                                                    config_data = json.load(f)
                                                orig_len = len(config_data) if isinstance(config_data, list) else 0
                                                if isinstance(config_data, list):
                                                    filtered = [item for item in config_data if item.get('audiotype') != del_video_name]
                                                    if len(filtered) != orig_len:
                                                        with open(custom_config_path, 'w', encoding='utf-8') as f:
                                                            json.dump(filtered, f, ensure_ascii=False, indent=2)
                                                        logger.info(f"已从 custom_config.json 移除 audiotype={del_video_name} 的配置（关联任务）")
                                        except Exception as e:
                                            logger.warning(f"更新 custom_config.json（关联任务）失败: {e}")
                                    except Exception as e:
                                        logger.warning(f"清理关联 action 任务相关文件时发生错误: {e}")

                                    # 删除任务并持久化
                                    with self.training_tasks_lock:
                                        if del_id in self.training_tasks:
                                            del self.training_tasks[del_id]
                                            logger.info(f"关联任务 {del_id} 删除成功")
                                            self.save_training_tasks_to_file()
                        except Exception as e:
                            logger.warning(f"遍历删除关联 action 任务时发生错误: {e}")
            except Exception as e:
                logger.warning(f"自动删除关联 action 任务时发生错误: {e}")

            # 删除任务（线程安全）
            logger.info(f"开始删除任务: {task_id}")
            with self.training_tasks_lock:
                if task_id in self.training_tasks:
                    del self.training_tasks[task_id]
            
            # 持久化保存
            self.save_training_tasks_to_file()
            logger.info(f"任务 {task_id} 删除成功")
            
            return web.json_response({
                "success": True,
                "message": "任务已删除",
                "task_id": task_id
            })
        except Exception as e:
            logger.error(f"删除训练任务失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "删除训练任务时发生错误"
            }, status=200)