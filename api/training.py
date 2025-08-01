"""
训练相关API接口
"""
import json
import asyncio
import time
import uuid
import urllib.parse
import tempfile
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
    
    def safe_get_training_task(self, task_id: str):
        """安全获取训练任务"""
        with self.training_tasks_lock:
            return self.training_tasks.get(task_id)
    
    def safe_set_training_task(self, task_id: str, task: TrainingTask):
        """安全设置训练任务"""
        with self.training_tasks_lock:
            self.training_tasks[task_id] = task
            # 保存到文件
            self.save_training_tasks_to_file()
    
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
            task.status = status
            task.progress = progress
            task.message = message
            if error:
                task.error = error
            if status in ["completed", "failed"]:
                task.end_time = time.time()
            logger.info(f"任务 {task_id} 进度更新: {status} - {progress}% - {message}")
            # 保存到文件
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
        # 验证token
        if self.auth_api:
            auth_result = await self._verify_auth(request)
            if auth_result:
                return auth_result
        
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.json_response({
                    "success": False,
                    "error": "请求体为空",
                    "message": "请提供有效的JSON数据"
                }, status=400)
            
            data = await request.json()
            video_name = data.get('video_name')
            video_url = data.get('video_url')  # 新增：支持视频URL
            train_type = data.get('type', 'avatar')  # avatar 或 action
            force_retrain = data.get('force_retrain', False)  # 是否强制重新训练
            
            # 参数验证
            if not video_name and not video_url:
                return web.json_response({
                    "success": False,
                    "error": "缺少必要参数",
                    "message": "请提供 video_name 或 video_url 参数"
                }, status=400)
            
            if train_type not in ['avatar', 'action']:
                return web.json_response({
                    "success": False,
                    "error": "无效的训练类型",
                    "message": "训练类型必须是 'avatar' 或 'action'"
                }, status=400)
            
            # 生成任务ID
            task_id = self.generate_task_id()
            
            # 创建训练任务
            task = TrainingTask(
                task_id=task_id,
                video_name=video_name or "unknown",
                video_url=video_url,
                train_type=train_type,
                force_retrain=force_retrain
            )
            
            # 保存任务
            self.safe_set_training_task(task_id, task)
            
            logger.info(f"创建训练任务: {task_id}, 视频名称={video_name or 'unknown'}, URL={video_url or '本地文件'}, 类型={train_type}")
            
            # 启动异步训练任务
            asyncio.create_task(self.execute_training_task(task_id))
            
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
            }, status=400)
        except Exception as e:
            logger.error(f"创建训练任务失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e),
                "message": "创建训练任务时发生错误"
            }, status=500)
    
    async def execute_training_task(self, task_id: str):
        """异步执行训练任务"""
        task = self.safe_get_training_task(task_id)
        if not task:
            logger.error(f"任务 {task_id} 不存在")
            return
        
        try:
            self.update_task_progress(task_id, "pending", 0, "开始处理训练任务")
            
            video_path = None
            is_url_video = False
            
            # 处理URL视频
            if task.video_url:
                try:
                    self.update_task_progress(task_id, "processing", 10, "正在处理视频文件")
                    
                    # 验证URL格式
                    parsed_url = urllib.parse.urlparse(task.video_url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        raise Exception("无效的URL格式")
                    
                    # 创建临时目录
                    temp_dir = Path(tempfile.gettempdir()) / "Unimed_videos"
                    temp_dir.mkdir(exist_ok=True)
                    
                    # 如果video_name为空或为"unknown"，则从URL中提取文件名
                    if not task.video_name or task.video_name == "unknown":
                        url_path = parsed_url.path
                        video_name = Path(url_path).stem
                        if not video_name:
                            video_name = f"video_{int(time.time())}"
                        task.video_name = video_name
                    # 否则保留传入的video_name作为记录的文件名
                    
                    # 确定文件扩展名
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
                    
                    # 构建本地文件路径
                    local_filename = f"{task.video_name}{ext}"
                    video_path = temp_dir / local_filename
                    task.video_path = video_path
                    task.is_url_video = True
                    is_url_video = True
                    
                    # 下载视频文件
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(task.video_url) as response:
                            if response.status != 200:
                                raise Exception(f"无法下载视频，HTTP状态码: {response.status}")
                            
                            # 检查文件大小
                            content_length = response.headers.get('content-length')
                            if content_length:
                                file_size = int(content_length)
                                if file_size > 500 * 1024 * 1024:  # 500MB限制
                                    raise Exception(f"视频文件过大 ({file_size / 1024 / 1024:.1f}MB)，最大支持500MB")
                            
                            # 写入文件
                            total_size = 0
                            with open(video_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                                    total_size += len(chunk)
                                    # 更新下载进度
                                    if content_length:
                                        download_progress = min(30, int(10 + (total_size / int(content_length)) * 20))
                                        self.update_task_progress(task_id, "processing", download_progress, f"正在获取视频文件 ({total_size / 1024 / 1024:.1f}MB)")
                    
                    self.update_task_progress(task_id, "training", 30, "视频文件获取完成，开始训练")
                    
                except Exception as e:
                    self.update_task_progress(task_id, "failed", 0, f"获取视频文件失败: {str(e)}", str(e))
                    return
            
            # 处理本地视频文件
            else:
                self.update_task_progress(task_id, "training", 10, "正在查找本地视频文件")
                
                # 查找视频文件
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                
                # 根据训练类型确定扫描目录
                if task.train_type == 'avatar':
                    scan_dirs = ['videos', 'data/videos', 'uploads']
                else:  # action
                    scan_dirs = ['action_videos', 'data/action_videos', 'uploads']
                
                for scan_dir in scan_dirs:
                    scan_path = Path(scan_dir)
                    if scan_path.exists():
                        for ext in video_extensions:
                            potential_path = scan_path / f"{task.video_name}{ext}"
                            if potential_path.exists():
                                video_path = potential_path
                                break
                            # 也检查大写扩展名
                            potential_path = scan_path / f"{task.video_name}{ext.upper()}"
                            if potential_path.exists():
                                video_path = potential_path
                                break
                        if video_path:
                            break
                
                if not video_path:
                    self.update_task_progress(task_id, "failed", 0, f"在目录 {', '.join(scan_dirs)} 中未找到视频文件: {task.video_name}", "视频文件未找到")
                    return
                
                task.video_path = video_path
                self.update_task_progress(task_id, "training", 20, "找到视频文件，开始训练")
            
            # 执行训练
            self.update_task_progress(task_id, "training", 40, "正在初始化训练环境")
            
            if task.train_type == 'avatar':
                from video_scanner import VideoScanner
                scanner = VideoScanner(
                    scan_directory=str(video_path.parent),
                    avatar_base_dir="data/avatars",
                    config_file="video_scanner_config.json" if Path("video_scanner_config.json").exists() else None
                )
                
                # 检查是否已经训练过（仅对本地文件）
                if not is_url_video and not task.force_retrain and scanner.is_trained(video_path):
                    self.update_task_progress(task_id, "failed", 0, f"视频 {task.video_name} 已经训练过，如需重新训练请设置 force_retrain=true", "视频已训练过")
                    return
                
                self.update_task_progress(task_id, "training", 60, "正在训练头像模型")
                success = scanner.train_video(video_path)
                
            else:  # action
                from action_scanner import ActionScanner
                scanner = ActionScanner(
                    scan_directory=str(video_path.parent),
                    action_base_dir="data/customvideo",
                    config_file="action_scanner_config.json" if Path("action_scanner_config.json").exists() else None
                )
                
                # 检查是否已经处理过（仅对本地文件）
                if not is_url_video and not task.force_retrain and scanner.is_processed(video_path):
                    self.update_task_progress(task_id, "failed", 0, f"视频 {task.video_name} 已经处理过，如需重新处理请设置 force_retrain=true", "视频已处理过")
                    return
                
                self.update_task_progress(task_id, "training", 60, "正在处理动作编排")
                success = scanner.process_video(video_path)
            
            if success:
                self.update_task_progress(task_id, "training", 90, "训练完成，正在清理临时文件")
                
                # 清理临时文件（如果是URL视频）
                if is_url_video and video_path and video_path.exists():
                    try:
                        video_path.unlink()
                        logger.info(f"已清理临时文件: {video_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
                
                self.update_task_progress(task_id, "completed", 100, f"视频 {task.video_name} {task.train_type}训练成功")
            else:
                # 清理临时文件（如果训练失败）
                if is_url_video and video_path and video_path.exists():
                    try:
                        video_path.unlink()
                        logger.info(f"训练失败，已清理临时文件: {video_path}")
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
                
                self.update_task_progress(task_id, "failed", 0, f"视频 {task.video_name} {task.train_type}训练失败，请检查日志", "训练失败")
                
        except Exception as e:
            # 清理临时文件（如果发生异常）
            if hasattr(task, 'video_path') and task.video_path and task.video_path.exists():
                try:
                    task.video_path.unlink()
                    logger.info(f"发生异常，已清理临时文件: {task.video_path}")
                except Exception as cleanup_error:
                    logger.warning(f"清理临时文件失败: {cleanup_error}")
            
            self.update_task_progress(task_id, "failed", 0, f"训练过程中发生错误: {str(e)}", str(e))
    
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
        # 验证token
        if self.auth_api:
            auth_result = await self._verify_auth(request)
            if auth_result:
                return auth_result
        
        try:
            task_id = request.match_info.get('task_id')
            if not task_id:
                return web.json_response({
                    "success": False,
                    "error": "缺少任务ID",
                    "message": "请提供任务ID"
                }, status=400)
            
            task = self.safe_get_training_task(task_id)
            if not task:
                return web.json_response({
                    "success": False,
                    "error": "任务不存在",
                    "message": f"任务ID {task_id} 不存在"
                }, status=404)
            
            # 计算运行时间
            duration = time.time() - task.start_time
            if task.end_time:
                duration = task.end_time - task.start_time
            
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
            }, status=500)
    
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
        # 验证token
        if self.auth_api:
            auth_result = await self._verify_auth(request)
            if auth_result:
                return auth_result
        
        try:
            with self.training_tasks_lock:
                tasks = []
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
            }, status=500)
    
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
        # 验证token
        if self.auth_api:
            auth_result = await self._verify_auth(request)
            if auth_result:
                return auth_result
        
        try:
            task_id = request.match_info.get('task_id')
            if not task_id:
                return web.json_response({
                    "success": False,
                    "error": "缺少任务ID",
                    "message": "请提供任务ID"
                }, status=400)
            
            task = self.safe_get_training_task(task_id)
            if not task:
                return web.json_response({
                    "success": False,
                    "error": "任务不存在",
                    "message": f"任务ID {task_id} 不存在"
                }, status=404)
            
            if task.status in ["completed", "failed", "cancelled"]:
                return web.json_response({
                    "success": False,
                    "error": "任务无法取消",
                    "message": f"任务状态为 {task.status}，无法取消"
                }, status=400)
            
            # 更新任务状态为取消
            self.update_task_progress(task_id, "cancelled", task.progress, "任务已取消")
            
            # 清理临时文件
            if hasattr(task, 'video_path') and task.video_path and task.video_path.exists():
                try:
                    task.video_path.unlink()
                    logger.info(f"取消任务，已清理临时文件: {task.video_path}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
            
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
            }, status=500) 