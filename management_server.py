import asyncio
import os
import sys
import time
from typing import Dict, Optional
from logger import logger

class BarrageManager:
    """管理 barrage_websocket.py 子进程，仅通过进程句柄控制。

    /barrage/start 仅需 sessionid：在启动前写入 config/barrage_config.json 的 default_sessionid。
    不依赖子进程管理HTTP端口。
    """
    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.started_at: float = 0.0
        self.args: Dict[str, str] = {}
        self.config_path: str = 'config/barrage_config.json'
        self._original_default_sessionid = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self, sessionid: int) -> Dict:
        if self.is_running():
            return {"ok": False, "error": "Barrage service already running"}

        # 1) 写入配置的 default_sessionid
        try:
            import json
            cfg_path = self.config_path
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            self._original_default_sessionid = cfg.get('default_sessionid')
            cfg['default_sessionid'] = int(sessionid)
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            logger.info(f"已将 {cfg_path} 的 default_sessionid 设置为 {sessionid}")
        except Exception as e:
            return {"ok": False, "error": f"更新配置失败: {e}"}

        # 2) 启动子进程（不传额外CLI参数，沿用脚本默认）
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'barrage_websocket.py')
        cmd = [sys.executable, script_path]
        logger.info(f"启动 barrage_websocket: {' '.join(cmd)}")
        try:
            self.process = await asyncio.create_subprocess_exec(*cmd)
        except Exception as e:
            # 失败时尝试回滚配置
            try:
                import json
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                cfg['default_sessionid'] = self._original_default_sessionid
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return {"ok": False, "error": f"启动子进程失败: {e}"}

        self.started_at = time.time()
        self.args = {'sessionid': int(sessionid)}
        return {"ok": True, "pid": self.process.pid}

    async def status(self) -> Dict:
        info = {
            'running': self.is_running(),
            'pid': getattr(self.process, 'pid', None),
            'started_at': self.started_at,
            'args': self.args,
        }
        return info

    async def stop(self) -> Dict:
        if not self.is_running():
            return {"ok": True, "message": "not running"}
        # 直接结束子进程
        try:
            self.process.terminate()
        except Exception:
            pass
        # 等待退出
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except Exception:
            logger.warning("子进程未按时退出，尝试kill")
            try:
                self.process.kill()
            except Exception:
                pass
        ret = self.process.returncode
        self.process = None
        # 尝试恢复配置
        try:
            import json
            with open(self.config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if self._original_default_sessionid is not None:
                cfg['default_sessionid'] = self._original_default_sessionid
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                logger.info("已恢复 default_sessionid 到原值")
        except Exception as e:
            logger.warning(f"恢复配置失败: {e}")
        return {"ok": True, "returncode": ret}

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
import aiohttp
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
    barrage_manager = BarrageManager()
    
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
    
    # 弹幕转发服务管理接口（基于 barrage_websocket.py）
    def api_ok(data: dict | None = None):
        from aiohttp import web
        return web.json_response({"code": 0, "msg": "ok", "data": data or {}})

    def api_err(message: str, code: int = 1):
        from aiohttp import web
        return web.json_response({"code": code, "msg": message, "data": {}})

    async def start_barrage(request: web.Request):
        data = await request.json()
        sessionid = int(data.get('sessionid'))
        result = await barrage_manager.start(sessionid)
        if not result.get('ok'):
            return api_err(result.get('error', 'start failed'))
        return api_ok(result)

    async def status_barrage(request: web.Request):
        result = await barrage_manager.status()
        return api_ok(result)

    async def stop_barrage(request: web.Request):
        result = await barrage_manager.stop()
        if not result.get('ok'):
            return api_err(result.get('error', 'stop failed'))
        return api_ok(result)

    app.router.add_post('/barrage/start', start_barrage)
    app.router.add_get('/barrage/status', status_barrage)
    app.router.add_post('/barrage/stop', stop_barrage)
    
    # ===== 四类配置 CRUD 接口 =====
    from pathlib import Path
    base_conf = Path('config')
    speech_file = base_conf / 'speech_config.json'
    sensitive_file = base_conf / 'sensitive_config.json'
    schedule_file = base_conf / 'schedule_config.json'
    barrage_cfg_file = base_conf / 'barrage_config.json'

    async def read_json_file(p: Path):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return True, json.load(f)
        except Exception as e:
            return False, str(e)

    async def write_json_file(p: Path, data: dict):
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True, None
        except Exception as e:
            return False, str(e)

    # 话术配置
    async def get_speech(request: web.Request):
        ok, res = await read_json_file(speech_file)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_speech(request: web.Request):
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        ok, err = await write_json_file(speech_file, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_speech(request: web.Request):
        default = {
            "templates": {"greeting": ["大家好，我是{avatar}，欢迎来到直播间！"], "fallback": ["这条消息我先跳过，继续看下一条～"]},
            "reply_rules": [{"match": "上链接", "template": "商品链接已置顶，{username}可以点击查看哦～"}],
            "gift_thanks": [{"min_price": 1, "template": "感谢{username}送出的{giftName}x{giftCount}！"}]
        }
        ok, err = await write_json_file(speech_file, default)
        return api_ok({"reset": ok}) if ok else api_err(f"重置失败: {err}")

    # 敏感词配置
    async def get_sensitive(request: web.Request):
        ok, res = await read_json_file(sensitive_file)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_sensitive(request: web.Request):
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        ok, err = await write_json_file(sensitive_file, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_sensitive(request: web.Request):
        default = {"blacklist": ["违禁词1", "违禁词2"], "strategy": "mask", "mask_char": "*"}
        ok, err = await write_json_file(sensitive_file, default)
        return api_ok({"reset": ok}) if ok else api_err(f"重置失败: {err}")

    # 定时任务配置
    async def get_schedule_cfg(request: web.Request):
        ok, res = await read_json_file(schedule_file)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_schedule_cfg(request: web.Request):
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        ok, err = await write_json_file(schedule_file, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_schedule_cfg(request: web.Request):
        default = {
            "auto_broadcast": {"enabled": True, "interval_sec": 180, "messages": ["关注不迷路，带你看好物～"]},
            "idle_fill": {"enabled": True, "idle_threshold_sec": 60, "messages": ["有想看的可以在弹幕里告诉我哦～"]}
        }
        ok, err = await write_json_file(schedule_file, default)
        return api_ok({"reset": ok}) if ok else api_err(f"重置失败: {err}")


    # 弹幕主配置（barrage_config.json）
    async def get_barrage_cfg(request: web.Request):
        ok, res = await read_json_file(barrage_cfg_file)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_barrage_cfg(request: web.Request):
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        ok, err = await write_json_file(barrage_cfg_file, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_barrage_cfg(request: web.Request):
        # 与当前文件中的结构保持一致的默认值
        default = {
            "_comment": "弹幕转发配置文件 - 控制各种消息类型的处理方式",
            "human_url": "http://127.0.0.1:8010/human",
            "_human_url_comment": "AI服务接口地址，用于发送处理后的消息",
            "default_sessionid": 1,
            "_default_sessionid_comment": "默认会话ID，用于区分不同的对话会话",
            "reply_control": {
                "_comment": "弹幕回复控制配置",
                "enabled": True,
                "reply_probability": 0.8,
                "_reply_probability_comment": "弹幕回复概率 (0.0-1.0)，0.3表示30%的弹幕会被回复",
                "max_replies_per_minute": 10,
                "_max_replies_per_minute_comment": "每分钟最大回复数量，防止过度回复"
            },
            "rules": {
                "global": {
                    "min_len": 1,
                    "max_len": 120,
                    "rate_limit_per_min": 60
                }
            },
            "types": {
                "DANMU": {
                    "_comment": "弹幕消息配置",
                    "enabled": True,
                    "_enabled_comment": "是否启用此类型消息的处理",
                    "action": "echo",
                    "_action_comment": "处理动作：chat=对话回复, echo=直接复述",
                    "interrupt": False,
                    "_interrupt_comment": "是否打断当前播放内容",
                    "min_length": 1,
                    "_min_length_comment": "弹幕最小长度，小于此长度的弹幕会被忽略",
                    "max_length": 120,
                    "_max_length_comment": "弹幕最大长度，超过会被截断"
                },
                "GIFT": {
                    "_comment": "礼物消息配置",
                    "enabled": True,
                    "action": "echo",
                    "template": "感谢{username}送出的{giftName}x{giftCount}！",
                    "interrupt": False,
                    "min_gift_price": 0,
                    "_min_gift_price_comment": "最小礼物价值，低于此价值的礼物不会触发感谢"
                },
                "SUPER_CHAT": {
                    "_comment": "醒目留言/SC配置",
                    "enabled": True,
                    "action": "echo",
                    "template": "感谢醒目留言，{username}：{content}",
                    "interrupt": True,
                    "_interrupt_comment": "SC通常会打断当前内容",
                    "min_price": 0,
                    "_min_price_comment": "最小SC价格"
                },
                "ENTER_ROOM": {
                    "_comment": "进入房间消息配置",
                    "enabled": False,
                    "_enabled_comment": "通常关闭，避免过多欢迎消息",
                    "action": "echo",
                    "template": "欢迎{username}进入直播间",
                    "interrupt": False
                },
                "LIKE": {
                    "_comment": "点赞消息配置",
                    "enabled": False,
                    "action": "echo",
                    "template": "{username} 点赞了直播",
                    "interrupt": False
                },
                "LIVE_STATUS_CHANGE": {
                    "_comment": "直播状态变更配置",
                    "enabled": False,
                    "action": "echo",
                    "template": "直播状态变更：{status}",
                    "interrupt": True
                },
                "ROOM_STATS": {
                    "_comment": "房间统计信息配置",
                    "enabled": False,
                    "action": "echo",
                    "template": "当前在线{online}，热度{hot}，点赞{likes}",
                    "interrupt": False
                },
                "SOCIAL": {
                    "_comment": "社交动作配置（关注、分享等）",
                    "enabled": False,
                    "action": "echo",
                    "template": "{username}{action}",
                    "interrupt": False
                }
            },
            "sessions": {
                "_comment": "为不同消息类型指定特定的会话ID",
                "_example": "DANMU: 681008, GIFT: 681009"
            }
        }
        ok, err = await write_json_file(barrage_cfg_file, default)
        return api_ok({"reset": ok}) if ok else api_err(f"重置失败: {err}")

    # 路由注册
    app.router.add_get('/speech_config', get_speech)
    app.router.add_put('/speech_config', put_speech)
    app.router.add_post('/speech_config/reset', reset_speech)

    app.router.add_get('/sensitive_config', get_sensitive)
    app.router.add_put('/sensitive_config', put_sensitive)
    app.router.add_post('/sensitive_config/reset', reset_sensitive)

    app.router.add_get('/schedule_config', get_schedule_cfg)
    app.router.add_put('/schedule_config', put_schedule_cfg)
    app.router.add_post('/schedule_config/reset', reset_schedule_cfg)


    app.router.add_get('/barrage_config', get_barrage_cfg)
    app.router.add_put('/barrage_config', put_barrage_cfg)
    app.router.add_post('/barrage_config/reset', reset_barrage_cfg)
    
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