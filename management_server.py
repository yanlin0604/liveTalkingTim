import asyncio
import os
import sys
import time
import json
import re
from typing import Dict, Optional
from logger import logger

class BarrageManager:
    """管理 barrage_websocket.py 多实例子进程。

    设计：
    - 按 sessionid 管理多个子进程，每个子进程使用独立配置文件：config/barrage_config.<sessionid>.json
    - /barrage/start 可重复调用以启动不同 sessionid 的实例
    - /barrage/stop 支持停止单个或全部
    """
    def __init__(self):
        self.base_config_path: str = 'config/barrage_config.json'
        # 其余三类外部配置基础路径（作为模板源；不存在则使用空配置生成会话隔离文件）
        self.base_speech_path: str = 'config/speech_config.json'
        self.base_sensitive_path: str = 'config/sensitive_config.json'
        self.base_schedule_path: str = 'config/schedule_config.json'
        # sessionid -> { process, config_path, started_at, args }
        self.sessions: Dict[str, dict] = {}

    def _refresh_sessions(self):
        """清理已退出的子进程记录，并删除临时配置文件。"""
        for sid, info in list(self.sessions.items()):
            p = info.get('process')
            if p is None or p.returncode is not None:
                # 先清理临时文件再删除记录
                try:
                    self._cleanup_session_files(info)
                finally:
                    self.sessions.pop(sid, None)

    def is_running(self, sessionid: Optional[str] = None) -> bool:
        self._refresh_sessions()
        if sessionid is None:
            return any(info.get('process') and info['process'].returncode is None for info in self.sessions.values())
        info = self.sessions.get(str(sessionid))
        return bool(info and info.get('process') and info['process'].returncode is None)

    def _make_session_config(self, sessionid: str) -> str:
        """兼容旧方法：仅生成 barrage_config.<sessionid>.json 并写入 default_sessionid。"""
        cfg_path = self.base_config_path
        # 若基础配置不存在，使用最小默认结构
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {
                "human_url": "http://127.0.0.1:8010/human",
                "default_sessionid": 0,
                "types": {},
                "rules": {"global": {"min_len": 1, "max_len": 120, "rate_limit_per_min": 60}}
            }
        cfg['default_sessionid'] = sessionid
        # 独立配置文件路径
        dir_name = os.path.dirname(cfg_path) or 'config'
        os.makedirs(dir_name, exist_ok=True)
        out_path = os.path.join(dir_name, f"barrage_config.{sessionid}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return out_path

    def _copy_json_file(self, src_path: str, out_path: str, default_obj: Optional[dict] = None):
        """复制JSON文件；若源不存在则写入 default_obj（或空对象）。"""
        os.makedirs(os.path.dirname(out_path) or 'config', exist_ok=True)
        try:
            if src_path and os.path.exists(src_path):
                with open(src_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = default_obj if isinstance(default_obj, dict) else {}
        except Exception:
            data = default_obj if isinstance(default_obj, dict) else {}
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"写入会话配置失败: {out_path} | {e}")

    def _prepare_session_configs(self, sessionid: str) -> Dict[str, str]:
        """为会话生成四类独立配置文件，返回路径映射。"""
        dir_name = os.path.dirname(self.base_config_path) or 'config'
        os.makedirs(dir_name, exist_ok=True)

        paths = {
            'barrage': os.path.join(dir_name, f"barrage_config.{sessionid}.json"),
            'speech': os.path.join(dir_name, f"speech_config.{sessionid}.json"),
            'sensitive': os.path.join(dir_name, f"sensitive_config.{sessionid}.json"),
            'schedule': os.path.join(dir_name, f"schedule_config.{sessionid}.json"),
        }
        # 生成 barrage（带 default_sessionid）
        barrage_path = self._make_session_config(sessionid)
        paths['barrage'] = barrage_path
        # 生成其余三个（复制或空对象）
        self._copy_json_file(self.base_speech_path, paths['speech'], default_obj={
            "templates": {"greeting": ["大家好，欢迎来到直播间！"], "fallback": ["这条消息我先跳过，继续看下一条～"]},
            "reply_rules": [],
            "gift_thanks": []
        })
        self._copy_json_file(self.base_sensitive_path, paths['sensitive'], default_obj={
            "blacklist": [], "strategy": "mask", "mask_char": "*", "whitelist": []
        })
        self._copy_json_file(self.base_schedule_path, paths['schedule'], default_obj={
            "auto_broadcast": {"enabled": False, "interval_sec": 180, "messages": []},
            "idle_fill": {"enabled": False, "idle_threshold_sec": 60, "messages": []}
        })
        return paths

    def _maybe_delete_file(self, path: Optional[str]):
        if not path:
            return
        try:
            if os.path.isfile(path):
                os.remove(path)
                logger.info(f"已删除会话临时配置: {path}")
        except Exception as e:
            logger.warning(f"删除临时配置失败: {path} | {e}")

    def _cleanup_session_files(self, info: Optional[dict]):
        """删除为会话生成的临时配置文件。"""
        if not info:
            return
        # 优先使用新的 config_paths
        paths = info.get('config_paths') if isinstance(info, dict) else None
        if isinstance(paths, dict):
            for p in paths.values():
                self._maybe_delete_file(p)
        # 为兼容旧结构，尝试删除单一 barrage_config.<sessionid>.json
        cfg_path = info.get('config_path') if isinstance(info, dict) else None
        if cfg_path:
            base = os.path.basename(cfg_path)
            if base.startswith('barrage_config.') and base.endswith('.json'):
                self._maybe_delete_file(cfg_path)

    def cleanup_orphan_configs(self, exclude: Optional[set] = None) -> Dict:
        """清理孤儿配置文件：删除 config/ 目录下不属于当前运行 sessions 的会话隔离配置文件。

        匹配的文件名格式：
        - barrage_config.<sid>.json
        - speech_config.<sid>.json
        - sensitive_config.<sid>.json
        - schedule_config.<sid>.json
        """
        exclude = exclude or set()
        dir_name = os.path.dirname(self.base_config_path) or 'config'
        os.makedirs(dir_name, exist_ok=True)
        pat = re.compile(r'^(barrage_config|speech_config|sensitive_config|schedule_config)\.(.+)\.json$')
        deleted = []
        try:
            for fn in os.listdir(dir_name):
                m = pat.match(fn)
                if not m:
                    continue
                sid = m.group(2)
                if sid in exclude:
                    continue
                # 当前管理器中未记录该会话，视为孤儿
                if sid not in self.sessions:
                    full_path = os.path.join(dir_name, fn)
                    try:
                        if os.path.isfile(full_path):
                            os.remove(full_path)
                            deleted.append(full_path)
                    except Exception as e:
                        logger.warning(f"清理孤儿配置失败: {full_path} | {e}")
        except Exception as e:
            logger.warning(f"扫描孤儿配置失败: {e}")
        if deleted:
            logger.info(f"清理孤儿配置完成，删除 {len(deleted)} 个文件")
        return {"deleted": deleted}

    async def _monitor_process(self, sessionid: str):
        """后台监控子进程，退出后自动清理会话配置并移除记录。"""
        info = self.sessions.get(sessionid)
        if not info:
            return
        p: asyncio.subprocess.Process = info.get('process')
        if not p:
            return
        try:
            await p.wait()
        except Exception:
            pass
        # 进程退出后进行清理（若 stop() 已处理，以下操作将幂等）
        try:
            self._cleanup_session_files(self.sessions.get(sessionid))
        finally:
            self.sessions.pop(sessionid, None)
            logger.info(f"barrage 子进程已退出并完成清理 | sessionid={sessionid}")

    async def start(self, sessionid: str) -> Dict:
        sessionid = str(sessionid)
        self._refresh_sessions()
        if self.is_running(sessionid):
            return {"ok": False, "error": f"session {sessionid} already running"}
        # 生成独立配置（四类）
        try:
            paths = self._prepare_session_configs(sessionid)
            cfg_path = paths['barrage']
        except Exception as e:
            return {"ok": False, "error": f"生成配置失败: {e}"}
        # 启动子进程
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'barrage_websocket.py')
        cmd = [
            sys.executable, script_path,
            '--config', paths['barrage'],
            '--speech', paths['speech'],
            '--sensitive', paths['sensitive'],
            '--schedule', paths['schedule']
        ]
        logger.info(f"启动 barrage_websocket | sessionid={sessionid} | {' '.join(cmd)}")
        try:
            process = await asyncio.create_subprocess_exec(*cmd)
        except Exception as e:
            return {"ok": False, "error": f"启动子进程失败: {e}"}
        self.sessions[sessionid] = {
            'process': process,
            'config_path': cfg_path,
            'config_paths': paths,
            'started_at': time.time(),
            'args': {'sessionid': sessionid}
        }
        # 启动后台监控，确保异常退出时也能清理临时配置
        try:
            asyncio.create_task(self._monitor_process(sessionid))
        except Exception:
            logger.warning(f"创建子进程监控任务失败 | sessionid={sessionid}")
        return {"ok": True, "pid": process.pid, "sessionid": sessionid, "config_path": cfg_path}

    async def status(self, sessionid: Optional[str] = None) -> Dict:
        self._refresh_sessions()
        if sessionid is not None:
            sid = str(sessionid)
            info = self.sessions.get(sid)
            if not info:
                return {'running': False, 'sessionid': sid, 'message': 'not found'}
            p = info.get('process')
            return {
                'running': bool(p and p.returncode is None),
                'pid': getattr(p, 'pid', None) if p else None,
                'started_at': info.get('started_at'),
                'args': info.get('args', {}),
                'config_path': info.get('config_path'),
                'config_paths': info.get('config_paths', {}),
                'sessionid': sid
            }
        # 汇总全部
        data = {}
        for sid, info in self.sessions.items():
            p = info.get('process')
            data[sid] = {
                'running': bool(p and p.returncode is None),
                'pid': getattr(p, 'pid', None) if p else None,
                'started_at': info.get('started_at'),
                'args': info.get('args', {}),
                'config_path': info.get('config_path'),
                'config_paths': info.get('config_paths', {})
            }
        running_total = sum(1 for v in data.values() if v['running'])
        running = running_total > 0
        # 兼容旧字段：始终提供一个 primary（取第一个运行中的实例）
        primary_pid = None
        primary_started_at = None
        primary_args = {}
        primary_sessionid = None
        if running:
            for sid, v in data.items():
                if v['running']:
                    primary_pid = v['pid']
                    primary_started_at = v['started_at']
                    primary_args = v['args']
                    primary_sessionid = sid
                    break
        return {
            'running': running,
            'pid': primary_pid,
            'started_at': primary_started_at,
            'args': primary_args,
            'sessionid': primary_sessionid,
            'running_total': running_total,
            'sessions': data
        }

    async def stop(self, sessionid: Optional[str] = None) -> Dict:
        self._refresh_sessions()
        async def _stop_one(sid: str) -> Dict:
            info = self.sessions.get(sid)
            if not info:
                return {'ok': True, 'sessionid': sid, 'message': 'not running'}
            p: asyncio.subprocess.Process = info.get('process')
            if p and p.returncode is None:
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(p.wait(), timeout=5)
                except Exception:
                    logger.warning(f"子进程未按时退出，尝试kill | sessionid={sid}")
                    try:
                        p.kill()
                    except Exception:
                        pass
            ret = p.returncode if p else 0
            # 清理会话临时配置
            try:
                self._cleanup_session_files(info)
            finally:
                self.sessions.pop(sid, None)
            return {'ok': True, 'sessionid': sid, 'returncode': ret}
        if sessionid is not None:
            return await _stop_one(str(sessionid))
        # 停止全部
        results = []
        for sid in list(self.sessions.keys()):
            results.append(await _stop_one(sid))
        return {'ok': True, 'stopped': results}

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
        # 读取请求体并处理sessionid前导零问题
        body_text = await request.text()
        data = json.loads(body_text)
        
        # 处理sessionid前导零丢失问题
        raw_sessionid = data.get('sessionid')
        if isinstance(raw_sessionid, int) and raw_sessionid != 0:
            import re
            # 在原始JSON文本中查找sessionid的原始值
            original_match = re.search(r'"sessionid"\s*:\s*"?(0\d+)"?', body_text)
            if original_match:
                sessionid = original_match.group(1)
                logger.info(f"🔧 检测到sessionid前导零丢失，已恢复为: {sessionid}")
            else:
                sessionid = str(raw_sessionid)
        else:
            sessionid = str(raw_sessionid)
            # 在启动前清理孤儿配置（不影响当前运行中的会话配置）
        try:
            barrage_manager.cleanup_orphan_configs(exclude=set(barrage_manager.sessions.keys()))
        except Exception as e:
            logger.warning(f"清理孤儿配置时出错: {e}")

        result = await barrage_manager.start(sessionid)
        if not result.get('ok'):
            return api_err(result.get('error', 'start failed'))
        return api_ok(result)

    async def status_barrage(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        result = await barrage_manager.status(sid)
        return api_ok(result)

    async def stop_barrage(request: web.Request):
        # 支持两种模式：
        # 1) 停止单个：POST /barrage/stop {"sessionid":"xxx"} 或 /barrage/stop?sessionid=xxx
        # 2) 停止全部：POST /barrage/stop（无 sessionid）
        sid = None
        stop_all = False
        # 读取 body
        try:
            body = await request.json()
            if isinstance(body, dict):
                sid = body.get('sessionid')
                allv = body.get('all')
                if isinstance(allv, bool):
                    stop_all = allv
                elif isinstance(allv, str):
                    stop_all = allv.lower() in ('1', 'true', 'yes')
                elif isinstance(allv, int):
                    stop_all = (allv == 1)
        except Exception:
            pass
        # 读取 query
        if sid is None:
            sid = request.rel_url.query.get('sessionid')
        q_all = request.rel_url.query.get('all')
        if q_all is not None and isinstance(q_all, str):
            if q_all.lower() in ('1', 'true', 'yes'):
                stop_all = True

        # 优先单个停止
        if sid:
            result = await barrage_manager.stop(sid)
            if isinstance(result, dict) and not result.get('ok', True) and 'error' in result:
                return api_err(result.get('error', 'stop failed'))
            return api_ok(result)

        # 未提供 sessionid
        if stop_all:
            result = await barrage_manager.stop(None)
            if isinstance(result, dict) and not result.get('ok', True) and 'error' in result:
                return api_err(result.get('error', 'stop failed'))
            return api_ok(result)

        # 向后兼容：若只有一个实例在运行，则默认停止该实例；若多个实例运行则报错提示
        status_all = await barrage_manager.status()
        running_total = int(status_all.get('running_total', 0) or 0)
        if running_total == 0:
            return api_ok({"ok": True, "message": "not running"})
        if running_total == 1:
            sessions = status_all.get('sessions', {}) or {}
            only_sid = None
            for k, v in sessions.items():
                if v.get('running'):
                    only_sid = k
                    break
            result = await barrage_manager.stop(only_sid)
            if isinstance(result, dict) and not result.get('ok', True) and 'error' in result:
                return api_err(result.get('error', 'stop failed'))
            return api_ok(result)

        # 多实例且未指定 sessionid 且未显式 all=true
        return api_err("multiple sessions running; specify sessionid or set all=true to stop all")

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

    def resolve_session_file(base_file: Path, sid: Optional[str]) -> Path:
        """根据 sessionid 返回隔离文件路径；无 sid 时返回基础路径。"""
        if not sid:
            return base_file
        stem = base_file.stem
        return base_file.with_name(f"{stem}.{sid}{base_file.suffix}")

    # 话术配置
    async def get_speech(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        p = resolve_session_file(speech_file, sid)
        ok, res = await read_json_file(p)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_speech(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        p = resolve_session_file(speech_file, sid)
        ok, err = await write_json_file(p, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_speech(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        default = {
            "templates": {"greeting": ["大家好，我是{avatar}，欢迎来到直播间！"], "fallback": ["这条消息我先跳过，继续看下一条～"]},
            "reply_rules": [{"match": "上链接", "template": "商品链接已置顶，{username}可以点击查看哦～"}],
            "gift_thanks": [{"min_price": 1, "template": "感谢{username}送出的{giftName}x{giftCount}！"}]
        }
        p = resolve_session_file(speech_file, sid)
        ok, err = await write_json_file(p, default)
        return api_ok({"reset": ok}) if ok else api_err(f"重置失败: {err}")

    # 敏感词配置
    async def get_sensitive(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        p = resolve_session_file(sensitive_file, sid)
        ok, res = await read_json_file(p)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_sensitive(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        p = resolve_session_file(sensitive_file, sid)
        ok, err = await write_json_file(p, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_sensitive(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        default = {"blacklist": ["违禁词1", "违禁词2"], "strategy": "mask", "mask_char": "*"}
        p = resolve_session_file(sensitive_file, sid)
        ok, err = await write_json_file(p, default)
        return api_ok({"reset": ok}) if ok else api_err(f"重置失败: {err}")

    # 定时任务配置
    async def get_schedule_cfg(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        p = resolve_session_file(schedule_file, sid)
        ok, res = await read_json_file(p)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_schedule_cfg(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        p = resolve_session_file(schedule_file, sid)
        ok, err = await write_json_file(p, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_schedule_cfg(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        default = {
            "auto_broadcast": {"enabled": True, "interval_sec": 180, "messages": ["关注不迷路，带你看好物～"]},
            "idle_fill": {"enabled": True, "idle_threshold_sec": 60, "messages": ["有想看的可以在弹幕里告诉我哦～"]}
        }
        p = resolve_session_file(schedule_file, sid)
        ok, err = await write_json_file(p, default)
        return api_ok({"reset": ok}) if ok else api_err(f"重置失败: {err}")


    # 弹幕主配置（barrage_config.json）
    async def get_barrage_cfg(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        p = resolve_session_file(barrage_cfg_file, sid)
        ok, res = await read_json_file(p)
        return api_ok(res) if ok else api_err(f"读取失败: {res}")

    async def put_barrage_cfg(request: web.Request):
        sid = request.rel_url.query.get('sessionid')
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        p = resolve_session_file(barrage_cfg_file, sid)
        ok, err = await write_json_file(p, data)
        return api_ok({"saved": ok}) if ok else api_err(f"保存失败: {err}")

    async def reset_barrage_cfg(request: web.Request):
        # 与当前文件中的结构保持一致的默认值
        sid = request.rel_url.query.get('sessionid')
        default = {
            "_comment": "弹幕转发配置文件 - 控制各种消息类型的处理方式",
            "human_url": "http://127.0.0.1:8010/human",
            "_human_url_comment": "AI服务接口地址，用于发送处理后的消息",
            "default_sessionid": sid if sid is not None else 1,
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
        p = resolve_session_file(barrage_cfg_file, sid)
        ok, err = await write_json_file(p, default)
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
    
    # ===== avatar_id 与 REF_FILE 映射配置 CRUD =====
    avatar_ref_file = base_conf / 'avatar_ref_config.json'
    # 配置文件位于 config/avatar_ref_config.json，结构如下：
    # {
    #   "_comment": "avatar_id 与 REF_FILE 一对一映射配置文件",
    #   "map": { "<avatar_id>": "<ref_file(字符串)>" }
    # }

    async def list_avatar_refs(request: web.Request):
        """
        列出全部 avatar_id -> REF_FILE 映射
        - 方法: GET /avatar_ref
        - 响应: {"code":0,"msg":"ok","data":{"map":{ "<avatar_id>":"<ref_file>" }}}
        """
        ok, res = await read_json_file(avatar_ref_file)
        if not ok:
            # 文件不存在或读取失败时，返回空集合
            return api_ok({"map": {}})
        return api_ok({"map": (res or {}).get('map', {})})

    async def create_avatar_ref(request: web.Request):
        """
        新增映射
        - 方法: POST /avatar_ref
        - 请求体: {"avatar_id":"string","ref_file":"string"}
        - 冲突: 若 avatar_id 已存在，返回错误
        - 响应: {"code":0,"msg":"ok","data":{"created":true,"avatar_id":"...","ref_file":"..."}}
        """
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        if not isinstance(data, dict):
            return api_err("请求体必须为对象(JSON)")
        avatar_id = data.get('avatar_id')
        ref_file = data.get('ref_file')
        if not avatar_id or not isinstance(avatar_id, str):
            return api_err("缺少或非法的 avatar_id，应为非空字符串")
        if not ref_file or not isinstance(ref_file, str):
            return api_err("缺少或非法的 ref_file，应为非空字符串")

        ok, res = await read_json_file(avatar_ref_file)
        if not ok:
            res = {"_comment": "avatar_id 与 REF_FILE 一对一映射配置文件", "map": {}}
        mapping = res.setdefault('map', {})
        if avatar_id in mapping:
            return api_err(f"avatar_id={avatar_id} 已存在，如需修改请使用 PUT /avatar_ref/{{avatar_id}}")
        mapping[avatar_id] = ref_file
        ok, err = await write_json_file(avatar_ref_file, res)
        return api_ok({"created": ok, "avatar_id": avatar_id, "ref_file": ref_file}) if ok else api_err(f"保存失败: {err}")

    async def get_avatar_ref(request: web.Request):
        """
        查询单个映射
        - 方法: GET /avatar_ref/{avatar_id}
        - 路径参数: avatar_id
        - 响应: {"code":0,"msg":"ok","data":{"avatar_id":"...","ref_file":"..."}}
        """
        avatar_id = request.match_info.get('avatar_id')
        ok, res = await read_json_file(avatar_ref_file)
        if not ok:
            return api_err(f"读取失败: {res}")
        mapping = (res or {}).get('map', {})
        if avatar_id not in mapping:
            return api_err(f"未找到 avatar_id={avatar_id} 的映射")
        return api_ok({"avatar_id": avatar_id, "ref_file": mapping.get(avatar_id)})

    async def put_avatar_ref(request: web.Request):
        """
        更新单个映射
        - 方法: PUT /avatar_ref/{avatar_id}
        - 路径参数: avatar_id
        - 请求体: {"ref_file":"string"}
        - 响应: {"code":0,"msg":"ok","data":{"saved":true,"avatar_id":"...","ref_file":"..."}}
        """
        avatar_id = request.match_info.get('avatar_id')
        try:
            data = await request.json()
        except Exception as e:
            return api_err(f"JSON解析失败: {e}")
        ref_file = data.get('ref_file') if isinstance(data, dict) else None
        if not ref_file or not isinstance(ref_file, str):
            return api_err("缺少或非法的 ref_file，应为非空字符串")
        # 读取现有数据（若不存在则初始化默认结构）
        ok, res = await read_json_file(avatar_ref_file)
        if not ok:
            res = {"_comment": "avatar_id 与 REF_FILE 一对一映射配置文件", "map": {}}
        mapping = res.setdefault('map', {})
        mapping[avatar_id] = ref_file
        ok, err = await write_json_file(avatar_ref_file, res)
        return api_ok({"saved": ok, "avatar_id": avatar_id, "ref_file": ref_file}) if ok else api_err(f"保存失败: {err}")

    async def delete_avatar_ref(request: web.Request):
        """
        删除单个映射
        - 方法: DELETE /avatar_ref/{avatar_id}
        - 路径参数: avatar_id
        - 响应: {"code":0,"msg":"ok","data":{"deleted":true,"avatar_id":"..."}}
        """
        avatar_id = request.match_info.get('avatar_id')
        ok, res = await read_json_file(avatar_ref_file)
        if not ok:
            return api_err(f"读取失败: {res}")
        mapping = (res or {}).get('map', {})
        if avatar_id in mapping:
            del mapping[avatar_id]
            # 持久化
            ok2, err = await write_json_file(avatar_ref_file, res)
            return api_ok({"deleted": ok2, "avatar_id": avatar_id}) if ok2 else api_err(f"删除失败: {err}")
        else:
            return api_err(f"未找到 avatar_id={avatar_id} 的映射")

    # 路由注册（avatar_ref）
    # - GET    /avatar_ref                 列表
    # - POST   /avatar_ref                 新增
    # - GET    /avatar_ref/{avatar_id}     查询单个
    # - PUT    /avatar_ref/{avatar_id}     更新单个
    # - DELETE /avatar_ref/{avatar_id}     删除单个
    app.router.add_get('/avatar_ref', list_avatar_refs)
    app.router.add_post('/avatar_ref', create_avatar_ref)
    app.router.add_get('/avatar_ref/{avatar_id}', get_avatar_ref)
    app.router.add_put('/avatar_ref/{avatar_id}', put_avatar_ref)
    app.router.add_delete('/avatar_ref/{avatar_id}', delete_avatar_ref)
    
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