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

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response
from dynamic_config import dynamic_config, start_config_monitoring, get_config, set_config
from config_callbacks import setup_config_callbacks

# 导入重构后的API模块
from api.webrtc import WebRTCAPI
from api.chat import ChatAPI
from api.config import ConfigAPI
from api.avatars import AvatarsAPI
from api.training import TrainingAPI, TrainingTask
from api.auth import AuthAPI

# 添加当前目录到Python路径，确保能找到swagger模块
import sys
import os
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

import argparse
import random
import shutil
import asyncio
import torch
import os
import glob
import urllib.parse
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from logger import logger
import gc


app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
opt = None
model = None
avatar = None

# 线程安全锁
import threading
nerfreals_lock = threading.Lock()

# 训练任务管理
training_tasks: Dict[str, Dict] = {}  # task_id -> task_info
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

# 定期清理过期任务（每小时执行一次）
def schedule_cleanup():
    """定期清理过期任务"""
    while True:
        try:
            time.sleep(3600)  # 每小时执行一次
            cleanup_old_completed_tasks()
        except Exception as e:
            logger.error(f"定期清理任务失败: {e}")

# 启动清理线程
cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
cleanup_thread.start()

def safe_get_nerfreal(sessionid):
    """安全获取nerfreal对象"""
    with nerfreals_lock:
        return nerfreals.get(sessionid)

def safe_set_nerfreal(sessionid, nerfreal):
    """安全设置nerfreal对象"""
    with nerfreals_lock:
        nerfreals[sessionid] = nerfreal

def safe_del_nerfreal(sessionid):
    """安全删除nerfreal对象"""
    with nerfreals_lock:
        if sessionid in nerfreals:
            del nerfreals[sessionid]
            return True
        return False

def safe_check_nerfreal(sessionid):
    """安全检查nerfreal对象是否存在"""
    with nerfreals_lock:
        return sessionid in nerfreals

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
        

#####webrtc###############################
pcs = set()

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int)->BaseReal:
    opt.sessionid=sessionid
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
        # 为新会话预热模型，确保推理稳定性
        logger.info(f"正在为会话 {sessionid} 预热模型...")
        from lipreal import warm_up
        model_res = 384 if '384' in str(model) else 256
        warm_up(opt.batch_size, model, model_res)
        logger.info(f"会话 {sessionid} 模型预热完成")
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
    # elif opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(opt,model,avatar)
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
    return nerfreal

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')

async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreal)
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch size (1-32, lower=less latency, higher=more efficient)")
    parser.add_argument('--auto_batch_size', action='store_true', help="automatically adjust batch size based on transport type")

    # 颜色匹配参数
    parser.add_argument('--enable_color_matching', action='store_true', default=True, help="enable color matching between generated lips and original face")
    parser.add_argument('--color_matching_strength', type=float, default=0.6, help="color matching strength (0.0-1.0, higher=stronger correction)")

    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")
    parser.add_argument('--use_custom_silent', action='store_true', default=True, help="use custom silent action")
    parser.add_argument('--custom_silent_audiotype', type=str, default='', help="custom silent action audiotype")

    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-XiaoxiaoNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    # LLM配置参数
    parser.add_argument('--llm_provider', type=str, default='dashscope', choices=['dashscope', 'ollama'],
                       help="LLM provider: dashscope (阿里云) or ollama (本地)")
    parser.add_argument('--llm_model', type=str, default='qwen-plus',
                       help="LLM model name (e.g., qwen-plus for dashscope, llama3.2 for ollama)")
    parser.add_argument('--llm_system_prompt', type=str, default='You are a helpful assistant.',
                       help="System prompt for LLM")
    parser.add_argument('--ollama_host', type=str, default='http://localhost:11434',
                       help="Ollama server host (only used when llm_provider=ollama)")

    parser.add_argument('--model', type=str, default='musetalk', help='model type: musetalk, wav2lip, ultralight')
    parser.add_argument('--model_path', type=str, default='', help='path to model file (auto-detect if empty)')
    parser.add_argument('--wav2lip_model_size', type=str, default='384', choices=['256', '384'], help='wav2lip model resolution')

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")
    
    # 动态配置文件参数
    parser.add_argument('--config_file', type=str, default='config.json', help="dynamic config file path")

    opt = parser.parse_args()
    
    # 初始化动态配置系统
    dynamic_config.config_file = opt.config_file
    
    # 从配置文件加载参数（如果存在）
    if os.path.exists(opt.config_file):
        logger.info(f"从配置文件加载参数: {opt.config_file}")
        config_data = dynamic_config.get_all()
        
        # 用配置文件的值覆盖命令行默认值（但不覆盖用户明确指定的参数）
        for key, value in config_data.items():
            if hasattr(opt, key) and value is not None:
                # 只有当命令行参数是默认值时才覆盖
                parser_default = parser.get_default(key)
                current_value = getattr(opt, key)
                if current_value == parser_default:
                    setattr(opt, key, value)
                    logger.info(f"从配置文件应用参数: {key} = {value}")
    
    # 启动配置文件监控
    start_config_monitoring(interval=2.0)
    
    # 设置配置变化回调
    setup_config_callbacks(opt, nerfreals)
    
    logger.info("动态配置系统已启动")

    # 智能配置调整
    if opt.auto_batch_size:
        if opt.transport == 'webrtc':
            # WebRTC实时场景：优先低延迟
            opt.batch_size = min(opt.batch_size, 8)
            logger.info(f"WebRTC模式：自动调整batch_size为 {opt.batch_size} (低延迟优化)")
        elif opt.transport == 'virtualcam':
            # 虚拟摄像头：平衡延迟和效率
            opt.batch_size = min(opt.batch_size, 12)
            logger.info(f"VirtualCam模式：自动调整batch_size为 {opt.batch_size} (平衡优化)")
        elif opt.transport == 'rtcpush':
            # 推流场景：可以接受稍高延迟
            opt.batch_size = min(opt.batch_size, 16)
            logger.info(f"RTC推流模式：自动调整batch_size为 {opt.batch_size} (效率优化)")

    # 配置验证和建议
    if opt.transport == 'webrtc' and opt.batch_size > 12:
        logger.warning(f"WebRTC模式下batch_size={opt.batch_size}可能导致延迟过高，建议使用 --batch_size 8 或 --auto_batch_size")

    if opt.fps > 50 and opt.batch_size > 16:
        logger.warning(f"高帧率({opt.fps})配合大batch_size({opt.batch_size})可能导致性能问题")

    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':       
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt) 
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)

        # 智能确定模型路径
        if opt.model_path:
            model_path = opt.model_path
        else:
            # 自动检测策略：
            # 1. 优先根据 wav2lip_model_size 参数
            # 2. 其次根据 avatar_id 中的数字
            # 3. 最后使用默认值

            model_size = opt.wav2lip_model_size
            if not model_size:
                if '384' in opt.avatar_id:
                    model_size = '384'
                elif '256' in opt.avatar_id:
                    model_size = '256'
                else:
                    model_size = '384'  # 默认384

            model_path = f"./models/wav2lip{model_size}.pth"

            # 检查文件是否存在，如果不存在尝试其他路径
            import os
            if not os.path.exists(model_path):
                alternative_paths = [
                    f"./models/wav2lip.pth",
                    f"./models/wav2lip_{model_size}.pth",
                    f"./wav2lip/models/wav2lip{model_size}.pth"
                ]
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        break
                else:
                    logger.warning(f"Model file not found: {model_path}")
                    logger.warning("Please ensure the model file exists or specify --model_path")

        logger.info(f"Loading wav2lip model from: {model_path}")
        model = load_model(model_path)
        avatar = load_avatar(opt.avatar_id)

        # 根据模型大小调整warm_up参数
        model_res = 384 if '384' in model_path else 256
        warm_up(opt.batch_size, model, model_res)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreal = build_nerfreal(0)
        safe_set_nerfreal(0, nerfreal)
        rendthrd = Thread(target=nerfreal.render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    # 程序启动时加载训练任务数据
    load_training_tasks_from_file()
    
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    
    # 初始化API模块
    webrtc_api = WebRTCAPI(build_nerfreal, nerfreals, nerfreals_lock, pcs)
    chat_api = ChatAPI(nerfreals, nerfreals_lock)
    config_api = ConfigAPI()
    avatars_api = AvatarsAPI()
    auth_api = AuthAPI()
    training_api = TrainingAPI(training_tasks, training_tasks_lock, auth_api)
    
    # WebRTC相关接口
    appasync.router.add_post("/offer", webrtc_api.offer)  # WebRTC连接建立，处理SDP offer
    appasync.router.add_get("/webrtc/status", webrtc_api.get_connection_status)  # 获取WebRTC连接状态

    # 文本交互接口
    appasync.router.add_post("/human", chat_api.human)  # 发送文本消息给数字人（支持echo/chat模式，可选打断）

    # 音频交互接口
    appasync.router.add_post("/humanaudio", chat_api.humanaudio)  # 发送音频数据给数字人进行语音识别
    appasync.router.add_post("/set_audiotype", chat_api.set_audiotype)  # 设置音频类型和参数
    appasync.router.add_post("/set_custom_silent", chat_api.set_custom_silent)  # 设置静音时是否使用自定义动作
    appasync.router.add_post("/record", chat_api.record)  # 录制功能控制接口

    # 对话控制接口
    appasync.router.add_post("/interrupt_talk", chat_api.interrupt_talk)  # 打断数字人当前说话
    appasync.router.add_post("/is_speaking", chat_api.is_speaking)  # 检查数字人是否正在说话
    
    # 头像管理接口
    appasync.router.add_get("/get_avatars", avatars_api.get_avatars)  # 获取可用头像列表
    appasync.router.add_get("/get_actions", avatars_api.get_actions)  # 获取可用动作列表
    appasync.router.add_get("/get_config_for_frontend", config_api.get_config_for_frontend)  # 获取前端配置
    
    # 配置管理接口
    appasync.router.add_get("/get_config", config_api.get_config_api)  # 获取当前配置
    appasync.router.add_post("/update_config", config_api.update_config_api)  # 更新配置参数
    appasync.router.add_post("/save_config", config_api.save_config_api)  # 保存配置到文件
    appasync.router.add_post("/reset_config", config_api.reset_config_api)  # 重置配置
    
    # 鉴权接口
    appasync.router.add_post("/auth/token", auth_api.get_token_api)  # 获取访问token
    appasync.router.add_post("/auth/revoke", auth_api.revoke_token_api)  # 撤销访问token
    appasync.router.add_post("/auth/verify", auth_api.verify_token_api)  # 验证访问token
    
    # 视频训练接口（需要认证）
    appasync.router.add_post("/train_video", training_api.train_video_api)  # 根据视频名称单独训练头像或动作
    appasync.router.add_get("/training/progress/{task_id}", training_api.get_training_progress)  # 获取训练任务进度
    appasync.router.add_get("/training/tasks", training_api.list_training_tasks)  # 获取所有训练任务列表
    appasync.router.add_post("/training/cancel/{task_id}", training_api.cancel_training_task)  # 取消训练任务
    
    # 添加Swagger文档
    create_swagger_docs(appasync)
    
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    logger.info('API文档地址: http://<serverip>:'+str(opt.listenport)+'/swagger')
    
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    
