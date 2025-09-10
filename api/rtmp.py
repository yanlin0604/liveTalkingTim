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

from aiohttp import web
import json
import uuid
from threading import Thread, Event
from logger import logger
import time

class RTMPAPI:
    """RTMP推流管理API"""
    
    def __init__(self, build_nerfreal, nerfreals, nerfreals_lock, opt):
        self.build_nerfreal = build_nerfreal
        self.nerfreals = nerfreals
        self.nerfreals_lock = nerfreals_lock
        self.opt = opt
        self.rtmp_sessions = {}  # sessionid: {'thread': Thread, 'quit_event': Event, 'status': str, 'push_url': str}
        
    def safe_get_nerfreal(self, sessionid):
        """安全获取nerfreal对象"""
        with self.nerfreals_lock:
            return self.nerfreals.get(sessionid)

    def safe_set_nerfreal(self, sessionid, nerfreal):
        """安全设置nerfreal对象"""
        with self.nerfreals_lock:
            self.nerfreals[sessionid] = nerfreal

    def safe_del_nerfreal(self, sessionid):
        """安全删除nerfreal对象"""
        with self.nerfreals_lock:
            if sessionid in self.nerfreals:
                del self.nerfreals[sessionid]
                return True
            return False

    async def start_rtmp_stream(self, request):
        """启动RTMP推流
        
        请求参数:
        - push_url: RTMP推流地址 (可选，默认使用配置中的push_url)
        - sessionid: 会话ID (可选，自动生成UUID)
        
        返回:
        - sessionid: 推流会话ID
        - push_url: 推流地址
        - status: 推流状态
        """
        try:
            data = await request.json() if request.content_type == 'application/json' else {}
        except:
            data = {}
            
        # 获取推流地址，优先使用请求参数，否则使用配置默认值
        push_url = data.get('push_url', self.opt.push_url)
        if not push_url or not push_url.startswith('rtmp://'):
            return web.json_response({
                'code': 400,
                'message': '无效的RTMP推流地址，必须以rtmp://开头',
                'data': None
            })
            
        # 获取或生成sessionid
        sessionid = data.get('sessionid')
        if not sessionid:
            sessionid = str(uuid.uuid4())
            
        # 检查会话是否已存在
        if sessionid in self.rtmp_sessions:
            session_info = self.rtmp_sessions[sessionid]
            if session_info['status'] == 'running':
                return web.json_response({
                    'code': 400,
                    'message': f'推流会话 {sessionid} 已在运行中',
                    'data': {
                        'sessionid': sessionid,
                        'push_url': session_info['push_url'],
                        'status': session_info['status']
                    }
                })
                
        logger.info(f"开始启动RTMP推流，sessionid={sessionid}, push_url={push_url}")
        
        try:
            # 构建nerfreal对象
            nerfreal = self.build_nerfreal(sessionid)
            if nerfreal is None:
                return web.json_response({
                    'code': 500,
                    'message': '构建数字人对象失败',
                    'data': None
                })
                
            # 设置推流地址
            nerfreal.opt.push_url = push_url
            nerfreal.opt.transport = 'rtmp'
            
            # 保存nerfreal对象
            self.safe_set_nerfreal(sessionid, nerfreal)
            
            # 创建退出事件
            thread_quit = Event()
            
            # 启动推流线程
            render_thread = Thread(target=nerfreal.render, args=(thread_quit,))
            render_thread.daemon = True
            render_thread.start()
            
            # 记录会话信息
            self.rtmp_sessions[sessionid] = {
                'thread': render_thread,
                'quit_event': thread_quit,
                'status': 'running',
                'push_url': push_url,
                'start_time': time.time()
            }
            
            logger.info(f"RTMP推流启动成功，sessionid={sessionid}, push_url={push_url}")
            
            return web.json_response({
                'code': 200,
                'message': 'RTMP推流启动成功',
                'data': {
                    'sessionid': sessionid,
                    'push_url': push_url,
                    'status': 'running'
                }
            })
            
        except Exception as e:
            logger.error(f"启动RTMP推流失败: {e}")
            logger.exception("启动RTMP推流详细异常信息")
            
            # 清理资源
            self.safe_del_nerfreal(sessionid)
            if sessionid in self.rtmp_sessions:
                del self.rtmp_sessions[sessionid]
                
            return web.json_response({
                'code': 500,
                'message': f'启动RTMP推流失败: {str(e)}',
                'data': None
            })

    async def stop_rtmp_stream(self, request):
        """停止RTMP推流
        
        请求参数:
        - sessionid: 推流会话ID
        
        返回:
        - sessionid: 推流会话ID
        - status: 推流状态
        """
        try:
            data = await request.json() if request.content_type == 'application/json' else {}
        except:
            data = {}
            
        sessionid = data.get('sessionid')
        if not sessionid:
            return web.json_response({
                'code': 400,
                'message': '缺少sessionid参数',
                'data': None
            })
            
        # 检查会话是否存在
        if sessionid not in self.rtmp_sessions:
            return web.json_response({
                'code': 404,
                'message': f'推流会话 {sessionid} 不存在',
                'data': None
            })
            
        session_info = self.rtmp_sessions[sessionid]
        
        try:
            logger.info(f"开始停止RTMP推流，sessionid={sessionid}")
            
            # 设置退出事件
            session_info['quit_event'].set()
            
            # 等待线程结束（最多等待5秒）
            session_info['thread'].join(timeout=5.0)
            
            # 清理nerfreal对象
            self.safe_del_nerfreal(sessionid)
            
            # 更新会话状态
            session_info['status'] = 'stopped'
            session_info['stop_time'] = time.time()
            
            logger.info(f"RTMP推流停止成功，sessionid={sessionid}")
            
            return web.json_response({
                'code': 200,
                'message': 'RTMP推流停止成功',
                'data': {
                    'sessionid': sessionid,
                    'status': 'stopped'
                }
            })
            
        except Exception as e:
            logger.error(f"停止RTMP推流失败: {e}")
            logger.exception("停止RTMP推流详细异常信息")
            
            return web.json_response({
                'code': 500,
                'message': f'停止RTMP推流失败: {str(e)}',
                'data': None
            })

    async def get_rtmp_status(self, request):
        """获取RTMP推流状态
        
        请求参数:
        - sessionid: 推流会话ID (可选，不提供则返回所有会话状态)
        
        返回:
        - sessions: 推流会话状态列表
        """
        try:
            sessionid = request.query.get('sessionid')
            
            if sessionid:
                # 获取指定会话状态
                if sessionid not in self.rtmp_sessions:
                    return web.json_response({
                        'code': 404,
                        'message': f'推流会话 {sessionid} 不存在',
                        'data': None
                    })
                    
                session_info = self.rtmp_sessions[sessionid]
                
                # 检查线程是否还在运行
                if session_info['status'] == 'running' and not session_info['thread'].is_alive():
                    session_info['status'] = 'stopped'
                    session_info['stop_time'] = time.time()
                    
                session_data = {
                    'sessionid': sessionid,
                    'push_url': session_info['push_url'],
                    'status': session_info['status'],
                    'start_time': session_info.get('start_time'),
                    'stop_time': session_info.get('stop_time'),
                    'thread_alive': session_info['thread'].is_alive() if session_info['thread'] else False
                }
                
                return web.json_response({
                    'code': 200,
                    'message': '获取推流状态成功',
                    'data': session_data
                })
            else:
                # 获取所有会话状态
                sessions = []
                for sid, session_info in self.rtmp_sessions.items():
                    # 检查线程是否还在运行
                    if session_info['status'] == 'running' and not session_info['thread'].is_alive():
                        session_info['status'] = 'stopped'
                        session_info['stop_time'] = time.time()
                        
                    sessions.append({
                        'sessionid': sid,
                        'push_url': session_info['push_url'],
                        'status': session_info['status'],
                        'start_time': session_info.get('start_time'),
                        'stop_time': session_info.get('stop_time'),
                        'thread_alive': session_info['thread'].is_alive() if session_info['thread'] else False
                    })
                    
                return web.json_response({
                    'code': 200,
                    'message': '获取推流状态成功',
                    'data': {
                        'sessions': sessions,
                        'total': len(sessions)
                    }
                })
                
        except Exception as e:
            logger.error(f"获取RTMP推流状态失败: {e}")
            logger.exception("获取RTMP推流状态详细异常信息")
            
            return web.json_response({
                'code': 500,
                'message': f'获取推流状态失败: {str(e)}',
                'data': None
            })

    async def list_rtmp_sessions(self, request):
        """列出所有RTMP推流会话
        
        返回:
        - sessions: 推流会话列表
        """
        try:
            sessions = []
            for sessionid, session_info in self.rtmp_sessions.items():
                # 检查线程是否还在运行
                if session_info['status'] == 'running' and not session_info['thread'].is_alive():
                    session_info['status'] = 'stopped'
                    session_info['stop_time'] = time.time()
                    
                sessions.append({
                    'sessionid': sessionid,
                    'push_url': session_info['push_url'],
                    'status': session_info['status'],
                    'start_time': session_info.get('start_time'),
                    'stop_time': session_info.get('stop_time'),
                    'duration': time.time() - session_info['start_time'] if session_info['status'] == 'running' else 
                               (session_info.get('stop_time', time.time()) - session_info['start_time'])
                })
                
            return web.json_response({
                'code': 200,
                'message': '获取推流会话列表成功',
                'data': {
                    'sessions': sessions,
                    'total': len(sessions),
                    'running': len([s for s in sessions if s['status'] == 'running']),
                    'stopped': len([s for s in sessions if s['status'] == 'stopped'])
                }
            })
            
        except Exception as e:
            logger.error(f"获取RTMP推流会话列表失败: {e}")
            logger.exception("获取RTMP推流会话列表详细异常信息")
            
            return web.json_response({
                'code': 500,
                'message': f'获取推流会话列表失败: {str(e)}',
                'data': None
            })

    async def set_rtmp_quality(self, request):
        """设置RTMP推流清晰度
        
        请求参数:
        - sessionid: 推流会话ID
        - quality: 清晰度级别 ('ultra', 'high', 'medium', 'low')
        
        返回:
        - sessionid: 推流会话ID
        - quality_info: 清晰度设置结果
        """
        try:
            data = await request.json() if request.content_type == 'application/json' else {}
        except:
            data = {}
            
        sessionid = data.get('sessionid')
        quality_level = data.get('quality')
        
        if not sessionid:
            return web.json_response({
                'code': 400,
                'message': '缺少sessionid参数',
                'data': None
            })
            
        if not quality_level:
            return web.json_response({
                'code': 400,
                'message': '缺少quality参数',
                'data': None
            })
            
        # 获取nerfreal对象
        nerfreal = self.safe_get_nerfreal(sessionid)
        if not nerfreal:
            return web.json_response({
                'code': 404,
                'message': f'推流会话 {sessionid} 不存在或未启动',
                'data': None
            })
            
        try:
            # 调用BaseReal的清晰度设置方法
            result = nerfreal.set_rtmp_quality(quality_level)
            
            if result['success']:
                logger.info(f"RTMP清晰度设置成功，sessionid={sessionid}, quality={quality_level}")
                return web.json_response({
                    'code': 200,
                    'message': result['message'],
                    'data': {
                        'sessionid': sessionid,
                        'quality_info': result
                    }
                })
            else:
                return web.json_response({
                    'code': 400,
                    'message': result['message'],
                    'data': None
                })
                
        except Exception as e:
            logger.error(f"设置RTMP清晰度失败: {e}")
            return web.json_response({
                'code': 500,
                'message': f'设置清晰度失败: {str(e)}',
                'data': None
            })

    async def get_rtmp_quality(self, request):
        """获取RTMP推流清晰度信息
        
        请求参数:
        - sessionid: 推流会话ID (可选，不提供则返回可用清晰度级别)
        
        返回:
        - quality_info: 清晰度信息
        """
        try:
            sessionid = request.query.get('sessionid')
            
            if sessionid:
                # 获取指定会话的清晰度信息
                nerfreal = self.safe_get_nerfreal(sessionid)
                if not nerfreal:
                    return web.json_response({
                        'code': 404,
                        'message': f'推流会话 {sessionid} 不存在或未启动',
                        'data': None
                    })
                    
                quality_info = nerfreal.get_rtmp_quality_info()
                
                return web.json_response({
                    'code': 200,
                    'message': '获取清晰度信息成功',
                    'data': {
                        'sessionid': sessionid,
                        'quality_info': quality_info
                    }
                })
            else:
                # 返回可用的清晰度级别（使用临时对象获取配置）
                temp_nerfreal = self.build_nerfreal('temp')
                if temp_nerfreal:
                    available_qualities = temp_nerfreal.get_available_rtmp_qualities()
                    # 清理临时对象
                    del temp_nerfreal
                    
                    return web.json_response({
                        'code': 200,
                        'message': '获取可用清晰度级别成功',
                        'data': {
                            'available_qualities': available_qualities
                        }
                    })
                else:
                    return web.json_response({
                        'code': 500,
                        'message': '无法获取清晰度配置',
                        'data': None
                    })
                    
        except Exception as e:
            logger.error(f"获取RTMP清晰度信息失败: {e}")
            return web.json_response({
                'code': 500,
                'message': f'获取清晰度信息失败: {str(e)}',
                'data': None
            })

    async def get_rtmp_stats(self, request):
        """获取RTMP推流统计信息
        
        请求参数:
        - sessionid: 推流会话ID
        
        返回:
        - stats: 推流统计信息
        """
        try:
            sessionid = request.query.get('sessionid')
            
            if not sessionid:
                return web.json_response({
                    'code': 400,
                    'message': '缺少sessionid参数',
                    'data': None
                })
                
            # 获取nerfreal对象
            nerfreal = self.safe_get_nerfreal(sessionid)
            if not nerfreal:
                return web.json_response({
                    'code': 404,
                    'message': f'推流会话 {sessionid} 不存在或未启动',
                    'data': None
                })
                
            # 获取RTMP统计信息
            stats = nerfreal.get_rtmp_stats()
            
            return web.json_response({
                'code': 200,
                'message': '获取推流统计信息成功',
                'data': {
                    'sessionid': sessionid,
                    'stats': stats
                }
            })
            
        except Exception as e:
            logger.error(f"获取RTMP推流统计信息失败: {e}")
            return web.json_response({
                'code': 500,
                'message': f'获取推流统计信息失败: {str(e)}',
                'data': None
            })
