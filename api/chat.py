"""
聊天和交互相关API接口
"""
import json
import asyncio
from aiohttp import web
from llm import llm_response
from logger import logger


class ChatAPI:
    """聊天和交互API接口类"""
    
    def __init__(self, nerfreals_dict, nerfreals_lock):
        self.nerfreals = nerfreals_dict
        self.nerfreals_lock = nerfreals_lock
    
    async def human(self, request):
        """
        文本交互接口

        功能：发送文本消息给数字人，支持直接播报和AI对话两种模式
        
        ---
        tags:
          - Chat
        summary: 文本交互
        description: 发送文本消息给数字人，支持直接播报和AI对话两种模式
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                text:
                  type: string
                  description: 要发送的文本内容
                type:
                  type: string
                  enum: [echo, chat]
                  description: 消息类型（echo=直接播报，chat=AI对话）
                interrupt:
                  type: boolean
                  description: 是否打断当前说话
                  default: false
                sessionid:
                  type: integer
                  description: 会话ID
                  default: 0
              required:
                - text
                - type
        responses:
          200:
            description: 发送成功
            schema:
              type: object
              properties:
                code:
                  type: integer
                  description: 状态码（0=成功，-1=失败）
                msg:
                  type: string
                  description: 状态消息
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Request body is empty"}
                    ),
                )
            
            params = await request.json()

            sessionid = params.get('sessionid', 0)
            
            # 验证必需参数
            if 'text' not in params:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Missing required parameter: text"}
                    ),
                )
            
            if 'type' not in params:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Missing required parameter: type"}
                    ),
                )
            
            # 使用锁保护访问
            with self.nerfreals_lock:
                if sessionid not in self.nerfreals:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Session {sessionid} not found"}
                        ),
                    )
                
                nerfreal = self.nerfreals[sessionid]
                
                if params.get('interrupt'):
                    nerfreal.flush_talk()

                if params['type'] == 'echo':
                    nerfreal.put_msg_txt(params['text'])
                elif params['type'] == 'chat':
                    asyncio.get_event_loop().run_in_executor(None, llm_response, params['text'], nerfreal)
                else:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Invalid type: {params['type']}. Must be 'echo' or 'chat'"}
                        ),
                    )

            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": 0, "msg": "ok"}
                ),
            )
        except json.JSONDecodeError as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": f"Invalid JSON format: {str(e)}"}
                ),
            )
        except Exception as e:
            logger.exception('exception:')
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": str(e)}
                ),
            )

    async def interrupt_talk(self, request):
        """
        打断数字人说话接口

        功能：立即停止数字人当前的说话，清空待播放的消息队列
        方法：POST
        参数：
            - sessionid: 会话ID（可选，默认0）
        返回：
            - code: 状态码（0=成功，-1=失败）
            - msg: 状态消息

        使用场景：
            - 用户需要紧急打断数字人
            - 切换话题时清空当前播放队列
            - 重置对话状态

        示例：
            {
                "sessionid": 0
            }
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Request body is empty"}
                    ),
                )
            
            params = await request.json()

            sessionid = params.get('sessionid', 0)
            
            # 使用锁保护访问
            with self.nerfreals_lock:
                if sessionid not in self.nerfreals:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Session {sessionid} not found"}
                        ),
                    )
                
                self.nerfreals[sessionid].flush_talk()
            
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": 0, "msg": "ok"}
                ),
            )
        except json.JSONDecodeError as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": f"Invalid JSON format: {str(e)}"}
                ),
            )
        except Exception as e:
            logger.exception('exception:')
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": str(e)}
                ),
            )

    async def humanaudio(self, request):
        """
        音频交互接口

        功能：接收用户音频文件，进行语音识别后转换为文本交互
        方法：POST (multipart/form-data)
        参数：
            - file: 音频文件（支持wav、mp3等格式）
            - sessionid: 会话ID（可选，默认0）
        返回：
            - code: 状态码（0=成功，-1=失败）
            - msg: 状态消息

        流程：
            1. 接收音频文件
            2. 进行语音识别转换为文本
            3. 自动调用相应的处理逻辑
        """
        try:
            form = await request.post()
            sessionid = int(form.get('sessionid', 0))
            
            fileobj = form["file"]
            filename = fileobj.filename
            filebytes = fileobj.file.read()
            
            # 使用锁保护访问
            with self.nerfreals_lock:
                if sessionid not in self.nerfreals:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Session {sessionid} not found"}
                        ),
                    )
                
                self.nerfreals[sessionid].put_audio_file(filebytes)

            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": 0, "msg": "ok"}
                ),
            )
        except Exception as e:
            logger.exception('exception:')
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": str(e)}
                ),
            )

    async def set_audiotype(self, request):
        """
        设置音频类型接口

        功能：设置数字人的音频播放类型和相关参数
        方法：POST
        参数：
            - sessionid: 会话ID（可选，默认0）
            - audiotype: 音频类型标识（可以是数字或字符串，对应custom_config.json中的audiotype）
            - reinit: 是否重新初始化（布尔值）
        返回：
            - code: 状态码（0=成功，-1=失败）
            - msg: 状态消息

        使用场景：
            - 切换不同的音频播放模式
            - 动态调整音频参数
            - 重置音频状态
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Request body is empty"}
                    ),
                )
            
            params = await request.json()

            sessionid = params.get('sessionid', 0)
            audiotype = params.get('audiotype')
            
            # 验证必需参数
            if audiotype is None:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Missing required parameter: audiotype"}
                    ),
                )
            
            # 使用锁保护访问
            with self.nerfreals_lock:
                if sessionid not in self.nerfreals:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Session {sessionid} not found"}
                        ),
                    )
                
                # 检查audiotype是否有效
                nerfreal = self.nerfreals[sessionid]
                if audiotype not in nerfreal.custom_index:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Audiotype '{audiotype}' not found in custom_config.json"}
                        ),
                    )
                
                nerfreal.set_custom_state(audiotype, params.get('reinit', True))

            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": 0, "msg": "ok"}
                ),
            )
        except json.JSONDecodeError as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": f"Invalid JSON format: {str(e)}"}
                ),
            )
        except Exception as e:
            logger.exception('exception:')
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": str(e)}
                ),
            )

    async def set_custom_silent(self, request):
        """
        设置静音时是否使用自定义动作接口

        功能：控制数字人在静音时是否自动使用audiotype=2的自定义动作
        方法：POST
        参数：
            - sessionid: 会话ID（可选，默认0）
            - enabled: 是否启用（布尔值）
        返回：
            - code: 状态码（0=成功，-1=失败）
            - msg: 状态消息
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Request body is empty"}
                    ),
                )
            
            params = await request.json()
            sessionid = params.get('sessionid', 0)
            
            enabled = params.get('enabled', True)
            
            # 使用锁保护访问
            with self.nerfreals_lock:
                if sessionid not in self.nerfreals:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Session {sessionid} not found"}
                        ),
                    )
                
                self.nerfreals[sessionid].set_use_custom_silent(enabled)
            
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": 0, "msg": "ok"}
                ),
            )
        except json.JSONDecodeError as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": f"Invalid JSON format: {str(e)}"}
                ),
            )
        except Exception as e:
            logger.exception('exception:')
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": str(e)}
                ),
            )

    async def record(self, request):
        """
        录制控制接口

        功能：控制数字人视频和音频的录制功能
        方法：POST
        参数：
            - sessionid: 会话ID（可选，默认0）
            - type: 录制操作类型
                * "start_record": 开始录制
                * "stop_record": 停止录制
            - path: 录制文件保存路径（开始录制时需要）
        返回：
            - code: 状态码（0=成功，-1=失败）
            - msg: 状态消息

        使用场景：
            - 录制数字人对话视频
            - 保存重要的交互内容
            - 生成演示材料

        示例：
            开始录制: {"type": "start_record", "path": "/path/to/video.mp4", "sessionid": 0}
            停止录制: {"type": "stop_record", "sessionid": 0}
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Request body is empty"}
                    ),
                )
            
            params = await request.json()
            sessionid = params.get('sessionid', 0)
            
            # 验证必需参数
            if 'type' not in params:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Missing required parameter: type"}
                    ),
                )
            
            # 使用锁保护访问
            with self.nerfreals_lock:
                if sessionid not in self.nerfreals:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Session {sessionid} not found"}
                        ),
                    )
                
                if params['type'] == 'start_record':
                    if 'path' not in params:
                        return web.Response(
                            content_type="application/json",
                            text=json.dumps(
                                {"code": -1, "msg": "Missing required parameter: path for start_record"}
                            ),
                        )
                    self.nerfreals[sessionid].start_recording(params['path'])
                elif params['type'] == 'stop_record':
                    self.nerfreals[sessionid].stop_recording()
                else:
                    return web.Response(
                        content_type="application/json",
                        text=json.dumps(
                            {"code": -1, "msg": f"Invalid type: {params['type']}. Must be 'start_record' or 'stop_record'"}
                        ),
                    )

            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": 0, "msg": "ok"}
                ),
            )
        except json.JSONDecodeError as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": f"Invalid JSON format: {str(e)}"}
                ),
            )
        except Exception as e:
            logger.exception('exception:')
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": str(e)}
                ),
            )

    async def is_speaking(self, request):
        """
        检查数字人说话状态接口

        功能：查询指定会话的数字人是否正在说话
        方法：POST
        参数：
            - sessionid: 会话ID（可选，默认0）
        返回：
            - code: 状态码（0=成功）
            - data: 说话状态（true=正在说话，false=未说话）

        使用场景：
            - 判断是否可以发送新消息
            - 实现智能打断逻辑
            - 监控数字人状态
            - 同步前端UI状态

        示例：
            请求: {"sessionid": 0}
            响应: {"code": 0, "data": true}
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Request body is empty"}
                    ),
                )
            
            params = await request.json()
            sessionid = params.get('sessionid', 0)
        except json.JSONDecodeError as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": f"Invalid JSON format: {str(e)}"}
                ),
            )
        except Exception as e:
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"code": -1, "msg": f"Request parsing error: {str(e)}"}
                ),
            )
        
        # 使用锁保护访问
        with self.nerfreals_lock:
            if sessionid not in self.nerfreals:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        "code": -1,
                        "msg": f"Session {sessionid} not found"
                    }),
                )
            
            nerfreal = self.nerfreals[sessionid]
        
        # 获取当前状态信息
        is_speaking = nerfreal.is_speaking()
        current_audiotype = getattr(nerfreal, '_last_silent_audiotype', None) if not is_speaking else None
        default_silent_audiotype = nerfreal.get_default_silent_audiotype()
        
        # 获取可用的audiotype列表
        available_audiotypes = list(nerfreal.custom_index.keys()) if hasattr(nerfreal, 'custom_index') else []
        
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0, 
                "data": {
                    "is_speaking": is_speaking,
                    "current_audiotype": current_audiotype,
                    "default_silent_audiotype": default_silent_audiotype,
                    "available_audiotypes": available_audiotypes
                }
            }),
        ) 