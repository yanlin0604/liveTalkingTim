"""
TTS试听相关API接口
"""
import json
import base64
import uuid
import requests
import asyncio
from aiohttp import web
from logger import logger


class TTSAPI:
    """TTS试听API接口类"""
    
    def __init__(self):
        # 豆包TTS配置
        self.doubao_config = {
            "appid": "8737889718",
            "access_token": "KitELbI5WB5yYEy4BxrD6lJnWWSaoXYb",
            "cluster": "volcano_tts",
            "host": "openspeech.bytedance.com"
        }
    
    async def preview_tts(self, request):
        """
        TTS试听接口

        功能：根据指定的TTS类型和语音类型生成试听音频
        
        ---
        tags:
          - TTS
        summary: TTS试听
        description: 根据指定的TTS类型和语音类型生成试听音频
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
                tts_type:
                  type: string
                  enum: [doubao]
                  description: TTS类型（目前支持doubao）
                voice_type:
                  type: string
                  description: 语音类型
                text:
                  type: string
                  description: 要试听的文本内容
                  default: "这是一个试听测试"
              required:
                - tts_type
                - voice_type
        responses:
          200:
            description: 试听成功
            schema:
              type: object
              properties:
                code:
                  type: integer
                  description: 状态码（0=成功，-1=失败）
                message:
                  type: string
                  description: 状态消息
                data:
                  type: object
                  properties:
                    audio_base64:
                      type: string
                      description: Base64编码的音频数据
                    audio_format:
                      type: string
                      description: 音频格式
        """
        try:
            # 检查请求体是否为空
            body = await request.text()
            if not body.strip():
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"success": False, "message": "Request body is empty"}
                    ),
                )
            
            params = await request.json()
            
            # 验证必需参数
            if 'tts_type' not in params:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"success": False, "message": "Missing required parameter: tts_type"}
                    ),
                )
            
            if 'voice_type' not in params:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"success": False, "message": "Missing required parameter: voice_type"}
                    ),
                )
            
            tts_type = params.get('tts_type')
            voice_type = params.get('voice_type')
            text = params.get('text', '这是一个试听测试')
            
            # 验证TTS类型
            if tts_type not in ['doubao']:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"success": False, "message": f"Unsupported TTS type: {tts_type}"}
                    ),
                )
            
            logger.info(f"TTS试听请求: tts_type={tts_type}, voice_type={voice_type}, text={text}")
            
            # 根据TTS类型调用相应的试听方法
            if tts_type == 'doubao':
                result = await self._preview_doubao_tts(voice_type, text)
            else:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"success": False, "message": f"TTS type {tts_type} not implemented yet"}
                    ),
                )
            
            if result['success']:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        "success": True,
                        "message": "试听成功",
                        "data": {
                            "audio_base64": result['audio_base64'],
                            "audio_format": "mp3"
                        }
                    }),
                )
            else:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps({
                        "success": False,
                        "message": f"试听失败: {result['error']}"
                    }),
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"success": False, "message": f"Invalid JSON format: {str(e)}"}
                ),
            )
        except Exception as e:
            logger.error(f"TTS试听接口异常: {e}")
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"success": False, "message": f"Internal server error: {str(e)}"}
                ),
            )
    
    async def _preview_doubao_tts(self, voice_type: str, text: str):
        """
        豆包TTS试听实现
        
        Args:
            voice_type: 语音类型
            text: 要试听的文本
            
        Returns:
            dict: 包含成功状态和音频数据的字典
        """
        try:
            # 构建请求URL
            api_url = f"https://{self.doubao_config['host']}/api/v1/tts"
            
            # 构建请求头
            header = {"Authorization": f"Bearer;{self.doubao_config['access_token']}"}
            
            # 构建请求体
            request_json = {
                "app": {
                    "appid": self.doubao_config['appid'],
                    "token": "access_token",
                    "cluster": self.doubao_config['cluster']
                },
                "user": {
                    "uid": "388808087185088"
                },
                "audio": {
                    "voice_type": voice_type,
                    "encoding": "mp3",
                    "speed_ratio": 1.0,
                    "volume_ratio": 1.0,
                    "pitch_ratio": 1.0,
                },
                "request": {
                    "reqid": str(uuid.uuid4()),
                    "text": text,
                    "text_type": "plain",
                    "operation": "query",
                    "with_frontend": 1,
                    "frontend_type": "unitTson"
                }
            }
            
            logger.info(f"豆包TTS请求: {json.dumps(request_json, ensure_ascii=False, indent=2)}")
            
            # 发送HTTP请求
            response = requests.post(api_url, json=request_json, headers=header)
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"豆包TTS响应: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                
                if "data" in response_data:
                    # 解码Base64音频数据
                    audio_data = base64.b64decode(response_data["data"])
                    
                    return {
                        "success": True,
                        "audio_base64": response_data["data"],
                        "audio_size": len(audio_data)
                    }
                else:
                    logger.error(f"豆包TTS响应中没有data字段: {response_data}")
                    return {
                        "success": False,
                        "error": "TTS服务返回数据格式错误"
                    }
            else:
                logger.error(f"豆包TTS请求失败: status_code={response.status_code}, response={response.text}")
                return {
                    "success": False,
                    "error": f"TTS服务请求失败: HTTP {response.status_code}"
                }
                
        except requests.RequestException as e:
            logger.error(f"豆包TTS网络请求异常: {e}")
            return {
                "success": False,
                "error": f"网络请求异常: {str(e)}"
            }
        except Exception as e:
            logger.error(f"豆包TTS试听异常: {e}")
            return {
                "success": False,
                "error": f"试听处理异常: {str(e)}"
            } 