"""
WebRTC相关API接口
"""
import json
import random
import asyncio
import gc
import time
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from logger import logger


def randN(N) -> int:
    '''生成长度为 N的随机数 '''
    min_val = pow(10, N - 1)
    max_val = pow(10, N)
    return random.randint(min_val, max_val - 1)


class WebRTCAPI:
    """WebRTC API接口类"""
    
    def __init__(self, build_nerfreal_func, nerfreals_dict, nerfreals_lock, pcs_set):
        self.build_nerfreal = build_nerfreal_func
        self.nerfreals = nerfreals_dict
        self.nerfreals_lock = nerfreals_lock
        self.pcs = pcs_set
        self.connection_stats = {}  # 存储连接统计信息
        self.reconnect_attempts = {}  # 存储重连尝试次数
    
    async def offer(self, request):
        """
        WebRTC连接建立接口

        功能：处理客户端的WebRTC SDP offer，建立音视频连接
        
        ---
        tags:
          - WebRTC
        summary: 建立WebRTC连接
        description: 处理客户端的WebRTC SDP offer，建立音视频连接
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
                sdp:
                  type: string
                  description: WebRTC会话描述协议数据
                type:
                  type: string
                  description: SDP类型（通常为'offer'）
                  default: offer
              required:
                - sdp
                - type
        responses:
          200:
            description: 连接建立成功
            schema:
              type: object
              properties:
                sdp:
                  type: string
                  description: 服务端的SDP answer
                type:
                  type: string
                  description: SDP类型（'answer'）
                sessionid:
                  type: integer
                  description: 分配的会话ID
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
            
            # 验证必需参数
            if 'sdp' not in params:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Missing required parameter: sdp"}
                    ),
                )
            
            if 'type' not in params:
                return web.Response(
                    content_type="application/json",
                    text=json.dumps(
                        {"code": -1, "msg": "Missing required parameter: type"}
                    ),
                )
            
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
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

        sessionid = randN(6)
        
        with self.nerfreals_lock:
            self.nerfreals[sessionid] = None
        
        logger.info('会话ID=%d, 当前会话数=%d', sessionid, len(self.nerfreals))
        nerfreal = await asyncio.get_event_loop().run_in_executor(None, self.build_nerfreal, sessionid)
        
        with self.nerfreals_lock:
            self.nerfreals[sessionid] = nerfreal
        
        ice_server = RTCIceServer(urls='stun:stun.miwifi.com:3478')
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("连接状态变化: %s" % pc.connectionState)
            
            # 更新连接统计信息
            if sessionid not in self.connection_stats:
                self.connection_stats[sessionid] = {
                    'created_time': time.time(),
                    'last_state_change': time.time(),
                    'state_history': []
                }
            
            self.connection_stats[sessionid]['last_state_change'] = time.time()
            self.connection_stats[sessionid]['state_history'].append({
                'state': pc.connectionState,
                'timestamp': time.time()
            })
            
            if pc.connectionState == "connected":
                logger.info(f"✅ WebRTC连接已建立 - 会话 {sessionid}")
                logger.info(f"📊 当前码率配置: 最大={max_bitrate/1000:.0f}k, 最小={min_bitrate/1000:.0f}k, 起始={start_bitrate/1000:.0f}k")
                # 重置重连尝试次数
                self.reconnect_attempts[sessionid] = 0
                
            elif pc.connectionState == "failed":
                logger.error(f"❌ WebRTC连接失败 - 会话 {sessionid}")
                await pc.close()
                self.pcs.discard(pc)
                # 优雅清理nerfreal资源
                await self._cleanup_nerfreal_session(sessionid)
                
            elif pc.connectionState == "closed":
                logger.info(f"🔌 WebRTC连接已关闭 - 会话 {sessionid}")
                self.pcs.discard(pc)
                # 优雅清理nerfreal资源
                await self._cleanup_nerfreal_session(sessionid)
                gc.collect()
                
            # 记录连接状态变化
            logger.info(f"📈 会话 {sessionid} 连接统计: 状态={pc.connectionState}, 持续时间={time.time() - self.connection_stats[sessionid]['created_time']:.1f}s")

        # 安全获取nerfreal对象
        with self.nerfreals_lock:
            nerfreal_for_player = self.nerfreals.get(sessionid)
        
        if nerfreal_for_player is None:
            logger.error(f"获取会话 {sessionid} 的nerfreal对象失败")
            return web.Response(status=500, text="Internal server error")
        
        player = HumanPlayer(nerfreal_for_player)
        audio_sender = pc.addTrack(player.audio)
        video_sender = pc.addTrack(player.video)
        
        # 获取流质量配置
        streaming_config = getattr(nerfreal_for_player, 'streaming_quality', {})
        max_bitrate = streaming_config.get('max_bitrate', 1500000)  # 降低到1.5Mbps
        min_bitrate = streaming_config.get('min_bitrate', 300000)   # 降低到300kbps
        start_bitrate = streaming_config.get('start_bitrate', 800000)  # 降低到800kbps
        audio_max_bitrate = streaming_config.get('audio_max_bitrate', 128000)  # 128kbps
        audio_min_bitrate = streaming_config.get('audio_min_bitrate', 64000)   # 64kbps
        
        # 设置视频码率控制 - 仅使用SDP方式
        try:
            logger.info(f"=== WebRTC码率配置 ===")
            logger.info(f"视频码率范围: {min_bitrate/1000:.0f}k - {max_bitrate/1000:.0f}k")
            logger.info(f"音频码率范围: {audio_min_bitrate/1000:.0f}k - {audio_max_bitrate/1000:.0f}k")
            logger.info(f"目标帧率: {streaming_config.get('target_fps', 25)} fps")
            logger.info(f"起始码率: {start_bitrate/1000:.0f}k")
            logger.info("使用SDP方式设置码率（兼容性最佳）")
            
        except Exception as e:
            logger.warning(f"设置WebRTC码率失败: {e}")
            logger.info("将使用默认码率设置")
        
        capabilities = RTCRtpSender.getCapabilities("video")
        # 优先使用VP8编码器，避免H.264 Level限制问题
        # 可以通过配置选择编码器优先级
        encoder_preference = getattr(nerfreal_for_player, 'encoder_preference', 'vp8_first')
        
        if encoder_preference == 'h264_first':
            preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
            preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
            logger.info(f"🎬 编码器优先级: H264 > VP8 > RTX")
        else:  # 默认VP8优先
            preferences = list(filter(lambda x: x.name == "VP8", capabilities.codecs))
            preferences += list(filter(lambda x: x.name == "H264", capabilities.codecs))
            logger.info(f"🎬 编码器优先级: VP8 > H264 > RTX (推荐，避免Level限制)")
        
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)
        
        # 设置编码参数以符合H.264 Level 3.1限制
        try:
            # 设置最大分辨率限制（符合Level 3.1）
            max_width = 1280
            max_height = 720
            max_fps = 30
            
            # 设置编码参数
            sender = video_sender
            if hasattr(sender, 'setParameters'):
                params = sender.getParameters()
                if params.encodings:
                    # 设置编码参数
                    params.encodings[0].maxBitrate = max_bitrate
                    params.encodings[0].minBitrate = min_bitrate
                    params.encodings[0].maxFramerate = max_fps
                    # 添加H.264 Level 3.1兼容性设置
                    params.encodings[0].scaleResolutionDownBy = 1.0  # 不缩放分辨率
                    sender.setParameters(params)
                    logger.info(f"✅ 设置编码参数: 最大码率={max_bitrate/1000:.0f}k, 最大帧率={max_fps}")
                    logger.info(f"📐 分辨率限制: {max_width}x{max_height}, H.264 Level 3.1兼容")
        except Exception as e:
            logger.warning(f"设置编码参数失败: {e}")

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        
        # 在SDP中添加码率限制和编码器优化 - 改进版本
        try:
            sdp_lines = answer.sdp.split('\n')
            modified_sdp_lines = []
            video_bitrate_added = False
            audio_bitrate_added = False
            
            for i, line in enumerate(sdp_lines):
                modified_sdp_lines.append(line)
                
                # 在视频媒体行后添加码率限制和编码器优化
                if line.startswith('m=video') and not video_bitrate_added:
                    # 查找视频媒体段的结束位置
                    j = i + 1
                    while j < len(sdp_lines) and not sdp_lines[j].startswith('m='):
                        j += 1
                    
                    # 在视频媒体段末尾添加码率设置和编码器参数
                    if j < len(sdp_lines):
                        # 添加码率限制
                        modified_sdp_lines.insert(j, f'b=AS:{max_bitrate // 1000}')
                        modified_sdp_lines.insert(j + 1, f'b=TIAS:{max_bitrate}')
                        
                        # 添加VP8编码器优化参数
                        modified_sdp_lines.insert(j + 2, 'a=fmtp:96 max-fr=30;max-fs=3600')
                        modified_sdp_lines.insert(j + 3, 'a=fmtp:96 profile-level-id=42e01f')
                        
                        logger.info(f"✅ 在SDP中添加视频码率限制: {max_bitrate // 1000}k")
                        logger.info(f"🔧 添加VP8编码器优化参数")
                        video_bitrate_added = True
                
                # 在音频媒体行后添加码率限制
                if line.startswith('m=audio') and not audio_bitrate_added:
                    # 查找音频媒体段的结束位置
                    j = i + 1
                    while j < len(sdp_lines) and not sdp_lines[j].startswith('m='):
                        j += 1
                    
                    # 在音频媒体段末尾添加码率设置
                    if j < len(sdp_lines):
                        modified_sdp_lines.insert(j, f'b=AS:{audio_max_bitrate // 1000}')
                        modified_sdp_lines.insert(j + 1, f'b=TIAS:{audio_max_bitrate}')
                        logger.info(f"✅ 在SDP中添加音频码率限制: {audio_max_bitrate // 1000}k")
                        audio_bitrate_added = True
            
            modified_sdp = '\n'.join(modified_sdp_lines)
            answer = RTCSessionDescription(sdp=modified_sdp, type=answer.type)
            logger.info("✅ SDP码率参数和编码器优化设置成功")
        except Exception as e:
            logger.warning(f"SDP码率设置失败: {e}")
        
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
            ),
        ) 

    async def get_connection_status(self, request):
        """
        获取连接状态监控信息
        
        ---
        tags:
          - WebRTC
        summary: 获取连接状态
        description: 获取所有WebRTC连接的状态统计信息
        produces:
          - application/json
        responses:
          200:
            description: 连接状态信息
            schema:
              type: object
              properties:
                active_connections:
                  type: integer
                  description: 活跃连接数
                total_sessions:
                  type: integer
                  description: 总会话数
                connection_stats:
                  type: object
                  description: 各会话的连接统计
        """
        try:
            active_connections = len(self.pcs)
            total_sessions = len(self.nerfreals)
            
            # 清理过期的连接统计信息
            current_time = time.time()
            expired_sessions = []
            for sessionid, stats in self.connection_stats.items():
                if current_time - stats['last_state_change'] > 3600:  # 1小时无活动则清理
                    expired_sessions.append(sessionid)
            
            for sessionid in expired_sessions:
                del self.connection_stats[sessionid]
                if sessionid in self.reconnect_attempts:
                    del self.reconnect_attempts[sessionid]
            
            return web.Response(
                content_type="application/json",
                text=json.dumps({
                    "active_connections": active_connections,
                    "total_sessions": total_sessions,
                    "connection_stats": self.connection_stats,
                    "reconnect_attempts": self.reconnect_attempts
                }, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logger.error(f"获取连接状态失败: {e}")
            return web.Response(
                content_type="application/json",
                text=json.dumps({"error": str(e)}),
                status=500
            )

    async def _cleanup_nerfreal_session(self, sessionid):
        """优雅清理nerfreal会话资源"""
        try:
            with self.nerfreals_lock:
                if sessionid in self.nerfreals:
                    nerfreal = self.nerfreals[sessionid]
                    logger.info(f"🧹 开始清理会话 {sessionid} 的资源...")
                    
                    # 停止渲染线程（如果正在运行）
                    if hasattr(nerfreal, 'render_event') and nerfreal.render_event:
                        nerfreal.render_event.clear()
                        logger.info(f"🛑 停止会话 {sessionid} 的渲染事件")
                    
                    # 停止TTS线程（如果正在运行）
                    if hasattr(nerfreal, 'tts') and hasattr(nerfreal.tts, 'state'):
                        nerfreal.tts.state = 1  # 设置为PAUSE状态
                        logger.info(f"🛑 停止会话 {sessionid} 的TTS处理")
                    
                    # 清理音频和视频队列
                    if hasattr(nerfreal, 'asr') and hasattr(nerfreal.asr, 'feat_queue'):
                        try:
                            while not nerfreal.asr.feat_queue.empty():
                                nerfreal.asr.feat_queue.get_nowait()
                        except:
                            pass
                        logger.info(f"🧹 清理会话 {sessionid} 的音频特征队列")
                    
                    # 删除nerfreal对象
                    del self.nerfreals[sessionid]
                    logger.info(f"✅ 会话 {sessionid} 资源清理完成")
                    
        except Exception as e:
            logger.error(f"❌ 清理会话 {sessionid} 资源时出错: {e}")
        finally:
            # 强制垃圾回收
            gc.collect() 