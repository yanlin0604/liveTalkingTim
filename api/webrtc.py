"""
WebRTC相关API接口
"""
import json
import random
import asyncio
import gc
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
        
        logger.info('sessionid=%d, session num=%d', sessionid, len(self.nerfreals))
        nerfreal = await asyncio.get_event_loop().run_in_executor(None, self.build_nerfreal, sessionid)
        
        with self.nerfreals_lock:
            self.nerfreals[sessionid] = nerfreal
        
        ice_server = RTCIceServer(urls='stun:stun.miwifi.com:3478')
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s" % pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
                with self.nerfreals_lock:
                    if sessionid in self.nerfreals:
                        del self.nerfreals[sessionid]
            if pc.connectionState == "closed":
                self.pcs.discard(pc)
                with self.nerfreals_lock:
                    if sessionid in self.nerfreals:
                        del self.nerfreals[sessionid]
                gc.collect()

        # 安全获取nerfreal对象
        with self.nerfreals_lock:
            nerfreal_for_player = self.nerfreals.get(sessionid)
        
        if nerfreal_for_player is None:
            logger.error(f"Failed to get nerfreal for session {sessionid}")
            return web.Response(status=500, text="Internal server error")
        
        player = HumanPlayer(nerfreal_for_player)
        audio_sender = pc.addTrack(player.audio)
        video_sender = pc.addTrack(player.video)
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
            ),
        ) 