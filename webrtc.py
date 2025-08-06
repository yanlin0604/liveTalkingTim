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

import asyncio
import json
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 0.040 #1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

# 配置aiortc模块的详细日志
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

# 启用aiortc相关模块的详细日志
# aiortc_logger = logging.getLogger('aiortc')
# aiortc_logger.setLevel(logging.DEBUG)

# 启用其他相关模块的日志
# rtp_logger = logging.getLogger('aiortc.rtcrtpsender')
# rtp_logger.setLevel(logging.DEBUG)

# rtp_receiver_logger = logging.getLogger('aiortc.rtcrtpreceiver')
# rtp_receiver_logger.setLevel(logging.DEBUG)

# connection_logger = logging.getLogger('aiortc.rtcicetransport')
# connection_logger.setLevel(logging.DEBUG)

# dtls_logger = logging.getLogger('aiortc.rtcdtlstransport')
# dtls_logger.setLevel(logging.DEBUG)

# sctp_logger = logging.getLogger('aiortc.rtcsctptransport')
# sctp_logger.setLevel(logging.DEBUG)

# peer_connection_logger = logging.getLogger('aiortc.rtcpeerconnection')
# peer_connection_logger.setLevel(logging.DEBUG)

# 启用更多相关模块的日志
# media_logger = logging.getLogger('aiortc.mediastreams')
# media_logger.setLevel(logging.DEBUG)

# codec_logger = logging.getLogger('aiortc.codecs')
# codec_logger.setLevel(logging.DEBUG)

# 启用av库的日志（用于音视频处理）
# av_logger = logging.getLogger('av')
# av_logger.setLevel(logging.INFO)  # 设置为INFO级别避免过于详细

# 启用asyncio的日志（用于异步操作）
# asyncio_logger = logging.getLogger('asyncio')
# asyncio_logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
from logger import logger as mylogger

# 添加日志输出说明
# mylogger.info("=== aiortc日志已启用 ===")
# mylogger.info("日志级别: DEBUG")
# mylogger.info("将输出以下模块的详细信息:")
# mylogger.info("- aiortc: 核心WebRTC功能")
# mylogger.info("- aiortc.rtcrtpsender: RTP发送器")
# mylogger.info("- aiortc.rtcrtpreceiver: RTP接收器") 
# mylogger.info("- aiortc.rtcicetransport: ICE传输")
# mylogger.info("- aiortc.rtcdtlstransport: DTLS传输")
# mylogger.info("- aiortc.rtcsctptransport: SCTP传输")
# mylogger.info("- aiortc.rtcpeerconnection: 对等连接")
# mylogger.info("- aiortc.mediastreams: 媒体流")
# mylogger.info("- aiortc.codecs: 编解码器")
# mylogger.info("=== 日志分析开始 ===")


class BitrateMonitor:
    """码率监控和自适应调整类"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.max_bitrate = self.config.get('max_bitrate', 2000000)
        self.min_bitrate = self.config.get('min_bitrate', 500000)
        self.current_bitrate = self.config.get('start_bitrate', 1000000)
        self.bitrate_history = []
        self.fps_history = []
        self.last_adjustment = time.time()
        self.adjustment_interval = 10  # 10秒调整一次
        
    def update_metrics(self, fps, queue_size):
        """更新性能指标"""
        current_time = time.time()
        self.fps_history.append(fps)
        self.bitrate_history.append(self.current_bitrate)
        
        # 保持历史记录在合理范围内
        if len(self.fps_history) > 10:
            self.fps_history.pop(0)
        if len(self.bitrate_history) > 10:
            self.bitrate_history.pop(0)
            
        # 检查是否需要调整码率
        if current_time - self.last_adjustment > self.adjustment_interval:
            self._adjust_bitrate()
            self.last_adjustment = current_time
    
    def _adjust_bitrate(self):
        """自适应调整码率"""
        if len(self.fps_history) < 3:
            return
            
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        target_fps = self.config.get('target_fps', 25)
        
        # mylogger.info(f"=== 码率自适应调整 ===")
        # mylogger.info(f"历史帧率: {[f'{fps:.1f}' for fps in self.fps_history[-3:]]}")
        # mylogger.info(f"平均帧率: {avg_fps:.1f} fps")
        # mylogger.info(f"目标帧率: {target_fps} fps")
        # mylogger.info(f"当前码率: {self.current_bitrate/1000:.0f}kbps")
        
        # 根据帧率调整码率
        if avg_fps < target_fps * 0.8:  # 帧率过低，降低码率
            new_bitrate = max(self.min_bitrate, self.current_bitrate * 0.8)
            # mylogger.info(f"帧率过低({avg_fps:.1f}fps < {target_fps*0.8:.1f}fps)，降低码率: {self.current_bitrate/1000:.0f}k -> {new_bitrate/1000:.0f}k")
            self.current_bitrate = new_bitrate
        elif avg_fps > target_fps * 1.2:  # 帧率过高，可以提高码率
            new_bitrate = min(self.max_bitrate, self.current_bitrate * 1.1)
            # mylogger.info(f"帧率良好({avg_fps:.1f}fps > {target_fps*1.2:.1f}fps)，提高码率: {self.current_bitrate/1000:.0f}k -> {new_bitrate/1000:.0f}k")
            self.current_bitrate = new_bitrate
        else:
            # mylogger.info(f"帧率正常({avg_fps:.1f}fps)，保持当前码率: {self.current_bitrate/1000:.0f}kbps")
            pass
    
    def get_current_bitrate(self):
        """获取当前码率"""
        return self.current_bitrate


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind, config=None):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
        
        # 使用配置参数
        self.config = config or {}
        self.max_queue_size = self.config.get('max_video_queue_size', 12)
        self.frame_drop_threshold = self.config.get('frame_drop_threshold', 12)
        
        # 初始化码率监控器
        self.bitrate_monitor = BitrateMonitor(self.config)
        
        # 打印码率配置参数
        if self.kind == 'video':
            # mylogger.info(f"=== WebRTC视频流配置 ===")
            # mylogger.info(f"目标帧率: {self.config.get('target_fps', 25)} fps")
            # mylogger.info(f"最大码率: {self.config.get('max_bitrate', 2000000)/1000:.0f}kbps")
            # mylogger.info(f"最小码率: {self.config.get('min_bitrate', 500000)/1000:.0f}kbps")
            # mylogger.info(f"起始码率: {self.config.get('start_bitrate', 1000000)/1000:.0f}kbps")
            # mylogger.info(f"最大队列大小: {self.max_queue_size}")
            # mylogger.info(f"帧丢弃阈值: {self.frame_drop_threshold}")
            pass
        elif self.kind == 'audio':
            # mylogger.info(f"=== WebRTC音频流配置 ===")
            # mylogger.info(f"音频最大码率: {self.config.get('audio_max_bitrate', 128000)/1000:.0f}kbps")
            # mylogger.info(f"音频最小码率: {self.config.get('audio_min_bitrate', 64000)/1000:.0f}kbps")
            # mylogger.info(f"音频采样率: {SAMPLE_RATE}Hz")
            # mylogger.info(f"音频包时长: {AUDIO_PTIME*1000:.0f}ms")
            pass
        
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                # 优化时间戳计算，确保稳定的帧率
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                
                # 改进的帧率控制
                expected_time = self._start + self.current_frame_count * VIDEO_PTIME
                current_time = time.time()
                wait_time = expected_time - current_time
                
                if wait_time > 0:
                    # 使用更精确的睡眠控制
                    await asyncio.sleep(wait_time)
                elif wait_time < -0.1:  # 如果延迟超过100ms，重置时间基准
                    # logger.warning(f"视频帧延迟过大({-wait_time:.3f}s)，重置时间基准")
                    self._start = current_time - self.current_frame_count * VIDEO_PTIME
                    
                # 添加时间戳日志
                # logger.debug(f"[WebRTC] 视频时间戳 - PTS: {self._timestamp}, 等待时间: {wait_time:.3f}s")
                    
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                # mylogger.info('video start:%f',self._start)
                # logger.info(f"[WebRTC] 视频流开始 - 起始时间: {self._start}")
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                # 优化音频时间戳计算
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                
                # 改进的音频帧率控制
                expected_time = self._start + self.current_frame_count * AUDIO_PTIME
                current_time = time.time()
                wait_time = expected_time - current_time
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                elif wait_time < -0.05:  # 如果延迟超过50ms，重置时间基准
                    # logger.warning(f"音频帧延迟过大({-wait_time:.3f}s)，重置时间基准")
                    self._start = current_time - self.current_frame_count * AUDIO_PTIME
                    
                # 添加音频时间戳日志
                # logger.debug(f"[WebRTC] 音频时间戳 - PTS: {self._timestamp}, 等待时间: {wait_time:.3f}s")
                    
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                # mylogger.info('audio start:%f',self._start)
                # logger.info(f"[WebRTC] 音频流开始 - 起始时间: {self._start}")
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # frame = self.frames[self.counter % 30]            
        self._player._start(self)
        
        # 添加详细的WebRTC日志
        # if self.kind == 'video':
        #     logger.debug(f"[WebRTC] 视频帧接收 - 队列大小: {self._queue.qsize()}")
        # else:
        #     logger.debug(f"[WebRTC] 音频帧接收 - 队列大小: {self._queue.qsize()}")
        
        # 添加队列管理
        if self.kind == 'video':
            # 视频队列管理
            if self._queue.qsize() > self.max_queue_size:  # 如果队列过大，丢弃旧帧
                try:
                    # 丢弃一些旧帧以保持实时性
                    for _ in range(self.frame_drop_threshold):
                        self._queue.get_nowait()
                    # logger.warning("视频队列过大，丢弃旧帧")
                    pass
                except asyncio.QueueEmpty:
                    pass
            elif self._queue.qsize() < 2:  # 如果队列过小，可能需要等待
                # logger.debug("视频队列较小，等待更多帧")
                pass
        
        frame,eventpoint = await self._queue.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if eventpoint:
            self._player.notify(eventpoint)
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                avg_fps = self.framecount/self.totaltime
                # mylogger.info(f"------actual avg final fps:{avg_fps:.4f}")
                
                # 添加帧率质量监控
                # if avg_fps < 20:  # 如果帧率过低
                #     mylogger.warning(f"视频帧率过低: {avg_fps:.2f} fps")
                # elif avg_fps > 30:  # 如果帧率过高
                #     mylogger.warning(f"视频帧率过高: {avg_fps:.2f} fps")
                
                # 码率质量建议
                # target_fps = self.config.get('target_fps', 25)
                # if abs(avg_fps - target_fps) > 5:
                #     mylogger.info(f"建议调整码率设置，当前帧率 {avg_fps:.2f} fps，目标帧率 {target_fps} fps")
                
                # 更新码率监控并打印详细信息
                self.bitrate_monitor.update_metrics(avg_fps, self._queue.qsize())
                current_bitrate = self.bitrate_monitor.get_current_bitrate()
                
                # 打印详细的码率信息
                # mylogger.info(f"=== 实时码率监控 ===")
                # mylogger.info(f"当前帧率: {avg_fps:.2f} fps")
                # mylogger.info(f"当前码率: {current_bitrate/1000:.0f}kbps")
                # mylogger.info(f"队列大小: {self._queue.qsize()}")
                # mylogger.info(f"配置码率范围: {self.config.get('min_bitrate', 500000)/1000:.0f}k - {self.config.get('max_bitrate', 2000000)/1000:.0f}k")
                
                # 计算码率使用率
                # max_bitrate = self.config.get('max_bitrate', 2000000)
                # bitrate_usage = (current_bitrate / max_bitrate) * 100
                # mylogger.info(f"码率使用率: {bitrate_usage:.1f}%")
                
                # 性能评估
                # if bitrate_usage > 80:
                #     mylogger.warning(f"码率使用率较高({bitrate_usage:.1f}%)，建议检查网络状况")
                # elif bitrate_usage < 30:
                #     mylogger.info(f"码率使用率较低({bitrate_usage:.1f}%)，可以考虑提高码率以获得更好质量")
                    
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        # logger.info(f"[WebRTC] 停止媒体流轨道 - 类型: {self.kind}")
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None
            # logger.info(f"[WebRTC] 媒体流轨道已停止 - 类型: {self.kind}")

def player_worker_thread(
    quit_event,
    loop,
    container,
    audio_track,
    video_track
):
    container.render(quit_event,loop,audio_track,video_track)

class HumanPlayer:

    def __init__(
        self, nerfreal, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        # 获取推流质量配置
        streaming_config = getattr(nerfreal, 'streaming_quality', {})
        
        self.__audio = PlayerStreamTrack(self, kind="audio", config=streaming_config)
        self.__video = PlayerStreamTrack(self, kind="video", config=streaming_config)

        self.__container = nerfreal

    def notify(self,eventpoint):
        self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        # logger.info(f"[WebRTC] 启动媒体轨道 - 类型: {track.kind}")
        
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            # logger.info("[WebRTC] 创建媒体播放器工作线程")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__audio,
                    self.__video                   
                ),
            )
            self.__thread.start()
            # logger.info("[WebRTC] 媒体播放器工作线程已启动")

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)
        # logger.info(f"[WebRTC] 停止媒体轨道 - 类型: {track.kind}")

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            # logger.info("[WebRTC] 停止媒体播放器工作线程")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None
            self.__log_debug("Worker thread stopped successfully")
            # logger.info("[WebRTC] 媒体播放器工作线程已停止")

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None
            self.__log_debug("Container reference cleared")
            # logger.info("[WebRTC] 容器引用已清除")

    def __log_debug(self, msg: str, *args) -> None:
        # mylogger.debug(f"HumanPlayer {msg}", *args)
        pass
