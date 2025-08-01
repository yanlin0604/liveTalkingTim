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

logging.basicConfig()
logger = logging.getLogger(__name__)
from logger import logger as mylogger


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
        
        mylogger.info(f"=== 码率自适应调整 ===")
        mylogger.info(f"历史帧率: {[f'{fps:.1f}' for fps in self.fps_history[-3:]]}")
        mylogger.info(f"平均帧率: {avg_fps:.1f} fps")
        mylogger.info(f"目标帧率: {target_fps} fps")
        mylogger.info(f"当前码率: {self.current_bitrate/1000:.0f}kbps")
        
        # 根据帧率调整码率
        if avg_fps < target_fps * 0.8:  # 帧率过低，降低码率
            new_bitrate = max(self.min_bitrate, self.current_bitrate * 0.8)
            mylogger.info(f"帧率过低({avg_fps:.1f}fps < {target_fps*0.8:.1f}fps)，降低码率: {self.current_bitrate/1000:.0f}k -> {new_bitrate/1000:.0f}k")
            self.current_bitrate = new_bitrate
        elif avg_fps > target_fps * 1.2:  # 帧率过高，可以提高码率
            new_bitrate = min(self.max_bitrate, self.current_bitrate * 1.1)
            mylogger.info(f"帧率良好({avg_fps:.1f}fps > {target_fps*1.2:.1f}fps)，提高码率: {self.current_bitrate/1000:.0f}k -> {new_bitrate/1000:.0f}k")
            self.current_bitrate = new_bitrate
        else:
            mylogger.info(f"帧率正常({avg_fps:.1f}fps)，保持当前码率: {self.current_bitrate/1000:.0f}kbps")
    
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
            mylogger.info(f"=== WebRTC视频流配置 ===")
            mylogger.info(f"目标帧率: {self.config.get('target_fps', 25)} fps")
            mylogger.info(f"最大码率: {self.config.get('max_bitrate', 2000000)/1000:.0f}kbps")
            mylogger.info(f"最小码率: {self.config.get('min_bitrate', 500000)/1000:.0f}kbps")
            mylogger.info(f"起始码率: {self.config.get('start_bitrate', 1000000)/1000:.0f}kbps")
            mylogger.info(f"最大队列大小: {self.max_queue_size}")
            mylogger.info(f"帧丢弃阈值: {self.frame_drop_threshold}")
        elif self.kind == 'audio':
            mylogger.info(f"=== WebRTC音频流配置 ===")
            mylogger.info(f"音频最大码率: {self.config.get('audio_max_bitrate', 128000)/1000:.0f}kbps")
            mylogger.info(f"音频最小码率: {self.config.get('audio_min_bitrate', 64000)/1000:.0f}kbps")
            mylogger.info(f"音频采样率: {SAMPLE_RATE}Hz")
            mylogger.info(f"音频包时长: {AUDIO_PTIME*1000:.0f}ms")
        
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
                    logger.warning(f"视频帧延迟过大({-wait_time:.3f}s)，重置时间基准")
                    self._start = current_time - self.current_frame_count * VIDEO_PTIME
                    
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f',self._start)
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
                    logger.warning(f"音频帧延迟过大({-wait_time:.3f}s)，重置时间基准")
                    self._start = current_time - self.current_frame_count * AUDIO_PTIME
                    
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f',self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # frame = self.frames[self.counter % 30]            
        self._player._start(self)
        
        # 添加队列管理
        if self.kind == 'video':
            # 视频队列管理
            if self._queue.qsize() > self.max_queue_size:  # 如果队列过大，丢弃旧帧
                try:
                    # 丢弃一些旧帧以保持实时性
                    for _ in range(self.frame_drop_threshold):
                        self._queue.get_nowait()
                    logger.warning("视频队列过大，丢弃旧帧")
                except asyncio.QueueEmpty:
                    pass
            elif self._queue.qsize() < 2:  # 如果队列过小，可能需要等待
                logger.debug("视频队列较小，等待更多帧")
        
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
                mylogger.info(f"------actual avg final fps:{avg_fps:.4f}")
                
                # 添加帧率质量监控
                if avg_fps < 20:  # 如果帧率过低
                    mylogger.warning(f"视频帧率过低: {avg_fps:.2f} fps")
                elif avg_fps > 30:  # 如果帧率过高
                    mylogger.warning(f"视频帧率过高: {avg_fps:.2f} fps")
                
                # 码率质量建议
                target_fps = self.config.get('target_fps', 25)
                if abs(avg_fps - target_fps) > 5:
                    mylogger.info(f"建议调整码率设置，当前帧率 {avg_fps:.2f} fps，目标帧率 {target_fps} fps")
                
                # 更新码率监控并打印详细信息
                self.bitrate_monitor.update_metrics(avg_fps, self._queue.qsize())
                current_bitrate = self.bitrate_monitor.get_current_bitrate()
                
                # 打印详细的码率信息
                mylogger.info(f"=== 实时码率监控 ===")
                mylogger.info(f"当前帧率: {avg_fps:.2f} fps")
                mylogger.info(f"当前码率: {current_bitrate/1000:.0f}kbps")
                mylogger.info(f"队列大小: {self._queue.qsize()}")
                mylogger.info(f"配置码率范围: {self.config.get('min_bitrate', 500000)/1000:.0f}k - {self.config.get('max_bitrate', 2000000)/1000:.0f}k")
                
                # 计算码率使用率
                max_bitrate = self.config.get('max_bitrate', 2000000)
                bitrate_usage = (current_bitrate / max_bitrate) * 100
                mylogger.info(f"码率使用率: {bitrate_usage:.1f}%")
                
                # 性能评估
                if bitrate_usage > 80:
                    mylogger.warning(f"码率使用率较高({bitrate_usage:.1f}%)，建议检查网络状况")
                elif bitrate_usage < 30:
                    mylogger.info(f"码率使用率较低({bitrate_usage:.1f}%)，可以考虑提高码率以获得更好质量")
                    
                self.framecount = 0
                self.totaltime=0
        return frame
    
    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None

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
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
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

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None
            self.__log_debug("Worker thread stopped successfully")

        if not self.__started and self.__container is not None:
            #self.__container.close()
            self.__container = None
            self.__log_debug("Container reference cleared")

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)
