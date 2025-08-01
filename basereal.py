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

import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS,DoubaoTTS
from logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def play_audio(quit_event,queue):        
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=16000,
        channels=1,
        format=8,
        output=True,
        output_device_index=1,
    )
    stream.start_stream()
    # while queue.qsize() <= 0:
    #     time.sleep(0.1)
    while not quit_event.is_set():
        stream.write(queue.get(block=True))
    stream.close()

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        elif opt.tts == "doubao":
            self.tts = DoubaoTTS(opt,self)
        
        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        # 动作状态管理
        self.curr_state=0  # 当前状态
        self.custom_img_cycle = {}  # 自定义图像序列
        self.custom_audio_cycle = {}  # 自定义音频序列
        self.custom_audio_index = {}  # 自定义音频索引
        self.custom_index = {}  # 自定义索引
        self.custom_opt = {}  # 自定义选项
        # 从配置文件读取自定义动作开关设置
        self.use_custom_silent = getattr(opt, 'use_custom_silent', True)
        # 从配置文件读取静默时使用的动作类型（可以为空）
        self.custom_silent_audiotype = getattr(opt, 'custom_silent_audiotype', "")
        
        # 读取推流质量配置
        self.streaming_quality = getattr(opt, 'streaming_quality', {})
        self.target_fps = self.streaming_quality.get('target_fps', 25.0)
        self.max_video_queue_size = self.streaming_quality.get('max_video_queue_size', 8)
        self.min_video_queue_size = self.streaming_quality.get('min_video_queue_size', 1)
        self.quality_check_interval = self.streaming_quality.get('quality_check_interval', 50)
        self.frame_drop_threshold = self.streaming_quality.get('frame_drop_threshold', 12)
        self.enable_quality_monitoring = self.streaming_quality.get('enable_quality_monitoring', True)
        self.enable_frame_rate_control = self.streaming_quality.get('enable_frame_rate_control', True)
        self.enable_queue_management = self.streaming_quality.get('enable_queue_management', True)
        
        logger.info(f"推流质量配置: 目标帧率={self.target_fps}fps, 最大队列={self.max_video_queue_size}, 最小队列={self.min_video_queue_size}")
        
        self.__loadcustom()

    def put_msg_txt(self,msg,eventpoint=None):
        self.tts.put_msg_txt(msg,eventpoint)
    
    def put_audio_frame(self,audio_chunk,eventpoint=None): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,eventpoint)

    def put_audio_file(self,filebyte): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        # 如果开启了静默自定义动作，只加载指定的动作
        if self.use_custom_silent:
            # 如果指定了具体的动作类型，只加载该动作
            if self.custom_silent_audiotype:
                target_audiotype = self.custom_silent_audiotype
                for item in self.opt.customopt:
                    if item['audiotype'] == target_audiotype:
                        logger.info(f"加载指定静默动作: {item}")
                        input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
                        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                        audiotype = item['audiotype']
                        self.custom_img_cycle[audiotype] = read_imgs(input_img_list)
                        self.custom_audio_cycle[audiotype], sample_rate = sf.read(item['audiopath'], dtype='float32')
                        self.custom_audio_index[audiotype] = 0
                        self.custom_index[audiotype] = 0
                        self.custom_opt[audiotype] = item
                        logger.info(f"成功加载指定静默动作 audiotype={audiotype}")
                        return
                logger.warning(f"未找到指定的静默动作 audiotype={target_audiotype}")
            # 如果没有指定动作类型，加载第一个可用动作
            if self.opt.customopt:
                item = self.opt.customopt[0]
                logger.info(f"加载第一个可用静默动作: {item}")
                input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                audiotype = item['audiotype']
                self.custom_img_cycle[audiotype] = read_imgs(input_img_list)
                self.custom_audio_cycle[audiotype], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[audiotype] = 0
                self.custom_index[audiotype] = 0
                self.custom_opt[audiotype] = item
                logger.info(f"成功加载第一个可用静默动作 audiotype={audiotype}")
                return
        else:
            # 如果未开启静默自定义动作，加载所有动作（保持原有行为）
            for item in self.opt.customopt:
                logger.info(item)
                input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                audiotype = item['audiotype']
                self.custom_img_cycle[audiotype] = read_imgs(input_img_list)
                self.custom_audio_cycle[audiotype], sample_rate = sf.read(item['audiopath'], dtype='float32')
                self.custom_audio_index[audiotype] = 0
                self.custom_index[audiotype] = 0
                self.custom_opt[audiotype] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return

        # 优化编码参数以提高视频质量稳定性
        bitrate = '2000k'      # 提高基础码率确保质量
        crf = 20              # 使用更高质量的CRF值（18-23为高质量范围）
        preset = 'slow'       # 使用slow获得更好的质量稳定性
        maxrate = '3000k'     # 设置更高的最大码率上限
        bufsize = '6000k'     # 增大缓冲区确保稳定性

        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p',
                    '-vcodec', "h264",
                    '-crf', str(crf),  # 优先使用CRF恒定质量模式
                    '-b:v', bitrate,  # 设置视频码率作为参考
                    '-maxrate', maxrate,  # 设置更高的最大码率上限
                    '-bufsize', bufsize,  # 增大缓冲区确保稳定性
                    '-preset', preset,  # 编码速度预设
                    '-profile:v', 'high',  # 使用高质量配置
                    '-level', '4.1',  # H.264标准
                    '-g', '50',  # 关键帧间隔，确保质量稳定
                    '-keyint_min', '25',  # 最小关键帧间隔
                    '-sc_threshold', '0',  # 禁用场景切换检测，保持稳定
                    '-tune', 'zerolatency',  # 优化低延迟
                    '-x264opts', 'no-scenecut=1:keyint=50:min-keyint=25',  # 强制关键帧间隔
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    #'-acodec','pcm_s16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    #'-f' , 'wav',                  
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()  #wait() 
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        cmd_combine_audio = f"ffmpeg -y -i temp{self.opt.sessionid}.aac -i temp{self.opt.sessionid}.mp4 -c:v copy -c:a copy data/record.mp4"
        os.system(cmd_combine_audio) 
        #os.remove(output_path)

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    def get_default_silent_audiotype(self):
        """获取静音时的默认动作类型"""
        # 如果开关开启，查找可用的自定义动作
        if self.use_custom_silent and self.custom_index:
            # 如果指定了具体的动作类型，优先使用指定的
            if self.custom_silent_audiotype and self.custom_silent_audiotype in self.custom_index:
                return self.custom_silent_audiotype
            # 否则返回第一个可用的audiotype
            return list(self.custom_index.keys())[0]
        # 否则返回1（静音状态）
        return 1

    def is_speaking(self):
        """检查当前是否在说话"""
        return getattr(self, 'speaking', False)

    def set_use_custom_silent(self, enabled):
        """设置静音时是否使用自定义动作"""
        self.use_custom_silent = enabled
        print(f"静音时使用自定义动作: {'开启' if enabled else '关闭'}")

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        # 使用配置文件中的参数
        target_fps = self.target_fps
        frame_interval = 1.0 / target_fps
        last_frame_time = time.perf_counter()
        
        # 使用配置文件中的队列管理参数
        max_video_queue_size = self.max_video_queue_size
        min_video_queue_size = self.min_video_queue_size
        
        # 使用配置文件中的质量监控参数
        frame_count = 0
        quality_check_interval = self.quality_check_interval
        
        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            _transition_duration = 0.1  # 进一步减少过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存
            
            # 更激进的缓动函数：快速过渡
            def ease_out_quad(t):
                """二次缓出函数，更快的结束"""
                return 1 - (1 - t) * (1 - t)
            
            def ease_in_quad(t):
                """二次缓入函数，更快的开始"""
                return t * t
        
        if self.opt.transport=='virtualcam':
            import pyvirtualcam
            vircam = None

            audio_tmp = queue.Queue(maxsize=3000)
            audio_thread = Thread(target=play_audio, args=(quit_event,audio_tmp,), daemon=True, name="pyaudio_stream")
            audio_thread.start()
        
        while not quit_event.is_set():
            # 帧率控制 - 使用配置的目标帧率
            if self.enable_frame_rate_control:
                current_time = time.perf_counter()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.perf_counter()
            
            # 实时推流队列管理 - 使用配置的队列大小
            if self.enable_queue_management and video_track and video_track._queue.qsize() > max_video_queue_size:
                # 队列过大时，丢弃一些旧帧以保持实时性
                try:
                    drop_count = min(3, video_track._queue.qsize() - max_video_queue_size + 2)
                    for _ in range(drop_count):
                        if loop:
                            # 使用线程安全的方式调用异步方法
                            asyncio.run_coroutine_threadsafe(
                                video_track._queue.get(), 
                                loop
                            )
                        else:
                            # 如果没有loop，使用同步方式
                            try:
                                video_track._queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                    logger.warning(f"视频队列过大({video_track._queue.qsize()})，丢弃{drop_count}帧")
                except Exception as e:
                    logger.warning(f"队列管理异常: {e}")
                continue
                
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            if enable_transition:
                # 改进的状态检测逻辑
                is_silent = audio_frames[0][1] != 0 and audio_frames[1][1] != 0
                current_speaking = not is_silent
                
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                _last_speaking = current_speaking

            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                # 静音时使用默认的静音动作类型
                audiotype = self.get_default_silent_audiotype()
                # 调试信息：显示当前使用的audiotype
                if hasattr(self, '_last_silent_audiotype') and self._last_silent_audiotype != audiotype:
                    print(f"静音状态切换到audiotype: {audiotype}")
                self._last_silent_audiotype = audiotype
                
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡，使用缓动函数
                    elapsed = time.time() - _transition_start
                    if elapsed < _transition_duration and _last_speaking_frame is not None:
                        t = elapsed / _transition_duration
                        alpha = ease_out_quad(t)  # 使用缓动函数
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡，使用缓动函数
                    elapsed = time.time() - _transition_start
                    if elapsed < _transition_duration and _last_silent_frame is not None:
                        t = elapsed / _transition_duration
                        alpha = ease_in_quad(t)  # 使用缓动函数
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            # 质量监控 - 使用配置的检查间隔
            frame_count += 1
            if self.enable_quality_monitoring and frame_count % quality_check_interval == 0:
                if video_track:
                    queue_size = video_track._queue.qsize()
                    logger.info(f"实时推流质量监控 - 队列大小: {queue_size}, 帧计数: {frame_count}, 目标帧率: {target_fps}fps")
                    
                    # 队列过小时可能存在处理延迟
                    if queue_size < min_video_queue_size:
                        logger.warning("视频队列过小，可能存在处理延迟")

            cv2.putText(combine_frame, "UNIMED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            else: #webrtc
                # 实时推流优化：确保图像质量和传输稳定性
                image = combine_frame
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                
                # 添加帧时间戳以确保同步
                new_frame.pts = int(frame_count * frame_interval * 90000)  # 90kHz时钟
                new_frame.time_base = Fraction(1, 90000)
                
                # 推送到队列 - 使用线程安全的方式调用异步方法
                if loop:
                    asyncio.run_coroutine_threadsafe(
                        video_track._queue.put((new_frame, None)), 
                        loop
                    )
                else:
                    # 如果没有loop，使用同步方式（不推荐，但作为fallback）
                    try:
                        video_track._queue.put_nowait((new_frame, None))
                    except asyncio.QueueFull:
                        # 队列满了，丢弃最旧的帧
                        try:
                            video_track._queue.get_nowait()
                            video_track._queue.put_nowait((new_frame, None))
                        except asyncio.QueueEmpty:
                            pass
                
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                if self.opt.transport=='virtualcam':
                    audio_tmp.put(frame.tobytes()) #TODO
                else: #webrtc
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    # 推送到音频队列 - 使用线程安全的方式调用异步方法
                    if loop:
                        asyncio.run_coroutine_threadsafe(
                            audio_track._queue.put((new_frame, eventpoint)), 
                            loop
                        )
                    else:
                        # 如果没有loop，使用同步方式（不推荐，但作为fallback）
                        try:
                            audio_track._queue.put_nowait((new_frame, eventpoint))
                        except asyncio.QueueFull:
                            # 队列满了，丢弃最旧的帧
                            try:
                                audio_track._queue.get_nowait()
                                audio_track._queue.put_nowait((new_frame, eventpoint))
                            except asyncio.QueueEmpty:
                                pass
                self.record_audio_data(frame)
            if self.opt.transport=='virtualcam':
                vircam.sleep_until_next_frame()
        if self.opt.transport=='virtualcam':
            audio_thread.join()
            vircam.close()
        logger.info('basereal process_frames thread stop') 
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1