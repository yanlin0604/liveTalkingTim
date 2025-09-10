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
from av.audio.resampler import AudioResampler

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
        
        # RTMP推流预初始化变量
        self.rtmp_container = None
        self.vstream = None
        self.astream = None
        self.audio_resampler = None
        self.rtmp_width = 0
        self.rtmp_height = 0
        self.video_frame_index = 0
        self.rtmp_initialized = False
        
        # RTMP重连控制变量
        self.rtmp_reconnect_count = 0
        self.rtmp_max_reconnects = 5  # 最大重连次数
        self.rtmp_last_reconnect_time = 0
        self.rtmp_reconnect_delay = 5.0  # 重连延迟（秒）
        self.rtmp_reconnect_backoff = 1.5  # 退避系数
        self.rtmp_connection_failed = False
        self.rtmp_url_in_use = None  # 当前使用的RTMP地址
        self.rtmp_cleanup_timeout = 10.0  # 清理超时时间
        
        # RTMP清晰度配置
        self.rtmp_quality_level = 'high'  # 默认高清
        self.rtmp_quality_configs = {
            'ultra': {  # 蓝光
                'name': '蓝光',
                'bitrate_factor': 1.5,
                'crf': 18,
                'preset': 'slow',
                'audio_bitrate': '192k'
            },
            'high': {  # 高清
                'name': '高清',
                'bitrate_factor': 1.0,
                'crf': 21,
                'preset': 'medium',
                'audio_bitrate': '128k'
            },
            'medium': {  # 普通
                'name': '普通',
                'bitrate_factor': 0.7,
                'crf': 24,
                'preset': 'fast',
                'audio_bitrate': '96k'
            },
            'low': {  # 流畅
                'name': '流畅',
                'bitrate_factor': 0.4,
                'crf': 28,
                'preset': 'faster',
                'audio_bitrate': '64k'
            }
        }
        
        print(f"=== BaseReal初始化TTS ===")
        print(f"opt.tts = {opt.tts}")
        print(f"opt对象: {opt}")
        
        logger.info(f"=== BaseReal初始化TTS ===")
        logger.info(f"opt.tts = {opt.tts}")
        logger.info(f"opt对象: {opt}")
        
        try:
            if opt.tts == "edgetts":
                logger.info("初始化EdgeTTS")
                self.tts = EdgeTTS(opt,self)
            elif opt.tts == "gpt-sovits":
                logger.info("初始化SovitsTTS")
                self.tts = SovitsTTS(opt,self)
            elif opt.tts == "xtts":
                logger.info("初始化XTTS")
                self.tts = XTTS(opt,self)
            elif opt.tts == "cosyvoice":
                logger.info("初始化CosyVoiceTTS")
                self.tts = CosyVoiceTTS(opt,self)
            elif opt.tts == "fishtts":
                logger.info("初始化FishTTS")
                self.tts = FishTTS(opt,self)
            elif opt.tts == "tencent":
                logger.info("初始化TencentTTS")
                self.tts = TencentTTS(opt,self)
            elif opt.tts == "doubao":
                logger.info("初始化DoubaoTTS")
                self.tts = DoubaoTTS(opt,self)
            else:
                logger.error(f"未知的TTS类型: {opt.tts}")
                logger.error("可用的TTS类型: edgetts, gpt-sovits, xtts, cosyvoice, fishtts, tencent, doubao")
                raise ValueError(f"未知的TTS类型: {opt.tts}")
                
            logger.info(f"TTS初始化成功: {type(self.tts).__name__}")
        except Exception as e:
            logger.error(f"TTS初始化失败: {e}")
            logger.exception("TTS初始化异常详情")
            raise
        
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
        
        # 多动作编排配置
        self.multi_action_mode = getattr(opt, 'multi_action_mode', 'single')  # single/random/sequence
        self.multi_action_list = getattr(opt, 'multi_action_list', [])  # 动作列表
        self.multi_action_interval = getattr(opt, 'multi_action_interval', 0)  # 动作切换间隔（帧数）
        # 新增：动作切换策略（interval=按帧间隔；on_complete=播放完整循环后切换）
        self.multi_action_switch_policy = getattr(opt, 'multi_action_switch_policy', 'interval')
        self.current_action_index = 0  # 当前动作索引（用于sequence模式）
        self.action_switch_counter = 0  # 动作切换计数器
        self.current_silent_audiotype = None  # 当前使用的静默动作类型
        
        # 记录静默自定义动作配置
        logger.info("=== 静默自定义动作配置 ===")
        logger.info(f"静默自定义动作开关: {'开启' if self.use_custom_silent else '关闭'}")
        logger.info(f"指定静默动作类型: {self.custom_silent_audiotype or '未指定'}")
        logger.info(f"多动作模式: {self.multi_action_mode}")
        logger.info(f"多动作列表: {self.multi_action_list}")
        logger.info(f"动作切换间隔: {self.multi_action_interval}帧")
        logger.info(f"动作切换策略: {self.multi_action_switch_policy}")
        logger.info(f"可用自定义动作配置数量: {len(opt.customopt) if hasattr(opt, 'customopt') and opt.customopt else 0}")
        
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
        
        # RTMP推流性能统计
        self.rtmp_stats = {
            'start_time': None,
            'total_frames': 0,
            'dropped_frames': 0,
            'encoding_errors': 0,
            'audio_errors': 0,
            'reconnections': 0,
            'avg_fps': 0.0,
            'current_bitrate': '0k',
            'last_stats_time': time.time(),
            'frame_times': [],
            'encoding_times': []
        }
        
        logger.info(f"推流质量配置: 目标帧率={self.target_fps}fps, 最大队列={self.max_video_queue_size}, 最小队列={self.min_video_queue_size}")
        
        # 加载自定义动作配置
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
        """加载自定义动作配置"""
        logger.info("=== 开始加载自定义动作配置 ===")
        logger.info(f"静默自定义动作开关: {'开启' if self.use_custom_silent else '关闭'}")
        logger.info(f"多动作模式: {self.multi_action_mode}")
        logger.info(f"多动作列表: {self.multi_action_list}")
        logger.info(f"可用自定义动作数量: {len(self.opt.customopt) if self.opt.customopt else 0}")
        
        # 如果开启了静默自定义动作
        if self.use_custom_silent:
            logger.info("静默自定义动作已开启，开始加载动作")
            
            # 根据多动作模式决定要加载的动作
            actions_to_load = []
            
            if self.multi_action_mode in ['random', 'sequence'] and self.multi_action_list:
                # 多动作模式：加载指定的多个动作
                logger.info(f"多动作模式({self.multi_action_mode})，加载动作列表: {self.multi_action_list}")
                actions_to_load = self.multi_action_list
            elif self.custom_silent_audiotype:
                # 单动作模式：加载指定的单个动作
                logger.info(f"单动作模式，加载指定动作: {self.custom_silent_audiotype}")
                actions_to_load = [self.custom_silent_audiotype]
            else:
                # 默认加载第一个可用动作
                if self.opt.customopt:
                    first_action = self.opt.customopt[0].get('audiotype')
                    logger.info(f"未指定动作，加载第一个可用动作: {first_action}")
                    actions_to_load = [first_action]
            
            # 加载所有需要的动作
            loaded_count = 0
            for target_audiotype in actions_to_load:
                logger.info(f"正在加载动作: {target_audiotype}")
                
                for item in self.opt.customopt:
                    if item['audiotype'] == target_audiotype:
                        logger.info(f"找到匹配的动作配置: {item}")
                        
                        try:
                            # 加载图像文件
                            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
                            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                            logger.info(f"找到图像文件数量: {len(input_img_list)}")
                            
                            audiotype = item['audiotype']
                            self.custom_img_cycle[audiotype] = read_imgs(input_img_list)
                            logger.info(f"成功加载图像帧数: {len(self.custom_img_cycle[audiotype])}")
                            
                            # 加载音频文件
                            self.custom_audio_cycle[audiotype], sample_rate = sf.read(item['audiopath'], dtype='float32')
                            logger.info(f"成功加载音频文件: 采样率={sample_rate}Hz, 时长={len(self.custom_audio_cycle[audiotype])/sample_rate:.2f}秒")
                            
                            # 初始化索引
                            self.custom_audio_index[audiotype] = 0
                            self.custom_index[audiotype] = 0
                            self.custom_opt[audiotype] = item
                            
                            loaded_count += 1
                            logger.info(f"✅ 成功加载动作 audiotype={audiotype}")
                            break
                        except Exception as e:
                            logger.error(f"加载动作 {target_audiotype} 失败: {e}")
                else:
                    logger.warning(f"❌ 未找到动作 audiotype={target_audiotype}")
            
            logger.info(f"成功加载 {loaded_count}/{len(actions_to_load)} 个动作")
            logger.info(f"当前已加载的自定义动作: {list(self.custom_index.keys())}")
            
            # 初始化第一个动作
            if self.custom_index:
                if self.multi_action_mode == 'random':
                    import random
                    self.current_silent_audiotype = random.choice(list(self.custom_index.keys()))
                else:
                    self.current_silent_audiotype = list(self.custom_index.keys())[0]
                logger.info(f"初始动作设置为: {self.current_silent_audiotype}")
        else:
            logger.info("静默自定义动作未开启，跳过自定义动作加载")
        
        logger.info("=== 自定义动作配置加载完成 ===")
        # else:
        #     # 如果未开启静默自定义动作，加载所有动作（保持原有行为）
        #     for item in self.opt.customopt:
        #         logger.info(item)
        #         input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
        #         input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        #         audiotype = item['audiotype']
        #         self.custom_img_cycle[audiotype] = read_imgs(input_img_list)
        #         self.custom_audio_cycle[audiotype], sample_rate = sf.read(item['audiopath'], dtype='float32')
        #         self.custom_audio_cycle[audiotype], sample_rate = sf.read(item['audiopath'], dtype='float32')
        #         self.custom_audio_index[audiotype] = 0
        #         self.custom_index[audiotype] = 0
        #         self.custom_opt[audiotype] = item

    def init_customindex(self):
        """初始化自定义动作索引"""
        logger.info("🔄 初始化自定义动作索引")
        old_state = self.curr_state
        self.curr_state = 0
        logger.debug(f"状态重置: {old_state} → {self.curr_state}")
        
        # 重置音频索引
        for key in self.custom_audio_index:
            old_audio_index = self.custom_audio_index[key]
            self.custom_audio_index[key] = 0
            logger.debug(f"重置音频索引 audiotype={key}: {old_audio_index} → 0")
        
        # 重置视频索引
        for key in self.custom_index:
            old_video_index = self.custom_index[key]
            self.custom_index[key] = 0
            logger.debug(f"重置视频索引 audiotype={key}: {old_video_index} → 0")
        
        logger.info("✅ 自定义动作索引初始化完成")

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
        """获取自定义动作的音频流"""
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        
        logger.debug(f"获取音频流 audiotype={audiotype}: 索引{idx}→{self.custom_audio_index[audiotype]}, 音频长度={len(stream)}")
        
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            old_state = self.curr_state
            self.curr_state = 1  # 当前视频不循环播放，切换到静音状态
            logger.info(f"🎵 自定义动作音频播放完成 audiotype={audiotype}, 状态切换: {old_state} → {self.curr_state}")
        
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        """设置自定义动作状态"""
        logger.info(f"🔄 设置自定义动作状态: audiotype={audiotype}, reinit={reinit}")
        print('set_custom_state:',audiotype)
        
        if self.custom_audio_index.get(audiotype) is None:
            logger.warning(f"❌ 指定的audiotype={audiotype}不存在，可用类型: {list(self.custom_audio_index.keys())}")
            return
        
        old_state = self.curr_state
        self.curr_state = audiotype
        logger.info(f"状态切换: {old_state} → {audiotype}")
        
        if reinit:
            old_audio_index = self.custom_audio_index[audiotype]
            old_video_index = self.custom_index[audiotype]
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
            logger.info(f"重置索引 audiotype={audiotype}: 音频{old_audio_index}→0, 视频{old_video_index}→0")
        else:
            logger.debug(f"保持当前索引 audiotype={audiotype}: 音频={self.custom_audio_index[audiotype]}, 视频={self.custom_index[audiotype]}")
    def get_default_silent_audiotype(self):
        """获取静音时的默认动作类型（支持多动作编排）"""
        logger.debug(f"获取默认静默动作类型 - 开关状态: {self.use_custom_silent}, 可用动作: {list(self.custom_index.keys()) if self.custom_index else '无'}")
        
        # 如果开关开启，查找可用的自定义动作
        if self.use_custom_silent and self.custom_index:
            # 多动作编排：random / sequence
            if self.multi_action_mode in ('random', 'sequence') and len(self.custom_index) > 1:
                policy = getattr(self, 'multi_action_switch_policy', 'interval')
                
                # 初始化当前动作
                if not self.current_silent_audiotype:
                    self.current_silent_audiotype = list(self.custom_index.keys())[0]
                    logger.debug(f"初始化当前静默动作: {self.current_silent_audiotype}")
                    return self.current_silent_audiotype
                
                if policy == 'on_complete':
                    # 完整循环后切换：当 index % size == 0 且 index>0 视为完成一轮
                    cur = self.current_silent_audiotype
                    if cur in self.custom_img_cycle and cur in self.custom_index:
                        size = len(self.custom_img_cycle[cur])
                        idx = self.custom_index[cur]
                        if size > 0 and idx > 0 and (idx % size == 0):
                            if self.multi_action_mode == 'random':
                                import random
                                candidates = list(self.custom_index.keys())
                                if cur in candidates and len(candidates) > 1:
                                    candidates.remove(cur)
                                self.current_silent_audiotype = random.choice(candidates)
                                logger.info(f"🎲[on_complete] 随机切换到动作: {self.current_silent_audiotype}")
                            else:
                                seq = list(self.custom_index.keys())
                                self.current_action_index = (self.current_action_index + 1) % len(seq)
                                self.current_silent_audiotype = seq[self.current_action_index]
                                logger.info(f"📝[on_complete] 顺序切换到动作: {self.current_silent_audiotype} (索引: {self.current_action_index})")
                    return self.current_silent_audiotype
                else:
                    # interval 策略（按帧间隔）
                    if self.multi_action_mode == 'random':
                        if self.action_switch_counter >= self.multi_action_interval:
                            import random
                            candidates = list(self.custom_index.keys())
                            if self.current_silent_audiotype in candidates and len(candidates) > 1:
                                candidates.remove(self.current_silent_audiotype)
                            self.current_silent_audiotype = random.choice(candidates)
                            self.action_switch_counter = 0
                            logger.info(f"🎲 随机切换到动作: {self.current_silent_audiotype}")
                        else:
                            self.action_switch_counter += 1
                        return self.current_silent_audiotype
                    else:
                        if self.action_switch_counter >= self.multi_action_interval:
                            seq = list(self.custom_index.keys())
                            self.current_action_index = (self.current_action_index + 1) % len(seq)
                            self.current_silent_audiotype = seq[self.current_action_index]
                            self.action_switch_counter = 0
                            logger.info(f"📝 顺序切换到动作: {self.current_silent_audiotype} (索引: {self.current_action_index})")
                        else:
                            self.action_switch_counter += 1
                        return self.current_silent_audiotype
            
            # 单动作模式或只有一个动作
            if self.current_silent_audiotype and self.current_silent_audiotype in self.custom_index:
                return self.current_silent_audiotype
            elif self.custom_silent_audiotype and self.custom_silent_audiotype in self.custom_index:
                logger.debug(f"使用指定的静默动作类型: {self.custom_silent_audiotype}")
                return self.custom_silent_audiotype
            else:
                default_audiotype = list(self.custom_index.keys())[0]
                logger.debug(f"使用第一个可用静默动作类型: {default_audiotype}")
                return default_audiotype
        
        # 否则返回1（静音状态）
        logger.debug("使用默认静音状态 (audiotype=1)")
        return 1

    def is_speaking(self):
        """检查当前是否在说话"""
        return getattr(self, 'speaking', False)

    def set_use_custom_silent(self, enabled):
        """设置静音时是否使用自定义动作"""
        old_status = self.use_custom_silent
        self.use_custom_silent = enabled
        logger.info(f"静默自定义动作状态变更: {'开启' if old_status else '关闭'} → {'开启' if enabled else '关闭'}")
        print(f"静音时使用自定义动作: {'开启' if enabled else '关闭'}")

    def set_custom_silent_audiotype(self, audiotype: str):
        """运行时设置静默自定义动作类型，并重新加载配置

        参数：
            audiotype: 目标动作类型（字符串，可为空字符串表示不指定）
        """
        old_type = getattr(self, 'custom_silent_audiotype', "")
        self.custom_silent_audiotype = audiotype or ""
        logger.info(f"静默动作类型变更: {old_type or '未指定'} → {self.custom_silent_audiotype or '未指定'}")
        # 重新加载自定义动作配置以应用新的选择
        try:
            self.__loadcustom()
            logger.info("静默自定义动作配置已重新加载")
        except Exception as e:
            logger.error(f"重新加载自定义动作失败: {e}")

    def reload_custom_actions(self):
        """对外暴露的重新加载自定义动作配置接口"""
        logger.info("收到请求：重新加载自定义动作配置")
        try:
            self.__loadcustom()
            logger.info("自定义动作配置重新加载完成")
        except Exception as e:
            logger.error(f"重新加载自定义动作失败: {e}")

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
        elif self.opt.transport=='rtmp':
            # RTMP推流预初始化（提前准备，减少首帧延迟）
            self._prepare_rtmp_streams()
        
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
                    logger.info(f"🔄 静音状态切换到audiotype: {audiotype}")
                    print(f"静音状态切换到audiotype: {audiotype}")
                self._last_silent_audiotype = audiotype
                
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    logger.debug(f"使用自定义静默动作 audiotype={audiotype}, 当前索引={self.custom_index[audiotype]}")
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                    logger.debug(f"自定义静默动作帧索引: {mirindex}/{len(self.custom_img_cycle[audiotype])}")
                else:
                    logger.debug(f"使用默认静默帧 audiotype={audiotype}, 帧索引={idx}")
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
                    # logger.info(f"实时推流质量监控 - 队列大小: {queue_size}, 帧计数: {frame_count}, 目标帧率: {target_fps}fps")
                    
                    # 队列过小时可能存在处理延迟
                    # if queue_size < min_video_queue_size:
                    #     logger.warning("视频队列过小，可能存在处理延迟")

            cv2.putText(combine_frame, "UNIMED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            elif self.opt.transport=='rtmp':
                # 零延迟RTMP推流处理
                if not self.rtmp_initialized:
                    self._initialize_rtmp_with_frame(combine_frame, target_fps)
                
                # 检查RTMP连接状态，如果连接彻底失败则跳过推流
                if self.rtmp_connection_failed and self.rtmp_reconnect_count >= self.rtmp_max_reconnects:
                    continue  # 跳过这一帧，避免无效操作
                
                if self.rtmp_initialized and not self.rtmp_connection_failed:
                    # 高效视频帧编码和推送
                    try:
                        # 获取当前队列大小用于自适应编码
                        current_queue_size = video_track._queue.qsize() if video_track else 0
                        
                        # 优化视频帧创建
                        vframe = VideoFrame.from_ndarray(combine_frame, format='bgr24')
                        if vframe.format.name != 'yuv420p':
                            vframe = vframe.reformat(format='yuv420p')
                        
                        vframe.pts = self.video_frame_index
                        vframe.time_base = Fraction(1, int(target_fps))
                        self.video_frame_index += 1
                        
                        # 批量编码减少系统调用
                        packets = list(self.vstream.encode(vframe))
                        for packet in packets:
                            self.rtmp_container.mux(packet)
                        
                        # 更新性能统计
                        self._update_rtmp_stats(True, False)
                        
                        # 每100帧检查一次编码性能并动态调整
                        if self.video_frame_index % 100 == 0:
                            self._check_and_adjust_encoding_quality(current_queue_size, target_fps)
                            self._log_rtmp_performance_stats()
                            
                    except (BrokenPipeError, ConnectionResetError, OSError) as e:
                        self._handle_rtmp_connection_error(f"RTMP 连接异常: {e}")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "broken pipe" in error_msg or "connection reset" in error_msg or "connection refused" in error_msg:
                            self._handle_rtmp_connection_error(f"RTMP 连接异常: {e}")
                        else:
                            # 其他编码异常，记录但不重连
                            self._update_rtmp_stats(False, False)
                            if self.rtmp_reconnect_count == 0:  # 只在首次出现时记录详细日志
                                logger.warning(f"RTMP 视频编码异常: {e}")
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
                elif self.opt.transport=='rtmp':
                    # 优化的音频处理
                    # 检查RTMP连接状态，如果连接彻底失败则跳过音频推流
                    if self.rtmp_connection_failed and self.rtmp_reconnect_count >= self.rtmp_max_reconnects:
                        continue  # 跳过音频处理，避免无效操作
                    
                    if self.rtmp_initialized and not self.rtmp_connection_failed:
                        success = self._optimize_audio_encoding(frame)
                        if not success:
                            # 音频编码失败时的处理
                            pass
                    else:
                        # 容器未初始化时跳过音频
                        pass
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
        if self.opt.transport=='rtmp':
            # 优雅关闭RTMP推流
            self._cleanup_rtmp()
        
        # 优雅停止：清理资源并记录日志
        logger.info('basereal process_frames thread stop - 优雅清理完成')
        
        # 清理队列中的剩余数据
        try:
            while not self.res_frame_queue.empty():
                self.res_frame_queue.get_nowait()
            logger.info('🧹 清理剩余帧队列数据')
        except:
            pass
            
        # 清理音频特征队列
        if hasattr(self, 'asr') and hasattr(self.asr, 'feat_queue'):
            try:
                while not self.asr.feat_queue.empty():
                    self.asr.feat_queue.get_nowait()
                logger.info('🧹 清理音频特征队列')
            except:
                pass
    
    def _prepare_rtmp_streams(self):
        """预初始化RTMP推流资源，减少首帧延迟"""
        try:
            logger.info("🚀 开始预初始化RTMP推流资源")
            
            # 预设默认分辨率，实际会在首帧时更新
            self.rtmp_width = 512
            self.rtmp_height = 512
            
            # 预创建容器但不立即打开连接
            logger.info("✅ RTMP推流资源预初始化完成")
            
        except Exception as e:
            logger.error(f"❌ RTMP推流资源预初始化失败: {e}")
    
    def _check_rtmp_url_availability(self, rtmp_url):
        """检查RTMP地址是否可用（避免重复推流）"""
        # 检查是否已经在使用同一地址
        if self.rtmp_url_in_use == rtmp_url:
            logger.warning(f"⚠️ RTMP地址已在使用: {rtmp_url}")
            return False
            
        return True
    
    def _initialize_rtmp_with_frame(self, frame, fps):
        """使用首帧初始化RTMP推流，实现零延迟启动"""
        try:
            if self.rtmp_initialized:
                return
                
            height, width, _ = frame.shape
            self.rtmp_width = width
            self.rtmp_height = height
            
            # 打开RTMP容器
            rtmp_url = getattr(self.opt, 'push_url', 'rtmp://localhost/live/stream')
            
            # 检查地址可用性
            if not self._check_rtmp_url_availability(rtmp_url):
                logger.error(f"❌ RTMP地址不可用，取消初始化: {rtmp_url}")
                return
            
            logger.info(f"🎬 使用首帧初始化RTMP推流: {width}x{height}@{fps}fps")
            logger.info(f"📍 占用RTMP地址: {rtmp_url}")
            self.rtmp_container = av.open(rtmp_url, 'w', format='flv')
            
            # 优化的视频编码参数配置
            video_codec = 'libx264'
            
            # 根据清晰度级别和分辨率动态调整编码参数
            quality_config = self.rtmp_quality_configs.get(self.rtmp_quality_level, self.rtmp_quality_configs['high'])
            
            # 基础码率根据分辨率确定
            if width * height <= 640 * 480:  # 低分辨率
                base_bitrate = 1500
            elif width * height <= 1280 * 720:  # 中分辨率
                base_bitrate = 2500
            else:  # 高分辨率
                base_bitrate = 4000
            
            # 应用清晰度因子
            actual_bitrate = int(base_bitrate * quality_config['bitrate_factor'])
            bitrate = f'{actual_bitrate}k'
            maxrate = f'{int(actual_bitrate * 1.4)}k'
            bufsize = f'{int(actual_bitrate * 2)}k'
            crf = quality_config['crf']
            preset = quality_config['preset']
            
            # 创建视频流
            self.vstream = self.rtmp_container.add_stream(video_codec, rate=fps)
            self.vstream.width = width
            self.vstream.height = height
            self.vstream.pix_fmt = 'yuv420p'
            
            # 优化的视频编码选项
            self.vstream.options = {
                'crf': str(crf),                    # 恒定质量因子
                'preset': preset,                   # 编码速度预设
                'tune': 'zerolatency',             # 零延迟优化
                'profile': 'high',                 # H.264高质量配置
                'level': '4.1',                    # H.264标准级别
                'b:v': bitrate,                    # 目标码率
                'maxrate': maxrate,                # 最大码率
                'bufsize': bufsize,                # 缓冲区大小
                'g': str(int(fps * 2)),           # GOP大小（2秒关键帧间隔）
                'keyint_min': str(int(fps)),      # 最小关键帧间隔
                'sc_threshold': '0',               # 禁用场景切换检测
                'bf': '0',                         # 禁用B帧提升实时性
                'refs': '1',                       # 参考帧数量
                'me_method': 'hex',                # 运动估计方法
                'subq': '6',                       # 子像素运动估计质量
                'trellis': '1',                    # Trellis量化
                'fast_pskip': '1',                 # 快速P帧跳过
                'flags': '+cgop',                  # 封闭GOP
                'x264opts': f'no-scenecut=1:keyint={int(fps * 2)}:min-keyint={int(fps)}:bframes=0'
            }
            
            # 创建音频流
            self.astream = self.rtmp_container.add_stream('aac', rate=44100)
            self.astream.channels = 1
            self.astream.layout = 'mono'
            
            # 优化的音频编码选项
            audio_bitrate = quality_config['audio_bitrate']
            self.astream.options = {
                'b:a': audio_bitrate,              # 音频码率
                'profile:a': 'aac_low',           # AAC低复杂度配置
                'ar': '44100',                     # 采样率
                'ac': '1'                          # 单声道
            }
            
            # 创建音频重采样器
            self.audio_resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=44100
            )
            
            self.video_frame_index = 0
            self.rtmp_initialized = True
            
            # 记录当前使用的地址
            self.rtmp_url_in_use = rtmp_url
            
            # 重置重连状态（连接成功后）
            self._reset_rtmp_reconnect_state()
            
            logger.info(f"✅ RTMP推流初始化成功: {rtmp_url}")
            logger.info(f"📹 视频参数: {width}x{height}@{fps}fps, 码率={bitrate}, CRF={crf}, 预设={preset}")
            logger.info(f"🎵 音频参数: AAC 44.1kHz 单声道 {audio_bitrate}")
            logger.info(f"🎯 清晰度级别: {quality_config['name']} ({self.rtmp_quality_level})")
            
        except Exception as e:
            logger.error(f"❌ RTMP推流初始化失败: {e}")
            self.rtmp_initialized = False
            self.rtmp_connection_failed = True  # 标记连接失败
            self._cleanup_rtmp()
    
    def _handle_rtmp_connection_error(self, error_msg):
        """处理RTMP连接错误，包含重连控制逻辑"""
        import time
        
        current_time = time.time()
        
        # 检查是否超过最大重连次数
        if self.rtmp_reconnect_count >= self.rtmp_max_reconnects:
            if not self.rtmp_connection_failed:
                logger.error(f"❌ RTMP重连次数已达上限({self.rtmp_max_reconnects})，执行完全重置")
                # 参考/rtmp/stop接口的重置逻辑，执行完全重置
                self._complete_rtmp_reset()
            return
        
        # 检测连续快速失败（在很短时间内多次失败）
        if (current_time - self.rtmp_last_reconnect_time < 2.0 and 
            self.rtmp_reconnect_count >= 2):
            # 连续快速失败，增加更长的冷却期
            extended_delay = self.rtmp_reconnect_delay * 3
            logger.warning(f"⚠️ 检测到连续快速失败，延长冷却期至 {extended_delay:.1f} 秒")
            self.rtmp_reconnect_delay = extended_delay
        
        # 检查重连间隔
        if current_time - self.rtmp_last_reconnect_time < self.rtmp_reconnect_delay:
            return  # 还在冷却期，不执行重连
        
        # 执行重连
        self.rtmp_reconnect_count += 1
        self.rtmp_last_reconnect_time = current_time
        
        # 只在前几次重连时输出详细日志
        if self.rtmp_reconnect_count <= 3:
            logger.warning(f"🔄 {error_msg}，第{self.rtmp_reconnect_count}次重连尝试")
        
        self._update_rtmp_stats(False, False)
        self._reinitialize_rtmp()
        
        # 增加重连延迟（指数退避），但设置上限
        self.rtmp_reconnect_delay = min(
            self.rtmp_reconnect_delay * self.rtmp_reconnect_backoff, 
            30.0  # 最大延迟30秒
        )
    
    def _reinitialize_rtmp(self):
        """重新初始化RTMP推流连接"""
        import time
        
        try:
            # 记录重连统计
            self.rtmp_stats['reconnections'] += 1
            
            # 强制清理之前的连接，确保完全断开
            logger.info("🔄 强制断开之前的RTMP连接")
            self._cleanup_rtmp(force_cleanup=True)
            
            # 等待更长时间确保连接完全释放，特别是在连续失败的情况下
            sleep_time = min(2.0 + (self.rtmp_reconnect_count * 0.5), 5.0)
            time.sleep(sleep_time)
            
            # 重置初始化状态，等待下一帧触发重新初始化
            self.rtmp_initialized = False
            self.video_frame_index = 0
            
            # 重要：暂时重置连接失败状态，给重连一个机会
            self.rtmp_connection_failed = False
            
            logger.info("✅ RTMP重连准备完成，等待下一帧触发重新初始化")
            
        except Exception as e:
            logger.error(f"❌ RTMP推流重新初始化失败: {e}")
            self.rtmp_stats['encoding_errors'] += 1
            # 即使出错也要强制清理
            self._cleanup_rtmp(force_cleanup=True)
            # 标记连接失败
            self.rtmp_connection_failed = True
    
    def _reset_rtmp_reconnect_state(self):
        """重置RTMP重连状态（连接成功后调用）"""
        if self.rtmp_reconnect_count > 0:
            logger.info(f"🎉 RTMP连接恢复正常，重置重连状态（之前重连{self.rtmp_reconnect_count}次）")
        
        self.rtmp_reconnect_count = 0
        self.rtmp_reconnect_delay = 5.0  # 重置为初始延迟
        self.rtmp_connection_failed = False
        self.rtmp_last_reconnect_time = 0
    
    def _complete_rtmp_reset(self):
        """完全重置RTMP状态（参考/rtmp/stop接口的重置逻辑）"""
        logger.warning("🔄 执行完全RTMP重置，参考stop接口逻辑")
        
        try:
            # 1. 强制清理所有RTMP资源
            self._cleanup_rtmp(force_cleanup=True)
            
            # 2. 重置所有RTMP相关状态变量
            self.rtmp_container = None
            self.vstream = None
            self.astream = None
            self.audio_resampler = None
            self.rtmp_width = 0
            self.rtmp_height = 0
            self.video_frame_index = 0
            self.rtmp_initialized = False
            
            # 3. 重置重连控制变量到初始状态
            self.rtmp_reconnect_count = 0
            self.rtmp_max_reconnects = 5
            self.rtmp_last_reconnect_time = 0
            self.rtmp_reconnect_delay = 5.0
            self.rtmp_reconnect_backoff = 1.5
            self.rtmp_connection_failed = False
            self.rtmp_url_in_use = None
            
            # 4. 重置统计信息
            if hasattr(self, 'rtmp_stats'):
                self.rtmp_stats = {
                    'frames_sent': 0,
                    'frames_dropped': 0,
                    'audio_frames_sent': 0,
                    'encoding_errors': 0,
                    'reconnections': 0,
                    'last_fps': 0,
                    'avg_encoding_time': 0,
                    'connection_start_time': 0
                }
            
            logger.info("✅ RTMP完全重置完成，所有状态已恢复到初始状态")
            
        except Exception as e:
            logger.error(f"❌ RTMP完全重置失败: {e}")
            # 即使重置失败，也要确保关键状态被重置
            self.rtmp_initialized = False
            self.rtmp_connection_failed = True
    
    def reset_rtmp_connection(self):
        """外部调用的RTMP连接重置方法（类似/rtmp/stop的效果）"""
        logger.info("🔄 外部触发RTMP连接重置")
        self._complete_rtmp_reset()
        return {
            'success': True,
            'message': 'RTMP连接已完全重置，可以重新开始推流'
        }
    
    def _cleanup_rtmp(self, force_cleanup=False):
        """清理RTMP推流资源"""
        import time
        import threading
        
        try:
            if force_cleanup:
                logger.warning("🔥 强制清理RTMP推流资源")
            else:
                logger.info("🧹 开始清理RTMP推流资源")
            
            cleanup_start_time = time.time()
            
            # 发送结束包（优雅关闭）
            if not force_cleanup and self.rtmp_initialized and self.vstream and self.astream:
                try:
                    # 设置超时机制
                    def flush_encoders():
                        try:
                            # 刷新视频编码器
                            for packet in self.vstream.encode(None):
                                self.rtmp_container.mux(packet)
                            
                            # 刷新音频编码器
                            for packet in self.astream.encode(None):
                                self.rtmp_container.mux(packet)
                        except Exception as e:
                            logger.warning(f"编码器刷新异常: {e}")
                    
                    # 使用线程执行刷新，避免阻塞
                    flush_thread = threading.Thread(target=flush_encoders, daemon=True)
                    flush_thread.start()
                    flush_thread.join(timeout=3.0)  # 最多等待3秒
                    
                    if flush_thread.is_alive():
                        logger.warning("编码器刷新超时，强制继续清理")
                        
                except Exception as e:
                    logger.warning(f"RTMP编码器刷新异常: {e}")
            
            # 强制关闭容器
            if self.rtmp_container:
                try:
                    # 设置关闭超时
                    def close_container():
                        try:
                            self.rtmp_container.close()
                        except Exception as e:
                            logger.warning(f"容器关闭异常: {e}")
                    
                    close_thread = threading.Thread(target=close_container, daemon=True)
                    close_thread.start()
                    close_thread.join(timeout=5.0)  # 最多等待5秒
                    
                    if close_thread.is_alive():
                        logger.warning("RTMP容器关闭超时，强制释放资源")
                    
                except Exception as e:
                    logger.warning(f"RTMP容器关闭异常: {e}")
            
            # 强制重置所有状态
            self.rtmp_container = None
            self.vstream = None
            self.astream = None
            self.audio_resampler = None
            self.rtmp_initialized = False
            self.video_frame_index = 0
            
            # 清理完成后释放地址占用
            if self.rtmp_url_in_use:
                logger.info(f"📍 释放RTMP地址占用: {self.rtmp_url_in_use}")
                self.rtmp_url_in_use = None
            
            cleanup_time = time.time() - cleanup_start_time
            logger.info(f"✅ RTMP推流资源清理完成 (耗时: {cleanup_time:.2f}秒)")
            
        except Exception as e:
            logger.error(f"❌ RTMP推流资源清理异常: {e}")
            # 即使出错也要重置状态
            self.rtmp_container = None
            self.vstream = None
            self.astream = None
            self.audio_resampler = None
            self.rtmp_initialized = False
            self.video_frame_index = 0
            self.rtmp_url_in_use = None
    
    def _get_adaptive_encoding_params(self, width, height, current_fps, queue_size=0):
        """根据分辨率、帧率和队列状态自适应调整编码参数"""
        pixel_count = width * height
        
        # 基础参数配置
        if pixel_count <= 320 * 240:  # 极低分辨率
            base_bitrate = 800
            base_crf = 25
            preset = 'ultrafast'
        elif pixel_count <= 640 * 480:  # 低分辨率
            base_bitrate = 1200
            base_crf = 23
            preset = 'faster'
        elif pixel_count <= 1280 * 720:  # 中分辨率
            base_bitrate = 2000
            base_crf = 21
            preset = 'medium'
        elif pixel_count <= 1920 * 1080:  # 高分辨率
            base_bitrate = 3500
            base_crf = 19
            preset = 'medium'
        else:  # 超高分辨率
            base_bitrate = 5000
            base_crf = 18
            preset = 'slow'
        
        # 根据帧率调整码率
        fps_factor = min(current_fps / 25.0, 1.5)  # 最大1.5倍调整
        adjusted_bitrate = int(base_bitrate * fps_factor)
        
        # 根据队列状态动态调整（如果队列过大，降低质量提升编码速度）
        if queue_size > 10:
            preset = 'ultrafast'
            base_crf = min(base_crf + 3, 28)  # 降低质量
            adjusted_bitrate = int(adjusted_bitrate * 0.8)  # 降低码率
        elif queue_size > 6:
            preset = 'faster'
            base_crf = min(base_crf + 1, 25)
            adjusted_bitrate = int(adjusted_bitrate * 0.9)
        
        return {
            'bitrate': f'{adjusted_bitrate}k',
            'maxrate': f'{int(adjusted_bitrate * 1.5)}k',
            'bufsize': f'{int(adjusted_bitrate * 2)}k',
            'crf': base_crf,
            'preset': preset
        }
    
    def _check_and_adjust_encoding_quality(self, queue_size, current_fps):
        """检查并动态调整编码质量以优化性能"""
        try:
            if not self.rtmp_initialized or not self.vstream:
                return
            
            # 获取当前编码参数
            current_params = self._get_adaptive_encoding_params(
                self.rtmp_width, self.rtmp_height, current_fps, queue_size
            )
            
            # 检查是否需要调整编码参数
            needs_adjustment = False
            
            # 队列过大时需要降低质量
            if queue_size > self.max_video_queue_size:
                needs_adjustment = True
                logger.info(f"📊 检测到队列过大({queue_size})，动态调整编码参数以提升性能")
            
            # 队列很小且性能良好时可以提升质量
            elif queue_size < self.min_video_queue_size and hasattr(self, '_last_adjustment_time'):
                time_since_last = time.time() - self._last_adjustment_time
                if time_since_last > 30:  # 30秒后才考虑提升质量
                    needs_adjustment = True
                    logger.info(f"📊 检测到队列较小({queue_size})且性能稳定，尝试提升编码质量")
            
            if needs_adjustment:
                self._last_adjustment_time = time.time()
                logger.debug(f"🔧 建议编码参数: {current_params}")
                
        except Exception as e:
            logger.warning(f"编码质量检查异常: {e}")
    
    def _optimize_audio_encoding(self, frame):
        """优化音频编码处理"""
        try:
            if not self.rtmp_initialized or not self.astream or not self.audio_resampler:
                return False
            
            # 创建音频帧
            new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
            new_frame.planes[0].update(frame.tobytes())
            new_frame.sample_rate = 16000
            
            # 批量处理音频重采样和编码
            resampled_frames = list(self.audio_resampler.resample(new_frame))
            
            # 批量编码和推送
            for rframe in resampled_frames:
                packets = list(self.astream.encode(rframe))
                for packet in packets:
                    self.rtmp_container.mux(packet)
            
            return True
            
        except Exception as e:
            logger.warning(f"音频编码优化异常: {e}")
            self._update_rtmp_stats(False, True)  # 记录音频错误
            return False
    
    def _update_rtmp_stats(self, video_success=True, audio_error=False):
        """更新RTMP推流性能统计"""
        try:
            current_time = time.time()
            
            # 初始化开始时间
            if self.rtmp_stats['start_time'] is None:
                self.rtmp_stats['start_time'] = current_time
            
            # 更新帧统计
            if video_success:
                self.rtmp_stats['total_frames'] += 1
                self.rtmp_stats['frame_times'].append(current_time)
                
                # 保持最近100帧的时间记录
                if len(self.rtmp_stats['frame_times']) > 100:
                    self.rtmp_stats['frame_times'].pop(0)
            else:
                # 视频编码失败
                self.rtmp_stats['encoding_errors'] += 1
            
            # 更新错误统计
            if audio_error:
                self.rtmp_stats['audio_errors'] += 1
            
            # 计算平均帧率
            if len(self.rtmp_stats['frame_times']) >= 2:
                time_span = self.rtmp_stats['frame_times'][-1] - self.rtmp_stats['frame_times'][0]
                if time_span > 0:
                    self.rtmp_stats['avg_fps'] = (len(self.rtmp_stats['frame_times']) - 1) / time_span
            
        except Exception as e:
            logger.warning(f"更新RTMP统计异常: {e}")
    
    def _log_rtmp_performance_stats(self):
        """记录RTMP推流性能统计信息"""
        try:
            current_time = time.time()
            
            # 每30秒记录一次详细统计
            if current_time - self.rtmp_stats['last_stats_time'] >= 30:
                self.rtmp_stats['last_stats_time'] = current_time
                
                # 计算运行时长
                if self.rtmp_stats['start_time']:
                    runtime = current_time - self.rtmp_stats['start_time']
                    runtime_str = f"{int(runtime//60)}分{int(runtime%60)}秒"
                else:
                    runtime_str = "未知"
                
                # 计算错误率
                total_frames = self.rtmp_stats['total_frames']
                error_rate = (self.rtmp_stats['encoding_errors'] / max(total_frames, 1)) * 100
                audio_error_rate = (self.rtmp_stats['audio_errors'] / max(total_frames, 1)) * 100
                
                logger.info("=" * 60)
                logger.info("📊 RTMP推流性能统计报告")
                logger.info("=" * 60)
                logger.info(f"🕐 运行时长: {runtime_str}")
                logger.info(f"🎬 总帧数: {total_frames}")
                logger.info(f"📈 平均帧率: {self.rtmp_stats['avg_fps']:.2f} fps")
                logger.info(f"🎯 目标帧率: {self.target_fps} fps")
                logger.info(f"📊 当前码率: {self.rtmp_stats['current_bitrate']}")
                logger.info(f"❌ 编码错误: {self.rtmp_stats['encoding_errors']} ({error_rate:.2f}%)")
                logger.info(f"🔊 音频错误: {self.rtmp_stats['audio_errors']} ({audio_error_rate:.2f}%)")
                logger.info(f"🔄 重连次数: {self.rtmp_stats['reconnections']}")
                logger.info(f"📦 推流分辨率: {self.rtmp_width}x{self.rtmp_height}")
                logger.info("=" * 60)
                
        except Exception as e:
            logger.warning(f"记录RTMP性能统计异常: {e}")
    
    def get_rtmp_stats(self):
        """获取RTMP推流统计信息（供API调用）"""
        try:
            current_time = time.time()
            stats = self.rtmp_stats.copy()
            
            # 计算运行时长
            if stats['start_time']:
                stats['runtime_seconds'] = current_time - stats['start_time']
            else:
                stats['runtime_seconds'] = 0
            
            # 计算错误率
            total_frames = stats['total_frames']
            stats['encoding_error_rate'] = (stats['encoding_errors'] / max(total_frames, 1)) * 100
            stats['audio_error_rate'] = (stats['audio_errors'] / max(total_frames, 1)) * 100
            
            # 添加推流状态信息
            stats['is_streaming'] = self.rtmp_initialized
            stats['resolution'] = f"{self.rtmp_width}x{self.rtmp_height}"
            stats['target_fps'] = self.target_fps
            
            return stats
            
        except Exception as e:
            logger.error(f"获取RTMP统计信息异常: {e}")
            return {}
    
    def set_rtmp_quality(self, quality_level):
        """设置RTMP推流清晰度级别
        
        Args:
            quality_level (str): 清晰度级别 ('ultra', 'high', 'medium', 'low')
            
        Returns:
            dict: 操作结果
        """
        try:
            if quality_level not in self.rtmp_quality_configs:
                available_levels = list(self.rtmp_quality_configs.keys())
                return {
                    'success': False,
                    'message': f'不支持的清晰度级别: {quality_level}，可用级别: {available_levels}'
                }
            
            old_level = self.rtmp_quality_level
            old_config = self.rtmp_quality_configs[old_level]
            new_config = self.rtmp_quality_configs[quality_level]
            
            self.rtmp_quality_level = quality_level
            
            # 如果RTMP正在推流，需要重新初始化以应用新的清晰度设置
            if self.rtmp_initialized:
                logger.info(f"🔄 清晰度从 {old_config['name']} 切换到 {new_config['name']}，重新初始化推流")
                self._reinitialize_rtmp()
            
            logger.info(f"✅ RTMP清晰度设置成功: {new_config['name']} ({quality_level})")
            
            return {
                'success': True,
                'message': f'清晰度已设置为: {new_config["name"]}',
                'old_quality': {'level': old_level, 'name': old_config['name']},
                'new_quality': {'level': quality_level, 'name': new_config['name']},
                'need_restart': self.rtmp_initialized
            }
            
        except Exception as e:
            logger.error(f"设置RTMP清晰度异常: {e}")
            return {
                'success': False,
                'message': f'设置清晰度失败: {str(e)}'
            }
    
    def get_rtmp_quality_info(self):
        """获取RTMP清晰度信息
        
        Returns:
            dict: 清晰度信息
        """
        try:
            current_config = self.rtmp_quality_configs[self.rtmp_quality_level]
            
            # 计算当前分辨率下的实际码率
            if self.rtmp_width > 0 and self.rtmp_height > 0:
                if self.rtmp_width * self.rtmp_height <= 640 * 480:
                    base_bitrate = 1500
                elif self.rtmp_width * self.rtmp_height <= 1280 * 720:
                    base_bitrate = 2500
                else:
                    base_bitrate = 4000
                
                actual_bitrate = int(base_bitrate * current_config['bitrate_factor'])
            else:
                actual_bitrate = None
            
            return {
                'current_level': self.rtmp_quality_level,
                'current_name': current_config['name'],
                'current_config': current_config,
                'actual_video_bitrate': f'{actual_bitrate}k' if actual_bitrate else None,
                'actual_audio_bitrate': current_config['audio_bitrate'],
                'available_levels': {
                    level: config['name'] for level, config in self.rtmp_quality_configs.items()
                },
                'is_streaming': self.rtmp_initialized,
                'resolution': f"{self.rtmp_width}x{self.rtmp_height}" if self.rtmp_width > 0 else None
            }
            
        except Exception as e:
            logger.error(f"获取RTMP清晰度信息异常: {e}")
            return {}
    
    def get_available_rtmp_qualities(self):
        """获取可用的RTMP清晰度级别列表
        
        Returns:
            list: 清晰度级别列表
        """
        return [
            {
                'level': level,
                'name': config['name'],
                'description': f"码率系数: {config['bitrate_factor']}x, CRF: {config['crf']}, 预设: {config['preset']}"
            }
            for level, config in self.rtmp_quality_configs.items()
        ]
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1