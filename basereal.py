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

        参数:
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
            # RTMP 推流相关对象延迟初始化（在拿到第一帧视频后再初始化尺寸）
            rtmp_container = None
            vstream = None
            astream = None
            audio_resampler = None
            rtmp_width = 0
            rtmp_height = 0
            video_frame_index = 0
        
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
                # 懒加载初始化 RTMP 输出（基于首帧尺寸）
                if rtmp_container is None:
                    height, width, _ = combine_frame.shape
                    rtmp_width, rtmp_height = width, height
                    try:
                        # 以 FLV 格式打开 RTMP 输出
                        rtmp_container = av.open(self.opt.push_url, mode='w', format='flv')
                        # 视频流（libx264）
                        vstream = rtmp_container.add_stream('libx264', rate=int(target_fps))
                        vstream.width = rtmp_width
                        vstream.height = rtmp_height
                        vstream.pix_fmt = 'yuv420p'
                        vstream.bit_rate = 2_000_000  # ~2Mbps
                        vstream.gop_size = int(target_fps * 2)  # 关键帧间隔
                        # 音频流（AAC），统一转为44.1kHz立体声，兼容性更好
                        astream = rtmp_container.add_stream('aac', rate=44100)
                        astream.layout = 'stereo'
                        astream.channels = 2
                        audio_resampler = AudioResampler(format='s16', layout='stereo', rate=44100)
                        logger.info(f"RTMP 推流初始化成功 -> {self.opt.push_url} ({rtmp_width}x{rtmp_height}@{int(target_fps)}fps)")
                    except Exception as e:
                        logger.error(f"RTMP 推流初始化失败: {e}")
                        # 出错则跳过本帧，避免阻塞
                        continue

                # 编码并推送视频帧
                try:
                    vframe = VideoFrame.from_ndarray(combine_frame, format="bgr24").reformat(width=rtmp_width, height=rtmp_height, format='yuv420p')
                    vframe.pts = video_frame_index
                    vframe.time_base = Fraction(1, int(target_fps))
                    video_frame_index += 1
                    for packet in vstream.encode(vframe):
                        rtmp_container.mux(packet)
                except Exception as e:
                    logger.warning(f"RTMP 视频编码/推送异常: {e}")
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
                    # 将16kHz单声道PCM重采样至44.1kHz立体声并编码推送
                    if 'rtmp_container' in locals() and rtmp_container is not None and astream is not None and audio_resampler is not None:
                        try:
                            new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                            new_frame.planes[0].update(frame.tobytes())
                            new_frame.sample_rate = 16000
                            for rframe in audio_resampler.resample(new_frame):
                                for packet in astream.encode(rframe):
                                    rtmp_container.mux(packet)
                        except Exception as e:
                            logger.warning(f"RTMP 音频编码/推送异常: {e}")
                    else:
                        # 若容器尚未初始化（通常等待首个视频帧），则跳过音频
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
            # 刷新缓冲并关闭容器
            try:
                if 'vstream' in locals() and vstream is not None and 'rtmp_container' in locals() and rtmp_container is not None:
                    for packet in vstream.encode(None):
                        rtmp_container.mux(packet)
                if 'astream' in locals() and astream is not None and 'rtmp_container' in locals() and rtmp_container is not None:
                    for packet in astream.encode(None):
                        rtmp_container.mux(packet)
                if 'rtmp_container' in locals() and rtmp_container is not None:
                    rtmp_container.close()
            except Exception as e:
                logger.warning(f"RTMP 结束时清理异常: {e}")
        
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
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1