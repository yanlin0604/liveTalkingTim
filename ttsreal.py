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
from __future__ import annotations
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

import os
import hmac
import hashlib
import base64
import json
import uuid
import logging
from logging.handlers import RotatingFileHandler

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
import copy,websockets,gzip

from threading import Thread, Event
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from basereal import BaseReal

from logger import logger
from dynamic_config import get_config, get_nested_config

# === TTS 专用日志（独立文件：logs/tts.log，大小轮转）===
_TTS_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
_TTS_LOG_PATH = os.path.join(_TTS_LOG_DIR, "tts.log")

def _setup_tts_logger():
    """初始化 TTS 专用日志记录器。
    - 文件: logs/tts.log
    - 轮转: 2MB, 保留 5 个历史
    - 编码: utf-8
    """
    tl = logging.getLogger("tts")
    if tl.handlers:
        return tl
    tl.setLevel(logging.DEBUG)
    try:
        os.makedirs(_TTS_LOG_DIR, exist_ok=True)
    except Exception:
        pass
    try:
        fh = RotatingFileHandler(_TTS_LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        tl.addHandler(fh)
    except Exception:
        pass
    tl.propagate = False
    return tl

tts_logger = _setup_tts_logger()
class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    def __init__(self, opt, parent:BaseReal):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING
        # 语速（1.0为原速），可在外部通过 opt.tts_speed 配置
        self.speed = getattr(opt, 'tts_speed', 1.0)

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg:str,eventpoint=None): 
        if len(msg)>0:
            self.msgqueue.put((msg,eventpoint))

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        
        # 优雅停止：清理资源并记录日志
        logger.info('ttsreal thread stop - 优雅清理完成')
        
        # 清理消息队列中的剩余数据
        try:
            while not self.msgqueue.empty():
                self.msgqueue.get_nowait()
            logger.info('🧹 清理TTS消息队列')
        except:
            pass
            
        # 清理输入流
        if hasattr(self, 'input_stream'):
            try:
                self.input_stream.seek(0)
                self.input_stream.truncate()
                logger.info('🧹 清理TTS输入流')
            except:
                pass
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg):
        voicename = self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        text,textevent = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint=None
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text,'msgevent':textevent}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            # 写死语速为0.6倍（明显较慢），使用SSML格式控制语速
            ssml_text = f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"><prosody rate="0.6">{text}</prosody></speak>'
            logger.info(f'EdgeTTS using SSML with rate 0.6: {ssml_text[:100]}...')
            communicate = edge_tts.Communicate(ssml_text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')

###########################################################################################
class FishTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        text,textevent = msg
        self.stream_tts(
            self.fish_speech(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def fish_speech(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'reference_id':reffile,
            'format':'wav',
            'streaming':True,
            'use_memory_cache':'on'
        }
        try:
            res = requests.post(
                f"{server_url}/v1/tts",
                json=req,
                stream=True,
                headers={
                    "content-type": "application/json",
                },
            )
            end = time.perf_counter()
            logger.info(f"fish_speech Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=17640): # 1764 44100*20ms*2
                #print('chunk len:',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"fish_speech Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('fishtts')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class SovitsTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]): 
        text,textevent = msg
        self.stream_tts(
            self.gpt_sovits(
                text=text,
                reffile=self.opt.REF_FILE,
                reftext=self.opt.REF_TEXT,
                language="zh", #en args.language,
                server_url=self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        # 动态获取语速配置
        speed_factor = getattr(self, 'speed', 1.0)
        
        # 打印请求参数
        logger.info(f"GPT-SoVITS 请求参数:")
        logger.info(f"  文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        logger.info(f"  参考音频: {reffile}")
        logger.info(f"  参考文本: {reftext}")
        logger.info(f"  语言: {language}")
        logger.info(f"  服务器地址: {server_url}")
        logger.info(f"  语速因子: {speed_factor}")
        
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'ogg',
            'streaming_mode':True,
            'speed_factor': speed_factor
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # 可以减少此值以获得更快响应，但会降低质量
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"GPT-SoVITS POST请求耗时: {end-start}s")

            if res.status_code != 200:
                logger.error("错误:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('音频块长度:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"GPT-SoVITS 首个音频块耗时: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('SoVITS异常')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[信息]TTS音频流 采样率{sample_rate}: 形状{stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[警告] 音频有{stream.shape[1]}个声道，仅使用第一个声道')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[警告] 音频采样率为{sample_rate}，重采样为{self.sample_rate}')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                #stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                #stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream=BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint={}
                    if first:
                        eventpoint={'status':'start','text':text}
                        eventpoint.update(**textevent) 
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent) 
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)

###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=9600): # 960 24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('cosyvoice')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class TencentTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.appid = os.getenv("TENCENT_APPID")
        self.secret_key = os.getenv("TENCENT_SECRET_KEY")
        self.secret_id = os.getenv("TENCENT_SECRET_ID")
        self.voice_type = int(opt.REF_FILE)
        self.codec = "pcm"
        self.sample_rate = 16000
        self.volume = 0
        self.speed = 0
    
    def __gen_signature(self, params):
        sort_dict = sorted(params.keys())
        sign_str = "POST" + _HOST + _PATH + "?"
        for key in sort_dict:
            sign_str = sign_str + key + "=" + str(params[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'),
                           sign_str.encode('utf-8'), hashlib.sha1).digest()
        s = base64.b64encode(hmacstr)
        s = s.decode('utf-8')
        return s

    def __gen_params(self, session_id, text):
        params = dict()
        params['Action'] = _ACTION
        params['AppId'] = int(self.appid)
        params['SecretId'] = self.secret_id
        params['ModelType'] = 1
        params['VoiceType'] = self.voice_type
        params['Codec'] = self.codec
        params['SampleRate'] = self.sample_rate
        params['Speed'] = self.speed
        params['Volume'] = self.volume
        params['SessionId'] = session_id
        params['Text'] = text

        timestamp = int(time.time())
        params['Timestamp'] = timestamp
        params['Expired'] = timestamp + 24 * 60 * 60
        return params

    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.tencent_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def tencent_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        session_id = str(uuid.uuid1())
        params = self.__gen_params(session_id, text)
        signature = self.__gen_signature(params)
        headers = {
            "Content-Type": "application/json",
            "Authorization": str(signature)
        }
        url = _PROTOCOL + _HOST + _PATH
        try:
            res = requests.post(url, headers=headers,
                          data=json.dumps(params), stream=True)
            
            end = time.perf_counter()
            logger.info(f"tencent Time to make POST: {end-start}s")
                
            first = True
        
            for chunk in res.iter_content(chunk_size=6400): # 640 16K*20ms*2
                #logger.info('chunk len:%d',len(chunk))
                if first:
                    try:
                        rsp = json.loads(chunk)
                        #response["Code"] = rsp["Response"]["Error"]["Code"]
                        #response["Message"] = rsp["Response"]["Error"]["Message"]
                        logger.error("tencent tts:%s",rsp["Response"]["Error"]["Message"])
                        return
                    except:
                        end = time.perf_counter()
                        logger.info(f"tencent Time to first chunk: {end-start}s")
                        first = False                    
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            logger.exception('tencent')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        last_stream = np.array([],dtype=np.float32)
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = np.concatenate((last_stream,stream))
                #stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        logger.info("发送音频开始事件")
                        first = False

                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:] #get the remain stream
        logger.info(f"=== 音频流处理完成 ===")
        logger.info("发送音频结束事件")

        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class DoubaoTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        print(f"opt: {opt}")
        logger.info(f"opt: {opt}")
        tts_logger.info("[DoubaoTTS] 初始化 opt=%s", opt)
        
        # 从动态配置中读取豆包TTS参数
        self.appid = get_config('DOUBAO_APPID', "")
        self.token = get_config('DOUBAO_TOKEN', "")
        
        # 打印日志参数
        print(f"豆包TTS参数: appid={self.appid}, token={self.token}")
        logger.info(f"豆包TTS参数: appid={self.appid}, token={self.token}")
        tts_logger.info("[DoubaoTTS] 参数: appid=%s, token_mask=%s...%s", self.appid, (self.token or '')[:6], (self.token or '')[-6:])
        
        # 从动态配置中读取音频参数
        self.speed_ratio = get_nested_config('doubao_audio.speed_ratio', 0.8)
        self.volume_ratio = get_nested_config('doubao_audio.volume_ratio', 1.0)
        self.pitch_ratio = get_nested_config('doubao_audio.pitch_ratio', 1.0)
        # 打印日志参数
        logger.info(f"豆包TTS参数: speed_ratio={self.speed_ratio}, volume_ratio={self.volume_ratio}, pitch_ratio={self.pitch_ratio}")
        tts_logger.info("[DoubaoTTS] 音频参数: speed=%.3f volume=%.3f pitch=%.3f", self.speed_ratio, self.volume_ratio, self.pitch_ratio)

        _cluster = 'volcano_tts'
        _host = "openspeech.bytedance.com"
        self.api_url = f"wss://{_host}/api/v1/tts/ws_binary"
        tts_logger.info("[DoubaoTTS] API: %s", self.api_url)
        
        self.request_json = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": _cluster
            },
            "user": {
                "uid": "xxx"
            },
            "audio": {
                "voice_type": "xxx",
                "encoding": "pcm",
                "rate": 16000,
                "speed_ratio": self.speed_ratio,
                "volume_ratio": self.volume_ratio,
                "pitch_ratio": self.pitch_ratio,
            },
            "request": {
                "reqid": "xxx",
                "text": "字节跳动语音合成。",
                "text_type": "plain",
                "operation": "xxx"
            }
        }

    async def doubao_voice(self, text): # -> Iterator[bytes]:
        start = time.perf_counter()
        voice_type = get_config('REF_FILE', '')

        logger.info(f"=== 豆包TTS开始 ===")
        logger.info(f"输入文本: {text}")
        logger.info(f"语音类型: {voice_type}")
        logger.info(f"APPID: {self.appid}")
        logger.info(f"TOKEN: {self.token[:10]}...{self.token[-10:] if len(self.token) > 20 else self.token}")
        logger.info(f"API URL: {self.api_url}")
        tts_logger.info("[DoubaoTTS] 开始，voice_type=%s text_len=%d", voice_type, len(text or ''))

        try:
            # 创建请求对象
            default_header = bytearray(b'\x11\x10\x11\x00')
            submit_request_json = copy.deepcopy(self.request_json)
            submit_request_json["user"]["uid"] = self.parent.sessionid
            submit_request_json["audio"]["voice_type"] = voice_type
            # 每次请求前读取最新的音频参数，确保热更新立即生效
            try:
                latest_speed = get_nested_config('doubao_audio.speed_ratio', self.speed_ratio)
                latest_volume = get_nested_config('doubao_audio.volume_ratio', self.volume_ratio)
                latest_pitch = get_nested_config('doubao_audio.pitch_ratio', self.pitch_ratio)
                submit_request_json["audio"]["speed_ratio"] = latest_speed
                submit_request_json["audio"]["volume_ratio"] = latest_volume
                submit_request_json["audio"]["pitch_ratio"] = latest_pitch
                # 同步更新到实例字段（可选）
                self.speed_ratio = latest_speed
                self.volume_ratio = latest_volume
                self.pitch_ratio = latest_pitch
                logger.info(f"已应用最新TTS音频参数: speed={latest_speed}, volume={latest_volume}, pitch={latest_pitch}")
                tts_logger.info("[DoubaoTTS] 应用最新音频参数: speed=%.3f volume=%.3f pitch=%.3f", latest_speed, latest_volume, latest_pitch)
            except Exception as _e:
                logger.warning(f"应用最新音频参数失败，继续使用旧参数: {_e}")
                tts_logger.warning("[DoubaoTTS] 应用最新音频参数失败: %s", _e)
            # 发送前对文本做严格清洗与规范化，避免服务端返回 illegal input text
            def _sanitize_text_for_doubao(raw: str) -> str:
                import re
                import unicodedata
                if raw is None:
                    return ""
                t = str(raw)
                # Unicode 规范化（兼容全角/半角等）
                t = unicodedata.normalize('NFKC', t)
                # 去除零宽/不可见字符与 BOM/控制符
                t = re.sub(r"[\u200b-\u200f\ufeff\u202a-\u202e]", "", t)
                t = ''.join(ch for ch in t if ch == '\n' or (ch >= ' ' and ch != '\x7f'))
                # 允许的字符范围：中英文、数字、常用中英文标点和括号等
                allowed = r"[^\u4e00-\u9fffA-Za-z0-9\s，。、；：！？,.!?\"'“”‘’（）()【】\[\]《》—…·\-:_]+"
                t = re.sub(allowed, "", t)
                # 统一多空白
                t = re.sub(r"\s+", " ", t).strip()
                # 长度保护（避免过长导致拒绝），可按需调整
                if len(t) > 500:
                    t = t[:500]
                return t

            _orig_text = text
            _clean_text = _sanitize_text_for_doubao(_orig_text)
            if _clean_text != _orig_text:
                tts_logger.info("[DoubaoTTS] 文本已清洗: 原len=%d 新len=%d 原预览='%s' 新预览='%s'",
                                len(_orig_text or ""), len(_clean_text), str(_orig_text or "")[:100], _clean_text[:100])
            # 若清洗后为空，则直接结束此次 TTS（不再兜底“好的。”，避免你反馈的“老是拼一个好的”）
            if not _clean_text:
                tts_logger.warning("[DoubaoTTS] 文本清洗后为空，本次不发送TTS请求")
                return

            submit_request_json["request"]["text"] = _clean_text
            submit_request_json["request"]["reqid"] = str(uuid.uuid4())
            submit_request_json["request"]["operation"] = "submit"

            logger.info(f"请求JSON: {json.dumps(submit_request_json, ensure_ascii=False, indent=2)}")
            tts_logger.debug("[DoubaoTTS] 请求JSON: %s", json.dumps(submit_request_json, ensure_ascii=False)[:1000])
            # 明确打印即将发送的文本（来自 request.text）
            _send_text = submit_request_json["request"].get("text", "")
            logger.info(f"发送文本长度: {len(_send_text)} 内容预览='{_send_text[:200]}'")
            tts_logger.info("[DoubaoTTS] 发送文本 len=%d 预览='%s'", len(_send_text), _send_text[:200])

            payload_bytes = str.encode(json.dumps(submit_request_json))
            logger.info(f"原始payload大小: {len(payload_bytes)} bytes")
            tts_logger.info("[DoubaoTTS] 原始payload大小=%d bytes", len(payload_bytes))

            payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
            logger.info(f"压缩后payload大小: {len(payload_bytes)} bytes")
            tts_logger.info("[DoubaoTTS] 压缩后payload大小=%d bytes", len(payload_bytes))

            full_client_request = bytearray(default_header)
            full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
            full_client_request.extend(payload_bytes)  # payload

            logger.info(f"完整请求大小: {len(full_client_request)} bytes")
            tts_logger.info("[DoubaoTTS] 完整请求大小=%d bytes", len(full_client_request))
            # 从将要发送的payload中预览文本（尝试解压+UTF-8 解码）
            try:
                _payload_preview = None
                try:
                    _decompressed = gzip.decompress(payload_bytes)
                    _payload_preview = _decompressed.decode('utf-8', errors='replace')
                    tts_logger.debug("[DoubaoTTS] 发送payload 解压后len=%d 文本预览='%s'", len(_payload_preview), _payload_preview[:400])
                except Exception:
                    _payload_preview = payload_bytes.decode('utf-8', errors='replace')
                    tts_logger.debug("[DoubaoTTS] 发送payload 直解len=%d 文本预览='%s'", len(_payload_preview), _payload_preview[:400])
            except Exception as _e:
                tts_logger.warning("[DoubaoTTS] payload文本预览失败: %s", _e)

            header = {"Authorization": f"Bearer; {self.token}"}
            logger.info(f"请求头: {header}")
            tts_logger.debug("[DoubaoTTS] 请求头: %s", header)

            first = True
            chunk_count = 0
            total_audio_size = 0
            # 记录最近一次收到音频的时间，用于无音频超时保护
            last_audio_time = time.perf_counter()

            logger.info("开始连接WebSocket...")
            tts_logger.info("[DoubaoTTS] 连接WebSocket: %s", self.api_url)
            try:
                # 优先使用旧版参数名 extra_headers，若不支持则回退到新版 additional_headers
                ws_connect = websockets.connect(self.api_url, extra_headers=header, ping_interval=None)
            except TypeError:
                # 回退到新版参数名
                ws_connect = websockets.connect(self.api_url, additional_headers=header, ping_interval=None)
            async with ws_connect as ws:
                logger.info("WebSocket连接成功，发送请求...")
                tts_logger.info("[DoubaoTTS] WebSocket连接成功，发送请求")
                await ws.send(full_client_request)
                logger.info("请求已发送，等待响应...")
                tts_logger.info("[DoubaoTTS] 请求已发送，等待响应")

                while True:
                    res = await ws.recv()
                    logger.info(f"收到响应，大小: {len(res)} bytes")
                    tts_logger.debug("[DoubaoTTS] 收到WS帧 bytes=%d", len(res))

                    header_size = res[0] & 0x0f
                    message_type = res[1] >> 4
                    message_type_specific_flags = res[1] & 0x0f
                    payload = res[header_size*4:]

                    logger.info(f"消息类型: 0x{message_type:x}, 标志: 0x{message_type_specific_flags:x}, 头部大小: {header_size}, payload大小: {len(payload)}")
                    tts_logger.debug("[DoubaoTTS] type=0x%x flags=0x%x header_size=%d payload=%d", message_type, message_type_specific_flags, header_size, len(payload))

                    if message_type == 0xb:  # audio-only server response
                        if message_type_specific_flags == 0:  # no sequence number as ACK
                            logger.info("收到ACK响应，payload大小为0")
                            tts_logger.info("[DoubaoTTS] 收到ACK，payload=0")
                            continue
                        else:
                            if first:
                                end = time.perf_counter()
                                logger.info(f"doubao tts Time to first chunk: {end-start}s")
                                tts_logger.info("[DoubaoTTS] 首包耗时=%.3fs", end - start)
                                first = False
                            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                            audio_payload = payload[8:]

                            chunk_count += 1
                            total_audio_size += len(audio_payload)
                            logger.info(f"音频块 #{chunk_count}: 序列号={sequence_number}, 声明大小={payload_size}, 实际大小={len(audio_payload)}")
                            tts_logger.info("[DoubaoTTS] 音频块#%d seq=%d decl=%d real=%d", chunk_count, sequence_number, payload_size, len(audio_payload))
                            # 更新最近音频时间
                            last_audio_time = time.perf_counter()

                            yield audio_payload

                        if sequence_number < 0:
                            logger.info(f"收到结束信号，序列号: {sequence_number}")
                            tts_logger.info("[DoubaoTTS] 收到结束信号 seq=%d", sequence_number)
                            break
                    else:
                        logger.warning(f"收到非音频消息类型: 0x{message_type:x}")
                        # 根据消息类型进行不同处理
                        if message_type == 0xf:  # 可能是控制消息或状态消息
                            # 增强对控制/状态payload的可观测性
                            try:
                                preview = payload[:120]
                                hex_preview = preview.hex() if len(preview) > 0 else "空"
                                tts_logger.debug("[DoubaoTTS] 控制/状态payload原始前120字节(hex)=%s", hex_preview)
                                text = None
                                if len(payload) > 0:
                                    # 控制消息也可能遵循同样的 4字节保留 + 4字节长度 + 压缩内容 的结构
                                    try:
                                        inner = payload
                                        if len(payload) >= 8:
                                            inner_size = int.from_bytes(payload[4:8], 'big', signed=False)
                                            inner = payload[8:8+inner_size]
                                        # 优先尝试解压
                                        try:
                                            decompressed = gzip.decompress(inner)
                                            text = decompressed.decode("utf-8", errors="replace")
                                            tts_logger.debug("[DoubaoTTS] 控制/状态payload(解包后) 解压len=%d 预览='%s'", len(text), text[:200])
                                        except Exception:
                                            text = inner.decode("utf-8", errors="replace")
                                            tts_logger.debug("[DoubaoTTS] 控制/状态payload(解包后) 直解len=%d 预览='%s'", len(text), text[:200])
                                    except Exception as _inner_e:
                                        # 退化为直接尝试原始payload
                                        try:
                                            decompressed = gzip.decompress(payload)
                                            text = decompressed.decode("utf-8", errors="replace")
                                            tts_logger.debug("[DoubaoTTS] 控制/状态payload 解压后len=%d 预览='%s'", len(text), text[:200])
                                        except Exception:
                                            text = payload.decode("utf-8", errors="replace")
                                            tts_logger.debug("[DoubaoTTS] 控制/状态payload 直解len=%d 预览='%s'", len(text), text[:200])
                                    # 尝试JSON解析，提取常见字段（使用模块级json，避免局部遮蔽）
                                    try:
                                        obj = json.loads(text)
                                        code = obj.get("code") or obj.get("status_code") or obj.get("err_no")
                                        status = obj.get("status") or obj.get("message") or obj.get("err_msg") or obj.get("task_status")
                                        err_msg = obj.get("error") or obj.get("msg")
                                        tts_logger.info("[DoubaoTTS] 控制/状态: code=%s status=%s error=%s keys=%s", str(code), str(status), str(err_msg), list(obj.keys()))
                                        # 发现明确错误字段则直接结束，避免挂起
                                        if err_msg:
                                            tts_logger.warning("[DoubaoTTS] 服务端错误: %s，结束会话", err_msg)
                                            break
                                        # 若明显是结束/拒绝/错误状态，结束本次会话，避免挂起
                                        status_l = str(status).lower() if status is not None else ""
                                        if status_l in ("end", "ended", "finish", "finished", "stopped") or \
                                           str(code) not in ("0", "None"):
                                            tts_logger.warning("[DoubaoTTS] 控制消息指示结束或错误，主动结束会话: code=%s status=%s", str(code), str(status))
                                            break
                                    except Exception:
                                        pass
                                else:
                                    tts_logger.info("[DoubaoTTS] 控制/状态消息：空payload")
                            except Exception as e:
                                tts_logger.exception("[DoubaoTTS] 控制/状态payload解析异常: %s", e)
                            # 无音频超时保护：若长时间未收到音频块，则结束本次会话，交由上层重试
                            now = time.perf_counter()
                            no_audio_secs = now - last_audio_time
                            if no_audio_secs > 5.0 and chunk_count == 0:
                                logger.warning(f"控制/状态持续且无音频超过 {no_audio_secs:.2f}s，结束本次会话")
                                tts_logger.warning("[DoubaoTTS] 控制/状态无音频超时(>5s)，结束会话")
                                break
                            logger.info("收到控制/状态消息，继续等待音频数据")
                            tts_logger.info("[DoubaoTTS] 控制/状态消息，继续")
                            continue
                        elif message_type == 0xc:  # 可能是错误消息
                            try:
                                # 尝试解压缩payload
                                if len(payload) > 0:
                                    try:
                                        # 尝试gzip解压
                                        decompressed = gzip.decompress(payload)
                                        error_msg = decompressed.decode('utf-8')
                                        logger.error(f"服务器错误信息(解压后): {error_msg}")
                                        tts_logger.error("[DoubaoTTS] 错误(解压后): %s", error_msg)
                                    except:
                                        # 直接解析
                                        error_msg = payload.decode('utf-8')
                                        logger.error(f"服务器错误信息: {error_msg}")
                                        tts_logger.error("[DoubaoTTS] 错误: %s", error_msg)
                                else:
                                    logger.warning("收到空的错误消息payload")
                                    tts_logger.warning("[DoubaoTTS] 空错误payload")
                            except Exception as decode_error:
                                logger.error(f"无法解析错误信息，解码异常: {str(decode_error)}")
                                logger.error(f"原始数据前100字节: {payload[:100]}")
                                # 尝试以十六进制显示
                                hex_data = payload[:50].hex() if len(payload) > 0 else "空数据"
                                logger.error(f"十六进制数据: {hex_data}")
                                tts_logger.exception("[DoubaoTTS] 错误解析异常: %s", decode_error)
                            break
                        else:
                            # 其他未知消息类型，记录并继续
                            logger.warning(f"未知消息类型 0x{message_type:x}，payload大小: {len(payload)}")
                            if len(payload) > 0:
                                hex_preview = payload[:20].hex() if len(payload) >= 20 else payload.hex()
                                logger.debug(f"payload预览(hex): {hex_preview}")
                                tts_logger.debug("[DoubaoTTS] 未知类型 payload_hex=%s", hex_preview)
                            continue
            logger.info(f"=== 豆包TTS完成 ===")
            logger.info(f"总共收到 {chunk_count} 个音频块，总大小: {total_audio_size} bytes")
            tts_logger.info("[DoubaoTTS] 完成，块数=%d 总字节=%d", chunk_count, total_audio_size)

        except Exception as e:
            logger.error(f"豆包TTS异常: {str(e)}")
            logger.exception('doubao详细异常信息')
            tts_logger.exception("[DoubaoTTS] 异常: %s", e)
        # # 检查响应状态码
        # if response.status_code == 200:
        #     # 处理响应数据
        #     audio_data = base64.b64decode(response.json().get('data'))
        #     yield audio_data
        # else:
        #     logger.error(f"请求失败，状态码: {response.status_code}")
        #     return

    def txt_to_audio(self, msg):
        text, textevent = msg
        logger.info(f"=== DoubaoTTS.txt_to_audio 开始 ===")
        logger.info(f"输入消息: text='{text}', textevent={textevent}")
        tts_logger.info("[DoubaoTTS] txt_to_audio 开始 text_len=%d", len(text or ''))

        try:
            asyncio.new_event_loop().run_until_complete(
                self.stream_tts(
                    self.doubao_voice(text),
                    msg
                )
            )
            logger.info("=== DoubaoTTS.txt_to_audio 完成 ===")
            tts_logger.info("[DoubaoTTS] txt_to_audio 完成")
        except Exception as e:
            logger.error(f"DoubaoTTS.txt_to_audio 异常: {str(e)}")
            logger.exception("详细异常信息")
            tts_logger.exception("[DoubaoTTS] txt_to_audio 异常: %s", e)

    async def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        logger.info(f"=== 开始音频流处理 ===")
        logger.info(f"处理文本: {text}")
        logger.info(f"chunk大小: {self.chunk}")
        tts_logger.info("[DoubaoTTS] 流式开始 chunk=%d", self.chunk)

        first = True
        last_stream = np.array([],dtype=np.float32)
        chunk_count = 0
        total_audio_frames = 0

        async for chunk in audio_stream:
            chunk_count += 1
            logger.info(f"收到音频块 #{chunk_count}, 大小: {len(chunk) if chunk else 0} bytes")
            tts_logger.debug("[DoubaoTTS] 收到音频块#%d bytes=%d", chunk_count, len(chunk) if chunk else 0)

            if chunk is not None and len(chunk) > 0:
                # 将字节数据转换为音频流
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                logger.info(f"转换后音频流长度: {len(stream)} samples")
                tts_logger.debug("[DoubaoTTS] 转换流 samples=%d", len(stream))

                stream = np.concatenate((last_stream,stream))
                logger.info(f"合并后音频流长度: {len(stream)} samples")
                tts_logger.debug("[DoubaoTTS] 合并后 samples=%d", len(stream))

                #stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                # byte_stream=BytesIO(buffer)
                # stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx = 0
                frames_in_chunk = 0

                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgenvent': textevent}
                        logger.info("发送音频开始事件")
                        tts_logger.info("[DoubaoTTS] 发送开始事件 text_len=%d", len(text or ''))
                        first = False

                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    frames_in_chunk += 1
                    total_audio_frames += 1
                    streamlen -= self.chunk
                    idx += self.chunk

                logger.info(f"本块处理了 {frames_in_chunk} 个音频帧")
                tts_logger.debug("[DoubaoTTS] 本块帧数=%d", frames_in_chunk)
                last_stream = stream[idx:] #get the remain stream
                logger.info(f"剩余音频流长度: {len(last_stream)} samples")
                tts_logger.debug("[DoubaoTTS] 残留 samples=%d", len(last_stream))
            else:
                logger.warning(f"收到空音频块 #{chunk_count}")
                tts_logger.warning("[DoubaoTTS] 空音频块#%d", chunk_count)

        logger.info(f"=== 音频流处理完成 ===")
        logger.info(f"总共处理 {chunk_count} 个音频块")
        logger.info(f"总共输出 {total_audio_frames} 个音频帧")
        logger.info("发送音频结束事件")
        tts_logger.info("[DoubaoTTS] 流式结束 块=%d 帧=%d", chunk_count, total_audio_frames)

        eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

###########################################################################################
class IndexTTS2(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        # IndexTTS2 配置参数
        self.server_url = opt.TTS_SERVER  # Gradio服务器地址，如 "http://127.0.0.1:7860/"
        self.ref_audio_path = opt.REF_FILE  # 参考音频文件路径
        self.max_tokens = getattr(opt, 'MAX_TOKENS', 120)  # 最大token数
        
        # 初始化Gradio客户端
        try:
            from gradio_client import Client, handle_file
            self.client = Client(self.server_url)
            self.handle_file = handle_file
            logger.info(f"IndexTTS2 Gradio客户端初始化成功: {self.server_url}")
        except ImportError:
            logger.error("IndexTTS2 需要安装 gradio_client: pip install gradio_client")
            raise
        except Exception as e:
            logger.error(f"IndexTTS2 Gradio客户端初始化失败: {e}")
            raise
        
    def txt_to_audio(self, msg):
        text, textevent = msg
        try:
            # 先进行文本分割
            segments = self.split_text(text)
            if not segments:
                logger.error("IndexTTS2 文本分割失败")
                return
            
            logger.info(f"IndexTTS2 文本分割为 {len(segments)} 个片段")
            
            # 循环生成每个片段的音频
            for i, segment_text in enumerate(segments):
                if self.state != State.RUNNING:
                    break
                    
                logger.info(f"IndexTTS2 正在生成第 {i+1}/{len(segments)} 段音频...")
                audio_file = self.indextts2_generate(segment_text)
                
                if audio_file:
                    # 为每个片段创建事件信息
                    segment_msg = (segment_text, textevent)
                    self.file_to_stream(audio_file, segment_msg, is_first=(i==0), is_last=(i==len(segments)-1))
                else:
                    logger.error(f"IndexTTS2 第 {i+1} 段音频生成失败")
                    
        except Exception as e:
            logger.exception(f"IndexTTS2 txt_to_audio 错误: {e}")

    def split_text(self, text):
        """使用 IndexTTS2 API 分割文本"""
        try:
            logger.info(f"IndexTTS2 开始分割文本，长度: {len(text)}")
            
            # 调用文本分割 API
            result = self.client.predict(
                text=text,
                max_text_tokens_per_segment=self.max_tokens,
                api_name="/on_input_text_change"
            )
            
            # 解析分割结果
            if 'value' in result and 'data' in result['value']:
                data = result['value']['data']
                logger.info(f"IndexTTS2 共分割为 {len(data)} 个片段")
                
                segments = []
                for i, item in enumerate(data):
                    序号 = item[0] + 1
                    分句内容 = item[1]
                    token数 = item[2]
                    logger.info(f"片段 {序号}: {len(分句内容)} 字符, {token数} tokens")
                    segments.append(分句内容)
                
                return segments
            else:
                logger.error(f"IndexTTS2 文本分割结果格式异常: {result}")
                return [text]  # 如果分割失败，返回原文本
                
        except Exception as e:
            logger.exception(f"IndexTTS2 文本分割失败: {e}")
            return [text]  # 如果分割失败，返回原文本

    def indextts2_generate(self, text):
        """调用 IndexTTS2 Gradio API 生成语音"""
        start = time.perf_counter()
        
        try:
            # 调用 gen_single API
            result = self.client.predict(
                emo_control_method="Same as the voice reference",
                prompt=self.handle_file(self.ref_audio_path),
                text=text,
                emo_ref_path=self.handle_file(self.ref_audio_path),
                emo_weight=0.8,
                vec1=0.5,
                vec2=0,
                vec3=0,
                vec4=0,
                vec5=0,
                vec6=0,
                vec7=0,
                vec8=0,
                emo_text="",
                emo_random=False,
                max_text_tokens_per_segment=self.max_tokens,
                param_16=True,
                param_17=0.8,
                param_18=30,
                param_19=0.8,
                param_20=0,
                param_21=3,
                param_22=10,
                param_23=1500,
                api_name="/gen_single"
            )
            
            end = time.perf_counter()
            logger.info(f"IndexTTS2 片段生成完成，耗时: {end-start:.2f}s")
            
            # 返回生成的音频文件路径
            if 'value' in result:
                audio_file = result['value']
                return audio_file
            else:
                logger.error(f"IndexTTS2 结果格式异常: {result}")
                return None
                
        except Exception as e:
            logger.exception(f"IndexTTS2 API调用失败: {e}")
            return None

    def file_to_stream(self, audio_file, msg, is_first=False, is_last=False):
        """将音频文件转换为音频流"""
        text, textevent = msg
        
        try:
            # 读取音频文件
            stream, sample_rate = sf.read(audio_file)
            logger.info(f'IndexTTS2 音频文件 {sample_rate}Hz: {stream.shape}')
            
            # 转换为float32
            stream = stream.astype(np.float32)
            
            # 如果是多声道，只取第一个声道
            if stream.ndim > 1:
                logger.info(f'IndexTTS2 音频有 {stream.shape[1]} 个声道，只使用第一个')
                stream = stream[:, 0]
            
            # 重采样到目标采样率
            if sample_rate != self.sample_rate and stream.shape[0] > 0:
                logger.info(f'IndexTTS2 重采样: {sample_rate}Hz -> {self.sample_rate}Hz')
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
            
            # 分块发送音频流
            streamlen = stream.shape[0]
            idx = 0
            first_chunk = True
            
            while streamlen >= self.chunk and self.state == State.RUNNING:
                eventpoint = None
                
                # 只在第一个片段的第一个chunk发送start事件
                if is_first and first_chunk:
                    eventpoint = {'status': 'start', 'text': text, 'msgevent': textevent}
                    first_chunk = False
                
                self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                idx += self.chunk
                streamlen -= self.chunk
            
            # 只在最后一个片段发送end事件
            if is_last:
                eventpoint = {'status': 'end', 'text': text, 'msgevent': textevent}
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
            
            # 清理临时文件
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logger.info(f"IndexTTS2 已删除临时文件: {audio_file}")
            except Exception as e:
                logger.warning(f"IndexTTS2 删除临时文件失败: {e}")
                
        except Exception as e:
            logger.exception(f"IndexTTS2 音频流处理失败: {e}")

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg):
        text,textevent = msg  
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            ),
            msg
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgevent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgevent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  