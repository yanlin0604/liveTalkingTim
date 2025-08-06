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
    def txt_to_audio(self,msg): 
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
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'ogg',
            'streaming_mode':True
        }
        # req["text"] = text
        # req["text_language"] = language
        # req["character"] = character
        # req["emotion"] = emotion
        # #req["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        # req["streaming_mode"] = True
        try:
            res = requests.post(
                f"{server_url}/tts",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            logger.info(f"gpt_sovits Time to make POST: {end-start}s")

            if res.status_code != 200:
                logger.error("Error:%s", res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                logger.info('chunk len:%d',len(chunk))
                if first:
                    end = time.perf_counter()
                    logger.info(f"gpt_sovits Time to first chunk: {end-start}s")
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            logger.exception('sovits')

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

    def stream_tts(self,audio_stream,msg):
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
        
        # 从动态配置中读取豆包TTS参数
        self.appid = get_config('DOUBAO_APPID', "")
        self.token = get_config('DOUBAO_TOKEN', "")
        
        # 打印日志参数
        print(f"豆包TTS参数: appid={self.appid}, token={self.token}")
        logger.info(f"豆包TTS参数: appid={self.appid}, token={self.token}")
        
        # 从动态配置中读取音频参数
        self.speed_ratio = get_nested_config('doubao_audio.speed_ratio', 0.8)
        self.volume_ratio = get_nested_config('doubao_audio.volume_ratio', 1.0)
        self.pitch_ratio = get_nested_config('doubao_audio.pitch_ratio', 1.0)
        # 打印日志参数
        logger.info(f"豆包TTS参数: speed_ratio={self.speed_ratio}, volume_ratio={self.volume_ratio}, pitch_ratio={self.pitch_ratio}")

        _cluster = 'volcano_tts'
        _host = "openspeech.bytedance.com"
        self.api_url = f"wss://{_host}/api/v1/tts/ws_binary"
        
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

        try:
            # 创建请求对象
            default_header = bytearray(b'\x11\x10\x11\x00')
            submit_request_json = copy.deepcopy(self.request_json)
            submit_request_json["user"]["uid"] = self.parent.sessionid
            submit_request_json["audio"]["voice_type"] = voice_type
            submit_request_json["request"]["text"] = text
            submit_request_json["request"]["reqid"] = str(uuid.uuid4())
            submit_request_json["request"]["operation"] = "submit"

            logger.info(f"请求JSON: {json.dumps(submit_request_json, ensure_ascii=False, indent=2)}")

            payload_bytes = str.encode(json.dumps(submit_request_json))
            logger.info(f"原始payload大小: {len(payload_bytes)} bytes")

            payload_bytes = gzip.compress(payload_bytes)  # if no compression, comment this line
            logger.info(f"压缩后payload大小: {len(payload_bytes)} bytes")

            full_client_request = bytearray(default_header)
            full_client_request.extend((len(payload_bytes)).to_bytes(4, 'big'))  # payload size(4 bytes)
            full_client_request.extend(payload_bytes)  # payload

            logger.info(f"完整请求大小: {len(full_client_request)} bytes")

            header = {"Authorization": f"Bearer; {self.token}"}
            logger.info(f"请求头: {header}")

            first = True
            chunk_count = 0
            total_audio_size = 0

            logger.info("开始连接WebSocket...")
            try:
                # 优先使用旧版参数名 extra_headers，若不支持则回退到新版 additional_headers
                ws_connect = websockets.connect(self.api_url, extra_headers=header, ping_interval=None)
            except TypeError:
                # 回退到新版参数名
                ws_connect = websockets.connect(self.api_url, additional_headers=header, ping_interval=None)
            async with ws_connect as ws:
                logger.info("WebSocket连接成功，发送请求...")
                await ws.send(full_client_request)
                logger.info("请求已发送，等待响应...")

                while True:
                    res = await ws.recv()
                    logger.info(f"收到响应，大小: {len(res)} bytes")

                    header_size = res[0] & 0x0f
                    message_type = res[1] >> 4
                    message_type_specific_flags = res[1] & 0x0f
                    payload = res[header_size*4:]

                    logger.info(f"消息类型: 0x{message_type:x}, 标志: 0x{message_type_specific_flags:x}, 头部大小: {header_size}, payload大小: {len(payload)}")

                    if message_type == 0xb:  # audio-only server response
                        if message_type_specific_flags == 0:  # no sequence number as ACK
                            logger.info("收到ACK响应，payload大小为0")
                            continue
                        else:
                            if first:
                                end = time.perf_counter()
                                logger.info(f"doubao tts Time to first chunk: {end-start}s")
                                first = False
                            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
                            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                            audio_payload = payload[8:]

                            chunk_count += 1
                            total_audio_size += len(audio_payload)
                            logger.info(f"音频块 #{chunk_count}: 序列号={sequence_number}, 声明大小={payload_size}, 实际大小={len(audio_payload)}")

                            yield audio_payload

                        if sequence_number < 0:
                            logger.info(f"收到结束信号，序列号: {sequence_number}")
                            break
                    else:
                        logger.warning(f"收到非音频消息类型: 0x{message_type:x}")
                        # 根据消息类型进行不同处理
                        if message_type == 0xf:  # 可能是控制消息或状态消息
                            logger.info("收到控制/状态消息，继续等待音频数据")
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
                                    except:
                                        # 直接解析
                                        error_msg = payload.decode('utf-8')
                                        logger.error(f"服务器错误信息: {error_msg}")
                                else:
                                    logger.warning("收到空的错误消息payload")
                            except Exception as decode_error:
                                logger.error(f"无法解析错误信息，解码异常: {str(decode_error)}")
                                logger.error(f"原始数据前100字节: {payload[:100]}")
                                # 尝试以十六进制显示
                                hex_data = payload[:50].hex() if len(payload) > 0 else "空数据"
                                logger.error(f"十六进制数据: {hex_data}")
                            break
                        else:
                            # 其他未知消息类型，记录并继续
                            logger.warning(f"未知消息类型 0x{message_type:x}，payload大小: {len(payload)}")
                            if len(payload) > 0:
                                hex_preview = payload[:20].hex() if len(payload) >= 20 else payload.hex()
                                logger.debug(f"payload预览(hex): {hex_preview}")
                            continue
            logger.info(f"=== 豆包TTS完成 ===")
            logger.info(f"总共收到 {chunk_count} 个音频块，总大小: {total_audio_size} bytes")

        except Exception as e:
            logger.error(f"豆包TTS异常: {str(e)}")
            logger.exception('doubao详细异常信息')
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

        try:
            asyncio.new_event_loop().run_until_complete(
                self.stream_tts(
                    self.doubao_voice(text),
                    msg
                )
            )
            logger.info("=== DoubaoTTS.txt_to_audio 完成 ===")
        except Exception as e:
            logger.error(f"DoubaoTTS.txt_to_audio 异常: {str(e)}")
            logger.exception("详细异常信息")

    async def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        logger.info(f"=== 开始音频流处理 ===")
        logger.info(f"处理文本: {text}")
        logger.info(f"chunk大小: {self.chunk}")

        first = True
        last_stream = np.array([],dtype=np.float32)
        chunk_count = 0
        total_audio_frames = 0

        async for chunk in audio_stream:
            chunk_count += 1
            logger.info(f"收到音频块 #{chunk_count}, 大小: {len(chunk) if chunk else 0} bytes")

            if chunk is not None and len(chunk) > 0:
                # 将字节数据转换为音频流
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                logger.info(f"转换后音频流长度: {len(stream)} samples")

                stream = np.concatenate((last_stream,stream))
                logger.info(f"合并后音频流长度: {len(stream)} samples")

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
                        first = False

                    self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
                    frames_in_chunk += 1
                    total_audio_frames += 1
                    streamlen -= self.chunk
                    idx += self.chunk

                logger.info(f"本块处理了 {frames_in_chunk} 个音频帧")
                last_stream = stream[idx:] #get the remain stream
                logger.info(f"剩余音频流长度: {len(last_stream)} samples")
            else:
                logger.warning(f"收到空音频块 #{chunk_count}")

        logger.info(f"=== 音频流处理完成 ===")
        logger.info(f"总共处理 {chunk_count} 个音频块")
        logger.info(f"总共输出 {total_audio_frames} 个音频帧")
        logger.info("发送音频结束事件")

        eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

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