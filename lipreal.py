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

# from .utils import *
import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp


from lipasr import LipASR
import asyncio
from av import AudioFrame, VideoFrame

# Wav2Lip 模型将在 load_model 函数中动态导入
from basereal import BaseReal

# from imgcache import ImgCache

from tqdm import tqdm
from logger import logger

device = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else "cpu"
    )
)
print("Using {} for inference.".format(device))


def _load(checkpoint_path):
    if device == "cuda":
        checkpoint = torch.load(checkpoint_path)  # ,weights_only=True
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


def detect_model_version(checkpoint_state_dict):
    """
    检测模型版本：通过检查特定层的存在和形状来判断模型架构
    """
    # 检查是否存在第7个face_encoder_blocks（384高分辨率版本特有）
    has_encoder_7 = any(
        "face_encoder_blocks.7." in key for key in checkpoint_state_dict.keys()
    )

    # 检查是否存在face_decoder_blocks.2.2（384版本特有的额外层）
    has_decoder_2_2 = any(
        "face_decoder_blocks.2.2." in key for key in checkpoint_state_dict.keys()
    )

    # 检查face_encoder_blocks.6.1的卷积核大小
    encoder_6_1_key = "face_encoder_blocks.6.1.conv_block.0.weight"
    if encoder_6_1_key in checkpoint_state_dict:
        kernel_shape = checkpoint_state_dict[encoder_6_1_key].shape
        logger.info(f"face_encoder_blocks.6.1 kernel shape: {kernel_shape}")
        if len(kernel_shape) >= 3 and kernel_shape[2] == 1:  # 1x1卷积 -> 384版本
            if has_encoder_7:
                return "384_full"  # 完整的384版本（8层）
            else:
                return "384_original"  # 原始384版本（7层）

    # 检查face_decoder_blocks.3的通道数
    decoder_3_key = "face_decoder_blocks.3.0.conv_block.0.weight"
    if decoder_3_key in checkpoint_state_dict:
        decoder_3_shape = checkpoint_state_dict[decoder_3_key].shape
        logger.info(f"face_decoder_blocks.3 shape: {decoder_3_shape}")
        if decoder_3_shape[0] == 768 and decoder_3_shape[1] == 384:  # 384版本特征
            return "384_original"
        elif decoder_3_shape[0] == 1024 and decoder_3_shape[1] == 512:  # 256版本特征
            return "256"

    # 默认返回256版本（向后兼容）
    logger.warning("无法确定模型版本，默认使用256版本")
    return "256"


def create_compatible_model(checkpoint_state_dict):
    """
    根据检查点创建兼容的模型架构
    """
    try:
        # 分析检查点中的层结构
        encoder_layers = set()
        decoder_layers = set()

        for key in checkpoint_state_dict.keys():
            if "face_encoder_blocks." in key:
                layer_num = int(key.split(".")[1])
                encoder_layers.add(layer_num)
            elif "face_decoder_blocks." in key:
                layer_num = int(key.split(".")[1])
                decoder_layers.add(layer_num)

        logger.info(f"检查点中的编码器层: {sorted(encoder_layers)}")
        logger.info(f"检查点中的解码器层: {sorted(decoder_layers)}")

        # 如果有8层编码器，尝试使用384版本但跳过不匹配的层
        if 7 in encoder_layers:
            logger.info("检测到8层编码器，使用384版本模型")
            from wav2lip.models.wav2lip import Wav2Lip as Wav2Lip384

            model = Wav2Lip384()

            # 尝试部分加载
            model_dict = model.state_dict()
            compatible_dict = {}

            for k, v in checkpoint_state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    logger.debug(f"跳过不兼容的层: {k}")

            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict)
            logger.info(f"成功加载 {len(compatible_dict)} 个兼容层")
            return model

        return None

    except Exception as e:
        logger.error(f"创建兼容模型失败: {e}")
        return None


def load_model(path):
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace("module.", "")] = v

    # 检测模型版本
    model_version = detect_model_version(new_s)
    logger.info(f"检测到模型版本: {model_version}")

    # 尝试加载策略列表
    strategies = []

    if "384" in model_version:
        strategies = [
            ("384版本", "wav2lip.models.wav2lip", "Wav2Lip"),
            ("256版本", "wav2lip.models.wav2lip_v2", "Wav2Lip"),
        ]
    else:
        strategies = [
            ("256版本", "wav2lip.models.wav2lip_v2", "Wav2Lip"),
            ("384版本", "wav2lip.models.wav2lip", "Wav2Lip"),
        ]

    # 尝试每种策略
    for strategy_name, module_name, class_name in strategies:
        try:
            logger.info(f"尝试使用{strategy_name}模型...")
            module = __import__(module_name, fromlist=[class_name])
            ModelClass = getattr(module, class_name)
            model = ModelClass()

            # 尝试加载，先严格模式，再宽松模式
            try:
                model.load_state_dict(new_s, strict=True)
                logger.info(f"✅ {strategy_name}模型加载成功（严格模式）")
                break
            except Exception:
                model.load_state_dict(new_s, strict=False)
                logger.info(f"✅ {strategy_name}模型加载成功（兼容模式）")
                break

        except Exception as e:
            logger.warning(f"❌ {strategy_name}模型加载失败: {str(e)[:100]}...")
            continue
    else:
        # 所有策略都失败，给出明确的解决建议
        logger.error("🚨 所有模型加载策略都失败了！")
        logger.error("💡 建议解决方案：")
        logger.error("1. 下载官方 wav2lip256.pth 模型")
        logger.error("2. 重命名为 wav2lip.pth 并放在 ./models/ 目录")
        logger.error("3. 下载地址：https://pan.quark.cn/s/83a750323ef0")
        raise RuntimeError("无法加载任何兼容的模型架构，请使用官方推荐的模型文件")

    model = model.to(device)
    return model.eval()


def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"

    with open(coords_path, "rb") as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, "*.[jpJP][pnPN]*[gG]"))
    input_img_list = sorted(
        input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    frame_list_cycle = read_imgs(input_img_list)
    # self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    input_face_list = glob.glob(os.path.join(face_imgs_path, "*.[jpJP][pnPN]*[gG]"))
    input_face_list = sorted(
        input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    face_list_cycle = read_imgs(input_face_list)

    return frame_list_cycle, face_list_cycle, coord_list_cycle


@torch.no_grad()
def warm_up(batch_size, model, modelres):
    # 增强预热函数 - 多轮预热确保稳定性
    logger.info("warmup model...")

    # 进行多轮预热，模拟真实使用场景
    for i in range(5):  # 增加预热轮数
        logger.info(f"warmup round {i+1}/5...")

        # 使用不同的随机数据模拟真实场景
        img_batch = torch.randn(batch_size, 6, modelres, modelres).to(device)
        mel_batch = torch.randn(batch_size, 1, 80, 16).to(device)

        # 执行推理
        _ = model(mel_batch, img_batch)

        # 清理GPU缓存，强制内存重新分配
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("warmup completed - model should be stable now")


def read_imgs(img_list):
    frames = []
    logger.info("reading images...")
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def __mirror_index(size, index):
    # size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


def inference(
    quit_event,
    batch_size,
    face_list_cycle,
    audio_feat_queue,
    audio_out_queue,
    res_frame_queue,
    model,
):

    # model = load_model("./models/wav2lip.pth")
    # input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    # input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # face_list_cycle = read_imgs(input_face_list)

    # input_latent_list_cycle = torch.load(latents_out_path)
    length = len(face_list_cycle)
    index = 0
    count = 0
    counttime = 0
    logger.info("start inference")
    while not quit_event.is_set():
        starttime = time.perf_counter()
        mel_batch = []
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        is_all_silence = True
        audio_frames = []
        for _ in range(batch_size * 2):
            frame, type, eventpoint = audio_out_queue.get()
            audio_frames.append((frame, type, eventpoint))
            if type == 0:
                is_all_silence = False

        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put(
                    (
                        None,
                        __mirror_index(length, index),
                        audio_frames[i * 2 : i * 2 + 2],
                    )
                )
                index = index + 1
        else:
            # print('infer=======')
            t = time.perf_counter()
            img_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length, index + i)
                face = face_list_cycle[idx]
                img_batch.append(face)
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, face.shape[0] // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
                device
            )
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(
                device
            )

            with torch.no_grad():
                pred = model(mel_batch, img_batch)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            counttime += time.perf_counter() - t
            count += batch_size
            # _totalframe += 1
            if count >= 100:
                logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                count = 0
                counttime = 0
            for i, res_frame in enumerate(pred):
                # self.__pushmedia(res_frame,loop,audio_track,video_track)
                res_frame_queue.put(
                    (
                        res_frame,
                        __mirror_index(length, index),
                        audio_frames[i * 2 : i * 2 + 2],
                    )
                )
                index = index + 1
            # print('total batch time:',time.perf_counter()-starttime)
    logger.info("lipreal inference processor stop")


class LipReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        # self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        # self.W = opt.W
        # self.H = opt.H

        self.fps = opt.fps  # 20 ms per frame

        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = Queue(self.batch_size * 2)  # mp.Queue
        # self.__loadavatar()
        self.model = model
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar

        # 加载蒙版图像
        mask_path = os.path.join(".", "models", "mask.png")
        self.mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_img is None:
            raise ValueError("找不到蒙版图像: " + mask_path)

        # 颜色匹配配置参数
        self.enable_color_matching = getattr(opt, 'enable_color_matching', True)
        self.color_matching_strength = getattr(opt, 'color_matching_strength', 0.6)

        # 添加渐变过渡相关变量
        self.last_generated_frame = None  # 保存最后一帧生成的面部
        self.transition_frames = 10  # 过渡帧数（约0.4秒）
        self.current_transition_frame = 0  # 当前过渡帧计数
        self.is_transitioning = False  # 是否正在过渡中

        self.asr = LipASR(opt, self)
        self.asr.warm_up()

        self.render_event = mp.Event()

    def __del__(self):
        logger.info(f"lipreal({self.sessionid}) delete")

    def paste_back_frame(self, pred_frame, idx: int):
        """将预测的面部帧粘贴回原始帧，如果静音则保留最后一帧的嘴型"""
        try:
            bbox = self.coord_list_cycle[idx]
            combine_frame = copy.deepcopy(self.frame_list_cycle[idx])

            frame_to_apply = None

            if pred_frame is not None:
                # 有新的预测帧，更新最后一帧
                self.last_generated_frame = pred_frame.copy()
                frame_to_apply = pred_frame
                logger.debug(
                    f"lipreal: 使用新预测帧，已更新last_generated_frame，idx={idx}"
                )
            elif self.last_generated_frame is not None:
                # 静音，但有历史帧，使用历史帧
                frame_to_apply = self.last_generated_frame
                logger.debug(f"lipreal: 静音状态，使用保留的最后一帧嘴型，idx={idx}")
            else:
                logger.debug(f"lipreal: 静音状态且无历史帧，使用原始帧，idx={idx}")

            if frame_to_apply is not None:
                # 如果有需要应用的帧，则进行融合
                y1, y2, x1, x2 = bbox
                res_frame = cv2.resize(
                    frame_to_apply.astype(np.uint8), (x2 - x1, y2 - y1)
                )

                if hasattr(self, "mask_img") and self.mask_img is not None:
                    combine_frame = self._apply_mask_blend(
                        combine_frame, res_frame, bbox
                    )
                else:
                    combine_frame[y1:y2, x1:x2] = res_frame

            # 如果 frame_to_apply is None (既没有新帧也没有历史帧)，则直接返回原始帧
            return combine_frame

        except Exception as e:
            logger.error(f"paste_back_frame error: {e}")
            return copy.deepcopy(self.frame_list_cycle[idx])

    def _apply_mask_blend(self, combine_frame, res_frame, bbox):
        """应用蒙版融合，包含颜色匹配"""
        y1, y2, x1, x2 = bbox

        # 调整蒙版大小到面部区域
        resized_mask = cv2.resize(self.mask_img, (x2 - x1, y2 - y1))
        _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 创建距离变换用于羽化效果
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        feather_radius = 15  # 羽化半径，控制融合边缘的柔和程度
        alpha = np.clip(dist / feather_radius, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)  # 高斯模糊平滑边缘

        # 获取原始面部和生成面部
        original_face = combine_frame[y1:y2, x1:x2].astype(np.float32)
        generated_face = res_frame.astype(np.float32)

        # 应用颜色匹配（如果启用）
        if self.enable_color_matching:
            color_matched_face = self._apply_color_matching(generated_face, original_face, alpha)
        else:
            color_matched_face = generated_face

        # 融合图像：alpha融合颜色匹配后的生成面部和原始面部
        blended_face = (
            alpha[..., None] * color_matched_face + (1 - alpha[..., None]) * original_face
        ).astype(np.uint8)

        combine_frame[y1:y2, x1:x2] = blended_face
        return combine_frame

    def _apply_color_matching(self, generated_face, original_face, alpha):
        """应用颜色匹配，使生成的嘴型颜色与原始面部更匹配"""
        try:
            # 创建嘴唇区域蒙版（alpha值较高的区域）
            lip_mask = (alpha > 0.3).astype(np.float32)

            if np.sum(lip_mask) == 0:
                return generated_face

            # 计算原始面部和生成面部在嘴唇区域的平均颜色
            original_lip_color = self._get_average_color(original_face, lip_mask)
            generated_lip_color = self._get_average_color(generated_face, lip_mask)

            # 计算颜色差异
            color_diff = original_lip_color - generated_lip_color

            # 应用颜色校正，强度根据alpha值调整
            color_matched_face = generated_face.copy()
            for c in range(3):  # BGR三个通道
                adjustment = color_diff[c] * alpha * self.color_matching_strength
                color_matched_face[:, :, c] = np.clip(
                    color_matched_face[:, :, c] + adjustment, 0, 255
                )

            return color_matched_face

        except Exception as e:
            logger.warning(f"颜色匹配失败，使用原始生成面部: {e}")
            return generated_face

    def _get_average_color(self, image, mask):
        """计算蒙版区域的平均颜色"""
        masked_pixels = image * mask[..., None]
        total_weight = np.sum(mask)
        if total_weight == 0:
            return np.array([0, 0, 0])

        avg_color = np.sum(masked_pixels, axis=(0, 1)) / total_weight
        return avg_color

    def _apply_transition_blend(
        self, combine_frame, res_frame, bbox, transition_weight
    ):
        """应用过渡融合"""
        y1, y2, x1, x2 = bbox

        # 调整蒙版大小到面部区域
        resized_mask = cv2.resize(self.mask_img, (x2 - x1, y2 - y1))
        _, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # 创建距离变换用于羽化效果
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        feather_radius = 15
        alpha = np.clip(dist / feather_radius, 0, 1)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

        # 应用过渡权重到alpha通道
        alpha = alpha * transition_weight

        # 融合图像
        original_face = combine_frame[y1:y2, x1:x2].astype(np.float32)
        generated_face = res_frame.astype(np.float32)
        blended_face = (
            alpha[..., None] * generated_face + (1 - alpha[..., None]) * original_face
        ).astype(np.uint8)

        combine_frame[y1:y2, x1:x2] = blended_face
        return combine_frame

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        # if self.opt.asr:
        #     self.asr.warm_up()

        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(
            target=self.process_frames,
            args=(quit_event, loop, audio_track, video_track),
        )
        process_thread.start()

        Thread(
            target=inference,
            args=(
                quit_event,
                self.batch_size,
                self.face_list_cycle,
                self.asr.feat_queue,
                self.asr.output_queue,
                self.res_frame_queue,
                self.model,
            ),
        ).start()  # mp.Process

        # self.render_event.set() #start infer process render
        count = 0
        totaltime = 0
        _starttime = time.perf_counter()
        # _totalframe=0
        while not quit_event.is_set():
            # update texture every frame
            # audio stream thread...
            t = time.perf_counter()
            self.asr.run_step()

            # if video_track._queue.qsize()>=2*self.opt.batch_size:
            #     print('sleep qsize=',video_track._queue.qsize())
            #     time.sleep(0.04*video_track._queue.qsize()*0.8)
            if video_track and video_track._queue.qsize() >= 5:
                logger.debug("[lipreal] sleep qsize=%d", video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)

            # delay = _starttime+_totalframe*0.04-time.perf_counter() #40ms
            # if delay > 0:
            #     time.sleep(delay)
        # self.render_event.clear() #end infer process render
        logger.info("lipreal thread stop")
