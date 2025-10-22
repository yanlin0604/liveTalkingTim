<div align="center">

**支持实时交互、直播推流、语音识别、AI对话的全栈数字人解决方案**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 📖 项目简介

LiveTalking 是一个功能完整的**智能数字人交互系统**，集成了先进的深度学习模型和实时流媒体技术，支持：

- 🎭 **交互式数字人** - 实时音视频对话，自然表情与唇形同步
- 📺 **直播数字人** - RTMP/WebRTC推流，支持多平台直播
- 🎤 **语音识别(ASR)** - 实时语音转文字，支持多种ASR引擎
- 🗣️ **文本转语音(TTS)** - 高质量语音合成，支持多种声音克隆
- 🤖 **AI大模型对话** - 集成多种LLM，智能对话交互
- 💬 **弹幕互动** - 实时弹幕监听与智能回复
- 🎬 **多种渲染模式** - MuseTalk/Wav2Lip/Ultralight 多种数字人生成方式

---

## ✨ 核心功能

### 🎯 交互数字人
- **实时对话交互**：支持语音/文本输入，实时生成数字人视频输出
- **自然表情同步**：基于深度学习的表情驱动，唇形精准同步
- **低延迟响应**：优化的异步处理架构，毫秒级响应
- **多人物切换**：支持动态切换不同数字人形象

### 📡 直播数字人
- **RTMP推流**：异步RTMP推流架构，支持抖音/快手/B站等主流平台
- **WebRTC推流**：低延迟实时流媒体传输
- **清晰度自适应**：支持蓝光/高清/普通/流畅四级清晰度动态切换
- **推流监控**：实时码率、帧率、丢帧率统计
- **断线重连**：智能重连机制，保障推流稳定性

### 🎤 语音识别(ASR)
支持多种ASR引擎：
- **MuseASR** - 高精度中文语音识别
- **HubertASR** - 基于Hubert模型的语音识别
- **LipASR** - 唇语识别增强

### 🗣️ 文本转语音(TTS)
集成多种TTS服务：
- **Edge TTS** - 微软Edge浏览器TTS
- **GPT-SoVITS** - 高质量声音克隆
- **CosyVoice** - 阿里云语音合成
- **豆包TTS** - 字节跳动语音服务
- **XTTS** - Coqui开源TTS
- **IndexTTS-V2** - blibli语音服务

支持语速、音量、音调实时调整

### 🤖 AI大模型集成
- **Ollama** - 本地部署大模型
- **DashScope** - 阿里云通义千问
- **MaxKB** - 知识库问答系统
- **Unimed** - 自定义API集成
- **Dify** - Dify知识库
- 支持流式对话和上下文记忆

### 💬 弹幕互动系统
- **实时弹幕监听**：WebSocket连接，毫秒级弹幕获取
- **智能优先回复**：弹幕回复优先于定时播报
- **定时播报**：支持自动播报和冷场填充
- **会话管理**：多会话隔离，自动过期清理

### 🎨 多种渲染引擎
- **MuseTalk** - 基于MuseV的高质量数字人生成
- **Wav2Lip** - 经典唇形同步模型，支持256/384分辨率
- **Ultralight** - 轻量级快速渲染方案

---

## 🏗️ 技术架构

```
LiveTalking/
├── 🎯 核心服务
│   ├── app.py                    # 主服务入口
│   ├── management_server.py      # 管理服务器
│   ├── basereal.py               # 核心实时处理引擎
│   └── barrage_websocket.py      # 弹幕WebSocket服务
│
├── 🎭 数字人渲染
│   ├── musetalk/                 # MuseTalk渲染模块
│   ├── wav2lip/                  # Wav2Lip渲染模块
│   └── ultralight/               # Ultralight渲染模块
│
├── 🎤 语音处理
│   ├── baseasr.py                # ASR基类
│   ├── museasr.py                # MuseASR实现
│   ├── hubertasr.py              # HubertASR实现
│   ├── lipasr.py                 # LipASR实现
│   └── ttsreal.py                # TTS统一接口
│
├── 🤖 AI大模型
│   ├── llm.py                    # LLM基类
│   └── llm_providers/            # LLM提供商实现
│       ├── ollama_provider.py
│       ├── dashscope_provider.py
│       ├── maxkb_provider.py
│       └── unimed_provider.py
│
├── 📡 流媒体推流
│   ├── rtmp/                     # RTMP推流组件
│   ├── webrtc.py                 # WebRTC推流
│   └── api/rtmp.py               # RTMP API接口
│
├── 🌐 Web管理界面
│   ├── web/                      # Web控制台
│   ├── templates/                # HTML模板
│   └── swagger.py                # API文档
│
└── ⚙️ 配置与工具
    ├── config.json               # 主配置文件
    ├── config_manager.py         # 配置管理器
    ├── dynamic_config.py         # 动态配置
    └── logger.py                 # 日志系统
```

---

## 🚀 快速开始

### 环境要求
- **Python**: 3.10+
- **CUDA**: 11.7+ (GPU推理)
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议 16GB+
- **显卡**: NVIDIA GPU (4GB+ VRAM)

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/yourusername/LiveTalking.git
cd LiveTalking
```

#### 2. 创建虚拟环境
```bash
conda create -n livetalking python=3.10
conda activate livetalking
```

#### 3. 安装依赖
```bash
# 安装PyTorch (根据您的CUDA版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt
```

#### 4. 下载模型文件
将预训练模型放置到 `models/` 目录：
- Wav2Lip模型: `models/checkpoint_step000760000.pth`
- MuseTalk模型: `models/musetalk/`
- 其他模型根据需要下载

#### 5. 配置系统
复制配置示例并修改：
```bash
cp config_examples/config.json config.json
# 编辑 config.json 配置您的参数
```

#### 6. 启动服务
```bash
# 方式1: 启动主服务
python app.py --config_file config.json

# 方式2: 使用启动脚本 (Linux/WSL)
bash start.sh

# 方式3: 启动管理服务器
python management_server.py
```

---

## 🎮 使用说明

### Web控制台
访问 `http://localhost:8010` 打开Web管理界面

主要功能：
- 📊 **实时监控** - 查看推流状态、FPS、码率等
- 🎨 **数字人管理** - 切换形象、调整参数
- ⚙️ **配置管理** - 动态调整TTS、LLM、推流配置
- 📝 **日志查看** - 实时日志输出和历史查询

### API接口

#### 文本转语音
```bash
POST /tts
Content-Type: application/json

{
  "text": "你好，我是数字人",
  "session_id": "session001"
}
```

#### 语音对话
```bash
POST /human
Content-Type: application/json

{
  "type": "full",
  "text": "今天天气怎么样？",
  "session_id": "session001"
}
```

#### 启动RTMP推流
```bash
POST /rtmp/start
Content-Type: application/json

{
  "rtmp_url": "rtmp://live.example.com/live/stream",
  "quality": "high"
}
```

#### 切换清晰度
```bash
POST /rtmp/switch_quality
Content-Type: application/json

{
  "quality": "low"  // ultra/high/medium/low
}
```

#### 弹幕互动
```bash
POST /barrage/start
Content-Type: application/json

{
  "sessionid": "barrage001",
  "mode": "ws",
  "ws_url": "ws://example.com/barrage"
}
```

完整API文档访问：`http://localhost:8010/swagger`

---

## ⚙️ 配置说明

### 主要配置项 (config.json)

#### 数字人渲染配置
```json
{
  "model": "wav2lip",              // 渲染模型: musetalk/wav2lip/ultralight
  "avatar_id": "avatar_name",      // 数字人ID
  "batch_size": 12,                // 推理批次大小
  "wav2lip_model_size": "384",     // Wav2Lip分辨率: 256/384
  "enable_color_matching": true,   // 启用颜色匹配
  "color_matching_strength": 0.9   // 颜色匹配强度
}
```

#### TTS配置
```json
{
  "tts": "doubao",                 // TTS类型: edgetts/xtts/gpt-sovits/cosyvoice/doubao
  "REF_FILE": "reference_audio",   // 参考音频
  "REF_TEXT": "参考文本",
  "TTS_SERVER": "http://127.0.0.1:9880",
  "doubao_audio": {
    "speed_ratio": 1,              // 语速 [0.2-3]
    "volume_ratio": 1,             // 音量 [0.1-3]
    "pitch_ratio": 1               // 音高 [0.1-3]
  }
}
```

#### LLM配置
```json
{
  "llm_provider": "ollama",        // LLM提供商: ollama/dashscope/maxkb/unimed
  "llm_model": "qwen2:0.5b",       // 模型名称
  "llm_system_prompt": "你是一个友善的助手",
  "llm_api_key": "",               // API密钥
  "ollama_host": "http://localhost:11434"
}
```

#### 推流配置
```json
{
  "transport": "webrtc",           // 传输方式: webrtc/rtcpush/virtualcam
  "push_url": "http://localhost:1985/rtc/v1/whip/",
  "streaming_quality": {
    "target_fps": 25,              // 目标帧率
    "max_bitrate": 1500000,        // 最大码率(bps)
    "min_bitrate": 300000,         // 最小码率(bps)
    "enable_quality_monitoring": true
  }
}
```

---

## 🎨 数字人素材管理

### 素材目录结构
```
data/
├── avatars/                       # 数字人形象
│   └── avatar_name/
│       ├── avatar.mp4            # 视频素材
│       ├── avatar.png            # 封面图片
│       └── coords.pkl            # 坐标数据
│
├── actions/                       # 动作素材
│   └── action_name/
│       └── audio.wav             # 动作音频
│
└── custom_config.json            # 自定义动作配置
```

### 扫描数字人素材
```bash
# 扫描动作素材
python start_action_scanner.py --scan_dir data/avatars

# 扫描视频素材
python video_scanner.py
```

---

## 🔧 高级功能

### 多动作模式
支持三种动作切换模式：
- **single** - 单个动作循环
- **random** - 随机切换动作
- **sequence** - 顺序播放动作

配置示例：
```json
{
  "multi_action_mode": "random",
  "multi_action_list": ["action1", "action2", "action3"],
  "multi_action_interval": 100,
  "multi_action_switch_policy": "interval"
}
```

### RTMP清晰度级别
- **蓝光 (ultra)**: CRF=18, 3600k码率, 192k音频
- **高清 (high)**: CRF=21, 2400k码率, 128k音频
- **普通 (medium)**: 1200k码率, 96k音频
- **流畅 (low)**: 600k码率, 64k音频

### 弹幕智能调度
- 弹幕回复优先于定时播报
- 检测正在说话状态，避免打断
- 支持延迟检查和重试机制
- 会话失效自动停服

---

## 📊 性能优化

### 推理优化
- **自动批处理** - 根据传输类型自动调整batch_size
- **队列管理** - 智能视频队列控制，避免积压
- **帧率控制** - 动态调整FPS，平衡质量与性能

### 推流优化
- **异步架构** - 非阻塞视频/音频队列处理
- **码率自适应** - 根据网络状况动态调整
- **断线重连** - 指数退避重连机制
- **时间戳管理** - 全局时间戳同步，避免冲突

### 内存优化
- **模型缓存** - 智能模型加载与卸载
- **队列限制** - 防止内存溢出
- **垃圾回收** - 及时释放资源

---

## 📝 日志与监控

### 日志文件
```
logs/
├── app.log                       # 主服务日志
├── management_server.log         # 管理服务日志
├── barrage_websocket.log         # 弹幕服务日志
└── rtmp_stream.log               # 推流日志
```

### 日志查看工具
```bash
# 启动Web日志查看器
python log_reader_web.py

# 访问 http://localhost:5000
```

### 性能监控
访问 `/rtmp/stats` 查看实时推流统计：
- 当前FPS
- 实时码率
- 编码错误率
- 重连次数
- 队列状态

---

## 🐛 故障排查

### 常见问题

#### 1. RTMP推流失败
```bash
# 检查RTMP URL是否正确
# 检查网络连接
# 查看日志: logs/rtmp_stream.log
```

#### 2. 唇形不同步
```bash
# 调整batch_size (降低延迟)
# 检查音频采样率 (必须44100Hz)
# 启用颜色匹配增强效果
```

#### 3. GPU内存不足
```bash
# 降低batch_size
# 降低模型分辨率 (384->256)
# 使用轻量级模型 (ultralight)
```

#### 4. TTS语音异常
```bash
# 检查TTS服务是否启动
# 验证TTS_SERVER地址
# 查看ttsreal.py日志
```

#### 5. LLM回复慢
```bash
# 使用本地Ollama小模型
# 检查网络延迟
# 启用流式响应
```

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 开发流程
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交改动 (`git commit -m '添加某个特性'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 代码规范
- Python代码遵循 PEP 8
- 使用 `black` 格式化代码
- 使用 `isort` 排序导入
- 添加中文注释说明


---

## 📄 开源协议

本项目基于 [Apache-2.0 License](LICENSE) 开源

---

## 🙏 致谢

本项目基于 [lipku/LiveTalking](https://github.com/lipku/LiveTalking) 进行改进和扩展开发。

同时感谢以下开源项目：
- [LiveTalking (原始项目)](https://github.com/lipku/LiveTalking) - 数字人实时交互框架
- [MuseTalk](https://github.com/TMElyralab/MuseTalk) - 音频驱动的数字人生成
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - 唇形同步模型
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - 声音克隆TTS
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 语音合成模型
- [Ollama](https://github.com/ollama/ollama) - 本地大语言模型运行环境

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个Star支持一下！**

</div>