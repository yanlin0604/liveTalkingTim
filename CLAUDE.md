# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LiveTalking 是一个数字人直播系统，支持实时语音交互、弹幕监听、多会话管理和RTMP推流。系统采用模块化架构，包含数字人渲染、语音合成、自然语言处理、弹幕处理等核心功能。

## 常用开发命令

### 服务启动和管理
```bash
# 启动主服务（默认分离式模式）
python app.py --config_file config.json

# 启动管理服务器
python management_server.py --port 8011

# 使用shell脚本启动（推荐）
./start.sh                    # 分离式服务模式
./start.sh --single          # 单服务模式
./start.sh --main-port 8020 --management-port 8021  # 自定义端口

# 停止服务
./stop.sh

# 检查服务状态
./status.sh
```

### 扫描和处理工具
```bash
# 扫描动作视频（默认模式）
python start_action_scanner.py

# 单次扫描
python start_action_scanner.py --once

# 指定扫描目录
python start_action_scanner.py --scan_dir my_action_videos

# 扫描头像视频
python start_scanner.py --mode action
python start_scanner.py --mode both
```

### 弹幕监听
```bash
# 独立启动弹幕监听
python barrage_websocket.py --mode ws --uri ws://127.0.0.1:8080/websocket \
  --human_url http://127.0.0.1:8010/human \
  --config config/barrage_config.json
```

## 核心架构

### 主要模块结构

1. **主服务 (app.py)**: 基于Flask的Web服务，提供数字人核心功能
   - 支持WebRTC、RTMP推流、虚拟摄像头等多种传输方式
   - 集成TTS、ASR、LLM等AI服务
   - 提供REST API和WebSocket接口

2. **管理服务器 (management_server.py)**: 统一的管理服务
   - 弹幕监听进程管理（多会话支持）
   - 配置热加载（话术模板、敏感词、定时任务等）
   - 动态会话管理和监控

3. **数字人渲染引擎 (basereal.py)**: 核心渲染模块
   - 支持多种数字人模型（musetalk、wav2lip、ultralight）
   - 实时音视频处理和同步
   - 多动作模式和颜色匹配

4. **弹幕处理系统 (barrage_websocket.py)**: 弹幕监听和响应
   - 支持多种弹幕类型（DANMU、GIFT、SUPER_CHAT等）
   - 智能回复规则和模板匹配
   - 敏感词过滤和限流控制

5. **API模块 (api/)**: 重构后的API接口
   - `api/webrtc.py` - WebRTC相关接口
   - `api/chat.py` - 聊天对话接口  
   - `api/rtmp.py` - RTMP推流接口
   - `api/tts.py` - 语音合成接口
   - `api/avatars.py` - 数字人管理接口

### 配置系统

- **主配置文件**: `config.json` - 系统核心配置
- **弹幕配置**: `config/barrage_config.json` - 弹幕处理配置
- **话术配置**: `config/speech_config.json` - 回复模板和规则
- **敏感词配置**: `config/sensitive_config.json` - 内容过滤配置
- **定时任务配置**: `config/schedule_config.json` - 自动播报和冷场填充

### AI服务集成

- **TTS服务**: 支持edge-tts、XTTS、GPT-Sovits、CosyVoice、豆包TTS
- **LLM服务**: 支持阿里云DashScope、Ollama本地模型、MaxKB知识库
- **ASR服务**: 支持多种语音识别引擎

## 关键特性

### 多会话管理
- 支持多个独立的数字人会话实例
- 每个会话使用独立的配置文件和进程
- 通过management_server.py统一管理

### 配置热加载
- 所有配置文件支持实时热加载
- 无需重启服务即可生效
- 自动检测配置文件变更

### 智能弹幕处理
- 基于规则的智能回复
- 支持正则表达式匹配
- 敏感词过滤和内容审核
- 冷场自动填充和定时播报

### 多种传输方式
- **WebRTC**: 低延迟实时通信
- **RTMP推流**: 直播平台推流
- **虚拟摄像头**: 视频会议软件集成

## 开发注意事项

### 配置文件管理
- 所有配置文件都有详细的字段说明（见config.json的_descriptions字段）
- 修改配置后系统会自动热加载，无需重启
- 新增配置项时需要在_descriptions中添加说明

### 日志管理
- 主服务日志: `logs/main.log`
- 管理服务器日志: `logs/management.log`
- 使用log_reader.py可以实时查看日志

### 性能优化
- batch_size参数影响推理延迟和效率
- auto_batch_size可根据传输类型自动调整
- 流式传输模式下需要关注队列管理和帧率控制

### 错误处理
- 所有API接口都返回统一格式：`{"code": 0, "msg": "ok", "data": {...}}`
- code=0表示成功，非零表示错误
- 重要操作都有日志记录和错误处理

## 故障排除

### 常见问题
1. **端口占用**: 使用`--main-port`和`--management-port`指定不同端口
2. **配置错误**: 检查config.json格式和必填字段
3. **模型加载失败**: 确认model_path路径正确且文件存在
4. **TTS服务不可用**: 检查TTS_SERVER地址和网络连接

### 调试技巧
- 使用`./status.sh`检查服务运行状态
- 查看日志文件了解详细错误信息
- 使用独立的启动脚本进行模块化测试