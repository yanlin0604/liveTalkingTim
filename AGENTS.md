# Repository Guidelines

## 项目结构与模块组织
LiveTalking 根目录同时存放核心服务与运行素材。主要 Python 入口涵盖 `app.py`、`management_server.py`、`barrage_websocket.py` 以及多类 `start_*.py` 启动脚本；流媒体管线集中在 `rtmp/`，语言模型适配位于 `llm_providers/`，语音与唇形处理拆分在 `musetalk/`、`wav2lip/`、`ultralight/`。可复用配置置于 `config/`，示例覆盖存放于 `config_examples/`。Web 控制台和工具在 `web/`，大模型权重与素材归档在 `models/`、`assets/`、`data/`，面向使用者的文档集中于 `docs/`。

## 构建、测试与开发命令
推荐先激活项目虚拟环境：
```bash
conda activate nerfstream
python app.py --config_file config.json        # 启动主服务栈
bash start.sh                                  # 在 Linux/WSL 启动 RTMP/WebRTC 辅助进程
python start_action_scanner.py --scan_dir ...  # 运行动作/头像扫描
python barrage_websocket.py --mode ws ...      # 独立模式下监听弹幕
```
如需扩展新流程，优先复用或继承现有 `start_*.py` 脚本，以保持部署方式一致。

## 代码风格与命名约定
Python 模块使用 snake_case 文件名与 4 空格缩进，遵循 PEP 8，并统一通过 `logger.py` 记录日志。JSON 配置键保持小写加下划线，便于脚本解析。`web/` 中的 HTML/JS 约定 2 空格缩进，DOM ID 采用 camelCase。提交前请运行 `black`（行宽 88）与 `isort`；若引入前端构建链，可补充 `eslint --fix`。

## 测试指南
当前缺少自动化覆盖，新功能需基于 `pytest` 自建测试。建议在 `tests/` 下镜像模块路径（如 `tests/test_management_server.py`），用功能点命名夹具。涉及集成改动时，可在 `examples/` 或 `docs/` 添加冒烟脚本并记录执行步骤。提交前务必运行 `python -m pytest -q`，关键日志请保存于 `logs/` 目录便于排查。

## 提交与合并请求规范
Git 历史存在占位提交（如 `1`）与规范主题并存。请采用命令式、72 字符以内的提交标题，并在正文说明配置/协议影响、回滚方案与风险。合并请求需关联任务编号，列出测试结果（包含 `pytest` 与手动接口/CURL 验证），UI 或 API 变更应附截图或请求示例。任何默认配置调整或新增密钥需求都要在描述中明确提醒。

## 沟通与语言要求
所有提交说明、评审讨论与自动应答须使用简体中文，确保团队对齐；若涉及第三方英文日志，请同时补充中文概述。

## 安全与配置提示
严禁提交真实凭据或私钥，将模板放入 `config_examples/` 并通过环境变量驱动实际配置（参考 `config_manager.py`）。大模型与音视频素材请保持在 `.gitignore` 控制范围，优先在 `docs/` 提供下载指引而非直接入库。