import time
import os
import re
import logging
import requests
import json
from logging.handlers import RotatingFileHandler
from basereal import BaseReal
from logger import logger

# MaxKB 会话缓存：sessionid -> chat_id 映射
_maxkb_chat_cache = {}

# 调试模式开关 - 设置为True启用DashScope详细调试
DASHSCOPE_DEBUG_MODE = False

if DASHSCOPE_DEBUG_MODE:
    from dashscope_debug import debug_dashscope_response

# === LLM 专用日志（独立文件，便于排查是否有发送文本以触发TTS）===
# 日志位置: logs/llm.log，大小轮转
_LLM_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
_LLM_LOG_PATH = os.path.join(_LLM_LOG_DIR, "llm.log")

def _setup_llm_logger():
    """初始化 LLM 专用日志记录器（单例）。
    - 文件: logs/llm.log
    - 轮转: 2MB, 保留 5 个历史
    - 编码: utf-8
    """
    ll = logging.getLogger("llm")
    if ll.handlers:
        return ll

    ll.setLevel(logging.DEBUG)
    try:
        os.makedirs(_LLM_LOG_DIR, exist_ok=True)
    except Exception:
        # 目录创建失败不应中断主流程，回退为仅控制台
        pass

    try:
        fh = RotatingFileHandler(_LLM_LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        ll.addHandler(fh)
    except Exception:
        # 文件句柄失败时不抛出，至少保留一个空记录器
        pass

    ll.propagate = False  # 不向上冒泡，避免与全局logger重复
    return ll

llm_logger = _setup_llm_logger()


def llm_response(message, nerfreal: BaseReal):
    """
    LLM响应函数，支持多种LLM提供商
    支持的提供商：dashscope（阿里云）、ollama（本地）、maxkb（MaxKB知识库）
    """
    start = time.perf_counter()
    logger.info(f" === LLM响应开始 ===")
    logger.info(f" 用户消息: '{message[:50]}{'...' if len(message) > 50 else ''}'")
    llm_logger.info("=== LLM响应开始 ===")
    llm_logger.info("用户消息长度=%d 预览='%s'", len(message or ""), (message or "")[:200])

    # 获取LLM配置
    llm_provider = getattr(nerfreal.opt, "llm_provider", "dashscope")
    logger.info(f" LLM提供商: {llm_provider}")
    llm_logger.info("提供商=%s", llm_provider)

    if llm_provider == "ollama":
        logger.info(" 使用Ollama本地模型")
        llm_logger.info("使用Ollama本地模型")
        _ollama_response(message, nerfreal, start)
    elif llm_provider == "maxkb":
        logger.info(" 使用MaxKB知识库")
        llm_logger.info("使用MaxKB知识库")
        _maxkb_response(message, nerfreal, start)
    else:
        logger.info(" 使用阿里云DashScope")
        llm_logger.info("使用阿里云DashScope")
        _dashscope_response(message, nerfreal, start)


def _dashscope_response(message, nerfreal: BaseReal, start_time):
    """阿里云DashScope响应处理"""
    from openai import OpenAI
    llm_api_key = getattr(nerfreal.opt, 'llm_api_key', '')
    llm_logger.info("使用DashScope API Key: %s", llm_api_key)
    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key= llm_api_key,
        # 填写DashScope SDK的base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    end = time.perf_counter()
    logger.info(f"llm Time init (dashscope): {end-start_time}s")
    llm_logger.debug("dashscope 初始化耗时=%.3fs", end - start_time)

    model = getattr(nerfreal.opt, 'llm_model', 'qwen-plus')
    system_prompt = getattr(
        nerfreal.opt, "llm_system_prompt", "你是一位乐于助人的助手。"
    )
    llm_logger.info("请求模型=%s system_prompt_len=%d msg_len=%d", model, len(system_prompt or ""), len(message or ""))
    llm_logger.debug("system_prompt预览: %s", (system_prompt or "")[:200])

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        stream=True,
        # 通过以下设置，在流式输出的最后一行展示token使用信息
        stream_options={"include_usage": True},
    )

    llm_logger.info("dashscope 流式响应开始")

    _process_stream_response(completion, nerfreal, start_time)


def _ollama_response(message, nerfreal: BaseReal, start_time):
    """Ollama本地模型响应处理"""
    try:
        import ollama
    except ImportError:
        logger.error(
            "ollama package not found. Please install it with: pip install ollama"
        )
        llm_logger.error("未安装ollama包，无法使用本地模型")
        nerfreal.put_msg_txt("抱歉，Ollama服务不可用。")
        return

    # 获取Ollama配置
    ollama_host = getattr(nerfreal.opt, "ollama_host", "http://localhost:11434")
    model = getattr(nerfreal.opt, "llm_model", "llama3.2")
    system_prompt = getattr(
        nerfreal.opt, "llm_system_prompt", "You are a helpful assistant."
    )
    llm_logger.info("请求模型=%s host=%s system_prompt_len=%d msg_len=%d", model, ollama_host, len(system_prompt or ""), len(message or ""))

    try:
        # 创建Ollama客户端
        client = ollama.Client(host=ollama_host)

        end = time.perf_counter()
        logger.info(f"llm Time init (ollama): {end-start_time}s")
        llm_logger.debug("ollama 初始化耗时=%.3fs", end - start_time)

        # 发送请求到Ollama
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            stream=True,
        )

        llm_logger.info("ollama 流式响应开始")

        _process_ollama_stream_response(response, nerfreal, start_time)

    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        llm_logger.exception("Ollama 请求失败: %s", e)
        nerfreal.put_msg_txt("抱歉，无法连接到Ollama服务。")


def _process_stream_response(completion, nerfreal: BaseReal, start_time):
    """处理OpenAI兼容的流式响应"""
    result = ""
    first = True
    total_chars = 0
    usage_logged = False

    for chunk in completion:
        if len(chunk.choices) > 0:
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start_time}s")
                llm_logger.info("首包到达耗时=%.3fs", end - start_time)
                first = False

            msg = chunk.choices[0].delta.content
            if msg:
                llm_logger.debug("收到delta长度=%d 预览='%s'", len(msg), msg[:120])
                result = _process_message_chunk(msg, result, nerfreal)
                total_chars += len(msg)

        # 兼容 DashScope include_usage: True 的末包用量信息
        try:
            if hasattr(chunk, "usage") and chunk.usage and not usage_logged:
                usage_logged = True
                llm_logger.info(
                    "usage: prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    getattr(chunk.usage, "prompt_tokens", None),
                    getattr(chunk.usage, "completion_tokens", None),
                    getattr(chunk.usage, "total_tokens", None),
                )
        except Exception:
            pass

    end = time.perf_counter()
    logger.info(f" LLM响应总时间: {end-start_time:.2f}s")
    llm_logger.info("流式结束，总时间=%.3fs，总字符数=%d", end - start_time, total_chars)
    if result:
        logger.info(f" 发送最终TTS文本: '{result}'")
        llm_logger.info("发送最终TTS文本，长度=%d 预览='%s'", len(result), result[:200])
        nerfreal.put_msg_txt(result)
    logger.info(" === LLM响应完成 ===")
    llm_logger.info("=== LLM响应完成（dashscope）===")


def _process_ollama_stream_response(response, nerfreal: BaseReal, start_time):
    """处理Ollama的流式响应"""
    result = ""
    first = True
    total_chars = 0

    for chunk in response:
        if first:
            end = time.perf_counter()
            logger.info(f"llm Time to first chunk: {end-start_time}s")
            llm_logger.info("首包到达耗时=%.3fs", end - start_time)
            first = False

        msg = chunk.get("message", {}).get("content", "")
        if msg:
            llm_logger.debug("收到delta长度=%d 预览='%s'", len(msg), msg[:120])
            result = _process_message_chunk(msg, result, nerfreal)
            total_chars += len(msg)

    end = time.perf_counter()
    logger.info(f" LLM响应总时间: {end-start_time:.2f}s")
    llm_logger.info("流式结束，总时间=%.3fs，总字符数=%d", end - start_time, total_chars)
    if result:
        # 清洗不可发音字符（例如表情符号），避免 TTS 无声
        def _sanitize_tts_text(text: str) -> str:
            import re
            # 去除常见 emoji 与变体选择符；保留中文、英文、数字与常见标点
            emoji_pattern = re.compile(
                "[\uFE0F\u200D\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAD6\U0001FAE0-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]",
                flags=re.UNICODE,
            )
            cleaned = emoji_pattern.sub("", text)
            return cleaned.strip()

        sanitized = _sanitize_tts_text(result)
        if sanitized:
            logger.info(f" 发送最终TTS文本: '{sanitized}'")
            llm_logger.info("发送最终TTS文本，长度=%d 预览='%s'", len(sanitized), sanitized[:200])
            nerfreal.put_msg_txt(sanitized)
        else:
            # 不再使用“好的。”兜底，避免用户听到重复拼接“好的”
            logger.warning(" 最终文本清洗后为空，本次不发送TTS文本")
            llm_logger.warning("清洗后为空，跳过TTS发送。原长=%d 预览='%s'", len(result), result[:200])
    logger.info(" === LLM响应完成 ===")
    llm_logger.info("=== LLM响应完成（ollama）===")


def _clean_text(s: str) -> str:
    """将大模型文本清洗为纯文本：
    - 去除Markdown代码块/行内反引号/粗斜体/删除线等标记
    - 去除标题/引用/列表等行首符号
    - Markdown链接/图片仅保留可读文字
    - 去除HTML标签与常见装饰性符号
    - 统一空白：换行/制表符转空格，多余空格合并
    """
    if not s:
        return s

    # 统一换行和制表为空格
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    # 去除围栏代码块 ```lang\n...``` 与 ```...```
    # 仅移除围栏标记，尽可能保留内部可读文字
    s = re.sub(r"```+\w*\n?", "", s)
    s = s.replace("```", "")

    # 去除行内代码反引号
    s = s.replace("`", "")

    # 行首的标题/引用/列表标记
    lines = s.split("\n")
    cleaned_lines = []
    for line in lines:
        # 去掉行首井号标题
        line = re.sub(r"^\s*#{1,6}\s+", "", line)
        # 去掉行首引用 >
        line = re.sub(r"^\s*>+\s+", "", line)
        # 去掉行首无序列表符号 - * +
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        # 去掉行首有序列表 1. 2. 等
        line = re.sub(r"^\s*\d+\.\s+", "", line)
        cleaned_lines.append(line)
    s = " ".join(cleaned_lines)

    # Markdown 图片与链接：保留可读文字部分
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)  # 图片
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)  # 链接

    # 去除常见的Markdown强调符号与删除线
    for ch in ["**", "__", "*", "_", "~~"]:
        s = s.replace(ch, "")

    # 去除HTML标签
    s = re.sub(r"<[^>]+>", "", s)

    # 去除常见装饰性符号（保留中英文标点）
    s = re.sub(r"[•·◦►–—]+", "", s)

    # 合并多余空格
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _maxkb_response(message, nerfreal: BaseReal, start_time):
    """MaxKB知识库响应处理"""
    # 获取MaxKB配置
    ollama_host = getattr(nerfreal.opt, "ollama_host", "http://localhost:8080")
    llm_api_key = getattr(nerfreal.opt, "llm_api_key", "")
    
    if not llm_api_key:
        logger.error("MaxKB API Key未配置")
        llm_logger.error("MaxKB API Key未配置")
        nerfreal.put_msg_txt("抱歉，MaxKB服务配置不完整。")
        return
    
    # 获取 sessionid 用于会话缓存
    sessionid = getattr(nerfreal, 'sessionid', 'default')
    llm_logger.info("使用MaxKB host=%s api_key=%s sessionid=%s msg_len=%d", ollama_host, llm_api_key[:10] + "...", sessionid, len(message or ""))
    
    try:
        # 第一步：获取或创建会话（基于 sessionid 缓存）
        chat_id = _maxkb_get_or_create_chat(ollama_host, llm_api_key, sessionid)
        if not chat_id:
            nerfreal.put_msg_txt("抱歉，无法创建MaxKB会话。")
            return
        
        end = time.perf_counter()
        logger.info(f"llm Time init (maxkb): {end-start_time}s")
        llm_logger.debug("maxkb 会话获取耗时=%.3fs chat_id=%s sessionid=%s", end - start_time, chat_id, sessionid)
        
        # 第二步：进行对话
        _maxkb_chat_message(ollama_host, llm_api_key, chat_id, message, nerfreal, start_time)
        
    except Exception as e:
        logger.error(f"MaxKB request failed: {e}")
        llm_logger.exception("MaxKB 请求失败: %s", e)
        nerfreal.put_msg_txt("抱歉，MaxKB服务暂时不可用。")


def _maxkb_get_or_create_chat(host, api_key, sessionid):
    """获取或创建MaxKB会话，基于sessionid缓存复用"""
    global _maxkb_chat_cache
    
    # 检查缓存中是否已有该 sessionid 的会话
    if sessionid in _maxkb_chat_cache:
        cached_chat_id = _maxkb_chat_cache[sessionid]
        llm_logger.info("复用已缓存的MaxKB会话 sessionid=%s chat_id=%s", sessionid, cached_chat_id)
        return cached_chat_id
    
    # 缓存中没有，创建新会话
    chat_id = _maxkb_create_new_chat(host, api_key)
    if chat_id:
        _maxkb_chat_cache[sessionid] = chat_id
        llm_logger.info("新建MaxKB会话并缓存 sessionid=%s chat_id=%s", sessionid, chat_id)
    
    return chat_id

def _maxkb_create_new_chat(host, api_key):
    """创建新的MaxKB会话，返回chat_id"""
    url = f"{host.rstrip('/')}/chat/api/open"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if data.get("code") == 200:
            chat_id = data.get("data")
            llm_logger.info("MaxKB会话创建成功 chat_id=%s", chat_id)
            return chat_id
        else:
            llm_logger.error("MaxKB会话创建失败 code=%s message=%s", data.get("code"), data.get("message"))
            return None
            
    except Exception as e:
        llm_logger.exception("MaxKB会话创建异常: %s", e)
        return None

def clear_maxkb_chat_cache(sessionid=None):
    """清除MaxKB会话缓存
    
    Args:
        sessionid: 指定要清除的sessionid，为None时清除所有缓存
    """
    global _maxkb_chat_cache
    
    if sessionid is None:
        # 清除所有缓存
        cleared_count = len(_maxkb_chat_cache)
        _maxkb_chat_cache.clear()
        llm_logger.info("已清除所有MaxKB会话缓存，共%d个", cleared_count)
    else:
        # 清除指定sessionid的缓存
        if sessionid in _maxkb_chat_cache:
            del _maxkb_chat_cache[sessionid]
            llm_logger.info("已清除MaxKB会话缓存 sessionid=%s", sessionid)
        else:
            llm_logger.warning("指定的sessionid不在缓存中 sessionid=%s", sessionid)


def _maxkb_chat_message(host, api_key, chat_id, message, nerfreal: BaseReal, start_time):
    """MaxKB对话消息处理"""
    url = f"{host.rstrip('/')}/chat/api/chat_message/{chat_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "message": message,
        "stream": True,
        "re_chat": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()
        
        llm_logger.info("maxkb 流式响应开始")
        _process_maxkb_stream_response(response, nerfreal, start_time)
        
    except Exception as e:
        llm_logger.exception("MaxKB对话请求异常: %s", e)
        raise


def _process_maxkb_stream_response(response, nerfreal: BaseReal, start_time):
    """处理MaxKB的流式响应"""
    result = ""
    first = True
    total_chars = 0
    
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
                
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start_time}s")
                llm_logger.info("首包到达耗时=%.3fs", end - start_time)
                first = False
            
            try:
                # 解析流式数据
                json_str = line[6:]  # 去掉 "data: " 前缀
                data = json.loads(json_str)
                
                content = data.get("content", "")
                if content:
                    llm_logger.debug("收到delta长度=%d 预览='%s'", len(content), content[:120])
                    result = _process_message_chunk(content, result, nerfreal)
                    total_chars += len(content)
                
                # 检查是否结束
                if data.get("is_end", False):
                    llm_logger.info("MaxKB响应结束标志")
                    break
                    
            except json.JSONDecodeError as e:
                llm_logger.warning("解析MaxKB流式数据失败: %s, line: %s", e, line[:200])
                continue
                
    except Exception as e:
        llm_logger.exception("处理MaxKB流式响应异常: %s", e)
    
    end = time.perf_counter()
    logger.info(f" LLM响应总时间: {end-start_time:.2f}s")
    llm_logger.info("流式结束，总时间=%.3fs，总字符数=%d", end - start_time, total_chars)
    
    if result:
        logger.info(f" 发送最终TTS文本: '{result}'")
        llm_logger.info("发送最终TTS文本，长度=%d 预览='%s'", len(result), result[:200])
        nerfreal.put_msg_txt(result)
    
    logger.info(" === LLM响应完成 ===")
    llm_logger.info("=== LLM响应完成（maxkb）===")


def _process_message_chunk(msg, result, nerfreal: BaseReal):
    """处理消息块，按标点符号分段"""
    msg = _clean_text(msg)
    lastpos = 0

    for i, char in enumerate(msg):
        if char in ",.!;:，。！？：；":
            result = result + msg[lastpos : i + 1]

            lastpos = i + 1
            if len(result) > 10:
                logger.info(f" 发送TTS文本: '{result}'")
                nerfreal.put_msg_txt(result)
                result = ""

    result = result + msg[lastpos:]
    return result
