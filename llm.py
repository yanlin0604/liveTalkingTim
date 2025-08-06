import time
import os
from basereal import BaseReal
from logger import logger

def llm_response(message, nerfreal: BaseReal):
    """
    LLM响应函数，支持多种LLM提供商
    支持的提供商：dashscope（阿里云）、ollama（本地）
    """
    start = time.perf_counter()
    logger.info(f"🤖 === LLM响应开始 ===")
    logger.info(f"💬 用户消息: '{message[:50]}{'...' if len(message) > 50 else ''}'")

    # 获取LLM配置
    llm_provider = getattr(nerfreal.opt, 'llm_provider', 'dashscope')
    logger.info(f"🔧 LLM提供商: {llm_provider}")

    if llm_provider == 'ollama':
        logger.info("🚀 使用Ollama本地模型")
        _ollama_response(message, nerfreal, start)
    else:
        logger.info("☁️ 使用阿里云DashScope")
        _dashscope_response(message, nerfreal, start)

def _dashscope_response(message, nerfreal: BaseReal, start_time):
    """阿里云DashScope响应处理"""
    from openai import OpenAI

    client = OpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 填写DashScope SDK的base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    end = time.perf_counter()
    logger.info(f"llm Time init (dashscope): {end-start_time}s")

    model = getattr(nerfreal.opt, 'llm_model', 'qwen-plus')
    system_prompt = getattr(nerfreal.opt, 'llm_system_prompt', 'You are a helpful assistant.')

    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system_prompt},
                  {'role': 'user', 'content': message}],
        stream=True,
        # 通过以下设置，在流式输出的最后一行展示token使用信息
        stream_options={"include_usage": True}
    )

    _process_stream_response(completion, nerfreal, start_time)

def _ollama_response(message, nerfreal: BaseReal, start_time):
    """Ollama本地模型响应处理"""
    try:
        import ollama
    except ImportError:
        logger.error("ollama package not found. Please install it with: pip install ollama")
        nerfreal.put_msg_txt("抱歉，Ollama服务不可用。")
        return

    # 获取Ollama配置
    ollama_host = getattr(nerfreal.opt, 'ollama_host', 'http://localhost:11434')
    model = getattr(nerfreal.opt, 'llm_model', 'llama3.2')
    system_prompt = getattr(nerfreal.opt, 'llm_system_prompt', 'You are a helpful assistant.')

    try:
        # 创建Ollama客户端
        client = ollama.Client(host=ollama_host)

        end = time.perf_counter()
        logger.info(f"llm Time init (ollama): {end-start_time}s")

        # 发送请求到Ollama
        response = client.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': message}
            ],
            stream=True
        )

        _process_ollama_stream_response(response, nerfreal, start_time)

    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        nerfreal.put_msg_txt("抱歉，无法连接到Ollama服务。")

def _process_stream_response(completion, nerfreal: BaseReal, start_time):
    """处理OpenAI兼容的流式响应"""
    result = ""
    first = True

    for chunk in completion:
        if len(chunk.choices) > 0:
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start_time}s")
                first = False

            msg = chunk.choices[0].delta.content
            if msg:
                result = _process_message_chunk(msg, result, nerfreal)

    end = time.perf_counter()
    logger.info(f"⏱️ LLM响应总时间: {end-start_time:.2f}s")
    if result:
        logger.info(f"🎤 发送最终TTS文本: '{result}'")
        nerfreal.put_msg_txt(result)
    logger.info("✅ === LLM响应完成 ===")

def _process_ollama_stream_response(response, nerfreal: BaseReal, start_time):
    """处理Ollama的流式响应"""
    result = ""
    first = True

    for chunk in response:
        if first:
            end = time.perf_counter()
            logger.info(f"llm Time to first chunk: {end-start_time}s")
            first = False

        msg = chunk.get('message', {}).get('content', '')
        if msg:
            result = _process_message_chunk(msg, result, nerfreal)

    end = time.perf_counter()
    logger.info(f"⏱️ LLM响应总时间: {end-start_time:.2f}s")
    if result:
        logger.info(f"🎤 发送最终TTS文本: '{result}'")
        nerfreal.put_msg_txt(result)
    logger.info("✅ === LLM响应完成 ===")

def _process_message_chunk(msg, result, nerfreal: BaseReal):
    """处理消息块，按标点符号分段"""
    lastpos = 0

    for i, char in enumerate(msg):
        if char in ",.!;:，。！？：；":
            result = result + msg[lastpos:i+1]
            lastpos = i+1
            if len(result) > 10:
                logger.info(f"🎤 发送TTS文本: '{result}'")
                nerfreal.put_msg_txt(result)
                result = ""

    result = result + msg[lastpos:]
    return result