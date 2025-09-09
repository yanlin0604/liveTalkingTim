import time
from openai import OpenAI
from basereal import BaseReal
from logger import logger
from .base_provider import BaseLLMProvider

# 调试模式开关 - 设置为True启用DashScope详细调试
DASHSCOPE_DEBUG_MODE = False

if DASHSCOPE_DEBUG_MODE:
    from dashscope_debug import debug_dashscope_response


class DashScopeProvider(BaseLLMProvider):
    """阿里云DashScope提供商"""
    
    def get_response(self, message: str, nerfreal: BaseReal, start_time: float):
        """阿里云DashScope响应处理"""
        llm_api_key = getattr(nerfreal.opt, 'llm_api_key', '')
        self.llm_logger.info("使用DashScope API Key: %s", llm_api_key)
        
        client = OpenAI(
            # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            api_key=llm_api_key,
            # 填写DashScope SDK的base_url
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        end = time.perf_counter()
        logger.info(f"llm Time init (dashscope): {end-start_time}s")
        self.llm_logger.debug("dashscope 初始化耗时=%.3fs", end - start_time)

        model = getattr(nerfreal.opt, 'llm_model', 'qwen-plus')
        system_prompt = getattr(
            nerfreal.opt, "llm_system_prompt", "你是一位乐于助人的助手。"
        )
        self.llm_logger.info("请求模型=%s system_prompt_len=%d msg_len=%d", model, len(system_prompt or ""), len(message or ""))
        self.llm_logger.debug("system_prompt预览: %s", (system_prompt or "")[:200])

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

        self.llm_logger.info("dashscope 流式响应开始")
        self._process_stream_response(completion, nerfreal, start_time)

    def _process_stream_response(self, completion, nerfreal: BaseReal, start_time: float):
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
                    self.llm_logger.info("首包到达耗时=%.3fs", end - start_time)
                    first = False

                msg = chunk.choices[0].delta.content
                if msg:
                    self.llm_logger.debug("收到delta长度=%d 预览='%s'", len(msg), msg[:120])
                    result = self._process_message_chunk(msg, result, nerfreal)
                    total_chars += len(msg)

            # 兼容 DashScope include_usage: True 的末包用量信息
            try:
                if hasattr(chunk, "usage") and chunk.usage and not usage_logged:
                    usage_logged = True
                    self.llm_logger.info(
                        "usage: prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                        getattr(chunk.usage, "prompt_tokens", None),
                        getattr(chunk.usage, "completion_tokens", None),
                        getattr(chunk.usage, "total_tokens", None),
                    )
            except Exception:
                pass

        end = time.perf_counter()
        logger.info(f" LLM响应总时间: {end-start_time:.2f}s")
        self.llm_logger.info("流式结束，总时间=%.3fs，总字符数=%d", end - start_time, total_chars)
        
        if result:
            logger.info(f" 发送最终TTS文本: '{result}'")
            self.llm_logger.info("发送最终TTS文本，长度=%d 预览='%s'", len(result), result[:200])
            nerfreal.put_msg_txt(result)
        
        logger.info(" === LLM响应完成 ===")
        self.llm_logger.info("=== LLM响应完成（dashscope）===")
