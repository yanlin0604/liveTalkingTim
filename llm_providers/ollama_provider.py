import time
import re
from basereal import BaseReal
from logger import logger
from .base_provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama本地模型提供商"""
    
    def get_response(self, message: str, nerfreal: BaseReal, start_time: float):
        """Ollama本地模型响应处理"""
        try:
            import ollama
        except ImportError:
            logger.error(
                "ollama package not found. Please install it with: pip install ollama"
            )
            self.llm_logger.error("未安装ollama包，无法使用本地模型")
            nerfreal.put_msg_txt("抱歉，Ollama服务不可用。")
            return

        # 获取Ollama配置
        ollama_host = getattr(nerfreal.opt, "ollama_host", "http://localhost:11434")
        model = getattr(nerfreal.opt, "llm_model", "llama3.2")
        system_prompt = getattr(
            nerfreal.opt, "llm_system_prompt", "You are a helpful assistant."
        )
        self.llm_logger.info("请求模型=%s host=%s system_prompt_len=%d msg_len=%d", model, ollama_host, len(system_prompt or ""), len(message or ""))

        try:
            # 创建Ollama客户端
            client = ollama.Client(host=ollama_host)

            end = time.perf_counter()
            logger.info(f"llm Time init (ollama): {end-start_time}s")
            self.llm_logger.debug("ollama 初始化耗时=%.3fs", end - start_time)

            # 发送请求到Ollama
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                stream=True,
            )

            self.llm_logger.info("ollama 流式响应开始")
            self._process_ollama_stream_response(response, nerfreal, start_time)

        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            self.llm_logger.exception("Ollama 请求失败: %s", e)
            nerfreal.put_msg_txt("抱歉，无法连接到Ollama服务。")

    def _process_ollama_stream_response(self, response, nerfreal: BaseReal, start_time: float):
        """处理Ollama的流式响应"""
        result = ""
        first = True
        total_chars = 0

        for chunk in response:
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end-start_time}s")
                self.llm_logger.info("首包到达耗时=%.3fs", end - start_time)
                first = False

            msg = chunk.get("message", {}).get("content", "")
            if msg:
                self.llm_logger.debug("收到delta长度=%d 预览='%s'", len(msg), msg[:120])
                result = self._process_message_chunk(msg, result, nerfreal)
                total_chars += len(msg)

        end = time.perf_counter()
        logger.info(f" LLM响应总时间: {end-start_time:.2f}s")
        self.llm_logger.info("流式结束，总时间=%.3fs，总字符数=%d", end - start_time, total_chars)
        
        if result:
            # 清洗不可发音字符（例如表情符号），避免 TTS 无声
            sanitized = self._sanitize_tts_text(result)
            if sanitized:
                logger.info(f" 发送最终TTS文本: '{sanitized}'")
                self.llm_logger.info("发送最终TTS文本，长度=%d 预览='%s'", len(sanitized), sanitized[:200])
                nerfreal.put_msg_txt(sanitized)
            else:
                # 不再使用"好的。"兜底，避免用户听到重复拼接"好的"
                logger.warning(" 最终文本清洗后为空，本次不发送TTS文本")
                self.llm_logger.warning("清洗后为空，跳过TTS发送。原长=%d 预览='%s'", len(result), result[:200])
        
        logger.info(" === LLM响应完成 ===")
        self.llm_logger.info("=== LLM响应完成（ollama）===")

    def _sanitize_tts_text(self, text: str) -> str:
        """清洗TTS文本，去除不可发音字符"""
        # 去除常见 emoji 与变体选择符；保留中文、英文、数字与常见标点
        emoji_pattern = re.compile(
            "[\uFE0F\u200D\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAD6\U0001FAE0-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]",
            flags=re.UNICODE,
        )
        cleaned = emoji_pattern.sub("", text)
        return cleaned.strip()
