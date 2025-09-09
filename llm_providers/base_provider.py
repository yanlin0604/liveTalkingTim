import time
import re
from abc import ABC, abstractmethod
from basereal import BaseReal
from logger import logger


class BaseLLMProvider(ABC):
    """LLM提供商基类"""
    
    def __init__(self, llm_logger):
        self.llm_logger = llm_logger
    
    @abstractmethod
    def get_response(self, message: str, nerfreal: BaseReal, start_time: float):
        """获取LLM响应的抽象方法
        
        Args:
            message: 用户消息
            nerfreal: BaseReal实例
            start_time: 开始时间
        """
        pass
    
    def _clean_text(self, s: str) -> str:
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

    def _process_message_chunk(self, msg: str, result: str, nerfreal: BaseReal) -> str:
        """处理消息块，按标点符号分段"""
        msg = self._clean_text(msg)
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
