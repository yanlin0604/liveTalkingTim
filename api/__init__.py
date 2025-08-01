# API package for Unimed
"""
Unimed API Package

This package contains all API modules for the Unimed digital human system.
"""

__version__ = "1.0.0"
__author__ = "Unmed Team"
__email__ = "zengyanlin99@gmail.com"

# 导入所有API类，方便外部直接使用
from .webrtc import WebRTCAPI
from .chat import ChatAPI
from .config import ConfigAPI
from .avatars import AvatarsAPI
from .training import TrainingAPI, TrainingTask

# 定义包的公共接口
__all__ = [
    'WebRTCAPI',
    'ChatAPI', 
    'ConfigAPI',
    'AvatarsAPI',
    'TrainingAPI',
    'TrainingTask'
]

# 包初始化时的日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"Unmed API Package v{__version__} initialized")