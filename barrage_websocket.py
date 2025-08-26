import argparse
import asyncio
import json
import logging
import os
import time
import random
from asyncio import Event
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Tuple, Dict, Any, Optional
import re

import aiohttp
from reactivestreams.subscriber import Subscriber
from reactivestreams.subscription import Subscription
from rsocket.helpers import single_transport_provider
from rsocket.payload import Payload
from rsocket.rsocket_client import RSocketClient
from rsocket.streams.stream_from_async_generator import StreamFromAsyncGenerator
from rsocket.transports.aiohttp_websocket import TransportAioHttpClient

subscribe_payload_json = {
    "data": {
        "taskIds": [],
        "cmd": "SUBSCRIBE"
    }
}


# ========================= 配置与模板工具函数 ========================= #

# 默认配置：八类消息的基础行为与模板
DEFAULT_CONFIG: Dict[str, Any] = {
    "human_url": "http://127.0.0.1:8010/human",
    "default_sessionid": 0,
    "types": {
        "DANMU": {
            "enabled": True,
            "action": "echo",
            "template": "{username}说：{content}",
            "interrupt": False,
            "min_length": 1,
            "max_length": 120
        },
        "GIFT": {
            "enabled": True,
            "action": "echo",
            "template": "感谢{username}送出的{giftName}x{giftCount}！",
            "interrupt": False,
            "min_gift_price": 0
        },
        "SUPER_CHAT": {
            "enabled": True,
            "action": "echo",
            "template": "感谢醒目留言，{username}：{content}",
            "interrupt": True,
            "min_price": 0
        },
        "ENTER_ROOM": {
            "enabled": False,
            "action": "echo",
            "template": "欢迎{username}进入直播间",
            "interrupt": False
        },
        "LIKE": {
            "enabled": False,
            "action": "echo",
            "template": "{username} 点赞了直播",
            "interrupt": False
        },
        "LIVE_STATUS_CHANGE": {
            "enabled": True,
            "action": "echo",
            "template": "直播状态变更：{status}",
            "interrupt": True
        },
        "ROOM_STATS": {
            "enabled": False,
            "action": "echo",
            "template": "当前在线{online}，热度{hot}，点赞{likes}",
            "interrupt": False
        },
        "SOCIAL": {
            "enabled": False,
            "action": "echo",
            "template": "{username}{action}",
            "interrupt": False
        }
    },
    # 可按类型单独指定sessionid；不配置则使用 default_sessionid
    "sessions": {
        # "DANMU": 0,
        # "GIFT": 0,
        # "SUPER_CHAT": 0
    }
}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """加载弹幕转发配置，若不存在则使用默认配置。"""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝
    if not path:
        return cfg
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                user_cfg = json.load(f)
            # 浅合并：human_url、default_sessionid、reply_control、types子项、sessions
            if isinstance(user_cfg, dict):
                if 'human_url' in user_cfg:
                    cfg['human_url'] = user_cfg['human_url']
                if 'default_sessionid' in user_cfg:
                    cfg['default_sessionid'] = user_cfg['default_sessionid']
                if 'reply_control' in user_cfg:
                    cfg['reply_control'] = user_cfg['reply_control']
                if 'types' in user_cfg and isinstance(user_cfg['types'], dict):
                    for k, v in user_cfg['types'].items():
                        if k in cfg['types'] and isinstance(v, dict):
                            cfg['types'][k].update(v)
                        else:
                            cfg['types'][k] = v
                if 'sessions' in user_cfg and isinstance(user_cfg['sessions'], dict):
                    cfg['sessions'].update(user_cfg['sessions'])
    except Exception as e:
        logging.warning(f"加载配置失败，使用默认配置: {e}")
    return cfg

# ===== 外部配置（话术/敏感词/定时/弹幕规则） ===== #
SPEECH_CFG: Dict[str, Any] = {}
SENSITIVE_CFG: Dict[str, Any] = {}
SCHEDULE_CFG: Dict[str, Any] = {}
RULES_CFG: Dict[str, Any] = {}

# 外部配置路径
SPEECH_PATH = 'config/speech_config.json'
SENSITIVE_PATH = 'config/sensitive_config.json'
SCHEDULE_PATH = 'config/schedule_config.json'
RULES_PATH = 'config/barrage_rules.json'

_RATE_WINDOW_START = 0.0
_RATE_COUNT = 0
_LAST_ACTIVITY_TS = 0.0

# 弹幕回复控制变量
_REPLY_WINDOW_START = 0.0
_REPLY_COUNT = 0

def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception as e:
        logging.warning(f"加载配置失败 {path}: {e}")
    return {}

def load_external_configs():
    """加载四类外部配置文件。"""
    global SPEECH_CFG, SENSITIVE_CFG, SCHEDULE_CFG, RULES_CFG
    SPEECH_CFG = _safe_load_json(SPEECH_PATH)
    SENSITIVE_CFG = _safe_load_json(SENSITIVE_PATH)
    SCHEDULE_CFG = _safe_load_json(SCHEDULE_PATH)
    RULES_CFG = _safe_load_json(RULES_PATH)
    logging.info("外部配置已加载：speech/sensitive/schedule/rules")

async def _config_watcher_loop(poll_sec: int = 2):
    """简单的轮询热加载，无需额外依赖。"""
    last: Dict[str, float] = {}
    paths = [SPEECH_PATH, SENSITIVE_PATH, SCHEDULE_PATH, RULES_PATH]
    # 初始化时间戳
    for p in paths:
        try:
            last[p] = os.path.getmtime(p) if os.path.exists(p) else 0.0
        except Exception:
            last[p] = 0.0
    while True:
        await asyncio.sleep(poll_sec)
        try:
            changed = False
            for p in paths:
                try:
                    mt = os.path.getmtime(p) if os.path.exists(p) else 0.0
                except Exception:
                    mt = 0.0
                if mt != last.get(p, 0.0):
                    changed = True
                    last[p] = mt
            if changed:
                load_external_configs()
                logging.info("检测到配置文件变更，已热加载")
        except Exception as e:
            logging.warning(f"配置热加载异常: {e}")

def apply_sensitive_filter(text: str) -> Optional[str]:
    """按敏感词配置过滤文本。返回 None 表示丢弃。"""
    if not text:
        return text
    
    # 检查白名单，如果包含白名单词汇则跳过过滤
    whitelist = (SENSITIVE_CFG.get('whitelist') or [])
    for white_word in whitelist:
        if white_word and white_word in text:
            logging.info(f"[敏感词跳过] 包含白名单词汇: {white_word}")
            return text
    
    bl = (SENSITIVE_CFG.get('blacklist') or [])
    strategy = (SENSITIVE_CFG.get('strategy') or 'mask').lower()
    mask_char = SENSITIVE_CFG.get('mask_char', '*')
    hit = False
    filtered = text
    
    for w in bl:
        if not w:
            continue
        if w in filtered:
            hit = True
            if strategy == 'mask':
                filtered = filtered.replace(w, mask_char * len(w))
            elif strategy == 'block' or strategy == 'drop':
                return None
    return filtered if (not hit or strategy == 'mask') else text

def should_reply_to_danmu(ctx: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """判断是否应该回复弹幕，基于概率和频率控制。"""
    global _REPLY_WINDOW_START, _REPLY_COUNT
    
    content = str(ctx.get('content', '') or '')
    
    # 敏感词过滤检查
    filtered_content = apply_sensitive_filter(content)
    if filtered_content is None:
        logging.info(f"[敏感词过滤] 弹幕被屏蔽: {content[:30]}")
        return False
    
    # 获取回复控制配置
    reply_ctrl = config.get('reply_control', {})
    if not reply_ctrl.get('enabled', True):
        logging.info(f"[回复控制] 回复功能已禁用")
        return False
    
    # 检查每分钟回复数量限制
    max_replies = reply_ctrl.get('max_replies_per_minute', 10)
    now = time.time()
    if now - _REPLY_WINDOW_START >= 60:
        _REPLY_WINDOW_START = now
        _REPLY_COUNT = 0
    
    if _REPLY_COUNT >= max_replies:
        logging.info(f"[回复控制] 达到每分钟最大回复数({max_replies})")
        return False
    
    priority_keywords = reply_ctrl.get('priority_keywords', [])
    has_priority = any(keyword in content for keyword in priority_keywords)
    
    logging.info(f"[关键词检查] 内容:'{content}' | 优先关键词:{priority_keywords} | 匹配:{has_priority}")
    
    if has_priority:
        logging.info(f"[优先回复] 包含关键词: {content[:30]}")
        _REPLY_COUNT += 1
        return True
    
    # 基于概率决定是否回复
    reply_prob = reply_ctrl.get('reply_probability', 0.8)
    if random.random() < reply_prob:
        _REPLY_COUNT += 1
        logging.info(f"[随机回复] 概率命中({reply_prob}): {content[:30]}")
        return True
    
    logging.info(f"[跳过回复] 概率未命中({reply_prob}): {content[:30]}")
    return False

def match_reply_template(msg_type: str, ctx: Dict[str, Any]) -> Optional[str]:
    """从话术配置中匹配模板，优先用于 DANMU 或通用。"""
    if msg_type != 'DANMU':
        return None
    
    content = str(ctx.get('content', '') or '')
    
    # 按优先级排序的回复规则
    rules = SPEECH_CFG.get('reply_rules') or []
    sorted_rules = sorted(rules, key=lambda x: x.get('priority', 0), reverse=True)
    
    for r in sorted_rules:
        pat = r.get('match')
        tpl = r.get('template')
        if not pat or not tpl:
            continue
        # 简单包含或正则匹配
        try:
            if pat in content or re.search(pat, content):
                logging.info(f"[规则匹配] 模式:{pat} | 内容:{content[:30]}")
                return tpl
        except re.error:
            if pat in content:
                logging.info(f"[规则匹配] 模式:{pat} | 内容:{content[:30]}")
                return tpl
    
    return None

def get_greeting_template() -> Optional[str]:
    """获取问候语模板。"""
    templates = SPEECH_CFG.get('templates', {}).get('greeting', [])
    if not templates:
        return None
    
    selected = random.choice(templates)
    logging.info(f"[问候模板] 选择: {selected}")
    return selected

def get_fallback_template() -> Optional[str]:
    """获取兜底回复模板。"""
    templates = SPEECH_CFG.get('templates', {}).get('fallback', [])
    if not templates:
        return None
    
    selected = random.choice(templates)
    logging.info(f"[兜底模板] 选择: {selected}")
    return selected

def get_gift_thanks_template(gift_price: float, ctx: Dict[str, Any]) -> Optional[str]:
    """根据礼物价格获取感谢模板。"""
    gift_thanks = SPEECH_CFG.get('gift_thanks', [])
    if not gift_thanks:
        return None
    
    # 按优先级排序，价格从高到低
    sorted_thanks = sorted(gift_thanks, key=lambda x: (x.get('priority', 0), x.get('min_price', 0)), reverse=True)
    
    for thanks in sorted_thanks:
        min_price = thanks.get('min_price', 0)
        if gift_price >= min_price:
            template = thanks.get('template', '')
            logging.info(f"[礼物感谢] 价格:{gift_price} >= {min_price}, 模板:{template}")
            return template
    
    return None

def get_random_reply_template(ctx: Dict[str, Any]) -> Optional[str]:
    """获取随机回复模板。"""
    templates = SPEECH_CFG.get('templates', {}).get('random_replies', [])
    if not templates:
        return None
    
    selected = random.choice(templates)
    logging.info(f"[随机模板] 选择: {selected}")
    return selected

def check_barrage_global(text_for_len: str) -> bool:
    """检查全局弹幕规则（长度与速率限制）。"""
    global _RATE_WINDOW_START, _RATE_COUNT
    g = RULES_CFG.get('global') or {}
    min_len = int(g.get('min_len', 0) or 0)
    max_len = int(g.get('max_len', 0) or 0)
    if min_len and len(text_for_len) < min_len:
        return False
    if max_len and len(text_for_len) > max_len:
        # 不直接过早拒绝，后续会对内容做截断，限流仍需计算
        pass
    # 速率限制
    limit = int(g.get('rate_limit_per_min', 0) or 0)
    if limit > 0:
        now = time.time()
        if now - _RATE_WINDOW_START >= 60:
            _RATE_WINDOW_START = now
            _RATE_COUNT = 0
        _RATE_COUNT += 1
        if _RATE_COUNT > limit:
            return False
    return True

def update_activity():
    global _LAST_ACTIVITY_TS
    _LAST_ACTIVITY_TS = time.time()

async def scheduler_loop(http_session: aiohttp.ClientSession, config: Dict[str, Any]):
    """内置调度：自动播报与冷场填充。"""
    auto_cfg = (SCHEDULE_CFG.get('auto_broadcast') or {})
    idle_cfg = (SCHEDULE_CFG.get('idle_fill') or {})
    auto_idx = 0
    while True:
        await asyncio.sleep(1)
        try:
            # 自动播报
            if auto_cfg.get('enabled') and (auto_cfg.get('messages')):
                interval = int(auto_cfg.get('interval_sec', 0) or 0)
                if interval > 0 and int(time.time()) % interval == 0:
                    msg = auto_cfg['messages'][auto_idx % len(auto_cfg['messages'])]
                    auto_idx += 1
                    auto_interrupt = bool(auto_cfg.get('interrupt', False))
                    auto_type = auto_cfg.get('msg_type')
                    await _send_human_text(http_session, config, str(msg), interrupt=auto_interrupt, msg_type=auto_type)
                    update_activity()
            # 冷场填充
            if idle_cfg.get('enabled') and (idle_cfg.get('messages')):
                th = int(idle_cfg.get('idle_threshold_sec', 0) or 0)
                if th > 0 and (time.time() - _LAST_ACTIVITY_TS) >= th:
                    idle_interrupt = bool(idle_cfg.get('interrupt', False))
                    idle_type = idle_cfg.get('msg_type')
                    await _send_human_text(http_session, config, str(idle_cfg['messages'][0]), interrupt=idle_interrupt, msg_type=idle_type)
                    update_activity()
        except Exception as e:
            logging.warning(f"调度任务异常: {e}")

async def _send_startup_greeting(http_session: aiohttp.ClientSession, config: Dict[str, Any]):
    """发送系统启动问候语。"""
    greeting_tpl = get_greeting_template()
    if not greeting_tpl:
        return
    
    # 构建上下文（用于模板渲染）
    ctx = {
        'avatar': '主播',  # 可以从配置中获取
        'username': '系统',
        'platform': 'system'
    }
    
    text = render_template(greeting_tpl, ctx)
    if not text:
        return
    
    logging.info(f"[启动问候] 发送问候语: {text}")
    await _send_human_text(http_session, config, text, interrupt=False, msg_type='greeting')

async def _send_human_text(http_session: aiohttp.ClientSession, config: Dict[str, Any], text: str, interrupt: bool, msg_type: Optional[str] = None):
    # 优先按类型路由其 sessionid
    if msg_type:
        sessionid = (config.get('sessions') or {}).get(str(msg_type), config.get('default_sessionid', 0))
    else:
        sessionid = config.get('default_sessionid', 0)
    payload = {
        "text": text,
        "type": "echo",
        "interrupt": interrupt,
        "sessionid": sessionid
    }
    try:
        async with http_session.post(config.get('human_url'), json=payload, timeout=10) as resp:
            if resp.status != 200:
                logging.warning(f"/human 返回非200: {resp.status}")
    except Exception as e:
        logging.warning(f"调用 /human 异常: {e}")

def flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    """将嵌套dict拍平成 'a.b' 键。"""
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            flatten(key, v, out)
    else:
        out[prefix] = value


def build_context(msg_dto: Dict[str, Any]) -> Dict[str, Any]:
    """基于统一JSON构建模板上下文：不再做历史兼容与推断。"""
    ctx: Dict[str, Any] = {}
    ctx['type'] = msg_dto.get('type', '')
    ctx['roomId'] = msg_dto.get('roomId', '')
    msg = msg_dto.get('msg', {}) or {}

    # 统一JSON的直接字段
    ctx['platform'] = msg.get('platform', '')
    ctx['username'] = msg.get('username', '')
    ctx['uid'] = msg.get('uid', '')
    ctx['content'] = msg.get('content', '')
    ctx['giftName'] = msg.get('giftName', '')
    ctx['giftCount'] = msg.get('giftCount', 1)
    # 兼容礼物/SC统一金额：giftPrice 与 price 都从统一JSON填充
    ctx['giftPrice'] = msg.get('giftPrice', 0)
    ctx['price'] = msg.get('price', 0)
    ctx['status'] = msg.get('status', '')
    ctx['online'] = msg.get('online', '')
    ctx['hot'] = msg.get('hot', '')
    ctx['likes'] = msg.get('likes', '')
    ctx['timestamp'] = msg.get('timestamp', 0)
    ctx['raw'] = msg.get('raw', '')

    # 同时提供扁平化点号字段（保留，便于模板扩展）
    flat: Dict[str, Any] = {}
    flatten("", msg_dto, flat)
    ctx.update(flat)
    return ctx


    


def normalize_incoming(obj: Any) -> list:
    """
    仅支持“统一 JSON”结构：
    {
      platform, type, roomId, userId, username, content,
      giftName, giftCount, price, status, online, hot, likes,
      timestamp, raw
    }
    返回 [{"type", "roomId", "msg"}]
    """
    res = []
    if obj is None:
        return res
    # 批量数组
    if isinstance(obj, list):
        for it in obj:
            res.extend(normalize_incoming(it))
        return res
    # 单对象（严格按统一JSON）
    if isinstance(obj, dict):
        if 'type' not in obj:
            return res
        unified_msg = {
            "platform": obj.get("platform"),
            "username": obj.get("username"),
            "uid": obj.get("userId"),
            "content": obj.get("content"),
            "giftName": obj.get("giftName"),
            "giftCount": obj.get("giftCount", 1),
            "giftPrice": obj.get("price", 0),
            "price": obj.get("price", 0),
            "status": obj.get("status"),
            "online": obj.get("online"),
            "hot": obj.get("hot"),
            "likes": obj.get("likes"),
            "timestamp": obj.get("timestamp"),
            "raw": obj.get("raw"),
        }
        res.append({
            "type": obj.get("type", ""),
            "roomId": obj.get("roomId", ""),
            "msg": unified_msg,
        })
        return res
    return res


def render_template(tpl: str, ctx: Dict[str, Any]) -> str:
    """安全渲染模板：使用 format_map，缺失字段置空。"""
    class SafeDict(dict):
        def __missing__(self, key):
            return ""

    try:
        return str(tpl).format_map(SafeDict(ctx))
    except Exception:
        return str(tpl)


def get_type_cfg(cfg: Dict[str, Any], msg_type: str) -> Dict[str, Any]:
    return (cfg.get('types') or {}).get(msg_type, {})


class ChannelSubscriber(Subscriber):
    def __init__(self, wait_for_responder_complete: Event, http_session: aiohttp.ClientSession,
                 human_url: str, config: Dict[str, Any]) -> None:
        super().__init__()
        self.subscription = None
        self._wait_for_responder_complete = wait_for_responder_complete
        self._http_session = http_session
        self._human_url = human_url
        self._config = config

    def on_subscribe(self, subscription: Subscription):
        self.subscription = subscription
        self.subscription.request(0x7FFFFFFF)

    # TODO 收到消息回调
    def on_next(self, value: Payload, is_complete=False):
        try:
            raw = json.loads(value.data)
        except Exception as e:
            logging.warning(f"收到非JSON消息，解析失败: {e}, 数据: {str(value.data)[:200]}")
            return

        msgs = normalize_incoming(raw)
        logging.info(f"[RSocket] 收到 {len(msgs)} 条消息")
        
        for msg_dto in msgs:
            if not isinstance(msg_dto, dict):
                continue
            msg_type = msg_dto.get('type')
            room_id = msg_dto.get('roomId', '')
            msg = msg_dto.get('msg', {}) or {}
            
            # 详细的消息接收日志
            if msg_type == "DANMU":
                badge_info = f"{msg.get('badgeLevel','')}{msg.get('badgeName','')}" if msg.get('badgeLevel',0) else ''
                logging.info(
                    f"[弹幕接收] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | "
                    f"徽章:{badge_info} | 内容:{msg.get('content','')} | 平台:{msg.get('platform','')}"
                )
            elif msg_type == "GIFT":
                badge_info = f"{msg.get('badgeLevel','')}{msg.get('badgeName','')}" if msg.get('badgeLevel',0) else ''
                action = msg.get('action') or '赠送'
                logging.info(
                    f"[礼物接收] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | "
                    f"徽章:{badge_info} | {action}:{msg.get('giftName','')}x{msg.get('giftCount',1)} | "
                    f"价值:{msg.get('giftPrice',0)} | 平台:{msg.get('platform','')}"
                )
            elif msg_type == "SUPER_CHAT":
                logging.info(
                    f"[醒目留言] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | "
                    f"内容:{msg.get('content','')} | 价格:{msg.get('price',0)} | 平台:{msg.get('platform','')}"
                )
            elif msg_type == "LIVE_STATUS_CHANGE":
                logging.info(
                    f"[直播状态] 房间:{room_id} | 状态变更:{msg.get('status','')} | 平台:{msg.get('platform','')}"
                )
            elif msg_type == "ENTER_ROOM":
                logging.info(
                    f"[进入房间] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | 平台:{msg.get('platform','')}"
                )
            elif msg_type == "LIKE":
                logging.info(
                    f"[点赞] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | 平台:{msg.get('platform','')}"
                )
            elif msg_type == "ROOM_STATS":
                logging.info(
                    f"[房间统计] 房间:{room_id} | 在线:{msg.get('online','')} | 热度:{msg.get('hot','')} | "
                    f"点赞:{msg.get('likes','')} | 平台:{msg.get('platform','')}"
                )
            else:
                logging.info(f"[未知消息] 类型:{msg_type} | 房间:{room_id} | 数据:{json.dumps(msg_dto, ensure_ascii=False)[:200]}")

            # 根据配置异步转发到 /human
            try:
                asyncio.get_event_loop().create_task(self._maybe_dispatch(msg_dto))
            except Exception as e:
                logging.error(f"[调度失败] 消息类型:{msg_type} | 房间:{room_id} | 错误:{e}")

        if is_complete:
            self._wait_for_responder_complete.set()

    def on_error(self, exception: Exception):
        logging.error('Error from server on channel' + str(exception))
        self._wait_for_responder_complete.set()

    def on_complete(self):
        logging.info('Completed from server on channel')
        self._wait_for_responder_complete.set()

    async def _maybe_dispatch(self, msg_dto: Dict[str, Any]):
        msg_type = msg_dto.get('type', '')
        cfg = get_type_cfg(self._config, msg_type)
        if not cfg or not cfg.get('enabled', False):
            logging.info(f"[消息跳过] 类型:{msg_type} | 原因:配置未启用或不存在")
            return

        ctx = build_context(msg_dto)

        # 基础过滤（长度 / 价格等）
        if msg_type == 'DANMU':
            text_raw = str(ctx.get('content', ''))
            min_len = int(cfg.get('min_length', 0) or 0)
            max_len = int(cfg.get('max_length', 0) or 0)
            if min_len and len(text_raw) < min_len:
                logging.info(f"[弹幕过滤] 用户:{ctx.get('username','')} | 原因:长度不足({len(text_raw)}<{min_len}) | 内容:{text_raw}")
                return
            if max_len and len(text_raw) > max_len:
                # 过长截断（防止刷屏）
                logging.info(f"[弹幕截断] 用户:{ctx.get('username','')} | 原长度:{len(text_raw)} | 截断至:{max_len}")
                ctx['content'] = text_raw[:max_len] + '…'
        elif msg_type == 'GIFT':
            min_price = float(cfg.get('min_gift_price', 0) or 0)
            price = 0.0
            try:
                price = float(ctx.get('giftPrice') or 0)
            except Exception:
                price = 0.0
            if price < min_price:
                logging.info(f"[礼物过滤] 用户:{ctx.get('username','')} | 原因:价格不足({price}<{min_price}) | 礼物:{ctx.get('giftName','')}")
                return
        elif msg_type == 'SUPER_CHAT':
            min_price = float(cfg.get('min_price', 0) or 0)
            price = 0.0
            try:
                price = float(ctx.get('price') or 0)
            except Exception:
                price = 0.0
            if price < min_price:
                logging.info(f"[醒目留言过滤] 用户:{ctx.get('username','')} | 原因:价格不足({price}<{min_price})")
                return

        # 对于弹幕消息，先判断是否应该回复
        if msg_type == 'DANMU':
            if not should_reply_to_danmu(ctx, self._config):
                return
        
        # 话术规则匹配优先
        tpl = match_reply_template(msg_type, ctx)
        if not tpl and msg_type == 'DANMU':
            # 如果没有匹配到特定规则，使用随机回复模板
            tpl = get_random_reply_template(ctx)
        if not tpl:
            # 最后使用默认模板
            tpl = cfg.get('template', '')
        
        text = render_template(tpl, ctx) if tpl else ctx.get('content', '')
        text = str(text).strip()
        if not text:
            return

        # 全局弹幕规则（长度与限流）
        if not check_barrage_global(text):
            logging.info(f"[全局过滤] 用户:{ctx.get('username','')} | 原因:违反全局规则 | 文本:{text[:30]}")
            return

        # 敏感词过滤
        original_text = text
        text = apply_sensitive_filter(text)
        if text is None:
            logging.info(f"[敏感词拦截] 用户:{ctx.get('username','')} | 原文本:{original_text[:50]}")
            return
        elif text != original_text:
            logging.info(f"[敏感词替换] 用户:{ctx.get('username','')} | 原文:{original_text[:30]} | 替换后:{text[:30]}")

        # 目标 sessionid 与是否打断
        sessionid = (self._config.get('sessions') or {}).get(msg_type, self._config.get('default_sessionid', 0))
        interrupt = bool(cfg.get('interrupt', False))

        payload = {
            "text": text,
            "type": str(cfg.get('action', 'echo')),
            "interrupt": interrupt,
            "sessionid": sessionid
        }

        # 发送到 /human
        logging.info(
            f"[/human请求] 类型:{msg_type} | 用户:{ctx.get('username','')} | "
            f"会话ID:{sessionid} | 打断:{interrupt} | 动作:{payload.get('type','')} | "
            f"文本:{text[:50]}{'...' if len(text) > 50 else ''}"
        )
        
        try:
            async with self._http_session.post(self._human_url, json=payload, timeout=10) as resp:
                response_text = await resp.text()
                if resp.status != 200:
                    logging.error(
                        f"[/human错误] 状态码:{resp.status} | 响应:{response_text} | "
                        f"请求:{json.dumps(payload, ensure_ascii=False)}"
                    )
                else:
                    try:
                        response_json = json.loads(response_text)
                        logging.info(
                            f"[/human成功] 代码:{response_json.get('code','')} | "
                            f"消息:{response_json.get('msg','')} | 会话ID:{sessionid}"
                        )
                    except json.JSONDecodeError:
                        logging.info(f"[/human成功] 响应:{response_text} | 会话ID:{sessionid}")
            update_activity()
        except asyncio.TimeoutError:
            logging.error(f"[/human超时] 请求超时(10秒) | 会话ID:{sessionid} | 文本:{text[:30]}")
        except Exception as e:
            logging.error(f"[/human异常] 网络错误:{e} | 会话ID:{sessionid} | 文本:{text[:30]}")


@asynccontextmanager
async def connect(websocket_uri):
    """
    创建一个Client，建立连接并return
    """
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(websocket_uri) as websocket:
            async with RSocketClient(
                    single_transport_provider(TransportAioHttpClient(websocket=websocket)),
                    keep_alive_period=timedelta(seconds=30),
                    max_lifetime_period=timedelta(days=1)
            ) as client:
                yield client


async def main(websocket_uri: str, human_url: str, config_path: Optional[str]):
    # 兼容两种模式：普通 WebSocket 推送（默认）与 RSocket
    mode = globals().get('_RUN_MODE', 'ws')
    if mode == 'ws':
        await ws_listen(websocket_uri, human_url, config_path)
        return

    # ===== RSocket 模式 =====
    cfg = load_config(config_path)
    load_external_configs()
    if human_url:
        cfg['human_url'] = human_url

    async with aiohttp.ClientSession() as http_session:
        # 启动调度任务
        update_activity()  # 启动静默，避免立即判定冷场
        asyncio.create_task(scheduler_loop(http_session, cfg))
        # 启动配置热加载
        asyncio.create_task(_config_watcher_loop())
        
        # 发送启动问候语
        await _send_startup_greeting(http_session, cfg)
        async with connect(websocket_uri) as client:
            channel_completion_event = Event()

            async def generator() -> AsyncGenerator[Tuple[Payload, bool], None]:
                yield Payload(
                    data=json.dumps(subscribe_payload_json["data"]).encode()
                ), False
                await Event().wait()

            stream = StreamFromAsyncGenerator(generator)
            requested = client.request_channel(Payload(), stream)
            requested.subscribe(ChannelSubscriber(channel_completion_event, http_session, cfg.get('human_url'), cfg))
            await channel_completion_event.wait()

async def ws_listen(websocket_uri: str, human_url: str, config_path: Optional[str]):
    """监听普通 WebSocket 推送服务，支持 Binary/Text 帧，UTF-8 解码并尝试 JSON 解析。"""
    cfg = load_config(config_path)
    load_external_configs()
    if human_url:
        cfg['human_url'] = human_url

    async with aiohttp.ClientSession() as http_session:
        # 启动调度任务
        update_activity()  # 启动静默
        asyncio.create_task(scheduler_loop(http_session, cfg))
        # 启动配置热加载
        asyncio.create_task(_config_watcher_loop())
        
        # 发送启动问候语
        await _send_startup_greeting(http_session, cfg)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(websocket_uri) as ws:
                    logging.info(f"WS 已连接: {websocket_uri}")
                    async for msg in ws:
                        text = None
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            try:
                                text = msg.data.decode('utf-8', errors='ignore')
                            except Exception:
                                text = None
                        elif msg.type == aiohttp.WSMsgType.TEXT:
                            text = msg.data
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logging.warning(f"WS 错误: {ws.exception()}")
                            break
                        else:
                            continue

                        if not text:
                            continue

                        # 尝试解析并规范化为一个或多个消息
                        try:
                            raw = json.loads(text)
                        except Exception:
                            logging.info(f"收到非JSON文本: {text[:200]}")
                            continue

                        msgs = normalize_incoming(raw)
                        if not msgs:
                            continue
                        
                        for msg_dto in msgs:
                            if not isinstance(msg_dto, dict):
                                continue
                            
                            msg_type = msg_dto.get('type')
                            room_id = msg_dto.get('roomId', '')
                            msg = msg_dto.get('msg', {}) or {}
                            
                            # 详细的消息接收日志（与RSocket保持一致）
                            if msg_type == "DANMU":
                                badge_info = f"{msg.get('badgeLevel','')}{msg.get('badgeName','')}" if msg.get('badgeLevel',0) else ''
                                logging.info(
                                    f"[弹幕接收] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | "
                                    f"徽章:{badge_info} | 内容:{msg.get('content','')} | 平台:{msg.get('platform','')}"
                                )
                            elif msg_type == "GIFT":
                                badge_info = f"{msg.get('badgeLevel','')}{msg.get('badgeName','')}" if msg.get('badgeLevel',0) else ''
                                action = msg.get('action') or '赠送'
                                logging.info(
                                    f"[礼物接收] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | "
                                    f"徽章:{badge_info} | {action}:{msg.get('giftName','')}x{msg.get('giftCount',1)} | "
                                    f"价值:{msg.get('giftPrice',0)} | 平台:{msg.get('platform','')}"
                                )
                            elif msg_type == "SUPER_CHAT":
                                logging.info(
                                    f"[醒目留言] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | "
                                    f"内容:{msg.get('content','')} | 价格:{msg.get('price',0)} | 平台:{msg.get('platform','')}"
                                )
                            elif msg_type == "LIVE_STATUS_CHANGE":
                                logging.info(
                                    f"[直播状态] 房间:{room_id} | 状态变更:{msg.get('status','')} | 平台:{msg.get('platform','')}"
                                )
                            elif msg_type == "ENTER_ROOM":
                                logging.info(
                                    f"[进入房间] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | 平台:{msg.get('platform','')}"
                                )
                            elif msg_type == "LIKE":
                                logging.info(
                                    f"[点赞] 房间:{room_id} | 用户:{msg.get('username','')}({msg.get('uid','')}) | 平台:{msg.get('platform','')}"
                                )
                            elif msg_type == "ROOM_STATS":
                                logging.info(
                                    f"[房间统计] 房间:{room_id} | 在线:{msg.get('online','')} | 热度:{msg.get('hot','')} | "
                                    f"点赞:{msg.get('likes','')} | 平台:{msg.get('platform','')}"
                                )
                            else:
                                logging.info(f"[未知消息] 类型:{msg_type} | 房间:{room_id} | 数据:{json.dumps(msg_dto, ensure_ascii=False)[:200]}")

                            # 调用 /human（按现有过滤与模板规则）
                            try:
                                await _ws_maybe_dispatch(http_session, cfg, msg_dto)
                            except Exception as e:
                                logging.error(f"[调度失败] 消息类型:{msg_type} | 房间:{room_id} | 错误:{e}")
        except Exception as e:
            logging.error(f"WS 连接失败: {e}")

async def _ws_maybe_dispatch(http_session: aiohttp.ClientSession, config: Dict[str, Any], msg_dto: Dict[str, Any]):
    msg_type = msg_dto.get('type', '')
    cfg = get_type_cfg(config, msg_type)
    if not cfg or not cfg.get('enabled', False):
        logging.info(f"[消息跳过] 类型:{msg_type} | 原因:配置未启用或不存在")
        return

    ctx = build_context(msg_dto)

    # 基础过滤（与 RSocket 逻辑保持一致）
    if msg_type == 'DANMU':
        text_raw = str(ctx.get('content', ''))
        min_len = int(cfg.get('min_length', 0) or 0)
        max_len = int(cfg.get('max_length', 0) or 0)
        if min_len and len(text_raw) < min_len:
            return
        if max_len and len(text_raw) > max_len:
            ctx['content'] = text_raw[:max_len] + '…'
    elif msg_type == 'GIFT':
        min_price = float(cfg.get('min_gift_price', 0) or 0)
        try:
            price = float(ctx.get('giftPrice') or 0)
        except Exception:
            price = 0.0
        if price < min_price:
            return
    elif msg_type == 'SUPER_CHAT':
        min_price = float(cfg.get('min_price', 0) or 0)
        try:
            price = float(ctx.get('price') or 0)
        except Exception:
            price = 0.0
        if price < min_price:
            return

    # 对于弹幕消息，先判断是否应该回复
    if msg_type == 'DANMU':
        if not should_reply_to_danmu(ctx, config):
            return
    
    # 构建模板
    tpl = cfg.get('template', '')
    
    # 动态选择回复模式（仅对弹幕）
    if msg_type == 'DANMU':
        content = str(ctx.get('content', '') or '')
        reply_ctrl = config.get('reply_control', {})
        chat_keywords = reply_ctrl.get('chat_keywords', [])
        
        # 检查是否需要使用chat模式
        needs_chat = any(keyword in content for keyword in chat_keywords)
        current_action = 'chat' if needs_chat else cfg.get('action', 'echo')
        
        logging.info(f"[模式选择] 内容:'{content[:20]}' | 模式:{current_action} | chat关键词:{needs_chat}")
        
        # 优先匹配话术规则（不论什么模式）
        matched_tpl = match_reply_template(msg_type, ctx)
        if matched_tpl:
            tpl = matched_tpl
        elif current_action == 'chat':
            # 使用随机回复模板
            random_tpl = get_random_reply_template(ctx)
            if random_tpl:
                tpl = random_tpl
            else:
                # 使用兜底模板
                fallback_tpl = get_fallback_template()
                if fallback_tpl:
                    tpl = fallback_tpl
                    logging.info(f"[兜底回复] 使用兜底模板: {ctx.get('content', '')[:30]}")
                elif not tpl:
                    logging.info(f"[跳过回复] 无可用模板: {ctx.get('content', '')[:30]}")
                    return
        else:  # echo模式
            # 使用随机回复模板
            random_tpl = get_random_reply_template(ctx)
            if random_tpl:
                tpl = random_tpl
            elif not tpl:
                logging.info(f"[随机回复跳过] 无可用模板: {ctx.get('content', '')[:30]}")
                return
        
        # 更新动作类型用于后续处理
        cfg = dict(cfg)  # 创建副本避免修改原配置
        cfg['action'] = current_action
    
    # 礼物感谢逻辑
    elif msg_type == 'GIFT':
        gift_price = float(ctx.get('giftPrice', 0) or ctx.get('price', 0) or 0)
        gift_thanks_tpl = get_gift_thanks_template(gift_price, ctx)
        if gift_thanks_tpl:
            tpl = gift_thanks_tpl
        elif not tpl:
            logging.info(f"[礼物跳过] 无可用模板，价格:{gift_price}")
            return
    
    # 检查模板是否存在
    if not tpl:
        logging.info(f"[模板缺失] 类型:{msg_type}")
        return
    
    text = render_template(tpl, ctx) if tpl else ctx.get('content', '')
    text = str(text).strip()
    if not text:
        return

    if not check_barrage_global(text):
        return
    text = apply_sensitive_filter(text)
    if text is None:
        return

    sessionid = (config.get('sessions') or {}).get(msg_type, config.get('default_sessionid', 0))
    interrupt = bool(cfg.get('interrupt', False))

    payload = {
        "text": text,
        "type": str(cfg.get('action', 'echo')),
        "interrupt": interrupt,
        "sessionid": sessionid
    }
    
    logging.info(
        f"[/human请求] 类型:{msg_type} | 用户:{ctx.get('username','')} | "
        f"会话ID:{sessionid} | 打断:{interrupt} | 动作:{payload.get('type','')} | "
        f"文本:{text[:50]}{'...' if len(text) > 50 else ''}"
    )
    
    try:
        async with http_session.post(config.get('human_url'), json=payload, timeout=10) as resp:
            response_text = await resp.text()
            if resp.status != 200:
                logging.error(
                    f"[/human错误] 状态码:{resp.status} | 响应:{response_text} | "
                    f"请求:{json.dumps(payload, ensure_ascii=False)}"
                )
            else:
                try:
                    response_json = json.loads(response_text)
                    logging.info(
                        f"[/human成功] 代码:{response_json.get('code','')} | "
                        f"消息:{response_json.get('msg','')} | 会话ID:{sessionid}"
                    )
                except json.JSONDecodeError:
                    logging.info(f"[/human成功] 响应:{response_text} | 会话ID:{sessionid}")
        update_activity()
    except asyncio.TimeoutError:
        logging.error(f"[/human超时] 请求超时(10秒) | 会话ID:{sessionid} | 文本:{text[:30]}")
    except Exception as e:
        logging.error(f"[/human异常] 网络错误:{e} | 会话ID:{sessionid} | 文本:{text[:30]}")


if __name__ == '__main__':
    """
    参考：https://github.com/rsocket/rsocket-py
    > First Run
    pip3 install rsocket
    pip3 install aiohttp
    
    python websocket.py -t taskId1 -t taskId2
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ws', choices=['ws', 'rsocket'], help='监听模式：ws(默认) 或 rsocket')
    parser.add_argument('--uri', default='ws://192.168.1.88:8080/websocket', type=str, help="WebSocket Server Uri")
    # parser.add_argument('-t', action='append', required=True, help="taskIds")
    parser.add_argument('--human_url', default='http://192.168.2.43:8010/human', type=str, help='aiohttp服务 /human 接口地址')
    parser.add_argument('--config', default='config/barrage_config.json', type=str, help='弹幕转发配置文件路径')
    args = parser.parse_args()

    uri = args.uri
    # subscribe_payload_json["data"]["taskIds"] = args.t
    # print(subscribe_payload_json)
    globals()['_RUN_MODE'] = args.mode
    asyncio.run(main(uri, args.human_url, args.config))
