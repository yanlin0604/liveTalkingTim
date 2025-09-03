import argparse
import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import random
from typing import Dict, Any, Optional
import re

import aiohttp


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
DEFAULT_CONFIG["rules"] = {
    "global": {
        "min_len": 1,
        "max_len": 120,
        "rate_limit_per_min": 60
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
                if 'rules' in user_cfg and isinstance(user_cfg['rules'], dict):
                    cfg['rules'] = user_cfg['rules']
    except Exception as e:
        logging.warning(f"加载配置失败，使用默认配置: {e}")
    return cfg

# ===== 外部配置（话术/敏感词/定时） ===== #
SPEECH_CFG: Dict[str, Any] = {}
SENSITIVE_CFG: Dict[str, Any] = {}
SCHEDULE_CFG: Dict[str, Any] = {}
BARRAGE_CFG: Dict[str, Any] = {}

# 外部配置路径
SPEECH_PATH = 'config/speech_config.json'
SENSITIVE_PATH = 'config/sensitive_config.json'
SCHEDULE_PATH = 'config/schedule_config.json'
BARRAGE_CFG_PATH = 'config/barrage_config.json'

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
        # 文件不存在时返回空字典，避免 None
        return {}
    except Exception as e:
        logging.warning(f"加载配置失败 {path}: {e}")
        return {}

def load_external_configs():
    """加载外部配置文件（speech/sensitive/schedule 以及 barrage_config）。"""
    global SPEECH_CFG, SENSITIVE_CFG, SCHEDULE_CFG, BARRAGE_CFG
    SPEECH_CFG = _safe_load_json(SPEECH_PATH)
    SENSITIVE_CFG = _safe_load_json(SENSITIVE_PATH)
    SCHEDULE_CFG = _safe_load_json(SCHEDULE_PATH)
    BARRAGE_CFG = _safe_load_json(BARRAGE_CFG_PATH)
    logging.info("外部配置已加载：speech/sensitive/schedule/barrage_config")

async def _config_watcher_loop(poll_sec: int = 2):
    """简单的轮询热加载，无需额外依赖。"""
    last: Dict[str, float] = {}
    paths = [SPEECH_PATH, SENSITIVE_PATH, SCHEDULE_PATH, BARRAGE_CFG_PATH]
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
                    ts = os.path.getmtime(p) if os.path.exists(p) else 0.0
                except Exception:
                    ts = 0.0
                if ts != last.get(p, -1.0):
                    last[p] = ts
                    changed = True
            if changed:
                load_external_configs()
                logging.info("外部配置已热加载。")
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
    
    # 仅基于概率决定是否回复（不再使用关键词强制回复）
    reply_prob = reply_ctrl.get('reply_probability', 0.8)
    if random.random() < reply_prob:
        _REPLY_COUNT += 1
        logging.info(f"[回复决定] 概率命中({reply_prob}): {content[:30]}")
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

def check_barrage_global(text_for_len: str, config: Dict[str, Any]) -> bool:
    """检查全局弹幕规则（长度与速率限制），从 barrage_config.rules.global 读取。"""
    global _RATE_WINDOW_START, _RATE_COUNT
    g = ((config.get('rules') or {}).get('global') or {})
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
    """内置调度：自动播报与冷场填充。
    修复：不使用 time % interval 方式，避免错过触发；增加详尽日志；冷场消息随机选择与防抖。
    """
    auto_idx = 0
    last_auto_ts = time.time()  # 上次自动播报时间
    last_idle_sent_ts = 0.0     # 上次冷场填充发送时间
    # 记录上一次的自动播报开关与间隔，用于热更新后防抖
    prev_auto_enabled: Optional[bool] = None
    current_interval: Optional[int] = None

    while True:
        await asyncio.sleep(1)
        try:
            now = time.time()

            # 每次循环使用最新的调度配置（由 _config_watcher_loop 热更新 SCHEDULE_CFG）
            auto_cfg = (SCHEDULE_CFG.get('auto_broadcast') or {})
            idle_cfg = (SCHEDULE_CFG.get('idle_fill') or {})

            logging.info(
                f"[调度任务] 当前时间: {now}, 是否启用自动播报: {auto_cfg.get('enabled')}, 是否启用冷场填充: {idle_cfg.get('enabled')}"
            )

            # 处理自动播报开关/间隔的变更：
            # - 首次赋值或从关闭->开启：重置 last_auto_ts，避免立刻触发
            # - 间隔改变：重置 last_auto_ts，按照新间隔重新计时
            enabled_now = bool(auto_cfg.get('enabled')) if auto_cfg else False
            interval_now = int((auto_cfg.get('interval_sec', 0) or 0)) if auto_cfg else 0
            if prev_auto_enabled is None:
                prev_auto_enabled = enabled_now
                current_interval = interval_now
                # 首次进入，无需调整 last_auto_ts（已在启动时设置为 now）
            else:
                if enabled_now and prev_auto_enabled is False:
                    last_auto_ts = now
                if interval_now != (current_interval or 0):
                    current_interval = interval_now
                    last_auto_ts = now
                prev_auto_enabled = enabled_now

            # 自动播报
            if auto_cfg.get('enabled') and (auto_cfg.get('messages')):
                interval = int(auto_cfg.get('interval_sec', 0) or 0)
                if interval > 0 and (now - last_auto_ts) >= interval:
                    msg = auto_cfg['messages'][auto_idx % len(auto_cfg['messages'])]
                    auto_idx += 1
                    auto_interrupt = bool(auto_cfg.get('interrupt', False))
                    auto_type = auto_cfg.get('msg_type')
                    logging.info(f"[定时-自动播报] 触发 | 间隔:{interval}s | 文本:{str(msg)[:50]}")
                    await _send_human_text(http_session, config, str(msg), interrupt=auto_interrupt, msg_type=auto_type)
                    last_auto_ts = now
                    update_activity()

            # 冷场填充
            if idle_cfg.get('enabled') and (idle_cfg.get('messages')):
                th = int(idle_cfg.get('idle_threshold_sec', 0) or 0)
                if th > 0 and (now - _LAST_ACTIVITY_TS) >= th and (now - last_idle_sent_ts) >= th:
                    idle_interrupt = bool(idle_cfg.get('interrupt', False))
                    idle_type = idle_cfg.get('msg_type')
                    # 随机选择一条冷场消息
                    try:
                        msg = random.choice(idle_cfg['messages'])
                    except Exception:
                        msg = idle_cfg['messages'][0]
                    logging.info(f"[定时-冷场填充] 触发 | 静默≥{th}s | 文本:{str(msg)[:50]}")
                    await _send_human_text(http_session, config, str(msg), interrupt=idle_interrupt, msg_type=idle_type)
                    last_idle_sent_ts = now
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
    # 便于排查：中文直显，附带 human_url 与 msg_type
    try:
        logging.info(
            "[HUMAN-PAYLOAD] url=%s | msg_type=%s | %s",
            str(config.get('human_url')),
            str(msg_type or ''),
            json.dumps(payload, ensure_ascii=False)
        )
    except Exception:
        logging.info("[HUMAN-PAYLOAD] %s", str(payload))
    try:
        async with http_session.post(config.get('human_url'), json=payload, timeout=10) as resp:
            response_text = await resp.text()
            if resp.status != 200:
                logging.warning(
                    "[/human错误] 状态码:%s | 响应:%s | 请求:%s",
                    resp.status,
                    response_text,
                    json.dumps(payload, ensure_ascii=False)
                )
            else:
                # 输出成功响应，尽量解析为 JSON 结构
                try:
                    response_json = json.loads(response_text)
                    logging.info("[/human成功] 代码:%s | 消息:%s | 原始:%s",
                                 str(response_json.get('code', '')),
                                 str(response_json.get('msg', '')),
                                 json.dumps(response_json, ensure_ascii=False))
                except json.JSONDecodeError:
                    logging.info("[/human成功] 响应:%s", response_text)
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


def _extract_json_objects(text: str) -> list:
    """
    从可能包含噪音/拼接/包裹的文本中提取一个或多个 JSON 对象或数组。
    处理要点：
    - 去除 BOM 与首尾空白
    - 直接整体解析；失败则按括号配对在顶层提取 {..} 或 [..]
    - 若解析结果为字符串且形如 JSON，再进行二次解析
    返回解析后的 Python 对象列表。
    """
    results = []

    if not isinstance(text, str) or not text:
        return results

    # 去 BOM/空白
    s = text.lstrip('\ufeff').strip()

    def try_load(piece: str):
        try:
            obj = json.loads(piece)
            # 如果是被字符串包裹的 JSON，再尝试二次解析
            if isinstance(obj, str):
                st = obj.lstrip('\ufeff').strip()
                if (st.startswith('{') and st.endswith('}')) or (st.startswith('[') and st.endswith(']')):
                    try:
                        return json.loads(st)
                    except Exception:
                        return obj
            return obj
        except Exception:
            return None

    # 1) 整体尝试
    obj = try_load(s)
    if obj is not None:
        return [obj]

    # 2) 基于括号配对切分（考虑字符串转义）
    in_string = False
    escape = False
    depth = 0
    start_idx = None
    open_char = None

    def is_open(c):
        return c in '{['

    def is_close(c):
        return c in '}]'

    pairs = {']': '[', '}': '{'}

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if depth == 0 and is_open(ch):
            start_idx = i
            open_char = ch
            depth = 1
            continue

        if depth > 0:
            if is_open(ch):
                depth += 1
            elif is_close(ch):
                # 合法配对才减少
                if pairs.get(ch) == open_char or depth > 1:
                    depth -= 1
                # 当回到顶层，截取候选
                if depth == 0 and start_idx is not None:
                    piece = s[start_idx:i+1]
                    parsed = try_load(piece)
                    if parsed is not None:
                        results.append(parsed)
                    start_idx = None
                    open_char = None

    if not results:
        # 维持原观测性：打印原文（截断避免刷屏）
        logging.info(f"收到非JSON文本: {s[:500]}")
    return results
def _extract_json_pieces(text: str) -> list:
    """
    与 _extract_json_objects 类似，但返回 (obj, start, end) 列表，
    便于调用方从原始文本中移除已消费的 JSON 片段并保留尾部未完成的残余。
    """
    pieces = []

    if not isinstance(text, str) or not text:
        return pieces

    s = text.lstrip('\ufeff').strip()

    def try_load(piece: str):
        try:
            obj = json.loads(piece)
            if isinstance(obj, str):
                st = obj.lstrip('\ufeff').strip()
                if (st.startswith('{') and st.endswith('}')) or (st.startswith('[') and st.endswith(']')):
                    try:
                        return json.loads(st)
                    except Exception:
                        return obj
            return obj
        except Exception:
            return None

    # 整体尝试（若成功，返回整段）
    obj = try_load(s)
    if obj is not None:
        # 计算在原始 text 中的位置（s 去除了前后空白，因此定位到原文需要查找）
        idx = text.find(s)
        if idx >= 0:
            pieces.append((obj, idx, idx + len(s)))
        else:
            pieces.append((obj, 0, len(text)))
        return pieces

    # 括号配对扫描，生成分片
    in_string = False
    escape = False
    depth = 0
    start_idx = None
    open_char = None

    def is_open(c):
        return c in '{['

    def is_close(c):
        return c in '}]'

    pairs = {']': '[', '}': '{'}

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if depth == 0 and is_open(ch):
            start_idx = i
            open_char = ch
            depth = 1
            continue

        if depth > 0:
            if is_open(ch):
                depth += 1
            elif is_close(ch):
                if pairs.get(ch) == open_char or depth > 1:
                    depth -= 1
                if depth == 0 and start_idx is not None:
                    piece = s[start_idx:i+1]
                    parsed = try_load(piece)
                    if parsed is not None:
                        # 映射回原始 text 的区间
                        base = text.find(s)
                        begin = (base if base >= 0 else 0) + start_idx
                        end = (base if base >= 0 else 0) + i + 1
                        pieces.append((parsed, begin, end))
                    start_idx = None
                    open_char = None

    return pieces

    # 去 BOM/空白
    s = text.lstrip('\ufeff').strip()

    def try_load(piece: str):
        try:
            obj = json.loads(piece)
            # 如果是被字符串包裹的 JSON，再尝试二次解析
            if isinstance(obj, str):
                st = obj.lstrip('\ufeff').strip()
                if (st.startswith('{') and st.endswith('}')) or (st.startswith('[') and st.endswith(']')):
                    try:
                        return json.loads(st)
                    except Exception:
                        return obj
            return obj
        except Exception:
            return None

    # 1) 整体尝试
    obj = try_load(s)
    if obj is not None:
        return [obj]

    # 2) 基于括号配对切分（考虑字符串转义）
    in_string = False
    escape = False
    depth = 0
    start_idx = None
    open_char = None

    def is_open(c):
        return c in '{['

    def is_close(c):
        return c in '}]'

    pairs = {']': '[', '}': '{'}

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if depth == 0 and is_open(ch):
            start_idx = i
            open_char = ch
            depth = 1
            continue

        if depth > 0:
            if is_open(ch):
                depth += 1
            elif is_close(ch):
                # 合法配对才减少
                if pairs.get(ch) == open_char or depth > 1:
                    depth -= 1
                # 当回到顶层，截取候选
                if depth == 0 and start_idx is not None:
                    piece = s[start_idx:i+1]
                    parsed = try_load(piece)
                    if parsed is not None:
                        results.append(parsed)
                    start_idx = None
                    open_char = None

    if not results:
        # 维持原观测性：打印原文（截断避免刷屏）
        logging.info(f"收到非JSON文本: {s[:500]}")
    return results

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


def _log_message_details(msg_dto: Dict[str, Any]):
    """记录消息详细信息的统一方法"""
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

def _decode_binary_data(data: bytes) -> Optional[str]:
    """尝试多种编码格式解码二进制数据"""
    # 支持的编码格式列表，按优先级排序
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'ascii']
    
    for encoding in encodings:
        try:
            decoded_text = data.decode(encoding, errors='strict')
            logging.debug(f"[解码成功] 使用编码: {encoding}, 数据长度: {len(decoded_text)}")
            return decoded_text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.debug(f"[解码失败] 编码 {encoding} 出现异常: {e}")
            continue
    
    # 如果所有编码都失败，尝试使用 UTF-8 忽略错误
    try:
        decoded_text = data.decode('utf-8', errors='ignore')
        logging.warning(f"[解码降级] 使用UTF-8忽略错误模式，原始长度: {len(data)}, 解码长度: {len(decoded_text)}")
        return decoded_text
    except Exception as e:
        logging.error(f"[解码完全失败] 无法解码二进制数据: {e}")
        return None

async def main(websocket_uri: str, human_url: str, config_path: Optional[str]):
    """主函数：启动 WebSocket 监听服务"""
    await ws_listen(websocket_uri, human_url, config_path)

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
                    buffer = ""
                    async for msg in ws:
                        text = None
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            # 使用统一的二进制解码方法
                            text = _decode_binary_data(msg.data)
                            if text is None:
                                continue
                            logging.debug(f"[WS] 二进制数据解码成功，长度: {len(text)}")
                        elif msg.type == aiohttp.WSMsgType.TEXT:
                            text = msg.data
                            logging.debug(f"[WS] 收到文本数据，长度: {len(text)}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logging.warning(f"WS 错误: {ws.exception()}")
                            break
                        else:
                            continue

                        if not text:
                            continue

                        # 累积缓冲，处理带噪音前缀或跨帧拼接
                        buffer += text

                        while True:
                            pieces = _extract_json_pieces(buffer)
                            if not pieces:
                                # 若始终无法解析，避免缓冲无限增长，最多保留 1MB 尾部
                                if len(buffer) > 1024 * 1024:
                                    buffer = buffer[-512 * 1024:]
                                break

                            # 顺序处理分片，仅保留最后一个分片后的尾部作为残余
                            last_end = 0
                            for raw, start, end in pieces:
                                last_end = max(last_end, end)
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

                            # 仅保留最后一个 JSON 片段之后的残余字符串
                            buffer = buffer[last_end:]
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
        # 优先匹配话术规则；未命中则直接走 chat，并使用原文
        matched_tpl = match_reply_template(msg_type, ctx)
        if matched_tpl:
            tpl = matched_tpl
            current_action = cfg.get('action', 'echo')
            logging.info(f"[路由] 话术命中 -> echo | 模板:{matched_tpl} | 文本:{content[:30]}")
        else:
            tpl = ''  # 使用原文
            current_action = 'chat'
            logging.info(f"[路由] 话术未命中 -> chat | 文本:{content[:30]}")

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
    
    # 检查模板是否存在；缺失时使用原文
    if not tpl:
        logging.info(f"[模板缺失] 类型:{msg_type} | 使用原文回复")
    
    text = render_template(tpl, ctx) if tpl else ctx.get('content', '')
    text = str(text).strip()
    if not text:
        return

    # 使用运行时的全局弹幕规则配置（优先 BARRAGE_CFG，其次 config）
    runtime_cfg = BARRAGE_CFG if BARRAGE_CFG else config
    if not check_barrage_global(text, runtime_cfg):
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


def setup_logging():
    """配置日志系统：同时输出到控制台和文件，支持日志轮转"""
    # 创建logs目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 文件处理器 - 支持日志轮转（每个文件最大10MB，保留5个备份）
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'barrage_websocket.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info("日志系统已初始化 - 输出到控制台和文件: logs/barrage_websocket.log")


if __name__ == '__main__':
    """
    WebSocket 弹幕监听服务
    
    使用方法:
    python barrage_websocket.py --uri ws://localhost:8080/websocket --human_url http://localhost:8010/human
    """
    # 初始化日志系统
    setup_logging()
    
    parser = argparse.ArgumentParser(description='WebSocket 弹幕监听服务')
    parser.add_argument('--uri', default='ws://192.168.1.88:8080/websocket', type=str, help="WebSocket Server Uri")
    parser.add_argument('--human_url', default='http://192.168.2.43:8010/human', type=str, help='aiohttp服务 /human 接口地址')
    parser.add_argument('--config', default='config/barrage_config.json', type=str, help='弹幕转发配置文件路径')
    args = parser.parse_args()

    asyncio.run(main(args.uri, args.human_url, args.config))
