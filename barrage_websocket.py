import argparse
import asyncio
import json
import logging
import os
import time
from asyncio import Event
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncGenerator, Tuple, Dict, Any, Optional

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
            # 浅合并：human_url、default_sessionid、types子项、sessions
            if isinstance(user_cfg, dict):
                if 'human_url' in user_cfg:
                    cfg['human_url'] = user_cfg['human_url']
                if 'default_sessionid' in user_cfg:
                    cfg['default_sessionid'] = user_cfg['default_sessionid']
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
        except Exception:
            logging.info(f"收到非JSON消息: {str(value.data)[:200]}")
            return

        msgs = normalize_incoming(raw)
        for msg_dto in msgs:
            if not isinstance(msg_dto, dict):
                continue
            msg_type = msg_dto.get('type')
            # 控制台输出（尽量兼容）
            if msg_type == "DANMU":
                msg = msg_dto.get('msg', {}) or {}
                logging.info(
                    f"{msg_dto.get('roomId','')} 收到弹幕 {str(msg.get('badgeLevel','')) + str(msg.get('badgeName','')) if msg.get('badgeLevel',0) else ''} "
                    f"{msg.get('username','')}({str(msg.get('uid',''))})：{msg.get('content','')}"
                )
            elif msg_type == "GIFT":
                msg = msg_dto.get('msg', {}) or {}
                action = msg.get('action') or '赠送'
                logging.info(
                    f"{msg_dto.get('roomId','')} 收到礼物 {str(msg.get('badgeLevel','')) + str(msg.get('badgeName','')) if msg.get('badgeLevel',0) else ''} "
                    f"{msg.get('username','')}({str(msg.get('uid',''))}) {action} {msg.get('giftName','')}x{str(msg.get('giftCount',1))}({str(msg.get('giftPrice',0))})"
                )
            # else:
                # logging.info("收到消息 " + json.dumps(msg_dto, ensure_ascii=False))

            # 根据配置异步转发到 /human
            try:
                asyncio.get_event_loop().create_task(self._maybe_dispatch(msg_dto))
            except Exception as e:
                logging.warning(f"调度/human失败: {e}")

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
            return

        ctx = build_context(msg_dto)

        # 基础过滤（长度 / 价格等）
        if msg_type == 'DANMU':
            text_raw = str(ctx.get('content', ''))
            min_len = int(cfg.get('min_length', 0) or 0)
            max_len = int(cfg.get('max_length', 0) or 0)
            if min_len and len(text_raw) < min_len:
                return
            if max_len and len(text_raw) > max_len:
                # 过长截断（防止刷屏）
                ctx['content'] = text_raw[:max_len] + '…'
        elif msg_type == 'GIFT':
            min_price = float(cfg.get('min_gift_price', 0) or 0)
            price = 0.0
            try:
                price = float(ctx.get('giftPrice') or 0)
            except Exception:
                price = 0.0
            if price < min_price:
                return
        elif msg_type == 'SUPER_CHAT':
            min_price = float(cfg.get('min_price', 0) or 0)
            price = 0.0
            try:
                price = float(ctx.get('price') or 0)
            except Exception:
                price = 0.0
            if price < min_price:
                return

        # 渲染模板
        tpl = cfg.get('template', '')
        text = render_template(tpl, ctx) if tpl else ctx.get('content', '')
        text = str(text).strip()
        if not text:
            return

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
        try:
            async with self._http_session.post(self._human_url, json=payload, timeout=10) as resp:
                if resp.status != 200:
                    logging.warning(f"/human 返回非200: {resp.status}")
                else:
                    txt = await resp.text()
                    logging.info(f"/human 响应: {txt}")
        except Exception as e:
            logging.warning(f"调用 /human 异常: {e}")


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
    if human_url:
        cfg['human_url'] = human_url

    async with aiohttp.ClientSession() as http_session:
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
    if human_url:
        cfg['human_url'] = human_url

    async with aiohttp.ClientSession() as http_session:
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
                            # logging.info("收到消息 " + json.dumps(msg_dto, ensure_ascii=False))

                            # 调用 /human（按现有过滤与模板规则）
                            try:
                                await _ws_maybe_dispatch(http_session, cfg, msg_dto)
                            except Exception as e:
                                logging.warning(f"调度/human失败: {e}")
        except Exception as e:
            logging.error(f"WS 连接失败: {e}")

async def _ws_maybe_dispatch(http_session: aiohttp.ClientSession, config: Dict[str, Any], msg_dto: Dict[str, Any]):
    msg_type = msg_dto.get('type', '')
    cfg = get_type_cfg(config, msg_type)
    if not cfg or not cfg.get('enabled', False):
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

    tpl = cfg.get('template', '')
    text = render_template(tpl, ctx) if tpl else ctx.get('content', '')
    text = str(text).strip()
    if not text:
        return

    sessionid = (config.get('sessions') or {}).get(msg_type, config.get('default_sessionid', 0))
    interrupt = bool(cfg.get('interrupt', False))

    payload = {
        "text": text,
        "type": str(cfg.get('action', 'echo')),
        "interrupt": interrupt,
        "sessionid": sessionid
    }
    logging.info(f"/human 请求: {payload}")
    try:
        async with http_session.post(config.get('human_url'), json=payload, timeout=10) as resp:
            if resp.status != 200:
                logging.warning(f"/human 返回非200: {resp.status}")
            else:
                txt = await resp.text()
                logging.info(f"/human 响应: {txt}")
    except Exception as e:
        logging.warning(f"调用 /human 异常: {e}")


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
    parser.add_argument('--uri', default='ws://127.0.0.1:8080/websocket', type=str, help="WebSocket Server Uri")
    # parser.add_argument('-t', action='append', required=True, help="taskIds")
    parser.add_argument('--human_url', default='http://192.168.2.43:8010/human', type=str, help='aiohttp服务 /human 接口地址')
    parser.add_argument('--config', default='config/barrage_config.json', type=str, help='弹幕转发配置文件路径')
    args = parser.parse_args()

    uri = args.uri
    # subscribe_payload_json["data"]["taskIds"] = args.t
    # print(subscribe_payload_json)
    globals()['_RUN_MODE'] = args.mode
    asyncio.run(main(uri, args.human_url, args.config))
