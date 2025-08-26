conda activate nerfstream
cd /mnt/disk1/ftp/file/60397193/LiveTalking
python app.py --config_file config.json

cd /mnt/disk1/ftp/file/60397193/LiveTalking
bash start.sh



# 使用默认配置
python start_action_scanner.py

# 单次扫描
python start_action_scanner.py --once

# 指定扫描目录
python start_action_scanner.py --scan_dir my_action_videos
============
# 只扫描动作视频
python start_scanner.py --mode action

# 同时扫描头像和动作视频
python start_scanner.py --mode both

# 单次扫描动作视频
python start_scanner.py --mode action --once

---

# 数字人弹幕监听与管理

本项目提供统一的管理服务与弹幕监听进程，支持动态会话、话术模板、敏感词过滤、定时任务（自动播报/冷场填充）、以及配置热加载。

## 管理接口（management_server.py）

所有接口统一返回结构：

```json
{"code": 0, "msg": "ok", "data": {...}}
```

- 启动弹幕监听
  - POST `/barrage/start`  body: `{ "sessionid": 123 }`
  - 说明：仅需传入会话ID。服务会写入 `config/barrage_config.json` 的 `default_sessionid` 并以子进程启动 `barrage_websocket.py`。

- 查询状态
  - GET `/barrage/status`

- 停止监听
  - POST `/barrage/stop`

- 四类配置 CRUD（GET=读取，PUT=保存/更新，POST=重置为默认）
  - 话术：`/config/speech`
  - 敏感词：`/config/sensitive`
  - 定时任务：`/config/schedule`
  - 弹幕规则：`/config/barrage_rules`

## 运行弹幕监听（独立模式）

```bash
python barrage_websocket.py --mode ws --uri ws://127.0.0.1:8080/websocket \
  --human_url http://127.0.0.1:8010/human \
  --config config/barrage_config.json
```

## 配置文件说明（config/ 目录）

### 1) barrage_config.json（转发基础配置）
- `human_url`: `/human` 接口地址
- `default_sessionid`: 默认会话ID
- `types`: 各类型开关、模板、阈值（示例：`DANMU.max_length`，`GIFT.min_gift_price`，`SUPER_CHAT.min_price`…）
- `sessions`: 可按类型覆盖会话ID，例如 `{ "DANMU": 123 }`

### 2) speech_config.json（话术模板与规则）
- `templates`: 通用模板集合（可选）
- `reply_rules`: 针对 `DANMU` 的匹配规则（按包含或正则匹配），命中后优先使用该模板。

示例：

```json
{
  "reply_rules": [
    {"match": "你好", "template": "你好，{username}!"},
    {"match": "(价格|多少钱)", "template": "可以看下置顶菜单哦～"}
  ]
}
```

### 3) sensitive_config.json（敏感词过滤）
- `blacklist`: ["词1", "词2"]
- `strategy`: `mask` 或 `block`/`drop`
- `mask_char`: 默认 `*`

行为：
- `mask`：将命中词用 `mask_char` 覆盖
- `block/drop`：直接丢弃该条输出

### 4) schedule_config.json（自动播报与冷场填充）

```json
{
  "auto_broadcast": {
    "enabled": true,
    "interval_sec": 60,
    "messages": ["欢迎来到直播间～"],
    "interrupt": false,
    "msg_type": "DANMU"
  },
  "idle_fill": {
    "enabled": true,
    "idle_threshold_sec": 120,
    "messages": ["有问题可以随时问我哦～"],
    "interrupt": false,
    "msg_type": "DANMU"
  }
}
```

- `interrupt`: 是否打断当前说话
- `msg_type`: 调度消息的逻辑类型，用于按 `barrage_config.json/sessions` 路由到不同会话；未配置则使用 `default_sessionid`

### 5) barrage_rules.json（全局弹幕规则）

```json
{
  "global": {
    "min_len": 0,
    "max_len": 200,
    "rate_limit_per_min": 120
  }
}
```

- `min_len`/`max_len`：全局长度限制（类型内仍可能进行截断，例如 `DANMU.max_length`）
- `rate_limit_per_min`：简单的每分钟全局限流（进程级）

## 热加载

- `barrage_websocket.py` 内置轮询热加载，每 2s 检查四个配置文件的修改时间，自动重新加载，无需重启。
- 变更时日志输出：`检测到配置文件变更，已热加载`。

## 处理流程要点（barrage_websocket.py）

- 统一解析消息 → 构建模板上下文 `build_context()`。
- 先按类型进行基础过滤（长度/金额阈值等）。
- 话术规则匹配 `match_reply_template()`（优先于类型模板）。
- 模板渲染 → 全局规则校验 `check_barrage_global()` → 敏感词过滤 `apply_sensitive_filter()`。
- 发送到 `/human`，按类型或默认会话路由；成功后 `update_activity()` 更新活跃时间。
- 定时任务 `scheduler_loop()`：定时广播与冷场填充，支持是否打断与按类型路由。

## 故障排查

- 确认管理服务与 `/human` 地址可达（`barrage_config.json.human_url`）。
- 查看日志中是否出现热加载提示与 `/human` 响应。
- 调整 `barrage_rules.json.global.rate_limit_per_min` 以避免过度限流。
- 已默认在启动时调用 `update_activity()`，避免立即判定冷场。