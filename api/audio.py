###############################################################################
#  Copyright (C) 2025 unimed
#  email: zengyanlin99@gmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

"""
音频上传与管理 API（基于 aiohttp）
- POST /audio/upload: 上传本地音频文件（multipart/form-data）
- POST /audio/upload_url: 通过远程 URL 下载并保存音频
- GET  /audio/list: 列出已保存的音频条目

说明：
- 文件统一保存至 data/audio/YYYYMMDD/ 目录
- 通过已注册的静态目录 /data 可直接访问，例如返回的 url: /data/audio/20250808/xxxx.wav
- 使用 data/audio_index.json 管理索引
"""

import aiohttp
from aiohttp import web
import asyncio
import os
import uuid
import json
import datetime
from pathlib import Path
import threading
from typing import Dict, Any
import contextlib

# 目录与索引文件
AUDIO_BASE_DIR = Path("data/audio")
AUDIO_INDEX_FILE = Path("data/audio_index.json")

# 并发安全
_audio_index_lock = threading.Lock()


def _ensure_dirs() -> None:
    AUDIO_BASE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not AUDIO_INDEX_FILE.exists():
        with AUDIO_INDEX_FILE.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


def _load_index() -> Dict[str, Any]:
    _ensure_dirs()
    try:
        with AUDIO_INDEX_FILE.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_index(index: Dict[str, Any]) -> None:
    _ensure_dirs()
    with AUDIO_INDEX_FILE.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _today_dir() -> Path:
    d = datetime.datetime.now().strftime("%Y%m%d")
    p = AUDIO_BASE_DIR / d
    p.mkdir(parents=True, exist_ok=True)
    return p


def _gen_id() -> str:
    return uuid.uuid4().hex


def _sanitize_filename(name: str, default: str = "audio.wav") -> str:
    """提取纯文件名：
    - 兼容 \ 与 / 分隔符，去除路径部分
    - 去除前后空白
    - 避免全空返回默认名
    """
    if not name:
        return default
    # 统一分隔符
    name = str(name).strip()
    name = name.replace("\\", "/")
    name = name.split("/")[-1]
    return name or default

class AudioAPI:
    """音频管理接口"""

    async def upload_file(self, request: web.Request) -> web.Response:
        """
        接收 multipart/form-data 上传，字段：
        - file: 必填，音频文件
        - filename: 可选，保存文件名（仅作为参考，最终会添加唯一前缀）
        返回：{ id, filename, path, url, source, uploaded_at }
        """
        reader = await request.multipart()
        target_dir = _today_dir()

        # 状态变量：在遇到file时立刻写入；循环结束后根据 filename 再统一重命名
        original_filename: str = ""        # 从上传 part.filename 获取
        desired_filename: str = ""         # 从表单文本字段 filename 获取
        saved_name: str = ""               # 实际落盘名
        final_path: Path | None = None
        size: int = 0
        record_id = _gen_id()

        async for part in reader:
            if part.name == "file":
                # 以 part.filename 作为原始名（客户端通常会带）
                original_filename = _sanitize_filename(part.filename or "audio.wav")
                suffix = "." + original_filename.split(".")[-1].lower() if "." in original_filename else ".wav"

                saved_name = original_filename
                if not saved_name.lower().endswith(suffix):
                    saved_name = saved_name + suffix
                path_candidate = target_dir / saved_name
                if path_candidate.exists():
                    uid = _gen_id()
                    stem = os.path.splitext(saved_name)[0]
                    saved_name = f"{stem}_{uid}{suffix}"
                    path_candidate = target_dir / saved_name

                final_path = path_candidate
                size = 0
                with final_path.open("wb") as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        size += len(chunk)
                        f.write(chunk)

            elif part.name == "filename":
                # 记录期望文件名，循环结束后统一处理（兼容 filename 在 file 之前的情况）
                try:
                    desired_filename = _sanitize_filename(await part.text())
                except Exception:
                    desired_filename = ""

        if not final_path:
            return web.json_response({"error": "missing file field"}, status=400)

        # 如果提供了期望文件名，在此统一尝试重命名
        if desired_filename:
            suffix = "." + desired_filename.split(".")[-1].lower() if "." in desired_filename else ".wav"
            new_name = desired_filename
            if not new_name.lower().endswith(suffix):
                new_name = new_name + suffix
            new_path = target_dir / new_name
            if new_path != final_path:
                if new_path.exists():
                    uid = _gen_id()
                    stem = os.path.splitext(new_name)[0]
                    new_name = f"{stem}_{uid}{suffix}"
                    new_path = target_dir / new_name
                try:
                    final_path.rename(new_path)
                    saved_name = new_name
                    final_path = new_path
                except Exception:
                    # 忽略重命名失败
                    pass

        # 展示用文件名优先取 desired_filename，其次 original_filename，最后 saved_name
        display_filename = desired_filename or original_filename or saved_name

        # 构造绝对路径与绝对URL
        abs_fs_path = str(final_path.resolve())
        rel_path = final_path.relative_to(Path("data"))
        url_path = f"/data/{rel_path.as_posix()}"
        absolute_url = f"{request.scheme}://{request.host}{url_path}"

        # 写入索引（以record_id为key）
        item = {
            "id": record_id,
            "filename": display_filename,
            "saved_filename": saved_name or final_path.name,
            "path": abs_fs_path,
            "url": absolute_url,
            "source": "upload",
            "uploaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "size": size,
        }
        with _audio_index_lock:
            index = _load_index()
            index[record_id] = item
            _save_index(index)

        return web.json_response(item)

    async def upload_url(self, request: web.Request) -> web.Response:
        """
        接收 JSON：{"url": "http://..."}
        由服务器下载保存到本地。
        返回：{ id, filename, path, url, source, uploaded_at }
        """
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        src_url = (data or {}).get("url", "").strip()
        if not src_url:
            return web.json_response({"error": "url is required"}, status=400)

        # 期望文件名（可选）：优先 JSON，其次 query 参数
        desired_name = ""
        if isinstance(data, dict):
            for key in ("filename", "name", "file_name", "title"):
                val = data.get(key)
                if val:
                    desired_name = _sanitize_filename(str(val))
                    break
        if not desired_name:
            q = request.rel_url.query
            for key in ("filename", "name"):
                val = q.get(key)
                if val:
                    desired_name = _sanitize_filename(str(val))
                    break

        # 推断原始文件名
        url_path_only = src_url.split("?")[0].split("#")[0]
        inferred_name = _sanitize_filename(os.path.basename(url_path_only) or "audio.wav")

        # 展示文件名优先 desired_name
        display_filename = desired_name or inferred_name
        # 后缀：优先从展示名取；若无则从推断名取；再无则 .wav
        if "." in display_filename:
            suffix = "." + display_filename.split(".")[-1].lower()
        elif "." in inferred_name:
            suffix = "." + inferred_name.split(".")[-1].lower()
        else:
            suffix = ".wav"

        # 保存名初始取展示名；若无后缀则补上
        target_dir = _today_dir()
        saved_name = display_filename
        if not saved_name.lower().endswith(suffix):
            saved_name = saved_name + suffix
        final_path = target_dir / saved_name
        if final_path.exists():
            uniq = _gen_id()
            stem = os.path.splitext(saved_name)[0]
            saved_name = f"{stem}_{uniq}{suffix}"
            final_path = target_dir / saved_name

        # 下载
        timeout = aiohttp.ClientTimeout(total=300)
        size = 0
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(src_url) as resp:
                    if resp.status != 200:
                        return web.json_response({"error": f"failed to download: {resp.status}"}, status=400)
                    with final_path.open("wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            if not chunk:
                                continue
                            size += len(chunk)
                            f.write(chunk)
            except Exception as e:
                # 清理失败文件
                with contextlib.suppress(Exception):
                    if final_path.exists():
                        final_path.unlink()
                return web.json_response({"error": f"download error: {e}"}, status=400)

        abs_fs_path = str(final_path.resolve())
        rel_path = final_path.relative_to(Path("data"))
        url_path = f"/data/{rel_path.as_posix()}"
        absolute_url = f"{request.scheme}://{request.host}{url_path}"

        record_id = _gen_id()
        item = {
            "id": record_id,
            "filename": display_filename,
            "saved_filename": saved_name,
            "path": abs_fs_path,
            "url": absolute_url,
            "source": "url",
            "original_url": src_url,
            "uploaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "size": size,
        }
        with _audio_index_lock:
            index = _load_index()
            index[record_id] = item
            _save_index(index)

        return web.json_response(item)

    async def list_audios(self, request: web.Request) -> web.Response:
        """列出所有音频条目"""
        with _audio_index_lock:
            index = _load_index()
        # 以时间倒序
        items = sorted(index.values(), key=lambda x: x.get("uploaded_at", ""), reverse=True)
        return web.json_response({"items": items, "count": len(items)})

    async def delete_audio(self, request: web.Request) -> web.Response:
        """删除音频：根据 id 删除索引记录并尝试删除文件
        支持以下方式提供 id：
        - 路径参数 /audio/{id}
        - 查询参数 ?id=xxx
        - JSON Body {"id": "xxx"}
        """
        # 1) 路径参数
        audio_id = request.match_info.get("id")
        # 2) 查询参数
        if not audio_id:
            audio_id = request.rel_url.query.get("id")
        # 3) JSON Body
        if not audio_id:
            with contextlib.suppress(Exception):
                data = await request.json()
                if isinstance(data, dict):
                    audio_id = str(data.get("id") or "").strip()

        if not audio_id:
            return web.json_response({"error": "id is required"}, status=400)

        with _audio_index_lock:
            index = _load_index()
            item = index.get(audio_id)
            if not item:
                return web.json_response({"error": "not found"}, status=404)

            # 尝试删除文件
            removed_file = False
            file_path = item.get("path")
            if file_path:
                p = Path(file_path)
                if p.exists():
                    with contextlib.suppress(Exception):
                        p.unlink()
                        removed_file = True

            # 删除索引项并保存
            del index[audio_id]
            _save_index(index)

        return web.json_response({
            "deleted": True,
            "id": audio_id,
            "removed_file": removed_file,
        })
