"""Async tosu WebSocket subscriber.

Subscribes concurrently to:
- /websocket/v2/precise — high-rate currentTime + key state (we ignore tosu's
  key state in favor of XQueryKeymap, which is faster and synchronous, but we
  keep currentTime).
- /websocket/v2 — general state, beatmap metadata, gameplay-state transitions.

Holds the most recent useful snapshot in thread-safe attributes. The capture
loop reads these synchronously from the main thread; this client runs in an
asyncio loop on a background thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import websockets

log = logging.getLogger(__name__)

PRECISE_URL = "ws://127.0.0.1:24050/websocket/v2/precise"
V2_URL = "ws://127.0.0.1:24050/websocket/v2"


@dataclass
class TosuSnapshot:
    # Precise (advances at the tosu Precise poll rate, e.g. 10 ms).
    current_time_ms: int = -1
    precise_recv_mono_ns: int = 0

    # v2 (advances at the tosu Common poll rate, e.g. 50 ms).
    state_name: str = ""           # "menu", "play", "songSelect", ...
    state_number: int = -1
    game_paused: bool = False
    map_md5: str = ""              # beatmap.checksum
    map_id: int = 0                # beatmap.id
    map_set: int = 0               # beatmap.set
    map_mode_number: int = -1      # 0 = std
    map_file: str = ""             # files.beatmap (relative to folders.songs)
    songs_folder: str = ""         # folders.songs
    v2_recv_mono_ns: int = 0


class TosuClient:
    """Background-thread asyncio client that maintains a TosuSnapshot."""

    def __init__(self, precise_url: str = PRECISE_URL, v2_url: str = V2_URL):
        self._precise_url = precise_url
        self._v2_url = v2_url
        self._snapshot = TosuSnapshot()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_evt: Optional[asyncio.Event] = None

    def snapshot(self) -> TosuSnapshot:
        with self._lock:
            s = self._snapshot
            return TosuSnapshot(
                current_time_ms=s.current_time_ms,
                precise_recv_mono_ns=s.precise_recv_mono_ns,
                state_name=s.state_name,
                state_number=s.state_number,
                game_paused=s.game_paused,
                map_md5=s.map_md5,
                map_id=s.map_id,
                map_set=s.map_set,
                map_mode_number=s.map_mode_number,
                map_file=s.map_file,
                songs_folder=s.songs_folder,
                v2_recv_mono_ns=s.v2_recv_mono_ns,
            )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="tosu-ws", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self._loop is not None and self._stop_evt is not None:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_evt = asyncio.Event()
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()
            self._loop = None
            self._stop_evt = None

    async def _main(self) -> None:
        tasks = [
            asyncio.create_task(self._consume(self._precise_url, self._apply_precise)),
            asyncio.create_task(self._consume(self._v2_url, self._apply_v2)),
        ]
        await self._stop_evt.wait()
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    async def _consume(self, url: str, handler) -> None:
        backoff = 0.5
        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(url, open_timeout=2, ping_interval=None) as ws:
                    backoff = 0.5
                    while not self._stop_evt.is_set():
                        msg = await ws.recv()
                        try:
                            d = json.loads(msg) if isinstance(msg, (str, bytes, bytearray)) else msg
                        except json.JSONDecodeError:
                            continue
                        try:
                            handler(d)
                        except Exception:
                            log.exception("handler error for %s", url)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("tosu %s disconnected: %s — retrying in %.1fs", url, e, backoff)
                try:
                    await asyncio.wait_for(self._stop_evt.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 5.0)

    def _apply_precise(self, d: dict) -> None:
        ct = d.get("currentTime")
        if ct is None:
            return
        now = time.monotonic_ns()
        with self._lock:
            self._snapshot.current_time_ms = int(ct)
            self._snapshot.precise_recv_mono_ns = now

    def _apply_v2(self, d: dict) -> None:
        state = d.get("state") or {}
        game = d.get("game") or {}
        beatmap = d.get("beatmap") or {}
        files = d.get("files") or {}
        folders = d.get("folders") or {}
        now = time.monotonic_ns()
        with self._lock:
            s = self._snapshot
            s.state_name = str(state.get("name", ""))
            s.state_number = int(state.get("number", -1))
            s.game_paused = bool(game.get("paused", False))
            s.map_md5 = str(beatmap.get("checksum", ""))
            s.map_id = int(beatmap.get("id") or 0)
            s.map_set = int(beatmap.get("set") or 0)
            mode = beatmap.get("mode") or {}
            s.map_mode_number = int(mode.get("number", -1))
            s.map_file = str(files.get("beatmap", ""))
            s.songs_folder = str(folders.get("songs", ""))
            s.v2_recv_mono_ns = now
