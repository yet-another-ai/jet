"""Best-effort diagnostics that never perform console or file I/O on the game thread."""

from __future__ import annotations

import json
import math
import queue
import threading
from pathlib import Path
from typing import Any, Self, TextIO


class RunOutput:
    """Start a daemon writer; a full queue drops diagnostics instead of blocking.

    close() normally drains accepted events. If its deadline expires, pending
    events are discarded and error reports the unfinished background operation.
    Python cannot interrupt a thread inside an OS write; the daemon can finish
    that operation later without delaying game controls or process shutdown.
    """

    def __init__(
        self, output_dir: Path, save_frames: bool = False, capacity: int = 64
    ) -> None:
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self._directory = Path(output_dir)
        self._save_frames = save_frames
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=capacity)
        self._stopping = threading.Event()
        self._discard = threading.Event()
        self._lock = threading.Lock()
        self._dropped = 0
        self._error: str | None = None
        self._worker = threading.Thread(
            target=self._run, name="doom-output", daemon=True
        )
        self._worker.start()

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def record(self, record: dict[str, Any]) -> None:
        # A shallow snapshot avoids later mutation of top-level result fields.
        self._enqueue(("record", record.copy()))

    def frame(self, index: int, png: bytes) -> None:
        self._enqueue(("frame", (index, png)))

    def message(self, text: str) -> None:
        self._enqueue(("message", text))

    def _enqueue(self, event: tuple[str, Any]) -> None:
        # Never hold this lock during background I/O or a blocking queue call.
        with self._lock:
            if self._stopping.is_set():
                self._dropped += 1
                return
            try:
                self._events.put_nowait(event)
            except queue.Full:
                self._dropped += 1

    def close(self, timeout_seconds: float = 2) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and nonnegative")
        with self._lock:
            self._stopping.set()
        self._worker.join(timeout=timeout_seconds)
        if self._worker.is_alive():
            self._discard.set()
            with self._lock:
                if self._error is None:
                    self._error = (
                        "Output worker did not finish before the close deadline"
                    )
            self._discard_queued()

    def _discard_queued(self) -> None:
        dropped = 0
        while True:
            try:
                self._events.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        with self._lock:
            self._dropped += dropped

    def _run(self) -> None:
        handling_event = False
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            with (self._directory / "decisions.jsonl").open(
                "w", encoding="utf-8"
            ) as log:
                while True:
                    try:
                        event = self._events.get(timeout=0.05)
                    except queue.Empty:
                        if self._stopping.is_set():
                            return
                        continue
                    if self._discard.is_set():
                        with self._lock:
                            self._dropped += 1
                        continue
                    handling_event = True
                    self._handle(log, event)
                    handling_event = False
        except Exception as exc:  # noqa: BLE001 - report all worker failures to the game thread
            with self._lock:
                self._stopping.set()
                self._dropped += int(handling_event)
                if self._error is None:
                    self._error = f"{type(exc).__name__}: {exc}"
            self._discard.set()
            self._discard_queued()

    def _handle(self, log: TextIO, event: tuple[str, Any]) -> None:
        kind, value = event
        if kind == "record":
            log.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            latency = value.get("latency_seconds")
            latency_text = (
                f"{latency:.2f}s" if isinstance(latency, (int, float)) else "?s"
            )
            status = (
                ("applied" if value["applied"] else "discarded")
                if "applied" in value
                else "recorded"
            )
            print(
                f"{value.get('request_id', '?')}: {value.get('choice', '?')} "
                f"{latency_text} {status}",
                flush=True,
            )
        elif kind == "frame":
            index, png = value
            (self._directory / "last-frame.png").write_bytes(png)
            if self._save_frames:
                frames = self._directory / "frames"
                frames.mkdir(exist_ok=True)
                (frames / f"{index:06d}.png").write_bytes(png)
        else:
            print(value, flush=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
