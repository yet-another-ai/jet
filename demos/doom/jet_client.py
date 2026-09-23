"""A persistent, asynchronous JSONL subprocess client using only the standard library."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class JetClientError(RuntimeError):
    """Jet rejected a request, violated its protocol, or stopped responding."""


@dataclass(frozen=True)
class DecisionResult:
    request_id: str
    response: dict[str, Any]
    latency_seconds: float


@dataclass(frozen=True)
class _Request:
    request_id: str
    payload: dict[str, Any]
    submitted_at: float


@dataclass(frozen=True)
class _Completion:
    result: DecisionResult | None = None
    error: JetClientError | None = None
    fatal: bool = False


class JetClient:
    """Start Jet immediately; submit/poll from the game thread without pipe I/O.

    Only one request may be outstanding, including a completed result not yet
    collected by poll(). Do not mutate a request after submitting it. A valid
    server error completes that request; transport/protocol errors end the client.
    """

    def __init__(
        self,
        command: list[str],
        stderr_path: Path,
        timeout_seconds: float = 60,
    ) -> None:
        if not command or not all(isinstance(part, str) for part in command):
            raise ValueError("command must be a nonempty list of strings")
        if not 0 < timeout_seconds < float("inf"):
            raise ValueError("timeout_seconds must be finite and positive")
        self._timeout = timeout_seconds
        self._requests: queue.Queue[_Request | None] = queue.Queue(maxsize=1)
        self._completions: queue.Queue[_Completion] = queue.Queue(maxsize=1)
        self._stopping = threading.Event()
        self._state_lock = threading.Lock()
        self._pending = False
        self._fatal_error: JetClientError | None = None
        self._timer: threading.Timer | None = None
        stderr_path = Path(stderr_path)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = stderr_path.open("ab")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
                shell=False,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )
        except (OSError, ValueError) as exc:
            self._stderr.close()
            raise JetClientError(f"Could not start Jet: {exc}") from exc
        self._worker = threading.Thread(target=self._run, name="jet-jsonl", daemon=True)
        try:
            self._worker.start()
        except RuntimeError:
            self._kill()
            self._process.wait(timeout=1)
            self._close_streams()
            raise

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._pending

    def submit(self, request: dict[str, Any]) -> None:
        """Enqueue immediately; JSON encoding and all pipe operations run off-thread."""
        if not isinstance(request, dict):
            raise TypeError("request must be a dictionary")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        with self._state_lock:
            if self._stopping.is_set():
                raise JetClientError("Jet client is closed")
            if self._fatal_error is not None:
                raise JetClientError(f"Jet client is unavailable: {self._fatal_error}")
            if self._pending:
                raise JetClientError(
                    "A Jet request is already in flight; poll it before submitting"
                )
            self._pending = True
            self._requests.put_nowait(
                _Request(request_id, request, time.perf_counter())
            )

    def poll(self) -> DecisionResult | None:
        """Return a completed decision or raise its error, without waiting for Jet."""
        try:
            completion = self._completions.get_nowait()
        except queue.Empty:
            return None
        with self._state_lock:
            self._pending = False
        if completion.error is not None:
            raise completion.error
        return completion.result

    def close(self) -> None:
        """Stop the child and join its worker with bounded waits; safe to repeat."""
        self._stopping.set()
        with self._state_lock:
            timer = self._timer
            self._pending = False
        if timer is not None:
            timer.cancel()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        if self._process.poll() is None:
            try:
                self._process.terminate()
            except OSError:
                pass
            try:
                self._process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._kill()
                try:
                    self._process.wait(timeout=0.5)
                except subprocess.TimeoutExpired as exc:
                    raise JetClientError(
                        "Jet did not exit after termination and kill"
                    ) from exc
        self._worker.join(timeout=1)
        if self._worker.is_alive():
            raise JetClientError("Jet exited but its JSONL worker did not stop")
        if timer is not None:
            timer.join(timeout=0.2)
        self._close_streams()

    def _close_streams(self) -> None:
        for stream in (self._process.stdin, self._process.stdout, self._stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    # Closing stdin can flush data into an already killed child.
                    # Continue closing stdout and the diagnostics file as well.
                    pass

    def _kill(self) -> None:
        try:
            self._process.kill()
        except OSError:
            pass

    def _publish(self, completion: _Completion) -> None:
        with self._state_lock:
            if completion.fatal:
                self._fatal_error = completion.error
            if not self._stopping.is_set():
                self._completions.put_nowait(completion)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                request = self._requests.get(timeout=0.05)
            except queue.Empty:
                return_code = self._process.poll()
                if return_code is not None:
                    # A completed request may still be waiting for poll(). Its
                    # result takes priority; report the exit after it is read.
                    with self._state_lock:
                        pending = self._pending
                    if not pending:
                        self._publish(
                            _Completion(
                                error=JetClientError(
                                    f"Jet exited with code {return_code}"
                                ),
                                fatal=True,
                            )
                        )
                        return
                continue
            if request is None or self._stopping.is_set():
                return
            completion = self._exchange(request)
            self._publish(completion)
            if completion.fatal:
                self._kill()
                return

    def _exchange(self, request: _Request) -> _Completion:
        # The timer kills the child to release BOTH a blocked stdin write and a
        # blocked stdout readline. poll() never waits on a process or a thread.
        guard = threading.Lock()
        finished = False
        timed_out = False

        def expire() -> None:
            nonlocal timed_out
            with guard:
                if not finished:
                    timed_out = True
                    self._kill()

        remaining = max(0, request.submitted_at + self._timeout - time.perf_counter())
        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        with self._state_lock:
            self._timer = timer
            timer.start()
        try:
            try:
                encoded = json.dumps(request.payload, allow_nan=False) + "\n"
            except (TypeError, ValueError, RecursionError) as exc:
                with guard:
                    finished = True
                if timed_out:
                    return self._timeout_error(request)
                return _Completion(
                    error=JetClientError(f"Request is not valid JSON: {exc}")
                )
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(encoded)
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            with guard:
                finished = True
            if timed_out:
                return self._timeout_error(request)
            if not line:
                return_code = self._process.poll()
                return _Completion(
                    error=JetClientError(
                        f"Jet closed stdout before replying to {request.request_id!r} "
                        f"(exit code: {return_code})"
                    ),
                    fatal=True,
                )
            try:
                response = json.loads(line)
            except (json.JSONDecodeError, RecursionError) as exc:
                return _Completion(
                    error=JetClientError(f"Jet returned invalid JSON: {exc}"),
                    fatal=True,
                )
            if not isinstance(response, dict):
                return _Completion(
                    error=JetClientError("Jet response must be a JSON object"),
                    fatal=True,
                )
            if response.get("request_id") != request.request_id:
                return _Completion(
                    error=JetClientError(
                        f"Jet response request_id does not match {request.request_id!r}"
                    ),
                    fatal=True,
                )
            if "error" in response:
                error = response["error"]
                if (
                    not isinstance(error, dict)
                    or not isinstance(error.get("code"), str)
                    or not isinstance(error.get("message"), str)
                ):
                    return _Completion(
                        error=JetClientError("Jet returned a malformed error envelope"),
                        fatal=True,
                    )
                return _Completion(
                    error=JetClientError(
                        f"Jet request {request.request_id!r} failed [{error['code']}]: {error['message']}"
                    )
                )
            return _Completion(
                result=DecisionResult(
                    request.request_id,
                    response,
                    time.perf_counter() - request.submitted_at,
                )
            )
        except (OSError, UnicodeError, ValueError) as exc:
            with guard:
                finished = True
            if timed_out:
                return self._timeout_error(request)
            return _Completion(
                error=JetClientError(f"Jet JSONL communication failed: {exc}"),
                fatal=True,
            )
        finally:
            with guard:
                finished = True
            timer.cancel()
            timer.join(timeout=0.2)
            with self._state_lock:
                self._timer = None

    def _timeout_error(self, request: _Request) -> _Completion:
        return _Completion(
            error=JetClientError(
                f"Jet request {request.request_id!r} timed out after {self._timeout:g} seconds"
            ),
            fatal=True,
        )
