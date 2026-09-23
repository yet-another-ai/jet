"""A separate-process Doom viewport and candidate-likelihood dashboard."""

from __future__ import annotations

import math
import multiprocessing
import os
import queue
from functools import lru_cache
from pathlib import Path
from typing import Any

WINDOW_SIZE = (1120, 680)
FRAME_RECT = (24, 128, 640, 400)
BACKGROUND = (12, 17, 27)
PANEL = (20, 28, 42)
BORDER = (39, 51, 70)
TEXT = (233, 240, 249)
MUTED = (136, 154, 179)
TEAL = (88, 222, 185)
AMBER = (246, 190, 91)
BLUE = (116, 170, 255)


def _probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= 1 or not math.isfinite(value):
        return None
    return float(value)


def probability_text(value: Any) -> str:
    """Missing/invalid values are unknown, while a real zero remains zero."""
    probability = _probability(value)
    return "--" if probability is None else f"{probability * 100:.1f}%"


def snapshot_status(snapshot: dict[str, Any]) -> tuple[str, tuple[int, int, int]]:
    if snapshot.get("thinking"):
        return "INFERENCE RUNNING", BLUE
    if snapshot.get("executing"):
        return "EXECUTING ACTION", TEAL
    if snapshot.get("settling"):
        return "OBSERVING RESULT", BLUE
    return (
        ("DECISION READY", TEAL)
        if snapshot.get("frame_id")
        else ("WAITING FOR JET", MUTED)
    )


@lru_cache(maxsize=20)
def _font(size: int, bold: bool = False):
    import pygame

    return pygame.font.SysFont("segoeui,dejavusans,arial", size, bold=bold)


def _text(
    surface,
    text: str,
    position: tuple[int, int],
    size: int = 16,
    color: tuple[int, int, int] = TEXT,
    bold: bool = False,
    max_width: int | None = None,
    right: bool = False,
) -> None:
    font = _font(size, bold)
    if max_width is not None:
        original = text
        while text and font.size(text)[0] > max_width:
            original = original[:-1]
            text = original + "…" if original else ""
    rendered = font.render(text, True, color)
    x, y = position
    surface.blit(rendered, (x - rendered.get_width() if right else x, y))


def _metric(value: Any, *, seconds: bool = False) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return "--"
    if seconds:
        return f"{value:.2f}s" if value >= 0 else "--"
    return f"{value:g}"


def render_snapshot(surface, snapshot: dict[str, Any], labels: dict[str, str]) -> None:
    """Draw one complete snapshot; no display events, filesystem, or Doom calls.

    The caller initializes pygame.font. Intended for the UI process's main
    thread; tests may render directly onto an offscreen Surface.
    """
    import pygame

    surface.fill(BACKGROUND)
    _text(surface, "JET", (24, 20), 29, TEAL, True)
    _text(surface, "/", (84, 24), 25, MUTED)
    _text(surface, "DOOM", (105, 23), 26, TEXT, True)
    _text(surface, "VISUAL DECISION MONITOR", (218, 33), 12, MUTED, True)
    pygame.draw.line(surface, BORDER, (24, 80), (1096, 80))

    completed = bool(snapshot.get("frame_id"))
    status, status_color = snapshot_status(snapshot)
    pygame.draw.rect(surface, PANEL, (840, 25, 256, 32), border_radius=16)
    pygame.draw.circle(surface, status_color, (858, 41), 4)
    _text(surface, status, (873, 31), 12, status_color, True)

    frame_heading = "LIVE GAME FRAME"
    if snapshot.get("game_tic") is not None:
        frame_heading += f"  ·  TIC {_metric(snapshot['game_tic'])}"
    _text(surface, frame_heading, (24, 101), 12, MUTED, True)
    _text(
        surface,
        f"EPISODE {_metric(snapshot.get('episode'))}",
        (664, 101),
        12,
        MUTED,
        True,
        right=True,
    )
    pygame.draw.rect(surface, (5, 8, 14), FRAME_RECT)
    pixels = snapshot.get("frame_rgb")
    size = snapshot.get("frame_size")
    valid_frame = (
        isinstance(pixels, bytes)
        and isinstance(size, (tuple, list))
        and len(size) == 2
        and all(isinstance(n, int) and n > 0 for n in size)
        and len(pixels) == size[0] * size[1] * 3
    )
    if valid_frame:
        width, height = size
        scale = min(FRAME_RECT[2] / width, FRAME_RECT[3] / height)
        scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        frame = pygame.image.frombuffer(pixels, (width, height), "RGB")
        frame = pygame.transform.scale(frame, scaled_size)
        surface.blit(
            frame,
            (
                FRAME_RECT[0] + (FRAME_RECT[2] - scaled_size[0]) // 2,
                FRAME_RECT[1] + (FRAME_RECT[3] - scaled_size[1]) // 2,
            ),
        )
    else:
        _text(
            surface,
            "Waiting for a game frame" if pixels is None else "Frame unavailable",
            (224, 318),
            19,
            MUTED,
        )
    pygame.draw.rect(surface, BORDER, FRAME_RECT, width=1)

    pygame.draw.rect(surface, PANEL, (24, 544, 640, 70), border_radius=10)
    metrics = (
        ("HEALTH", snapshot.get("health"), False),
        ("AMMO", snapshot.get("ammo"), False),
        ("LAST INFERENCE", snapshot.get("latency_seconds"), True),
    )
    for index, (label, value, seconds) in enumerate(metrics):
        x = 42 + index * 211
        _text(surface, label, (x, 555), 11, MUTED, True)
        _text(surface, _metric(value, seconds=seconds), (x, 574), 24, TEXT, True)
        if index:
            pygame.draw.line(surface, BORDER, (x - 16, 559), (x - 16, 600))
    _text(surface, "LAST COMPLETED FRAME", (24, 630), 11, MUTED, True)
    frame_label = str(snapshot.get("frame_id") or "No decision yet")
    if completed and snapshot.get("decision_game_tic") is not None:
        frame_label += f"  ·  tic {_metric(snapshot['decision_game_tic'])}"
    _text(surface, frame_label, (24, 648), 14, TEXT, max_width=640)

    _text(surface, "Candidate probabilities", (688, 97), 23, TEXT, True)
    _text(
        surface,
        "Last decision · relative likelihood of each action",
        (688, 128),
        13,
        MUTED,
    )
    probabilities = snapshot.get("probabilities")
    probabilities = probabilities if isinstance(probabilities, dict) else {}
    choice = snapshot.get("choice") if completed else None
    applied = snapshot.get("applied") is True
    selection_color = TEAL if applied else AMBER
    for index, (key, label) in enumerate(labels.items()):
        y = 160 + index * 48
        selected = key == choice
        row_color = (
            (23, 51, 47)
            if selected and applied
            else (53, 43, 28)
            if selected
            else PANEL
        )
        pygame.draw.rect(surface, row_color, (688, y, 408, 42), border_radius=8)
        if selected:
            pygame.draw.rect(
                surface, selection_color, (692, y + 8, 3, 26), border_radius=2
            )
        _text(
            surface,
            label,
            (704, y + 4),
            15,
            selection_color if selected else TEXT,
            selected,
            max_width=292,
        )
        value = probabilities.get(key)
        _text(
            surface,
            probability_text(value),
            (1080, y + 4),
            15,
            selection_color if selected else MUTED,
            selected,
            right=True,
        )
        pygame.draw.rect(surface, BORDER, (704, y + 31, 376, 4), border_radius=2)
        probability = _probability(value)
        if probability is not None and probability > 0:
            width = round(376 * probability)
            if width:
                pygame.draw.rect(
                    surface,
                    selection_color if selected else BLUE,
                    (704, y + 31, width, 4),
                    border_radius=2,
                )
    if completed and choice in labels:
        selected_status = (
            "Last decision applied" if applied else "Last decision not applied (stale)"
        )
        pygame.draw.circle(surface, selection_color, (694, 609), 4)
        _text(surface, selected_status, (706, 600), 13, selection_color)
    else:
        _text(surface, "No completed decision", (688, 600), 13, MUTED)
    _text(surface, "These scores are not success probabilities.", (688, 632), 12, MUTED)
    _text(surface, "Esc or close this window to stop the run.", (688, 651), 12, MUTED)


def _dashboard_main(
    labels: dict[str, str], screenshot_path: str | None, snapshots, stopped, failed
) -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pygame = None
    try:
        import pygame

        pygame.display.init()
        pygame.font.init()
        surface = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Jet · Doom visual decisions")
        clock = pygame.time.Clock()
        snapshot: dict[str, Any] = {}
        saved_decision = None
        while not stopped.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    stopped.set()
            if stopped.is_set():
                break
            try:
                while True:
                    snapshot = snapshots.get_nowait()
            except queue.Empty:
                pass
            render_snapshot(surface, snapshot, labels)
            pygame.display.flip()
            decision = (snapshot.get("episode"), snapshot.get("frame_id"))
            if screenshot_path and decision[1] and decision != saved_decision:
                path = Path(screenshot_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(surface, str(path))
                saved_decision = decision
            clock.tick(30)
    except Exception:
        # Publish the failure before stopped: the game may observe the stop
        # while this process is still shutting down and exitcode is still None.
        failed.set()
        raise
    finally:
        # Signal before SDL shutdown: even a failed/blocked cleanup cannot leave
        # the game assuming its control window is healthy.
        stopped.set()
        if pygame is not None:
            pygame.quit()


class Dashboard:
    """Spawn the UI; publish only complete snapshots from the game thread.

    A saturated queue drops the new snapshot. Every later snapshot must repeat
    the last completed decision, so a dropped display frame cannot lose that
    decision's probabilities. No queue reads or image serialization occur here.
    """

    def __init__(
        self, action_labels: dict[str, str], screenshot_path: Path | None = None
    ) -> None:
        if len(action_labels) != 9:
            raise ValueError("The Doom dashboard requires exactly nine action labels")
        context = multiprocessing.get_context("spawn")
        self._snapshots = context.Queue(maxsize=1)
        # A feeder blocked on a stopped UI must not delay interpreter shutdown.
        self._snapshots.cancel_join_thread()
        self._stopped = context.Event()
        self._failed = context.Event()
        self._closed = False
        self.dropped_snapshots = 0
        self._process = context.Process(
            target=_dashboard_main,
            args=(
                dict(action_labels),
                str(screenshot_path) if screenshot_path else None,
                self._snapshots,
                self._stopped,
                self._failed,
            ),
            name="doom-dashboard",
            daemon=True,
        )
        try:
            self._process.start()
        except Exception:
            self._snapshots.close()
            raise

    @property
    def stop_requested(self) -> bool:
        return self._closed or self._stopped.is_set() or not self._process.is_alive()

    @property
    def error(self) -> str | None:
        if self._failed.is_set():
            return "Dashboard UI failed during initialization, rendering, or screenshot output"
        exitcode = self._process.exitcode
        if not self._closed and exitcode not in (None, 0):
            return f"Dashboard process exited unexpectedly with code {exitcode}"
        return None

    def publish(self, snapshot: dict[str, Any]) -> None:
        if self.stop_requested:
            return
        snapshot = snapshot.copy()
        if isinstance(snapshot.get("probabilities"), dict):
            snapshot["probabilities"] = snapshot["probabilities"].copy()
        try:
            self._snapshots.put_nowait(snapshot)
        except queue.Full:
            self.dropped_snapshots += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stopped.set()
        self._process.join(timeout=0.75)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=0.5)
        self._snapshots.close()
