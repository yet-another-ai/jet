"""Offscreen rendering and real spawn-process transport, without Doom or a model."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__:
    from . import dashboard
else:
    import dashboard


LABELS = {
    "forward": "Move forward",
    "backward": "Move backward",
    "strafe_left": "Strafe left",
    "strafe_right": "Strafe right",
    "turn_left": "Turn left",
    "turn_right": "Turn right",
    "fire": "Fire weapon",
    "use": "Use door",
    "wait": "Stay still",
}


def _paused_ui(labels, screenshot_path, snapshots, stopped, failed) -> None:
    stopped.wait(timeout=10)


def _hung_ui(labels, screenshot_path, snapshots, stopped, failed) -> None:
    time.sleep(10)


def _escape_ui(labels, screenshot_path, snapshots, stopped, failed) -> None:
    import pygame

    pygame.display.init()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    dashboard._dashboard_main(labels, screenshot_path, snapshots, stopped, failed)


def _failed_but_running_ui(labels, screenshot_path, snapshots, stopped, failed) -> None:
    failed.set()
    stopped.set()
    time.sleep(10)


def sample_snapshot() -> dict:
    return {
        "frame_rgb": bytes((130, 40, 20)) * (160 * 100),
        "frame_size": (160, 100),
        "episode": 3,
        "frame_id": "episode-3-frame-42",
        "choice": "forward",
        "probabilities": {
            key: 0.6 if index == 0 else 0.05 for index, key in enumerate(LABELS)
        },
        "applied": True,
        "latency_seconds": 1.23,
        "thinking": True,
        "health": 85,
        "ammo": 20,
    }


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = patch.dict(
            os.environ, {"SDL_VIDEODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
        )
        cls.environment.start()
        import pygame

        cls.pygame = pygame
        pygame.font.init()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pygame.quit()
        dashboard._font.cache_clear()
        cls.environment.stop()

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_missing_and_invalid_probabilities_are_not_zero(self) -> None:
        for invalid in (
            None,
            float("nan"),
            float("inf"),
            -0.1,
            1.1,
            10**1000,
            "0.5",
            True,
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(dashboard.probability_text(invalid), "--")
        self.assertEqual(dashboard.probability_text(0), "0.0%")
        self.assertEqual(dashboard.probability_text(0.125), "12.5%")
        self.assertEqual(dashboard.probability_text(1), "100.0%")

    def test_render_frame_and_highlight_applied_or_stale_selection(self) -> None:
        surface = self.pygame.Surface(dashboard.WINDOW_SIZE)
        snapshot = sample_snapshot()
        dashboard.render_snapshot(surface, snapshot, LABELS)
        self.assertEqual(tuple(surface.get_at((300, 300)))[:3], (130, 40, 20))
        self.assertEqual(tuple(surface.get_at((693, 180)))[:3], dashboard.TEAL)
        snapshot["applied"] = False
        dashboard.render_snapshot(surface, snapshot, LABELS)
        self.assertEqual(tuple(surface.get_at((693, 180)))[:3], dashboard.AMBER)
        snapshot.update(frame_id=None, choice=None, probabilities=None)
        dashboard.render_snapshot(surface, snapshot, LABELS)
        self.assertEqual(tuple(surface.get_at((693, 180)))[:3], dashboard.PANEL)

    def test_action_phases_only_change_status_and_preserve_last_scores(self) -> None:
        surface = self.pygame.Surface(dashboard.WINDOW_SIZE)
        snapshot = sample_snapshot()
        probabilities = snapshot["probabilities"].copy()
        dashboard.render_snapshot(surface, snapshot, LABELS)
        panel_rect = (688, 97, 408, 571)
        original_panel = self.pygame.image.tobytes(
            surface.subsurface(panel_rect), "RGB"
        )
        cases = (
            (
                {"thinking": True, "executing": True, "settling": True},
                "INFERENCE RUNNING",
            ),
            (
                {"thinking": False, "executing": True, "settling": True},
                "EXECUTING ACTION",
            ),
            (
                {"thinking": False, "executing": False, "settling": True},
                "OBSERVING RESULT",
            ),
            (
                {"thinking": False, "executing": False, "settling": False},
                "DECISION READY",
            ),
        )
        for flags, expected in cases:
            with self.subTest(status=expected):
                snapshot.update(flags)
                with patch.object(dashboard, "_text", wraps=dashboard._text) as text:
                    dashboard.render_snapshot(surface, snapshot, LABELS)
                rendered_text = [call.args[1] for call in text.call_args_list]
                self.assertIn(expected, rendered_text)
                self.assertEqual(
                    [value for value in rendered_text if value.endswith("%")],
                    [dashboard.probability_text(probabilities[key]) for key in LABELS],
                )
                self.assertEqual(
                    self.pygame.image.tobytes(surface.subsurface(panel_rect), "RGB"),
                    original_panel,
                )
                self.assertEqual(snapshot["probabilities"], probabilities)
        self.assertEqual(dashboard.snapshot_status({})[0], "WAITING FOR JET")
        self.assertEqual(
            dashboard.snapshot_status({"frame_id": "done"})[0], "DECISION READY"
        )
        self.assertLessEqual(dashboard._font(15, True).size(LABELS["use"])[0], 292)

    def test_render_preserves_frame_aspect_and_handles_missing_or_bad_pixels(
        self,
    ) -> None:
        surface = self.pygame.Surface(dashboard.WINDOW_SIZE)
        snapshot = sample_snapshot()
        snapshot.update(frame_rgb=bytes((200, 50, 20)) * 8, frame_size=(4, 2))
        dashboard.render_snapshot(surface, snapshot, LABELS)
        self.assertEqual(tuple(surface.get_at((300, 300)))[:3], (200, 50, 20))
        self.assertNotEqual(tuple(surface.get_at((300, 132)))[:3], (200, 50, 20))
        for pixels in (None, b"bad length"):
            snapshot["frame_rgb"] = pixels
            dashboard.render_snapshot(surface, snapshot, LABELS)
        self.pygame.image.save(surface, self.directory / "render.png")
        self.assertGreater((self.directory / "render.png").stat().st_size, 0)

    def test_real_ui_process_renders_screenshot_and_closes(self) -> None:
        screenshot = self.directory / "dashboard.png"
        ui = dashboard.Dashboard(LABELS, screenshot_path=screenshot)
        try:
            deadline = time.perf_counter() + 8
            while not screenshot.exists() and time.perf_counter() < deadline:
                ui.publish(sample_snapshot())
                self.assertFalse(ui.stop_requested)
                time.sleep(0.02)
            self.assertTrue(
                screenshot.exists(), "UI process did not render its snapshot"
            )
            # The child may still be finishing its first save; stop joins it.
        finally:
            ui.close()
        self.assertEqual(
            self.pygame.image.load(screenshot).get_size(), dashboard.WINDOW_SIZE
        )
        self.assertTrue(ui.stop_requested)
        self.assertFalse(ui._process.is_alive())
        ui.close()

    def test_saturated_real_queue_drops_without_waiting_for_ui(self) -> None:
        with patch.object(dashboard, "_dashboard_main", _paused_ui):
            ui = dashboard.Dashboard(LABELS)
        try:
            snapshot = sample_snapshot()
            started = time.perf_counter()
            for _ in range(1000):
                ui.publish(snapshot)
            self.assertLess(time.perf_counter() - started, 0.5)
            self.assertGreater(ui.dropped_snapshots, 0)
            self.assertFalse(ui.stop_requested)
        finally:
            ui.close()

    def test_child_failure_requests_game_stop(self) -> None:
        with patch.object(dashboard, "_dashboard_main", _paused_ui):
            ui = dashboard.Dashboard(LABELS)
        try:
            ui._process.terminate()
            ui._process.join(timeout=2)
            self.assertTrue(ui.stop_requested)
            self.assertFalse(ui._failed.is_set())
            self.assertIn("exited unexpectedly", ui.error or "")
        finally:
            ui.close()

    def test_real_ui_initialization_failure_is_not_a_normal_close(self) -> None:
        with patch.dict(
            os.environ, {"SDL_VIDEODRIVER": "jet-test-missing-video-driver"}
        ):
            ui = dashboard.Dashboard(LABELS)
        try:
            deadline = time.perf_counter() + 4
            while not ui.stop_requested and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertTrue(ui.stop_requested)
            self.assertTrue(ui._failed.is_set())
            self.assertIn("Dashboard UI failed", ui.error or "")
        finally:
            ui.close()
        self.assertIsNotNone(ui.error)

    def test_failure_is_visible_before_child_finishes_exiting(self) -> None:
        with patch.object(dashboard, "_dashboard_main", _failed_but_running_ui):
            ui = dashboard.Dashboard(LABELS)
        try:
            deadline = time.perf_counter() + 4
            while not ui.stop_requested and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertTrue(ui.stop_requested)
            self.assertIsNone(ui._process.exitcode)
            self.assertIn("Dashboard UI failed", ui.error or "")
        finally:
            ui.close()

    def test_escape_in_ui_process_requests_game_stop(self) -> None:
        with patch.object(dashboard, "_dashboard_main", _escape_ui):
            ui = dashboard.Dashboard(LABELS)
        try:
            deadline = time.perf_counter() + 4
            while not ui.stop_requested and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertTrue(ui.stop_requested)
            self.assertTrue(ui._stopped.is_set())
            self.assertIsNone(ui.error)
        finally:
            ui.close()
        self.assertIsNone(ui.error)

    def test_close_terminates_a_ui_that_ignores_stop(self) -> None:
        with patch.object(dashboard, "_dashboard_main", _hung_ui):
            ui = dashboard.Dashboard(LABELS)
        started = time.perf_counter()
        ui.close()
        self.assertLess(time.perf_counter() - started, 2.5)
        self.assertFalse(ui._process.is_alive())
        self.assertIsNone(ui.error)


if __name__ == "__main__":
    unittest.main()
