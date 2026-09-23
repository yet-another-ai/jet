from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import play
from jet_client import DecisionResult
from PIL import Image


class ControlsTests(unittest.TestCase):
    def test_slow_inference_does_not_extend_a_button_lease(self):
        controls = play.Controls()
        frame = play.PendingFrame("one", 1, 10.0, 35)
        self.assertEqual(
            controls.apply(frame, "forward", 11.0, 3), play.ACTIONS["forward"].buttons
        )
        self.assertIsNone(controls.release_due(11.49))
        self.assertEqual(controls.release_due(11.5), play.RELEASE)
        self.assertIsNone(controls.release_due(12.0))

    def test_previous_episode_and_expired_frames_cannot_press_buttons(self):
        controls = play.Controls()
        frame = play.PendingFrame("old", 1, 10.0, 35)
        self.assertEqual(controls.reset(), play.RELEASE)
        self.assertIsNone(controls.apply(frame, "fire", 10.5, 3))
        fresh_episode = play.PendingFrame("stale", 2, 10.0, 35)
        self.assertIsNone(controls.apply(fresh_episode, "fire", 13.1, 3))
        self.assertIsNone(controls.deadline)

    def test_turn_is_bounded_and_wait_releases_previous_input(self):
        controls = play.Controls()
        frame = play.PendingFrame("turn", 1, 10.0, 35)
        controls.apply(frame, "turn_left", 10.1, 3)
        self.assertAlmostEqual(controls.deadline, 10.35)
        self.assertEqual(controls.apply(frame, "wait", 10.2, 3), play.RELEASE)
        with self.assertRaisesRegex(ValueError, "unknown action"):
            controls.apply(frame, "jump_to_exit", 10.2, 3)

    def test_request_contains_exactly_160_by_100_rgb_image(self):
        png = play.encode_frame(Image.new("RGBA", (1280, 720), (255, 0, 0, 255)))
        request = play.decision_request("frame", png, {})
        source = request["images"][0]["source"]
        with Image.open(io.BytesIO(base64.b64decode(source["data"]))) as decoded:
            self.assertEqual(decoded.size, (160, 100))
            self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(
            set(request["questions"]["action"]["criteria"]), set(play.ACTIONS)
        )

    def test_malformed_warmup_and_decisions_are_rejected(self):
        for response in [
            {},
            {"answers": []},
            {"answers": {"action": {"type": "score"}}},
            {"answers": {"action": {"type": "choice", "choice": []}}},
        ]:
            with self.subTest(response=response), self.assertRaises(ValueError):
                play.selected_action(response)

    def test_visual_progress_counts_unchanged_movement_and_resets(self):
        progress = play.VisualProgress()
        dark = play.encode_frame(Image.new("RGB", play.FRAME_SIZE, "black"))
        light = play.encode_frame(Image.new("RGB", play.FRAME_SIZE, "white"))
        self.assertEqual(progress.observe(dark)["unchanged_moves"], 0)
        progress.begin(dark, "forward")
        self.assertEqual(progress.observe(dark)["unchanged_moves"], 1)
        progress.begin(dark, "strafe_left")
        self.assertEqual(progress.observe(dark)["unchanged_moves"], 2)
        progress.begin(dark, "forward")
        self.assertEqual(progress.observe(light)["unchanged_moves"], 0)
        progress.begin(light, "backward")
        self.assertEqual(progress.observe(light)["unchanged_moves"], 1)
        progress.begin(light, "fire")
        self.assertEqual(progress.observe(light)["unchanged_moves"], 0)

    def test_use_feedback_compares_execution_baseline_and_counts_only_once(self):
        progress = play.VisualProgress()
        old_request = play.encode_frame(Image.new("RGB", play.FRAME_SIZE, "black"))
        live_before_action = play.encode_frame(
            Image.new("RGB", play.FRAME_SIZE, "white")
        )
        progress.observe(old_request)
        progress.begin(live_before_action, "use")
        self.assertEqual(progress.observe(live_before_action)["unchanged_uses"], 1)
        self.assertEqual(progress.observe(live_before_action)["unchanged_uses"], 1)
        progress.begin(live_before_action, "use")
        self.assertEqual(progress.observe(live_before_action)["unchanged_uses"], 2)
        progress.begin(live_before_action, "use")
        self.assertEqual(progress.observe(old_request)["unchanged_uses"], 0)

    def test_use_approaches_then_presses_use_and_observes_after_settling(self):
        controls = play.Controls()
        frame = play.PendingFrame("door", 1, 10.0, 35)
        self.assertEqual(
            controls.apply(frame, "use", 10.0, 3), play.ACTIONS["forward"].buttons
        )
        self.assertFalse(controls.ready_for_observation(10.49))
        self.assertIsNone(controls.release_due(10.49))
        # A late tick must still send USE rather than skip it entirely.
        self.assertEqual(controls.release_due(10.7), (0, 0, 0, 0, 0, 0, 1))
        self.assertFalse(controls.ready_for_observation(10.79))
        self.assertIsNone(controls.release_due(10.79))
        self.assertEqual(controls.release_due(10.81), play.RELEASE)
        self.assertFalse(controls.ready_for_observation(10.9))
        self.assertTrue(controls.ready_for_observation(10.97))

    def test_reset_cancels_macro_and_settling_before_new_episode(self):
        controls = play.Controls()
        frame = play.PendingFrame("door", 1, 10.0, 35)
        controls.apply(frame, "use", 10.0, 3)
        self.assertEqual(controls.reset(), play.RELEASE)
        self.assertIsNone(controls.release_due(10.5))
        self.assertTrue(controls.ready_for_observation(10.5))
        self.assertEqual(controls.remaining_phases, [])
        self.assertIsNone(controls.apply(frame, "use", 10.5, 3))


class MainLoopTests(unittest.TestCase):
    def args(self, output: Path) -> argparse.Namespace:
        return argparse.Namespace(
            output_dir=output,
            save_frames=False,
            iwad=None,
            map="map01",
            skill=1,
            seed=7,
            decision_hz=2.0,
            max_frame_age=3.0,
            request_timeout=10.0,
            headless=True,
            duration=2.0,
            episodes=0,
        )

    def test_closing_dashboard_releases_controls_and_closes_every_worker(self):
        client = Mock(busy=False)
        client.poll.side_effect = [
            DecisionResult(
                request_id,
                {"answers": {"action": {"type": "choice", "choice": choice}}},
                0.1,
            )
            if request_id
            else None
            for request_id, choice in [
                ("warmup", "wait"),
                (None, None),
                ("episode-1-frame-1", "forward"),
            ]
        ]
        game = Mock()
        game.is_running.return_value = True
        game.is_player_dead.return_value = False
        game.is_episode_finished.return_value = False
        game.get_state.return_value = SimpleNamespace(
            tic=10,
            screen_buffer=Mock(shape=(400, 640, 3)),
            game_variables=[100, 50],
        )
        dashboard = Mock(error=None)
        type(dashboard).stop_requested = PropertyMock(side_effect=[False, False, True])
        diagnostics = Mock(dropped_events=0, error=None)
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory) / "run")
            args.headless = False
            with (
                patch.object(play, "jet_command", return_value=["fake-jet"]),
                patch.object(play, "JetClient", return_value=client),
                patch.object(play, "configure_game", return_value=game),
                patch.object(play, "Dashboard", return_value=dashboard),
                patch.object(play, "RunOutput", return_value=diagnostics),
                patch.object(play.time, "perf_counter", return_value=100.0),
                patch.object(
                    play.Image, "fromarray", return_value=Image.new("RGB", (160, 100))
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                play.run(args)
        self.assertIn(
            list(play.ACTIONS["forward"].buttons),
            [call.args[0] for call in game.set_action.call_args_list],
        )
        self.assertEqual(game.set_action.call_args.args[0], list(play.RELEASE))
        game.close.assert_called_once()
        client.close.assert_called_once()
        dashboard.close.assert_called_once()
        diagnostics.close.assert_called_once()
        self.assertIsNone(dashboard.publish.call_args.args[0]["probabilities"])

    def test_dashboard_failure_marks_run_failed_and_still_releases_controls(self):
        client = Mock()
        client.poll.return_value = DecisionResult(
            "warmup", {"answers": {"action": {"type": "choice", "choice": "wait"}}}, 0.1
        )
        game = Mock()
        game.is_running.return_value = True
        dashboard = Mock(stop_requested=True, error="Dashboard process failed")
        diagnostics = Mock(dropped_events=0, error=None)
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory) / "run")
            args.headless = False
            with (
                patch.object(play, "jet_command", return_value=["fake-jet"]),
                patch.object(play, "JetClient", return_value=client),
                patch.object(play, "configure_game", return_value=game),
                patch.object(play, "Dashboard", return_value=dashboard),
                patch.object(play, "RunOutput", return_value=diagnostics),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "Dashboard process failed"),
            ):
                play.run(args)
            summary = json.loads((args.output_dir / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(game.set_action.call_args.args[0], list(play.RELEASE))
        game.close.assert_called_once()
        client.close.assert_called_once()
        dashboard.close.assert_called_once()
        diagnostics.close.assert_called_once()

    def test_episode_reset_drains_old_request_before_submitting_fresh_frame(self):
        test = self
        clock = SimpleNamespace(tic=0)

        def now():
            return 100.0 + clock.tic / 35

        class FakeGame:
            episode = 1
            episode_tic = 0
            initialized = False
            closed = False

            def __init__(self):
                self.actions = []
                self.restarts = []
                self.busy_tics = []

            def init(self):
                self.initialized = True

            def is_running(self):
                return self.initialized and not self.closed

            def is_episode_finished(self):
                return self.is_player_dead()

            def is_player_dead(self):
                return self.episode == 1 and self.episode_tic >= 3

            def new_episode(self):
                self.restarts.append((clock.tic, client.busy))
                self.episode += 1
                self.episode_tic = 0

            def set_action(self, buttons):
                self.actions.append((self.episode, clock.tic, tuple(buttons)))

            def advance_action(self, tics, update_state):
                test.assertEqual((tics, update_state), (1, True))
                clock.tic += 1
                self.episode_tic += 1
                test.assertLess(clock.tic, 100, "main loop exceeded its duration")
                if client.busy:
                    self.busy_tics.append((self.episode, clock.tic))

            def get_state(self):
                return SimpleNamespace(
                    tic=self.episode_tic,
                    screen_buffer=object(),
                    game_variables=[100, 50],
                )

            def close(self):
                self.closed = True

        game = FakeGame()

        class FakeClient:
            pending = None
            closed = False

            def __init__(self):
                self.submissions = []
                self.completed = []

            @property
            def busy(self):
                return self.pending is not None

            def submit(self, request):
                test.assertIsNone(self.pending, "submitted a second in-flight request")
                request_id = request["request_id"]
                delay = 0 if request_id == "warmup" else 25 if game.episode == 1 else 2
                self.pending = (request_id, clock.tic + delay, now())
                if request_id != "warmup":
                    self.submissions.append((request_id, now()))

            def poll(self):
                if self.pending is None or clock.tic < self.pending[1]:
                    return None
                request_id, _, submitted_at = self.pending
                self.pending = None
                choice = (
                    "wait"
                    if request_id == "warmup"
                    else "fire"
                    if request_id.startswith("episode-1-")
                    else "forward"
                )
                self.completed.append((request_id, game.episode, clock.tic))
                return DecisionResult(
                    request_id,
                    {
                        "request_id": request_id,
                        "answers": {"action": {"type": "choice", "choice": choice}},
                    },
                    now() - submitted_at,
                )

            def close(self):
                self.closed = True

        client = FakeClient()
        diagnostics = Mock(dropped_events=0, error=None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.object(play, "jet_command", return_value=["fake-jet"]),
                patch.object(play, "JetClient", return_value=client),
                patch.object(play, "configure_game", return_value=game),
                patch.object(play, "RunOutput", return_value=diagnostics),
                patch.object(play.time, "perf_counter", side_effect=now),
                patch.object(
                    play.Image, "fromarray", return_value=Image.new("RGB", (160, 100))
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                play.run(self.args(output))
            summary = json.loads((output / "summary.json").read_text())

        self.assertEqual(game.restarts, [(3, True)])
        old_completion = next(
            result for result in client.completed if result[0] == "episode-1-frame-1"
        )
        self.assertEqual(old_completion[1], 2)
        self.assertTrue(client.submissions[1][0].startswith("episode-2-"))
        self.assertGreaterEqual(
            client.submissions[1][1], 100.0 + old_completion[2] / 35
        )
        self.assertGreaterEqual(sum(episode == 2 for episode, _ in game.busy_tics), 20)
        records = [call.args[0] for call in diagnostics.record.call_args_list]
        self.assertEqual(records[0]["request_id"], "episode-1-frame-1")
        self.assertFalse(records[0]["applied"])
        self.assertTrue(
            any(record["episode"] == 2 and record["applied"] for record in records)
        )
        pressed = [
            (episode, buttons)
            for episode, _, buttons in game.actions
            if buttons != play.RELEASE
        ]
        self.assertTrue(pressed)
        self.assertTrue(
            all(
                episode == 2 and buttons == play.ACTIONS["forward"].buttons
                for episode, buttons in pressed
            )
        )
        self.assertGreaterEqual(len(client.submissions), 3)
        for previous, current in zip(client.submissions, client.submissions[1:]):
            self.assertGreaterEqual(current[1] - previous[1], 0.5 - 1e-9)
        # A fast model cannot trigger another screenshot during forward motion.
        first_fresh_completion = next(
            item for item in client.completed if item[0].startswith("episode-2-")
        )
        self.assertGreaterEqual(
            client.submissions[2][1] - (100.0 + first_fresh_completion[2] / 35),
            play.ACTIONS["forward"].duration + play.OBSERVATION_SETTLE_SECONDS,
        )
        self.assertEqual(summary["deaths"], 1)
        self.assertEqual(summary["episodes_started"], 2)
        self.assertEqual(summary["stale_decisions_discarded"], 1)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(game.closed)
        self.assertTrue(client.closed)
        diagnostics.close.assert_called_once()

    def test_malformed_warmup_fails_before_configuring_game_and_closes_client(self):
        client = Mock()
        client.poll.return_value = DecisionResult(
            "warmup", {"request_id": "warmup"}, 0.01
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with (
                patch.object(play, "jet_command", return_value=["fake-jet"]),
                patch.object(play, "JetClient", return_value=client),
                patch.object(play, "configure_game") as configure_game,
                patch.object(play, "RunOutput") as run_output,
                patch.object(play.time, "perf_counter", return_value=100.0),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "action answer of type choice"),
            ):
                play.run(self.args(output))
            summary = json.loads((output / "summary.json").read_text())

        self.assertEqual(client.submit.call_args.args[0]["request_id"], "warmup")
        configure_game.assert_not_called()
        run_output.assert_not_called()
        client.close.assert_called_once()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["episodes_started"], 0)
        self.assertEqual(summary["frames_submitted"], 0)


class DashboardStateTests(unittest.TestCase):
    def test_last_scores_survive_pending_inference_but_not_episode_reset(self):
        display = play.DashboardState()
        frame = SimpleNamespace(
            tic=70,
            screen_buffer=SimpleNamespace(
                shape=(100, 160, 3), tobytes=lambda: bytes(160 * 100 * 3)
            ),
            game_variables=[90, 40],
        )
        initial = display.snapshot(frame, thinking=True)
        self.assertIsNone(initial["probabilities"])
        self.assertIsNone(initial["choice"])
        record = {
            "episode": 1,
            "request_id": "episode-1-frame-1",
            "game_tic": 35,
            "choice": "fire",
            "probabilities": {"fire": 0.8, "forward": 0.2},
            "latency_seconds": 1.0,
            "applied": True,
        }
        display.complete(record)
        pending = display.snapshot(frame, thinking=True)
        self.assertEqual(pending["probabilities"], record["probabilities"])
        self.assertEqual(pending["choice"], "fire")
        self.assertEqual(pending["frame_id"], "episode-1-frame-1")
        self.assertEqual((pending["game_tic"], pending["decision_game_tic"]), (70, 35))
        self.assertEqual(pending["frame_size"], (160, 100))
        self.assertEqual(len(pending["frame_rgb"]), 160 * 100 * 3)

        display.reset(2)
        display.complete(record)
        restarted = display.snapshot(frame, thinking=True)
        self.assertEqual(restarted["episode"], 2)
        self.assertIsNone(restarted["probabilities"])
        self.assertIsNone(restarted["choice"])

        display.complete({**record, "episode": 2, "applied": False})
        expired = display.snapshot(frame, thinking=False)
        self.assertEqual(expired["probabilities"], record["probabilities"])
        self.assertFalse(expired["applied"])


@unittest.skipUnless(
    os.environ.get("JET_DOOM_ENGINE_SMOKE") == "1", "requires real Doom engine"
)
class DoomEngineSmokeTests(unittest.TestCase):
    def test_engine_advances_while_slow_model_is_busy(self):
        fake_jet = """
import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request['request_id'] != 'warmup':
        time.sleep(0.65)
    print(json.dumps({'request_id': request['request_id'], 'answers': {
        'action': {'type': 'choice', 'choice': 'forward'}}, 'usage': {}}), flush=True)
"""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = argparse.Namespace(
                output_dir=output,
                save_frames=True,
                iwad=None,
                map="map01",
                skill=1,
                seed=7,
                decision_hz=2.0,
                max_frame_age=3.0,
                request_timeout=10.0,
                headless=True,
                duration=2.5,
                episodes=1,
            )
            with (
                patch.object(
                    play,
                    "jet_command",
                    return_value=[sys.executable, "-u", "-c", fake_jet],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                play.run(args)
            summary = json.loads((output / "summary.json").read_text())
            records = [
                json.loads(line)
                for line in (output / "decisions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(summary["status"], "completed")
            self.assertGreaterEqual(summary["decisions_applied"], 2)
            self.assertGreater(records[1]["game_tic"] - records[0]["game_tic"], 15)
            self.assertGreaterEqual(records[0]["latency_seconds"], 0.6)
            with Image.open(output / "last-frame.png") as frame:
                self.assertEqual(frame.size, (160, 100))


if __name__ == "__main__":
    unittest.main()
