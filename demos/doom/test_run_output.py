"""The diagnostics queue must not delay the game while storage/console is slow."""

import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

if __package__:
    from .run_output import RunOutput
else:
    from run_output import RunOutput


class RunOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output_dir = Path(directory.name) / "run"

    def test_normal_output_is_flushed_and_frames_are_saved(self) -> None:
        record = {
            "request_id": "frame-1",
            "choice": "left",
            "latency_seconds": 0.125,
            "applied": True,
            "note": "中文",
        }
        console = io.StringIO()
        with (
            redirect_stdout(console),
            RunOutput(self.output_dir, save_frames=True) as output,
        ):
            output.record(record)
            output.frame(1, b"first png")
            output.frame(2, b"second png")
            output.message("ready")
        self.assertIsNone(output.error)
        self.assertEqual(output.dropped_events, 0)
        self.assertEqual(
            json.loads((self.output_dir / "decisions.jsonl").read_text("utf-8")), record
        )
        self.assertEqual(
            (self.output_dir / "last-frame.png").read_bytes(), b"second png"
        )
        self.assertEqual(
            (self.output_dir / "frames/000001.png").read_bytes(), b"first png"
        )
        self.assertEqual(
            (self.output_dir / "frames/000002.png").read_bytes(), b"second png"
        )
        self.assertIn("frame-1: left 0.12s applied", console.getvalue())
        self.assertIn("ready", console.getvalue())
        self.assertFalse(output._worker.is_alive())
        output.close()

    def test_frames_are_not_archived_unless_requested(self) -> None:
        with RunOutput(self.output_dir) as output:
            output.frame(1, b"png")
        self.assertEqual((self.output_dir / "last-frame.png").read_bytes(), b"png")
        self.assertFalse((self.output_dir / "frames").exists())
        self.assertIsNone(output.error)

    def test_slow_worker_never_blocks_enqueue_and_queue_drops_at_capacity(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def slow_handle(*_: object) -> None:
            entered.set()
            release.wait(timeout=3)

        with patch.object(RunOutput, "_handle", slow_handle):
            output = RunOutput(self.output_dir, capacity=2)
            try:
                output.message("block the worker")
                self.assertTrue(entered.wait(timeout=1))
                started = time.perf_counter()
                output.record({"request_id": "one"})
                output.frame(2, b"png")
                output.message("dropped 1")
                output.record({"request_id": "dropped 2"})
                output.frame(3, b"dropped 3")
                self.assertLess(time.perf_counter() - started, 0.1)
                self.assertEqual(output.dropped_events, 3)
                self.assertEqual(output._events.qsize(), 2)
                started = time.perf_counter()
                output.close(timeout_seconds=0.02)
                self.assertLess(time.perf_counter() - started, 0.3)
                self.assertEqual(output.dropped_events, 5)
                self.assertIn("close deadline", output.error or "")
                self.assertTrue(output._worker.daemon)
                output.message("after close")
                self.assertEqual(output.dropped_events, 6)
            finally:
                release.set()
                output.close(timeout_seconds=1)
        self.assertFalse(output._worker.is_alive())

    def test_console_block_does_not_prevent_log_flush_or_bounded_close(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def slow_print(*_: object, **__: object) -> None:
            entered.set()
            release.wait(timeout=3)

        with patch("builtins.print", slow_print):
            output = RunOutput(self.output_dir)
            try:
                output.record({"request_id": "flushed", "choice": "wait"})
                self.assertTrue(entered.wait(timeout=1))
                self.assertIn(
                    "flushed", (self.output_dir / "decisions.jsonl").read_text("utf-8")
                )
                started = time.perf_counter()
                output.frame(1, b"queued")
                output.close(timeout_seconds=0.02)
                self.assertLess(time.perf_counter() - started, 0.3)
                self.assertEqual(output.dropped_events, 1)
            finally:
                release.set()
                output.close(timeout_seconds=1)

    def test_background_write_error_is_exposed_and_later_events_are_dropped(
        self,
    ) -> None:
        with patch.object(Path, "write_bytes", side_effect=OSError("disk full")):
            output = RunOutput(self.output_dir)
            output.frame(1, b"png")
            output.message("never written")
            output.close(timeout_seconds=1)
        self.assertIn("disk full", output.error or "")
        self.assertEqual(output.dropped_events, 2)
        output.record({"request_id": "after failure"})
        self.assertEqual(output.dropped_events, 3)
        self.assertFalse(output._worker.is_alive())

    def test_invalid_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RunOutput(self.output_dir, capacity=0)


if __name__ == "__main__":
    unittest.main()
