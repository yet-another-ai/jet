"""Exercise the real subprocess/pipe transport without a model or ViZDoom."""

import sys
import tempfile
import time
import unittest
from pathlib import Path

if __package__:
    from .jet_client import DecisionResult, JetClient, JetClientError
else:
    from jet_client import DecisionResult, JetClient, JetClientError


class JetClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.stderr_path = Path(self.directory.name) / "jet.stderr.log"

    def client(self, script: str, timeout: float = 3) -> JetClient:
        client = JetClient(
            [sys.executable, "-u", "-c", script], self.stderr_path, timeout
        )
        self.addCleanup(client.close)
        return client

    def result(self, client: JetClient, timeout: float = 5) -> DecisionResult:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = client.poll()
            if result is not None:
                return result
            time.sleep(0.005)
        self.fail("client did not return a result or error")

    def test_response_arrives_without_eof_and_process_handles_another_request(
        self,
    ) -> None:
        client = self.client("""
import json, sys
sys.stderr.write('native diagnostics\\n' * 10000)
sys.stderr.flush()
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({'request_id': request['request_id'], 'answers': {'action': {'choice': 'left'}}}), flush=True)
""")
        self.assertIsNone(client.poll())
        for request_id in ("frame-1", "frame-2"):
            client.submit({"request_id": request_id, "image": "fake-base64"})
            self.assertTrue(client.busy)
            result = self.result(client)
            self.assertEqual(result.request_id, request_id)
            self.assertEqual(result.response["answers"]["action"]["choice"], "left")
            self.assertGreater(result.latency_seconds, 0)
            self.assertFalse(client.busy)
            self.assertIsNone(client._process.poll())
        self.assertGreater(self.stderr_path.stat().st_size, 65536)

    def test_submit_and_poll_do_not_wait_and_reject_concurrent_requests(self) -> None:
        client = self.client("""
import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    time.sleep(0.4)
    print(json.dumps({'request_id': request['request_id']}), flush=True)
""")
        before = time.monotonic()
        client.submit({"request_id": "first"})
        self.assertIsNone(client.poll())
        self.assertLess(time.monotonic() - before, 0.2)
        with self.assertRaisesRegex(JetClientError, "already in flight"):
            client.submit({"request_id": "second"})
        self.result(client)

    def test_uncollected_result_is_still_busy(self) -> None:
        client = self.client("""
import json, sys
for line in sys.stdin:
    print(json.dumps({'request_id': json.loads(line)['request_id']}), flush=True)
""")
        client.submit({"request_id": "first"})
        deadline = time.monotonic() + 3
        while client._completions.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(client._completions.empty())
        self.assertTrue(client.busy)
        with self.assertRaisesRegex(JetClientError, "already in flight"):
            client.submit({"request_id": "second"})
        self.result(client)

    def test_server_error_is_clear_and_next_request_can_succeed(self) -> None:
        client = self.client("""
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    response = {'request_id': request['request_id']}
    if request['request_id'] == 'bad':
        response['error'] = {'code': 'invalid_image', 'message': 'PNG is corrupt'}
    print(json.dumps(response), flush=True)
""")
        client.submit({"request_id": "bad"})
        with self.assertRaisesRegex(JetClientError, r"invalid_image.*PNG is corrupt"):
            self.result(client)
        self.assertFalse(client.busy)
        client.submit({"request_id": "good"})
        self.assertEqual(self.result(client).request_id, "good")

    def test_protocol_errors_end_the_client(self) -> None:
        cases = [
            ("[]", "must be a JSON object"),
            ("{not-json}", "invalid JSON"),
            ('{"request_id":"other"}', "does not match"),
            ('{"request_id":"frame","error":"bad"}', "malformed error"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                client = self.client(
                    "import sys, time\nsys.stdin.readline()\n"
                    f"print({response!r}, flush=True)\ntime.sleep(10)"
                )
                client.submit({"request_id": "frame"})
                with self.assertRaisesRegex(JetClientError, expected):
                    self.result(client)
                with self.assertRaisesRegex(JetClientError, "unavailable"):
                    client.submit({"request_id": "next"})
                client.close()

    def test_eof_and_process_exit_are_reported(self) -> None:
        client = self.client("import sys\nsys.stdin.readline()\nsys.exit(7)")
        client.submit({"request_id": "frame"})
        with self.assertRaisesRegex(JetClientError, "closed stdout|exited"):
            self.result(client)
        self.assertFalse(client.busy)

    def test_idle_process_exit_is_reported(self) -> None:
        client = self.client("raise SystemExit(9)")
        with self.assertRaisesRegex(JetClientError, "exited with code 9"):
            self.result(client)

    def test_timeout_kills_child_and_releases_reader_without_close(self) -> None:
        client = self.client(
            "import sys, time\nsys.stdin.readline()\ntime.sleep(30)", timeout=0.2
        )
        client.submit({"request_id": "slow"})
        with self.assertRaisesRegex(JetClientError, "slow.*timed out"):
            self.result(client)
        client._worker.join(timeout=1)
        self.assertFalse(client._worker.is_alive())
        self.assertIsNotNone(client._process.poll())

    def test_timeout_also_releases_a_blocked_stdin_write(self) -> None:
        client = self.client("import time\ntime.sleep(30)", timeout=0.2)
        client.submit({"request_id": "large", "data": "x" * 2_000_000})
        with self.assertRaisesRegex(JetClientError, "large.*timed out"):
            self.result(client)
        client._worker.join(timeout=1)
        self.assertFalse(client._worker.is_alive())

    def test_close_interrupts_pending_request_and_is_idempotent(self) -> None:
        client = self.client("import sys, time\nsys.stdin.readline()\ntime.sleep(30)")
        client.submit({"request_id": "frame"})
        time.sleep(0.05)
        before = time.monotonic()
        client.close()
        self.assertLess(time.monotonic() - before, 2.5)
        client.close()
        self.assertFalse(client.busy)
        self.assertFalse(client._worker.is_alive())
        self.assertIsNotNone(client._process.poll())
        with self.assertRaisesRegex(JetClientError, "closed"):
            client.submit({"request_id": "next"})

    def test_bad_request_validation_and_worker_json_error(self) -> None:
        client = self.client("import time\ntime.sleep(30)")
        for request in ({}, {"request_id": ""}, {"request_id": 1}):
            with self.assertRaises(ValueError):
                client.submit(request)
        client.submit({"request_id": "bad", "data": object()})
        with self.assertRaisesRegex(JetClientError, "not valid JSON"):
            self.result(client)
        self.assertFalse(client.busy)


if __name__ == "__main__":
    unittest.main()
