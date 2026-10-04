"""Admission checks use a controlled clock and actual competing threads."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from demo.request_limits import (
    GLOBAL_REQUESTS_PER_MINUTE, RequestGate, RequestRejected,
)


class RequestLimitTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.gate = RequestGate(clock=lambda: self.now)

    def test_busy_request_never_enters_work_and_errors_release_slot(self):
        with self.assertRaisesRegex(ValueError, "inference failed"):
            with self.gate.admit():
                with self.assertRaisesRegex(RequestRejected, "busy"):
                    with self.gate.admit():
                        self.fail("A busy request must not run model loading or inference")
                raise ValueError("inference failed")
        with self.gate.admit():
            pass

    def test_cooldown_expires_and_rejections_do_not_consume_global_budget(self):
        for _ in range(GLOBAL_REQUESTS_PER_MINUTE + 1):
            with self.assertRaisesRegex(RequestRejected, "wait 10 seconds"):
                with self.gate.admit(last_finished=self.now):
                    self.fail("Cooldown must reject the request")
        self.now += 10
        with self.gate.admit(last_finished=100.0):
            pass

    def test_new_sessions_share_the_global_limit_and_window_expires(self):
        for _ in range(GLOBAL_REQUESTS_PER_MINUTE):
            # None simulates a fresh session, with no per-session cooldown.
            with self.gate.admit(last_finished=None):
                pass
        for _ in range(50):
            with self.assertRaisesRegex(RequestRejected, "request limit"):
                with self.gate.admit():
                    self.fail("New sessions must not bypass the global cap")
        self.assertEqual(len(self.gate._accepted), GLOBAL_REQUESTS_PER_MINUTE)
        self.now += 60
        with self.gate.admit():
            pass
        self.assertEqual(len(self.gate._accepted), 1)

    def test_competing_threads_cannot_enter_while_one_request_is_running(self):
        started = threading.Event()
        finish = threading.Event()

        def active_request():
            with self.gate.admit():
                started.set()
                if not finish.wait(timeout=5):
                    raise AssertionError("Test request was not released")

        def competing_request():
            try:
                with self.gate.admit():
                    return "accepted"
            except RequestRejected:
                return "rejected"

        with ThreadPoolExecutor(max_workers=5) as executor:
            active = executor.submit(active_request)
            try:
                self.assertTrue(started.wait(timeout=5))
                results = list(executor.map(lambda _: competing_request(), range(20)))
                self.assertEqual(results, ["rejected"] * 20)
            finally:
                finish.set()
            active.result(timeout=5)
        with self.gate.admit():
            pass


if __name__ == "__main__":
    unittest.main()
