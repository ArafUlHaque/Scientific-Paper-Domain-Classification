"""Bound inference work across all sessions in one Streamlit process."""
import math
import threading
import time
from collections import deque
from contextlib import contextmanager

SESSION_COOLDOWN_SECONDS = 10
GLOBAL_REQUESTS_PER_MINUTE = 12


class RequestRejected(Exception):
    """A safe, visitor-facing admission rejection."""


class RequestGate:
    """One active request, no inference queue, and bounded rolling-window state.

    This gate must be shared with st.cache_resource. Session timestamps are kept
    in session_state, so the gate never retains abstracts or visitor identifiers.
    Limits reset on process restart and are not a connection-level DDoS defense.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._active = False
        self._accepted = deque()

    @contextmanager
    def admit(self, last_finished=None):
        with self._lock:
            now = self._clock()
            if last_finished is not None:
                remaining = SESSION_COOLDOWN_SECONDS - (now - last_finished)
                if remaining > 0:
                    raise RequestRejected(f"Please wait {math.ceil(remaining)} seconds before classifying again.")
            if self._active:
                raise RequestRejected("The classifier is busy. Please try again shortly.")
            while self._accepted and self._accepted[0] <= now - 60:
                self._accepted.popleft()
            if len(self._accepted) >= GLOBAL_REQUESTS_PER_MINUTE:
                remaining = math.ceil(60 - (now - self._accepted[0]))
                raise RequestRejected(f"The demo has reached its request limit. Please try again in {remaining} seconds.")
            self._accepted.append(now)
            self._active = True
        try:
            yield
        finally:
            with self._lock:
                self._active = False
