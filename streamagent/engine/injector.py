"""ObsInjector: Thread-safe queue for injecting observations into the token stream.

The ObsInjector bridges the sync Environment thread and async KVStream,
allowing observations to be queued non-blocking and drained per token step.
"""

import threading
from collections import deque
from typing import Deque

from streamagent.engine.interfaces import Observation, ObsInjectorProtocol


class ObsInjector(ObsInjectorProtocol):
    """Thread-safe observation queue for stream injection.

    Called from Environment thread via put(); drained by KVStream per token.
    """

    def __init__(self, maxsize: int = 256):
        """Initialize the observation injector.

        Args:
            maxsize: Maximum queue size. Oldest observations are dropped on overflow.
        """
        self.maxsize = maxsize
        self._queue: Deque[Observation] = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def put(self, obs: Observation) -> None:
        """Non-blocking enqueue from Environment thread.

        If the queue is full, the oldest observation is silently dropped.

        Args:
            obs: The observation to enqueue.
        """
        with self._lock:
            self._queue.append(obs)

    def get_pending(self) -> list[Observation]:
        """Drain all pending observations.

        Called by KVStream per token step (sync context). Returns all currently
        queued observations and clears the queue.

        Returns:
            List of all pending observations (empty list if queue is empty).
        """
        with self._lock:
            result = list(self._queue)
            self._queue.clear()
            return result

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if no pending observations, False otherwise.
        """
        with self._lock:
            return len(self._queue) == 0
