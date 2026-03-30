"""SinkCache: StreamingLLM attention-sink KV cache management.

Implements the attention-sink eviction strategy from Xiao et al. ICLR 2024
for fixed-capacity KV caches with three segments:

  1. Attention sinks (indices [0, n_sinks)): always kept, never evicted
  2. Pinned prefill tokens (indices [n_sinks, n_sinks + n_pinned)): always kept
  3. FIFO rolling window (indices [n_sinks + n_pinned, capacity)): evicts oldest

Reference: StreamingLLM: Efficient Streaming Language Models with
Attention Sinks (Xiao et al., ICLR 2024)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from streamagent.engine.interfaces import CacheStats


class SinkCache:
    """Fixed-capacity KV cache with attention-sink and pinned segments.

    Memory layout (within context window of size `capacity`):
        [0 .. n_sinks-1]              = attention sink tokens
        [n_sinks .. n_sinks+n_pinned-1] = pinned prefill tokens
        [n_sinks+n_pinned .. capacity)  = FIFO rolling window

    Tokens are added in order: first n_sinks slots, then n_pinned slots,
    then the rolling window. Only the rolling window evicts on overflow.
    """

    def __init__(
        self,
        capacity: int,
        n_sinks: int = 4,
        n_pinned: int = 0,
    ) -> None:
        """Initialize SinkCache.

        Args:
            capacity: Maximum number of tokens in the cache (must be > 0)
            n_sinks: Number of attention sink tokens (must be >= 0)
            n_pinned: Number of pinned prefill tokens (must be >= 0)

        Raises:
            ValueError: If capacity <= 0, or n_sinks + n_pinned >= capacity,
                       or n_sinks < 0, or n_pinned < 0
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if n_sinks < 0:
            raise ValueError(f"n_sinks must be >= 0, got {n_sinks}")
        if n_pinned < 0:
            raise ValueError(f"n_pinned must be >= 0, got {n_pinned}")

        reserved = n_sinks + n_pinned
        if reserved >= capacity:
            raise ValueError(
                f"n_sinks + n_pinned ({reserved}) must be < capacity ({capacity})"
            )

        self._capacity = capacity
        self._n_sinks = n_sinks
        self._n_pinned = n_pinned

        # Sinks and pinned segments stored as lists
        self._sinks: list[int] = []
        self._pinned: list[int] = []

        # FIFO rolling window: queue of token_ids
        # Maximum size = capacity - n_sinks - n_pinned
        self._rolling_window: deque[int] = deque(
            maxlen=capacity - n_sinks - n_pinned
        )

        # Counter for evicted tokens
        self._n_evicted = 0

    def add(self, token_id: int, position: int) -> bool:
        """Add a token to the cache.

        Tokens are placed in order:
        1. First n_sinks positions go to sinks
        2. Next n_pinned positions go to pinned
        3. Remaining positions go to rolling window (FIFO, evicts oldest)

        Args:
            token_id: The token ID to add
            position: The absolute sequence position (monotonically increasing)

        Returns:
            True if eviction occurred, False otherwise
        """
        # Determine where this token goes based on current count
        current_total = len(self._sinks) + len(self._pinned) + len(self._rolling_window)

        if current_total < self._n_sinks:
            # Fill sinks first
            self._sinks.append(token_id)
            return False

        elif current_total < self._n_sinks + self._n_pinned:
            # Fill pinned next
            self._pinned.append(token_id)
            return False

        else:
            # Fill rolling window; this may cause eviction
            is_full = len(self._rolling_window) == self._rolling_window.maxlen
            self._rolling_window.append(token_id)

            if is_full:
                self._n_evicted += 1
                return True
            return False

    def evict_if_needed(self) -> bool:
        """Check if the cache is full and would need to evict on next add.

        Returns:
            True if the cache is at capacity and rolling window is full

        Note:
            This method does NOT actually evict; it only checks the condition.
        """
        current_total = len(self._sinks) + len(self._pinned) + len(self._rolling_window)
        if current_total < self._capacity:
            return False

        # At capacity - check if rolling window is full
        return len(self._rolling_window) == self._rolling_window.maxlen

    def is_full(self) -> bool:
        """Check if rolling window is nearly full (1 slot or less remaining).

        Returns:
            True if rolling window has <=1 slots remaining (next add will evict or be close)
        """
        remaining = self._rolling_window.maxlen - len(self._rolling_window)
        return remaining <= 1

    def get_stats(self) -> CacheStats:
        """Get current cache statistics.

        Returns:
            CacheStats with utilization, rolling_capacity, and eviction counts
        """
        n_used = len(self._sinks) + len(self._pinned) + len(self._rolling_window)
        return CacheStats(
            capacity=self._capacity,
            n_sinks=self._n_sinks,
            n_pinned=self._n_pinned,
            n_rolling=self._rolling_window.maxlen,
            n_used=n_used,
            n_evicted=self._n_evicted,
        )

    def reset(self) -> None:
        """Clear all tokens and reset eviction counter."""
        self._sinks.clear()
        self._pinned.clear()
        self._rolling_window.clear()
        self._n_evicted = 0

    def __len__(self) -> int:
        """Return the number of tokens currently in the cache.

        Includes sinks, pinned, and rolling window tokens.
        """
        return len(self._sinks) + len(self._pinned) + len(self._rolling_window)
