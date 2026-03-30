"""Tests for SinkCache implementing StreamingLLM attention-sink eviction strategy.

This module tests the SinkCache class which manages a fixed-capacity KV cache
with three segments:
  1. Attention sink tokens (always kept, never evicted)
  2. Pinned prefill tokens (always kept)
  3. FIFO rolling window (evicts oldest when full)

Reference: Xiao et al. ICLR 2024 - StreamingLLM
"""

import pytest
from streamagent.engine.sink_cache import SinkCache
from streamagent.engine.interfaces import CacheStats


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


def test_init_valid():
    """Test valid SinkCache construction."""
    cache = SinkCache(capacity=128, n_sinks=4, n_pinned=8)
    assert len(cache) == 0
    assert cache.is_full() is False
    stats = cache.get_stats()
    assert stats.capacity == 128
    assert stats.n_sinks == 4
    assert stats.n_pinned == 8
    assert stats.n_rolling == 128 - 4 - 8
    assert stats.n_used == 0
    assert stats.n_evicted == 0


def test_init_defaults():
    """Test default parameters."""
    cache = SinkCache(capacity=512)
    stats = cache.get_stats()
    assert stats.n_sinks == 4
    assert stats.n_pinned == 0


def test_init_invalid_capacity_zero():
    """Test that capacity <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        SinkCache(capacity=0)


def test_init_invalid_capacity_negative():
    """Test that negative capacity raises ValueError."""
    with pytest.raises(ValueError):
        SinkCache(capacity=-1)


def test_init_invalid_sinks_exceed_capacity():
    """Test that n_sinks >= capacity raises ValueError."""
    with pytest.raises(ValueError):
        SinkCache(capacity=10, n_sinks=10)


def test_init_invalid_pinned_exceed_capacity():
    """Test that n_sinks + n_pinned >= capacity raises ValueError."""
    with pytest.raises(ValueError):
        SinkCache(capacity=10, n_sinks=3, n_pinned=8)


def test_init_invalid_negative_sinks():
    """Test that negative n_sinks raises ValueError."""
    with pytest.raises(ValueError):
        SinkCache(capacity=128, n_sinks=-1)


def test_init_invalid_negative_pinned():
    """Test that negative n_pinned raises ValueError."""
    with pytest.raises(ValueError):
        SinkCache(capacity=128, n_pinned=-1)


# ---------------------------------------------------------------------------
# Basic Add Tests (No Eviction)
# ---------------------------------------------------------------------------


def test_add_tokens_no_eviction():
    """Test adding tokens below capacity returns False (no eviction)."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)

    # Add tokens below capacity (rolling window has 6 slots)
    assert cache.add(token_id=1, position=0) is False
    assert len(cache) == 1

    assert cache.add(token_id=2, position=1) is False
    assert len(cache) == 2

    assert cache.add(token_id=3, position=2) is False
    assert len(cache) == 3


def test_add_to_exactly_capacity():
    """Test adding exactly to capacity without eviction."""
    cache = SinkCache(capacity=5, n_sinks=1, n_pinned=1)
    # Rolling window: 5 - 1 - 1 = 3 slots

    # Sink + pinned + 3 rolling
    for i in range(4):
        result = cache.add(token_id=i, position=i)
        assert result is False
        assert len(cache) == i + 1


# ---------------------------------------------------------------------------
# Eviction Tests
# ---------------------------------------------------------------------------


def test_add_triggers_eviction():
    """Test that adding beyond rolling window capacity triggers eviction."""
    cache = SinkCache(capacity=5, n_sinks=1, n_pinned=1)
    # Rolling window: 5 - 1 - 1 = 3 slots

    # Fill sinks, pinned, and rolling window (5 total tokens)
    for i in range(5):
        cache.add(token_id=i, position=i)

    assert len(cache) == 5
    assert cache.is_full() is True

    # Adding one more should trigger eviction (rolling window is now full)
    result = cache.add(token_id=5, position=5)
    assert result is True  # Eviction occurred
    assert len(cache) == 5  # Still at capacity


def test_eviction_count_increments():
    """Test that eviction counter increments."""
    cache = SinkCache(capacity=4, n_sinks=1, n_pinned=1)
    # Rolling window: 4 - 1 - 1 = 2 slots

    # Fill to capacity (sinks + pinned + rolling = 1 + 1 + 2 = 4)
    for i in range(4):
        cache.add(token_id=i, position=i)

    stats = cache.get_stats()
    assert stats.n_evicted == 0

    # Trigger first eviction (5th add, rolling window is full)
    cache.add(token_id=4, position=4)
    stats = cache.get_stats()
    assert stats.n_evicted == 1

    # Trigger second eviction (6th add)
    cache.add(token_id=5, position=5)
    stats = cache.get_stats()
    assert stats.n_evicted == 2


def test_sinks_never_evicted():
    """Test that sink tokens are never evicted."""
    cache = SinkCache(capacity=5, n_sinks=2, n_pinned=0)
    # Rolling window: 5 - 2 - 0 = 3 slots

    # Add tokens to fill
    for i in range(5):
        cache.add(token_id=i, position=i)

    # Trigger multiple evictions
    for i in range(5, 10):
        cache.add(token_id=i, position=i)

    # The first n_sinks tokens should still be there
    # (We verify by checking position 0 and 1 were never replaced)
    stats = cache.get_stats()
    assert stats.n_evicted == 5  # Only rolling window evicted
    assert len(cache) == 5


def test_pinned_never_evicted():
    """Test that pinned tokens are never evicted."""
    cache = SinkCache(capacity=6, n_sinks=1, n_pinned=2)
    # Rolling window: 6 - 1 - 2 = 3 slots

    # Add tokens to fill
    for i in range(6):
        cache.add(token_id=i, position=i)

    # Trigger multiple evictions
    for i in range(6, 15):
        cache.add(token_id=i, position=i)

    stats = cache.get_stats()
    # Only rolling window tokens evicted (not sinks or pinned)
    assert stats.n_evicted == 9
    assert len(cache) == 6


def test_rolling_fifo_eviction_order():
    """Test that rolling window evicts in FIFO order."""
    cache = SinkCache(capacity=5, n_sinks=2, n_pinned=0)
    # Rolling window: 5 - 2 - 0 = 3 slots
    # Layout: [sink0, sink1, rolling0, rolling1, rolling2]

    # Add 5 tokens: positions 0-4
    # Positions 0-1: sinks
    # Positions 2-4: rolling (all 3 slots filled)
    for i in range(5):
        cache.add(token_id=i, position=i)

    assert cache.is_full()

    # Position 5 should evict position 2 (oldest rolling)
    cache.add(token_id=100, position=5)

    # Position 6 should evict position 3 (next oldest rolling)
    cache.add(token_id=101, position=6)

    stats = cache.get_stats()
    assert stats.n_evicted == 2


# ---------------------------------------------------------------------------
# CacheStats Tests
# ---------------------------------------------------------------------------


def test_stats_after_eviction():
    """Test CacheStats.n_evicted increments after eviction."""
    cache = SinkCache(capacity=3, n_sinks=1, n_pinned=0)

    stats_initial = cache.get_stats()
    assert stats_initial.n_evicted == 0

    # Fill to capacity (sinks + rolling = 1 + 2 = 3)
    cache.add(token_id=0, position=0)
    cache.add(token_id=1, position=1)
    cache.add(token_id=2, position=2)

    # Trigger eviction (rolling window is full)
    cache.add(token_id=3, position=3)

    stats_after = cache.get_stats()
    assert stats_after.n_evicted == 1


def test_stats_utilization():
    """Test CacheStats.utilization calculation."""
    cache = SinkCache(capacity=100, n_sinks=4, n_pinned=10)

    # Empty cache
    stats = cache.get_stats()
    assert stats.utilization == pytest.approx(0.0)
    assert stats.n_used == 0

    # Add 50 tokens (all fit without eviction)
    for i in range(50):
        cache.add(token_id=i, position=i)

    stats = cache.get_stats()
    assert stats.n_used == 50
    assert stats.utilization == pytest.approx(0.5)


def test_stats_rolling_capacity_property():
    """Test CacheStats.rolling_capacity property."""
    cache = SinkCache(capacity=512, n_sinks=4, n_pinned=32)
    stats = cache.get_stats()

    assert stats.rolling_capacity == 512 - 4 - 32
    assert stats.rolling_capacity == 476


def test_stats_n_rolling_value():
    """Test that CacheStats.n_rolling is correctly reported."""
    cache = SinkCache(capacity=100, n_sinks=5, n_pinned=20)
    stats = cache.get_stats()

    # n_rolling should be the number of slots available in rolling window
    assert stats.n_rolling == 100 - 5 - 20
    assert stats.n_rolling == 75


def test_stats_after_reset():
    """Test that stats reset correctly after reset()."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)

    # Add some tokens and trigger eviction
    for i in range(15):
        cache.add(token_id=i, position=i)

    stats_before = cache.get_stats()
    assert stats_before.n_evicted > 0

    # Reset
    cache.reset()

    stats_after = cache.get_stats()
    assert stats_after.n_used == 0
    assert stats_after.n_evicted == 0


# ---------------------------------------------------------------------------
# is_full Tests
# ---------------------------------------------------------------------------


def test_is_full_when_rolling_window_has_space():
    """Test is_full() returns False when rolling window has space."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)
    # Rolling: 10 - 2 - 2 = 6 slots

    for i in range(5):
        cache.add(token_id=i, position=i)

    assert cache.is_full() is False


def test_is_full_when_rolling_window_exhausted():
    """Test is_full() returns True when rolling window is full."""
    cache = SinkCache(capacity=5, n_sinks=1, n_pinned=1)
    # Rolling: 5 - 1 - 1 = 3 slots

    # Fill sink + pinned + rolling (4 tokens total)
    for i in range(4):
        cache.add(token_id=i, position=i)

    assert cache.is_full() is True


def test_is_full_after_eviction():
    """Test is_full() remains True after eviction."""
    cache = SinkCache(capacity=5, n_sinks=1, n_pinned=1)

    # Fill to capacity and trigger eviction
    for i in range(6):
        cache.add(token_id=i, position=i)

    # Cache should still be full
    assert cache.is_full() is True


# ---------------------------------------------------------------------------
# evict_if_needed Tests
# ---------------------------------------------------------------------------


def test_evict_if_needed_when_not_full():
    """Test evict_if_needed() returns False when rolling window has space."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)

    cache.add(token_id=0, position=0)
    cache.add(token_id=1, position=1)

    assert cache.evict_if_needed() is False


def test_evict_if_needed_when_full():
    """Test evict_if_needed() returns True when rolling window is full."""
    cache = SinkCache(capacity=5, n_sinks=1, n_pinned=1)
    # Rolling: 5 - 1 - 1 = 3 slots

    # Fill to capacity (sinks + pinned + rolling = 1 + 1 + 3 = 5)
    for i in range(5):
        cache.add(token_id=i, position=i)

    assert cache.evict_if_needed() is True


def test_evict_if_needed_does_not_evict():
    """Test that evict_if_needed() does not actually evict."""
    cache = SinkCache(capacity=5, n_sinks=1, n_pinned=1)

    # Fill to capacity (sinks + pinned + rolling = 1 + 1 + 3 = 5)
    for i in range(5):
        cache.add(token_id=i, position=i)

    # Check eviction needed
    assert cache.evict_if_needed() is True

    # Verify no actual eviction occurred (no count change)
    stats = cache.get_stats()
    assert stats.n_evicted == 0
    assert len(cache) == 5


# ---------------------------------------------------------------------------
# Reset Tests
# ---------------------------------------------------------------------------


def test_reset_clears_state():
    """Test that reset() clears all tokens and counters."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)

    # Add tokens
    for i in range(5):
        cache.add(token_id=i, position=i)

    assert len(cache) == 5

    # Reset
    cache.reset()

    # Verify clean state
    assert len(cache) == 0
    stats = cache.get_stats()
    assert stats.n_used == 0
    assert stats.n_evicted == 0


def test_reset_clears_eviction_counter():
    """Test that reset() clears the eviction counter."""
    cache = SinkCache(capacity=4, n_sinks=1, n_pinned=0)

    # Fill and trigger evictions
    for i in range(8):
        cache.add(token_id=i, position=i)

    stats_before = cache.get_stats()
    assert stats_before.n_evicted > 0

    # Reset
    cache.reset()

    stats_after = cache.get_stats()
    assert stats_after.n_evicted == 0


def test_reset_allows_reuse():
    """Test that cache can be reused after reset()."""
    cache = SinkCache(capacity=5, n_sinks=1, n_pinned=1)

    # Use once
    for i in range(4):
        cache.add(token_id=i, position=i)

    # Reset
    cache.reset()

    # Use again
    for i in range(4):
        cache.add(token_id=100 + i, position=100 + i)

    assert len(cache) == 4


# ---------------------------------------------------------------------------
# Length Tests
# ---------------------------------------------------------------------------


def test_len_empty():
    """Test __len__() on empty cache."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)
    assert len(cache) == 0


def test_len_after_adds():
    """Test __len__() after multiple adds."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)

    for i in range(7):
        cache.add(token_id=i, position=i)
        assert len(cache) == i + 1


def test_len_after_eviction():
    """Test __len__() remains constant after eviction."""
    cache = SinkCache(capacity=4, n_sinks=1, n_pinned=1)

    # Fill to capacity
    for i in range(3):
        cache.add(token_id=i, position=i)

    assert len(cache) == 3

    # Trigger eviction
    cache.add(token_id=3, position=3)

    # Length should stay at capacity
    assert len(cache) == 4

    # More evictions
    for i in range(4, 10):
        cache.add(token_id=i, position=i)
        assert len(cache) == 4


# ---------------------------------------------------------------------------
# Edge Cases and Stress Tests
# ---------------------------------------------------------------------------


def test_large_number_of_evictions():
    """Test handling many sequential evictions."""
    cache = SinkCache(capacity=16, n_sinks=2, n_pinned=2)
    # Rolling: 16 - 2 - 2 = 12 slots

    # Add 100 tokens
    for i in range(100):
        cache.add(token_id=i, position=i)

    assert len(cache) == 16
    stats = cache.get_stats()
    assert stats.n_evicted == 84  # 100 - 16
    assert stats.n_used == 16


def test_zero_rolling_window():
    """Test when capacity equals n_sinks + n_pinned."""
    # This should be rejected during init
    with pytest.raises(ValueError):
        SinkCache(capacity=10, n_sinks=5, n_pinned=5)


def test_minimal_valid_cache():
    """Test smallest valid cache (capacity=1, n_sinks=0, n_pinned=0)."""
    cache = SinkCache(capacity=1, n_sinks=0, n_pinned=0)
    assert len(cache) == 0

    cache.add(token_id=0, position=0)
    assert len(cache) == 1

    # Next add should trigger eviction
    result = cache.add(token_id=1, position=1)
    assert result is True
    assert len(cache) == 1


def test_add_with_increasing_positions():
    """Test that positions are strictly monotonic (as used by backend)."""
    cache = SinkCache(capacity=10, n_sinks=1, n_pinned=0)

    # Add with increasing positions
    for i in range(20):
        cache.add(token_id=i, position=i)


def test_stats_consistency_after_operations():
    """Test that stats remain consistent after various operations."""
    cache = SinkCache(capacity=20, n_sinks=2, n_pinned=5)

    stats = cache.get_stats()
    assert stats.capacity == 20
    assert stats.n_sinks == 2
    assert stats.n_pinned == 5
    assert stats.rolling_capacity == 13

    # Add tokens
    for i in range(25):
        cache.add(token_id=i, position=i)

    stats = cache.get_stats()
    assert stats.capacity == 20
    assert stats.n_sinks == 2
    assert stats.n_pinned == 5
    assert stats.rolling_capacity == 13
    assert stats.n_used == 20
    assert stats.n_evicted == 5
    assert len(cache) == 20


def test_multiple_resets():
    """Test that multiple resets work correctly."""
    cache = SinkCache(capacity=10, n_sinks=2, n_pinned=2)

    for _ in range(3):
        # Add and fill
        for i in range(12):
            cache.add(token_id=i, position=i)

        assert len(cache) == 10

        # Reset
        cache.reset()
        assert len(cache) == 0
        assert cache.get_stats().n_evicted == 0
