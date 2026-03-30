"""Tests for KVStream — the async never-terminated generation loop.

All tests use a MockBackend so no real model is required.
"""

from __future__ import annotations

import asyncio
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest

from streamagent.engine.interfaces import (
    BackendProtocol,
    CacheStats,
    Observation,
    ObsInjectorProtocol,
    Token,
)
from streamagent.engine.kv_stream import KVStream, KVStreamConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockBackend(BackendProtocol):
    """Deterministic backend: tokenize by char-codes, forward returns next_id=id+1."""

    def __init__(self, ctx: int = 512) -> None:
        self._ctx = ctx
        self._loaded = False
        self.forward_calls: list[tuple[int, int]] = []

    def load_model(self, model_path: str, **kwargs: object) -> None:
        self._loaded = True

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def detokenize(self, ids: list[int]) -> str:
        return "".join(chr(i % 128) for i in ids)

    def prefill(self, input_ids: list[int]) -> int:
        return len(input_ids)

    def forward_one(self, token_id: int, cache_position: int) -> tuple[int, float]:
        self.forward_calls.append((token_id, cache_position))
        # cycle through ASCII printable range to produce deterministic output
        next_id = (token_id % 95) + 32  # stays in printable ASCII
        return next_id, -1.0

    @property
    def context_length(self) -> int:
        return self._ctx


def make_stream(
    system_prompt: str = "<goal>test</goal>",
    window_length: int = 64,
    sink_tokens: int = 4,
    backend: BackendProtocol | None = None,
) -> KVStream:
    cfg = KVStreamConfig(
        model_id="mock",
        backend="llama",
        sink_tokens=sink_tokens,
        window_length=window_length,
    )
    stream = KVStream(cfg, system_prompt)
    stream._backend = backend or MockBackend()
    return stream


async def collect_tokens(stream: KVStream, n: int) -> list[Token]:
    tokens: list[Token] = []
    async for tok in stream.run():
        tokens.append(tok)
        if len(tokens) >= n:
            break
    return tokens


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = KVStreamConfig(model_id="x", backend="llama")
    assert cfg.sink_tokens == 4
    assert cfg.window_length == 2048
    assert cfg.temperature == 0.6


def test_config_custom() -> None:
    cfg = KVStreamConfig(model_id="x", backend="hf", sink_tokens=2, window_length=512)
    assert cfg.sink_tokens == 2
    assert cfg.window_length == 512


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_kvstream_construction() -> None:
    stream = make_stream()
    assert stream is not None


def test_kvstream_cache_stats_before_start() -> None:
    stream = make_stream()
    stats = stream.cache_stats
    assert isinstance(stats, CacheStats)
    assert stats.n_used == 0


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def test_prefill_sets_position() -> None:
    backend = MockBackend()
    stream = make_stream(system_prompt="hello", backend=backend)
    stream.start()
    # position should equal prefill length (len("hello") = 5)
    assert stream._position == len("hello")


def test_prefill_calls_backend_prefill() -> None:
    backend = MockBackend()
    stream = make_stream(system_prompt="hi", backend=backend)
    stream.start()
    # backend.prefill was called with encoded "hi"
    assert stream._position == 2  # len("hi")


# ---------------------------------------------------------------------------
# Generation loop — basic token yield
# ---------------------------------------------------------------------------


def test_run_yields_tokens() -> None:
    stream = make_stream()
    stream.start()
    tokens = asyncio.run(collect_tokens(stream, 5))
    assert len(tokens) == 5


def test_run_yields_token_objects() -> None:
    stream = make_stream()
    stream.start()
    tokens = asyncio.run(collect_tokens(stream, 3))
    for tok in tokens:
        assert isinstance(tok, Token)
        assert isinstance(tok.id, int)
        assert isinstance(tok.text, str)


def test_cache_position_strictly_monotonic() -> None:
    """cache_position must increment by 1 on every forward call."""
    backend = MockBackend()
    stream = make_stream(system_prompt="ab", backend=backend)
    stream.start()
    asyncio.run(collect_tokens(stream, 10))
    positions = [pos for _, pos in backend.forward_calls]
    # positions start after prefill (prefill length = 2)
    for i in range(1, len(positions)):
        assert positions[i] == positions[i - 1] + 1, (
            f"Non-monotonic at index {i}: {positions[i-1]} → {positions[i]}"
        )


def test_cache_position_starts_after_prefill() -> None:
    backend = MockBackend()
    stream = make_stream(system_prompt="xyz", backend=backend)  # prefill len = 3
    stream.start()
    asyncio.run(collect_tokens(stream, 1))
    first_pos = backend.forward_calls[0][1]
    assert first_pos == 3  # must continue from prefill length


# ---------------------------------------------------------------------------
# Observation injection
# ---------------------------------------------------------------------------


def test_inject_queues_observation() -> None:
    stream = make_stream()
    stream.start()
    stream.inject("<obs type='test'>hello</obs>")
    assert not stream._injector.empty()


def test_inject_consumed_before_next_token() -> None:
    """After injecting, the injector should be empty after the next token is generated."""
    stream = make_stream()
    stream.start()
    stream.inject("<obs type='test'>x</obs>")

    async def _run() -> None:
        async for _ in stream.run():
            # after first token, injection must have been drained
            assert stream._injector.empty()
            break

    asyncio.run(_run())


def test_inject_position_still_monotonic_after_injection() -> None:
    """cache_position must remain monotonic even after observation injection."""
    backend = MockBackend()
    stream = make_stream(system_prompt="a", backend=backend)
    stream.start()

    async def _run() -> None:
        count = 0
        async for _ in stream.run():
            count += 1
            if count == 3:
                stream.inject("<obs type='x'>hi</obs>")
            if count == 8:
                break

    asyncio.run(_run())

    positions = [pos for _, pos in backend.forward_calls]
    for i in range(1, len(positions)):
        assert positions[i] == positions[i - 1] + 1, (
            f"Monotonicity broken at {i}: {positions[i-1]} → {positions[i]}"
        )


# ---------------------------------------------------------------------------
# cache_stats property
# ---------------------------------------------------------------------------


def test_cache_stats_increments() -> None:
    stream = make_stream(system_prompt="a")
    stream.start()
    asyncio.run(collect_tokens(stream, 5))
    stats = stream.cache_stats
    assert stats.n_used > 0


def test_cache_stats_has_correct_type() -> None:
    stream = make_stream()
    stream.start()
    stats = stream.cache_stats
    assert isinstance(stats, CacheStats)


# ---------------------------------------------------------------------------
# SinkCache integration — eviction
# ---------------------------------------------------------------------------


def test_eviction_does_not_break_generation() -> None:
    """Generate more tokens than window_length; eviction must not raise."""
    stream = make_stream(system_prompt="a", window_length=16, sink_tokens=2)
    stream.start()
    # generate 32 tokens (2x the window — forces eviction)
    tokens = asyncio.run(collect_tokens(stream, 32))
    assert len(tokens) == 32


def test_eviction_increments_evicted_count() -> None:
    stream = make_stream(system_prompt="a", window_length=10, sink_tokens=2)
    stream.start()
    asyncio.run(collect_tokens(stream, 20))
    assert stream.cache_stats.n_evicted > 0


# ---------------------------------------------------------------------------
# Stop / cleanup
# ---------------------------------------------------------------------------


def test_stop_halts_generation() -> None:
    stream = make_stream()
    stream.start()

    async def _run() -> int:
        count = 0
        async for _ in stream.run():
            count += 1
            if count == 3:
                stream.stop()
        return count

    total = asyncio.run(_run())
    # may generate 3 or 4 (stop is cooperative), but not unbounded
    assert total <= 5
