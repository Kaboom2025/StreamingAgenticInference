"""Integration tests for the eval harness using ScriptedBackend.

These tests verify that:
- RecoveryEvents are created for each injection→act pair
- recovery_tokens = act_position - injection_position
- The harness correctly detects episode termination (done=True from env)
- The chunking baseline formula produces values ≥ streaming values
- ALFWorld episode loop sets is_failure and is_correct correctly
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional

import pytest

from streamagent.engine.interfaces import Action, BackendProtocol, CacheStats, Observation
from streamagent.engine.kv_stream import KVStream, KVStreamConfig
from streamagent.env.gridworld import GridWorld
from streamagent.env.scenarios import GridCell, Scenario
from streamagent.eval.harness import run_alfworld_episode, run_gridworld_episode


# ---------------------------------------------------------------------------
# Minimal scripted backend (char-per-token, same as test_integration.py)
# ---------------------------------------------------------------------------


class ScriptedBackend(BackendProtocol):
    """Emits chars from a script; returns space when script is exhausted."""

    def __init__(self, script: list[str], ctx: int = 512) -> None:
        self._tokens: deque[str] = deque("".join(script))
        self._ctx = ctx

    def load_model(self, model_path: str, **kwargs: object) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def detokenize(self, ids: list[int]) -> str:
        return "".join(chr(i % 128) for i in ids)

    def prefill(self, input_ids: list[int]) -> int:
        return len(input_ids)

    def inject_one(self, token_id: int, cache_position: int) -> None:
        pass

    def forward_one(self, token_id: int, cache_position: int) -> tuple[int, float]:
        if self._tokens:
            return ord(self._tokens.popleft()), -1.0
        return ord(" "), -1.0

    @property
    def context_length(self) -> int:
        return self._ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SCENARIO = Scenario(
    name="test_2x2",
    width=2,
    height=2,
    start=GridCell(0, 0),
    goal=GridCell(1, 1),
    walls=frozenset(),
    max_steps=50,
)

SYSTEM_PROMPT = "Navigate to the goal."


def _make_stream(script: list[str]) -> KVStream:
    config = KVStreamConfig(
        model_id="unused",
        backend="llama",   # won't be loaded; we inject _backend directly
    )
    stream = KVStream(config, SYSTEM_PROMPT)
    stream._backend = ScriptedBackend(script)
    stream._started = True
    # Position starts at 0 for the scripted backend (no real prefill)
    stream._last_prefill_token_id = 0
    return stream


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_act_creates_recovery_event():
    """One act tag → one RecoveryEvent."""
    # Script emits: think tokens, then one act, then spaces
    script = ['<act cmd="move south"/>']
    stream = _make_stream(script)
    env = GridWorld(SIMPLE_SCENARIO)

    metrics = run_gridworld_episode(stream, env, max_tokens=200)

    assert len(metrics.recovery_events) >= 1
    assert metrics.recovery_events[0].action == "move south"


def test_recovery_tokens_positive():
    """recovery_tokens must be positive (act comes after injection)."""
    script = ['<act cmd="move east"/>']
    stream = _make_stream(script)
    env = GridWorld(SIMPLE_SCENARIO)

    metrics = run_gridworld_episode(stream, env, max_tokens=200)

    for event in metrics.recovery_events:
        assert event.recovery_tokens > 0


def test_act_position_greater_than_injection_position():
    """act_position > injection_position for every event."""
    script = ['<act cmd="move south"/>']
    stream = _make_stream(script)
    env = GridWorld(SIMPLE_SCENARIO)

    metrics = run_gridworld_episode(stream, env, max_tokens=200)

    for event in metrics.recovery_events:
        assert event.act_position > event.injection_position


def test_chunking_baseline_gte_streaming():
    """Chunking latency is always ≥ streaming latency."""
    script = ['<act cmd="move east"/>']
    stream = _make_stream(script)
    env = GridWorld(SIMPLE_SCENARIO)

    metrics = run_gridworld_episode(stream, env, max_tokens=200)

    assert metrics.mean_chunking_recovery_tokens(32) >= metrics.mean_recovery_tokens


def test_token_count_matches_generated_tokens():
    """total_tokens reflects how many tokens were yielded."""
    # Short script so we hit max_tokens cap
    script = ['<act cmd="look"/>']
    stream = _make_stream(script)
    env = GridWorld(SIMPLE_SCENARIO)

    metrics = run_gridworld_episode(stream, env, max_tokens=50)

    assert 0 < metrics.total_tokens <= 50


def test_scenario_name_preserved():
    script = ['<act cmd="move east"/>']
    stream = _make_stream(script)
    env = GridWorld(SIMPLE_SCENARIO)

    metrics = run_gridworld_episode(stream, env, max_tokens=200)

    assert metrics.scenario_name == "test_2x2"


def test_obs_content_is_non_empty():
    """Each recovery event carries the observation content that triggered it."""
    script = ['<act cmd="move south"/>']
    stream = _make_stream(script)
    env = GridWorld(SIMPLE_SCENARIO)

    metrics = run_gridworld_episode(stream, env, max_tokens=200)

    for event in metrics.recovery_events:
        assert len(event.obs_content) > 0


def test_kv_stream_position_property():
    """KVStream.position exposes the internal _position counter."""
    config = KVStreamConfig(model_id="unused", backend="llama")
    stream = KVStream(config, "prompt")
    stream._backend = ScriptedBackend([])
    stream._started = True
    stream._position = 42
    assert stream.position == 42


# ---------------------------------------------------------------------------
# Mock ALFWorldEnv for harness tests
# ---------------------------------------------------------------------------


class _MockALFWorldEnv:
    """Minimal ALFWorld env for harness testing.

    obs_sequence: list of (content, reward, done) tuples returned by step().
    First entry is returned by reset().
    """

    def __init__(self, obs_sequence: list[tuple[str, float, bool]]) -> None:
        self._seq = obs_sequence
        self._idx = 0

    def reset(self) -> Observation:
        self._idx = 0
        return Observation(type="alfworld", content=self._seq[0][0])

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        self._idx = min(self._idx + 1, len(self._seq) - 1)
        content, reward, done = self._seq[self._idx]
        return Observation(type="alfworld", content=content), reward, done

    def register_injector(self, injector: object) -> None:
        pass

    def render(self) -> str:
        return self._seq[self._idx][0]


# ---------------------------------------------------------------------------
# ALFWorld harness tests
# ---------------------------------------------------------------------------


def test_alfworld_episode_task_type_propagated():
    """task_type parameter ends up in the returned EpisodeMetrics."""
    obs_seq = [
        ("You are in a kitchen.", 0.0, False),
        ("You pick up the apple.", 1.0, True),
    ]
    stream = _make_stream(['<act cmd="take" obj="apple"/>'])
    env = _MockALFWorldEnv(obs_seq)

    metrics = run_alfworld_episode(stream, env, task_type="pick", max_tokens=200)

    assert metrics.task_type == "pick"


def test_alfworld_episode_returns_episode_metrics():
    """run_alfworld_episode returns an EpisodeMetrics with recovery events."""
    obs_seq = [
        ("You are in a kitchen.", 0.0, False),
        ("You pick up the apple.", 1.0, True),
    ]
    stream = _make_stream(['<act cmd="take" obj="apple"/>'])
    env = _MockALFWorldEnv(obs_seq)

    metrics = run_alfworld_episode(stream, env, max_tokens=200)

    assert len(metrics.recovery_events) >= 1


def test_alfworld_episode_failure_detection():
    """is_failure=True when env returns a FAILURE_STRING observation."""
    obs_seq = [
        ("You are in a kitchen.", 0.0, False),
        ("Nothing happens.", 0.0, False),  # failure string
        ("You pick up the apple.", 1.0, True),
    ]
    stream = _make_stream(['<act cmd="goto" obj="microwave"/><act cmd="take" obj="apple"/>'])
    env = _MockALFWorldEnv(obs_seq)

    metrics = run_alfworld_episode(stream, env, max_tokens=400)

    # At least one event should have is_failure=True
    assert any(e.is_failure for e in metrics.recovery_events)


def test_alfworld_episode_non_failure_obs_not_marked():
    """is_failure=False for normal observations."""
    obs_seq = [
        ("You are in a kitchen.", 0.0, False),
        ("You pick up the apple.", 1.0, True),
    ]
    stream = _make_stream(['<act cmd="take" obj="apple"/>'])
    env = _MockALFWorldEnv(obs_seq)

    metrics = run_alfworld_episode(stream, env, max_tokens=200)

    # First event is triggered by the reset obs — not a failure
    assert metrics.recovery_events[0].is_failure is False


def test_alfworld_episode_is_correct_set_after_failure():
    """is_correct is not None on a failure event when judge can classify it."""
    obs_seq = [
        ("You are in a kitchen.", 0.0, False),
        ("Nothing happens.", 0.0, False),  # failure
        ("You look around.", 1.0, True),
    ]
    # goto (causes failure) → look (recovery)
    stream = _make_stream(['<act cmd="goto" obj="microwave"/><act cmd="look"/>'])
    env = _MockALFWorldEnv(obs_seq)

    metrics = run_alfworld_episode(stream, env, max_tokens=400)

    failure_events = [e for e in metrics.recovery_events if e.is_failure]
    assert len(failure_events) >= 1
    # "look" after a failure → is_correct should be True (rule 2)
    assert failure_events[0].is_correct is True
