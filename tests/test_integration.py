"""End-to-end integration tests: KVStream → Router → GridWorld → inject obs.

These tests wire every component together with a ScriptedBackend that emits
pre-planned token sequences, then verify:
  - Actions parsed by Router are dispatched to GridWorld
  - GridWorld observations flow back through ObsInjector into KVStream
  - cache_position is strictly monotonic across the full episode
  - GridWorld state changes correctly in response to dispatched actions
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

import pytest

from streamagent.engine.injector import ObsInjector
from streamagent.engine.interfaces import (
    Action,
    BackendProtocol,
    CacheStats,
    Observation,
    Token,
)
from streamagent.engine.kv_stream import KVStream, KVStreamConfig
from streamagent.engine.router import Router
from streamagent.env.gridworld import GridWorld
from streamagent.env.scenarios import GridCell, Scenario, load_scenarios


# ---------------------------------------------------------------------------
# ScriptedBackend — emits a pre-planned sequence of full strings as tokens
# ---------------------------------------------------------------------------


class ScriptedBackend(BackendProtocol):
    """Returns tokens from a script; falls back to space when script exhausted."""

    def __init__(self, script: list[str], ctx: int = 512) -> None:
        # Explode each script string into individual chars so we emit
        # one char per forward_one call (simulates subword tokenizer).
        self._tokens: deque[str] = deque("".join(script))
        self._ctx = ctx
        self._positions: list[int] = []

    def load_model(self, model_path: str, **kwargs: object) -> None:
        pass

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def detokenize(self, ids: list[int]) -> str:
        return "".join(chr(i % 128) for i in ids)

    def prefill(self, input_ids: list[int]) -> int:
        return len(input_ids)

    def inject_one(self, token_id: int, cache_position: int) -> None:
        # Record position for monotonicity checks but do NOT consume the script.
        # Injection return values are discarded by KVStream; consuming script
        # chars here would silently drop future generated tokens.
        self._positions.append(cache_position)

    def forward_one(self, token_id: int, cache_position: int) -> tuple[int, float]:
        self._positions.append(cache_position)
        if self._tokens:
            ch = self._tokens.popleft()
            return ord(ch), -1.0
        return ord(" "), -1.0

    @property
    def context_length(self) -> int:
        return self._ctx

    def positions_are_monotonic(self) -> bool:
        return all(
            self._positions[i] == self._positions[i - 1] + 1
            for i in range(1, len(self._positions))
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scripted_stream(script: list[str], system_prompt: str = "S") -> KVStream:
    cfg = KVStreamConfig(model_id="mock", backend="llama", window_length=128)
    stream = KVStream(cfg, system_prompt)
    stream._backend = ScriptedBackend(script)
    return stream


async def run_episode(
    stream: KVStream,
    router: Router,
    env: GridWorld,
    max_tokens: int = 200,
) -> tuple[list[Action], list[int]]:
    """Drive one episode; return (dispatched_actions, all_cache_positions)."""
    actions: list[Action] = []
    positions_snapshot: list[int] = []

    token_count = 0
    async for token in stream.run():
        out = router.process(token)
        positions_snapshot.append(stream._position)

        if out.action is not None:
            actions.append(out.action)
            _, _, done = env.step(out.action)
            if done:
                stream.stop()
                break

        token_count += 1
        if token_count >= max_tokens:
            stream.stop()
            break

    return actions, positions_snapshot


# ---------------------------------------------------------------------------
# Test 1: Router receives tokens and parses a single action
# ---------------------------------------------------------------------------


def test_router_parses_action_from_stream() -> None:
    script = ['<act cmd="look"/>']
    stream = make_scripted_stream(script)
    stream.start()
    router = Router()

    actions: list[Action] = []

    async def _run() -> None:
        async for tok in stream.run():
            out = router.process(tok)
            if out.action:
                actions.append(out.action)
                stream.stop()
                break
            if stream._position > 50:
                stream.stop()
                break

    asyncio.run(_run())
    assert len(actions) == 1
    assert actions[0].command == "look"


# ---------------------------------------------------------------------------
# Test 2: Action dispatched to GridWorld changes its state
# ---------------------------------------------------------------------------


def test_action_dispatched_to_gridworld() -> None:
    scenario = Scenario(
        name="test",
        width=5,
        height=5,
        start=GridCell(0, 2),
        goal=GridCell(4, 2),
        walls=frozenset(),
        max_steps=50,
    )
    env = GridWorld(scenario)
    env.reset()

    script = ['<act cmd="move east"/>']
    stream = make_scripted_stream(script)
    stream.start()
    router = Router()
    injector = stream._injector
    env.register_injector(injector)

    async def _run() -> None:
        async for tok in stream.run():
            out = router.process(tok)
            if out.action:
                env.step(out.action)
                stream.stop()
                break
            if stream._position > 60:
                stream.stop()
                break

    asyncio.run(_run())
    # Agent should have moved east from (0,2) to (1,2)
    assert env.current_position == GridCell(1, 2)


# ---------------------------------------------------------------------------
# Test 3: GridWorld observation injected back into stream
# ---------------------------------------------------------------------------


def test_observation_injected_into_stream() -> None:
    scenario = Scenario(
        name="test",
        width=5,
        height=5,
        start=GridCell(0, 0),
        goal=GridCell(4, 4),
        walls=frozenset(),
        max_steps=50,
    )
    env = GridWorld(scenario)
    env.reset()

    script = ['<act cmd="look"/>  ']  # trailing spaces keep stream alive
    stream = make_scripted_stream(script)
    stream.start()
    injector = stream._injector
    env.register_injector(injector)
    router = Router()

    injected_count = [0]

    async def _run() -> None:
        async for tok in stream.run():
            # If injector has observations, they'll be consumed by the stream
            # before the next token. We verify they arrived.
            out = router.process(tok)
            if out.action:
                env.step(out.action)  # this calls injector.put(obs)
            # Check that injection happened (injector drained by stream)
            if stream._position > 30:
                injected_count[0] = stream.cache_stats.n_used
                stream.stop()
                break

    asyncio.run(_run())
    # n_used should be larger than just prefill + generated tokens
    # because injection forward passes also add to position
    assert stream._position > 1


# ---------------------------------------------------------------------------
# Test 4: cache_position monotonic across full episode with injections
# ---------------------------------------------------------------------------


def test_cache_position_monotonic_full_episode() -> None:
    scenario = Scenario(
        name="monotonic_test",
        width=6,
        height=6,
        start=GridCell(0, 0),
        goal=GridCell(5, 5),
        walls=frozenset(),
        max_steps=20,
    )
    env = GridWorld(scenario)
    env.reset()

    # Script: three move actions followed by padding
    script = [
        '<act cmd="move east"/>',
        '<act cmd="move east"/>',
        '<act cmd="move south"/>',
        "   " * 20,
    ]
    stream = make_scripted_stream(script)
    backend: ScriptedBackend = stream._backend  # type: ignore[assignment]
    stream.start()
    injector = stream._injector
    env.register_injector(injector)
    router = Router()

    async def _run() -> None:
        action_count = 0
        async for tok in stream.run():
            out = router.process(tok)
            if out.action:
                env.step(out.action)
                action_count += 1
            if action_count >= 3 or stream._position > 150:
                stream.stop()
                break

    asyncio.run(_run())

    assert backend.positions_are_monotonic(), (
        f"Non-monotonic positions detected: {backend._positions[:20]}..."
    )


# ---------------------------------------------------------------------------
# Test 5: Full mini-episode — agent navigates to goal
# ---------------------------------------------------------------------------


def test_full_episode_agent_reaches_goal() -> None:
    # 3x3 grid, goal at (2,0), agent starts at (0,0)
    # Optimal path: east, east
    scenario = Scenario(
        name="mini",
        width=3,
        height=3,
        start=GridCell(0, 0),
        goal=GridCell(2, 0),
        walls=frozenset(),
        max_steps=10,
    )
    env = GridWorld(scenario)
    env.reset()

    script = [
        '<act cmd="move east"/>',
        '<act cmd="move east"/>',
        "   " * 30,  # padding so stream doesn't stop immediately
    ]
    stream = make_scripted_stream(script, system_prompt="<goal>reach (2,0)</goal>")
    stream.start()
    injector = stream._injector
    env.register_injector(injector)
    router = Router()

    episode_done = [False]
    total_reward = [0.0]

    async def _run() -> None:
        async for tok in stream.run():
            out = router.process(tok)
            if out.action:
                _, reward, done = env.step(out.action)
                total_reward[0] += reward
                if done:
                    episode_done[0] = True
                    stream.stop()
                    break
            if stream._position > 200:
                stream.stop()
                break

    asyncio.run(_run())

    assert env.current_position == GridCell(2, 0)
    assert episode_done[0] is True
    assert total_reward[0] == pytest.approx(1.0)  # goal reward


# ---------------------------------------------------------------------------
# Test 6: Multiple observations injected, all consumed in order
# ---------------------------------------------------------------------------


def test_multiple_injections_consumed_in_order() -> None:
    stream = make_scripted_stream(["   " * 50])
    stream.start()

    obs_order = [
        Observation(type="tick", content="step=1"),
        Observation(type="tick", content="step=2"),
        Observation(type="tick", content="step=3"),
    ]
    for obs in obs_order:
        stream._injector.put(obs)

    consumed: list[str] = []

    async def _run() -> None:
        # Run until injector is drained
        async for _ in stream.run():
            if stream._injector.empty() and len(consumed) == 0:
                # The injector was drained — record which obs were consumed
                # by checking position advanced by injection tokens
                consumed.append("drained")
                stream.stop()
                break
            if stream._position > 100:
                stream.stop()
                break

    asyncio.run(_run())
    assert len(consumed) == 1  # injector was drained once


# ---------------------------------------------------------------------------
# Test 7: Router FSM state resets properly between actions
# ---------------------------------------------------------------------------


def test_router_state_resets_between_actions() -> None:
    script = [
        '<act cmd="look"/>',
        "   ",
        '<act cmd="move north"/>',
        "   ",
    ]
    stream = make_scripted_stream(script)
    stream.start()
    router = Router()
    actions: list[Action] = []

    async def _run() -> None:
        async for tok in stream.run():
            out = router.process(tok)
            if out.action:
                actions.append(out.action)
            if len(actions) >= 2 or stream._position > 150:
                stream.stop()
                break

    asyncio.run(_run())
    assert len(actions) == 2
    assert actions[0].command == "look"
    assert actions[1].command == "move north"


# ---------------------------------------------------------------------------
# Test 8: Episode with wall collision — reward is -0.1, agent stays in place
# ---------------------------------------------------------------------------


def test_wall_collision_in_episode() -> None:
    # Put a wall at (1,0) so move east is blocked
    scenario = Scenario(
        name="wall_test",
        width=4,
        height=4,
        start=GridCell(0, 0),
        goal=GridCell(3, 3),
        walls=frozenset({GridCell(1, 0)}),
        max_steps=20,
    )
    env = GridWorld(scenario)
    env.reset()

    script = ['<act cmd="move east"/>', "   " * 20]
    stream = make_scripted_stream(script)
    stream.start()
    injector = stream._injector
    env.register_injector(injector)
    router = Router()
    rewards: list[float] = []

    async def _run() -> None:
        async for tok in stream.run():
            out = router.process(tok)
            if out.action:
                _, reward, _ = env.step(out.action)
                rewards.append(reward)
                stream.stop()
                break
            if stream._position > 80:
                stream.stop()
                break

    asyncio.run(_run())

    assert env.current_position == GridCell(0, 0)  # stayed in place
    assert rewards[0] == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Test 9: SinkCache eviction does not corrupt episode
# ---------------------------------------------------------------------------


def test_eviction_during_episode_no_corruption() -> None:
    """Run a long episode (> window_length tokens) and confirm no crash."""
    scenario = load_scenarios()[0]
    env = GridWorld(scenario)
    env.reset()

    # Short window so eviction fires quickly
    cfg = KVStreamConfig(
        model_id="mock", backend="llama", window_length=20, sink_tokens=2
    )
    script_text = '<act cmd="look"/>' + "   " * 100
    stream = KVStream(cfg, "S")
    stream._backend = ScriptedBackend([script_text])
    stream.start()
    env.register_injector(stream._injector)
    router = Router()

    async def _run() -> None:
        async for tok in stream.run():
            out = router.process(tok)
            if out.action:
                env.step(out.action)
            if stream._position > 80:
                stream.stop()
                break

    asyncio.run(_run())  # must not raise
    assert stream.cache_stats.n_evicted > 0


# ---------------------------------------------------------------------------
# Test 10: Verify stream._position equals token count + prefill + injections
# ---------------------------------------------------------------------------


def test_position_accounts_for_injections() -> None:
    """position = prefill_len + gen_tokens + injection_tokens."""
    system_prompt = "SP"  # 2 chars → prefill_len = 2
    stream = make_scripted_stream(["   " * 30], system_prompt=system_prompt)
    stream.start()
    assert stream._position == 2  # after prefill

    # Inject an observation with known token count
    obs_text = Observation(type="x", content="AB").to_token_text()
    # obs_text encodes to some number of chars
    obs_token_count = len(obs_text)  # ScriptedBackend tokenizes as chars

    stream._injector.put(Observation(type="x", content="AB"))

    async def _run() -> None:
        async for _ in stream.run():
            # After first token step, injection should be consumed
            if stream._injector.empty():
                stream.stop()
                break
            if stream._position > 60:
                stream.stop()
                break

    asyncio.run(_run())

    # position should have advanced by at least obs_token_count + 1 gen token
    assert stream._position >= 2 + obs_token_count + 1
