"""Eval harness for recovery-latency measurement.

Runs a single episode with any KVStream-compatible backend and records a
RecoveryEvent for each observation→act pair, measuring latency in tokens.

Chunking baseline is computed analytically from streaming results: given
chunk_size C and a streaming injection at position P, the chunking model
would not see the injection until the next boundary at P + (-P % C), adding
that delay to the streaming recovery count.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from streamagent.engine.interfaces import Action
from streamagent.engine.kv_stream import KVStream
from streamagent.engine.router import Router
from streamagent.env.gridworld import GridWorld
from streamagent.eval.metrics import EpisodeMetrics, RecoveryEvent

if TYPE_CHECKING:
    from streamagent.env.alfworld_env import ALFWorldEnv

# Sentinel type alias
_PendingInjection = tuple[int, float, str]   # (position, wall_time, obs_content)


async def _run_episode_async(
    stream: KVStream,
    env: GridWorld,
    max_tokens: int,
) -> EpisodeMetrics:
    """Core async episode loop.

    Wires KVStream → Router → GridWorld, injecting each env observation back
    into the stream and recording injection/act positions for latency metrics.

    Args:
        stream: Initialised KVStream (start() already called, or will be called
            by run() on first iteration).
        env: GridWorld instance (not yet reset).
        max_tokens: Hard cap on generated tokens to prevent infinite loops.

    Returns:
        EpisodeMetrics with one RecoveryEvent per (injection, act) pair.
    """
    router = Router()
    recovery_events: list[RecoveryEvent] = []
    pending: Optional[_PendingInjection] = None
    solved = False
    token_count = 0

    # Inject the initial observation so the model knows its starting state.
    obs = env.reset()
    inj_pos = stream.position
    stream.inject(obs.to_token_text())
    pending = (inj_pos, time.perf_counter(), obs.content)

    async for token in stream.run():
        token_count += 1
        output = router.process(token)

        if output.action is not None:
            act_pos = stream.position
            act_t = time.perf_counter()

            # Pair this act with the most recent pending injection.
            if pending is not None:
                inj_p, inj_t, obs_content = pending
                recovery_events.append(
                    RecoveryEvent(
                        injection_position=inj_p,
                        act_position=act_pos,
                        action=output.action.command,
                        obs_content=obs_content,
                        injection_time=inj_t,
                        act_time=act_t,
                    )
                )
                pending = None

            # Step the env and enqueue the resulting observation.
            obs, reward, done = env.step(output.action)
            inj_pos = stream.position
            stream.inject(obs.to_token_text())
            pending = (inj_pos, time.perf_counter(), obs.content)

            if done:
                solved = reward > 0.0
                break

        if token_count >= max_tokens:
            break

    stream.stop()

    return EpisodeMetrics(
        scenario_name=env.scenario.name,
        total_tokens=token_count,
        solved=solved,
        recovery_events=recovery_events,
    )


async def _run_alfworld_episode_async(
    stream: KVStream,
    env: "ALFWorldEnv",
    task_type: str = "",
    max_tokens: int = 4000,
) -> EpisodeMetrics:
    """Core async ALFWorld episode loop.

    Sets is_failure on each RecoveryEvent when the triggering observation
    contains a FAILURE_STRING, then calls the recovery judge to set is_correct.

    Args:
        stream: Initialised KVStream (backend injected for testing).
        env: ALFWorldEnv instance (not yet reset).
        task_type: ALFWorld task category (e.g. "pick_heat") stored in metrics.
        max_tokens: Hard cap on generated tokens.

    Returns:
        EpisodeMetrics with is_failure/is_correct fields populated.
    """
    from streamagent.env.alfworld_env import FAILURE_STRINGS
    from streamagent.eval.alfworld_judge import is_correct_recovery

    router = Router()
    recovery_events: list[RecoveryEvent] = []
    pending: Optional[_PendingInjection] = None
    pending_is_failure: bool = False
    last_action: Optional[Action] = None
    solved = False
    token_count = 0

    obs = env.reset()
    inj_pos = stream.position
    stream.inject(obs.to_token_text())
    pending = (inj_pos, time.perf_counter(), obs.content)

    async for token in stream.run():
        token_count += 1
        output = router.process(token)

        if output.action is not None:
            act_pos = stream.position
            act_t = time.perf_counter()
            current_action = output.action

            if pending is not None:
                inj_p, inj_t, obs_content = pending
                is_failure = pending_is_failure

                is_correct: Optional[bool] = None
                if is_failure and last_action is not None:
                    is_correct = is_correct_recovery(obs_content, current_action, last_action)

                recovery_events.append(
                    RecoveryEvent(
                        injection_position=inj_p,
                        act_position=act_pos,
                        action=current_action.command,
                        obs_content=obs_content,
                        injection_time=inj_t,
                        act_time=act_t,
                        is_failure=is_failure,
                        is_correct=is_correct,
                    )
                )
                pending = None

            last_action = current_action

            obs, reward, done = env.step(current_action)
            inj_pos = stream.position
            stream.inject(obs.to_token_text())
            pending = (inj_pos, time.perf_counter(), obs.content)
            pending_is_failure = any(fs in obs.content for fs in FAILURE_STRINGS)

            if done:
                solved = reward > 0.0
                break

        if token_count >= max_tokens:
            break

    stream.stop()

    return EpisodeMetrics(
        scenario_name="alfworld",
        total_tokens=token_count,
        solved=solved,
        task_type=task_type,
        recovery_events=recovery_events,
    )


def run_alfworld_episode(
    stream: KVStream,
    env: "ALFWorldEnv",
    task_type: str = "",
    max_tokens: int = 4000,
) -> EpisodeMetrics:
    """Run a single ALFWorld episode and return recovery-latency metrics.

    Synchronous entry point; internally runs an asyncio event loop.

    Args:
        stream: KVStream backed by any BackendProtocol implementation.
        env: ALFWorldEnv to interact with.
        task_type: ALFWorld task category string (e.g. "pick_heat").
        max_tokens: Hard cap on generated tokens per episode.

    Returns:
        EpisodeMetrics with is_failure/is_correct fields populated.
    """
    return asyncio.run(_run_alfworld_episode_async(stream, env, task_type, max_tokens))


def run_gridworld_episode(
    stream: KVStream,
    env: GridWorld,
    max_tokens: int = 2000,
) -> EpisodeMetrics:
    """Run a GridWorld episode and return recovery-latency metrics.

    Synchronous entry point; internally runs an asyncio event loop.

    Args:
        stream: KVStream backed by any BackendProtocol implementation.
            The stream must not have been started yet (or may be started with
            a backend already injected for testing).
        env: GridWorld to navigate.
        max_tokens: Hard cap on generated tokens per episode.

    Returns:
        EpisodeMetrics containing per-event recovery_tokens and episode-level
        aggregates. Call metrics.mean_chunking_recovery_tokens(chunk_size) to
        compare against the chunking baseline.

    Example::

        stream = KVStream(config, system_prompt)
        stream._backend = ScriptedBackend(script)   # inject mock for tests
        stream._started = True

        env = GridWorld(load_scenarios()[0])
        metrics = run_gridworld_episode(stream, env, max_tokens=500)

        print(f"mean recovery: {metrics.mean_recovery_tokens:.1f} tokens")
        print(f"vs chunk-32:   {metrics.mean_chunking_recovery_tokens(32):.1f} tokens")
    """
    return asyncio.run(_run_episode_async(stream, env, max_tokens))
