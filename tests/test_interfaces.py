"""Tests for core interfaces and dataclasses."""

import pytest
from streamagent.engine.interfaces import (
    Action,
    BackendProtocol,
    CacheStats,
    Environment,
    Observation,
    ObsInjectorProtocol,
    RouterOutput,
    RouterState,
    Token,
)


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


def test_token_immutable():
    t = Token(id=42, text="hello")
    with pytest.raises(Exception):  # frozen dataclass
        t.id = 99  # type: ignore[misc]


def test_token_defaults():
    t = Token(id=1, text="x")
    assert t.log_prob == 0.0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


def test_action_fields():
    a = Action(command="move north", raw='<act cmd="move north"/>')
    assert a.command == "move north"
    assert "move north" in a.raw


def test_action_params_default_none():
    a = Action(command="look", raw='<act cmd="look"/>')
    assert a.params is None


def test_action_params_dict():
    a = Action(command="pick", raw='<act cmd="pick" obj="apple"/>', params={"obj": "apple"})
    assert a.params == {"obj": "apple"}


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def test_observation_to_token_text():
    obs = Observation(type="gridworld", content="You are in a room.")
    text = obs.to_token_text()
    assert text.startswith('<obs type="gridworld">')
    assert "You are in a room." in text
    assert text.endswith("</obs>")


def test_observation_immutable():
    obs = Observation(type="gridworld", content="hi")
    with pytest.raises(Exception):
        obs.type = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------


def test_cache_stats_utilization():
    stats = CacheStats(
        capacity=100,
        n_sinks=4,
        n_pinned=10,
        n_rolling=50,
        n_used=64,
    )
    assert stats.utilization == pytest.approx(0.64)


def test_cache_stats_utilization_zero_capacity():
    stats = CacheStats(capacity=0, n_sinks=0, n_pinned=0, n_rolling=0, n_used=0)
    assert stats.utilization == 0.0


def test_cache_stats_rolling_capacity():
    stats = CacheStats(capacity=512, n_sinks=4, n_pinned=32, n_rolling=400, n_used=436)
    assert stats.rolling_capacity == 512 - 4 - 32


# ---------------------------------------------------------------------------
# RouterState / RouterOutput
# ---------------------------------------------------------------------------


def test_router_states_distinct():
    states = list(RouterState)
    assert len(states) == len(set(states))


def test_router_output_defaults():
    t = Token(id=5, text="a")
    out = RouterOutput(token=t, state=RouterState.PASSTHROUGH)
    assert out.action is None
    assert out.obs_complete is False
    assert out.timeout_reset is False


def test_router_output_with_action():
    t = Token(id=10, text="/>")
    action = Action(command="look", raw='<act cmd="look"/>')
    out = RouterOutput(token=t, state=RouterState.PASSTHROUGH, action=action)
    assert out.action is not None
    assert out.action.command == "look"


# ---------------------------------------------------------------------------
# ABCs cannot be instantiated
# ---------------------------------------------------------------------------


def test_environment_is_abstract():
    with pytest.raises(TypeError):
        Environment()  # type: ignore[abstract]


def test_obs_injector_is_abstract():
    with pytest.raises(TypeError):
        ObsInjectorProtocol()  # type: ignore[abstract]


def test_backend_is_abstract():
    with pytest.raises(TypeError):
        BackendProtocol()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete stub satisfies Environment ABC
# ---------------------------------------------------------------------------


class _StubEnv(Environment):
    def reset(self) -> Observation:
        return Observation(type="test", content="start")

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        return Observation(type="test", content="ok"), 1.0, False

    def register_injector(self, injector: ObsInjectorProtocol) -> None:
        pass

    def render(self) -> str:
        return "stub"


def test_concrete_env_instantiates():
    env = _StubEnv()
    obs = env.reset()
    assert obs.type == "test"
    obs2, reward, done = env.step(Action(command="noop", raw=""))
    assert reward == 1.0
    assert not done
    assert env.render() == "stub"
