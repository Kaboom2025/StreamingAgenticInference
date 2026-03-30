"""Tests for streamagent.env.alfworld_env.ALFWorldEnv.

Uses sys.modules injection to mock alfworld and yaml so tests run without
either package installed.
"""

import sys
import types

import pytest

from streamagent.engine.interfaces import Action, Observation


# ---------------------------------------------------------------------------
# Mock alfworld + yaml before importing ALFWorldEnv
# ---------------------------------------------------------------------------

def _install_mocks() -> None:
    """Inject fake alfworld and yaml modules, and patch open() for missing configs."""
    import builtins
    import io
    import os

    _real_open = builtins.open

    def _patched_open(path: object, *args: object, **kwargs: object) -> object:
        # Intercept non-existent YAML config paths used in tests.
        if str(path).endswith(".yaml") and not os.path.exists(str(path)):
            return io.StringIO("")
        return _real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    builtins.open = _patched_open  # type: ignore[assignment]

    # yaml — only inject a stub if the real package is not installed
    if "yaml" not in sys.modules:
        try:
            import importlib.util as _ilu
            if _ilu.find_spec("yaml") is None:
                yaml_mod = types.ModuleType("yaml")
                yaml_mod.safe_load = lambda f: {}  # type: ignore[attr-defined]
                sys.modules["yaml"] = yaml_mod
        except Exception:
            pass

    # alfworld
    alfworld_mod = types.ModuleType("alfworld")
    env_mod = types.ModuleType("alfworld.env")

    class _MockBatchEnv:
        def __init__(self, config: object, train_eval: str) -> None:
            self._obs_seq = [
                "You are in a kitchen. There is an apple on the counter.",
                "Nothing happens.",
                "You take the apple.",
            ]
            self._idx = 0
            self._done = False

        def __len__(self) -> int:
            return 1

        def reset(self) -> tuple[list[str], dict]:
            self._idx = 0
            self._done = False
            return [self._obs_seq[0]], {"won": [False]}

        def step(self, action: str) -> tuple[list[str], list[float], list[bool], dict]:
            self._idx += 1
            obs = self._obs_seq[min(self._idx, len(self._obs_seq) - 1)]
            done = self._idx >= len(self._obs_seq) - 1
            self._done = done
            return [obs], [1.0 if done else 0.0], [done], {"won": [done]}

    env_mod.AlfredTWEnv = _MockBatchEnv  # type: ignore[attr-defined]
    alfworld_mod.env = env_mod  # type: ignore[attr-defined]
    sys.modules.setdefault("alfworld", alfworld_mod)
    sys.modules.setdefault("alfworld.env", env_mod)


_install_mocks()

from streamagent.env.alfworld_env import ALFWorldEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeInjector:
    def __init__(self) -> None:
        self.injected: list[Observation] = []

    def put(self, obs: Observation) -> None:
        self.injected.append(obs)

    def get_pending(self) -> list[Observation]:
        items = list(self.injected)
        self.injected.clear()
        return items

    def empty(self) -> bool:
        return len(self.injected) == 0


def _act(cmd: str, **params: str) -> Action:
    raw = f'<act cmd="{cmd}"' + "".join(f' {k}="{v}"' for k, v in params.items()) + "/>"
    return Action(command=cmd, raw=raw, params=params or None)


# ---------------------------------------------------------------------------
# Environment ABC tests
# ---------------------------------------------------------------------------

def test_reset_returns_observation() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.type == "alfworld"
    assert len(obs.content) > 0


def test_step_returns_tuple() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    env.reset()
    obs, reward, done = env.step(_act("goto", obj="kitchen"))
    assert isinstance(obs, Observation)
    assert obs.type == "alfworld"
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_failure_string_zeroes_reward() -> None:
    """step() returning a FAILURE_STRING obs must have reward=0.0."""
    env = ALFWorldEnv(config_path="fake/config.yaml")
    env.reset()
    obs, reward, done = env.step(_act("goto", obj="kitchen"))
    assert "Nothing happens." in obs.content
    assert reward == 0.0


def test_success_sets_done() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    env.reset()
    env.step(_act("goto", obj="kitchen"))       # step 1 → failure
    _, _, done = env.step(_act("take", obj="apple", **{"from": "counter"}))
    assert done is True


def test_register_injector() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    env.register_injector(_FakeInjector())


def test_render_after_reset() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    env.reset()
    assert len(env.render()) > 0


# ---------------------------------------------------------------------------
# _map_action tests
# ---------------------------------------------------------------------------

def test_map_goto() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("goto", obj="microwave")) == "go to microwave"


def test_map_take() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("take", obj="cup", **{"from": "table"})) == "take cup from table"


def test_map_take_no_from() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("take", obj="cup")) == "take cup from it"


def test_map_put() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("put", obj="cup", on="counter")) == "put cup in/on counter"


def test_map_open() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("open", obj="fridge")) == "open fridge"


def test_map_close() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("close", obj="fridge")) == "close fridge"


def test_map_heat() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("heat", obj="cup", **{"with": "microwave"})) == "heat cup with microwave"


def test_map_cool() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("cool", obj="cup", **{"with": "fridge"})) == "cool cup with fridge"


def test_map_clean() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("clean", obj="cup", **{"with": "sink"})) == "clean cup with sink"


def test_map_examine() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("examine", obj="cup")) == "examine cup"


def test_map_look() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("look")) == "look"


def test_map_inventory() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("inventory")) == "inventory"


def test_map_done() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("done")) == "DONE"


def test_map_unknown_falls_back_to_look() -> None:
    env = ALFWorldEnv(config_path="fake/config.yaml")
    assert env._map_action(_act("frobnicate", obj="something")) == "look"
