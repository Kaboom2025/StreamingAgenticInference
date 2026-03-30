"""ALFWorld environment wrapper conforming to the Environment ABC.

Wraps alfworld.env.AlfredTWEnv (text-world batch env) as a single-episode
Environment. alfworld and yaml are imported lazily inside reset() so the
module can be imported and tested without either package installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from streamagent.engine.interfaces import (
    Action,
    Environment,
    Observation,
    ObsInjectorProtocol,
)

if TYPE_CHECKING:
    pass

FAILURE_STRINGS: list[str] = [
    "Nothing happens.",
    "You can't",
    "That's not something you can",
    "There is no",
    "I can't see",
]


class ALFWorldEnv(Environment):
    """Single-episode ALFWorld text-world environment.

    Wraps alfworld's AlfredTWEnv batch interface (batch size = 1).
    alfworld and yaml are imported lazily on first reset() call.

    Args:
        config_path: Path to the alfworld YAML config file.
        split: Dataset split to evaluate on.
    """

    def __init__(
        self,
        config_path: str,
        split: str = "eval_out_of_distribution",
    ) -> None:
        self._config_path = config_path
        self._split = split
        self._env: object | None = None
        self._current_obs: str = ""
        self._injector: ObsInjectorProtocol | None = None

    # ------------------------------------------------------------------
    # Environment ABC
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Load alfworld lazily and reset to the initial state."""
        if self._env is None:
            import alfworld.env  # type: ignore[import]
            config = self._load_config()
            self._env = alfworld.env.AlfredTWEnv(config, self._split)

        obs_list, _ = self._env.reset()  # type: ignore[union-attr]
        self._current_obs = obs_list[0]
        return Observation(type="alfworld", content=self._current_obs)

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Execute action and return (observation, reward, done).

        Reward is 0.0 when the observation contains a failure string, otherwise
        the reward from the underlying environment.
        """
        alfworld_action = self._map_action(action)
        obs_list, reward_list, done_list, _ = self._env.step(alfworld_action)  # type: ignore[union-attr]
        self._current_obs = obs_list[0]
        reward = float(reward_list[0])
        done = bool(done_list[0])

        if any(fs in self._current_obs for fs in FAILURE_STRINGS):
            reward = 0.0

        return Observation(type="alfworld", content=self._current_obs), reward, done

    def register_injector(self, injector: ObsInjectorProtocol) -> None:
        """Store the injector handle for mid-stream observation injection."""
        self._injector = injector

    def render(self) -> str:
        """Return the current observation text."""
        return self._current_obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_action(self, action: Action) -> str:
        """Map a structured <act> tag to an ALFWorld natural-language action string.

        ALFWorld expects strings like "go to microwave" or "take cup from table".
        Unknown commands fall back to "look" (safe, information-gathering).

        Args:
            action: Parsed action from the token router.

        Returns:
            ALFWorld action string.
        """
        cmd = action.command
        p = action.params or {}

        mapping: dict[str, object] = {
            "goto":      lambda: f"go to {p['obj']}",
            "take":      lambda: f"take {p['obj']} from {p.get('from', 'it')}",
            "put":       lambda: f"put {p['obj']} in/on {p.get('on', p.get('in', 'it'))}",
            "open":      lambda: f"open {p['obj']}",
            "close":     lambda: f"close {p['obj']}",
            "heat":      lambda: f"heat {p['obj']} with {p.get('with', 'microwave')}",
            "cool":      lambda: f"cool {p['obj']} with {p.get('with', 'fridge')}",
            "clean":     lambda: f"clean {p['obj']} with {p.get('with', 'sink')}",
            "examine":   lambda: f"examine {p['obj']}",
            "toggle":    lambda: f"use {p['obj']}",
            "look":      lambda: "look",
            "inventory": lambda: "inventory",
            "done":      lambda: "DONE",
        }

        fn = mapping.get(cmd)
        if fn is None:
            return "look"
        return fn()  # type: ignore[operator]

    def _load_config(self) -> dict:  # type: ignore[type-arg]
        """Load alfworld YAML config from disk.

        Returns:
            Parsed config dict.
        """
        import yaml  # type: ignore[import]
        with open(self._config_path) as f:
            return yaml.safe_load(f)
