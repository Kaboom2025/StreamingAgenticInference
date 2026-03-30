"""CalcEnv — minimal single-tool environment for end-to-end demos.

The agent receives a math expression problem and has one tool:

    <act cmd="eval 23*7"/>

The environment evaluates the expression and injects the result as an
observation.  The episode terminates after one successful eval (or one error).

This is the simplest possible environment that exercises the full pipeline:
  KVStream → Router (parses <act>) → CalcEnv → ObsInjector → KVStream
"""

from __future__ import annotations

import ast
import operator
from typing import Optional

from streamagent.engine.interfaces import (
    Action,
    Environment,
    Observation,
    ObsInjectorProtocol,
)

# Whitelist of safe operators for eval
_SAFE_OPS: dict = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a numeric expression without using Python's eval().

    Only supports arithmetic on numeric literals — no variable access,
    no function calls, no imports.

    Raises:
        ValueError: if the expression contains unsupported constructs.
    """
    def _eval(node: ast.expr) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported literal: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
            return op_fn(_eval(node.operand))
        raise ValueError(f"Unsupported node: {type(node).__name__}")

    tree = ast.parse(expr.strip(), mode="eval")
    return _eval(tree.body)


def _format_result(value: float) -> str:
    """Return int string when value is whole, float string otherwise."""
    if value == int(value):
        return str(int(value))
    return str(value)


class CalcEnv(Environment):
    """Single-tool calculator environment.

    The agent calls ``<act cmd="eval <expr>"/>`` once to compute a numeric
    expression.  The result is injected as an observation and the episode ends.

    Attributes:
        problem: The math problem string shown to the agent in the system prompt.
    """

    def __init__(self, problem: str) -> None:
        """Initialize CalcEnv with a problem description.

        Args:
            problem: Natural-language math problem, e.g. "What is 23 * 7?".
        """
        self.problem = problem
        self._injector: Optional[ObsInjectorProtocol] = None
        self._done = False
        self._last_result: Optional[str] = None

    # ------------------------------------------------------------------
    # Environment ABC
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset episode state."""
        self._done = False
        self._last_result = None
        return Observation(type="problem", content=self.problem)

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Evaluate the expression in the action command.

        Expected command format: ``eval <expr>``
        e.g. ``eval 23*7``

        Args:
            action: Action with command string.

        Returns:
            Tuple of (observation, reward, done).
            reward=1.0 on success, -1.0 on error.
        """
        raw = action.command.strip()
        if not raw.startswith("eval"):
            obs = Observation(type="error", content=f"unknown command: {raw!r}")
            self._inject(obs)
            return obs, -1.0, True

        expr = raw[len("eval"):].strip()

        try:
            value = _safe_eval(expr)
            result_str = _format_result(value)
            self._last_result = result_str
            obs = Observation(type="result", content=result_str)
            self._inject(obs)
            self._done = True
            return obs, 1.0, True
        except (ValueError, ZeroDivisionError, SyntaxError) as exc:
            obs = Observation(type="error", content=str(exc))
            self._inject(obs)
            self._done = True
            return obs, -1.0, True

    def register_injector(self, injector: ObsInjectorProtocol) -> None:
        """Register injector to receive observations."""
        self._injector = injector

    def render(self) -> str:
        """Return current state as a string."""
        if self._last_result is not None:
            return f"Problem: {self.problem}  Result: {self._last_result}"
        return f"Problem: {self.problem}  (unsolved)"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _inject(self, obs: Observation) -> None:
        if self._injector is not None:
            self._injector.put(obs)
