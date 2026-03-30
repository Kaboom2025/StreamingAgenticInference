"""Unit tests for CalcEnv."""

from __future__ import annotations

import pytest

from streamagent.engine.interfaces import Action, Observation
from streamagent.env.calc_env import CalcEnv, _safe_eval


# ---------------------------------------------------------------------------
# _safe_eval
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_addition(self) -> None:
        assert _safe_eval("3 + 4") == 7.0

    def test_subtraction(self) -> None:
        assert _safe_eval("10 - 3") == 7.0

    def test_multiplication(self) -> None:
        assert _safe_eval("23 * 7") == 161.0

    def test_division(self) -> None:
        assert _safe_eval("10 / 4") == pytest.approx(2.5)

    def test_floor_division(self) -> None:
        assert _safe_eval("10 // 3") == 3.0

    def test_modulo(self) -> None:
        assert _safe_eval("10 % 3") == 1.0

    def test_power(self) -> None:
        assert _safe_eval("2 ** 8") == 256.0

    def test_unary_negation(self) -> None:
        assert _safe_eval("-5") == -5.0

    def test_unary_positive(self) -> None:
        assert _safe_eval("+3") == 3.0

    def test_nested_expression(self) -> None:
        assert _safe_eval("(2 + 3) * 4") == 20.0

    def test_float_literal(self) -> None:
        assert _safe_eval("1.5 + 0.5") == pytest.approx(2.0)

    def test_zero_division_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            _safe_eval("1 / 0")

    def test_variable_access_raises(self) -> None:
        with pytest.raises(ValueError):
            _safe_eval("x + 1")

    def test_function_call_raises(self) -> None:
        with pytest.raises((ValueError, AttributeError)):
            _safe_eval("abs(-1)")

    def test_string_literal_raises(self) -> None:
        with pytest.raises(ValueError):
            _safe_eval("'hello'")

    def test_import_raises(self) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            _safe_eval("__import__('os')")


# ---------------------------------------------------------------------------
# CalcEnv.reset
# ---------------------------------------------------------------------------


class TestCalcEnvReset:
    def test_reset_returns_observation(self) -> None:
        env = CalcEnv("What is 2+2?")
        obs = env.reset()
        assert isinstance(obs, Observation)

    def test_reset_obs_type(self) -> None:
        env = CalcEnv("What is 2+2?")
        obs = env.reset()
        assert obs.type == "problem"

    def test_reset_obs_content_contains_problem(self) -> None:
        env = CalcEnv("What is 2+2?")
        obs = env.reset()
        assert "2+2" in obs.content

    def test_reset_clears_done(self) -> None:
        env = CalcEnv("What is 2+2?")
        env.reset()
        env.step(Action(command="eval 2+2", raw=""))
        env.reset()
        assert env._done is False

    def test_reset_clears_last_result(self) -> None:
        env = CalcEnv("What is 2+2?")
        env.reset()
        env.step(Action(command="eval 2+2", raw=""))
        env.reset()
        assert env._last_result is None


# ---------------------------------------------------------------------------
# CalcEnv.step — success cases
# ---------------------------------------------------------------------------


class TestCalcEnvStep:
    def test_eval_returns_correct_result(self) -> None:
        env = CalcEnv("What is 23*7?")
        env.reset()
        obs, reward, done = env.step(Action(command="eval 23*7", raw=""))
        assert obs.content == "161"

    def test_eval_reward_on_success(self) -> None:
        env = CalcEnv("")
        env.reset()
        _, reward, _ = env.step(Action(command="eval 5+5", raw=""))
        assert reward == pytest.approx(1.0)

    def test_eval_done_on_success(self) -> None:
        env = CalcEnv("")
        env.reset()
        _, _, done = env.step(Action(command="eval 5+5", raw=""))
        assert done is True

    def test_eval_result_type(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, _, _ = env.step(Action(command="eval 5+5", raw=""))
        assert obs.type == "result"

    def test_eval_stores_last_result(self) -> None:
        env = CalcEnv("")
        env.reset()
        env.step(Action(command="eval 12*12", raw=""))
        assert env._last_result == "144"

    def test_float_result_formatted(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, _, _ = env.step(Action(command="eval 7/2", raw=""))
        assert obs.content == "3.5"

    def test_integer_result_no_decimal(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, _, _ = env.step(Action(command="eval 6*6", raw=""))
        assert "." not in obs.content
        assert obs.content == "36"

    def test_eval_with_spaces_in_expr(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, _, _ = env.step(Action(command="eval 3 + 4", raw=""))
        assert obs.content == "7"

    def test_power_operator(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, _, _ = env.step(Action(command="eval 2**10", raw=""))
        assert obs.content == "1024"


# ---------------------------------------------------------------------------
# CalcEnv.step — error cases
# ---------------------------------------------------------------------------


class TestCalcEnvStepErrors:
    def test_division_by_zero_returns_error_obs(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, reward, done = env.step(Action(command="eval 1/0", raw=""))
        assert obs.type == "error"
        assert reward == pytest.approx(-1.0)
        assert done is True

    def test_invalid_expression_returns_error(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, reward, done = env.step(Action(command="eval not_a_number", raw=""))
        assert obs.type == "error"
        assert reward == pytest.approx(-1.0)

    def test_unknown_command_returns_error(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, reward, done = env.step(Action(command="move north", raw=""))
        assert obs.type == "error"
        assert reward == pytest.approx(-1.0)
        assert done is True

    def test_empty_eval_returns_error(self) -> None:
        env = CalcEnv("")
        env.reset()
        obs, _, done = env.step(Action(command="eval", raw=""))
        assert obs.type == "error"
        assert done is True


# ---------------------------------------------------------------------------
# CalcEnv.register_injector — obs are forwarded
# ---------------------------------------------------------------------------


class TestCalcEnvInjector:
    def test_injector_receives_result_obs(self) -> None:
        from streamagent.engine.injector import ObsInjector

        env = CalcEnv("")
        env.reset()
        injector = ObsInjector()
        env.register_injector(injector)

        env.step(Action(command="eval 6*6", raw=""))
        pending = injector.get_pending()
        assert len(pending) == 1
        assert pending[0].content == "36"

    def test_injector_receives_error_obs(self) -> None:
        from streamagent.engine.injector import ObsInjector

        env = CalcEnv("")
        env.reset()
        injector = ObsInjector()
        env.register_injector(injector)

        env.step(Action(command="eval 1/0", raw=""))
        pending = injector.get_pending()
        assert len(pending) == 1
        assert pending[0].type == "error"

    def test_no_injector_does_not_raise(self) -> None:
        env = CalcEnv("")
        env.reset()
        # No injector registered — must not raise
        env.step(Action(command="eval 2+2", raw=""))


# ---------------------------------------------------------------------------
# CalcEnv.render
# ---------------------------------------------------------------------------


class TestCalcEnvRender:
    def test_render_unsolved(self) -> None:
        env = CalcEnv("What is 2+2?")
        env.reset()
        rendered = env.render()
        assert "What is 2+2?" in rendered
        assert "unsolved" in rendered

    def test_render_after_solve(self) -> None:
        env = CalcEnv("What is 2+2?")
        env.reset()
        env.step(Action(command="eval 2+2", raw=""))
        rendered = env.render()
        assert "4" in rendered
