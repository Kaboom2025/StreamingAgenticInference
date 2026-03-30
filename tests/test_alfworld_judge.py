"""Tests for streamagent.eval.alfworld_judge.is_correct_recovery."""

import pytest

from streamagent.engine.interfaces import Action
from streamagent.eval.alfworld_judge import is_correct_recovery

# Helpers — pre-built Action objects
_LOOK    = Action(command="look",      raw='<act cmd="look"/>')
_INV     = Action(command="inventory", raw='<act cmd="inventory"/>')
_EXAMINE = Action(command="examine",   raw='<act cmd="examine" obj="cup"/>', params={"obj": "cup"})
_GOTO_K  = Action(command="goto",      raw='<act cmd="goto" obj="kitchen"/>', params={"obj": "kitchen"})
_GOTO_L  = Action(command="goto",      raw='<act cmd="goto" obj="living room"/>', params={"obj": "living room"})
_TAKE    = Action(command="take",      raw='<act cmd="take" obj="apple"/>', params={"obj": "apple"})
_OPEN    = Action(command="open",      raw='<act cmd="open" obj="fridge"/>', params={"obj": "fridge"})

FAILURE_OBS = "Nothing happens."


def test_repeat_exact_is_false() -> None:
    """Repeating the exact same command+params is never a recovery."""
    assert is_correct_recovery(FAILURE_OBS, _GOTO_K, _GOTO_K) is False


def test_look_is_true() -> None:
    """'look' is always a valid recovery regardless of prior action."""
    assert is_correct_recovery(FAILURE_OBS, _LOOK, _TAKE) is True


def test_inventory_is_true() -> None:
    """'inventory' is always a valid recovery."""
    assert is_correct_recovery(FAILURE_OBS, _INV, _TAKE) is True


def test_examine_is_true() -> None:
    """'examine <obj>' is always a valid recovery."""
    assert is_correct_recovery(FAILURE_OBS, _EXAMINE, _TAKE) is True


def test_goto_different_obj_is_true() -> None:
    """goto with a different destination is a valid pivot."""
    assert is_correct_recovery(FAILURE_OBS, _GOTO_L, _GOTO_K) is True


def test_goto_same_obj_is_false() -> None:
    """goto with same destination — repeat, covered by rule 1."""
    assert is_correct_recovery(FAILURE_OBS, _GOTO_K, _GOTO_K) is False


def test_goto_after_non_goto_prior_is_true() -> None:
    """goto when prior was not a goto is a valid pivot."""
    assert is_correct_recovery(FAILURE_OBS, _GOTO_K, _TAKE) is True


def test_ambiguous_open_is_none() -> None:
    """'open' doesn't match any positive rule → None."""
    assert is_correct_recovery(FAILURE_OBS, _OPEN, _TAKE) is None


def test_failure_obs_not_used_in_logic() -> None:
    """The failure_obs param doesn't change the outcome (reserved for future use)."""
    assert is_correct_recovery("You can't do that.", _LOOK, _TAKE) is True
    assert is_correct_recovery("There is no door.", _OPEN, _TAKE) is None
