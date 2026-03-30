"""Recovery judge for ALFWorld episodes.

Classifies whether an action after a failure constitutes a valid recovery.
"""

from __future__ import annotations

from streamagent.engine.interfaces import Action


def is_correct_recovery(
    failure_obs: str,
    recovery_action: Action,
    prior_action: Action,
) -> bool | None:
    """Classify whether recovery_action is a valid response to a failure.

    Rules (evaluated in order):
    1. Same command AND same params as prior → False (repeating the failed action)
    2. command in ("look", "inventory", "examine") → True (info-gathering)
    3. command == "goto" with a different destination obj → True (spatial pivot)
    4. Anything else → None (ambiguous — excluded from recovery rate metric)

    Args:
        failure_obs: The failure observation text (reserved for future rule
            extensions; current rules do not use it).
        recovery_action: The action the agent takes after the failure.
        prior_action: The action that caused the failure observation.

    Returns:
        True if definitely a recovery, False if definitely not, None if ambiguous.
    """
    # Rule 1: exact repeat
    if (recovery_action.command == prior_action.command and
            recovery_action.params == prior_action.params):
        return False

    # Rule 2: information-gathering commands
    if recovery_action.command in ("look", "inventory", "examine"):
        return True

    # Rule 3: going somewhere different
    if recovery_action.command == "goto":
        prior_obj = (prior_action.params or {}).get("obj", "")
        recovery_obj = (recovery_action.params or {}).get("obj", "")
        return recovery_obj != prior_obj

    return None
