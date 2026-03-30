"""Router FSM: Deterministic token router for parsing inline tags.

The Router processes tokens one at a time and detects:
- <act cmd="..."/> tags → returns Action
- <obs type="...">...</obs> tags → returns obs_complete flag
- All other tokens pass through unchanged
"""

import xml.etree.ElementTree as ET
from typing import Optional

from streamagent.engine.interfaces import (
    ACT_CLOSE,
    ACT_OPEN_PREFIX,
    OBS_CLOSE,
    OBS_OPEN_PREFIX,
    OBS_OPEN_SUFFIX,
    Action,
    RouterOutput,
    RouterState,
    Token,
)


class Router:
    """Deterministic FSM for parsing inline tags from token stream."""

    def __init__(self, tag_timeout: int = 50):
        """Initialize the router FSM.

        Args:
            tag_timeout: Number of tokens to wait before aborting MAYBE_TAG state.
        """
        self.tag_timeout = tag_timeout
        self._state = RouterState.PASSTHROUGH
        self._accumulator = ""
        self._token_count = 0

    def process(self, token: Token) -> RouterOutput:
        """Process one token and return the routing decision.

        Args:
            token: The token to process.

        Returns:
            RouterOutput with the next state, action (if complete), and flags.
        """
        if self._state == RouterState.PASSTHROUGH:
            return self._process_passthrough(token)
        elif self._state == RouterState.MAYBE_TAG:
            return self._process_maybe_tag(token)
        elif self._state == RouterState.IN_ACT_TAG:
            return self._process_act_tag(token)
        elif self._state == RouterState.IN_OBS_TAG:
            return self._process_obs_tag(token)
        else:
            # Shouldn't reach here
            return RouterOutput(token=token, state=self._state)

    def _process_passthrough(self, token: Token) -> RouterOutput:
        """Process token in PASSTHROUGH state.

        Transitions to MAYBE_TAG if '<' is seen.
        """
        if "<" in token.text:
            self._state = RouterState.MAYBE_TAG
            self._accumulator = token.text
            self._token_count = 1
            return RouterOutput(token=token, state=RouterState.MAYBE_TAG)
        return RouterOutput(token=token, state=RouterState.PASSTHROUGH)

    def _process_maybe_tag(self, token: Token) -> RouterOutput:
        """Process token in MAYBE_TAG state.

        Classifies whether the tag is an action or observation tag.
        If no classification after tag_timeout tokens, resets to PASSTHROUGH.
        """
        self._accumulator += token.text
        self._token_count += 1

        # Check for timeout first (before classification)
        if self._token_count > self.tag_timeout:
            self._state = RouterState.PASSTHROUGH
            self._accumulator = ""
            self._token_count = 0
            return RouterOutput(
                token=token,
                state=RouterState.PASSTHROUGH,
                timeout_reset=True,
            )

        # Check for action tag start
        if ACT_OPEN_PREFIX in self._accumulator:
            self._state = RouterState.IN_ACT_TAG
            return RouterOutput(token=token, state=RouterState.IN_ACT_TAG)

        # Check for observation tag start
        if OBS_OPEN_PREFIX in self._accumulator:
            self._state = RouterState.IN_OBS_TAG
            return RouterOutput(token=token, state=RouterState.IN_OBS_TAG)

        # Still waiting for classification
        return RouterOutput(token=token, state=RouterState.MAYBE_TAG)

    def _process_act_tag(self, token: Token) -> RouterOutput:
        """Process token in IN_ACT_TAG state.

        Accumulates until '/>' is seen, then parses the action.
        """
        self._accumulator += token.text

        if ACT_CLOSE in self._accumulator:
            # Tag is complete; extract command
            action = self._parse_action_tag(self._accumulator)
            self._state = RouterState.PASSTHROUGH
            self._accumulator = ""
            self._token_count = 0
            return RouterOutput(
                token=token,
                state=RouterState.PASSTHROUGH,
                action=action,
            )

        return RouterOutput(token=token, state=RouterState.IN_ACT_TAG)

    def _process_obs_tag(self, token: Token) -> RouterOutput:
        """Process token in IN_OBS_TAG state.

        Accumulates until '</obs>' is seen, then signals completion.
        """
        self._accumulator += token.text

        if OBS_CLOSE in self._accumulator:
            # Tag is complete
            self._state = RouterState.PASSTHROUGH
            self._accumulator = ""
            self._token_count = 0
            return RouterOutput(
                token=token,
                state=RouterState.PASSTHROUGH,
                obs_complete=True,
            )

        return RouterOutput(token=token, state=RouterState.IN_OBS_TAG)

    def _parse_action_tag(self, tag_text: str) -> Action:
        """Extract command and params from <act cmd="..." .../> tag.

        Uses XML parsing to extract all attributes. Falls back to string-based
        extraction of `cmd` only if the tag is malformed XML.

        Args:
            tag_text: The accumulated tag text.

        Returns:
            Action with the extracted command and optional params dict.
        """
        act_start = tag_text.find("<act ")
        if act_start == -1:
            return Action(command="", raw=tag_text, params=None)

        act_end = tag_text.find("/>", act_start)
        if act_end == -1:
            act_xml = tag_text[act_start:] + "/>"
        else:
            act_xml = tag_text[act_start:act_end + 2]

        try:
            el = ET.fromstring(act_xml)
            attribs = dict(el.attrib)
            cmd = attribs.pop("cmd", "")
            return Action(command=cmd, raw=tag_text, params=attribs or None)
        except ET.ParseError:
            # Fallback: extract cmd value via string search
            start_idx = tag_text.find(ACT_OPEN_PREFIX)
            if start_idx == -1:
                return Action(command="", raw=tag_text, params=None)
            start_idx += len(ACT_OPEN_PREFIX)
            end_idx = tag_text.find('"', start_idx)
            if end_idx == -1:
                end_idx = len(tag_text)
            return Action(command=tag_text[start_idx:end_idx], raw=tag_text, params=None)

    def reset(self) -> None:
        """Reset the FSM to initial state."""
        self._state = RouterState.PASSTHROUGH
        self._accumulator = ""
        self._token_count = 0
