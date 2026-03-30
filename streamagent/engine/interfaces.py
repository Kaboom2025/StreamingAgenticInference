"""Core interfaces, dataclasses, and ABCs for StreamAgent.

All engine components depend on this module. The Environment ABC is the ONLY
communication boundary between engine/ and env/.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Optional


# ---------------------------------------------------------------------------
# Token vocabulary constants (prompt-level only, no fine-tuning required)
# ---------------------------------------------------------------------------

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ACT_OPEN_PREFIX = '<act cmd="'
ACT_CLOSE = "/>"
OBS_OPEN_PREFIX = '<obs type="'
OBS_OPEN_SUFFIX = '">'
OBS_CLOSE = "</obs>"
GOAL_OPEN = "<goal>"
GOAL_CLOSE = "</goal>"
MEM_OPEN = "<mem>"
MEM_CLOSE = "</mem>"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """A single generated token from KVStream."""

    id: int
    text: str
    log_prob: float = 0.0


@dataclass
class Action:
    """Parsed action extracted from <act cmd="..."/> tag."""

    command: str
    raw: str  # full tag text including surrounding tokens
    params: dict[str, str] | None = None  # additional attributes, e.g. {"obj": "cup"}


@dataclass(frozen=True)
class Observation:
    """Observation from environment to be injected into the stream."""

    type: str  # e.g. "gridworld", "alfworld"
    content: str

    def to_token_text(self) -> str:
        """Render as the observation token sequence to inject."""
        return f'{OBS_OPEN_PREFIX}{self.type}{OBS_OPEN_SUFFIX}{self.content}{OBS_CLOSE}'


@dataclass
class CacheStats:
    """Runtime statistics for the SinkCache."""

    capacity: int
    n_sinks: int
    n_pinned: int
    n_rolling: int
    n_used: int
    n_evicted: int = 0

    @property
    def utilization(self) -> float:
        return self.n_used / self.capacity if self.capacity > 0 else 0.0

    @property
    def rolling_capacity(self) -> int:
        return self.capacity - self.n_sinks - self.n_pinned


# ---------------------------------------------------------------------------
# Router types
# ---------------------------------------------------------------------------


class RouterState(Enum):
    """FSM states for the token router."""

    PASSTHROUGH = auto()
    MAYBE_TAG = auto()       # seen '<', waiting to classify
    IN_ACT_TAG = auto()      # inside <act .../>
    IN_OBS_TAG = auto()      # inside <obs ...>...</obs>  (injection path)


@dataclass
class RouterOutput:
    """Result of routing a single token."""

    token: Token
    state: RouterState
    action: Optional[Action] = None       # set when a complete <act/> is parsed
    obs_complete: bool = False             # set when </obs> closes an injection
    timeout_reset: bool = False            # set when 50-token timeout fires


# ---------------------------------------------------------------------------
# Environment ABC  (the ONLY cross-boundary interface)
# ---------------------------------------------------------------------------


class Environment(ABC):
    """Abstract base for all environments.

    Called synchronously from the agent loop; must NOT block for > ~50 ms.
    The injector is registered once before the episode begins.
    """

    @abstractmethod
    def reset(self) -> Observation:
        """Reset to initial state; return the first observation."""
        ...

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Execute action; return (observation, reward, done)."""
        ...

    @abstractmethod
    def register_injector(self, injector: "ObsInjectorProtocol") -> None:
        """Give the environment a handle to push observations into the stream."""
        ...

    @abstractmethod
    def render(self) -> str:
        """Return a human-readable string of the current state."""
        ...


# ---------------------------------------------------------------------------
# ObsInjector protocol (structural typing to avoid circular imports)
# ---------------------------------------------------------------------------


class ObsInjectorProtocol(ABC):
    """Minimal interface that Environment implementations depend on."""

    @abstractmethod
    def put(self, obs: Observation) -> None:
        """Non-blocking enqueue from Environment thread."""
        ...

    @abstractmethod
    def get_pending(self) -> list[Observation]:
        """Drain all pending observations; called by KVStream per token step."""
        ...

    @abstractmethod
    def empty(self) -> bool:
        """True if no pending observations."""
        ...


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class BackendProtocol(ABC):
    """Minimal interface for LLM backends (llama-cpp, HF, MLX)."""

    @abstractmethod
    def load_model(self, model_path: str, **kwargs: object) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Encode text to token ids."""
        ...

    @abstractmethod
    def detokenize(self, ids: list[int]) -> str:
        """Decode token ids to text."""
        ...

    @abstractmethod
    def prefill(self, input_ids: list[int]) -> int:
        """Run prefill pass; return length of KV cache after prefill."""
        ...

    @abstractmethod
    def forward_one(
        self,
        token_id: int,
        cache_position: int,
    ) -> tuple[int, float]:
        """Single forward pass; return (next_token_id, log_prob).

        cache_position MUST increment strictly monotonically across ALL calls
        (including observation injection passes) to maintain RoPE correctness.
        """
        ...

    def inject_one(self, token_id: int, cache_position: int) -> None:
        """Process one forced observation token into the KV cache.

        The return value is intentionally void — the next generated token is
        determined by the subsequent forward_one call, not by this pass.

        Default implementation calls forward_one and discards the result.
        Override in test backends to avoid side-effects on scripted output.

        cache_position MUST increment strictly monotonically (same invariant
        as forward_one).
        """
        self.forward_one(token_id, cache_position)

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Maximum context length supported by the loaded model."""
        ...
