"""Evaluation metrics for StreamAgent episodes."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class RecoveryEvent:
    """Records one injection→act pair for latency measurement.

    injection_position: stream position when inject() was called (before obs tokens
        are consumed). This is the moment the environment's response arrived.
    act_position: stream position when the <act .../> tag completed.
    recovery_tokens: act_position - injection_position — the number of forward
        passes the model needed from injection enqueue to completed act tag.
        Includes obs injection tokens + think tokens + act tag tokens.
    is_failure: True when the triggering observation contained a failure string.
    is_correct: Result from the recovery judge (True/False/None = ambiguous).
    """

    injection_position: int
    act_position: int
    action: str          # command from <act cmd="..."/>
    obs_content: str     # raw observation text that triggered this event
    injection_time: float = 0.0   # wall-clock seconds (time.perf_counter)
    act_time: float = 0.0
    is_failure: bool = False
    is_correct: Optional[bool] = None   # set by judge after the fact

    @property
    def recovery_tokens(self) -> int:
        """Tokens from injection enqueue to act completion."""
        return self.act_position - self.injection_position

    def chunking_recovery_tokens(self, chunk_size: int) -> int:
        """Simulated chunking baseline recovery tokens.

        Chunking can only inject at boundaries aligned to chunk_size.
        Adds the delay (tokens until next boundary) to the streaming value.

        Args:
            chunk_size: Fixed generation chunk length for the baseline.

        Returns:
            recovery_tokens + delay_to_next_boundary
        """
        delay = (-self.injection_position) % chunk_size
        return self.recovery_tokens + delay


@dataclass
class EpisodeMetrics:
    """Aggregate metrics for one episode."""

    scenario_name: str
    total_tokens: int
    solved: bool
    task_type: str = ""   # ALFWorld task type, e.g. "pick_heat"
    recovery_events: list[RecoveryEvent] = field(default_factory=list)

    @property
    def had_failures(self) -> bool:
        """True if at least one failure observation was injected this episode."""
        return any(e.is_failure for e in self.recovery_events)

    @property
    def mean_recovery_tokens(self) -> float:
        """Mean streaming recovery latency in tokens."""
        if not self.recovery_events:
            return 0.0
        return statistics.mean(e.recovery_tokens for e in self.recovery_events)

    @property
    def median_recovery_tokens(self) -> float:
        """Median streaming recovery latency in tokens."""
        if not self.recovery_events:
            return 0.0
        return statistics.median(e.recovery_tokens for e in self.recovery_events)

    def mean_chunking_recovery_tokens(self, chunk_size: int) -> float:
        """Mean chunking-baseline recovery latency in tokens.

        Args:
            chunk_size: Fixed chunk length for the baseline simulation.
        """
        if not self.recovery_events:
            return 0.0
        return statistics.mean(
            e.chunking_recovery_tokens(chunk_size) for e in self.recovery_events
        )

    def speedup_vs_chunking(self, chunk_size: int) -> float:
        """Ratio of chunking mean to streaming mean (>1 means streaming is faster).

        Args:
            chunk_size: Fixed chunk length for the baseline simulation.
        """
        stream_mean = self.mean_recovery_tokens
        if stream_mean == 0:
            return 1.0
        return self.mean_chunking_recovery_tokens(chunk_size) / stream_mean


@dataclass
class BenchmarkResult:
    """Aggregate results across a batch of episodes.

    Partitioned into no_failures / with_failures subsets to isolate the
    architecture's advantage (the streaming benefit only appears in episodes
    that had mid-episode failures).
    """

    episodes: list[EpisodeMetrics] = field(default_factory=list)

    @property
    def no_failures(self) -> list[EpisodeMetrics]:
        """Episodes with zero failure observations."""
        return [e for e in self.episodes if not e.had_failures]

    @property
    def with_failures(self) -> list[EpisodeMetrics]:
        """Episodes with at least one failure observation."""
        return [e for e in self.episodes if e.had_failures]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.solved) / len(self.episodes)

    def success_rate_for(self, subset: list[EpisodeMetrics]) -> float:
        if not subset:
            return 0.0
        return sum(1 for e in subset if e.solved) / len(subset)
