"""Unit tests for EpisodeMetrics, RecoveryEvent, and BenchmarkResult dataclasses."""

import pytest

from streamagent.eval.metrics import BenchmarkResult, EpisodeMetrics, RecoveryEvent


# ---------------------------------------------------------------------------
# RecoveryEvent
# ---------------------------------------------------------------------------


def test_recovery_tokens_is_difference():
    event = RecoveryEvent(
        injection_position=100,
        act_position=120,
        action="move north",
        obs_content="Agent at (1, 1).",
    )
    assert event.recovery_tokens == 20


def test_chunking_recovery_tokens_at_boundary():
    # injection_position already on a chunk boundary → no delay added
    event = RecoveryEvent(
        injection_position=64,
        act_position=84,
        action="move east",
        obs_content="obs",
    )
    # 64 % 32 == 0, delay = 0
    assert event.chunking_recovery_tokens(32) == 20


def test_chunking_recovery_tokens_mid_chunk():
    # injection at position 10, chunk_size=32 → delay = 32 - 10 = 22
    event = RecoveryEvent(
        injection_position=10,
        act_position=30,
        action="move south",
        obs_content="obs",
    )
    assert event.chunking_recovery_tokens(32) == 20 + 22


def test_chunking_recovery_always_gte_streaming():
    for inj_pos in range(0, 100, 7):
        event = RecoveryEvent(
            injection_position=inj_pos,
            act_position=inj_pos + 15,
            action="look",
            obs_content="obs",
        )
        assert event.chunking_recovery_tokens(32) >= event.recovery_tokens


# ---------------------------------------------------------------------------
# EpisodeMetrics — empty
# ---------------------------------------------------------------------------


def test_empty_metrics_mean_is_zero():
    m = EpisodeMetrics(scenario_name="s", total_tokens=0, solved=False)
    assert m.mean_recovery_tokens == 0.0


def test_empty_metrics_median_is_zero():
    m = EpisodeMetrics(scenario_name="s", total_tokens=0, solved=False)
    assert m.median_recovery_tokens == 0.0


def test_empty_metrics_chunking_is_zero():
    m = EpisodeMetrics(scenario_name="s", total_tokens=0, solved=False)
    assert m.mean_chunking_recovery_tokens(32) == 0.0


# ---------------------------------------------------------------------------
# EpisodeMetrics — with events
# ---------------------------------------------------------------------------


def _make_event(inj: int, act: int) -> RecoveryEvent:
    return RecoveryEvent(
        injection_position=inj,
        act_position=act,
        action="move east",
        obs_content="obs",
    )


def test_mean_recovery_single_event():
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=50,
        solved=True,
        recovery_events=[_make_event(10, 30)],
    )
    assert m.mean_recovery_tokens == 20.0


def test_mean_recovery_multiple_events():
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=200,
        solved=True,
        recovery_events=[_make_event(0, 20), _make_event(50, 80)],  # 20, 30
    )
    assert m.mean_recovery_tokens == 25.0


def test_median_recovery_odd_count():
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=300,
        solved=True,
        recovery_events=[
            _make_event(0, 10),   # 10
            _make_event(50, 65),  # 15
            _make_event(100, 130),  # 30
        ],
    )
    assert m.median_recovery_tokens == 15.0


def test_mean_chunking_greater_than_mean_streaming():
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=200,
        solved=True,
        recovery_events=[_make_event(10, 30), _make_event(55, 80)],
    )
    assert m.mean_chunking_recovery_tokens(32) >= m.mean_recovery_tokens


def test_speedup_vs_chunking_above_one():
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=200,
        solved=True,
        recovery_events=[
            _make_event(10, 30),  # recovery=20, chunking=20+22=42 (inj@10, chunk=32)
        ],
    )
    # speedup = chunking_mean / stream_mean = 42/20 > 1
    assert m.speedup_vs_chunking(32) > 1.0


def test_speedup_at_boundary_is_one():
    # injection exactly on boundary → no delay, speedup should be exactly 1
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=100,
        solved=True,
        recovery_events=[_make_event(32, 52)],  # inj@32, chunk=32, delay=0
    )
    assert m.speedup_vs_chunking(32) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RecoveryEvent — is_failure and is_correct fields
# ---------------------------------------------------------------------------


def test_recovery_event_is_failure_defaults_false():
    event = _make_event(10, 30)
    assert event.is_failure is False


def test_recovery_event_is_correct_defaults_none():
    event = _make_event(10, 30)
    assert event.is_correct is None


def test_recovery_event_is_failure_can_be_set():
    event = RecoveryEvent(
        injection_position=10,
        act_position=30,
        action="goto",
        obs_content="Nothing happens.",
        is_failure=True,
    )
    assert event.is_failure is True


def test_recovery_event_is_correct_can_be_set():
    event = RecoveryEvent(
        injection_position=10,
        act_position=30,
        action="look",
        obs_content="Nothing happens.",
        is_failure=True,
        is_correct=True,
    )
    assert event.is_correct is True


# ---------------------------------------------------------------------------
# EpisodeMetrics — task_type and had_failures
# ---------------------------------------------------------------------------


def test_episode_metrics_task_type_defaults_empty():
    m = EpisodeMetrics(scenario_name="s", total_tokens=0, solved=False)
    assert m.task_type == ""


def test_episode_metrics_task_type_can_be_set():
    m = EpisodeMetrics(scenario_name="s", total_tokens=0, solved=False, task_type="pick_heat")
    assert m.task_type == "pick_heat"


def test_had_failures_false_when_no_failure_events():
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=50,
        solved=True,
        recovery_events=[_make_event(0, 20)],
    )
    assert m.had_failures is False


def test_had_failures_true_when_any_failure_event():
    fail_event = RecoveryEvent(
        injection_position=0,
        act_position=20,
        action="look",
        obs_content="Nothing happens.",
        is_failure=True,
    )
    m = EpisodeMetrics(
        scenario_name="s",
        total_tokens=50,
        solved=True,
        recovery_events=[_make_event(30, 50), fail_event],
    )
    assert m.had_failures is True


def test_had_failures_false_when_no_events():
    m = EpisodeMetrics(scenario_name="s", total_tokens=0, solved=False)
    assert m.had_failures is False


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


def _make_episode(solved: bool, has_failure: bool, task_type: str = "") -> EpisodeMetrics:
    events = []
    if has_failure:
        events.append(
            RecoveryEvent(
                injection_position=0,
                act_position=20,
                action="look",
                obs_content="Nothing happens.",
                is_failure=True,
            )
        )
    return EpisodeMetrics(
        scenario_name="s",
        total_tokens=100,
        solved=solved,
        task_type=task_type,
        recovery_events=events,
    )


def test_benchmark_result_empty():
    br = BenchmarkResult()
    assert br.episodes == []
    assert br.success_rate == 0.0
    assert br.no_failures == []
    assert br.with_failures == []


def test_benchmark_result_no_failures_partition():
    ep1 = _make_episode(True, has_failure=False)
    ep2 = _make_episode(False, has_failure=True)
    br = BenchmarkResult(episodes=[ep1, ep2])
    assert br.no_failures == [ep1]
    assert br.with_failures == [ep2]


def test_benchmark_result_success_rate():
    episodes = [
        _make_episode(True, False),
        _make_episode(True, False),
        _make_episode(False, True),
    ]
    br = BenchmarkResult(episodes=episodes)
    assert br.success_rate == pytest.approx(2 / 3)


def test_benchmark_result_success_rate_for_subset():
    episodes = [
        _make_episode(True, True),
        _make_episode(False, True),
        _make_episode(True, False),
    ]
    br = BenchmarkResult(episodes=episodes)
    assert br.success_rate_for(br.with_failures) == pytest.approx(0.5)
    assert br.success_rate_for(br.no_failures) == pytest.approx(1.0)


def test_benchmark_result_success_rate_for_empty_subset():
    br = BenchmarkResult(episodes=[_make_episode(True, False)])
    assert br.success_rate_for([]) == 0.0


def test_benchmark_result_all_no_failures():
    episodes = [_make_episode(True, False), _make_episode(False, False)]
    br = BenchmarkResult(episodes=episodes)
    assert len(br.no_failures) == 2
    assert len(br.with_failures) == 0
