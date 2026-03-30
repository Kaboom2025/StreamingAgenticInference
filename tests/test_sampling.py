"""Unit tests for LlamaBackend sampling logic (temperature/top-k/top-p)."""

import ctypes
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from streamagent.engine.backends.llama_backend import LlamaBackend


def _make_backend(
    temperature: float = 0.6, top_k: int = 50, top_p: float = 0.9
) -> LlamaBackend:
    """Create a LlamaBackend with specified sampling parameters."""
    return LlamaBackend(temperature=temperature, top_k=top_k, top_p=top_p)


def _setup_logits(backend: LlamaBackend, values: list[float]) -> None:
    """Set up mock model to return the given logits."""
    n = len(values)
    mock_model = MagicMock()
    mock_model.n_vocab.return_value = n
    backend._model = mock_model


def _ctypes_array(*values: float) -> ctypes.Array[ctypes.c_float]:  # type: ignore
    """Create a ctypes array from floating-point values."""
    arr_type = ctypes.c_float * len(values)
    return arr_type(*values)


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_greedy_selects_argmax(mock_get_logits: MagicMock) -> None:
    """Test that temperature=0.0 always selects the token with highest logit."""
    logits = [0.1, 0.2, 5.0, 0.3]  # token 2 is clear winner
    mock_get_logits.return_value = _ctypes_array(*logits)

    backend = _make_backend(temperature=0.0)
    _setup_logits(backend, logits)

    token_id, _ = backend._sample_logits()
    assert token_id == 2


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_greedy_log_prob_is_negative(mock_get_logits: MagicMock) -> None:
    """Test that log_prob is negative (and < 0) in greedy mode."""
    logits = [1.0, 2.0, 3.0, 1.5]
    mock_get_logits.return_value = _ctypes_array(*logits)

    backend = _make_backend(temperature=0.0)
    _setup_logits(backend, logits)

    _, log_prob = backend._sample_logits()
    assert log_prob < 0.0


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_temperature_affects_distribution(mock_get_logits: MagicMock) -> None:
    """Test that higher temperature leads to more uniform distribution.

    With low temperature, high-logit tokens dominate.
    With high temperature, distribution is more uniform.
    """
    logits = [0.1, 0.2, 5.0, 0.3]

    # Low temperature (0.1) should concentrate mass on token 2
    low_temp_samples = []
    for _ in range(100):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=0.1, top_k=50, top_p=0.9)
        _setup_logits(backend, logits)
        token_id, _ = backend._sample_logits()
        low_temp_samples.append(token_id)

    # High temperature (2.0) should have more diverse samples
    high_temp_samples = []
    for _ in range(100):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=2.0, top_k=50, top_p=0.9)
        _setup_logits(backend, logits)
        token_id, _ = backend._sample_logits()
        high_temp_samples.append(token_id)

    # Low temp should favor token 2
    low_temp_count_2 = sum(1 for t in low_temp_samples if t == 2)
    # High temp should have token 2 less frequently
    high_temp_count_2 = sum(1 for t in high_temp_samples if t == 2)

    assert low_temp_count_2 > high_temp_count_2


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_top_k_limits_candidates(mock_get_logits: MagicMock) -> None:
    """Test that top_k=2 restricts sampling to only the top-2 tokens."""
    # Create logits where token 2 is best, token 3 is second-best,
    # and tokens 0, 1 are far behind
    logits = [0.0, 0.1, 5.0, 4.0, -10.0, -10.0]

    samples = []
    for _ in range(100):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=0.6, top_k=2, top_p=0.9)
        _setup_logits(backend, logits)
        token_id, _ = backend._sample_logits()
        samples.append(token_id)

    # All samples should be either token 2 or token 3
    assert all(t in [2, 3] for t in samples), f"Got unexpected tokens: {set(samples)}"


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_top_p_limits_candidates(mock_get_logits: MagicMock) -> None:
    """Test that top_p restricts sampling to nucleus (top cumulative probability).

    With top_p=0.5, we keep the smallest set of tokens that sum to >= 50% probability.
    """
    # Create logits with a clear dominant token
    logits = [5.0, 4.0, 3.0, -10.0, -10.0]

    samples = []
    for _ in range(100):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=0.6, top_k=50, top_p=0.5)
        _setup_logits(backend, logits)
        token_id, _ = backend._sample_logits()
        samples.append(token_id)

    # With top_p=0.5 and skewed logits, we should only sample from a small set
    unique_tokens = set(samples)
    # Should be roughly 1-2 unique tokens (the top ones that sum to >= 50%)
    assert len(unique_tokens) <= 2, f"top_p=0.5 should limit to <= 2 tokens, got {unique_tokens}"


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_all_probability_mass_used(mock_get_logits: MagicMock) -> None:
    """Test that after top-k and top-p filtering, probabilities sum to ~1."""
    logits = [1.0, 2.0, 3.0, 0.5, 0.1]
    mock_get_logits.return_value = _ctypes_array(*logits)

    backend = _make_backend(temperature=0.6, top_k=3, top_p=0.95)
    _setup_logits(backend, logits)

    # We can't directly inspect probs, but we can verify that:
    # (1) sampling produces valid token IDs
    # (2) log_prob is reasonable (not -inf, not 0)
    for _ in range(10):
        mock_get_logits.return_value = _ctypes_array(*logits)
        token_id, log_prob = backend._sample_logits()
        assert 0 <= token_id < len(logits)
        assert log_prob < 0.0  # log of prob < 1 is negative


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_greedy_vs_stochastic_same_winner(mock_get_logits: MagicMock) -> None:
    """Test that greedy (temp=0) and very-low-temp still favor the argmax token."""
    logits = [0.5, 1.0, 2.0, 0.8]
    argmax_token = 2

    # Greedy sampling
    mock_get_logits.return_value = _ctypes_array(*logits)
    greedy_backend = _make_backend(temperature=0.0)
    _setup_logits(greedy_backend, logits)
    greedy_token, _ = greedy_backend._sample_logits()

    # Very low temperature sampling
    low_temp_samples = []
    for _ in range(50):
        mock_get_logits.return_value = _ctypes_array(*logits)
        low_temp_backend = _make_backend(temperature=0.01)
        _setup_logits(low_temp_backend, logits)
        token_id, _ = low_temp_backend._sample_logits()
        low_temp_samples.append(token_id)

    # Greedy should always be argmax
    assert greedy_token == argmax_token

    # Low temp should mostly be argmax
    low_temp_count = sum(1 for t in low_temp_samples if t == argmax_token)
    assert low_temp_count >= 45  # at least 90% should be the best token


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_sampling_produces_valid_tokens(mock_get_logits: MagicMock) -> None:
    """Test that sampling always produces valid token IDs within vocab range."""
    logits = [0.5, 1.0, 2.0, 0.8, -5.0, 0.2]
    n_vocab = len(logits)

    for _ in range(50):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=0.6, top_k=10, top_p=0.9)
        _setup_logits(backend, logits)
        token_id, _ = backend._sample_logits()
        assert 0 <= token_id < n_vocab


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_extreme_top_k_larger_than_vocab(mock_get_logits: MagicMock) -> None:
    """Test that top_k > vocab_size is handled gracefully."""
    logits = [1.0, 2.0, 3.0]
    n_vocab = len(logits)

    # top_k=1000 > vocab_size, should still work
    for _ in range(20):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=0.6, top_k=1000, top_p=0.9)
        _setup_logits(backend, logits)
        token_id, log_prob = backend._sample_logits()
        assert 0 <= token_id < n_vocab
        assert log_prob < 0.0


@patch("streamagent.engine.backends.llama_backend.llama_get_logits")
def test_top_p_at_boundary(mock_get_logits: MagicMock) -> None:
    """Test top_p edge cases: top_p=0.99999 and top_p=1.0."""
    logits = [1.0, 2.0, 3.0, 0.5, 0.1]

    # top_p=0.99999 should allow almost all tokens
    samples_high_p = []
    for _ in range(50):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=0.6, top_k=50, top_p=0.99999)
        _setup_logits(backend, logits)
        token_id, _ = backend._sample_logits()
        samples_high_p.append(token_id)

    # top_p=1.0 should allow all tokens
    samples_full_p = []
    for _ in range(50):
        mock_get_logits.return_value = _ctypes_array(*logits)
        backend = _make_backend(temperature=0.6, top_k=50, top_p=1.0)
        _setup_logits(backend, logits)
        token_id, _ = backend._sample_logits()
        samples_full_p.append(token_id)

    # Both should have similar diversity
    unique_high_p = len(set(samples_high_p))
    unique_full_p = len(set(samples_full_p))
    # With high top_p, we should see multiple unique tokens
    assert unique_high_p >= 2
    assert unique_full_p >= 2
