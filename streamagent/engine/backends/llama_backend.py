"""LLaMA backend using llama-cpp-python."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from llama_cpp import Llama, llama_get_logits

from streamagent.engine.interfaces import BackendProtocol


class LlamaBackend(BackendProtocol):
    """Backend wrapping llama-cpp-python (llama.cpp).

    Supports GGUF quantized models with Metal acceleration on macOS.
    """

    def __init__(
        self,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> None:
        """Initialize LlamaBackend with no loaded model.

        Args:
            temperature: Sampling temperature (0.0 = greedy, >1.0 = more random).
            top_k: Number of highest probability tokens to keep for sampling.
            top_p: Cumulative probability for nucleus sampling (0.0-1.0).
        """
        self._model: Llama | None = None
        self._context_length: int = 0
        self._prefill_logits_ready: bool = False
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def load_model(self, model_path: str, **kwargs: object) -> None:
        """Load a GGUF model using llama-cpp-python.

        Args:
            model_path: Path to the .gguf model file.
            n_ctx: Maximum context length (default: 4096).
            n_gpu_layers: Number of layers to offload to GPU (default: -1 for all).
                         -1 enables Metal acceleration on macOS.
        """
        n_ctx: int = int(cast(int | None, kwargs.get("n_ctx")) or 4096)
        n_gpu_layers: int = int(cast(int | None, kwargs.get("n_gpu_layers")) or -1)

        self._model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            verbose=False,
        )
        self._context_length = n_ctx

    def tokenize(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model.tokenize(text.encode(), special=True)

    def detokenize(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text string.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model.detokenize(ids).decode("utf-8", errors="replace")

    def prefill(self, input_ids: list[int]) -> int:
        """Run prefill pass (evaluates the entire sequence).

        Args:
            input_ids: List of token IDs to prefill with.

        Returns:
            Length of KV cache after prefill (which equals len(input_ids)).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        self._model.eval(input_ids)
        self._prefill_logits_ready = True
        return len(input_ids)

    def forward_one(
        self,
        token_id: int,
        cache_position: int,
    ) -> tuple[int, float]:
        """Single token forward pass (autoregressive decoding step).

        Args:
            token_id: The token ID to evaluate.
            cache_position: Current position in KV cache.
                           Must increment monotonically across all calls.

        Returns:
            Tuple of (next_token_id: int, log_prob: float).
            next_token_id is the argmax of logits.
            log_prob is the log-probability of next_token_id.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        if self._prefill_logits_ready:
            # Logits from prefill are still valid — skip eval to avoid
            # double-processing the last prompt token.
            self._prefill_logits_ready = False
        else:
            self._model.eval([token_id])

        return self._sample_logits()

    def inject_one(self, token_id: int, cache_position: int) -> None:
        """Process one forced observation token into the KV cache.

        Runs a forward pass to update the KV cache but discards the logits.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        self._prefill_logits_ready = False
        self._model.eval([token_id])

    @property
    def context_length(self) -> int:
        """Maximum context length of the loaded model.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._context_length

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_logits(self) -> tuple[int, float]:
        """Read logits and sample next token with temperature/top-k/top-p.

        Returns:
            Tuple of (next_token_id: int, log_prob: float).
        """
        assert self._model is not None
        n_vocab = self._model.n_vocab()
        logits_ptr = llama_get_logits(self._model._ctx.ctx)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy().astype(np.float64)

        if self._temperature == 0.0:
            # Greedy: argmax
            next_token_id = int(np.argmax(logits))
            shifted = logits - logits[next_token_id]
            log_sum_exp = logits[next_token_id] + math.log(float(np.sum(np.exp(shifted))))
            log_prob = float(logits[next_token_id]) - log_sum_exp
            return next_token_id, log_prob

        # Temperature scaling
        logits = logits / self._temperature

        # Numerically stable softmax → probabilities
        shifted = logits - np.max(logits)
        probs = np.exp(shifted)
        probs /= probs.sum()

        # Top-k filtering
        if 0 < self._top_k < n_vocab:
            top_k_indices = np.argpartition(probs, -self._top_k)[-self._top_k :]
            mask = np.zeros(n_vocab, dtype=bool)
            mask[top_k_indices] = True
            probs = np.where(mask, probs, 0.0)
            probs /= probs.sum()

        # Top-p (nucleus) filtering
        if self._top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            # keep the smallest set whose cumulative prob >= top_p
            cutoff = int(np.searchsorted(cumsum, self._top_p, side="right")) + 1
            cutoff = max(1, min(cutoff, n_vocab))
            keep = sorted_indices[:cutoff]
            mask = np.zeros(n_vocab, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)
            probs /= probs.sum()

        # Sample
        next_token_id = int(np.random.choice(n_vocab, p=probs))
        log_prob = float(np.log(max(probs[next_token_id], 1e-40)))

        return next_token_id, log_prob
