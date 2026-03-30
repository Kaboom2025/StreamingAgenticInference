"""HuggingFace Transformers backend."""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamagent.engine.interfaces import BackendProtocol


class HFBackend(BackendProtocol):
    """Backend using HuggingFace transformers with PyTorch.

    Supports all HuggingFace causal language models via AutoTokenizer and
    AutoModelForCausalLM. Handles KV cache automatically.
    """

    def __init__(self) -> None:
        """Initialize HFBackend with no loaded model."""
        self._tokenizer: Any = None
        self._model: Any = None
        self._past_key_values: Any = None
        self._context_length: int = 0

    def load_model(self, model_path: str, **kwargs: object) -> None:
        """Load a HuggingFace model with tokenizer.

        Args:
            model_path: HuggingFace model ID or local path.
            n_ctx: Maximum context length (default: from config, fallback 4096).
                  Overrides model.config.max_position_embeddings if provided.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
        )

        # Get context length from model config, or override with kwargs
        if "n_ctx" in kwargs:
            self._context_length = int(cast(int | None, kwargs["n_ctx"]) or 4096)
        else:
            ctx_from_config = getattr(
                self._model.config, "max_position_embeddings", 4096
            )
            self._context_length = int(ctx_from_config)

    def tokenize(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        result: list[int] = self._tokenizer.encode(text, add_special_tokens=False)
        return result

    def detokenize(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text string.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        result: str = self._tokenizer.decode(ids, skip_special_tokens=False)
        return result

    def prefill(self, input_ids: list[int]) -> int:
        """Run prefill pass (evaluates the entire sequence).

        Stores KV cache for use in subsequent forward_one calls.

        Args:
            input_ids: List of token IDs to prefill with.

        Returns:
            Length of KV cache after prefill (which equals len(input_ids)).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        input_tensor = torch.tensor([input_ids])
        with torch.no_grad():
            out = self._model(input_tensor, use_cache=True)
        self._past_key_values = out.past_key_values
        return len(input_ids)

    def forward_one(
        self,
        token_id: int,
        cache_position: int,
    ) -> tuple[int, float]:
        """Single token forward pass with KV cache.

        Args:
            token_id: The token ID to evaluate.
            cache_position: Current position in KV cache.
                           Must increment monotonically across all calls.

        Returns:
            Tuple of (next_token_id: int, log_prob: float).
            next_token_id is the argmax of logits.
            log_prob is the log-probability of next_token_id.

        Raises:
            RuntimeError: If model is not loaded or prefill not called.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        input_tensor = torch.tensor([[token_id]])
        pos_tensor = torch.tensor([[cache_position]])

        with torch.no_grad():
            out = self._model(
                input_tensor,
                past_key_values=self._past_key_values,
                position_ids=pos_tensor,
                use_cache=True,
            )
        self._past_key_values = out.past_key_values

        # Extract logits for the last token
        logits = out.logits[0, -1]  # shape: [vocab_size]
        next_token_id = int(logits.argmax())
        log_prob = float(torch.log_softmax(logits, dim=-1)[next_token_id])

        return next_token_id, log_prob

    @property
    def context_length(self) -> int:
        """Maximum context length of the loaded model.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._context_length
