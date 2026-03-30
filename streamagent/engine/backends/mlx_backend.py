"""MLX backend for Apple Silicon (stub implementation)."""

from __future__ import annotations

from streamagent.engine.interfaces import BackendProtocol


class MLXBackend(BackendProtocol):
    """Stub implementation of MLX backend for Apple Silicon.

    Full implementation pending MLX library integration.
    """

    def load_model(self, model_path: str, **kwargs: object) -> None:
        """Not yet implemented.

        Args:
            model_path: Model path.
            **kwargs: Additional arguments.

        Raises:
            NotImplementedError: This backend is not yet implemented.
        """
        raise NotImplementedError("MLXBackend is not yet implemented")

    def tokenize(self, text: str) -> list[int]:
        """Not yet implemented.

        Args:
            text: Text to tokenize.

        Raises:
            NotImplementedError: This backend is not yet implemented.
        """
        raise NotImplementedError("MLXBackend is not yet implemented")

    def detokenize(self, ids: list[int]) -> str:
        """Not yet implemented.

        Args:
            ids: Token IDs to decode.

        Raises:
            NotImplementedError: This backend is not yet implemented.
        """
        raise NotImplementedError("MLXBackend is not yet implemented")

    def prefill(self, input_ids: list[int]) -> int:
        """Not yet implemented.

        Args:
            input_ids: Token IDs to prefill.

        Raises:
            NotImplementedError: This backend is not yet implemented.
        """
        raise NotImplementedError("MLXBackend is not yet implemented")

    def forward_one(
        self,
        token_id: int,
        cache_position: int,
    ) -> tuple[int, float]:
        """Not yet implemented.

        Args:
            token_id: Token ID to evaluate.
            cache_position: Current cache position.

        Raises:
            NotImplementedError: This backend is not yet implemented.
        """
        raise NotImplementedError("MLXBackend is not yet implemented")

    @property
    def context_length(self) -> int:
        """Not yet implemented.

        Raises:
            NotImplementedError: This backend is not yet implemented.
        """
        raise NotImplementedError("MLXBackend is not yet implemented")
