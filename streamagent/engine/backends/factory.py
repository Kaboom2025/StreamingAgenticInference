"""Factory for creating LLM backend instances."""

from __future__ import annotations

from streamagent.engine.backends.hf_backend import HFBackend
from streamagent.engine.backends.llama_backend import LlamaBackend
from streamagent.engine.backends.mlx_backend import MLXBackend
from streamagent.engine.interfaces import BackendProtocol


def create_backend(
    backend_type: str,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.9,
) -> BackendProtocol:
    """Create a backend instance by type.

    Args:
        backend_type: One of "llama", "hf", or "mlx".
        temperature: Sampling temperature (0.0 = greedy, >1.0 = more random).
        top_k: Number of highest probability tokens to keep for sampling.
        top_p: Cumulative probability for nucleus sampling (0.0-1.0).

    Returns:
        Backend instance implementing BackendProtocol.

    Raises:
        ValueError: If backend_type is unknown.
    """
    if backend_type == "llama":
        return LlamaBackend(temperature=temperature, top_k=top_k, top_p=top_p)
    elif backend_type == "hf":
        return HFBackend()
    elif backend_type == "mlx":
        return MLXBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_type!r}")
