"""KVStream — never-terminated LLM generation loop with persistent KV cache.

Critical invariant: cache_position increments strictly monotonically across
ALL forward passes (generation + injection). Breaking this causes silent RoPE
attention collapse.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, Literal, Optional

from streamagent.engine.injector import ObsInjector
from streamagent.engine.interfaces import BackendProtocol, CacheStats, Token
from streamagent.engine.sink_cache import SinkCache


@dataclass
class KVStreamConfig:
    model_id: str
    backend: Literal["hf", "llama", "mlx"]
    sink_tokens: int = 4
    window_length: int = 2048
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.9
    max_think_tokens_per_step: int = 64
    device: str = "auto"
    quantization: Optional[str] = None


class KVStream:
    """A never-terminated LLM generation process.

    Owns the model, the SinkCache, and the async generation loop.
    Accepts mid-stream observation injections via inject().
    Yields Token objects to the caller (typically the Router).
    """

    def __init__(self, config: KVStreamConfig, system_prompt: str) -> None:
        self._config = config
        self._system_prompt = system_prompt

        # Set by start() or injected directly in tests via stream._backend
        self._backend: Optional[BackendProtocol] = None

        self._sink_cache = SinkCache(
            capacity=config.window_length,
            n_sinks=config.sink_tokens,
        )
        self._injector = ObsInjector()

        # Absolute position counter — never resets within an episode
        self._position: int = 0
        # Token id used as the first generation input after prefill
        self._last_prefill_token_id: int = 0
        self._started: bool = False
        self._stop_requested: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Prefill with system prompt, initialise position counter."""
        if self._backend is None:
            self._backend = self._load_backend()

        prompt_ids = self._backend.tokenize(self._system_prompt)
        if not prompt_ids:
            prompt_ids = [0]

        prefill_len = self._backend.prefill(prompt_ids)
        self._position = prefill_len

        # Seed cache with prefill tokens
        for i, tok_id in enumerate(prompt_ids):
            self._sink_cache.add(tok_id, i)

        self._last_prefill_token_id = prompt_ids[-1]
        self._started = True

    async def run(self) -> AsyncGenerator[Token, None]:
        """Async generator — yields one Token per iteration, forever.

        Before each token:
          1. Drain injector queue and run injection forward passes
             (each injection token increments _position by 1).
          2. Generate one token (increments _position by 1).
          3. Yield the token.
        """
        if not self._started:
            self.start()

        current_id = self._last_prefill_token_id

        while not self._stop_requested:
            # ----------------------------------------------------------------
            # 1. Drain observation injection queue
            # ----------------------------------------------------------------
            pending = self._injector.get_pending()
            for obs in pending:
                obs_text = obs.to_token_text()
                obs_ids = self._backend.tokenize(obs_text)  # type: ignore[union-attr]
                for tok_id in obs_ids:
                    self._backend.inject_one(  # type: ignore[union-attr]
                        tok_id, self._position
                    )
                    self._sink_cache.add(tok_id, self._position)
                    self._position += 1

            # ----------------------------------------------------------------
            # 2. Generate one token
            # ----------------------------------------------------------------
            next_id, log_prob = self._backend.forward_one(  # type: ignore[union-attr]
                current_id, self._position
            )
            self._sink_cache.add(current_id, self._position)
            self._position += 1
            current_id = next_id

            # ----------------------------------------------------------------
            # 3. Yield
            # ----------------------------------------------------------------
            text = self._backend.detokenize([next_id])  # type: ignore[union-attr]
            yield Token(id=next_id, text=text, log_prob=log_prob)

            # Cooperative yield — allow other coroutines to run
            await asyncio.sleep(0)

    def inject(self, obs_text: str) -> None:
        """Thread-safe. Enqueue a raw observation string for injection.

        The text is wrapped as an Observation and placed in the injector queue.
        KVStream drains the queue before generating the next token.
        """
        from streamagent.engine.interfaces import Observation

        obs = Observation(type="injected", content=obs_text)
        self._injector.put(obs)

    def stop(self) -> None:
        """Signal the generation loop to stop after the current token."""
        self._stop_requested = True

    @property
    def position(self) -> int:
        """Current absolute token position in the KV cache."""
        return self._position

    @property
    def cache_stats(self) -> CacheStats:
        return self._sink_cache.get_stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_backend(self) -> BackendProtocol:
        """Instantiate backend from config.backend string."""
        from streamagent.engine.backends.factory import create_backend

        backend = create_backend(
            self._config.backend,
            temperature=self._config.temperature,
            top_k=self._config.top_k,
            top_p=self._config.top_p,
        )
        backend.load_model(self._config.model_id)
        return backend
