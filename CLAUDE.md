# StreamAgent — CLAUDE.md

## What this project is

StreamAgent is a streaming agentic inference engine. An LLM runs in a **never-terminated generation loop** (`KVStream`), parsing its own output token-by-token for `<act>` tags, executing actions, and injecting observations back mid-stream — without restarting context or re-prefilling.

The two halves are cleanly separated:
- `streamagent/engine/` — framework-agnostic inference runtime (KV cache, router, injector, backends)
- `streamagent/env/` — environments that conform to the `Environment` ABC

**The engine has zero knowledge of any environment. Environments have zero knowledge of the model. They communicate only through `engine/interfaces.py`.**

---

## Repository layout

```
streamagent/
├── engine/
│   ├── interfaces.py       # BackendProtocol, Environment ABC, Action, Observation, Token
│   ├── kv_stream.py        # Core: async generation loop + KVStreamConfig
│   ├── sink_cache.py       # StreamingLLM attention-sink eviction policy
│   ├── router.py           # Deterministic FSM: <act> → Action, <obs> passthrough
│   ├── injector.py         # Thread-safe observation injection queue
│   └── backends/
│       ├── llama_backend.py  # llama-cpp-python (GGUF, Metal — primary)
│       ├── hf_backend.py     # HuggingFace transformers
│       └── mlx_backend.py    # Apple MLX (Mac only)
├── env/
│   ├── calc_env.py         # Single-tool eval environment (full pipeline proven)
│   ├── gridworld.py        # 2D grid navigation
│   ├── renderer.py         # ASCII renderer
│   └── scenarios.py        # Named test scenarios
├── eval/                   # TODO: harness.py, metrics.py (not yet built)
├── configs/                # TODO: YAML config loader (not yet built)
└── scripts/                # TODO: CLI entrypoints (not yet built)
tests/
demo_calc.py                # End-to-end demo with real GGUF model
```

---

## Critical invariant

`cache_position` increments **strictly monotonically** across ALL forward passes — generation steps AND observation injection steps. Violating this silently breaks RoPE positional encoding and produces garbage output with no error.

This invariant is enforced in `KVStream`. Never call `backend.forward_one()` or `backend.inject_one()` outside of `KVStream` unless you are absolutely certain you are maintaining this sequence.

---

## Current state

**257 tests, 97.7% coverage, all green.**

Implemented and tested:
- `BackendProtocol`, `LlamaBackend`, `HFBackend`, `MLXBackend`
- `KVStream` + `KVStreamConfig` (temperature=0.6, top_k=50, top_p=0.9)
- `SinkCache` — sliding window with sink token retention
- `Router` — FSM parsing `<think>`, `<act cmd="..."/>`, `<obs type="...">` tags
- `ObsInjector` — thread-safe queue
- `CalcEnv` — single-tool env, proves full pipeline end-to-end
- `GridWorld` — multi-step 2D navigation env

Not yet built:
- `eval/` harness and metrics
- `configs/` YAML loader
- `scripts/` CLI entrypoints

---

## What's next (in order)

1. **Step 3 — Eval harness**: `eval/metrics.py` + `eval/harness.py`. Batch-run CalcEnv/GridWorld, collect recovery latency (tokens from obs injection to next valid `<act>`), success rate, step efficiency.
2. **Step 4 — CLI**: `scripts/run_agent.py`. YAML config → `KVStreamConfig` → run loop with structured logging. Replace `demo_calc.py`.
3. **Step 5 — Harder envs**: Multi-tool environment or ALFWorld wrapper.

---

## Development rules

### Tests first
All new functionality follows TDD: write the test (RED), implement (GREEN), refactor. Coverage must stay ≥ 80% (enforced by pytest config). Use the `tdd-guide` agent.

### No backend imports in tests by default
`test_backends.py` segfaults if `llama-cpp-python` is not installed (Metal initializes on import). Guard any test file that imports llama-related code with `pytest.importorskip` **before** any llama import:

```python
pytest.importorskip("llama_cpp")

from streamagent.engine.backends.llama_backend import LlamaBackend  # safe after skip guard
```

When mocking `forward_one` or `_sample_logits`, never pass a `MagicMock` to `llama_get_logits` — it is a C function and will segfault. Use the ctypes helper in `test_backends.py`:

```python
def _ctypes_float_array(*values: float):
    arr_type = ctypes.c_float * len(values)
    return arr_type(*values)
```

Always pass `temperature=0.0` to `LlamaBackend()` in unit tests that assert on a specific token ID (greedy = deterministic).

### Async tests
All async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (set in `pyproject.toml`). No manual `asyncio.run()`.

### Type annotations required
All function signatures must have type annotations. `mypy --strict` is the target. `ignore_missing_imports = true` is set for optional backends.

### Line length: 100 (ruff)
Linter is `ruff`, formatter is the ruff formatter. Target: `py310`.

### Run tests
```bash
pytest                              # full suite with coverage
pytest tests/test_kv_stream.py      # single file
pytest -k "not backend"             # skip backend integration tests
```

---

## Agents to use

| Situation | Agent |
|-----------|-------|
| New feature or env | `tdd-guide` |
| Architecture decisions (eval harness design, new env interface) | `dl-expert` or `architect` |
| After writing code | `code-reviewer` + `python-reviewer` |
| Deep learning / transformer questions about the model side | `dl-expert` |
| Build or type errors | `build-error-resolver` |

---

## Models present on disk

```
SmolLM2-135M-Instruct-Q4_K_M.gguf      # fast, good for local dev/testing
qwen2.5-1.5b-instruct-q4_k_m.gguf      # primary demo model
```

Primary backend: `llama-cpp-python` with Metal on macOS.

---

## Token vocabulary (defined in `engine/interfaces.py`)

| Tag | Meaning |
|-----|---------|
| `<think>...</think>` | Model's internal reasoning (passed through) |
| `<act cmd="..."/>` | Tool call — parsed by Router → `Action` |
| `<obs type="...">...</obs>` | Environment observation injected mid-stream |
| `<goal>...</goal>` | Episode goal in system prompt |
| `<mem>...</mem>` | Persistent memory slot |

These are **prompt-level tokens only** — no fine-tuning required. The model learns to use them from the system prompt.
