# StreamAgent — Current Status

_Last updated: 2026-03-29 (session 4)_

---

## What it is

StreamAgent is a streaming agentic inference engine: a language model runs in a **never-terminated generation loop** (KVStream), parsing its own output token-by-token for `<act>` tags, executing tool calls, and injecting observations back mid-stream — all without restarting the context or re-prefilling.

---

## Architecture

```
KVStreamConfig
      │
      ▼
  KVStream ──── SinkCache (sliding window KV eviction)
      │               │
      │           BackendProtocol
      │          ┌────┴───────────────┐
      │      LlamaBackend         HFBackend / MLXBackend
      │      (llama.cpp, GGUF,    (HuggingFace / Apple MLX)
      │       Metal on macOS)
      │
      ▼
   Router (FSM)
   Parses token stream for <think>, <act>, <obs> tags
      │
      ├── <act cmd="..."/>  ──►  Environment.step()
      │                               │
      │                        CalcEnv / GridWorld
      │                               │
      └── <obs>...</obs>  ◄──  ObsInjector.put(obs)
```

**Key invariant**: `cache_position` increments strictly monotonically across all forward passes (generation + injection). Broken monotonicity causes silent RoPE attention collapse.

---

## Implemented & tested

| Component | File | Notes |
|-----------|------|-------|
| `BackendProtocol` | `engine/interfaces.py` | `prefill`, `forward_one`, `inject_one`, `tokenize`, `detokenize`, `context_length` |
| `LlamaBackend` | `engine/backends/llama_backend.py` | llama-cpp-python, GGUF, Metal; temperature/top-k/top-p sampling |
| `HFBackend` | `engine/backends/hf_backend.py` | HuggingFace transformers |
| `MLXBackend` | `engine/backends/mlx_backend.py` | Apple MLX |
| `KVStream` | `engine/kv_stream.py` | Async generation loop, prefill, injection |
| `KVStreamConfig` | `engine/kv_stream.py` | `temperature=0.6`, `top_k=50`, `top_p=0.9` (wired through to backend) |
| `SinkCache` | `engine/sink_cache.py` | Sliding window with sink token retention |
| `Router` | `engine/router.py` | Deterministic FSM; `<act>` → Action (with XML attribute parsing for `params`), `<obs>` passthrough |
| `ObsInjector` | `engine/injector.py` | Thread-safe injection queue |
| `CalcEnv` | `env/calc_env.py` | Single-tool (eval) environment; proves full pipeline |
| `GridWorld` | `env/gridworld.py` | 2D grid navigation; multi-step planning |
| `Environment ABC` | `engine/interfaces.py` | `reset`, `step`, `register_injector` |
| `ALFWorldEnv` | `env/alfworld_env.py` | ALFWorld text-world wrapper; lazy import, failure detection via `FAILURE_STRINGS` |
| `alfworld_judge` | `eval/alfworld_judge.py` | Recovery classifier: repeat→False, look/inventory/examine/goto-different→True, else→None |
| `RecoveryEvent` | `eval/metrics.py` | Per-event injection→act token latency; `is_failure`, `is_correct` fields for ALFWorld judge |
| `EpisodeMetrics` | `eval/metrics.py` | Mean/median recovery tokens, chunking baseline, `speedup_vs_chunking`, `had_failures` flag |
| `BenchmarkResult` | `eval/metrics.py` | Batch aggregation; partitions into `no_failures` / `with_failures` subsets; `success_rate` |
| Eval harness | `eval/harness.py` | `run_gridworld_episode()` + `run_alfworld_episode()` — async episode loops, position-stamped injection/act events, failure detection, judge wiring |

**Test suite**: 328 tests, **97.63% coverage**, all green.

### Action dataclass — `params` field added

`Action` is no longer `frozen=True`. It now carries an optional `params: dict[str, str] | None` for extra `<act>` attributes (e.g. `<act cmd="pick" obj="apple"/>`). `Router._parse_action_tag` uses `xml.etree.ElementTree` to extract all attributes, with a string-based `cmd`-only fallback for malformed XML.

---

## Sampling (LlamaBackend)

`_sample_logits()` implements the full pipeline:

1. `temperature == 0.0` → pure greedy argmax (deterministic)
2. Temperature scaling → numerically stable softmax
3. Top-k: keep the k highest-probability tokens (`np.argpartition`)
4. Top-p (nucleus): keep the smallest set whose cumulative probability ≥ p
5. `np.random.choice` over the filtered distribution

Defaults: `temperature=0.6`, `top_k=50`, `top_p=0.9`.

---

## What exists but is empty/stub

| Path | Status |
|------|--------|
| `streamagent/eval/harness.py` | GridWorld episode runner done; ALFWorld episode loop not yet built |
| `streamagent/configs/__init__.py` | Empty — YAML config loader not yet built |
| `streamagent/scripts/__init__.py` | Empty — CLI entrypoints not yet built |

---

## Demo

`demo_calc.py` — a script that runs one CalcEnv episode end-to-end with a real GGUF model. Two models are present in the repo root:

- `SmolLM2-135M-Instruct-Q4_K_M.gguf`
- `qwen2.5-1.5b-instruct-q4_k_m.gguf`

---

## What's next

### Step 3 — Eval harness (partial — GridWorld done)

**Done:**
- `eval/metrics.py` — `RecoveryEvent`, `EpisodeMetrics`, `BenchmarkResult`; chunking baseline formula; failure/correct partitioning fields for ALFWorld
- `eval/harness.py` — `run_gridworld_episode()` measuring recovery latency in tokens with analytical chunking baseline
- `KVStream.position` property — exposes token position counter for harness instrumentation

**Still needed to get numbers:**
- A run script that feeds a real GGUF model through `run_gridworld_episode` across all 5 scenarios and prints the table
- ALFWorld episode loop in `harness.py` once ALFWorld integration is ready

### Step 4 — CLI entrypoints _(deferred)_
`streamagent/scripts/run_agent.py` — YAML config → `KVStreamConfig` → run loop. Come back after real eval numbers exist.

### Step 5 — ALFWorld integration _(complete — needs real model run)_

All phases implemented and tested:

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 | Done | `ALFWorldEnv` wrapper, `FAILURE_STRINGS`, `_map_action` (12 commands), `_load_config` |
| Phase 2 | Done | `alfworld_judge.is_correct_recovery(failure_obs, recovery_action, prior_action)` |
| Phase 3 | Done | `run_alfworld_episode()` in `harness.py`; `is_failure`/`is_correct` wired through judge |
| Phase 4 | Done | `configs/alfworld_eval.yaml`, `requirements-eval.txt` → `alfworld>=0.3.3` |
| Phase 5 | Done | Full mocked integration test suite (328 tests, 97.63% coverage) |

**Still needed to get numbers:**
- Install alfworld + real data, then run `run_alfworld_episode` across all 6 task types with Qwen3-8B Q4_K_M
