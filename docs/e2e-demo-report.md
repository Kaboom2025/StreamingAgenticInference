# End-to-End Demo Report

## What we tested

Ran the full StreamAgent pipeline with a **real LLM** (Qwen2.5-1.5B-Instruct, Q4 GGUF, ~1 GB) to verify the architecture works end-to-end — not just with mock/scripted backends but with actual neural network inference.

**Task**: CalcEnv — the simplest environment that exercises the full pipeline.
The LLM receives a math problem, must emit `<act cmd="eval EXPR"/>`, CalcEnv evaluates it, and the result is injected back into the KV cache mid-stream.

**Result**: Working. `347 * 821 = 284887` and `2**16 - 1 = 65535` both completed successfully with reward 1.0.

## Bugs found and fixed

### 1. `BackendProtocol` missing `inject_one` method

**Symptom**: Integration test `test_full_episode_agent_reaches_goal` failed — agent only executed one action instead of two.

**Root cause**: `ScriptedBackend.forward_one` was called for both generation and observation injection. During injection, the return values are discarded, but each call consumed a character from the scripted sequence. The second action's characters were eaten during the first observation's injection.

**Fix**: Added `inject_one(token_id, cache_position) -> None` to `BackendProtocol` with a default implementation that delegates to `forward_one`. `ScriptedBackend` overrides it to track positions without consuming the script. `KVStream` now calls `inject_one` during observation injection.

**Files**: `interfaces.py`, `kv_stream.py`, `test_integration.py`

### 2. `LlamaBackend.tokenize` not parsing special tokens

**Symptom**: Model generated `!!!!!!!...` (garbage) — completely incoherent output.

**Root cause**: `llama_cpp.Llama.tokenize()` defaults to `special=False`, which tokenizes `<|im_start|>` as literal character sequences (`<`, `|`, `i`, `m`, ...) instead of the single special token ID that the model was trained with.

**Fix**: Pass `special=True` to `self._model.tokenize()`.

**File**: `llama_backend.py`

### 3. `LlamaBackend.forward_one` reading wrong logits buffer

**Symptom**: Even with special tokens fixed, model still generated `!!!!...` (argmax was always token ID 0).

**Root cause**: The backend read logits from `self._model.scores[-1]`, which is a `(n_ctx, n_vocab)` numpy array. With `logits_all=False`, this array is all zeros — the actual logits are in a separate C buffer accessed via `llama_get_logits()`.

**Fix**: Replaced `self._model.scores[-1]` with `llama_get_logits(self._model._ctx.ctx)` and read logits as a numpy view of the C pointer. Also switched argmax/log-prob computation to numpy for correctness and performance.

**File**: `llama_backend.py`

### 4. Double-processing the last prefill token

**Symptom**: After fixes 2 and 3, model generated close-to-correct output but with subtle token-level corruption.

**Root cause**: `KVStream.start()` calls `prefill(all_tokens)` which processes the entire prompt including the last token. Then `run()` calls `forward_one(last_token, ...)` which re-processes that same token via `model.eval([token_id])`. This double-feeds the last token into the KV cache, corrupting the model's attention state.

**Fix**: Added a `_prefill_logits_ready` flag to `LlamaBackend`. After `prefill`, the flag is set. The first `forward_one` call skips `model.eval()` and just reads the logits that are already available from prefill. Subsequent calls eval normally.

**File**: `llama_backend.py`

## Test status

| Suite | Count | Status |
|-------|-------|--------|
| Unit tests (interfaces, kv_stream, sink_cache, router, injector, gridworld, scenarios, calc_env) | 192 | All pass |
| Integration tests (ScriptedBackend full pipeline) | 10 | All pass |
| Backend tests | ~6 | Segfault (pre-existing llama-cpp Metal issue, not our code) |
| **Real LLM demo** | 2 expressions | **Both pass** |

## Architecture validated

```
System Prompt → LlamaBackend.prefill()
                     ↓
              KVStream.run() loop:
                  1. Drain ObsInjector → inject_one() per obs token
                  2. forward_one() → next token
                  3. yield Token
                     ↓
              Router.process(token)
                  FSM: PASSTHROUGH → MAYBE_TAG → IN_ACT_TAG → parse action
                     ↓
              CalcEnv.step(action)
                  _safe_eval(expr) → Observation("result", "284887")
                  injector.put(obs) → queued for next KVStream iteration
                     ↓
              (cycle repeats — obs injected into KV cache, generation continues)
```

## Model notes

- **SmolLM2-135M-Instruct** (~95 MB): Too small. Cannot follow custom tag format even with proper chat template. Outputs essays instead of structured tags.
- **Qwen2.5-1.5B-Instruct** (~1 GB): Works reliably. Follows `<act cmd="..."/>` format with a few-shot example in the prompt. Needs to be forced to use the tool (it can do simple math in its head and skips the tool).
