# StreamAgent TDD Quick Start Guide

**A 1-page reference for the test implementation order**

---

## Executive Summary

This project requires **4 independent test files** in strict order:

| Phase | File | Tests | Days | Why First |
|-------|------|-------|------|-----------|
| 1 | `test_sink_cache.py` | 12+ | 1-2 | Foundation: all generation depends on cache invariants |
| 2 | `test_router.py` | 11+ | 2-3 | Token FSM: stateful, deterministic, no model needed |
| 3 | `test_injector.py` | 9+ | 3-3.5 | Queue: simple threading, independent from cache |
| 4 | `test_gridworld.py` | 18+ | 3.5-4 | Environment: fully independent physics engine |

**Total: 50+ unit tests, 80%+ coverage, 0 model loading**

---

## Phase 1: SinkCache (DAYS 1-2)

### CRITICAL INVARIANT: cache_position Must Be Strictly Monotonic

```python
# This test must pass before any other implementation
def test_cache_position_strictly_monotonic():
    positions = []
    for i in range(100):
        cache.update(...)
        positions.append(cache.cache_position)
    assert positions == list(range(100))  # NO SKIPS, NO REPEATS
```

### Key Test Cases
1. **FIFO eviction**: Oldest non-sink token evicted first
2. **Sink preservation**: Tokens 0..num_sinks-1 NEVER evict
3. **Pinned tokens**: Goal tokens stay forever
4. **Attention mask**: Rebuild after eviction, handle non-contiguous positions
5. **Multiple evictions**: 100-token generation, verify sequence

### Quick Test Run
```bash
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheMonotonicity::test_cache_position_strictly_monotonic -v
# Must pass 100% before proceeding
```

---

## Phase 2: Router (DAYS 2-3)

### FSM State Machine

```
PASSTHROUGH  ──(<)──► MAYBE_TAG
MAYBE_TAG    ──(act)──► IN_ACT_TAG  ──(/>)──► PASSTHROUGH (dispatch)
MAYBE_TAG    ──(obs)──► IN_OBS_TAG  ──</obs>──► PASSTHROUGH (silence)
MAYBE_TAG    ──(other)──► PASSTHROUGH (emit '<' + token)
```

### Key Test Cases
1. **State transitions**: All 6 transitions tested
2. **Act tag parsing**: `<act cmd="move" dir="N"/>` → ActTag(cmd="move", params={"dir": "N"})
3. **Malformed tags**: 50-token timeout in IN_ACT_TAG, log event, return to PASSTHROUGH
4. **Obs silencing**: `<obs type="...">...</obs>` → no output, no handler
5. **Multi-token tags**: Tag spans multiple tokens, accumulate correctly

### Quick Test Run
```bash
pytest streamagent/tests/test_router.py::TestRouterFSM -v
# Verify all state transitions work
```

---

## Phase 3: Injector (DAYS 3-3.5)

### Thread-Safe Queue with Priority

```python
# Core operations
injector.put(obs)                  # Non-blocking, thread-safe
pending = injector.get_pending()   # Returns all pending, flushes queue
formatted = injector.format_obs(obs)  # Renders XML: <obs type="...">...</obs>
```

### Key Test Cases
1. **Thread safety**: 4 threads, 10 puts each, no corruption
2. **Overflow**: max_queue_size=32, drops oldest when full
3. **Priority**: Higher priority first in get_pending()
4. **Formatting**: Correct XML for collision, enemy_near, tick, full_grid
5. **Batching**: Multiple obs in one get_pending()

### Quick Test Run
```bash
pytest streamagent/tests/test_injector.py::TestInjectorThreadSafety -v
# Verify concurrent access works
```

---

## Phase 4: GridWorld (DAYS 3.5-4)

### Physics Engine (No Engine Dependencies)

```python
gridworld.step(action)  # → (done: bool, info: dict)
# Events in info["events"]: collision, enemy_near, goal_reached, death, tick
```

### Key Test Cases
1. **Movement**: N/S/E/W all work, update position
2. **Walls**: Blocked movement, collision event, position unchanged
3. **Enemy patrol**: Deterministic loop, position[i+1] = path[(idx+1) % len(path)]
4. **Enemy near**: distance ≤ 2 → event with dist and bearing
5. **Goal reached**: agent_pos == goal_pos → done=True
6. **Death**: agent_pos == enemy_pos → done=True, death event priority=10
7. **Tick heartbeat**: Every 10 steps, inject tick event
8. **Max steps**: done=True when step_count >= max_steps

### Quick Test Run
```bash
pytest streamagent/tests/test_gridworld.py::TestGridWorldPhysics -v
# Verify movement and collision logic
```

---

## TDD Workflow for Each Phase

For each file, follow this cycle **50+ times**:

### 1. RED: Write Failing Test
```python
def test_feature_x():
    result = function_under_test()
    assert result == expected_value
```

### 2. RUN: Verify it Fails
```bash
pytest streamagent/tests/test_file.py::TestClass::test_feature_x -v
# Expected: FAILED (not implemented yet)
```

### 3. GREEN: Implement Minimal Code
```python
def function_under_test():
    return expected_value  # Hardcoded is OK if test is specific enough
```

### 4. RUN: Verify it Passes
```bash
pytest streamagent/tests/test_file.py::TestClass::test_feature_x -v
# Expected: PASSED
```

### 5. REFACTOR: Improve Implementation
```python
def function_under_test():
    # Generalized implementation, still pass all tests
    return compute_value_correctly()
```

### 6. COVERAGE: Check 80%+
```bash
pytest streamagent/tests/test_file.py --cov=streamagent.engine.component
# Coverage must be 80%+ for branches, functions, lines, statements
```

---

## Mock Strategy (Avoid Heavy Dependencies)

### DO NOT Load Models
```python
# ❌ WRONG: ~10 second load, requires CUDA/Metal
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
```

### DO Mock Everything
```python
# ✅ CORRECT: Instant, no dependencies
from unittest.mock import MagicMock
model = MagicMock()
model.forward = MagicMock(return_value=MagicMock(logits=torch.randn(1, 1, 32000)))
```

### Standard Fixtures
```python
@pytest.fixture
def mock_llm_backend():
    """Mock model backend - instant, no loading"""
    backend = MagicMock()
    backend.forward = MagicMock(return_value=MagicMock(logits=torch.randn(1, 1, 32000)))
    return backend

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer - no weights"""
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: "".join(str(id) for id in ids))
    return tokenizer

@pytest.fixture
def mock_injector():
    """Mock injector for GridWorld"""
    return MagicMock(spec=ObsInjector)
```

---

## Coverage Targets

| Component | Target | Critical Tests |
|-----------|--------|-----------------|
| SinkCache | 85%+ | cache_position monotonic, FIFO eviction, pinned tokens |
| Router | 85%+ | FSM transitions, act parsing, malformed recovery |
| Injector | 80%+ | thread safety, overflow, priority |
| GridWorld | 80%+ | movement, collision, enemy patrol, termination |

**Running full coverage report:**
```bash
pytest streamagent/tests/ --cov=streamagent --cov-report=term-missing
```

---

## Testing Order within Each Phase

### SinkCache Order
1. Basic append (no eviction)
2. Single eviction (FIFO)
3. Sink preservation (never evict sinks)
4. Pinned token protection
5. Attention mask rebuild
6. **cache_position monotonicity** (MUST PASS EARLY)
7. Multiple evictions (100-token run)
8. Edge cases (empty, full, boundaries)

### Router Order
1. PASSTHROUGH state (basic passthrough)
2. MAYBE_TAG transition (saw '<')
3. IN_ACT_TAG state (confirmed act)
4. Act tag parsing (extract cmd, params)
5. Complete dispatch (call handler, return to PASSTHROUGH)
6. Malformed timeout (50-token limit)
7. Obs silencing
8. Multi-token tags
9. Edge cases (whitespace, newlines)

### Injector Order
1. Basic put() and get_pending()
2. Multiple observations batched
3. format_obs() for each type
4. Queue overflow (drop oldest)
5. Priority ordering
6. Thread safety (concurrent puts)
7. Empty queue
8. Queue persistence (until get_pending)

### GridWorld Order
1. Valid movement (N/S/E/W)
2. Wall collision
3. Boundary conditions
4. Enemy patrol loop
5. Enemy near detection (distance ≤ 2)
6. Enemy bearing calculation
7. Goal reached
8. Death event
9. Tick heartbeat
10. Max steps termination
11. Reset and render
12. Scenarios build correctly

---

## Common Test Failures and Fixes

### SinkCache: cache_position Not Monotonic
**Problem**: Position skips or repeats after eviction
**Fix**: cache_position is an absolute counter, independent from storage index
```python
# ✅ CORRECT:
self.cache_position += 1  # increment regardless of eviction
# ❌ WRONG:
self.cache_position = len(self.cache)  # tied to storage size
```

### Router: FSM Stuck in IN_ACT_TAG
**Problem**: Never transitions back to PASSTHROUGH
**Fix**: Token counter not incremented, timeout condition wrong
```python
# ✅ CORRECT:
if self.tokens_in_buffer >= 50:
    # timeout, return to PASSTHROUGH
# ❌ WRONG:
if "</>" in self.buffer:  # what if token is split?
```

### Injector: Thread Race Condition
**Problem**: Queue corrupted, observations lost
**Fix**: Add lock around queue operations
```python
import threading
self._lock = threading.Lock()

def put(self, obs):
    with self._lock:
        self.queue.append(obs)
```

### GridWorld: Event Not Detected
**Problem**: Enemy near not firing when distance = 2
**Fix**: Manhattan distance calculation
```python
# ✅ CORRECT:
dist = abs(ax - ex) + abs(ay - ey)
# ❌ WRONG:
dist = (ax - ex)**2 + (ay - ey)**2  # this is squared Euclidean
```

---

## File Locations (Absolute Paths)

```
Test files (write here):
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_sink_cache.py
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_router.py
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_injector.py
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_gridworld.py

Implementation files (write after tests pass):
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/engine/sink_cache.py
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/engine/router.py
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/engine/injector.py
  /Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/env/gridworld.py

Configuration:
  /Users/saalik/Documents/Projects/StreamingAgenticInference/TDD_IMPLEMENTATION_STRATEGY.md (full spec)
  /Users/saalik/Documents/Projects/StreamingAgenticInference/TDD_QUICK_START.md (this file)
```

---

## Quick Command Reference

```bash
# Run all tests
pytest streamagent/tests/ -v

# Run one file
pytest streamagent/tests/test_sink_cache.py -v

# Run one test
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheFIFOEviction::test_fifo_eviction_preserves_first_4_sinks -v

# Coverage report
pytest streamagent/tests/ --cov=streamagent --cov-report=term-missing

# Coverage HTML report
pytest streamagent/tests/ --cov=streamagent --cov-report=html
# Open: htmlcov/index.html

# Stop at first failure
pytest streamagent/tests/ -x

# Verbose output (show print statements)
pytest streamagent/tests/ -v -s

# Run with 4 parallel workers (faster)
pytest streamagent/tests/ -n 4
```

---

## Success Criteria

- [x] All 50+ tests written first (RED)
- [x] All tests fail initially (not implemented)
- [x] Minimal implementations written (GREEN)
- [x] All tests pass
- [x] Coverage 80%+ on all components
- [x] No model loading in any test (instant runs)
- [x] No integration with KVStream yet (Phase 5)

---

**Next Step**: Begin Phase 1 (test_sink_cache.py). Write all 12+ test cases, verify they fail, then implement SinkCache.

