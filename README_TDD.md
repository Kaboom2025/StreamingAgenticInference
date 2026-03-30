# StreamAgent TDD Implementation Guide

**Complete Test-Driven Development Specification**

---

## Overview

This guide specifies a **test-first approach** for implementing the StreamAgent architecture in 4 independent phases over 4 days. All 50+ tests are specified before any implementation code is written, following strict TDD principles.

**Key metric**: 80%+ test coverage across all components. All tests must pass in isolation with zero model loading.

---

## Documentation

### Where to Start

| Your Role | Start Here |
|-----------|-----------|
| **Developer building tests** | [TDD_QUICK_START.md](./TDD_QUICK_START.md) (1 page) |
| **Need all details** | [TDD_IMPLEMENTATION_STRATEGY.md](./TDD_IMPLEMENTATION_STRATEGY.md) (50 pages) |
| **Navigation & reference** | [TDD_INDEX.md](./TDD_INDEX.md) (14 KB) |
| **Test structure overview** | [TEST_STRUCTURE.md](./TEST_STRUCTURE.md) (8 KB) |
| **Architecture & spec** | [streamagent_spec.md](./streamagent_spec.md) (original spec) |

### Document Contents

**TDD_QUICK_START.md** — Quick reference
- Phase breakdown (what to implement when)
- FSM diagram for Router
- Mock strategy
- Common failures and fixes
- CLI command cheatsheet

**TDD_IMPLEMENTATION_STRATEGY.md** — Complete specification
- Part 1: Implementation order and timeline
- Part 2: SinkCache tests (12+ tests, exact assertions)
- Part 3: Router tests (11+ tests, exact assertions)
- Part 4: Injector tests (9+ tests, exact assertions)
- Part 5: GridWorld tests (18+ tests, exact assertions)
- Part 6: Mocking strategy (global)
- Part 7: Local test execution
- Part 8: Coverage targets
- Part 9: Debugging guide
- Part 10: Checklist

**TDD_INDEX.md** — Navigation guide
- For each component: file location, test count, target coverage
- Test specification details (cross-references to full spec)
- Implementation order
- Common issues and solutions
- Testing commands cheat sheet

**TEST_STRUCTURE.md** — Test inventory
- Complete list of 50+ test cases organized by class
- Fixture definitions for each component
- Mock strategy summary
- Test execution timeline
- Coverage verification commands

---

## Quick Start (5 Minutes)

### 1. Read the One-Page Reference
```bash
open TDD_QUICK_START.md
```

### 2. Understand the 4 Phases

| Phase | Component | Tests | Days | Why? |
|-------|-----------|-------|------|------|
| 1 | SinkCache | 12+ | 1-2 | Foundation: cache invariants |
| 2 | Router | 11+ | 2-3 | FSM: stateful tag detection |
| 3 | Injector | 9+ | 3-3.5 | Queue: threading |
| 4 | GridWorld | 18+ | 3.5-4 | Physics: independent env |

### 3. Start Phase 1
```bash
cd /Users/saalik/Documents/Projects/StreamingAgenticInference

# Create test file
touch streamagent/tests/test_sink_cache.py

# Read the spec
less TDD_IMPLEMENTATION_STRATEGY.md  # Jump to Part 2

# Write all 12+ failing tests
# Run: pytest streamagent/tests/test_sink_cache.py -v
# Expected: 12 FAILED (not implemented yet)
```

### 4. Follow Red-Green-Refactor
```python
# RED: Write failing test
def test_fifo_eviction_preserves_first_4_sinks():
    cache = SinkCache(num_sinks=4, window_length=8)
    # append tokens, verify sinks never evicted
    assert cache.evicted_count == 1

# RUN: Verify it fails
# pytest streamagent/tests/test_sink_cache.py::TestSinkCacheFIFOEviction::test_fifo_eviction_preserves_first_4_sinks -v
# Expected: FAILED

# GREEN: Implement minimal code
class SinkCache:
    def evict_if_needed(self):
        # Evict oldest non-sink token
        if len(self.cache) > self.capacity:
            self.evicted_positions.append(self.cache[self.num_sinks])
            self.cache.pop(self.num_sinks)

# RUN: Verify it passes
# pytest streamagent/tests/test_sink_cache.py::TestSinkCacheFIFOEviction::test_fifo_eviction_preserves_first_4_sinks -v
# Expected: PASSED

# REFACTOR: Improve implementation (keep tests green)
# Coverage: pytest streamagent/tests/test_sink_cache.py --cov --cov-report=term-missing
```

---

## Detailed Breakdown

### Phase 1: SinkCache (Days 1-2)

**What to implement**: `streamagent/engine/sink_cache.py`

**Test file**: `streamagent/tests/test_sink_cache.py`

**Key test cases**:
1. FIFO eviction preserves sink tokens (0..num_sinks-1)
2. Pinned tokens (goal tokens) never evict
3. Attention mask rebuild with non-contiguous positions
4. **CRITICAL**: cache_position strictly monotonic (no skips, no repeats)
5. Multiple eviction rounds (100+ token generation)

**Why first**: All downstream components depend on correct cache behavior. If cache_position drifts, RoPE attention collapses.

**Pass criteria**:
- All 12+ tests green
- 85%+ coverage
- No model loading
- <1 second per test

---

### Phase 2: Router (Days 2-3)

**What to implement**: `streamagent/engine/router.py`

**Test file**: `streamagent/tests/test_router.py`

**Key test cases**:
1. FSM state transitions (6 total):
   - PASSTHROUGH → MAYBE_TAG (saw '<')
   - MAYBE_TAG → IN_ACT_TAG (saw 'act')
   - IN_ACT_TAG → PASSTHROUGH (saw '/>', dispatch)
   - PASSTHROUGH (malformed, timeout)
   - MAYBE_TAG → IN_OBS_TAG (saw 'obs')
   - IN_OBS_TAG → PASSTHROUGH (silence)

2. Act tag parsing: `<act cmd="move" dir="N"/>` → ActTag(cmd="move", params={"dir": "N"})
3. Malformed tag timeout: 50 tokens in IN_ACT_TAG state
4. Observation tag silencing: `<obs>...</obs>` not echoed

**Why second**: Independent from SinkCache. Stateless except FSM. Deterministic.

**Pass criteria**:
- All 11+ tests green
- 85%+ coverage
- FSM never stuck (timeout works)
- Multi-token tags accumulate correctly

---

### Phase 3: Injector (Days 3-3.5)

**What to implement**: `streamagent/engine/injector.py`

**Test file**: `streamagent/tests/test_injector.py`

**Key test cases**:
1. Thread-safe put() from 4 concurrent threads
2. get_pending() returns all pending obs and flushes queue
3. Queue overflow: max_queue_size=32, drop oldest
4. Priority ordering: higher priority first
5. format_obs() generates: `<obs type="collision">wall at N</obs>`

**Why third**: Independent from SinkCache and Router. Simple threading.

**Pass criteria**:
- All 9+ tests green
- 80%+ coverage
- Thread-safe (no race conditions)
- Priority ordering correct

---

### Phase 4: GridWorld (Days 3.5-4)

**What to implement**: `streamagent/env/gridworld.py`

**Test file**: `streamagent/tests/test_gridworld.py`

**Key test cases**:
1. Movement (N/S/E/W): position updates correctly
2. Wall collision: blocked movement, collision event
3. Enemy patrol: deterministic loop, path_index update
4. Event detection:
   - enemy_near: distance ≤ 2, bearing (N/S/E/W/NE/NW/SE/SW)
   - goal_reached: agent_pos == goal_pos
   - death: agent_pos == enemy_pos, done=True
   - tick: every 10 steps
5. Episode termination: max_steps reached

**Why fourth**: Fully independent. No engine dependencies except interfaces.

**Pass criteria**:
- All 18+ tests green
- 80%+ coverage
- Physics correct (Manhattan distance, patrol loops)
- All events detected correctly

---

## Key Test Files Location

All files absolute paths (required for bash calls):

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
```

---

## Critical Invariants

### SinkCache: cache_position Must Be Strictly Monotonic

```python
# This is the SMOKING GUN test
def test_cache_position_strictly_monotonic():
    cache = SinkCache(num_sinks=4, window_length=8)

    for i in range(100):
        cache.update(...)
        # Position counter must be exactly i, no skips or repeats
        assert cache.cache_position == i

    # Verify sequence
    assert positions == list(range(100))
```

**Why critical**: If cache_position drifts, RoPE (rotary position embedding) collapses and attention fails. This is THE invariant the entire system depends on.

### Router: FSM Never Stuck in IN_ACT_TAG

```python
# Router must timeout and recover
def test_malformed_tag_50_token_timeout():
    router = Router()

    # Feed 50+ garbage tokens after '<act' (no completion)
    for _ in range(50):
        router.feed(Token(id=999, text="garbage"))

    # Must timeout, log event, return to PASSTHROUGH
    assert router.state == RouterState.PASSTHROUGH
    assert router.malformed_tag_count == 1
```

### Injector: Thread Safety

```python
# Multiple threads must not corrupt queue
def test_thread_safe_concurrent_puts():
    injector = ObsInjector(max_queue_size=32)

    def thread_puts(thread_id):
        for i in range(10):
            obs = Observation(type=f"event_{thread_id}_{i}", payload="")
            injector.put(obs)

    # 4 threads × 10 puts = 40 observations
    # Queue size 32, so oldest 8 dropped (FIFO)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        [executor.submit(thread_puts, i) for i in range(4)]

    pending = injector.get_pending()
    assert len(pending) == 32  # Not corrupted, exactly 32
```

### GridWorld: Correct Physics

```python
# Manhattan distance, not Euclidean
def test_enemy_near_distance_calculation():
    agent = (5, 5)
    enemy = (5, 3)  # distance = |5-5| + |5-3| = 2

    dist = abs(agent[0] - enemy[0]) + abs(agent[1] - enemy[1])
    assert dist == 2
    assert dist <= 2  # enemy_near event fires
```

---

## Mock Strategy (Critical for Speed)

### DO NOT Load Models

```python
# ❌ WRONG: ~10 second load
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
```

### DO Use Instant Mocks

```python
# ✅ CORRECT: Instant, no loading
from unittest.mock import MagicMock
import torch

model = MagicMock()
model.forward = MagicMock(return_value=MagicMock(logits=torch.randn(1, 1, 32000)))
```

### Standard Fixtures

All components use these fixtures:

```python
@pytest.fixture
def mock_llm_backend():
    backend = MagicMock()
    backend.forward = MagicMock(return_value=MagicMock(logits=torch.randn(1, 1, 32000)))
    return backend

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: "".join(str(id) for id in ids))
    return tokenizer

@pytest.fixture
def mock_injector():
    return MagicMock(spec=ObsInjector)
```

---

## Coverage Targets

```
Component       Target  Why
─────────────   ─────   ──────────────────────────
SinkCache       85%+    Critical: cache invariants
Router          85%+    Critical: FSM correctness
Injector        80%+    Supporting: queue ops
GridWorld       80%+    Supporting: physics

Overall         82%+    All components
```

**Verification**:
```bash
pytest streamagent/tests/ --cov=streamagent --cov-report=term-missing
# Must show 80%+ for branches, functions, lines, statements
```

---

## Testing Commands

### Run All Tests
```bash
pytest streamagent/tests/ -v
```

### Run One Phase
```bash
pytest streamagent/tests/test_sink_cache.py -v
pytest streamagent/tests/test_router.py -v
pytest streamagent/tests/test_injector.py -v
pytest streamagent/tests/test_gridworld.py -v
```

### Run One Test Class
```bash
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheMonotonicity -v
```

### Run One Test
```bash
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheMonotonicity::test_cache_position_strictly_monotonic -v
```

### Coverage Report
```bash
pytest streamagent/tests/ --cov=streamagent --cov-report=term-missing
pytest streamagent/tests/ --cov=streamagent --cov-report=html
# Open htmlcov/index.html in browser
```

### Parallel Execution (Faster)
```bash
pip install pytest-xdist
pytest streamagent/tests/ -n 4  # 4 parallel workers
```

### Verbose Output
```bash
pytest streamagent/tests/ -v -s  # Show print statements
```

### Stop at First Failure
```bash
pytest streamagent/tests/ -x
```

---

## Common Failures and Solutions

### cache_position Not Monotonic
**Problem**: Position skips or repeats
**Fix**: cache_position is absolute counter, increment before/after every update, regardless of eviction
```python
# ✅ CORRECT
self.cache_position += 1

# ❌ WRONG
self.cache_position = len(self.cache)  # Tied to size
```

### Router FSM Stuck in IN_ACT_TAG
**Problem**: Never transitions back
**Fix**: Token counter not incremented, or timeout condition wrong
```python
# ✅ CORRECT
if self.tokens_in_buffer >= 50:
    # timeout, recover

# ❌ WRONG
if "</>" in self.buffer:  # What if split across tokens?
```

### Thread Race in Injector
**Problem**: Queue corrupted or observations lost
**Fix**: Add lock around all queue operations
```python
import threading
self._lock = threading.Lock()

def put(self, obs):
    with self._lock:
        self.queue.append(obs)
```

### Enemy Near Not Detected
**Problem**: Distance = 2 but no event
**Fix**: Manhattan distance calculation
```python
# ✅ CORRECT
dist = abs(ax - ex) + abs(ay - ey)

# ❌ WRONG
dist = (ax - ex)**2 + (ay - ey)**2  # Squared Euclidean
```

---

## Success Metrics

- [x] 50+ tests specified before any implementation
- [x] All tests fail initially (RED state)
- [x] Minimal implementations written (GREEN state)
- [x] All tests pass
- [x] 82%+ coverage across all components
- [x] Zero model loading (instant test runs)
- [x] All tests independent (can run in any order)

---

## Next Steps

1. **Read** TDD_QUICK_START.md (1 page, 5 min)
2. **Understand** test structure (TEST_STRUCTURE.md, 5 min)
3. **Start Phase 1**: Write all SinkCache tests (2 hours)
4. **Run**: Verify all tests fail (RED)
5. **Implement**: Minimal SinkCache code (4 hours)
6. **Run**: Verify all tests pass (GREEN)
7. **Repeat** for Phases 2-4

**Estimated timeline**: 4 days to all 50+ tests passing, 80%+ coverage.

---

## Documentation Files

All files in `/Users/saalik/Documents/Projects/StreamingAgenticInference/`:

- **README_TDD.md** ← You are here
- **TDD_QUICK_START.md** — One-page reference
- **TDD_IMPLEMENTATION_STRATEGY.md** — Complete 50-page specification
- **TDD_INDEX.md** — Navigation and cross-reference
- **TEST_STRUCTURE.md** — Test inventory (50+ cases)
- **streamagent_spec.md** — Original architecture specification

---

**Ready to begin? Start with TDD_QUICK_START.md, then implement Phase 1 (SinkCache).**

