# StreamAgent TDD Documentation Index

**Complete reference for all test specifications and implementation order**

---

## Document Overview

| Document | Purpose | Audience | Length |
|----------|---------|----------|--------|
| **TDD_QUICK_START.md** | One-page reference | Developers starting Phase 1 | 1 page |
| **TDD_IMPLEMENTATION_STRATEGY.md** | Complete test specifications | Test writers, implementation engineers | 50 pages |
| **This file (TDD_INDEX.md)** | Navigation and cross-reference | All readers | This file |

---

## Part A: For Developers (Start Here)

### If you're implementing test_sink_cache.py first:
1. Read: **TDD_QUICK_START.md** → Phase 1: SinkCache section
2. Read: **TDD_IMPLEMENTATION_STRATEGY.md** → Part 2: SinkCache Test Specification
3. Key assertion to nail: `cache_position` must be **strictly monotonic**
4. Run: `pytest streamagent/tests/test_sink_cache.py::TestSinkCacheMonotonicity -v`

### If you're implementing test_router.py:
1. Read: **TDD_QUICK_START.md** → Phase 2: Router section
2. Read: **TDD_IMPLEMENTATION_STRATEGY.md** → Part 3: Router Test Specification
3. Key diagram: FSM state machine (6 transitions)
4. Run: `pytest streamagent/tests/test_router.py::TestRouterFSM -v`

### If you're implementing test_injector.py:
1. Read: **TDD_QUICK_START.md** → Phase 3: Injector section
2. Read: **TDD_IMPLEMENTATION_STRATEGY.md** → Part 4: Injector Test Specification
3. Key requirement: Thread-safe queue with priority ordering
4. Run: `pytest streamagent/tests/test_injector.py::TestInjectorThreadSafety -v`

### If you're implementing test_gridworld.py:
1. Read: **TDD_QUICK_START.md** → Phase 4: GridWorld section
2. Read: **TDD_IMPLEMENTATION_STRATEGY.md** → Part 5: GridWorld Test Specification
3. Key mechanics: Manhattan distance, patrol loops, event injection
4. Run: `pytest streamagent/tests/test_gridworld.py::TestGridWorldPhysics -v`

---

## Part B: Test Specification Details

### SinkCache Tests (test_sink_cache.py)

**Location in full spec**: TDD_IMPLEMENTATION_STRATEGY.md → Part 2

**Test count**: 12+
**Target coverage**: 85%+
**Implementation file**: streamagent/engine/sink_cache.py

**Key test cases**:
1. FIFO eviction preserves sinks (0..num_sinks-1)
2. Pinned tokens (goal tokens) never evict
3. Attention mask rebuild with non-contiguous positions
4. **CRITICAL: cache_position monotonicity** (no skips, no repeats)
5. Multiple eviction rounds (100+ tokens)
6. KV tensor shape validation
7. Quantization edge cases

**Quick reference**:
```python
# The CRITICAL test every other component depends on:
def test_cache_position_strictly_monotonic():
    for i in range(100):
        cache.update(...)
        assert cache.cache_position == i  # MUST be exactly i, no skips
```

**Mocking strategy**: No model loading. Use torch.randn() for KV tensors.

---

### Router Tests (test_router.py)

**Location in full spec**: TDD_IMPLEMENTATION_STRATEGY.md → Part 3

**Test count**: 11+
**Target coverage**: 85%+
**Implementation file**: streamagent/engine/router.py

**Key test cases**:
1. FSM: PASSTHROUGH → MAYBE_TAG → IN_ACT_TAG → PASSTHROUGH
2. Act tag parsing: `<act cmd="move" dir="N"/>` → ActTag object
3. Malformed tag timeout: 50 tokens in IN_ACT_TAG, then fail
4. Observation tag silencing: `<obs>...</obs>` not echoed
5. Multi-token tag accumulation (tag spans multiple tokens)
6. Whitespace handling (extra spaces, newlines)
7. Handler dispatch (call registered handler)

**FSM diagram**:
```
PASSTHROUGH ──(<)──► MAYBE_TAG
MAYBE_TAG ──(act)──► IN_ACT_TAG ──(/>)──► PASSTHROUGH [dispatch handler]
MAYBE_TAG ──(obs)──► IN_OBS_TAG ──</obs>──► PASSTHROUGH [silence]
MAYBE_TAG ──(other)──► PASSTHROUGH [emit '<']
```

**Mocking strategy**: No tokenizer loading. Use MagicMock with side_effect.

---

### Injector Tests (test_injector.py)

**Location in full spec**: TDD_IMPLEMENTATION_STRATEGY.md → Part 4

**Test count**: 9+
**Target coverage**: 80%+
**Implementation file**: streamagent/engine/injector.py

**Key test cases**:
1. Thread-safe put() from multiple threads
2. get_pending() batches all observations and flushes
3. Queue overflow: drop oldest when max_queue_size exceeded
4. Priority ordering: higher priority first
5. format_obs() generates correct XML
6. Empty queue handling
7. Queue persistence until get_pending()

**Quick API reference**:
```python
injector.put(obs: Observation)  # thread-safe, non-blocking
pending = injector.get_pending()  # returns List[str], flushes queue
formatted = injector.format_obs(obs: Observation)  # → '<obs type="...">...</obs>'
```

**Mocking strategy**: No external dependencies. Observation is simple dataclass.

---

### GridWorld Tests (test_gridworld.py)

**Location in full spec**: TDD_IMPLEMENTATION_STRATEGY.md → Part 5

**Test count**: 18+
**Target coverage**: 80%+
**Implementation file**: streamagent/env/gridworld.py

**Key test cases**:
1. Movement: N/S/E/W all cardinal directions
2. Wall blocking: position unchanged, collision event
3. Boundary conditions: grid edge as wall
4. Enemy patrol: deterministic loop, path_index update
5. Enemy near detection: distance ≤ 2, bearing calculation
6. Goal reached: agent_pos == goal_pos → done=True
7. Death: agent_pos == enemy_pos → done=True, priority=10
8. Tick heartbeat: every 10 steps
9. Max steps termination: step_count >= max_steps → done=True
10. Event priority ordering in injector
11. Reset and render output

**Physics formulas**:
```python
# Movement
new_pos = (pos[0] + dx, pos[1] + dy)  # (col, row)
where: N=(0,-1), S=(0,+1), E=(+1,0), W=(-1,0)

# Distance
manhattan = abs(ax - ex) + abs(ay - ey)

# Bearing (8 directions)
if dy < 0: bearing = "N"
if dy > 0: bearing = "S"
if dx > 0: bearing = "E"
if dx < 0: bearing = "W"
# Combine: NE, NW, SE, SW
```

**Mocking strategy**: Mock injector to verify put() calls. No model, no backend.

---

## Part C: Implementation Order

### Phase 1: SinkCache (Days 1-2)

**Why first**: All generation depends on cache invariants. Cache position monotonicity must be rock-solid before proceeding.

**Steps**:
1. Write all 12+ tests (all fail initially)
2. Implement minimal SinkCache to pass first test (FIFO eviction)
3. Implement pinned token protection
4. Implement attention mask rebuild (critical for non-contiguous positions)
5. **VERIFY: cache_position monotonicity** (run test 50+ times)
6. Implement edge cases (multiple evictions, quantization)
7. Run full coverage: `pytest test_sink_cache.py --cov --cov-report=term-missing`
8. Target: 85%+ coverage

**Pass criteria**: All tests green, no skipped assertions

---

### Phase 2: Router FSM (Days 2-3)

**Why second**: Independent from SinkCache. Stateless except for FSM. No model needed.

**Steps**:
1. Write all 11+ tests (all fail initially)
2. Implement FSM state machine (6 transitions)
3. Implement PASSTHROUGH state
4. Implement MAYBE_TAG detection
5. Implement IN_ACT_TAG state and tag parsing
6. Implement malformed tag timeout (50-token limit)
7. Implement obs tag silencing
8. Implement handler dispatch
9. Run full coverage
10. Target: 85%+ coverage

**Pass criteria**: All FSM transitions tested and working, malformed tags timeout correctly

---

### Phase 3: Injector (Days 3-3.5)

**Why third**: Independent from SinkCache and Router. Builds only on dataclasses.

**Steps**:
1. Write all 9+ tests (all fail initially)
2. Implement basic queue (put, get_pending)
3. Implement queue overflow (max_queue_size)
4. Implement priority ordering
5. Implement format_obs() for all types
6. Add thread-safety locks
7. Run full coverage
8. Target: 80%+ coverage

**Pass criteria**: Concurrent puts don't corrupt state, priority ordering verified

---

### Phase 4: GridWorld (Days 3.5-4)

**Why fourth**: Fully independent environment. Physics engine, not inference engine.

**Steps**:
1. Write all 18+ tests (all fail initially)
2. Implement movement (N/S/E/W)
3. Implement wall collision detection
4. Implement boundary conditions
5. Implement enemy patrol loop
6. Implement event detection (enemy_near, goal_reached, death)
7. Implement tick heartbeat
8. Implement episode termination
9. Implement reset and render
10. Run full coverage
11. Target: 80%+ coverage

**Pass criteria**: All physics correct, all events detected, episode terminates properly

---

### Phase 5: Integration (Day 5+, sketch only)

**Out of scope for this TDD specification** (will be written by another agent).

---

## Part D: Quick Reference by Component

### SinkCache
- **File**: streamagent/engine/sink_cache.py
- **Tests**: streamagent/tests/test_sink_cache.py
- **Critical invariant**: cache_position strictly monotonic
- **Key method**: `evict_if_needed()` + `get_attention_mask()`
- **Coverage target**: 85%+

### Router
- **File**: streamagent/engine/router.py
- **Tests**: streamagent/tests/test_router.py
- **Critical invariant**: FSM never stuck (50-token timeout)
- **Key method**: `feed(token)` → RouterOutput
- **Coverage target**: 85%+

### Injector
- **File**: streamagent/engine/injector.py
- **Tests**: streamagent/tests/test_injector.py
- **Critical invariant**: Thread-safe, priority-ordered
- **Key method**: `put(obs)`, `get_pending()`
- **Coverage target**: 80%+

### GridWorld
- **File**: streamagent/env/gridworld.py
- **Tests**: streamagent/tests/test_gridworld.py
- **Critical invariant**: Deterministic physics, correct event firing
- **Key method**: `step(action)` → (done, info)
- **Coverage target**: 80%+

---

## Part E: Common Issues and Solutions

### Issue: cache_position Not Monotonic (SinkCache)
**Symptom**: Test fails at positions 0, 1, 2, ... 99 but position skips or repeats
**Root cause**: cache_position tied to internal storage index instead of absolute counter
**Solution**: Increment position counter before/after every update, independently of evictions
**Test**: Run `test_cache_position_strictly_monotonic` 100+ times

### Issue: FSM Stuck in IN_ACT_TAG (Router)
**Symptom**: Router never returns to PASSTHROUGH after seeing `<act`
**Root cause**: Token counter not incremented, or timeout condition never triggers
**Solution**: Increment counter in every `feed()` call, timeout >= 50 tokens
**Test**: Run `test_malformed_tag_50_token_timeout`

### Issue: Thread Corruption in Injector
**Symptom**: Observations lost or queue size exceeds limit with concurrent puts
**Root cause**: No synchronization (lock) on queue operations
**Solution**: Use `threading.Lock()` around put() and get_pending()
**Test**: Run `test_thread_safe_concurrent_puts` with 4 threads

### Issue: Enemy Near Not Detected (GridWorld)
**Symptom**: Enemy at Manhattan distance 2 but no event fired
**Root cause**: Distance calculation uses squared Euclidean instead of Manhattan
**Solution**: dist = abs(dx) + abs(dy), not sqrt(dx² + dy²)
**Test**: Run `test_enemy_near_detection` with explicit positions

---

## Part F: Testing Commands Cheat Sheet

```bash
# Phase 1: SinkCache
pytest streamagent/tests/test_sink_cache.py -v
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheMonotonicity::test_cache_position_strictly_monotonic -v
pytest streamagent/tests/test_sink_cache.py --cov=streamagent.engine.sink_cache --cov-report=term-missing

# Phase 2: Router
pytest streamagent/tests/test_router.py -v
pytest streamagent/tests/test_router.py::TestRouterFSM -v
pytest streamagent/tests/test_router.py --cov=streamagent.engine.router --cov-report=term-missing

# Phase 3: Injector
pytest streamagent/tests/test_injector.py -v
pytest streamagent/tests/test_injector.py::TestInjectorThreadSafety -v
pytest streamagent/tests/test_injector.py --cov=streamagent.engine.injector --cov-report=term-missing

# Phase 4: GridWorld
pytest streamagent/tests/test_gridworld.py -v
pytest streamagent/tests/test_gridworld.py::TestGridWorldPhysics -v
pytest streamagent/tests/test_gridworld.py --cov=streamagent.env.gridworld --cov-report=term-missing

# All tests
pytest streamagent/tests/ -v
pytest streamagent/tests/ --cov=streamagent --cov-report=html

# With parallel workers (faster)
pytest streamagent/tests/ -n 4

# Stop at first failure
pytest streamagent/tests/ -x

# Show print statements
pytest streamagent/tests/ -v -s
```

---

## Part G: File Locations (Absolute Paths)

### Test Files (Write Here)
```
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_sink_cache.py
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_router.py
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_injector.py
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/tests/test_gridworld.py
```

### Implementation Files (Write After Tests Pass)
```
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/engine/sink_cache.py
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/engine/router.py
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/engine/injector.py
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent/env/gridworld.py
```

### Documentation Files
```
/Users/saalik/Documents/Projects/StreamingAgenticInference/TDD_QUICK_START.md
/Users/saalik/Documents/Projects/StreamingAgenticInference/TDD_IMPLEMENTATION_STRATEGY.md (this was created)
/Users/saalik/Documents/Projects/StreamingAgenticInference/TDD_INDEX.md (this file)
/Users/saalik/Documents/Projects/StreamingAgenticInference/streamagent_spec.md
```

---

## Part H: Key Sections in Full Spec

### For SinkCache Deep Dive
- TDD_IMPLEMENTATION_STRATEGY.md → Part 2: Sections 1-7
- Focus on: Test cases 1-7, especially "Cache Position Monotonicity"

### For Router Deep Dive
- TDD_IMPLEMENTATION_STRATEGY.md → Part 3: Sections 1-11
- Focus on: Test cases 4 (complete dispatch), 7 (malformed timeout)

### For Injector Deep Dive
- TDD_IMPLEMENTATION_STRATEGY.md → Part 4: Sections 1-9
- Focus on: Test cases 6 (thread safety), 4 (overflow)

### For GridWorld Deep Dive
- TDD_IMPLEMENTATION_STRATEGY.md → Part 5: Sections 1-18
- Focus on: Test cases 7 (death), 8 (bearing), 13 (tick)

---

## Summary

| Phase | File | Tests | Days | Status |
|-------|------|-------|------|--------|
| 1 | test_sink_cache.py | 12+ | 1-2 | Tests specified ✓ |
| 2 | test_router.py | 11+ | 2-3 | Tests specified ✓ |
| 3 | test_injector.py | 9+ | 3-3.5 | Tests specified ✓ |
| 4 | test_gridworld.py | 18+ | 3.5-4 | Tests specified ✓ |
| 5 | test_kv_stream.py | TBD | 5+ | To be created |

**Total: 50+ unit tests specified, 0% complete (all in RED state)**

---

**Next Step**: Choose a phase above and begin writing tests. All tests must fail before any implementation code is written.

