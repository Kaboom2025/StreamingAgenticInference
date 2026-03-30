# StreamAgent TDD Documentation — START HERE

**Complete Test-Driven Development Specification for StreamAgent**

---

## What Is This?

This is a **test-first implementation guide** for the StreamAgent architecture. All 50+ test cases are specified before any implementation code is written, following strict TDD principles (Red → Green → Refactor).

**Timeline**: 4 days to implement all 4 components with 80%+ test coverage.

---

## 5-Minute Quick Start

### Step 1: Understand the 4 Phases

| Phase | Component | Tests | Days | Critical Invariant |
|-------|-----------|-------|------|-------------------|
| 1 | SinkCache | 12+ | 1-2 | cache_position must be strictly monotonic |
| 2 | Router | 11+ | 2-3 | FSM never stuck (50-token timeout) |
| 3 | Injector | 9+ | 3-3.5 | Thread-safe with priority ordering |
| 4 | GridWorld | 18+ | 3.5-4 | Correct physics (Manhattan distance) |

### Step 2: Choose Your Starting Point

**I want to jump right in:**
→ Read [TDD_QUICK_START.md](./TDD_QUICK_START.md) (1 page, 5 min)

**I need to understand the architecture first:**
→ Read [README_TDD.md](./README_TDD.md) (15 KB, 10 min)

**I'm implementing SinkCache tests:**
→ Read [TDD_IMPLEMENTATION_STRATEGY.md](./TDD_IMPLEMENTATION_STRATEGY.md) Part 2 (10 pages)

**I need a reference guide:**
→ Read [TDD_INDEX.md](./TDD_INDEX.md) (14 KB, quick lookup)

**I want to see all test cases:**
→ Read [TEST_STRUCTURE.md](./TEST_STRUCTURE.md) (8 KB)

### Step 3: Run Your First Test

```bash
cd /Users/saalik/Documents/Projects/StreamingAgenticInference

# Phase 1: SinkCache
pytest streamagent/tests/test_sink_cache.py -v
# Expected: 12 FAILED (tests not implemented yet)
```

---

## Document Map

```
00_START_HERE.md (this file)
├─ README_TDD.md ..................... Main entry point (15 KB)
│
├─ TDD_QUICK_START.md ............... 1-page reference for developers
│  ├─ Phase breakdown
│  ├─ FSM diagram
│  ├─ Mock strategy
│  └─ Common failures
│
├─ TDD_IMPLEMENTATION_STRATEGY.md .... Complete 50-page specification
│  ├─ Part 1: Implementation order
│  ├─ Part 2: SinkCache tests (12+)
│  ├─ Part 3: Router tests (11+)
│  ├─ Part 4: Injector tests (9+)
│  ├─ Part 5: GridWorld tests (18+)
│  ├─ Part 6: Mock strategy
│  ├─ Part 7: Local testing
│  ├─ Part 8: Coverage targets
│  ├─ Part 9: Debugging
│  └─ Part 10: Checklist
│
├─ TDD_INDEX.md ..................... Navigation guide
│  ├─ Component-by-component reference
│  ├─ Quick CLI commands
│  └─ File locations (absolute paths)
│
├─ TEST_STRUCTURE.md ................ Test inventory
│  ├─ Complete list of 50+ test cases
│  ├─ Test fixtures
│  ├─ Mock strategy
│  └─ Execution timeline
│
└─ streamagent_spec.md .............. Original architecture spec
```

---

## Component Overview

### Phase 1: SinkCache (test_sink_cache.py)

**What it tests**: KV cache with StreamingLLM eviction policy

**Key tests**:
1. FIFO eviction preserves sink tokens (0..num_sinks-1)
2. Pinned tokens (goal tokens) never evict
3. Attention mask rebuild with non-contiguous positions
4. **CRITICAL**: cache_position strictly monotonic (THE foundational invariant)
5. Multiple eviction rounds (100+ token generation)

**Why first**: All downstream components depend on correct cache behavior.

---

### Phase 2: Router (test_router.py)

**What it tests**: Token FSM router for tag detection and dispatch

**Key tests**:
1. FSM state transitions (6 total)
2. Act tag parsing: `<act cmd="move" dir="N"/>` → ActTag object
3. Malformed tag recovery: 50-token timeout
4. Observation tag silencing: `<obs>...</obs>` not echoed
5. Multi-token tag accumulation

**Why second**: Independent from SinkCache. Stateless except FSM.

---

### Phase 3: Injector (test_injector.py)

**What it tests**: Thread-safe observation queue with priority ordering

**Key tests**:
1. Thread-safe put() from concurrent threads
2. get_pending() batches and flushes
3. Queue overflow: drop oldest when full
4. Priority ordering: higher priority first
5. format_obs() generates correct XML

**Why third**: Independent from SinkCache and Router. Simple threading.

---

### Phase 4: GridWorld (test_gridworld.py)

**What it tests**: 2D game environment physics and event detection

**Key tests**:
1. Movement (N/S/E/W) and wall collision
2. Enemy patrol deterministic loop
3. Event detection (enemy_near, goal_reached, death, tick)
4. Episode termination (max_steps)
5. Correct physics (Manhattan distance)

**Why fourth**: Fully independent. No engine dependencies.

---

## How to Use These Documents

### For Quick Reference
→ **TDD_QUICK_START.md**
- One page, all essential info
- Use when you need a quick lookup

### For Complete Understanding
→ **README_TDD.md**
- 15 KB, covers everything
- Start here if this is your first time

### For Implementation Details
→ **TDD_IMPLEMENTATION_STRATEGY.md**
- 50 pages, exhaustive specification
- Use when implementing a specific component
- Jump to the relevant Part (2-5) for your component

### For Navigation
→ **TDD_INDEX.md**
- Quick reference by component
- CLI commands cheat sheet
- File locations (absolute paths)

### For Test Overview
→ **TEST_STRUCTURE.md**
- Complete test case inventory
- All 50+ tests listed and organized
- Test timeline and execution roadmap

### For Architecture
→ **streamagent_spec.md**
- Original specification
- Understand the system design
- Reference the architecture diagram

---

## Critical Success Metrics

### SinkCache (Phase 1)
- [ ] All 12+ tests pass
- [ ] cache_position is strictly monotonic (THE critical invariant)
- [ ] 85%+ coverage
- [ ] No model loading

### Router (Phase 2)
- [ ] All 11+ tests pass
- [ ] FSM transitions all working
- [ ] 50-token malformed tag timeout verified
- [ ] 85%+ coverage

### Injector (Phase 3)
- [ ] All 9+ tests pass
- [ ] Thread safety verified (4 concurrent threads)
- [ ] Priority ordering correct
- [ ] 80%+ coverage

### GridWorld (Phase 4)
- [ ] All 18+ tests pass
- [ ] Physics correct (Manhattan distance)
- [ ] All events detected correctly
- [ ] 80%+ coverage

### Overall
- [ ] 50+ tests total
- [ ] 82%+ coverage
- [ ] All tests independent
- [ ] Zero model loading (all instant)

---

## Quick Command Reference

```bash
# Run all tests
pytest streamagent/tests/ -v

# Run one phase
pytest streamagent/tests/test_sink_cache.py -v
pytest streamagent/tests/test_router.py -v
pytest streamagent/tests/test_injector.py -v
pytest streamagent/tests/test_gridworld.py -v

# Run one test class
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheMonotonicity -v

# Run one test
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheMonotonicity::test_cache_position_strictly_monotonic -v

# Coverage report
pytest streamagent/tests/ --cov=streamagent --cov-report=term-missing
pytest streamagent/tests/ --cov=streamagent --cov-report=html

# Parallel (faster)
pytest streamagent/tests/ -n 4

# Verbose (show print statements)
pytest streamagent/tests/ -v -s
```

---

## File Locations

All absolute paths (for bash calls):

```
/Users/saalik/Documents/Projects/StreamingAgenticInference/

Test files (to be created):
  streamagent/tests/test_sink_cache.py
  streamagent/tests/test_router.py
  streamagent/tests/test_injector.py
  streamagent/tests/test_gridworld.py

Implementation files (to be created after tests pass):
  streamagent/engine/sink_cache.py
  streamagent/engine/router.py
  streamagent/engine/injector.py
  streamagent/env/gridworld.py

Documentation files (already created):
  00_START_HERE.md (this file)
  README_TDD.md
  TDD_QUICK_START.md
  TDD_IMPLEMENTATION_STRATEGY.md
  TDD_INDEX.md
  TEST_STRUCTURE.md
  streamagent_spec.md
```

---

## Next Steps

### ✓ You Are Here
This is the entry point. Congratulations!

### Choose Your Next Document
1. **If you have 5 minutes**: Read [TDD_QUICK_START.md](./TDD_QUICK_START.md)
2. **If you have 10 minutes**: Read [README_TDD.md](./README_TDD.md)
3. **If you have time**: Jump to [TDD_IMPLEMENTATION_STRATEGY.md](./TDD_IMPLEMENTATION_STRATEGY.md) Part 2 (SinkCache)

### Start Phase 1
1. Create `streamagent/tests/test_sink_cache.py`
2. Write all 12+ test cases (they will all fail)
3. Implement minimal `SinkCache` class
4. Run tests until all pass
5. Verify 85%+ coverage

### Timeline
- **Days 1-2**: Phase 1 (SinkCache)
- **Days 2-3**: Phase 2 (Router)
- **Days 3-3.5**: Phase 3 (Injector)
- **Days 3.5-4**: Phase 4 (GridWorld)
- **Day 5+**: Integration tests (not in scope)

---

## Summary

| Document | Length | Purpose | Read Time |
|----------|--------|---------|-----------|
| 00_START_HERE.md | This file | Entry point | 2 min |
| TDD_QUICK_START.md | 1 page | Quick reference | 5 min |
| README_TDD.md | 15 KB | Main guide | 10 min |
| TDD_IMPLEMENTATION_STRATEGY.md | 50 KB | Complete spec | 1-2 hours |
| TDD_INDEX.md | 14 KB | Navigation | On demand |
| TEST_STRUCTURE.md | 8 KB | Test inventory | 5 min |
| streamagent_spec.md | 46 KB | Architecture | On demand |

**Total**: 3,617 lines of documentation, 50+ test cases specified.

---

**Ready? Start with [TDD_QUICK_START.md](./TDD_QUICK_START.md) →**

