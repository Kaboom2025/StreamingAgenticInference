# StreamAgent Test Structure Roadmap

**Complete test class and case inventory for all 4 components**

---

## test_sink_cache.py (12+ Tests)

### TestSinkCacheFIFOEviction
```python
def test_fifo_eviction_preserves_first_4_sinks()
def test_fifo_eviction_sequence()
def test_multiple_eviction_rounds()
```

### TestSinkCacheAttentionMask
```python
def test_attention_mask_non_contiguous_after_eviction()
def test_get_attention_mask_correct_shape_and_values()
def test_attention_mask_causal_property()
```

### TestSinkCachePinnedTokens
```python
def test_pinned_tokens_never_evict()
def test_pin_token_api()
```

### TestSinkCacheMonotonicity (CRITICAL)
```python
def test_cache_position_strictly_monotonic()
def test_cache_position_never_skips_or_repeats()
```

### TestSinkCacheKVTensors
```python
def test_kv_tensors_valid_shape_after_eviction()
def test_kv_tensors_no_nan_or_inf()
```

---

## test_router.py (11+ Tests)

### TestRouterFSM
```python
def test_fsm_passthrough_to_maybe_tag()
def test_fsm_maybe_tag_to_in_act_tag()
def test_fsm_maybe_tag_to_passthrough_false_alarm()
def test_fsm_in_act_tag_complete_dispatch()
def test_fsm_in_obs_tag_silencing()
```

### TestRouterActTagParsing
```python
def test_act_tag_parsing_single_param()
def test_act_tag_parsing_multiple_params()
def test_act_tag_parsing_self_closing()
def test_act_tag_parsing_with_content()
```

### TestRouterMalformedTags
```python
def test_malformed_tag_50_token_timeout()
def test_incomplete_act_tag_at_episode_end()
```

### TestRouterMultiTokenTags
```python
def test_multi_token_tag_act_spread_across_tokens()
def test_tag_detection_with_extra_whitespace()
def test_act_tag_newlines_in_tag()
```

### TestRouterObsTagSilencing
```python
def test_obs_tag_completely_silenced()
def test_obs_tag_handler_not_called()
```

---

## test_injector.py (9+ Tests)

### TestInjectorQueueManagement
```python
def test_put_single_obs_get_pending()
def test_get_pending_batches_multiple_obs()
def test_get_pending_empty_queue()
def test_queue_persists_until_get_pending()
```

### TestInjectorObsFormatting
```python
def test_format_obs_collision()
def test_format_obs_enemy_near_with_attributes()
def test_format_obs_full_grid()
```

### TestInjectorOverflow
```python
def test_overflow_drops_oldest()
```

### TestInjectorPriority
```python
def test_priority_ordering_higher_first()
```

### TestInjectorThreadSafety
```python
def test_thread_safe_concurrent_puts()
```

---

## test_gridworld.py (18+ Tests)

### TestGridWorldPhysics
```python
def test_agent_moves_north()
def test_movement_all_cardinal_directions()
def test_agent_blocked_by_wall()
def test_agent_blocked_at_grid_boundary()
```

### TestGridWorldEnemyPatrol
```python
def test_enemy_patrol_deterministic_loop()
def test_enemy_speed_2_advances_2_steps_per_tick()
```

### TestGridWorldEventDetection
```python
def test_enemy_near_event_triggers_at_distance_2()
def test_enemy_near_bearing_direction()
def test_no_enemy_near_event_at_distance_3()
def test_goal_near_distance_2()
def test_goal_reached_event()
def test_death_event_agent_on_enemy()
```

### TestGridWorldTickHeartbeat
```python
def test_tick_heartbeat_every_10_steps()
```

### TestGridWorldTermination
```python
def test_episode_terminates_at_max_steps()
```

### TestGridWorldIntegration
```python
def test_event_priority_ordering_in_injection()
def test_reset_returns_full_grid_obs()
def test_render_ascii_grid()
```

### TestGridWorldScenarios
```python
def test_scenario_static_maze_builds()
def test_scenario_single_patrol_builds()
def test_scenario_dual_patrol_builds()
```

---

## Test Statistics

```
┌─────────────────────┬───────┬────────┬──────────┐
│ Component           │ Tests │ Target │ Days     │
├─────────────────────┼───────┼────────┼──────────┤
│ SinkCache           │ 12+   │ 85%+   │ 1-2      │
│ Router              │ 11+   │ 85%+   │ 2-3      │
│ Injector            │ 9+    │ 80%+   │ 3-3.5    │
│ GridWorld           │ 18+   │ 80%+   │ 3.5-4    │
├─────────────────────┼───────┼────────┼──────────┤
│ TOTAL               │ 50+   │ 82%+   │ 4 days   │
└─────────────────────┴───────┴────────┴──────────┘
```

---

## Test Fixtures by Component

### SinkCache Fixtures
```python
@pytest.fixture
def simple_cache()
    # num_layers=2, num_heads=4, head_dim=64, num_sinks=4, window_length=8

@pytest.fixture
def cache_with_pinned()
    # Cache with pinned token IDs pre-marked

@pytest.fixture
def mock_kv_tensors()
    # Generator for random KV tensors of appropriate shape
```

### Router Fixtures
```python
@pytest.fixture
def router()
    # Fresh Router instance

@pytest.fixture
def router_with_handlers()
    # Router with mock handlers registered for "move", "wait"

@pytest.fixture
def mock_tokenizer()
    # Mock tokenizer with decode side_effect
```

### Injector Fixtures
```python
@pytest.fixture
def injector()
    # Fresh injector with default queue size

@pytest.fixture
def injector_small()
    # Injector with max_queue_size=3 for overflow testing

@pytest.fixture
def sample_observations()
    # Dict of reusable Observation objects
```

### GridWorld Fixtures
```python
@pytest.fixture
def gridworld()
    # Fresh GridWorld instance (10x10)

@pytest.fixture
def gridworld_reset()
    # GridWorld after reset to "static_maze"

@pytest.fixture
def mock_injector()
    # Mock ObsInjector to capture put() calls
```

---

## Mock Strategy Summary

### Mocks to Use (Instant, No Loading)
```python
# Model backend
backend = MagicMock()
backend.forward = MagicMock(return_value=MagicMock(logits=torch.randn(1, 1, 32000)))

# Tokenizer
tokenizer = MagicMock()
tokenizer.decode = MagicMock(side_effect=lambda ids: "decoded_text")

# Injector (for GridWorld tests)
injector = MagicMock(spec=ObsInjector)
injector.put = MagicMock()

# KV tensors (for SinkCache tests)
key = torch.randn(batch, heads, seq_len, head_dim)
value = torch.randn(batch, heads, seq_len, head_dim)
```

### What NOT to Mock (Real Objects)
```python
# Real dataclasses
obs = Observation(type="collision", payload="wall")
action = Action(cmd="move", params={"dir": "N"})

# Real physics logic
dist = abs(dx) + abs(dy)  # Manhattan distance

# Real grid state
state = GridState(width=10, height=10, ...)
```

---

## Test Execution Timeline

### Day 1 Morning
- [ ] Write all test_sink_cache.py tests (all RED)
- [ ] Verify they all fail
- [ ] Run: `pytest test_sink_cache.py -v` → 12 FAILED

### Day 1 Afternoon
- [ ] Implement SinkCache.update() → test FIFO eviction
- [ ] Implement SinkCache.evict_if_needed() → test multiple rounds
- [ ] Implement SinkCache.pin_token() → test pinned tokens
- [ ] Implement SinkCache.get_attention_mask() → test non-contiguous

### Day 1 Evening
- [ ] **CRITICAL**: Implement cache_position monotonicity
- [ ] Run: `pytest test_sink_cache.py::TestSinkCacheMonotonicity -v` → PASSED
- [ ] Run full suite: `pytest test_sink_cache.py -v` → All GREEN

### Day 2 Morning
- [ ] Coverage check: `pytest test_sink_cache.py --cov` → Target 85%+
- [ ] Write all test_router.py tests (all RED)
- [ ] Verify they all fail

### Day 2 Afternoon
- [ ] Implement Router FSM state transitions
- [ ] Implement PASSTHROUGH state handling
- [ ] Implement MAYBE_TAG detection (`<`)
- [ ] Implement IN_ACT_TAG state

### Day 2 Evening
- [ ] Implement ActTag parsing (XML extraction)
- [ ] Implement malformed tag timeout (50-token limit)
- [ ] Implement obs tag silencing
- [ ] Run: `pytest test_router.py -v` → All GREEN

### Day 3 Morning
- [ ] Coverage check: `pytest test_router.py --cov` → Target 85%+
- [ ] Write all test_injector.py tests (all RED)
- [ ] Verify they all fail

### Day 3 Afternoon
- [ ] Implement Injector queue (put, get_pending)
- [ ] Implement queue overflow (drop oldest)
- [ ] Implement priority ordering
- [ ] Implement format_obs()

### Day 3 Evening
- [ ] Add thread-safety locks
- [ ] Run: `pytest test_injector.py -v` → All GREEN
- [ ] Coverage check: `pytest test_injector.py --cov` → Target 80%+

### Day 4 Morning
- [ ] Write all test_gridworld.py tests (all RED)
- [ ] Verify they all fail
- [ ] Implement movement (N/S/E/W)

### Day 4 Afternoon
- [ ] Implement wall collision detection
- [ ] Implement enemy patrol loop
- [ ] Implement event detection

### Day 4 Evening
- [ ] Implement episode termination
- [ ] Run: `pytest test_gridworld.py -v` → All GREEN
- [ ] Coverage check: `pytest test_gridworld.py --cov` → Target 80%+

### Day 5+
- [ ] Run all tests: `pytest streamagent/tests/ -v` → All GREEN
- [ ] Final coverage: `pytest streamagent/tests/ --cov` → All 80%+
- [ ] Begin Phase 5 (KVStream integration tests)

---

## Coverage Verification

### Phase 1: SinkCache
```bash
pytest streamagent/tests/test_sink_cache.py --cov=streamagent.engine.sink_cache --cov-report=term-missing
# Expected: 85%+ on branches, functions, lines, statements
```

### Phase 2: Router
```bash
pytest streamagent/tests/test_router.py --cov=streamagent.engine.router --cov-report=term-missing
# Expected: 85%+ on branches, functions, lines, statements
```

### Phase 3: Injector
```bash
pytest streamagent/tests/test_injector.py --cov=streamagent.engine.injector --cov-report=term-missing
# Expected: 80%+ on branches, functions, lines, statements
```

### Phase 4: GridWorld
```bash
pytest streamagent/tests/test_gridworld.py --cov=streamagent.env.gridworld --cov-report=term-missing
# Expected: 80%+ on branches, functions, lines, statements
```

### Full Suite
```bash
pytest streamagent/tests/ --cov=streamagent --cov-report=term-missing
# Expected: 82%+ overall on all components
```

---

## Key Success Metrics

- [x] 50+ tests specified (all currently in RED state)
- [x] 85%+ coverage target for critical components (SinkCache, Router)
- [x] 80%+ coverage target for supporting components (Injector, GridWorld)
- [x] All tests independent, can run in any order
- [x] No model loading, all tests instant (<1 second per test)
- [x] Zero implementation code written yet (pure TDD)
- [x] Execution timeline: 4 days to all tests GREEN

---

## File References

**Quick Start**: TDD_QUICK_START.md
**Full Spec**: TDD_IMPLEMENTATION_STRATEGY.md (1859 lines)
**Navigation**: TDD_INDEX.md
**This File**: TEST_STRUCTURE.md

---

**Status**: All test specifications complete. Ready for implementation.

