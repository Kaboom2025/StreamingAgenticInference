# StreamAgent: TDD Implementation Order and Test Strategy

**Version 1.0 — Test-Driven Development Roadmap**

This document specifies the test-first order for building all four core engine components. Each section includes:
- Test class structure
- Key test cases with exact assertions
- Fixtures and mocks required
- When and why to write each test
- Coverage targets

**Target: 80%+ coverage across all components. All implementation follows from failing tests.**

---

## Part 1: Test Implementation Order (High-Level Timeline)

### Phase 1: Foundation (SinkCache + Router FSM)
Tests must pass in isolation, with heavy mocking of the model backend.

```
Week 1:
  test_sink_cache.py
    ├─ Test FIFO eviction mechanics
    ├─ Test pinned token protection
    ├─ Test attention mask rebuild
    └─ Test cache_position monotonicity ← CRITICAL INVARIANT

  test_router.py
    ├─ Test FSM state transitions
    ├─ Test act tag parsing
    ├─ Test malformed tag recovery
    └─ Test obs tag silencing
```

### Phase 2: Integration (Injector)
Builds on Router tests; adds concurrency validation.

```
Week 1.5:
  test_injector.py
    ├─ Test thread-safe put()
    ├─ Test queue overflow behavior
    ├─ Test format_obs() output
    └─ Test priority ordering
```

### Phase 3: Environment (GridWorld)
Fully independent; no engine dependencies except interfaces.

```
Week 2:
  test_gridworld.py
    ├─ Test physics (wall blocking, movement)
    ├─ Test enemy patrol loops
    ├─ Test event detection (enemy_near, goal_reached, death)
    └─ Test episode termination
```

### Phase 4: Integration Tests (Later)
After individual components work. Only sketch here — full KVStream tests require model loading.

```
Week 3+:
  test_kv_stream.py (to be created)
    ├─ Mock model backend + observe full generation loop
    ├─ Test injection timing and cache updates
    └─ Test recovery latency on GridWorld playthrough
```

---

## Part 2: SinkCache Test Specification

**File**: `streamagent/tests/test_sink_cache.py`

### Test Class Structure

```python
class TestSinkCacheFIFOEviction:
    """Tests for standard FIFO eviction mode (default)."""

class TestSinkCacheAttentionMask:
    """Tests for attention mask rebuild after eviction."""

class TestSinkCachePinnedTokens:
    """Tests for goal token protection."""

class TestSinkCacheMonotonicity:
    """CRITICAL: cache_position must never skip or repeat."""

class TestSinkCacheImportanceEviction:
    """Optional: H2O-style importance-based eviction."""
```

### Detailed Test Cases

#### 1. FIFO Eviction Preserves Sink Tokens

```python
def test_fifo_eviction_preserves_first_4_sinks():
    """
    Setup:
      num_sinks=4, window_length=8 → total capacity = 12 tokens
      Create cache, append tokens 0-11 (filling sinks + window)
      Append token 12 (triggers eviction)

    Expected behavior:
      - Tokens 0-3 (sinks) remain pinned forever
      - Token 4 (first non-sink, first window token) is evicted
      - Cache now holds tokens [0-3] (sinks) + [5-12] (window)
      - Evicted count = 1

    Assertions:
      assert cache.evicted_count == 1
      assert cache.effective_seq_len == 12  # 4 sinks + 8 window
      # Verify sink tokens still present in cache.key_states[0]
      assert cache.get_attention_mask(query_pos=12).sum() == 12
```

#### 2. Eviction Sequence (FIFO Order)

```python
def test_fifo_eviction_sequence():
    """
    Append 20 tokens to a cache with capacity 12.
    Verify eviction happens in strict FIFO order.

    Assertions:
      After each append beyond capacity:
        - Oldest non-sink, non-pinned token is evicted
        - No sink tokens ever evicted
        - No pinned tokens ever evicted
        - Eviction count increments by 1
      Cache state after appending token 20:
        - Tokens 0-3: sinks (never evicted)
        - Tokens 13-20: current window (8 newest non-sink)
        - Tokens 4-12: evicted
        - assert cache.evicted_count == 9  # 4-12 inclusive
```

#### 3. Pinned Token Protection (Goal Tokens)

```python
def test_pinned_tokens_never_evict():
    """
    Setup:
      num_sinks=4, window_length=8, capacity=12
      Append tokens 0-19, pinning tokens 2, 6, 11

    Expected behavior:
      When token 12 arrives:
        - Token 2 is pinned → skip it
        - Token 5 (first non-pinned, non-sink) is evicted
      When token 15 arrives:
        - Token 6 is pinned → skip it
        - Token 7 (next non-pinned) is evicted
      And so on...

    Assertions:
      for token_id in [2, 6, 11]:
        assert token_id not in cache.evicted_positions
      # Verify these tokens still in cache.key_states
      assert cache.get_attention_mask(query_pos=19).sum() == 12
      # Only non-pinned, non-sink tokens evicted
      assert cache.evicted_count == len([4, 5, 7, 8, 9, 10, 12, 13, 14])
```

#### 4. Attention Mask Rebuild (Non-Contiguous Positions)

```python
def test_attention_mask_non_contiguous_after_eviction():
    """
    Key insight: After evicting token 5, cache holds tokens [0-4, 6-12].
    The attention mask must reflect these non-contiguous absolute positions.

    Setup:
      Append tokens 0-12, evict token 5.
      Cache now holds positions [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
      Query position = 12

    Expected behavior:
      Mask = get_attention_mask(query_pos=12)
      Mask shape: [12] (one row per token in cache)
      Mask[i] = 1 if position[i] <= query_pos, else 0
      → All 12 tokens have position <= 12 → mask should be all 1s

    Assertions:
      mask = cache.get_attention_mask(query_pos=12)
      assert mask.shape == torch.Size([12])
      assert mask.sum() == 12
      assert (mask == 1).all()

    Verify boundary case:
      query_pos = 5 (would attend to positions <= 5)
      mask = cache.get_attention_mask(query_pos=5)
      Positions in cache: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
      Causal attention: can only attend to positions <= 5
      → tokens at positions [0, 1, 2, 3, 4] attend: yes
      → tokens at positions [6, 7, ...] attend: no
      assert mask[:5].sum() == 5
      assert mask[5:].sum() == 0
```

#### 5. Cache Position Monotonicity (CRITICAL INVARIANT)

```python
def test_cache_position_strictly_monotonic():
    """
    This is the smoking gun test. If cache_position breaks, RoPE collapses.

    Setup:
      Run generation loop for 100 tokens with 3 evictions.
      At each step:
        1. Append token (position increments by 1)
        2. Call evict_if_needed()
        3. Record cache_position value
        4. Build attention mask

    Expected behavior:
      cache_position is an absolute counter: [0, 1, 2, 3, ..., 99]
      It NEVER skips (no jumps), NEVER repeats (no resets)
      Eviction affects internal storage layout, NOT the position counter

    Assertions:
      positions = [cache.cache_position after each update]
      assert len(positions) == 100
      assert positions == list(range(100))
      # Verify with explicit increment check
      for i in range(1, len(positions)):
        assert positions[i] == positions[i-1] + 1

    In context of attention mask:
      After 3 evictions with 8 non-sink tokens in window:
        - Internal storage: 8 tokens stored at indices [0-7]
        - Absolute positions: [92, 93, 94, 95, 96, 97, 98, 99]
        - At query_pos=99, mask must account for actual positions, not indices
        - assert cache.get_attention_mask(99)[i] == 1 for all i (all attend)
```

#### 6. Eviction Preserves KV Tensor Validity

```python
def test_kv_tensors_valid_shape_after_eviction():
    """
    Verify that after eviction, KV tensors are correctly reshaped.

    Setup:
      num_layers=4, num_heads=8, head_dim=64, window_length=10
      Append 15 tokens (5 evictions), query at position 14

    Expected behavior:
      key_states shape: [4 layers, num_heads, effective_seq_len, head_dim]
      value_states shape: [4 layers, num_heads, effective_seq_len, head_dim]
      effective_seq_len should be window_length (10) after evictions

    Assertions:
      key_states, value_states = cache.get_tensors()
      assert key_states.shape == torch.Size([4, 8, 10, 64])
      assert value_states.shape == torch.Size([4, 8, 10, 64])
      # No NaN, no infinite values
      assert not torch.isnan(key_states).any()
      assert not torch.isinf(key_states).any()
```

#### 7. Multiple Eviction Rounds

```python
def test_multiple_eviction_rounds():
    """
    Simulate a 100-token generation with repeated evictions.

    Setup:
      window_length=20, num_sinks=4, capacity=24
      Append 100 tokens (3 full rounds of 32 tokens)

    Expected behavior:
      Round 1: tokens 0-23 in cache, token 24+ triggers evictions
      Round 2: tokens 0-3 (sinks) + tokens 21-39 in cache, tokens 4-20 evicted
      Round 3: tokens 0-3 + tokens 40-59 in cache, tokens 21-39 evicted
      Round 4: tokens 0-3 + tokens 60-79 in cache, tokens 40-59 evicted

    Assertions:
      assert cache.evicted_count == 76  # tokens 4-79
      assert cache.effective_seq_len == 24
      # Verify sinks always present
      mask = cache.get_attention_mask(99)
      assert mask.shape[0] == 24
```

### Fixtures Required

```python
@pytest.fixture
def simple_cache():
    """Basic cache: num_layers=2, num_heads=4, head_dim=64, num_sinks=4, window_length=8"""
    return SinkCache(
        num_layers=2,
        num_heads=4,
        head_dim=64,
        num_sink_tokens=4,
        window_length=8,
        eviction_policy="fifo",
    )

@pytest.fixture
def cache_with_pinned():
    """Cache with pinned tokens marked (e.g., goal tokens)"""
    cache = SinkCache(
        num_layers=2, num_heads=4, head_dim=64,
        num_sink_tokens=4, window_length=8,
        eviction_policy="fifo",
        pinned_token_ids={2, 6, 11},  # Will be set during execution
    )
    return cache

@pytest.fixture
def mock_kv_tensors():
    """Generate random KV tensors of appropriate shape"""
    def _make_tensors(batch_size=1, seq_len=1, num_layers=2, num_heads=4, head_dim=64):
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        return key, value
    return _make_tensors
```

### Mock Strategy

**Do NOT load actual models. Mock all forward passes.**

```python
@pytest.fixture
def mock_model():
    """Mock model that returns dummy logits without weight loading"""
    model = MagicMock()
    model.config.num_hidden_layers = 2
    model.config.num_attention_heads = 4
    model.config.hidden_size = 256
    # Don't load weights; just return shape-correct tensors
    model.forward = MagicMock(return_value=MagicMock(logits=torch.randn(1, 1, 32000)))
    return model
```

### Coverage Checklist

- [ ] FIFO eviction: append, evict, verify oldest non-sink gone
- [ ] Sink preservation: sink tokens never evicted under any circumstance
- [ ] Pinned tokens: goal tokens stay forever
- [ ] Attention mask: correct shape, correct causal values, non-contiguous handling
- [ ] Cache position: strictly monotonic across 100+ steps with evictions
- [ ] KV tensor shapes: always [num_heads, effective_seq_len, head_dim]
- [ ] Edge cases: empty cache, full capacity, single eviction, bulk evictions
- [ ] Quantization tolerance: if using Q4 KV cache, verify F16/Q4 conversion doesn't NaN

---

## Part 3: Router Test Specification

**File**: `streamagent/tests/test_router.py`

### Test Class Structure

```python
class TestRouterFSM:
    """FSM state machine transitions."""

class TestRouterActTagParsing:
    """Parse complete <act> tags to ActTag dataclass."""

class TestRouterMalformedTags:
    """Malformed tag recovery and timeout handling."""

class TestRouterObsTagSilencing:
    """Observation tags are silenced (not output)."""

class TestRouterMultiTokenTags:
    """Tags spanning multiple tokens are handled correctly."""
```

### Detailed Test Cases

#### 1. FSM: PASSTHROUGH → MAYBE_TAG

```python
def test_fsm_passthrough_to_maybe_tag():
    """
    Feed normal text tokens, then see '<'.

    Setup:
      router = Router()
      Feed tokens: "Hello", " ", "world", " ", "<"

    Expected behavior:
      Tokens "Hello", " ", "world", " " → PASSTHROUGH state, yielded as-is
      Token "<" → transition to MAYBE_TAG state, buffered (not yielded)
      Output: "Hello world "

    Assertions:
      router.state == RouterState.MAYBE_TAG
      router.buffer == "<"
      output == "Hello world "
```

#### 2. FSM: MAYBE_TAG → IN_ACT_TAG (confirmed)

```python
def test_fsm_maybe_tag_to_in_act_tag():
    """
    After seeing '<', check next characters for 'act' keyword.

    Setup:
      router in MAYBE_TAG state, buffer="<"
      Feed: "act", " "

    Expected behavior:
      Tokens "a", "c", "t" decoded → buffer becomes "<act"
      Next token " " → still collecting, stay in IN_ACT_TAG
      No tokens yielded yet

    Assertions:
      router.state == RouterState.IN_ACT_TAG
      router.buffer.startswith("<act")
      output == ""
```

#### 3. FSM: MAYBE_TAG → PASSTHROUGH (false alarm)

```python
def test_fsm_maybe_tag_to_passthrough_false_alarm():
    """
    After '<', next token is not 'act' or 'obs' → emit buffered '<' and continue.

    Setup:
      router in MAYBE_TAG state, buffer="<"
      Feed: "b" (not 'a' for act, not 'o' for obs)

    Expected behavior:
      Detect mismatch, transition to PASSTHROUGH
      Emit buffered "<" + new token "b"
      Continue normal passthrough

    Assertions:
      router.state == RouterState.PASSTHROUGH
      router.buffer == ""
      output == "<b"
```

#### 4. FSM: IN_ACT_TAG → PASSTHROUGH (complete tag)

```python
def test_fsm_in_act_tag_complete_dispatch():
    """
    Model generates: <act cmd="move" dir="N"/>
    Router should parse, dispatch, and return to PASSTHROUGH.

    Setup:
      router = Router()
      register_handler("move", mock_handler)
      Feed: "<", "act", " ", 'cmd', "=", '"', "move", '"', ...]

    Expected behavior:
      In IN_ACT_TAG state, accumulating buffer
      See "/>" → buffer complete
      Parse XML: <act cmd="move" dir="N"/>
      Extract: cmd="move", params={"dir": "N"}
      Dispatch: call registered handler with ActTag(cmd="move", params={"dir": "N"})
      Transition to PASSTHROUGH, buffer cleared
      Output: SILENT (act tags are not echoed)

    Assertions:
      mock_handler.called
      mock_handler.call_args[0][0].cmd == "move"
      mock_handler.call_args[0][0].params == {"dir": "N"}
      router.state == RouterState.PASSTHROUGH
      output == ""  # act tags are not output
```

#### 5. Act Tag Parsing: Multiple Parameters

```python
def test_act_tag_parsing_multiple_params():
    """
    Parse <act cmd="take" obj="key" from="shelf"/>

    Setup:
      raw = '<act cmd="take" obj="key" from="shelf"/>'

    Expected behavior:
      _parse_act_tag(raw) extracts:
        cmd = "take"
        params = {"obj": "key", "from": "shelf"}

    Assertions:
      tag = router._parse_act_tag(raw)
      assert tag.cmd == "take"
      assert tag.params == {"obj": "key", "from": "shelf"}
```

#### 6. Act Tag Parsing: Self-closing vs. Container

```python
def test_act_tag_parsing_self_closing():
    """<act cmd="wait"/> → self-closing, no closing tag needed"""
    raw = '<act cmd="wait"/>'
    tag = router._parse_act_tag(raw)
    assert tag.cmd == "wait"
    assert tag.params == {}

def test_act_tag_parsing_with_content():
    """<act cmd="think">reasoning text</act> → extract cmd, ignore content"""
    raw = '<act cmd="think">I should move north</act>'
    tag = router._parse_act_tag(raw)
    assert tag.cmd == "think"
    # Content is discarded for act tags (only used for obs)
```

#### 7. Malformed Tag Timeout

```python
def test_malformed_tag_50_token_timeout():
    """
    Model emits <act but never completes (no />).
    After 50 tokens in IN_ACT_TAG state, timeout and discard.

    Setup:
      router in IN_ACT_TAG state
      Feed 50 tokens of garbage (no "/>" anywhere)

    Expected behavior:
      At token 50, timeout triggers
      Log "malformed_tag" event
      Clear buffer, transition to PASSTHROUGH
      Emit buffered junk as THINK tokens
      Next tokens pass through normally

    Assertions:
      router.malformed_tag_count == 1
      router.state == RouterState.PASSTHROUGH
      router.buffer == ""
      # Tokens 1-49 emitted as junk, token 50+ pass through
```

#### 8. Observation Tag Silencing

```python
def test_obs_tag_silenced():
    """
    Injected obs tags: <obs type="collision">...</obs>
    These are input (from injector), not output. Router silences them.

    Setup:
      router = Router()
      Feed: "<", "obs", " ", "type", ...
           (full <obs type="collision">wall at N</obs>)

    Expected behavior:
      Transition to IN_OBS_TAG state
      Accumulate until "</obs>"
      Discard entire tag (do not emit)
      Transition to PASSTHROUGH
      No handler registered for obs tags

    Assertions:
      router.state == RouterState.PASSTHROUGH
      output == ""  # entire obs tag silenced
      router.buffer == ""
```

#### 9. Multi-Token Tag Detection

```python
def test_multi_token_tag_act_spread_across_tokens():
    """
    Model might tokenize: "<act" + " cmd=" + '"move"' + " dir=" + '"N"' + "/>"
    Router must accumulate across token boundaries and recognize completion.

    Setup:
      tokenizer splits <act cmd="move" dir="N"/> into 6 tokens
      Feed each token one by one

    Expected behavior:
      Token 1: "<act" → buffer = "<act"
      Token 2: " cmd=" → buffer = "<act cmd="
      Token 3: '"move"' → buffer = '<act cmd="move"'
      ... continue ...
      Token 6: "/>" → buffer complete, parse and dispatch

    Assertions:
      After token 6:
        router.state == RouterState.PASSTHROUGH
        handler.called with ActTag(cmd="move", params={"dir": "N"})
```

#### 10. Tag Detection with Whitespace Variations

```python
def test_act_tag_with_extra_whitespace():
    """<act   cmd  =  "move"   dir  =  "N"   />"""
    raw = '<act   cmd  =  "move"   dir  =  "N"   />'
    tag = router._parse_act_tag(raw)
    assert tag.cmd == "move"
    assert tag.params["dir"] == "N"

def test_act_tag_newlines_in_tag():
    """<act\ncmd="move"\ndir="N"\n/>"""
    raw = '<act\ncmd="move"\ndir="N"\n/>'
    tag = router._parse_act_tag(raw)
    assert tag.cmd == "move"
```

#### 11. Edge Case: Incomplete Act Tag at Episode End

```python
def test_flush_incomplete_act_tag():
    """
    Episode ends with buffer = "<act cmd="move""
    Calling flush() should handle gracefully.

    Setup:
      router in IN_ACT_TAG state, buffer incomplete
      Call flush()

    Expected behavior:
      Log "incomplete_tag" event
      Return to PASSTHROUGH
      Optionally emit buffer as regular text or discard

    Assertions:
      router.state == RouterState.PASSTHROUGH
      router.buffer == ""
```

### Fixtures Required

```python
@pytest.fixture
def router():
    """Fresh router instance"""
    return Router()

@pytest.fixture
def router_with_handlers(router):
    """Router with mock handlers registered"""
    move_handler = MagicMock()
    wait_handler = MagicMock()
    router.register_handler("move", move_handler)
    router.register_handler("wait", wait_handler)
    return router, {"move": move_handler, "wait": wait_handler}

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that decodes token IDs to text"""
    tokenizer = MagicMock()
    tokenizer.decode = MagicMock(side_effect=lambda ids: "".join(
        {"<": "<", "a": "a", "c": "c", "t": "t"}.get(str(id), str(id))
    ))
    return tokenizer
```

### Mock Strategy

```python
# No model loading needed; just test FSM and parsing logic
# Mock Token objects:
@pytest.fixture
def mock_tokens():
    """Generate tokens with text content"""
    def _make_token(text):
        token = MagicMock()
        token.text = text
        return token
    return _make_token
```

### Coverage Checklist

- [ ] FSM: all 6 transitions (PASSTHROUGH→MAYBE_TAG, MAYBE_TAG→IN_ACT_TAG, etc.)
- [ ] Parsing: cmd extraction, parameter extraction, XML robustness
- [ ] Malformed tags: 50-token timeout, logging, recovery
- [ ] Obs silencing: tags detected, handlers not called, no output
- [ ] Multi-token tags: accumulation across token boundaries
- [ ] Whitespace handling: extra spaces, newlines, tabs
- [ ] Edge cases: empty buffer, EOF in middle of tag, duplicate handlers
- [ ] Handler dispatch: correct handler called with correct ActTag

---

## Part 4: Injector Test Specification

**File**: `streamagent/tests/test_injector.py`

### Test Class Structure

```python
class TestInjectorThreadSafety:
    """Concurrent put() from multiple threads."""

class TestInjectorQueueManagement:
    """get_pending(), overflow, queue state."""

class TestInjectorObsFormatting:
    """format_obs() generates correct XML."""

class TestInjectorPriority:
    """Higher priority observations come first."""
```

### Detailed Test Cases

#### 1. Basic put() and get_pending()

```python
def test_put_single_obs_get_pending():
    """
    Put one observation, retrieve it.

    Setup:
      injector = ObsInjector(max_queue_size=32)
      obs = Observation(type="collision", payload="wall at N", priority=1)
      injector.put(obs)

    Expected behavior:
      pending = injector.get_pending()
      Returns list of one formatted string
      Queue is flushed (empty after get_pending)

    Assertions:
      assert len(pending) == 1
      assert '<obs type="collision">' in pending[0]
      assert 'wall at N' in pending[0]
      # Verify queue now empty
      assert len(injector.get_pending()) == 0
```

#### 2. Multiple Observations Batched

```python
def test_get_pending_batches_multiple_obs():
    """
    Put 3 observations, get_pending returns all 3.

    Setup:
      injector = ObsInjector()
      obs1 = Observation(type="enemy_near", payload='dist="2" bearing="N"')
      obs2 = Observation(type="goal_near", payload='dist="1"')
      obs3 = Observation(type="tick", payload='step="10"')
      injector.put(obs1)
      injector.put(obs2)
      injector.put(obs3)

    Expected behavior:
      pending = injector.get_pending()
      Returns list of 3 formatted strings in priority order

    Assertions:
      assert len(pending) == 3
      # Verify order (assuming same priority, FIFO)
      assert 'type="enemy_near"' in pending[0]
      assert 'type="goal_near"' in pending[1]
      assert 'type="tick"' in pending[2]
      # Queue flushed
      assert injector.get_pending() == []
```

#### 3. format_obs() Correctness

```python
def test_format_obs_collision():
    """<obs type="collision">wall at N</obs>"""
    obs = Observation(type="collision", payload="wall at N")
    formatted = injector.format_obs(obs)
    assert formatted == '<obs type="collision">wall at N</obs>'

def test_format_obs_enemy_near_with_attributes():
    """<obs type="enemy_near" dist="2" bearing="NW"/>"""
    obs = Observation(type="enemy_near", payload='dist="2" bearing="NW"')
    formatted = injector.format_obs(obs)
    expected = '<obs type="enemy_near" dist="2" bearing="NW"/>'
    assert formatted == expected

def test_format_obs_full_grid():
    """Full grid payload (ASCII art)"""
    grid_ascii = "W W W\nW A G\nW W W"
    obs = Observation(type="full_grid", payload=grid_ascii)
    formatted = injector.format_obs(obs)
    assert '<obs type="full_grid">' in formatted
    assert 'W A G' in formatted
    assert '</obs>' in formatted
```

#### 4. Queue Overflow: Drop Oldest

```python
def test_overflow_drops_oldest():
    """
    max_queue_size=3
    Put 5 observations → oldest 2 are dropped

    Setup:
      injector = ObsInjector(max_queue_size=3)
      obs1, obs2, obs3, obs4, obs5 = [Observation(...) for _ in range(5)]
      for obs in [obs1, obs2, obs3, obs4, obs5]:
        injector.put(obs)

    Expected behavior:
      Queue now contains [obs3, obs4, obs5]
      obs1 and obs2 are dropped
      get_pending() returns 3 formatted strings

    Assertions:
      pending = injector.get_pending()
      assert len(pending) == 3
      # obs3 comes first
      assert obs3.type in pending[0]
```

#### 5. Priority Ordering

```python
def test_priority_ordering_higher_first():
    """
    Higher priority observations come first in get_pending().

    Setup:
      obs_low = Observation(type="tick", payload='step="10"', priority=0)
      obs_med = Observation(type="enemy_near", payload='dist="3"', priority=2)
      obs_high = Observation(type="death", payload='caught', priority=10)
      injector.put(obs_low)
      injector.put(obs_high)
      injector.put(obs_med)

    Expected behavior:
      get_pending() returns [high, med, low]
      Sorted by priority descending

    Assertions:
      pending = injector.get_pending()
      assert 'death' in pending[0]
      assert 'enemy_near' in pending[1]
      assert 'tick' in pending[2]
```

#### 6. Thread Safety: Concurrent Puts

```python
def test_thread_safe_concurrent_puts():
    """
    Multiple threads calling put() simultaneously.

    Setup:
      injector = ObsInjector(max_queue_size=32)
      Create 4 threads, each puts 10 observations
      Total: 40 observations, but only 32 in queue (oldest 8 dropped)

    Expected behavior:
      No crashes, no race conditions
      Final queue size ≤ 32
      No data corruption
      get_pending() returns all buffered observations

    Assertions:
      import threading
      def thread_puts(thread_id):
        for i in range(10):
          obs = Observation(type=f"event_{thread_id}_{i}", payload="")
          injector.put(obs)

      threads = [threading.Thread(target=thread_puts, args=(i,)) for i in range(4)]
      for t in threads: t.start()
      for t in threads: t.join()

      pending = injector.get_pending()
      assert len(pending) <= 32
      # Oldest 8 dropped (40 - 32)
      assert len(pending) == 32
```

#### 7. Empty Queue

```python
def test_get_pending_empty_queue():
    """get_pending() on empty queue returns empty list."""
    injector = ObsInjector()
    pending = injector.get_pending()
    assert pending == []
```

#### 8. Queue Persistence Until Flush

```python
def test_queue_persists_until_get_pending():
    """
    Observations stay in queue until get_pending() called.

    Setup:
      injector.put(obs1)
      injector.put(obs2)

    Expected behavior:
      # Queue still has 2 observations (not yet flushed)
      get_pending() returns 2, flushes
      Next get_pending() returns [] (empty)

    Assertions:
      # Before flush: can check internal queue size
      assert injector._queue_size() == 2  # if exposed
      pending1 = injector.get_pending()
      assert len(pending1) == 2
      pending2 = injector.get_pending()
      assert len(pending2) == 0
```

#### 9. Duplicate Type Handling (Optional)

```python
def test_dedup_same_type_within_batch():
    """
    If multiple enemy_near events in same batch, deduplicate?
    (Spec says injector deduplicates within one tick)

    Setup:
      obs1 = Observation(type="enemy_near", payload='dist="2" bearing="N"')
      obs2 = Observation(type="enemy_near", payload='dist="3" bearing="E"')
      injector.put(obs1)
      injector.put(obs2)

    Expected behavior (if dedup enabled):
      get_pending() returns 1 (keep highest priority dist, or latest?)

    Or no dedup:
      get_pending() returns 2 (keep both)

    Assertions:
      pending = injector.get_pending()
      # Verify behavior matches spec
```

### Fixtures Required

```python
@pytest.fixture
def injector():
    """Fresh injector with default queue size"""
    return ObsInjector(max_queue_size=32)

@pytest.fixture
def injector_small():
    """Injector with small queue for overflow testing"""
    return ObsInjector(max_queue_size=3)

@pytest.fixture
def sample_observations():
    """Reusable observation objects"""
    return {
        "collision": Observation(type="collision", payload="wall at N"),
        "enemy_near": Observation(type="enemy_near", payload='dist="2" bearing="N"', priority=2),
        "goal_reached": Observation(type="goal_reached", payload="task complete", priority=10),
        "tick": Observation(type="tick", payload='step="10"', priority=0),
    }
```

### Mock Strategy

```python
# No external dependencies; Observation is a simple dataclass
# No mocking needed beyond normal test fixtures
```

### Coverage Checklist

- [ ] put() and get_pending(): basic queue operations
- [ ] Batching: multiple obs returned in one get_pending()
- [ ] format_obs(): correct XML for all obs types (collision, enemy_near, goal_reached, tick, full_grid)
- [ ] Overflow: max_queue_size enforced, oldest dropped
- [ ] Priority: higher priority first, stable sort for equal priority
- [ ] Thread safety: concurrent puts don't corrupt state
- [ ] Empty queue: no errors on empty
- [ ] Queue flushing: get_pending() empties the queue
- [ ] Payload encoding: special characters (quotes, newlines) handled

---

## Part 5: GridWorld Test Specification

**File**: `streamagent/tests/test_gridworld.py`

### Test Class Structure

```python
class TestGridWorldPhysics:
    """Movement, walls, boundary conditions."""

class TestGridWorldEnemyPatrol:
    """Enemy movement along patrol paths."""

class TestGridWorldEventDetection:
    """Collision, enemy_near, goal_reached, death, tick."""

class TestGridWorldScenarios:
    """Named scenarios build correctly."""

class TestGridWorldEpisodeTermination:
    """max_steps, goal reached, death conditions."""
```

### Detailed Test Cases

#### 1. Basic Movement: Valid Moves

```python
def test_agent_moves_north():
    """
    Agent at (5, 5), execute move N → (5, 4)

    Setup:
      gridworld.reset("static_maze")
      agent_pos = (5, 5)
      action = Action(cmd="move", params={"dir": "N"})

    Expected behavior:
      done, info = gridworld.step(action)
      Agent now at (5, 4) (column same, row -1)
      No events (unless other triggers)
      done = False (no goal, no death)

    Assertions:
      assert gridworld.state.agent_pos == (5, 4)
      assert done == False
      assert "events" in info
      assert len(info["events"]) == 0  # no collision
```

#### 2. Movement: Cardinal Directions

```python
def test_movement_all_cardinal_directions():
    """N/S/E/W all work correctly"""
    gridworld.reset("static_maze")
    gridworld.state.agent_pos = (5, 5)

    # North
    gridworld.step(Action(cmd="move", params={"dir": "N"}))
    assert gridworld.state.agent_pos == (5, 4)

    # South
    gridworld.step(Action(cmd="move", params={"dir": "S"}))
    assert gridworld.state.agent_pos == (5, 5)

    # East
    gridworld.step(Action(cmd="move", params={"dir": "E"}))
    assert gridworld.state.agent_pos == (6, 5)

    # West
    gridworld.step(Action(cmd="move", params={"dir": "W"}))
    assert gridworld.state.agent_pos == (5, 5)
```

#### 3. Wall Blocking

```python
def test_agent_blocked_by_wall():
    """
    Agent at (5, 5), wall at (5, 4) (north), move N → position unchanged, collision event

    Setup:
      gridworld.reset("static_maze")
      walls = FrozenSet({(5, 4), ...})
      agent_pos = (5, 5)
      action = Action(cmd="move", params={"dir": "N"})

    Expected behavior:
      done, info = gridworld.step(action)
      Agent stays at (5, 5) (move rejected)
      collision event injected: "wall at N"
      done = False

    Assertions:
      assert gridworld.state.agent_pos == (5, 5)  # unchanged
      assert len(info["events"]) >= 1
      collision_obs = [e for e in info["events"] if e.type == "collision"]
      assert len(collision_obs) == 1
      assert "wall at N" in collision_obs[0].payload
```

#### 4. Boundary Conditions (Grid Edge)

```python
def test_agent_blocked_at_grid_boundary():
    """
    Agent at (0, 0) (top-left), move W or N → rejected

    Setup:
      gridworld = GridWorld(width=10, height=10)
      gridworld.reset()
      gridworld.state.agent_pos = (0, 0)

    Expected behavior:
      step(Action(cmd="move", params={"dir": "W"}))
      Agent stays at (0, 0)
      Collision event (boundary treated like wall)

    Assertions:
      assert gridworld.state.agent_pos == (0, 0)
      events = info["events"]
      collision = [e for e in events if e.type == "collision"]
      assert len(collision) >= 1
```

#### 5. Enemy Patrol: Deterministic Loop

```python
def test_enemy_patrol_deterministic_loop():
    """
    Enemy has patrol_path = [(3,3), (4,3), (5,3), (4,3)].
    Each step, enemy moves to next position in loop.

    Setup:
      gridworld.reset("single_patrol")
      enemy = gridworld.state.enemies[0]
      initial_pos = enemy.pos
      initial_path_index = enemy.path_index

    Expected behavior:
      For 4 steps, enemy visits each patrol position in order, then repeats

    Assertions:
      path = enemy.patrol_path
      for step_num in range(8):
        done, info = gridworld.step(Action(cmd="wait", params={}))
        expected_idx = (initial_path_index + step_num + 1) % len(path)
        expected_pos = path[expected_idx]
        assert enemy.pos == expected_pos
```

#### 6. Enemy Speed Multiplier

```python
def test_enemy_speed_2_advances_2_steps_per_tick():
    """
    Enemy with speed=2 advances 2 steps per game tick.

    Setup:
      enemy = EnemyState(
          pos=(3, 3),
          patrol_path=[(3,3), (4,3), (5,3), (6,3)],
          path_index=0,
          speed=2
      )

    Expected behavior:
      At step 1: path_index = (0 + 2) % 4 = 2, pos = (5, 3)
      At step 2: path_index = (2 + 2) % 4 = 0, pos = (3, 3)

    Assertions:
      assert enemy.path_index == 2
      assert enemy.pos == (5, 3)
```

#### 7. Enemy Near Detection (radius ≤ 2)

```python
def test_enemy_near_event_triggers_at_distance_2():
    """
    Manhattan distance from agent to enemy ≤ 2 → enemy_near event

    Setup:
      agent_pos = (5, 5)
      enemy_pos = (5, 3)  # distance = |5-5| + |5-3| = 2
      gridworld.reset("single_patrol")
      gridworld.state.agent_pos = (5, 5)
      gridworld.state.enemies[0].pos = (5, 3)

    Expected behavior:
      step(Action(...))
      enemy_near event fires with dist="2"

    Assertions:
      done, info = gridworld.step(Action(cmd="wait", params={}))
      enemy_near_obs = [e for e in info["events"] if e.type == "enemy_near"]
      assert len(enemy_near_obs) == 1
      assert 'dist="2"' in enemy_near_obs[0].payload
```

#### 8. Enemy Near: Bearing Calculation

```python
def test_enemy_near_bearing_direction():
    """
    Enemy at different bearing (N/S/E/W/NE/etc.) → bearing field set correctly

    Setup:
      agent_pos = (5, 5)
      enemy_pos = (5, 3)  # north
      gridworld.reset()
      gridworld.state.agent_pos = (5, 5)
      gridworld.state.enemies[0].pos = (5, 3)

    Expected behavior:
      done, info = gridworld.step(Action(cmd="wait", params={}))
      enemy_near event includes bearing="N"

    Assertions:
      enemy_near_obs = [e for e in info["events"] if e.type == "enemy_near"]
      assert 'bearing="N"' in enemy_near_obs[0].payload

    Also test:
      enemy at (6, 5) → bearing="E"
      enemy at (6, 4) → bearing="NE"
      enemy at (4, 6) → bearing="SW"
```

#### 9. Enemy Not Near (distance > 2)

```python
def test_no_enemy_near_event_at_distance_3():
    """
    Manhattan distance > 2 → no enemy_near event

    Setup:
      agent_pos = (5, 5)
      enemy_pos = (5, 2)  # distance = 3
      gridworld.reset()

    Expected behavior:
      step(Action(...))
      No enemy_near event

    Assertions:
      done, info = gridworld.step(Action(cmd="wait", params={}))
      enemy_near_obs = [e for e in info["events"] if e.type == "enemy_near"]
      assert len(enemy_near_obs) == 0
```

#### 10. Death: Agent Collides with Enemy

```python
def test_death_event_agent_on_enemy():
    """
    agent_pos == enemy_pos → death event, done=True, episode terminates

    Setup:
      agent_pos = (5, 5)
      enemy_pos = (5, 5)
      gridworld.reset("single_patrol")
      gridworld.state.agent_pos = (5, 5)
      gridworld.state.enemies[0].pos = (5, 5)

    Expected behavior:
      done, info = gridworld.step(Action(cmd="wait", params={}))
      death event fires with payload "agent caught by enemy"
      done = True
      episode terminates

    Assertions:
      assert done == True
      death_obs = [e for e in info["events"] if e.type == "death"]
      assert len(death_obs) == 1
      assert death_obs[0].priority == 10  # highest priority
```

#### 11. Goal Near Detection

```python
def test_goal_near_distance_2():
    """
    Manhattan distance to goal ≤ 2 → goal_near event

    Setup:
      agent_pos = (5, 5)
      goal_pos = (6, 5)  # distance = 1
      gridworld.reset()

    Expected behavior:
      step(Action(...))
      goal_near event fires with dist="1"

    Assertions:
      goal_near_obs = [e for e in info["events"] if e.type == "goal_near"]
      assert len(goal_near_obs) == 1
      assert 'dist="1"' in goal_near_obs[0].payload
```

#### 12. Goal Reached (Agent on Goal)

```python
def test_goal_reached_event():
    """
    agent_pos == goal_pos → goal_reached event, done=True

    Setup:
      agent_pos = (5, 5)
      goal_pos = (5, 5)
      gridworld.reset("static_maze")
      gridworld.state.agent_pos = (5, 5)
      gridworld.state.goal_pos = (5, 5)

    Expected behavior:
      step(Action(...))
      goal_reached event fires
      done = True

    Assertions:
      assert done == True
      goal_obs = [e for e in info["events"] if e.type == "goal_reached"]
      assert len(goal_obs) == 1
      assert goal_obs[0].priority == 10
```

#### 13. Tick Heartbeat (Every 10 Steps)

```python
def test_tick_heartbeat_every_10_steps():
    """
    Every 10 steps (step_count % 10 == 0), inject tick event

    Setup:
      gridworld.reset("static_maze")

    Expected behavior:
      step 0: tick event (step=0)
      step 1-9: no tick
      step 10: tick event (step=10)
      step 11-19: no tick
      step 20: tick event (step=20)

    Assertions:
      for step in range(30):
        done, info = gridworld.step(Action(cmd="wait", params={}))
        tick_obs = [e for e in info["events"] if e.type == "tick"]
        if step % 10 == 0:
          assert len(tick_obs) == 1
          assert f'step="{step}"' in tick_obs[0].payload
        else:
          assert len(tick_obs) == 0
```

#### 14. Max Steps Episode Termination

```python
def test_episode_terminates_at_max_steps():
    """
    After step_count reaches max_steps, done=True

    Setup:
      gridworld.reset("static_maze")
      gridworld.state.max_steps = 50

    Expected behavior:
      After 50 steps, done=True

    Assertions:
      for step in range(50):
        done, info = gridworld.step(Action(cmd="wait", params={}))
        if step < 49:
          assert done == False
        else:
          assert done == True
```

#### 15. Event Priority Ordering

```python
def test_event_priority_ordering_in_injection():
    """
    Multiple events in same step are injected in priority order.

    Setup:
      agent_pos = goal_pos = enemy_pos = (5, 5)
      (simultaneous: goal_reached, death, both high priority)

    Expected behavior:
      get_pending() from injector returns events sorted by priority

    Assertions:
      # This is more integration-level, but verify that
      # injector.put() is called for each event in priority order
      # via injector.put(obs) and later get_pending() respects priority
```

#### 16. Reset Returns Initial Observation

```python
def test_reset_returns_full_grid_obs():
    """
    reset(scenario) returns initial full_grid observation

    Setup:
      gridworld.reset("static_maze")

    Expected behavior:
      initial_obs = gridworld._full_grid_obs()
      Type is Observation(type="full_grid")
      Payload is ASCII grid string

    Assertions:
      assert initial_obs.type == "full_grid"
      assert "W" in initial_obs.payload  # walls
      assert "A" in initial_obs.payload  # agent
      assert "G" in initial_obs.payload  # goal
```

#### 17. Render Output

```python
def test_render_ascii_grid():
    """
    render() produces correct ASCII representation

    Setup:
      gridworld.reset("static_maze")

    Expected behavior:
      render() returns string with grid visualized:
        W = wall
        A = agent
        G = goal
        E = enemy
        . = empty

    Assertions:
      output = gridworld.render()
      assert "W" in output
      assert "A" in output
      assert "G" in output
```

#### 18. Scenario Builders

```python
def test_scenario_static_maze_builds():
    """scenarios.build("static_maze") returns valid GridState"""
    state = scenarios.build("static_maze")
    assert state.width == 10
    assert state.height == 10
    assert len(state.walls) > 0
    assert len(state.enemies) == 0  # no enemies in static maze

def test_scenario_single_patrol_builds():
    """scenarios.build("single_patrol") has one enemy"""
    state = scenarios.build("single_patrol")
    assert len(state.enemies) == 1
    assert len(state.enemies[0].patrol_path) > 0

def test_scenario_dual_patrol_builds():
    """scenarios.build("dual_patrol") has two enemies"""
    state = scenarios.build("dual_patrol")
    assert len(state.enemies) == 2
```

### Fixtures Required

```python
@pytest.fixture
def gridworld():
    """Fresh GridWorld instance with default dimensions"""
    return GridWorld(width=10, height=10)

@pytest.fixture
def gridworld_reset(gridworld):
    """GridWorld after reset to static_maze"""
    gridworld.reset("static_maze")
    return gridworld

@pytest.fixture
def mock_injector():
    """Mock injector to capture put() calls"""
    injector = MagicMock(spec=ObsInjector)
    injector.put = MagicMock()
    return injector
```

### Mock Strategy

```python
# GridWorld depends only on interfaces.py (Environment ABC, Action, Observation)
# No model loading; no external dependencies
# Injector can be mocked to verify observations are queued correctly

@pytest.fixture
def gridworld_with_mock_injector(gridworld, mock_injector):
    """GridWorld with mock injector to verify put() calls"""
    gridworld.register_injector(mock_injector)
    return gridworld, mock_injector
```

### Coverage Checklist

- [ ] Movement: N/S/E/W all cardinal directions
- [ ] Wall blocking: position unchanged, collision event fired
- [ ] Boundary: edge of grid treated as wall
- [ ] Enemy patrol: deterministic loop, speed multiplier
- [ ] Enemy near: distance ≤ 2, bearing calculation (8 directions)
- [ ] Enemy far: distance > 2, no event
- [ ] Death: agent on enemy cell, done=True
- [ ] Goal near: distance ≤ 2
- [ ] Goal reached: agent on goal, done=True
- [ ] Tick heartbeat: every 10 steps
- [ ] Max steps: episode terminates at limit
- [ ] Event priority: higher priority events first in injector queue
- [ ] Reset: returns full_grid observation
- [ ] Render: ASCII output correct
- [ ] Scenarios: all named scenarios build correctly
- [ ] Edge cases: agent at (0, 0), goal at edge, multiple events same step

---

## Part 6: TDD Execution Order and Milestones

### Milestone 1: SinkCache (Days 1-2)

**Why first**: The most fundamental component. Errors here cascade into all downstream tests.

**Execution**:
1. Write test class skeleton (all tests defined, all fail)
2. Implement minimal SinkCache to pass first test (FIFO eviction)
3. Implement pinned token logic
4. Implement attention mask rebuild
5. **CRITICAL**: Verify cache_position monotonicity invariant passes
6. Run full suite; target 80%+ coverage

**Pass criteria**:
```bash
pytest streamagent/tests/test_sink_cache.py -v --cov=streamagent.engine.sink_cache
# Expected: 12+ tests, all passing, 80%+ coverage
```

### Milestone 2: Router (Days 2-3)

**Why second**: Independent from SinkCache. FSM logic is deterministic, no model needed.

**Execution**:
1. Write test class skeleton
2. Implement FSM transitions (PASSTHROUGH → MAYBE_TAG → IN_ACT_TAG)
3. Implement ActTag parsing with XML.etree
4. Implement obs tag silencing
5. Implement 50-token malformed tag timeout
6. Run full suite; target 80%+ coverage

**Pass criteria**:
```bash
pytest streamagent/tests/test_router.py -v --cov=streamagent.engine.router
# Expected: 11+ tests, all passing, 80%+ coverage
```

### Milestone 3: Injector (Days 3-3.5)

**Why third**: Depends on dataclasses (Action, Observation), not on Router or SinkCache.

**Execution**:
1. Write test class skeleton
2. Implement put() and get_pending()
3. Implement queue overflow (drop oldest)
4. Implement priority ordering
5. Add thread-safety locks
6. Run full suite; target 80%+ coverage

**Pass criteria**:
```bash
pytest streamagent/tests/test_injector.py -v --cov=streamagent.engine.injector
# Expected: 9+ tests, all passing, 80%+ coverage
```

### Milestone 4: GridWorld (Days 3.5-4)

**Why fourth**: Fully independent. Tests environment physics, not engine.

**Execution**:
1. Write test class skeleton
2. Implement movement (N/S/E/W)
3. Implement wall collision
4. Implement enemy patrol loop
5. Implement event detection (enemy_near, goal_reached, death, tick)
6. Implement episode termination
7. Run full suite; target 80%+ coverage

**Pass criteria**:
```bash
pytest streamagent/tests/test_gridworld.py -v --cov=streamagent.env.gridworld
# Expected: 18+ tests, all passing, 80%+ coverage
```

### Milestone 5: Integration Tests (Day 5+)

**Why last**: Requires all four components working together.

**Execution** (sketch only — full implementation after unit tests pass):
1. Create test_kv_stream.py (not in scope of this spec)
2. Mock model backend
3. Test generation loop with injections
4. Test recovery latency metric
5. Test GridWorld + KVStream integration

---

## Part 7: Mock Strategy (Global)

### Model Backend Mocking (Used Everywhere)

```python
@pytest.fixture
def mock_llm_backend():
    """
    Mock the backend (llama-cpp-python, HF transformers, mlx-lm).
    Do NOT load actual model weights.
    """
    backend = MagicMock()

    # Mock forward pass
    output = MagicMock()
    output.logits = torch.randn(1, 1, 32000)  # vocab size
    backend.forward = MagicMock(return_value=output)

    # Mock cache update
    backend.update_cache = MagicMock(return_value=(
        torch.randn(1, 8, 10, 64),  # key_states
        torch.randn(1, 8, 10, 64),  # value_states
    ))

    return backend
```

### Avoid Heavy Dependencies in Unit Tests

**DO NOT**:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")  # ❌ SLOW, HEAVY
```

**DO**:
```python
model = MagicMock()
model.forward = MagicMock(return_value=MagicMock(logits=torch.randn(...)))  # ✅ INSTANT
```

### Tokenizer Mocking (for Router tests)

```python
@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for token-to-text conversion"""
    tokenizer = MagicMock()

    # Token text mapping (simplified)
    token_map = {
        1: "<",
        2: "a",
        3: "c",
        4: "t",
        # ... etc
    }

    tokenizer.decode = MagicMock(side_effect=lambda ids: "".join(
        token_map.get(id, str(id)) for id in ids
    ))

    tokenizer.encode = MagicMock(side_effect=lambda text: [1, 2, 3, 4])  # simplified

    return tokenizer
```

### Concurrency Testing (Injector)

```python
import threading
import concurrent.futures

def test_injector_thread_safety_concurrent_puts(injector):
    """Multiple threads calling put() simultaneously"""

    def thread_worker(thread_id, num_puts):
        for i in range(num_puts):
            obs = Observation(
                type=f"event_{thread_id}_{i}",
                payload=f"payload {thread_id} {i}"
            )
            injector.put(obs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(thread_worker, tid, 10)
            for tid in range(4)
        ]
        for f in futures:
            f.result()  # wait for all to complete

    pending = injector.get_pending()
    assert len(pending) <= 32  # queue size limit
```

---

## Part 8: Running Tests Locally

### Install Dependencies

```bash
cd /Users/saalik/Documents/Projects/StreamingAgenticInference

# Create venv
python -m venv venv
source venv/bin/activate

# Install test deps
pip install pytest pytest-cov pytest-asyncio pytest-mock numpy torch pyyaml
```

### Run All Tests

```bash
pytest streamagent/tests/ -v --cov=streamagent --cov-report=html
```

### Run Single Test File

```bash
pytest streamagent/tests/test_sink_cache.py -v --cov=streamagent.engine.sink_cache
```

### Run Single Test

```bash
pytest streamagent/tests/test_sink_cache.py::TestSinkCacheFIFOEviction::test_fifo_eviction_preserves_first_4_sinks -v
```

### Coverage Report

```bash
pytest streamagent/tests/ --cov=streamagent --cov-report=term-missing
# Must show: 80%+ branches, functions, lines, statements
```

---

## Part 9: Coverage Targets

**MANDATORY: 80%+ for all components**

### SinkCache: Target 85%+
- Eviction logic (FIFO, importance)
- Attention mask rebuild
- Pinned token protection
- Cache position tracking

### Router: Target 85%+
- FSM state transitions
- Tag parsing and dispatch
- Malformed tag recovery
- Multi-token tags

### Injector: Target 80%+
- Queue management
- Priority ordering
- Thread safety
- Observation formatting

### GridWorld: Target 80%+
- Movement and collision
- Enemy patrol
- Event detection
- Episode termination

---

## Part 10: Debugging Common Test Failures

### SinkCache: cache_position Not Monotonic

**Problem**: After eviction, cache_position skips or repeats
**Root cause**: Internal storage index drift
**Debug**:
```python
print(f"Position before: {cache.cache_position}")
cache.evict_if_needed()
print(f"Position after: {cache.cache_position}")
print(f"Internal indices: {cache._internal_indices}")
```
**Fix**: Ensure position counter is independent from storage index

### Router: FSM Stuck in IN_ACT_TAG

**Problem**: Timeout not triggering, router never returns to PASSTHROUGH
**Root cause**: Token counter not incremented, or timeout condition wrong
**Debug**:
```python
print(f"Tokens in ACT_TAG state: {router.tokens_in_buffer}")
print(f"State: {router.state}")
```
**Fix**: Increment token counter every feed(), check timeout >= 50

### Injector: Thread Safety Race

**Problem**: Queue size exceeds max or observations lost
**Root cause**: No lock on put(), get_pending()
**Debug**:
```python
import threading
lock = threading.Lock()  # add to Injector.__init__
with lock:
    self.queue.append(obs)
```
**Fix**: Add lock around queue operations

### GridWorld: Event Not Fired

**Problem**: Enemy near detection fails
**Root cause**: Manhattan distance calculation wrong
**Debug**:
```python
dist = abs(agent[0] - enemy[0]) + abs(agent[1] - enemy[1])
print(f"Agent: {agent}, Enemy: {enemy}, Distance: {dist}")
```
**Fix**: Verify Manhattan distance formula: |dx| + |dy|

---

## Summary: TDD Implementation Checklist

### Phase 1: Foundation (Days 1-2)
- [ ] test_sink_cache.py: 12+ tests, 85%+ coverage
  - [ ] FIFO eviction preserves sinks
  - [ ] Pinned tokens protected
  - [ ] Attention mask rebuild
  - [ ] **cache_position monotonic** (CRITICAL)

### Phase 2: FSM (Days 2-3)
- [ ] test_router.py: 11+ tests, 85%+ coverage
  - [ ] FSM transitions
  - [ ] Act tag parsing
  - [ ] Malformed tag timeout (50 tokens)
  - [ ] Obs tag silencing

### Phase 3: Queue (Days 3-3.5)
- [ ] test_injector.py: 9+ tests, 80%+ coverage
  - [ ] Thread-safe put()
  - [ ] get_pending() batching
  - [ ] Queue overflow (drop oldest)
  - [ ] Priority ordering

### Phase 4: Environment (Days 3.5-4)
- [ ] test_gridworld.py: 18+ tests, 80%+ coverage
  - [ ] Movement (N/S/E/W)
  - [ ] Wall collision
  - [ ] Enemy patrol loop
  - [ ] Event detection (enemy_near, goal_reached, death, tick)
  - [ ] Episode termination

### Phase 5: Integration (Day 5+, sketch only)
- [ ] test_kv_stream.py (to be created)
  - [ ] Mock model backend
  - [ ] Generation loop with injections
  - [ ] Recovery latency measurement

---

**End of TDD Specification**

This document is complete and implementation-ready. All tests are independent, can run in parallel, and require zero model loading.

