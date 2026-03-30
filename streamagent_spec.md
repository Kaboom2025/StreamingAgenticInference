# StreamAgent: Technical Specification
**Version 0.1 — Working Draft**

---

## Overview

StreamAgent is two independent systems with a clean interface between them:

1. **The Inference Engine** (`streamagent/engine/`) — a framework-agnostic continuous LLM runtime that maintains a never-terminated token stream with a bounded KV cache, a mid-stream observation injection API, and a token routing layer that dispatches structured tags to registered handlers.

2. **The Gridworld Environment** (`streamagent/env/`) — a deterministic, headless 2D game environment that conforms to the Engine's `Environment` interface, used as the primary test bed.

The engine has zero knowledge of the game. The game has zero knowledge of the model. They communicate only through the `Environment` interface and the token vocabulary defined in the system prompt.

```
┌─────────────────────────────────────────────────────────┐
│                   INFERENCE ENGINE                       │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │ KVStream │───►│  Router   │───►│ ObsInjector      │  │
│  │ (model)  │    │ (tag FSM) │    │ (async queue)    │  │
│  └──────────┘    └─────┬─────┘    └──────────────────┘  │
│                        │ <act> tag                       │
└────────────────────────┼────────────────────────────────┘
                         │ Environment interface
┌────────────────────────┼────────────────────────────────┐
│                   GRIDWORLD ENV                          │
│                        │                                 │
│  ┌──────────────────────▼──────────────────────────────┐ │
│  │  step(action) → obs  │  GridWorld (10×10)           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
streamagent/
├── engine/
│   ├── __init__.py
│   ├── kv_stream.py          # Core: persistent KV cache + generation loop
│   ├── sink_cache.py         # StreamingLLM attention-sink eviction policy
│   ├── router.py             # Token router: FSM tag detection + dispatch
│   ├── injector.py           # Async observation injection queue
│   ├── interfaces.py         # Environment ABC + Action/Obs dataclasses
│   └── backends/
│       ├── hf_backend.py     # HuggingFace transformers backend
│       ├── llama_backend.py  # llama-cpp-python backend
│       └── mlx_backend.py    # mlx-lm backend (Mac only)
├── env/
│   ├── __init__.py
│   ├── gridworld.py          # GridWorld: state, physics, event detection
│   ├── alfworld_env.py       # ALFWorld wrapper implementing Environment ABC
│   ├── renderer.py           # ASCII + optional rich terminal renderer
│   └── scenarios.py          # Named test scenarios (static maze, patrol, etc.)
├── eval/
│   ├── harness.py            # Runs N episodes, collects metrics
│   ├── metrics.py            # Recovery time, completion rate, step efficiency
│   └── baseline_chunker.py   # Action-chunking baseline for comparison
├── configs/
│   ├── qwen3_8b_mac.yaml              # primary — llama-cpp-python + Metal
│   ├── qwen3_8b_colab_t4.yaml         # eval only — HF transformers 4-bit
│   └── qwen3_8b_colab_a100.yaml       # eval only — HF transformers FP16
├── scripts/
│   ├── run_agent.py          # CLI entrypoint
│   └── run_eval.py           # Evaluation suite entrypoint
└── tests/
    ├── test_router.py
    ├── test_sink_cache.py
    ├── test_injector.py
    └── test_gridworld.py
```

---

## Part 1: The Inference Engine

### 1.1 Token Vocabulary

The system uses a fixed set of structured tags added to the model's vocabulary at the **prompt level only** — no tokenizer modification, no fine-tuning. The model is instructed to use these tags in the system prompt.

```
Tag class        Format                               Emitted by
─────────────    ─────────────────────────────────    ──────────
Think            <think>...</think>                   Model
Act              <act cmd="CMD" [params]/> or         Model
                 <act cmd="CMD" [params]>
Observation      <obs type="TYPE">PAYLOAD</obs>       Engine (injected)
Goal             <goal>DESCRIPTION</goal>             System prompt (pinned)
Memory           <mem>COMPRESSED_HISTORY</mem>        Engine (injected on compression)
```

**Act commands** (all single-token params for fast generation):
```
<act cmd="move" dir="N"/>          # cardinal: N S E W
<act cmd="wait"/>                  # hold current position one tick
<act cmd="look"/>                  # request fresh full-grid observation
<act cmd="done"/>                  # agent signals task complete
```

**Observation types**:
```
<obs type="collision">wall at N</obs>
<obs type="enemy_near" dist="3" bearing="NW"/>
<obs type="goal_near" dist="2" bearing="E"/>
<obs type="tick">step=47</obs>                 # periodic heartbeat
<obs type="full_grid">...ASCII...</obs>        # response to <act cmd="look"/>
```

### 1.2 KVStream — Core Generation Loop

`KVStream` is the central object. It owns the model, the cache, and the generation loop. It is started once and never stopped during an episode.

```python
# engine/kv_stream.py

@dataclass
class KVStreamConfig:
    model_id: str                        # HF model ID or local path
    backend: Literal["hf", "llama", "mlx"]
    sink_tokens: int = 4                 # attention sink size (StreamingLLM)
    window_length: int = 2048            # total KV cache slots
    temperature: float = 0.6
    max_think_tokens_per_step: int = 64  # max reasoning tokens before forcing act check
    device: str = "auto"
    quantization: Optional[str] = None  # "4bit", "8bit", None


class KVStream:
    """
    A never-terminated LLM generation process.
    Maintains a persistent KV cache with attention-sink eviction.
    Accepts async observation injections.
    Yields Token objects to the Router.
    """

    def __init__(self, config: KVStreamConfig, system_prompt: str): ...

    def start(self) -> None:
        """Initialize model, build system prompt KV cache, begin generation loop."""

    async def run(self) -> AsyncGenerator[Token, None]:
        """
        Main loop. Yields tokens one at a time.
        After each token, checks injector queue.
        If injection pending, pauses generation, runs injection forward passes,
        resumes generation from updated cache state.
        """

    def inject(self, obs: str) -> None:
        """
        Thread-safe. Enqueue an observation string for injection.
        Called by the Environment via the Router's callback.
        """

    @property
    def cache_stats(self) -> CacheStats:
        """Returns current sink_used, window_used, total_evicted, position_counter."""
```

**Generation loop pseudocode** (the core of the system):

```python
async def run(self):
    position = self.prefill_length   # absolute position counter
    current_ids = self.last_prefill_token

    while True:
        # 1. Check injection queue before every token
        while not self.injector.empty():
            obs_text = self.injector.get_nowait()
            obs_ids = self.tokenizer.encode(obs_text)

            for tok_id in obs_ids:
                tok_tensor = torch.tensor([[tok_id]])
                cache_pos  = torch.tensor([position])
                attn_mask  = self._build_mask(position)

                self.model(
                    input_ids=tok_tensor,
                    cache_position=cache_pos,
                    attention_mask=attn_mask,
                    past_key_values=self.cache,
                    use_cache=True,
                )
                position += 1
                self.cache.evict_if_needed()   # StreamingLLM eviction

        # 2. Generate one token
        cache_pos = torch.tensor([position])
        attn_mask = self._build_mask(position)

        with torch.no_grad():
            logits = self.model(
                input_ids=current_ids,
                cache_position=cache_pos,
                attention_mask=attn_mask,
                past_key_values=self.cache,
                use_cache=True,
            ).logits[:, -1, :]

        next_id = self._sample(logits)   # temperature sampling
        current_ids = torch.tensor([[next_id]])
        position += 1
        self.cache.evict_if_needed()

        # 3. Yield to router
        yield Token(id=next_id, text=self.tokenizer.decode([next_id]))
```

**Critical invariant**: `cache_position` must increment monotonically and never skip values. The SinkCache remaps internal storage positions, but the absolute position counter seen by RoPE must be strictly increasing. Breaking this causes attention to collapse.

### 1.3 SinkCache — Attention-Sink Eviction Policy

`SinkCache` wraps the model's KV tensors. It implements the StreamingLLM eviction policy with an optional importance-based eviction extension.

```python
# engine/sink_cache.py

class SinkCache:
    """
    KV cache with attention-sink preservation (StreamingLLM, Xiao et al. ICLR 2024).

    Memory layout:
      [0 : num_sinks]              ← pinned forever (attention sinks)
      [num_sinks : num_sinks + P]  ← pinned tokens (goals, landmarks)
      [num_sinks + P : window]     ← rolling FIFO eviction zone

    When rolling zone exceeds capacity:
      - Standard mode: evict oldest token (FIFO)
      - Importance mode: evict lowest-attention token (H2O-style)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_sink_tokens: int = 4,
        window_length: int = 2048,
        eviction_policy: Literal["fifo", "importance"] = "fifo",
        pinned_token_ids: Optional[Set[int]] = None,   # token IDs that never evict
    ): ...

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV states. Evict if over capacity.
        Returns full (key, value) tensors for attention computation.
        """

    def evict_if_needed(self) -> int:
        """Evict one token if over capacity. Returns number evicted."""

    def pin_token(self, abs_position: int) -> None:
        """Mark a token at absolute position as never-evictable."""

    def get_attention_mask(self, query_position: int) -> torch.Tensor:
        """
        Build causal mask for current cache state.
        Accounts for non-contiguous positions after eviction.
        """

    @property
    def effective_seq_len(self) -> int:
        """Number of tokens currently in cache (sinks + pinned + window)."""
```

**The eviction zone** operates on remapped positions. After evicting token at internal index `k`, all tokens at indices `k+1...n` shift down by one internally. Externally, their absolute positions are unchanged — only the internal storage index changes. The attention mask is rebuilt from scratch each step to reflect the actual set of positions in cache.

**Pinned token protocol**: On startup, the system runs a forward pass for the system prompt including `<goal>...</goal>` and marks all `<goal>` token positions as pinned. These never leave the cache regardless of episode length.

### 1.4 Router — Token FSM and Dispatch

The Router sits between the KVStream output and all downstream consumers. It maintains a finite-state machine over decoded text to detect structured tags.

```python
# engine/router.py

class RouterState(Enum):
    PASSTHROUGH   = "passthrough"    # normal think text
    MAYBE_TAG     = "maybe_tag"      # saw '<', waiting to confirm
    IN_ACT_TAG    = "in_act_tag"     # confirmed <act, collecting
    IN_OBS_TAG    = "in_obs_tag"     # injected obs being echoed (skip)


class Router:
    """
    Stateful token router. Maintains a text buffer and FSM state.
    Dispatches complete <act> tags to registered handlers.
    Passes <think> text to the trace logger.
    Silences injected <obs> tokens (they are input, not output).
    """

    def __init__(self): ...

    def register_handler(self, cmd: str, fn: Callable[[ActTag], None]) -> None:
        """Register a handler for a specific act command."""

    def feed(self, token: Token) -> RouterOutput:
        """
        Process one token. Returns RouterOutput indicating disposition:
          - THINK: normal reasoning text (log/display)
          - ACT_DISPATCHED: complete act tag was parsed and handler called
          - ACT_PARTIAL: inside act tag, accumulating
          - SILENT: observation echo or other suppressed token
        """

    def _parse_act_tag(self, raw: str) -> ActTag:
        """
        Parse '<act cmd="move" dir="N"/>' into ActTag(cmd="move", params={"dir": "N"}).
        Uses xml.etree.ElementTree for robustness.
        """

    def flush(self) -> None:
        """Force-flush partial buffer. Called on episode end."""
```

**FSM transitions**:
```
PASSTHROUGH  ──── saw '<' ────►  MAYBE_TAG
MAYBE_TAG    ──── text="act" ──► IN_ACT_TAG
MAYBE_TAG    ──── text="obs" ──► IN_OBS_TAG
MAYBE_TAG    ──── other ───────► PASSTHROUGH (emit buffered '<' + token)
IN_ACT_TAG   ──── saw '/>' ────► PASSTHROUGH (dispatch, clear buffer)
IN_OBS_TAG   ──── saw '</obs>' ► PASSTHROUGH (silence, clear buffer)
```

**Malformed tag recovery**: If the model generates `<act` followed by a EOS token or an incompatible token sequence (detected by a 50-token timeout in `IN_ACT_TAG`), the router discards the buffer, transitions to `PASSTHROUGH`, and logs a malformed-tag event. This is counted in eval metrics.

### 1.5 Injector — Async Observation Queue

```python
# engine/injector.py

class ObsInjector:
    """
    Thread-safe queue connecting the Environment (producer)
    to the KVStream generation loop (consumer).

    Observations are held until the generation loop reaches a
    safe injection point (between tokens). They are then encoded
    and run through the model as forward passes, extending the KV cache.
    """

    def __init__(self, max_queue_size: int = 32): ...

    def put(self, obs: Observation) -> None:
        """Called by Environment. Non-blocking. Drops oldest if full."""

    def get_pending(self) -> List[str]:
        """
        Called by KVStream at each token step.
        Returns all pending observations formatted as injection strings.
        Flushes the queue.
        """

    def format_obs(self, obs: Observation) -> str:
        """Render Observation dataclass to <obs type="...">...</obs> string."""
```

**Injection timing**: Observations are not injected immediately. The KVStream checks the queue before generating each token. If observations are pending, it processes ALL of them in one batch before generating the next token. This means an observation injected during the middle of a `<think>` block will be visible to the model at the next token boundary — worst case latency is one inter-token interval (~15-50ms at target speeds).

### 1.6 Environment Interface

This is the contract between the engine and any test environment.

```python
# engine/interfaces.py

@dataclass
class Action:
    cmd: str
    params: Dict[str, str]


@dataclass
class Observation:
    type: str
    payload: str
    priority: int = 0     # higher priority observations preempt lower ones in queue


class Environment(ABC):
    """
    Abstract base class for any environment the streaming agent can control.
    The engine only calls step() and reset().
    The environment calls injector.put() when events occur.
    """

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment. Returns initial full-grid observation."""

    @abstractmethod
    def step(self, action: Action) -> Tuple[bool, Dict]:
        """
        Execute action. Returns (done, info).
        Environment is responsible for calling injector.put() for any
        events generated by this action (collision, enemy_near, etc.)
        """

    @abstractmethod
    def register_injector(self, injector: ObsInjector) -> None:
        """Give environment a reference to the injector."""

    @abstractmethod
    def render(self) -> str:
        """Return ASCII representation of current state."""
```

---

## Part 2: The Gridworld Environment

The Gridworld is a **separate, self-contained module**. It implements `Environment`. It has no imports from `engine/` except `interfaces.py`.

### 2.1 State Representation

```python
# env/gridworld.py

@dataclass
class GridState:
    width: int                          # default 10
    height: int                         # default 10
    agent_pos: Tuple[int, int]          # (col, row), 0-indexed
    goal_pos: Tuple[int, int]
    walls: FrozenSet[Tuple[int, int]]   # immutable set of wall cells
    enemies: List[EnemyState]
    step_count: int
    max_steps: int                      # episode terminates at this count


@dataclass
class EnemyState:
    pos: Tuple[int, int]
    patrol_path: List[Tuple[int, int]]  # deterministic loop
    path_index: int
    speed: int = 1                      # steps per game tick


# Cell encoding for ASCII render
CELL_CHARS = {
    "empty":  ".",
    "wall":   "W",
    "agent":  "A",
    "goal":   "G",
    "enemy":  "E",
    "both":   "X",   # agent and enemy on same cell = collision loss
}
```

### 2.2 Physics and Tick Model

```python
class GridWorld(Environment):

    def reset(self, scenario: str = "default") -> Observation:
        self.state = scenarios.build(scenario)
        return self._full_grid_obs()

    def step(self, action: Action) -> Tuple[bool, Dict]:
        """
        One game tick:
          1. Apply agent action
          2. Move all enemies one step along patrol path
          3. Detect events (collision, enemy_near, goal_near, goal_reached)
          4. Inject event observations into injector queue
          5. Return (done, info)
        """
        done = False
        events = []

        # 1. Agent move
        new_pos = self._apply_move(self.state.agent_pos, action)
        if new_pos in self.state.walls:
            new_pos = self.state.agent_pos   # blocked
            events.append(Observation("collision", f"wall at {action.params.get('dir','?')}"))
        self.state.agent_pos = new_pos

        # 2. Enemy moves
        for enemy in self.state.enemies:
            enemy.path_index = (enemy.path_index + enemy.speed) % len(enemy.patrol_path)
            enemy.pos = enemy.patrol_path[enemy.path_index]

        # 3. Event detection
        for enemy in self.state.enemies:
            dist = self._manhattan(self.state.agent_pos, enemy.pos)
            bearing = self._bearing(self.state.agent_pos, enemy.pos)
            if dist <= 2:
                events.append(Observation("enemy_near", "", priority=2))
                events[-1].payload = f'dist="{dist}" bearing="{bearing}"'
            if self.state.agent_pos == enemy.pos:
                events.append(Observation("death", "agent caught by enemy", priority=10))
                done = True

        goal_dist = self._manhattan(self.state.agent_pos, self.state.goal_pos)
        if goal_dist <= 2:
            events.append(Observation("goal_near", f'dist="{goal_dist}"'))
        if self.state.agent_pos == self.state.goal_pos:
            events.append(Observation("goal_reached", "task complete", priority=10))
            done = True

        if self.state.step_count >= self.state.max_steps:
            done = True

        # 4. Periodic tick heartbeat (every 10 steps)
        if self.state.step_count % 10 == 0:
            events.append(Observation("tick", f'step="{self.state.step_count}"'))

        # 5. Inject all events
        for obs in sorted(events, key=lambda o: o.priority):
            self.injector.put(obs)

        self.state.step_count += 1
        return done, {"events": events, "step": self.state.step_count}

    def render(self) -> str:
        """Return 10×10 ASCII grid with coordinates."""
```

**Why separate tick handling from injection**: The environment generates events synchronously during `step()`. The injector queue is async — the model may not see an observation until several tokens after it was enqueued. This is intentional and mirrors real-world embodied AI latency. Evaluation metrics account for this injection delay.

### 2.3 Scenarios

```python
# env/scenarios.py

SCENARIOS = {
    "static_maze": GridState(
        # 10×10 maze with fixed walls, no enemies
        # baseline: does the agent navigate at all?
    ),
    "single_patrol": GridState(
        # one enemy on a 6-cell patrol loop
        # primary test for observation-reactive replanning
    ),
    "dual_patrol": GridState(
        # two enemies on crossing patrol paths
        # tests multi-threat awareness
    ),
    "item_fetch": GridState(
        # agent must collect a key (intermediate goal) before reaching chest
        # tests long-horizon planning with pinned goal tokens
    ),
    "moving_goal": GridState(
        # goal cell changes every 20 steps (new obs injected)
        # tests goal-update handling in persistent KV stream
    ),
}
```

---

## Part 3: Evaluation Harness

### 3.0 Evaluation Strategy

The paper requires two evaluation environments serving different purposes.

**The gridworld** is the controlled ablation environment. Every variable is observable and deterministic. Enemy patrol paths are fixed seeds. Optimal paths are BFS-computable. The recovery latency metric is unambiguous because the triggering event (enemy enters radius ≤ 2) and the correct response (agent increases distance) are both rule-verifiable without a judge model. This is where the mechanism is isolated and measured.

**ALFWorld** is the credibility environment. It is an established text-based interactive benchmark (Shridhar et al. 2021) that reviewers already know and trust. Tasks involve navigating rooms, picking up objects, and completing multi-step household goals (e.g. "put a heated mug on the coffee table"). Episodes average 20-50 steps. We report standard ALFWorld completion rate alongside recovery latency on the subset of episodes that contain mid-task failures, giving reviewers both a familiar number to anchor to and the new metric that explains it.

**Why ALFWorld specifically, not NetHack or AgentBench**: ALFWorld tasks are long enough for the architecture's advantage to appear (>20 steps), the feedback loop is textual and clean (no pixel observation pipeline), and the failure modes (wrong room, empty receptacle, full hands) map cleanly to the recovery latency definition. NetHack is too complex to isolate the mechanism from model quality. AgentBench's OS tasks have too much variance.

**Where ALFWorld will not show a large advantage**: Tasks that are essentially a fixed known sequence with no mid-task failures. Both agents follow the same path and complete identically. The streaming architecture adds no value here -- this is expected and should be stated in the paper.

**The key theoretical claim**: For ALFWorld tasks that contain at least one mid-task failure event, the streaming agent's completion rate and recovery latency should both improve over the ReAct baseline. On tasks with zero mid-task failures, the two agents should be statistically equivalent. This is the falsifiable prediction.

```
Evaluation structure:
  Gridworld (controlled ablation)    ALFWorld (external validity)
  ─────────────────────────────      ────────────────────────────
  Primary metric: recovery latency   Primary metric: task completion rate
  Secondary: completion rate         Secondary: recovery latency on failure subset
  Purpose: isolate the mechanism     Purpose: credibility with reviewers
  Episodes: 20 × 4 scenarios         Episodes: full ALFWorld eval split (~134 tasks)
  Baselines: chunking (k=1,3,5)      Baselines: ReAct (Yao et al. 2023)
```

### 3.1 Metrics

```python
# eval/metrics.py

@dataclass
class EpisodeMetrics:
    completed: bool                     # reached goal before max_steps
    steps_taken: int
    optimal_steps: int                  # BFS shortest path at episode start (gridworld only)
    step_efficiency: float              # optimal / actual (1.0 = perfect)
    n_unexpected_events: int            # events not caused by agent's own action
    n_malformed_acts: int               # router rejected malformed <act> tags
    n_obs_injected: int                 # total observations injected
    had_mid_task_failure: bool          # at least one recovery event occurred
    recovery_events: List[RecoveryEvent]
    token_trace: List[Token]            # full token log for qualitative analysis
    environment: str                    # "gridworld" or "alfworld"


@dataclass
class RecoveryEvent:
    """
    Formal definition (applies to both environments):

    T_trigger: the step at which an unexpected event is injected into the stream.
               "Unexpected" = not a direct response to the agent's last action.
               Gridworld: enemy enters radius ≤ 2 on enemy's patrol tick.
               ALFWorld: action fails due to world state the agent did not anticipate
                         (empty receptacle, wrong room, object not interactable).

    T_recover: the first step at which the agent takes a contextually correct action
               in response to the triggering event.
               Correctness is verified by a deterministic rule-based judge:
               Gridworld: agent's manhattan distance to enemy increased.
               ALFWorld: agent's next action is semantically appropriate to the
                         failure (e.g. goes to search a different location after
                         "receptacle is empty").

    Recovery latency = T_recover - T_trigger (in agent steps).

    A recovery latency of 1 means the agent responded on the very next action.
    For a chunking agent with chunk_size=5, minimum recovery latency is 1
    (if event fires just before chunk boundary) and maximum is chunk_size
    (if event fires just after chunk boundary). Mean expected latency ≈ chunk_size/2.
    For the streaming agent, minimum and expected recovery latency is both 1.
    """
    trigger_step: int
    trigger_type: str                   # "enemy_near", "alfworld_failure", etc.
    resolved_step: int
    recovery_steps: int                 # T_recover - T_trigger
    agent_responded_correctly: bool     # rule-based judge verdict
    tokens_between: int                 # think tokens generated between trigger and response


@dataclass
class BenchmarkResult:
    environment: str
    scenario: str
    n_episodes: int
    completion_rate: float
    mean_step_efficiency: float         # gridworld only
    # Recovery latency stats — computed only on episodes with mid-task failures
    n_episodes_with_failures: int
    mean_recovery_steps: float          # KEY METRIC
    p50_recovery_steps: float
    p95_recovery_steps: float
    correct_recovery_rate: float        # % of recovery events judged correct
    malformed_act_rate: float
```

### 3.2 Baselines

Three baselines, all using the same Qwen3-8B Q4 model and temperature 0.6:

```python
# eval/baseline_chunker.py

class ActionChunkingBaseline:
    """
    Discrete call-and-respond loop. Represents the dominant current paradigm.

    At each step:
      1. Build prompt: system prompt + full current state observation
      2. Call model.generate(max_new_tokens=CHUNK_SIZE * avg_tokens_per_act)
      3. Parse all <act> tags from response
      4. Execute them sequentially against the environment
      5. Any events that fire mid-chunk are NOT seen until the chunk exhausts
         and the next generate() call includes the updated state

    Run with chunk_size = 1, 3, 5 to show recovery latency scales linearly
    with chunk size. This is the smoking gun for the paper:
      chunk_size=1  → mean recovery ≈ 1.0 steps  (effectively ReAct)
      chunk_size=3  → mean recovery ≈ 2.0 steps
      chunk_size=5  → mean recovery ≈ 3.0 steps
      streaming     → mean recovery ≈ 1.0 steps  (same as chunk_size=1)
                                                   but with lower per-action
                                                   latency and no cold-start
                                                   prefill cost per step
    """
    def __init__(self, model, tokenizer, chunk_size: int = 5): ...
    def run_episode(self, env: Environment) -> EpisodeMetrics: ...


class ReActBaseline:
    """
    Standard ReAct (Yao et al. 2023): Thought → Action → Observation, one step at a time.
    Used specifically for ALFWorld comparison since ReAct is the published baseline there.
    Equivalent to ActionChunkingBaseline with chunk_size=1 but with explicit
    Thought/Action/Observation formatting matching the ReAct paper.
    """
    def __init__(self, model, tokenizer): ...
    def run_episode(self, env: Environment) -> EpisodeMetrics: ...
```

**The important nuance about chunk_size=1 vs streaming**: At chunk_size=1 (ReAct), recovery latency approaches 1 step -- same as streaming. The difference is not recovery latency at chunk_size=1, it is **cost per step**. ReAct at chunk_size=1 runs a full prefill for every single action (O(n²) attention over full context). The streaming agent amortizes this -- no prefill cost per step, just one additional decode step. This matters for long episodes and is a secondary contribution of the paper.

### 3.3 ALFWorld Integration

ALFWorld requires wrapping the existing ALFWorld environment to implement the `Environment` ABC.

```python
# env/alfworld_env.py

class ALFWorldEnv(Environment):
    """
    Wraps the alfworld.agents.environment package to conform to the Engine's
    Environment interface.

    Key differences from GridWorld:
    - Actions are natural language strings, not structured <act> tags
      → The act tag cmd is mapped to ALFWorld action strings via a lookup
      → e.g. <act cmd="goto" obj="microwave"/> → "go to microwave"
    - Observations are natural language descriptions, not structured events
      → ALFWorld feedback is already text; wrap in <obs type="alfworld"> tags
    - "Unexpected" events are action failures (not enemy spawns)
      → Detected when ALFWorld returns "Nothing happens." or "You can't..."
      → These are injected as high-priority observations triggering recovery tracking

    Install: pip install alfworld
    Data:    export ALFWORLD_DATA=<path>; alfworld-download
    """

    ALFWORLD_ACTION_MAP = {
        ("goto",   "obj"):  "go to {obj}",
        ("take",   "obj"):  "take {obj} from {container}",
        ("put",    "obj"):  "put {obj} in/on {container}",
        ("open",   "obj"):  "open {obj}",
        ("close",  "obj"):  "close {obj}",
        ("toggle", "obj"):  "use {obj}",
        ("examine","obj"):  "examine {obj}",
        ("done",   None):   "DONE",
    }

    def step(self, action: Action) -> Tuple[bool, Dict]:
        alfworld_action = self._map_action(action)
        obs_text, reward, done, info = self.env.step([alfworld_action])

        # Detect failure events and inject as high-priority observations
        if any(fail in obs_text[0] for fail in ["Nothing happens", "can't", "isn't"]):
            self.injector.put(Observation(
                type="alfworld_failure",
                payload=obs_text[0].strip(),
                priority=5,
            ))
        else:
            self.injector.put(Observation(
                type="alfworld_feedback",
                payload=obs_text[0].strip(),
                priority=1,
            ))

        return done, {"reward": reward, "raw_obs": obs_text[0]}
```

### 3.4 Harness

```python
# eval/harness.py

def run_benchmark(
    agent_config: KVStreamConfig,
    environment: Literal["gridworld", "alfworld", "both"],
    n_gridworld_episodes_per_scenario: int = 20,
    compare_baselines: List[str] = ["react", "chunk_1", "chunk_3", "chunk_5"],
    seed: int = 42,
    results_dir: str = "./results",
) -> Dict[str, BenchmarkResult]:
    """
    Runs full evaluation suite.
    Seeds RNG for deterministic episode ordering across all agent types.
    Writes EpisodeMetrics to results_dir after each episode (resume-safe).
    Returns BenchmarkResult dict keyed by f"{environment}/{scenario}/{agent}".
    """
```

**Episode counts for the paper**:
```
Gridworld:
  4 scenarios × 20 episodes × 4 agent types (stream + chunk_1/3/5) = 320 episodes

ALFWorld:
  134 eval tasks × 2 agent types (stream + react) = 268 episodes
  (ALFWorld tasks are deterministic so no seed variation needed)

Total: 588 episodes
Estimated runtime on Colab A100: ~6-8 hours
```

---

## Part 4: System Prompt

The system prompt is the only place the model is told how to behave. No fine-tuning. The prompt establishes the token vocabulary, the cognitive loop, and the pinned goal.

```
You are an autonomous game agent. You think continuously and act by emitting
structured tags into your reasoning stream.

THINKING: Write your reasoning freely between actions. Think out loud about
what you observe, what threats exist, and what you intend to do next.

ACTING: When you decide to move or wait, emit exactly one act tag:
  <act cmd="move" dir="N"/>   — move north (N/S/E/W)
  <act cmd="wait"/>           — stay in place one tick
  <act cmd="look"/>           — request full grid view
  <act cmd="done"/>           — signal task complete

OBSERVATIONS: The environment will inject observations into your stream:
  <obs type="collision">...</obs>       — you hit a wall
  <obs type="enemy_near" dist="X" bearing="Y"/>  — enemy within 2 cells
  <obs type="goal_near" dist="X"/>      — goal within 2 cells
  <obs type="tick" step="X"/>           — periodic step counter
  <obs type="full_grid">...</obs>       — full grid view

React to observations immediately in your next reasoning tokens.
After each observation, decide whether to act or continue observing.
You do not need to act every step — thinking carefully before acting is good.

<goal>Navigate to G while avoiding enemies E. Walls W block movement.</goal>
```

---

## Part 5: Hardware Configuration

### Deployment Decision

**All engine development and live agent runs happen locally on Mac. Colab is used exclusively for evaluation batch runs.**

The "never-terminated process" requirement is fundamentally incompatible with Colab's session model. Colab Pro sessions are hard-killed after 12-24 hours and have a ~90 minute idle timeout. A KV cache that gets destroyed mid-episode is not a continuous stream. The architecture's core invariant — one unbroken process, one monotonically incrementing position counter — requires a machine you control.

Colab is appropriate for `eval/harness.py` because each episode is bounded (max 200 steps) and stateless across episodes. The eval harness creates a fresh engine instance per episode — no persistent session needed.

```
Mac (M2/M3 Pro, 18-36GB unified memory)
  ├── All engine/ development
  ├── All env/ development
  ├── Unit tests
  ├── Single-episode interactive runs (scripts/run_agent.py)
  └── The persistent KV stream — never restart mid-episode

Colab Pro (T4 or A100, eval only)
  └── scripts/run_eval.py — 20 episodes × 4 scenarios × 2 agents
      Each episode = fresh engine instance, bounded at 200 steps
      No session persistence required
```

### Mac — Primary (llama-cpp-python + Metal)

M2/M3 Pro with 18-36GB unified memory runs Qwen3-8B Q4_K_M comfortably. The unified memory architecture benefits the persistent cache pattern specifically: model weights and KV cache tensors share the same pool with no VRAM/RAM transfer overhead.

```yaml
# configs/qwen3_8b_mac.yaml
backend: llama
model_id: bartowski/Qwen_Qwen3-8B-GGUF
model_file: Qwen3-8B-Q4_K_M.gguf   # ~5.5GB on disk
n_gpu_layers: -1                     # full Metal offload — must verify with Activity Monitor
n_ctx: 4096                          # 4096 sufficient; 8192 needs Q3_K_M on 18GB
sink_tokens: 4
window_length: 2048                  # effective rolling window: 2044 tokens
temperature: 0.6
n_threads: 8
expected_tok_s: 30-45               # M2/M3 Pro 18-36GB
vram_budget_gb: ~12                  # 5.5 model + 6.5 KV cache at n_ctx=4096
```

**Install**:
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall

huggingface-cli download bartowski/Qwen_Qwen3-8B-GGUF \
    --include "Qwen3-8B-Q4_K_M.gguf" \
    --local-dir ./models

# Verify Metal offload
python -c "
from llama_cpp import Llama
m = Llama('./models/Qwen3-8B-Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=4096, verbose=True)
# Check stderr for: 'ggml_metal_init: allocating...' and layer offload count
"
```

**Memory headroom by config**:
```
18GB unified memory:
  Q4_K_M (5.5GB) + n_ctx=4096 KV (~6GB F16) = ~11.5GB  ✅ comfortable
  Q4_K_M (5.5GB) + n_ctx=8192 KV (~12GB F16) = ~17.5GB  ⚠️  tight, use Q3_K_M instead

36GB unified memory:
  Q4_K_M (5.5GB) + n_ctx=8192 KV (~12GB F16) = ~17.5GB  ✅ fine
```

### Colab T4 — Eval Only

```yaml
# configs/qwen3_8b_colab_t4.yaml
backend: hf
model_id: Qwen/Qwen3-8B
load_in_4bit: true              # bitsandbytes NF4
bnb_4bit_compute_dtype: float16
bnb_4bit_use_double_quant: true
sink_tokens: 4
window_length: 1024             # conservative: 16GB VRAM, 8B 4-bit ~5GB + cache
temperature: 0.6
device_map: auto
expected_tok_s: 50-80
use_case: eval_only             # do not use for persistent stream runs
```

### Colab A100 — Eval Only (preferred if available)

```yaml
# configs/qwen3_8b_colab_a100.yaml
backend: hf
model_id: Qwen/Qwen3-8B
torch_dtype: float16            # full precision — no quality compromise for eval
sink_tokens: 4
window_length: 4096
temperature: 0.6
device_map: auto
expected_tok_s: 100-130
use_case: eval_only
```

**Colab eval setup** (`scripts/run_eval.py` preamble):
```python
# Mount Drive for results persistence across sessions
from google.colab import drive
drive.mount('/content/drive')
RESULTS_DIR = '/content/drive/MyDrive/streamagent_results/'

# Each episode writes its EpisodeMetrics to Drive immediately on completion
# If session dies mid-eval, resume from last completed episode index
```

---

## Part 6: Key Implementation Risks and Mitigations

| Risk | Description | Mitigation |
|------|-------------|------------|
| Cache position drift | `cache_position` must be strictly monotonic across injections. Off-by-one errors cause RoPE collapse | Unit test: assert `cache_position[-1] == prev + 1` after every step |
| Attention mask shape mismatch | After eviction, mask shape changes. Must rebuild every step | `SinkCache.get_attention_mask()` is authoritative; never cache the mask externally |
| KV precision loss under repeated eviction | llama.cpp quantized KV (Q4/Q8) degrades after thousands of evictions on Mac | Use F16 KV cache (`kv_cache_type=f16` in llama.cpp); use eviction batch size ≥256 tokens to reduce RoPE shift accumulation |
| Malformed `<act>` tag mid-generation | Model may start `<act` but fail to close it (EOS, mid-token injection) | Router 50-token timeout; fallback to `<act cmd="wait"/>` on timeout |
| Observation flood | Environment fires many events simultaneously; model CoT interrupted continuously | Injector batches all pending observations into one injection pass; deduplicate same-type obs within one tick |
| Metal offload not fully engaged | Some llama.cpp builds silently fall back to CPU layers on Mac | Verify with `verbose=True` on init: all 32 layers must show `[Metal]`; re-install with `CMAKE_ARGS="-DGGML_METAL=on"` if not |
| Colab eval session dies mid-run | T4 sessions can be preempted mid-eval batch | `run_eval.py` writes each `EpisodeMetrics` to Drive immediately on episode completion; resume flag skips already-completed episodes |

---

## Part 7: The Paper

### Research Question

> **Does eliminating the observe-reason-act boundary in LLM-based game agents reduce recovery latency from unexpected environmental events, and does this translate to higher task completion on established benchmarks?**

### Contribution Claims

**C1 — Architecture**: A continuous inference engine where a reasoning LLM runs as a never-terminated process, emitting interleaved `<think>` and `<act>` tokens over a persistent StreamingLLM KV cache, with asynchronous mid-stream observation injection. No fine-tuning. No new model weights.

**C2 — Metric**: A formal definition of *recovery latency* -- steps between an unexpected event injection and the agent's first contextually correct response -- with a deterministic rule-based judge applicable to both gridworld and ALFWorld environments. This metric exposes a gap that standard completion-rate metrics miss entirely.

**C3 — Empirical finding**: Recovery latency scales linearly with chunk size in action-chunking agents. The streaming agent breaks this scaling, achieving recovery latency ≈ 1 step regardless of episode length, while eliminating per-step prefill cost for long episodes.

### Paper Structure (Workshop / Short Paper Target)

```
1. Introduction
   - The observe-reason-act loop as the dominant paradigm
   - Its structural weakness: events that fire mid-chunk are invisible until next call
   - Our claim: make the process never-terminate; observations inject into live stream

2. Related Work
   - StreamingLLM (KV cache management)
   - ReAct (discrete think/act/observe loop — our primary baseline)
   - Action Chunking / ACT (chunking baseline)
   - ThinkStream / VST (streaming CoT for video — not embodied control)
   - SIMA 2 (embodied game agent — not continuous autoregressive)
   - Gap: no prior work combines all five properties (Table 1 from prior spec)

3. Architecture
   - KVStream: never-terminated generation loop
   - SinkCache: attention-sink eviction for bounded memory
   - ObsInjector: async queue + mid-stream forward pass injection
   - Token Router: FSM-based tag detection and dispatch
   - Diagram from spec overview

4. Experimental Setup
   4.1 Gridworld (controlled ablation)
       - 4 scenarios, 20 episodes each
       - Baselines: chunk_size = 1, 3, 5
       - Metrics: recovery latency, completion rate
   4.2 ALFWorld (external validity)
       - Full eval split (134 tasks)
       - Baseline: ReAct (Yao et al. 2023)
       - Metrics: completion rate, recovery latency on failure subset

5. Results
   5.1 Gridworld: recovery latency vs chunk size (the key graph)
       Expected: linear scaling for chunking, flat ≈1 for streaming
   5.2 ALFWorld: completion rate on tasks WITH vs WITHOUT mid-task failures
       Expected: equivalent on zero-failure tasks; streaming wins on failure tasks
   5.3 Ablation: recovery latency vs event density (events per step)
       Expected: crossover point where chunking collapses, streaming degrades gracefully

6. Discussion
   - Where the architecture does not help (short episodes, no unexpected events)
   - The prefill cost argument for long episodes
   - Limitations: model quality dominates at low event density; no visual observations

7. Conclusion
```

### Target Venues

```
Primary targets (4-6 page workshop papers):
  NeurIPS 2025 Workshop on Embodied AI
  NeurIPS 2025 Workshop on Interactive Learning
  ICLR 2026 Workshop on Agent Learning in Open-Endedness

Secondary targets (short paper, 4 pages):
  COLM 2026 (Conference on Language Modeling)
  EMNLP 2025 short papers

Full paper (8 pages) — only if numbers are very clean:
  ICLR 2026 (main track)
  NeurIPS 2026 (main track)
```

### Falsifiability

The paper is falsifiable in a clean way, which reviewers respect.

If recovery latency for the streaming agent is **not** significantly lower than chunk_size=1 (ReAct), the architecture provides no reactive advantage and the C3 claim fails. This is not fatal -- it would mean the model's reasoning speed is the bottleneck, not the loop architecture, which is itself a finding worth reporting.

If completion rate on ALFWorld failure-subset tasks is **equivalent** for streaming and ReAct, the external validity claim weakens. The paper can still stand on gridworld + metric contribution alone, but the scope narrows to a workshop paper rather than main track.

**The result that would kill the paper entirely**: streaming agent performing worse than ReAct on both environments. This would indicate that continuous generation without episode boundaries degrades coherence more than it gains in reactivity -- possible if the SinkCache evicts too much relevant context. Mitigated by the pinned `<goal>` token design.

---

## Dependencies

```
# requirements.txt — Mac (primary)
llama-cpp-python>=0.3.16   # primary Mac backend — install with CMAKE_ARGS="-DGGML_METAL=on"
huggingface_hub>=0.24.0    # model download
numpy>=1.26.0
pyyaml>=6.0
rich>=13.0                 # terminal renderer (optional)
pytest>=8.0

# requirements-eval.txt — Colab only (eval runs)
torch>=2.3.0
transformers>=4.44.0       # SinkCache added in 4.44
accelerate>=0.30.0
bitsandbytes>=0.43.0       # 4-bit quantization on T4
huggingface_hub>=0.24.0
numpy>=1.26.0
pyyaml>=6.0
alfworld>=0.3.3            # ALFWorld evaluation environment
pytest>=8.0

# Optional Mac alternative backend
mlx-lm>=0.31.2             # faster on M-series but no native SinkCache
```

**ALFWorld data setup** (Colab, one-time):
```bash
pip install alfworld
export ALFWORLD_DATA=/content/alfworld_data
alfworld-download
# Downloads ~1.5GB of task data to ALFWORLD_DATA
# Mount to Drive to avoid re-downloading across sessions:
# ln -s /content/drive/MyDrive/alfworld_data $ALFWORLD_DATA
```

---

*This spec describes the minimum system required to run the experiment. The engine is designed to be environment-agnostic — any environment implementing the `Environment` ABC can be swapped in without modifying the engine.*
