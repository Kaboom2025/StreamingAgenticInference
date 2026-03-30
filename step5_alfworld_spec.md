# Step 5: ALFWorld Integration Spec
**Prerequisite: Step 3 (eval harness) complete and GridWorld numbers in hand**

---

## What we're building

A wrapper that makes ALFWorld look identical to `CalcEnv` and `GridWorld` from the engine's perspective. The engine sees the same `Environment` ABC it already talks to. Nothing in `engine/` changes.

```
streamagent/
└── env/
    ├── calc_env.py          ← already done
    ├── gridworld.py         ← already done
    └── alfworld_env.py      ← this spec
```

---

## What ALFWorld actually is

ALFWorld is a text-based environment of household tasks. Under the hood it's a PDDL simulator with a text interface. Tasks look like:

```
Task:   "Put a hot cup on the counter"
State:  "You are in the kitchen. You see a fridge, a microwave, a counter."
Action: "go to microwave"
Obs:    "You open the microwave. You see a cup inside."
Action: "take cup from microwave"
Obs:    "You pick up the cup."
...
```

Episodes average 20-50 steps. There are 6 task types: `pick_and_place`, `pick_two_obj`, `look_at_obj`, `pick_clean`, `pick_heat`, `pick_cool`. The eval split has 134 tasks.

The key property for this paper: **action failures fire mid-episode as text observations.** When the agent tries to take an object that isn't there, or go to a room it can't reach, ALFWorld returns `"Nothing happens."` or `"You can't do that."` These are the "unexpected events" that the streaming architecture should handle faster than ReAct.

---

## Install and data

```bash
# requirements-eval.txt addition
pip install alfworld>=0.3.3

# One-time data download (~1.5GB)
export ALFWORLD_DATA=/path/to/alfworld_data
alfworld-download

# On Colab: persist to Drive to avoid re-downloading
export ALFWORLD_DATA=/content/drive/MyDrive/alfworld_data
alfworld-download  # only needed once
```

---

## The wrapper

```python
# env/alfworld_env.py

import alfworld.agents.environment as alf_env
from engine.interfaces import Environment, Action, Observation, ObsInjector
from typing import Tuple, Dict, Optional
import re


# Strings ALFWorld returns when an action fails unexpectedly
FAILURE_STRINGS = [
    "Nothing happens.",
    "You can't",
    "That's not something you can",
    "There is no",
    "I can't see",
]


class ALFWorldEnv(Environment):
    """
    Wraps ALFWorld's InteractiveWADITHEnvironment to implement Environment ABC.

    The engine talks to this exactly as it talks to GridWorld.
    The wrapper is responsible for:
      1. Mapping <act> tag params to ALFWorld action strings
      2. Detecting failure observations and injecting them as high-priority obs
      3. Detecting success and returning done=True
    """

    def __init__(self, config_path: str, task_type: str = "all", split: str = "eval"):
        self.config_path = config_path
        self.task_type = task_type
        self.split = split
        self.env = None
        self.injector: Optional[ObsInjector] = None
        self.current_task: Optional[str] = None
        self._step_count = 0

    def register_injector(self, injector: ObsInjector) -> None:
        self.injector = injector

    def reset(self) -> Observation:
        if self.env is None:
            self.env = alf_env.AlfredTWEnv(
                config=self._load_config(),
                train_eval=self.split,
            )
            self.env = self.env.init_env(batch_size=1)

        obs, info = self.env.reset()
        self._step_count = 0
        self.current_task = obs[0]   # ALFWorld includes task description in first obs

        return Observation(
            type="alfworld_reset",
            payload=obs[0].strip(),
            priority=0,
        )

    def step(self, action: Action) -> Tuple[bool, Dict]:
        assert self.injector is not None, "call register_injector() before step()"

        alfworld_action = self._map_action(action)
        obs, scores, dones, info = self.env.step([alfworld_action])

        raw_obs = obs[0].strip()
        done = dones[0]
        self._step_count += 1

        # Determine observation priority based on content
        is_failure = any(f in raw_obs for f in FAILURE_STRINGS)
        is_success = done and scores[0] > 0

        if is_success:
            self.injector.put(Observation(
                type="alfworld_success",
                payload=raw_obs,
                priority=10,
            ))
        elif is_failure:
            # This is the key injection: unexpected failure mid-episode
            # Triggers recovery latency measurement
            self.injector.put(Observation(
                type="alfworld_failure",
                payload=raw_obs,
                priority=5,
            ))
        else:
            self.injector.put(Observation(
                type="alfworld_feedback",
                payload=raw_obs,
                priority=1,
            ))

        return done, {
            "score": scores[0],
            "raw_obs": raw_obs,
            "is_failure": is_failure,
            "step": self._step_count,
        }

    def render(self) -> str:
        return f"[ALFWorld step {self._step_count}] task: {self.current_task}"

    def _map_action(self, action: Action) -> str:
        """
        Map <act> tag to ALFWorld action string.

        ALFWorld expects natural language like "go to microwave" or
        "take cup from microwave". We map structured act params to these.

        Examples:
          <act cmd="goto" obj="microwave"/>        → "go to microwave"
          <act cmd="take" obj="cup" from="table"/> → "take cup from table"
          <act cmd="put" obj="cup" on="counter"/>  → "put cup in/on counter"
          <act cmd="open" obj="fridge"/>           → "open fridge"
          <act cmd="close" obj="fridge"/>          → "close fridge"
          <act cmd="heat" obj="cup" with="microwave"/> → "heat cup with microwave"
          <act cmd="cool" obj="cup" with="fridge"/>    → "cool cup with fridge"
          <act cmd="clean" obj="cup" with="sink"/>     → "clean cup with sink"
          <act cmd="examine" obj="cup"/>           → "examine cup"
          <act cmd="look"/>                        → "look"
          <act cmd="inventory"/>                   → "inventory"
          <act cmd="done"/>                        → "DONE"
        """
        cmd = action.cmd
        p = action.params

        mapping = {
            "goto":      lambda: f"go to {p['obj']}",
            "take":      lambda: f"take {p['obj']} from {p.get('from', 'it')}",
            "put":       lambda: f"put {p['obj']} in/on {p.get('on', p.get('in', 'it'))}",
            "open":      lambda: f"open {p['obj']}",
            "close":     lambda: f"close {p['obj']}",
            "heat":      lambda: f"heat {p['obj']} with {p.get('with', 'microwave')}",
            "cool":      lambda: f"cool {p['obj']} with {p.get('with', 'fridge')}",
            "clean":     lambda: f"clean {p['obj']} with {p.get('with', 'sink')}",
            "examine":   lambda: f"examine {p['obj']}",
            "toggle":    lambda: f"use {p['obj']}",
            "look":      lambda: "look",
            "inventory": lambda: "inventory",
            "done":      lambda: "DONE",
        }

        if cmd not in mapping:
            # Unknown command — return look as safe fallback
            return "look"

        return mapping[cmd]()

    def _load_config(self) -> dict:
        import yaml
        with open(self.config_path) as f:
            return yaml.safe_load(f)
```

---

## Updated system prompt for ALFWorld

The gridworld system prompt won't work here. ALFWorld needs a different action vocabulary and the model needs to understand household tasks. Keep the tag format identical -- only the act commands and examples change.

```
You are an autonomous household task agent. You think continuously and act
by emitting structured tags into your reasoning stream.

THINKING: Reason freely between actions. Think about where you are, what
you need, and what the most likely next step is.

ACTING: Emit exactly one act tag when you decide to act:
  <act cmd="goto" obj="microwave"/>
  <act cmd="take" obj="cup" from="table"/>
  <act cmd="put" obj="cup" on="counter"/>
  <act cmd="open" obj="fridge"/>
  <act cmd="close" obj="fridge"/>
  <act cmd="heat" obj="cup" with="microwave"/>
  <act cmd="cool" obj="cup" with="fridge"/>
  <act cmd="clean" obj="cup" with="sink"/>
  <act cmd="examine" obj="cup"/>
  <act cmd="look"/>
  <act cmd="inventory"/>
  <act cmd="done"/>

OBSERVATIONS: The environment injects feedback into your stream:
  <obs type="alfworld_feedback">You open the fridge. You see a cup.</obs>
  <obs type="alfworld_failure">Nothing happens.</obs>
  <obs type="alfworld_success">Task complete.</obs>

When you see a failure observation, stop and reconsider. The action did not
work. Try a different approach rather than repeating the same action.

FEW-SHOT EXAMPLE:
I need to find a heated cup. Let me start by looking around.
<act cmd="look"/>
<obs type="alfworld_feedback">You are in the kitchen. You see a microwave, a fridge, a counter with a cup on it.</obs>
There's a cup on the counter. I'll take it then heat it in the microwave.
<act cmd="take" obj="cup" from="counter"/>
<obs type="alfworld_feedback">You pick up the cup.</obs>
Good. Now go to the microwave to heat it.
<act cmd="goto" obj="microwave"/>
<obs type="alfworld_feedback">You arrive at the microwave.</obs>
<act cmd="heat" obj="cup" with="microwave"/>
<obs type="alfworld_feedback">You heat the cup in the microwave.</obs>
Now put it on the counter as instructed.
<act cmd="put" obj="cup" on="counter"/>
<obs type="alfworld_success">Task complete.</obs>
<act cmd="done"/>
```

---

## Recovery latency in ALFWorld

The gridworld definition used manhattan distance to measure correct recovery. ALFWorld needs a different judge because there's no spatial metric.

```python
# eval/alfworld_judge.py

def is_correct_recovery(
    failure_obs: str,
    recovery_action: Action,
    prior_action: Action,
) -> bool:
    """
    Rule-based judge for ALFWorld recovery correctness.
    Returns True if the recovery action is a sensible response to the failure.

    Rules (in order):
    1. If prior_action == recovery_action: WRONG (repeating failed action)
    2. If recovery_action.cmd == "look" or "inventory": CORRECT
       (gathering more information is always a valid recovery)
    3. If recovery_action.cmd == "goto" and obj != prior_action.params.get("obj"):
       CORRECT (going somewhere different is a valid pivot)
    4. If recovery_action.cmd == "examine": CORRECT
       (inspecting the environment is valid)
    5. Otherwise: AMBIGUOUS (log but don't count either way)

    Note: this judge is intentionally conservative. It only marks clearly
    wrong recoveries as wrong (repeating the failed action). Ambiguous
    cases are excluded from the recovery rate metric to avoid noise.
    """
    if recovery_action.cmd == prior_action.cmd and \
       recovery_action.params == prior_action.params:
        return False   # repeated the exact same failed action

    if recovery_action.cmd in ("look", "inventory", "examine"):
        return True    # information-gathering is always valid

    if recovery_action.cmd == "goto":
        prior_obj = prior_action.params.get("obj", "")
        recovery_obj = recovery_action.params.get("obj", "")
        return recovery_obj != prior_obj   # going somewhere different

    return None   # ambiguous — exclude from metric
```

---

## What changes in the eval harness

The harness already supports swapping environments via the `Environment` ABC. The only additions needed:

```python
# eval/harness.py additions

def run_alfworld_benchmark(
    agent_config: KVStreamConfig,
    alfworld_config_path: str,
    task_types: List[str] = ["all"],   # or specific: ["pick_heat", "pick_cool"]
    compare_react: bool = True,
    results_dir: str = "./results/alfworld",
) -> Dict[str, BenchmarkResult]:
    """
    Runs full ALFWorld eval split (134 tasks).
    Each task = one episode = one fresh engine instance.
    Writes results per-episode to results_dir immediately (resume-safe).

    Key addition vs gridworld harness:
    - Splits results by task_type for granular analysis
    - Splits results by had_mid_task_failure (True/False)
      The False subset should show no difference between agents.
      The True subset is where the architecture's advantage appears.
    """
```

**The critical split**: after running all 134 tasks, partition episodes into:
- `no_failures`: zero `alfworld_failure` observations injected
- `with_failures`: one or more `alfworld_failure` observations

Report completion rate separately for each partition. If the architecture works, `with_failures` completion rate should be higher for the streaming agent. `no_failures` should be equivalent. That's the cleanest result for a paper.

---

## Episode count and runtime

```
ALFWorld eval split: 134 tasks (fixed, deterministic — no seeds needed)

Agents:
  - StreamAgent (continuous KV stream)
  - ReAct baseline (discrete loop, chunk_size=1)

Total episodes: 134 × 2 = 268

Runtime estimate:
  Colab A100, Qwen3-8B FP16, ~100 tok/s:
  Average episode ~300 tokens → ~3s per episode
  268 episodes → ~15 minutes total

  Colab T4, Qwen3-8B 4-bit, ~60 tok/s:
  Average episode ~300 tokens → ~5s per episode
  268 episodes → ~25 minutes total
```

ALFWorld is fast enough that a full eval run fits comfortably inside a single Colab session with room to spare.

---

## File checklist for Step 5

```
New files:
  env/alfworld_env.py          ← wrapper (this spec)
  eval/alfworld_judge.py       ← recovery correctness judge
  configs/alfworld_eval.yaml   ← alfworld config path + task type

Modified files:
  eval/harness.py              ← add run_alfworld_benchmark()
  eval/metrics.py              ← add alfworld_judge call in RecoveryEvent
  configs/qwen3_8b_colab_a100.yaml  ← add alfworld system prompt path
  requirements-eval.txt        ← add alfworld>=0.3.3
```

---

## Order of operations

1. Get GridWorld eval numbers first (Step 3). Don't start Step 5 until you have a recovery latency number from GridWorld that makes sense.
2. Install ALFWorld on Colab, run one episode manually with `demo_alfworld.py` to verify the wrapper maps actions correctly.
3. Run 10 episodes of each agent on `pick_and_place` tasks only (simplest task type). Check that failure observations are being injected and that the recovery judge is firing.
4. Run full 134-task eval split once numbers look stable.

The single biggest risk is action mapping -- if the model generates `<act cmd="take" obj="cup"/>` but forgets the `from` parameter, the mapped action is `"take cup from it"` which ALFWorld won't understand. Watch the malformed/unmapped action rate in early runs and fix the system prompt examples if it's high.
