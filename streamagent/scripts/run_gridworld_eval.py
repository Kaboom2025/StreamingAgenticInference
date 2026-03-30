"""GridWorld streaming vs chunking benchmark.

Runs N episodes per scenario with the streaming agent, then computes
chunking-baseline recovery latency analytically from the same results.
One model run produces all table columns — no second inference pass needed.

Usage:
    python -m streamagent.scripts.run_gridworld_eval
    python -m streamagent.scripts.run_gridworld_eval --n-episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from streamagent.engine.kv_stream import KVStream, KVStreamConfig
from streamagent.env.gridworld import GridWorld
from streamagent.env.scenarios import load_scenarios
from streamagent.eval.harness import run_gridworld_episode
from streamagent.eval.metrics import EpisodeMetrics

_REPO_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = _REPO_ROOT / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
SYSTEM_PROMPT_PATH = _REPO_ROOT / "configs" / "gridworld_system_prompt.txt"

CHUNK_SIZES = [1, 3, 5]


def _build_config(model_id: str | None = None) -> KVStreamConfig:
    return KVStreamConfig(
        model_id=model_id if model_id is not None else str(MODEL_PATH),
        backend="llama",
        sink_tokens=4,
        window_length=2048,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
    )


def _mean_recovery(metrics_list: list[EpisodeMetrics]) -> float:
    if not metrics_list:
        return 0.0
    return sum(m.mean_recovery_tokens for m in metrics_list) / len(metrics_list)


def _mean_chunking(metrics_list: list[EpisodeMetrics], chunk_size: int) -> float:
    if not metrics_list:
        return 0.0
    return sum(m.mean_chunking_recovery_tokens(chunk_size) for m in metrics_list) / len(metrics_list)


def _print_table(results: dict[str, list[EpisodeMetrics]]) -> None:
    print("\n\n=== RESULTS ===")
    header = f"{'scenario':<20} {'stream':>10} {'chunk_1':>10} {'chunk_3':>10} {'chunk_5':>10}"
    print(header)
    print("-" * 62)
    for scenario_name, metrics in results.items():
        s = _mean_recovery(metrics)
        c1 = _mean_chunking(metrics, 1)
        c3 = _mean_chunking(metrics, 3)
        c5 = _mean_chunking(metrics, 5)
        print(f"{scenario_name:<20} {s:>10.1f} {c1:>10.1f} {c3:>10.1f} {c5:>10.1f}")
    print("\nMetric: mean recovery latency in tokens (lower = better)")
    print("simple_4x4 should have 0 recovery events if no enemies are present")


def main(n_episodes: int = 5, model_id: str | None = None) -> None:
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    config = _build_config(model_id=model_id)
    scenarios = load_scenarios()
    results: dict[str, list[EpisodeMetrics]] = {}

    for scenario in scenarios:
        print(f"\n--- {scenario.name} ---")
        metrics: list[EpisodeMetrics] = []

        for ep in range(n_episodes):
            env = GridWorld(scenario=scenario)
            stream = KVStream(config=config, system_prompt=system_prompt)
            m = run_gridworld_episode(stream, env, max_tokens=2000)
            metrics.append(m)

            recovery_count = len(m.recovery_events)
            print(
                f"  [ep={ep}] solved={m.solved} "
                f"recovery_events={recovery_count} "
                f"mean_recovery_tokens={m.mean_recovery_tokens:.1f}"
            )

            # Print analytical chunking numbers for this episode
            for chunk_size in CHUNK_SIZES:
                chunk_lat = m.mean_chunking_recovery_tokens(chunk_size)
                print(f"    [chunk={chunk_size}] mean_recovery_tokens={chunk_lat:.1f}")

        results[scenario.name] = metrics

    _print_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld streaming eval")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes per scenario (default: 5)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Path to GGUF model file (default: qwen2.5-1.5b repo-root path)",
    )
    args = parser.parse_args()
    main(n_episodes=args.n_episodes, model_id=args.model_id)
