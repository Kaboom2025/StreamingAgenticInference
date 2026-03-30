"""End-to-end demo: CalcEnv with a real LLM via llama-cpp-python.

Usage:
    python demo_calc.py
    python demo_calc.py --model qwen2.5-1.5b-instruct-q4_k_m.gguf
    python demo_calc.py --expr "17 * 43"
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from llama_cpp import Llama

from streamagent.engine.kv_stream import KVStream, KVStreamConfig
from streamagent.engine.router import Router
from streamagent.env.calc_env import CalcEnv

# ---------------------------------------------------------------------------
# Chat template (chatml — used by Qwen2.5, SmolLM2, Mistral-instruct variants)
# ---------------------------------------------------------------------------

def build_prompt(problem: str) -> str:
    system = (
        "You are a calculator assistant. You MUST use the calc tool to "
        "answer. NEVER compute in your head.\n\n"
        "Tool syntax (self-closing XML tag):\n"
        '<act cmd="eval EXPRESSION"/>\n\n'
        "You will then receive:\n"
        '<obs type="result">ANSWER</obs>\n\n'
        "Then state the final answer."
    )
    # Few-shot example
    example_user = "Compute: 15 + 28"
    example_asst = '<act cmd="eval 15+28"/>'
    user = f"Compute: {problem}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{example_user}<|im_end|>\n"
        f"<|im_start|>assistant\n{example_asst}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(model_path: str, problem: str, max_new_tokens: int = 200) -> None:
    env = CalcEnv(problem)
    env.reset()

    # Grab EOS token id before handing model to KVStream
    _probe = Llama(model_path, n_ctx=16, verbose=False)
    eos_token_id: int = _probe.token_eos()
    del _probe

    cfg = KVStreamConfig(
        model_id=model_path,
        backend="llama",
        window_length=2048,
        sink_tokens=4,
    )
    prompt = build_prompt(problem)
    stream = KVStream(cfg, prompt)

    print(f"Model : {model_path}")
    print(f"Problem: {problem}\n")
    stream.start()
    prefill_pos = stream._position
    print("-" * 60)

    env.register_injector(stream._injector)
    router = Router()
    action_fired = False

    async for tok in stream.run():
        print(tok.text, end="", flush=True)
        out = router.process(tok)

        if out.action:
            action_fired = True
            _, reward, done = env.step(out.action)
            print(f"\n[env] expr={out.action.command!r}  "
                  f"→ {env._last_result}  reward={reward}")
            if done:
                stream.stop()
                break

        # Stop on EOS or generation budget
        if tok.id == eos_token_id:
            stream.stop()
            break
        if (stream._position - prefill_pos) >= max_new_tokens:
            stream.stop()
            break

    print("\n" + "-" * 60)

    if action_fired:
        print(f"\n{env.render()}")
    else:
        print("\nModel did not emit an <act> tag.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CalcEnv end-to-end demo")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    parser.add_argument("--expr", default="23 * 7")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    asyncio.run(run(args.model, args.expr, args.max_new_tokens))


if __name__ == "__main__":
    main()
