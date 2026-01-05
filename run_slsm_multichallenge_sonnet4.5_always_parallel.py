import json
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio

from src.data_loader import DataLoader
from src.models.openai import OpenAIModel
from src.models.claude import ClaudeModel
from src.slsm_wrapper import (
    SLSMConfig,
    SLSMController,
    SLSMWrapper,
)

# =========================
# Parallel config
# =========================
CONCURRENCY = 8

# =========================
# Config
# =========================
BENCHMARK_FILE = "data/benchmark_questions.jsonl"
OUTPUT_FILE = "data/final_model_responses/sonnet4.5_slsm-gpt-4o-mini_always_inject.jsonl"

UNDERLYING_MODEL = "claude-sonnet-4-5-20250929"
CONTROLLER_MODEL = "gpt-4o-mini"

# Safety: do not overwrite unless you want to
if os.path.exists(OUTPUT_FILE):
    raise RuntimeError(f"Output file already exists: {OUTPUT_FILE}")

# =========================
# Load benchmark
# =========================
dl = DataLoader(input_file=BENCHMARK_FILE)
dl.load_data()
conversations = dl.get_conversations()

print(f"Loaded {len(conversations)} conversations")

# =========================
# Build SLSM components
# =========================
# Controller: OpenAI
controller_llm = OpenAIModel(
    model=CONTROLLER_MODEL,
    temp=0,
)

cfg = SLSMConfig(
    # inject="on_risk",   # IMPORTANT: use on_risk for fair comparison
    inject="always",      # sanity mode
    note_max_items=6,
)

controller = SLSMController(controller_llm, cfg)
wrapper = SLSMWrapper(controller, cfg)

# Underlying: Claude
underlying_llm = ClaudeModel(
    model=UNDERLYING_MODEL,
    temp=0,
    max_tokens=2048,
)

# =========================
# Run benchmark (parallel)
# =========================
async def run_one(conv, wrapper, underlying_llm, sem):
    messages = conv.conversation
    qid = conv.question_id

    async with sem:
        try:
            # wrapper.generate_last_turn is typically sync -> run in a thread
            response = await asyncio.to_thread(
                wrapper.generate_last_turn,
                underlying_llm=underlying_llm,
                original_conversation=messages,
            )
        except Exception as e:
            response = f"[ERROR] {type(e).__name__}: {e}"

    return {
        "question_id": qid,
        "model": f"{UNDERLYING_MODEL}+SLSM({CONTROLLER_MODEL})",
        "response": response,
    }

async def main(conversations, wrapper, underlying_llm, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [run_one(conv, wrapper, underlying_llm, sem) for conv in conversations]

    results = await tqdm_asyncio.gather(*tasks, desc="Running SLSM-controlled Claude")

    with open(output_file, "w", encoding="utf-8") as fout:
        for rec in results:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Safety checks for keys
    assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in env"
    assert os.getenv("ANTHROPIC_API_KEY"), "Missing ANTHROPIC_API_KEY in env"

    asyncio.run(main(conversations, wrapper, underlying_llm, OUTPUT_FILE))
    print(f"\nDone. Results saved to:\n  {OUTPUT_FILE}")
