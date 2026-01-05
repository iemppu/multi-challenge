# run_slsm_haiku-4.5_controlled_by_haiku-4.5_always_parallel.py
import json
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio

from src.data_loader import DataLoader
from src.models.claude import ClaudeModel
from src.models.openai import OpenAIModel

from src.slsm_wrapper import (
    SLSMConfig,
    SLSMController,
    SLSMWrapper,
)

# =========================
# Config
# =========================
CONCURRENCY = 8

BENCHMARK_FILE = "data/benchmark_questions.jsonl"
OUTPUT_FILE = "data/final_model_responses/haiku-4.5_slsm-haiku-4.5_always.jsonl"

UNDERLYING_MODEL = "claude-haiku-4-5-20251001"
CONTROLLER_MODEL = "claude-haiku-4-5-20251001" 

# -------------------------
# Env sanity (keys)
# -------------------------
if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError("Missing ANTHROPIC_API_KEY in environment.")

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
controller_llm = ClaudeModel(
    model=CONTROLLER_MODEL,
    temp=0,
    max_tokens=2048,
)

cfg = SLSMConfig(
    inject="always",  
    disable_controller=False,
    note_max_items=6,
)

controller = SLSMController(controller_llm, cfg)
wrapper = SLSMWrapper(controller, cfg)

# NOTE:
# Some SDK clients are not guaranteed thread-safe. If you see flaky errors,
# set SHARE_UNDERLYING_LLM = False to create one ClaudeModel per task.
SHARE_UNDERLYING_LLM = True

underlying_llm_shared = None
if SHARE_UNDERLYING_LLM:
    underlying_llm_shared = ClaudeModel(
        model=UNDERLYING_MODEL,
        temp=0,
        max_tokens=2048,
    )

# =========================
# Run benchmark (parallel)
# =========================
async def run_one(conv, wrapper: SLSMWrapper, sem: asyncio.Semaphore):
    messages = conv.conversation
    qid = conv.question_id

    async with sem:
        try:
            # Create per-task underlying LLM if desired (avoids thread-safety issues)
            underlying_llm = underlying_llm_shared
            if underlying_llm is None:
                underlying_llm = ClaudeModel(model=UNDERLYING_MODEL, temp=0, max_tokens=2048)

            # wrapper.generate_last_turn is most likely sync -> run in threadpool
            response = await asyncio.to_thread(
                wrapper.generate_last_turn,
                underlying_llm=underlying_llm,
                original_conversation=messages,
            )

        except Exception as e:
            response = f"[ERROR] {type(e).__name__}: {str(e)}"

    return {
        "question_id": qid,
        "model": f"{UNDERLYING_MODEL}+SLSM({CONTROLLER_MODEL})",
        "response": response,
    }

async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [run_one(conv, wrapper, sem) for conv in conversations]

    results = await tqdm_asyncio.gather(
        *tasks,
        desc=f"Running SLSM-controlled {UNDERLYING_MODEL}",
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for rec in results:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
    print(f"\nDone. Results saved to:\n  {OUTPUT_FILE}")
