# run_slsm_gemini2.5-flash-lite_controlled_by_gemini2.5-flash-lite_always_parallel.py
import json
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio

from src.data_loader import DataLoader
from src.models.gemini import GeminiModel
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
OUTPUT_FILE = "data/final_model_responses/gemini-2.5-flash-lite_slsm-gemini-2.5-flash-lite_always.jsonl"

UNDERLYING_MODEL = "gemini-2.5-flash-lite"
CONTROLLER_MODEL = "gemini-2.5-flash-lite" 

# -------------------------
# Env sanity (keys)
# -------------------------
# GeminiModel reads GOOGLE_API_KEY by default; many setups use GEMINI_API_KEY.
if os.getenv("GOOGLE_API_KEY") is None and os.getenv("GEMINI_API_KEY") is not None:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) in environment.")

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
controller_llm = GeminiModel(
    model=CONTROLLER_MODEL,
    temp=0,
)

cfg = SLSMConfig(
    inject="always",   # IMPORTANT: use on_risk for fair comparison
    disable_controller=False,
    note_max_items=6,
)

controller = SLSMController(controller_llm, cfg)
wrapper = SLSMWrapper(controller, cfg)

# NOTE:
# Some SDK clients are not guaranteed thread-safe. If you see flaky errors,
# set SHARE_UNDERLYING_LLM = False to create one GeminiModel per task.
SHARE_UNDERLYING_LLM = True

underlying_llm_shared = None
if SHARE_UNDERLYING_LLM:
    underlying_llm_shared = GeminiModel(
        model=UNDERLYING_MODEL,
        temp=0,
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
                underlying_llm = GeminiModel(model=UNDERLYING_MODEL, temp=0)

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
