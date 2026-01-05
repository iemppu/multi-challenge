import json
import os
import asyncio
from tqdm.asyncio import tqdm_asyncio

from src.data_loader import DataLoader
from src.models.openai import OpenAIModel
from src.slsm_wrapper import (
    SLSMConfig,
    SLSMController,
    SLSMWrapper,
)

CONCURRENCY = 4

# =========================
# Config
# =========================
BENCHMARK_FILE = "data/benchmark_questions.jsonl"
OUTPUT_FILE = "data/final_model_responses/gpt-4o-2024-08-06_slsm-gpt-4o-mini_on_detection.jsonl"

UNDERLYING_MODEL = "gpt-4o-2024-08-06"
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
controller_llm = OpenAIModel(
    model=CONTROLLER_MODEL,
    temp=0,
)

cfg = SLSMConfig(
    inject="on_detection",      # IMPORTANT: use on_risk for fair comparison
    note_max_items=6,
)

controller = SLSMController(controller_llm, cfg)
wrapper = SLSMWrapper(controller, cfg)

underlying_llm = OpenAIModel(
    model=UNDERLYING_MODEL,
    temp=0,
)

# =========================
# Run benchmark
# =========================
async def run_one(conv, wrapper, underlying_llm, sem):
    messages = conv.conversation
    qid = conv.question_id

    async with sem:
        try:
            # 如果 wrapper.generate_last_turn 是 async，直接 await
            # response = await wrapper.generate_last_turn(
            #     underlying_llm=underlying_llm,
            #     original_conversation=messages,
            # )

            # 如果它是同步函数（最常见），用 to_thread 丢到线程池跑
            response = await asyncio.to_thread(
                wrapper.generate_last_turn,
                underlying_llm=underlying_llm,
                original_conversation=messages,
            )

        except Exception as e:
            response = f"[ERROR] {str(e)}"

    return {
        "question_id": qid,
        "model": f"{UNDERLYING_MODEL}+SLSM({CONTROLLER_MODEL})",
        "response": response,
    }

async def main(conversations, wrapper, underlying_llm, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [run_one(conv, wrapper, underlying_llm, sem) for conv in conversations]

    # tqdm_asyncio.gather 会显示进度条
    results = await tqdm_asyncio.gather(*tasks, desc="Running SLSM-controlled")

    # 写文件（串行）
    with open(output_file, "w", encoding="utf-8") as fout:
        for rec in results:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main(conversations, wrapper, underlying_llm, OUTPUT_FILE))
    print(f"\nDone. Results saved to:\n  {OUTPUT_FILE}")
