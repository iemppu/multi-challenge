import json
import os
from tqdm import tqdm

from src.data_loader import DataLoader
from src.models.openai import OpenAIModel
from src.slsm_wrapper import (
    SLSMConfig,
    SLSMController,
    SLSMWrapper,
)

# =========================
# Config
# =========================
BENCHMARK_FILE = "data/benchmark_questions.jsonl"
OUTPUT_FILE = "data/final_model_responses/gpt-4o-2024-08-06_slsm-gpt-4o-mini.jsonl"

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
    inject="on_risk",      # IMPORTANT: use on_risk for fair comparison
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
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for conv in tqdm(conversations, desc="Running SLSM-controlled GPT-4o")[:50]:
        messages = conv.conversation
        qid = conv.question_id

        try:
            response = wrapper.generate_last_turn(
                underlying_llm=underlying_llm,
                original_conversation=messages,
            )
        except Exception as e:
            # Fail-safe: record error as response text
            response = f"[ERROR] {str(e)}"

        record = {
            "question_id": qid,
            "model": f"{UNDERLYING_MODEL}+SLSM({CONTROLLER_MODEL})",
            "response": response,
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nDone. Results saved to:\n  {OUTPUT_FILE}")
