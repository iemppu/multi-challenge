# run_slsm_multichallenge_gemini25pro.py
import json
import os
from tqdm import tqdm

from src.data_loader import DataLoader
from src.models.openai import OpenAIModel
from src.models.gemini import GeminiModel
from src.slsm_wrapper import (
    SLSMConfig,
    SLSMController,
    SLSMWrapper,
)

# =========================
# Config
# =========================
BENCHMARK_FILE = "data/benchmark_questions.jsonl"
OUTPUT_FILE = "data/final_model_responses/gemini-2.5-pro_slsm-gpt-4o-mini.jsonl"

UNDERLYING_MODEL = "gemini-2.5-pro"
CONTROLLER_MODEL = "gpt-4o-mini"

# -------------------------
# Env sanity (keys)
# -------------------------
# Your GeminiModel reads GOOGLE_API_KEY by default; many setups use GEMINI_API_KEY.
if os.getenv("GOOGLE_API_KEY") is None and os.getenv("GEMINI_API_KEY") is not None:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")
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

underlying_llm = GeminiModel(
    model=UNDERLYING_MODEL,
    temp=0,
)

# =========================
# Run benchmark
# =========================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for conv in tqdm(conversations[:50], desc=f"Running SLSM-controlled {UNDERLYING_MODEL}"):
        messages = conv.conversation
        qid = conv.question_id

        try:
            response = wrapper.generate_last_turn(
                underlying_llm=underlying_llm,
                original_conversation=messages,
            )
        except Exception as e:
            # Fail-safe: record error as response text
            response = f"[ERROR] {type(e).__name__}: {str(e)}"

        record = {
            "question_id": qid,
            "model": f"{UNDERLYING_MODEL}+SLSM({CONTROLLER_MODEL})",
            "response": response,
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nDone. Results saved to:\n  {OUTPUT_FILE}")
