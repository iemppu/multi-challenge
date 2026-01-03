# %cd /content/multi-challenge
# !cat > run_judge_eval.py << 'PY'
import os, json, argparse
from collections import defaultdict

from typing import Dict, List

from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.result_parser import ResultParser

# def load_responses_jsonl(path: str):
#     """
#     Loads a responses jsonl file like data/final_model_responses/*_responses.jsonl
#     and returns: Dict[question_id -> List[str]]
#     Tries common key patterns robustly.
#     """
#     responses = defaultdict(list)
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             obj = json.loads(line)

#             # Common patterns: question_id or id
#             qid = obj.get("question_id", obj.get("id", obj.get("qid")))
#             if qid is None:
#                 raise ValueError(f"Cannot find question id key in: {obj.keys()}")

#             # Common patterns: list of attempts or single response
#             if "responses" in obj and isinstance(obj["responses"], list):
#                 for r in obj["responses"]:
#                     if isinstance(r, str):
#                         responses[qid].append(r)
#                     elif isinstance(r, dict):
#                         # sometimes {"content": "..."} / {"response": "..."}
#                         responses[qid].append(r.get("content", r.get("response", "")))
#             elif "response" in obj:
#                 responses[qid].append(obj["response"])
#             elif "output" in obj:
#                 responses[qid].append(obj["output"])
#             else:
#                 # fall back: try 'attempts'
#                 atts = obj.get("attempts")
#                 if isinstance(atts, list):
#                     for r in atts:
#                         responses[qid].append(r if isinstance(r, str) else r.get("content",""))
#                 else:
#                     raise ValueError(f"Cannot find responses in: {obj.keys()}")
#     return responses

def load_responses_jsonl(path: str) -> Dict[str, List[str]]:
    responses: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["QUESTION_ID"]
            resp = obj["RESPONSE"]
            # 确保是 List[str]
            if isinstance(resp, str):
                resp = [resp]
            responses[qid] = resp
    return responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/benchmark_questions.jsonl")
    ap.add_argument("--responses", required=True, help="path to *_responses.jsonl")
    ap.add_argument("--out_json", required=True, help="save raw evaluation results json")
    ap.add_argument("--out_csv", required=True, help="save raw detailed csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=3, help="attempts per question (for CSV formatting)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
        
    dl = DataLoader(input_file=args.questions)
    dl.load_data()
    conversations = dl.get_conversations()

    responses = load_responses_jsonl(args.responses)

    ev = Evaluator(conversations=conversations, responses=responses)
    results = ev.evaluate(max_workers=args.workers)

    # save raw json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # compute scores
    rp = ResultParser(results)
    scores = rp.calculate_scores()
    print("\n=== SCORES ===")
    print(json.dumps(scores, indent=2))

    # save detailed csv
    rp.save_raw_output(args.out_csv, conversations, responses, attempts=args.attempts)
    print(f"\nSaved: {args.out_json}\nSaved: {args.out_csv}")

if __name__ == "__main__":
    main()
