#!/usr/bin/env python3
"""
SLSM Multi-Challenge CLI - OpenRouter Provider

Mirrors interface.py but uses OpenRouter as the provider.
Reads configuration from config_openrouter.yaml.

Usage:
    python scripts/interface_openrouter.py run --config scripts/config_openrouter.yaml
    python scripts/interface_openrouter.py run --num-samples 10 --parallel true
    python scripts/interface_openrouter.py test --model openai/gpt-4o-mini
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm


def setup_project_path():
    """Add project root to sys.path."""
    project_root = Path(__file__).parent.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    return project_root


PROJECT_ROOT = setup_project_path()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "scripts" / "config_openrouter.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"WARNING: Config file not found: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env():
    """Load environment variables from .env file."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
    else:
        print(f"WARNING: .env file not found at {env_path}")


def get_openrouter_model(
    model_name: str,
    temperature: float,
    seed: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """Create OpenRouter model instance."""
    from src.models.openrouter import OpenRouterModel
    return OpenRouterModel(model=model_name, temp=temperature, seed=seed, top_p=top_p)


def build_output_filename(underlying_model: str, controller_model: str, enable_slsm: bool, tag: str = None) -> str:
    """Build output filename based on model names."""
    # Sanitize model names for filename (replace / with -)
    underlying_safe = underlying_model.replace("/", "-")
    controller_safe = controller_model.replace("/", "-")

    if enable_slsm:
        base = f"{underlying_safe}_slsm-{controller_safe}"
    else:
        base = f"{underlying_safe}_baseline"

    if tag:
        base = f"{base}_{tag}"

    return f"{base}.jsonl"


def save_experiment_config(output_file: str, config: Dict[str, Any]):
    """Save experiment configuration to a .txt file alongside the output."""
    from datetime import datetime

    config_file = output_file.replace(".jsonl", ".txt")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)

    with open(config_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT CONFIGURATION (OpenRouter)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output file: {output_file}\n")
        f.write("\n")

        for section, values in config.items():
            f.write(f"[{section}]\n")
            if isinstance(values, dict):
                for k, v in values.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {values}\n")
            f.write("\n")

    return config_file


def cmd_run(args, cfg: Dict[str, Any]):
    """Run SLSM benchmark with OpenRouter."""
    from src.data_loader import DataLoader
    from src.slsm_wrapper import SLSMConfig, SLSMController, SLSMWrapper

    # Get configuration with CLI overrides
    paths_cfg = cfg.get("paths", {})
    models_cfg = cfg.get("models", {})
    slsm_cfg = cfg.get("slsm", {})
    run_cfg = cfg.get("run", {})

    benchmark_file = args.benchmark or paths_cfg.get("benchmark_file", "data/benchmark_questions.jsonl")
    output_dir = args.output_dir or paths_cfg.get("output_dir", "data/final_model_responses")
    num_samples = args.num_samples if args.num_samples is not None else run_cfg.get("num_samples")
    enable_slsm = args.enable_slsm if args.enable_slsm is not None else run_cfg.get("enable_slsm", True)
    skip_existing = run_cfg.get("skip_existing", True)

    # Parallel configuration
    parallel = args.parallel if args.parallel is not None else run_cfg.get("parallel", False)
    num_workers = args.num_workers if args.num_workers is not None else run_cfg.get("num_workers", 3)

    # Model configuration (CLI args override yaml config)
    underlying_cfg = models_cfg.get("underlying", {})
    controller_cfg = models_cfg.get("controller", {})

    underlying_model = args.underlying_model or underlying_cfg.get("name", "openai/gpt-4o-2024-08-06")
    underlying_temp = args.underlying_temp if args.underlying_temp is not None else underlying_cfg.get("temperature", 0.0)
    underlying_seed = args.seed if args.seed is not None else underlying_cfg.get("seed")
    underlying_top_p = args.top_p if args.top_p is not None else underlying_cfg.get("top_p")

    controller_model = args.controller_model or controller_cfg.get("name", "openai/gpt-4o-mini")
    controller_temp = args.controller_temp if args.controller_temp is not None else controller_cfg.get("temperature", 0.0)
    controller_seed = controller_cfg.get("seed")  # Controller usually doesn't need seed
    controller_top_p = controller_cfg.get("top_p")

    # Build output path
    tag = args.tag or run_cfg.get("tag")
    output_filename = build_output_filename(underlying_model, controller_model, enable_slsm, tag)
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(output_dir, output_filename)

    # Check existing
    if skip_existing and os.path.exists(output_file) and not args.force:
        print(f"Output file already exists: {output_file}")
        print("Use --force to overwrite.")
        return

    # Load benchmark
    print(f"Loading benchmark from: {benchmark_file}")
    dl = DataLoader(input_file=benchmark_file)
    dl.load_data()
    conversations = dl.get_conversations()
    print(f"Loaded {len(conversations)} conversations")

    # Slice if requested
    if num_samples is not None:
        conversations = conversations[:num_samples]
        print(f"Processing first {num_samples} samples")

    # Create models
    print(f"Underlying model (OpenRouter): {underlying_model}")
    if underlying_seed is not None:
        print(f"  seed={underlying_seed}, top_p={underlying_top_p}")
    underlying_llm = get_openrouter_model(
        underlying_model, underlying_temp,
        seed=underlying_seed, top_p=underlying_top_p
    )

    # Setup SLSM wrapper if enabled
    wrapper = None
    if enable_slsm:
        print(f"Controller model (OpenRouter): {controller_model}")
        controller_llm = get_openrouter_model(
            controller_model, controller_temp,
            seed=controller_seed, top_p=controller_top_p
        )

        slsm_config = SLSMConfig(
            inject=slsm_cfg.get("inject", "on_risk"),
            risk_modes=tuple(slsm_cfg.get("risk_modes", ["verify", "clarify"])),
            note_max_items=slsm_cfg.get("note_max_items", 6),
            controller_max_tokens=slsm_cfg.get("controller_max_tokens", 1200),
            gate_facts_by_evidence=slsm_cfg.get("gate_facts_by_evidence", True),
        )

        controller = SLSMController(controller_llm, slsm_config)
        wrapper = SLSMWrapper(controller, slsm_config)
        print(f"SLSM enabled with inject={slsm_config.inject}")
    else:
        print("SLSM disabled (baseline mode)")

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save experiment configuration
    experiment_config = {
        "paths": {
            "benchmark_file": benchmark_file,
            "output_file": output_file,
        },
        "underlying_model": {
            "provider": "openrouter",
            "name": underlying_model,
            "temperature": underlying_temp,
            "seed": underlying_seed,
            "top_p": underlying_top_p,
        },
        "controller_model": {
            "provider": "openrouter",
            "name": controller_model,
            "temperature": controller_temp,
            "seed": controller_seed,
            "top_p": controller_top_p,
        },
        "slsm": {
            "enabled": enable_slsm,
            "inject": slsm_cfg.get("inject", "on_risk") if enable_slsm else "N/A",
            "risk_modes": slsm_cfg.get("risk_modes", ["verify", "clarify"]) if enable_slsm else "N/A",
            "note_max_items": slsm_cfg.get("note_max_items", 6) if enable_slsm else "N/A",
            "gate_facts_by_evidence": slsm_cfg.get("gate_facts_by_evidence", True) if enable_slsm else "N/A",
        },
        "run": {
            "num_samples": num_samples if num_samples else "all",
            "total_conversations": len(conversations),
            "tag": tag or "none",
            "parallel": parallel,
            "num_workers": num_workers if parallel else "N/A",
        },
    }
    config_file = save_experiment_config(output_file, experiment_config)
    print(f"Experiment config saved to: {config_file}")
    if parallel:
        print(f"Parallel mode: {num_workers} workers")

    # Build model name
    if enable_slsm:
        model_name = f"{underlying_model}+SLSM({controller_model})"
    else:
        model_name = underlying_model

    # Worker function for processing a single conversation
    def process_conversation(conv) -> Tuple[str, str]:
        """Process a single conversation, returns (question_id, response)."""
        messages = conv.conversation
        qid = conv.question_id

        try:
            if enable_slsm:
                # Create per-thread SLSM wrapper to avoid state conflicts
                thread_controller_llm = get_openrouter_model(
                    controller_model, controller_temp,
                    seed=controller_seed, top_p=controller_top_p
                )
                thread_slsm_config = SLSMConfig(
                    inject=slsm_cfg.get("inject", "on_risk"),
                    risk_modes=tuple(slsm_cfg.get("risk_modes", ["verify", "clarify"])),
                    note_max_items=slsm_cfg.get("note_max_items", 6),
                    controller_max_tokens=slsm_cfg.get("controller_max_tokens", 1200),
                    gate_facts_by_evidence=slsm_cfg.get("gate_facts_by_evidence", True),
                )
                thread_controller = SLSMController(thread_controller_llm, thread_slsm_config)
                thread_wrapper = SLSMWrapper(thread_controller, thread_slsm_config)
                response = thread_wrapper.generate_last_turn(
                    underlying_llm=underlying_llm,
                    original_conversation=messages,
                )
            else:
                response = underlying_llm.generate(messages)
        except Exception as e:
            response = f"[ERROR] {str(e)}"

        return qid, response

    # Run benchmark
    desc = "Running SLSM (OpenRouter)" if enable_slsm else "Running baseline (OpenRouter)"
    results = []

    if parallel and num_workers > 1:
        # Parallel execution
        print(f"Starting parallel execution with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_conversation, conv): conv.question_id for conv in conversations}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                qid = futures[future]
                try:
                    result_qid, response = future.result()
                    results.append((result_qid, response))
                except Exception as e:
                    results.append((qid, f"[ERROR] {str(e)}"))
    else:
        # Sequential execution (original behavior)
        for conv in tqdm(conversations, desc=desc):
            qid, response = process_conversation(conv)
            results.append((qid, response))

    # Write results to file (preserving order by question_id from conversations)
    qid_to_response = {qid: resp for qid, resp in results}
    with open(output_file, "w", encoding="utf-8") as fout:
        for conv in conversations:
            qid = conv.question_id
            response = qid_to_response.get(qid, "[ERROR] Missing result")
            record = {
                "question_id": qid,
                "model": model_name,
                "response": response,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Results saved to: {output_file}")


def cmd_test(args):
    """Test OpenRouter connection with a simple prompt."""
    from src.models.openrouter import OpenRouterModel

    model_name = args.model or "openai/gpt-4o-mini"
    seed = getattr(args, 'seed', None)
    top_p = getattr(args, 'top_p', None)
    print(f"Testing OpenRouter with model: {model_name}")
    if seed is not None:
        print(f"  seed={seed}, top_p={top_p}")

    try:
        model = OpenRouterModel(model=model_name, temp=0.0, seed=seed, top_p=top_p)
        response = model.generate("Say 'Hello from OpenRouter!' in exactly 5 words.")
        print(f"Success! Response: {response}")
        print(f"Model info: {model.get_model_info()}")
    except Exception as e:
        print(f"Failed: {e}")


def cmd_eval(args, cfg: Dict[str, Any]):
    """Run judge evaluation on responses."""
    import subprocess

    eval_cfg = cfg.get("evaluation", {})
    paths_cfg = cfg.get("paths", {})

    responses_file = args.responses
    if not responses_file:
        print("ERROR: --responses is required")
        return

    output_dir = args.output_dir or paths_cfg.get("eval_output_dir", "outputs")
    workers = args.workers if args.workers is not None else eval_cfg.get("workers", 1)
    attempts = args.attempts if args.attempts is not None else eval_cfg.get("attempts", 1)

    # Build output paths
    basename = Path(responses_file).stem
    out_json = os.path.join(output_dir, f"{basename}_judge_results.json")
    out_csv = os.path.join(output_dir, f"{basename}_judge_results.csv")
    out_scores = os.path.join(output_dir, f"{basename}_scores.json")

    os.makedirs(output_dir, exist_ok=True)

    # Check if results already exist
    if os.path.exists(out_json) and os.path.exists(out_csv):
        print(f"Found existing evaluation results:")
        print(f"  JSON: {out_json}")
        print(f"  CSV:  {out_csv}")
        print(f"\nComputing scores from existing results...")

        # Load existing results and compute scores
        from src.result_parser import ResultParser
        with open(out_json, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Filter out invalid results (missing responses) if skip_missing is enabled
        if getattr(args, 'skip_missing', True):
            valid_results = [r for r in results if not r.get("reasoning", "").startswith("NA - Question ID")]
            if len(valid_results) < len(results):
                print(f"Filtered results: {len(results)} -> {len(valid_results)} (removed missing responses)")
        else:
            valid_results = results

        rp = ResultParser(valid_results)
        scores = rp.calculate_scores()

        print("\n=== SCORES ===")
        print(json.dumps(scores, indent=2))

        # Save scores
        with open(out_scores, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

        print(f"\nScores saved to: {out_scores}")
        return

    # Run evaluation via subprocess
    cmd = [
        sys.executable, "-m", "run_judge_eval",
        "--responses", responses_file,
        "--out_json", out_json,
        "--out_csv", out_csv,
        "--workers", str(workers),
        "--attempts", str(attempts),
    ]
    # Add --skip_missing if enabled (default: True)
    if getattr(args, 'skip_missing', True):
        cmd.append("--skip_missing")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"\nEvaluation complete.")
        print(f"  JSON:   {out_json}")
        print(f"  CSV:    {out_csv}")
        print(f"  Scores: {out_scores}")
    else:
        print(f"\nEvaluation failed with code {result.returncode}")


def cmd_compare(args, cfg: Dict[str, Any]):
    """Compare two response files."""
    import pandas as pd
    import numpy as np

    file_a = args.file_a
    file_b = args.file_b

    if not os.path.exists(file_a) or not os.path.exists(file_b):
        print("ERROR: Both files must exist")
        return

    # Load CSVs
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    print(f"File A: {file_a} ({len(df_a)} rows)")
    print(f"File B: {file_b} ({len(df_b)} rows)")

    # Find ID column
    id_col = None
    for c in ["QUESTION_ID", "question_id", "qid", "id"]:
        if c in df_a.columns and c in df_b.columns:
            id_col = c
            break

    if id_col is None:
        print("ERROR: Cannot find common ID column")
        return

    # Merge
    df = df_a.merge(df_b, on=id_col, how="inner", suffixes=("_A", "_B"))
    print(f"Merged: {len(df)} rows")

    # Find numeric pairs
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    pairs = []
    for c in num_cols:
        if c.endswith("_A"):
            c2 = c[:-2] + "_B"
            if c2 in df.columns:
                pairs.append((c, c2))

    if not pairs:
        print("No numeric metric pairs found")
        return

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for a_col, b_col in pairs:
        metric = a_col[:-2]
        x = df[a_col].to_numpy()
        y = df[b_col].to_numpy()

        mean_a = np.mean(x)
        mean_b = np.mean(y)
        delta = mean_b - mean_a
        win = np.mean(y > x)
        tie = np.mean(y == x)
        lose = np.mean(y < x)

        print(f"\n{metric}:")
        print(f"  A mean: {mean_a:.4f}")
        print(f"  B mean: {mean_b:.4f}")
        print(f"  Delta:  {delta:+.4f}")
        print(f"  Win/Tie/Lose: {win*100:.1f}% / {tie*100:.1f}% / {lose*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="SLSM Multi-Challenge CLI - OpenRouter Provider",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models (examples):
  openai/gpt-4o-2024-08-06    GPT-4o
  openai/gpt-4o-mini          GPT-4o Mini (cheap)
  google/gemini-2.0-flash-001 Gemini 2.0 Flash (fast & cheap)
  google/gemini-2.5-flash     Gemini 2.5 Flash
  anthropic/claude-3.5-sonnet Claude 3.5 Sonnet
        """
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config YAML file (default: scripts/config_openrouter.yaml)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run SLSM benchmark")
    run_parser.add_argument("--benchmark", "-b", type=str, help="Benchmark file path")
    run_parser.add_argument("--output", "-o", type=str, help="Output file path")
    run_parser.add_argument("--output-dir", type=str, help="Output directory")
    run_parser.add_argument("--num-samples", "-n", type=int, help="Number of samples to process")
    run_parser.add_argument("--underlying-model", type=str, help="Underlying model (e.g., openai/gpt-4o-2024-08-06)")
    run_parser.add_argument("--underlying-temp", type=float, help="Underlying model temperature")
    run_parser.add_argument("--controller-model", type=str, help="Controller model (e.g., openai/gpt-4o-mini)")
    run_parser.add_argument("--controller-temp", type=float, help="Controller model temperature")
    run_parser.add_argument("--enable-slsm", type=lambda x: x.lower() == "true", help="Enable SLSM wrapper")
    run_parser.add_argument("--tag", "-t", type=str, help="Experiment tag for output filename")
    run_parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing output")
    run_parser.add_argument("--parallel", "-p", type=lambda x: x.lower() == "true", help="Enable parallel processing")
    run_parser.add_argument("--num-workers", "-w", type=int, help="Number of parallel workers")
    run_parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility (e.g., 42)")
    run_parser.add_argument("--top-p", type=float, help="Top-p (nucleus sampling) value (e.g., 1.0)")

    # --- test command ---
    test_parser = subparsers.add_parser("test", help="Test OpenRouter connection")
    test_parser.add_argument("--model", "-m", type=str, help="Model to test")
    test_parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    test_parser.add_argument("--top-p", type=float, help="Top-p (nucleus sampling) value")

    # --- eval command ---
    eval_parser = subparsers.add_parser("eval", help="Run judge evaluation")
    eval_parser.add_argument("--responses", "-r", type=str, required=True, help="Response JSONL file")
    eval_parser.add_argument("--output-dir", type=str, help="Output directory")
    eval_parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    eval_parser.add_argument("--attempts", "-a", type=int, help="Number of judge attempts")
    eval_parser.add_argument("--skip-missing", action="store_true", default=False,
                             help="Skip questions without responses (calculate score over subset)")
    eval_parser.add_argument("--no-skip-missing", dest="skip_missing", action="store_false",
                             help="Count missing responses as failures (default, use full 273 as denominator)")

    # --- compare command ---
    compare_parser = subparsers.add_parser("compare", help="Compare two result CSVs")
    compare_parser.add_argument("file_a", type=str, help="First CSV file")
    compare_parser.add_argument("file_b", type=str, help="Second CSV file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Load environment and config
    load_env()
    cfg = load_config(args.config)

    # Dispatch command
    if args.command == "run":
        cmd_run(args, cfg)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "eval":
        cmd_eval(args, cfg)
    elif args.command == "compare":
        cmd_compare(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
