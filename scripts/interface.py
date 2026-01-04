#!/usr/bin/env python3
"""
SLSM Multi-Challenge Command Line Interface

Usage:
    python scripts/interface.py run --config scripts/config.yaml
    python scripts/interface.py run --num-samples 10
    python scripts/interface.py eval --responses data/final_model_responses/xxx.jsonl
    python scripts/interface.py test --index 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
        config_path = PROJECT_ROOT / "scripts" / "config.yaml"
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


def get_model_provider(provider: str, model_name: str, temperature: float):
    """Create model provider instance based on configuration."""
    if provider == "openai":
        from src.models.openai import OpenAIModel
        return OpenAIModel(model=model_name, temp=temperature)
    elif provider == "gemini":
        from src.models.gemini import GeminiModel
        return GeminiModel(model=model_name, temp=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def build_output_filename(underlying_model: str, controller_model: str, enable_slsm: bool, tag: str = None) -> str:
    """Build output filename based on model names and optional tag."""
    if enable_slsm:
        base = f"{underlying_model}_slsm-{controller_model}"
    else:
        base = f"{underlying_model}_baseline"

    if tag:
        base = f"{base}_{tag}"

    return f"{base}.jsonl"


def save_experiment_config(output_file: str, config: Dict[str, Any]):
    """Save experiment configuration to a .txt file alongside the output."""
    from datetime import datetime

    config_file = output_file.replace(".jsonl", ".txt")

    with open(config_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("EXPERIMENT CONFIGURATION\n")
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
    """Run SLSM benchmark."""
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

    # Model configuration (CLI args override yaml config)
    underlying_cfg = models_cfg.get("underlying", {})
    controller_cfg = models_cfg.get("controller", {})

    underlying_provider = args.underlying_provider or underlying_cfg.get("provider", "openai")
    underlying_model = args.underlying_model or underlying_cfg.get("name", "gpt-4o-2024-08-06")
    underlying_temp = args.underlying_temp if args.underlying_temp is not None else underlying_cfg.get("temperature", 0.0)

    controller_provider = args.controller_provider or controller_cfg.get("provider", "openai")
    controller_model = args.controller_model or controller_cfg.get("name", "gpt-4o-mini")
    controller_temp = args.controller_temp if args.controller_temp is not None else controller_cfg.get("temperature", 0.0)

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
    print(f"Underlying model: {underlying_provider}/{underlying_model}")
    underlying_llm = get_model_provider(underlying_provider, underlying_model, underlying_temp)

    # Setup SLSM wrapper if enabled
    wrapper = None
    if enable_slsm:
        print(f"Controller model: {controller_provider}/{controller_model}")
        controller_llm = get_model_provider(controller_provider, controller_model, controller_temp)

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
            "provider": underlying_provider,
            "name": underlying_model,
            "temperature": underlying_temp,
        },
        "controller_model": {
            "provider": controller_provider,
            "name": controller_model,
            "temperature": controller_temp,
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
        },
    }
    config_file = save_experiment_config(output_file, experiment_config)
    print(f"Experiment config saved to: {config_file}")

    # Run benchmark
    desc = "Running SLSM" if enable_slsm else "Running baseline"
    with open(output_file, "w", encoding="utf-8") as fout:
        for conv in tqdm(conversations, desc=desc):
            messages = conv.conversation
            qid = conv.question_id

            try:
                if wrapper:
                    response = wrapper.generate_last_turn(
                        underlying_llm=underlying_llm,
                        original_conversation=messages,
                    )
                else:
                    response = underlying_llm.generate(messages)
            except Exception as e:
                response = f"[ERROR] {str(e)}"

            # Build model name
            if enable_slsm:
                model_name = f"{underlying_model}+SLSM({controller_model})"
            else:
                model_name = underlying_model

            record = {
                "question_id": qid,
                "model": model_name,
                "response": response,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Results saved to: {output_file}")


def cmd_eval(args, cfg: Dict[str, Any]):
    """Run judge evaluation on responses."""
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

    os.makedirs(output_dir, exist_ok=True)

    # Run evaluation via subprocess
    import subprocess
    cmd = [
        sys.executable, "-m", "run_judge_eval",
        "--responses", responses_file,
        "--out_json", out_json,
        "--out_csv", out_csv,
        "--workers", str(workers),
        "--attempts", str(attempts),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print(f"\nEvaluation complete.")
        print(f"  JSON: {out_json}")
        print(f"  CSV:  {out_csv}")
    else:
        print(f"\nEvaluation failed with code {result.returncode}")


def cmd_test(args, cfg: Dict[str, Any]):
    """Test SLSM on a single conversation."""
    from src.data_loader import DataLoader
    from src.slsm_wrapper import SLSMConfig, SLSMController, SLSMWrapper

    paths_cfg = cfg.get("paths", {})
    models_cfg = cfg.get("models", {})
    slsm_cfg = cfg.get("slsm", {})

    benchmark_file = paths_cfg.get("benchmark_file", "data/benchmark_questions.jsonl")

    # Load benchmark
    dl = DataLoader(input_file=benchmark_file)
    dl.load_data()
    conversations = dl.get_conversations()

    # Get conversation
    index = args.index
    if index >= len(conversations):
        print(f"ERROR: Index {index} out of range (max {len(conversations) - 1})")
        return

    conv = conversations[index]
    messages = conv.conversation

    print("=" * 60)
    print(f"Conversation {index} (ID: {conv.question_id})")
    print(f"Axis: {conv.axis}")
    print("=" * 60)

    # Print conversation
    for i, m in enumerate(messages):
        role = m.get("role", "unknown")
        content = m.get("content", "")
        preview = content if len(content) <= 300 else content[:300] + "..."
        print(f"\n[Turn {i}] {role.upper()}:")
        print(preview)

    print("\n" + "=" * 60)

    # Get model configuration
    underlying_cfg = models_cfg.get("underlying", {})
    controller_cfg = models_cfg.get("controller", {})

    underlying_llm = get_model_provider(
        underlying_cfg.get("provider", "openai"),
        underlying_cfg.get("name", "gpt-4o-2024-08-06"),
        underlying_cfg.get("temperature", 0.0),
    )

    controller_llm = get_model_provider(
        controller_cfg.get("provider", "openai"),
        controller_cfg.get("name", "gpt-4o-mini"),
        controller_cfg.get("temperature", 0.0),
    )

    # Create SLSM wrapper
    slsm_config = SLSMConfig(
        inject="always",  # Force injection for testing
        note_max_items=slsm_cfg.get("note_max_items", 6),
    )

    controller = SLSMController(controller_llm, slsm_config)
    wrapper = SLSMWrapper(controller, slsm_config)

    # Generate responses
    print("\nGenerating baseline response...")
    baseline_resp = underlying_llm.generate(messages)

    print("Generating SLSM response...")
    slsm_resp = wrapper.generate_last_turn(
        underlying_llm=underlying_llm,
        original_conversation=messages,
    )

    # Inspect state
    state = wrapper.track_state(messages)

    print("\n" + "=" * 60)
    print("BASELINE RESPONSE:")
    print("=" * 60)
    print(baseline_resp[:800] if len(baseline_resp) > 800 else baseline_resp)

    print("\n" + "=" * 60)
    print("SLSM RESPONSE:")
    print("=" * 60)
    print(slsm_resp[:800] if len(slsm_resp) > 800 else slsm_resp)

    print("\n" + "=" * 60)
    print("SLSM STATE:")
    print("=" * 60)
    print(f"Facts: {len(state.facts)}")
    for f in state.facts[:3]:
        print(f"  - {f.get('text', '')[:100]}")
    print(f"Constraints: {len(state.constraints)}")
    for c in state.constraints[:3]:
        print(f"  - [{c.get('status', '')}] {c.get('text', '')[:100]}")
    print(f"Plan mode: {state.plan.get('mode', 'unknown')}")

    print("\n" + "=" * 60)
    print("MEMORY NOTE (injected):")
    print("=" * 60)
    note = state.to_compact_note()
    print(note if note else "(empty)")


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
        description="SLSM Multi-Challenge CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config YAML file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run SLSM benchmark")
    run_parser.add_argument("--benchmark", "-b", type=str, help="Benchmark file path")
    run_parser.add_argument("--output", "-o", type=str, help="Output file path")
    run_parser.add_argument("--output-dir", type=str, help="Output directory")
    run_parser.add_argument("--num-samples", "-n", type=int, help="Number of samples to process")
    run_parser.add_argument("--underlying-provider", type=str, help="Underlying model provider")
    run_parser.add_argument("--underlying-model", type=str, help="Underlying model name")
    run_parser.add_argument("--underlying-temp", type=float, help="Underlying model temperature")
    run_parser.add_argument("--controller-provider", type=str, help="Controller model provider")
    run_parser.add_argument("--controller-model", type=str, help="Controller model name")
    run_parser.add_argument("--controller-temp", type=float, help="Controller model temperature")
    run_parser.add_argument("--enable-slsm", type=lambda x: x.lower() == "true", help="Enable SLSM wrapper")
    run_parser.add_argument("--tag", "-t", type=str, help="Experiment tag for output filename (e.g., exp1, test, v2)")
    run_parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing output")

    # --- eval command ---
    eval_parser = subparsers.add_parser("eval", help="Run judge evaluation")
    eval_parser.add_argument("--responses", "-r", type=str, required=True, help="Response JSONL file")
    eval_parser.add_argument("--output-dir", type=str, help="Output directory")
    eval_parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    eval_parser.add_argument("--attempts", "-a", type=int, help="Number of judge attempts")

    # --- test command ---
    test_parser = subparsers.add_parser("test", help="Test SLSM on single conversation")
    test_parser.add_argument("--index", "-i", type=int, default=0, help="Conversation index")

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
    elif args.command == "eval":
        cmd_eval(args, cfg)
    elif args.command == "test":
        cmd_test(args, cfg)
    elif args.command == "compare":
        cmd_compare(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
