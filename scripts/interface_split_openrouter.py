#!/usr/bin/env python3
"""
SLSM Multi-Challenge CLI - Split/Incremental Write Version

Writes each response to a separate JSON file for crash-resilient operation.
Cache structure: data/cache/<tag>/<qid>.json or <qid>_empty.json

Usage:
    python scripts/interface_split_openrouter.py --config <config.yaml> run
    python scripts/interface_split_openrouter.py --config <config.yaml> status
    python scripts/interface_split_openrouter.py --config <config.yaml> rerun
    python scripts/interface_split_openrouter.py --config <config.yaml> merge
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Import shared utilities from interface_openrouter.py
from interface_openrouter import (
    setup_project_path,
    load_config,
    load_env,
    get_openrouter_model,
    verify_model_availability,
    build_output_filename,
    PROJECT_ROOT,
)


# =============================================================================
# Cache Management Functions
# =============================================================================

def get_cache_dir(tag: str) -> Path:
    """Get cache directory for a given tag."""
    cache_dir = PROJECT_ROOT / "data" / "cache" / tag
    return cache_dir


def ensure_cache_dir(tag: str) -> Path:
    """Ensure cache directory exists and return path."""
    cache_dir = get_cache_dir(tag)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def write_response_to_cache(
    cache_dir: Path,
    qid: str,
    response: str,
    model_name: str,
) -> Path:
    """
    Write a single response to cache.

    Returns the path to the written file.
    Empty responses are saved as <qid>_empty.json.
    """
    is_empty = not response or not response.strip()

    # Remove old file if exists (either empty or non-empty version)
    for suffix in [".json", "_empty.json"]:
        old_file = cache_dir / f"{qid}{suffix}"
        if old_file.exists():
            old_file.unlink()

    # Determine filename
    if is_empty:
        filename = f"{qid}_empty.json"
    else:
        filename = f"{qid}.json"

    filepath = cache_dir / filename

    record = {
        "question_id": qid,
        "model": model_name,
        "response": response,
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return filepath


def read_cache_responses(cache_dir: Path) -> List[Dict[str, Any]]:
    """Read all responses from cache directory."""
    responses = []

    if not cache_dir.exists():
        return responses

    for filepath in cache_dir.glob("*.json"):
        # Skip config file
        if filepath.name == "_config.json":
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            responses.append(data)

    return responses


def get_cache_status(cache_dir: Path, total_questions: int = 273) -> Dict[str, Any]:
    """
    Get cache status.

    Returns dict with:
        - total: total expected questions
        - completed: non-empty responses
        - empty: empty responses
        - missing: not yet processed
    """
    if not cache_dir.exists():
        return {
            "total": total_questions,
            "completed": 0,
            "empty": 0,
            "missing": total_questions,
        }

    completed = 0
    empty = 0

    for filepath in cache_dir.glob("*.json"):
        if filepath.name == "_config.json":
            continue

        if filepath.name.endswith("_empty.json"):
            empty += 1
        else:
            completed += 1

    return {
        "total": total_questions,
        "completed": completed,
        "empty": empty,
        "missing": total_questions - completed - empty,
    }


def get_completed_qids(cache_dir: Path) -> set:
    """Get set of question IDs that have non-empty responses."""
    completed = set()

    if not cache_dir.exists():
        return completed

    for filepath in cache_dir.glob("*.json"):
        if filepath.name == "_config.json":
            continue
        if filepath.name.endswith("_empty.json"):
            continue

        # Extract qid from filename
        qid = filepath.stem
        completed.add(qid)

    return completed


def get_empty_qids(cache_dir: Path) -> set:
    """Get set of question IDs that have empty responses."""
    empty = set()

    if not cache_dir.exists():
        return empty

    for filepath in cache_dir.glob("*_empty.json"):
        # Extract qid from filename (remove _empty suffix)
        qid = filepath.stem.replace("_empty", "")
        empty.add(qid)

    return empty


def save_config_to_cache(cache_dir: Path, config: Dict[str, Any]):
    """Save experiment config to cache for reference."""
    config_file = cache_dir / "_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# =============================================================================
# Commands
# =============================================================================

def cmd_run(args, cfg: Dict[str, Any]):
    """Run SLSM benchmark with incremental write to cache."""
    from src.data_loader import DataLoader
    from src.slsm_wrapper import SLSMConfig, SLSMController, SLSMWrapper

    # Get configuration
    paths_cfg = cfg.get("paths", {})
    models_cfg = cfg.get("models", {})
    slsm_cfg = cfg.get("slsm", {})
    run_cfg = cfg.get("run", {})

    # Tag is required
    tag = args.tag or run_cfg.get("tag")
    if not tag:
        print("ERROR: tag is required (via --tag or run.tag in config)")
        sys.exit(1)

    benchmark_file = args.benchmark or paths_cfg.get("benchmark_file", "data/benchmark_questions.jsonl")
    num_samples = args.num_samples if args.num_samples is not None else run_cfg.get("num_samples")
    enable_slsm = args.enable_slsm if args.enable_slsm is not None else run_cfg.get("enable_slsm", True)

    # Parallel configuration
    parallel = args.parallel if args.parallel is not None else run_cfg.get("parallel", False)
    num_workers = args.num_workers if args.num_workers is not None else run_cfg.get("num_workers", 2)

    # Model configuration
    underlying_cfg = models_cfg.get("underlying", {})
    controller_cfg = models_cfg.get("controller", {})

    underlying_model = args.underlying_model or underlying_cfg.get("name")
    if not underlying_model:
        print("ERROR: underlying model name is required")
        sys.exit(1)

    underlying_temp = args.underlying_temp if args.underlying_temp is not None else underlying_cfg.get("temperature", 0.0)
    underlying_seed = args.seed if args.seed is not None else underlying_cfg.get("seed")
    underlying_top_p = args.top_p if args.top_p is not None else underlying_cfg.get("top_p")

    controller_model = args.controller_model or controller_cfg.get("name")
    if not controller_model:
        print("ERROR: controller model name is required")
        sys.exit(1)

    controller_temp = args.controller_temp if args.controller_temp is not None else controller_cfg.get("temperature", 0.0)
    controller_seed = controller_cfg.get("seed")
    controller_top_p = controller_cfg.get("top_p")

    # Setup cache directory
    cache_dir = ensure_cache_dir(tag)
    print(f"Cache directory: {cache_dir}")

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

    # Check what's already completed
    completed_qids = get_completed_qids(cache_dir)
    if completed_qids:
        print(f"Found {len(completed_qids)} already completed responses in cache")
        # Filter out completed
        conversations = [c for c in conversations if c.question_id not in completed_qids]
        print(f"Remaining to process: {len(conversations)}")

    if not conversations:
        print("All conversations already processed. Use 'rerun' to retry empty responses.")
        return

    # Create models
    print(f"Underlying model (OpenRouter): {underlying_model}")
    if underlying_seed is not None:
        print(f"  seed={underlying_seed}, top_p={underlying_top_p}")
    underlying_llm = get_openrouter_model(
        underlying_model, underlying_temp,
        seed=underlying_seed, top_p=underlying_top_p
    )

    # Build model name
    if enable_slsm:
        model_name = f"{underlying_model}+SLSM({controller_model})"
        print(f"Controller model (OpenRouter): {controller_model}")
        print(f"SLSM enabled with inject={slsm_cfg.get('inject', 'on_risk')}")
    else:
        model_name = underlying_model
        print("SLSM disabled (baseline mode)")

    # Save config to cache
    experiment_config = {
        "tag": tag,
        "underlying_model": underlying_model,
        "controller_model": controller_model,
        "enable_slsm": enable_slsm,
        "slsm": slsm_cfg,
        "created_at": datetime.now().isoformat(),
    }
    save_config_to_cache(cache_dir, experiment_config)

    # Worker function with immediate write
    def process_and_save(conv) -> Tuple[str, bool]:
        """Process a single conversation and save to cache immediately."""
        messages = conv.conversation
        qid = conv.question_id

        try:
            if enable_slsm:
                thread_controller_llm = get_openrouter_model(
                    controller_model, controller_temp,
                    seed=controller_seed, top_p=controller_top_p
                )
                thread_slsm_config = SLSMConfig(
                    disable_controller=slsm_cfg.get("disable_controller", False),
                    memory_mode=slsm_cfg.get("memory_mode", "structured"),
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

        # Write to cache immediately
        write_response_to_cache(cache_dir, qid, response, model_name)

        is_success = bool(response and response.strip() and not response.startswith("[ERROR]"))
        return qid, is_success

    # Run
    print(f"Parallel mode: {num_workers} workers")
    success_count = 0
    empty_count = 0

    if parallel and num_workers > 1:
        print(f"Starting parallel execution with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_and_save, conv): conv.question_id for conv in conversations}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                qid = futures[future]
                try:
                    _, is_success = future.result()
                    if is_success:
                        success_count += 1
                    else:
                        empty_count += 1
                except Exception as e:
                    empty_count += 1
                    # Still write error to cache
                    write_response_to_cache(cache_dir, qid, f"[ERROR] {str(e)}", model_name)
    else:
        for conv in tqdm(conversations, desc="Processing"):
            _, is_success = process_and_save(conv)
            if is_success:
                success_count += 1
            else:
                empty_count += 1

    # Final status
    status = get_cache_status(cache_dir)
    print(f"\nRun complete:")
    print(f"  This run: {success_count} success, {empty_count} empty/error")
    print(f"  Cache total: {status['completed']} completed, {status['empty']} empty, {status['missing']} missing")

    # Auto merge
    output_dir = paths_cfg.get("output_dir", "data/final_model_responses")
    output_file = os.path.join(output_dir, f"{tag}.jsonl")

    responses = read_cache_responses(cache_dir)
    if responses:
        responses.sort(key=lambda x: x.get("question_id", ""))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for r in responses:
                output_record = {
                    "question_id": r.get("question_id"),
                    "model": r.get("model"),
                    "response": r.get("response"),
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        completed = sum(1 for r in responses if r.get("response") and r.get("response").strip())
        print(f"\nAuto-merged {len(responses)} responses to: {output_file}")
        print(f"  Completed: {completed}, Empty: {len(responses) - completed}")

    if status['empty'] > 0:
        print(f"\nUse 'rerun' to retry {status['empty']} empty responses")


def cmd_status(args, cfg: Dict[str, Any]):
    """Show cache status."""
    run_cfg = cfg.get("run", {})
    tag = args.tag or run_cfg.get("tag")

    if not tag:
        print("ERROR: tag is required (via --tag or run.tag in config)")
        sys.exit(1)

    cache_dir = get_cache_dir(tag)

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    status = get_cache_status(cache_dir)

    print(f"Cache: {cache_dir}")
    print(f"  Total expected: {status['total']}")
    print(f"  Completed: {status['completed']}")
    print(f"  Empty: {status['empty']}")
    print(f"  Missing: {status['missing']}")

    if status['completed'] + status['empty'] > 0:
        completion_rate = status['completed'] / (status['completed'] + status['empty']) * 100
        print(f"  Success rate: {completion_rate:.1f}%")


def cmd_rerun(args, cfg: Dict[str, Any]):
    """Rerun only empty responses."""
    from src.data_loader import DataLoader
    from src.slsm_wrapper import SLSMConfig, SLSMController, SLSMWrapper
    import time

    run_cfg = cfg.get("run", {})
    paths_cfg = cfg.get("paths", {})
    models_cfg = cfg.get("models", {})
    slsm_cfg = cfg.get("slsm", {})

    tag = args.tag or run_cfg.get("tag")
    if not tag:
        print("ERROR: tag is required (via --tag or run.tag in config)")
        sys.exit(1)

    cache_dir = get_cache_dir(tag)
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    # Find empty responses
    empty_qids = get_empty_qids(cache_dir)
    if not empty_qids:
        print("No empty responses found. Nothing to rerun.")
        return

    print(f"Found {len(empty_qids)} empty responses to rerun")

    # Load benchmark
    benchmark_file = args.benchmark or paths_cfg.get("benchmark_file", "data/benchmark_questions.jsonl")
    dl = DataLoader(input_file=benchmark_file)
    dl.load_data()
    all_conversations = dl.get_conversations()

    # Filter to empty qids
    qid_to_conv = {conv.question_id: conv for conv in all_conversations}
    conversations = [qid_to_conv[qid] for qid in empty_qids if qid in qid_to_conv]

    if not conversations:
        print("No matching conversations found in benchmark")
        return

    # Model config
    underlying_cfg = models_cfg.get("underlying", {})
    controller_cfg = models_cfg.get("controller", {})

    underlying_model = args.underlying_model or underlying_cfg.get("name")
    underlying_temp = underlying_cfg.get("temperature", 0.0)
    underlying_seed = underlying_cfg.get("seed")
    underlying_top_p = underlying_cfg.get("top_p")

    controller_model = args.controller_model or controller_cfg.get("name")
    controller_temp = controller_cfg.get("temperature", 0.0)
    controller_seed = controller_cfg.get("seed")
    controller_top_p = controller_cfg.get("top_p")

    enable_slsm = run_cfg.get("enable_slsm", True)
    num_workers = args.num_workers if args.num_workers is not None else run_cfg.get("num_workers", 2)
    max_retries = args.max_retries if args.max_retries is not None else 3
    retry_delay = args.retry_delay if args.retry_delay is not None else 2.0

    # Create models
    print(f"Underlying model: {underlying_model}")
    underlying_llm = get_openrouter_model(
        underlying_model, underlying_temp,
        seed=underlying_seed, top_p=underlying_top_p
    )

    if enable_slsm:
        model_name = f"{underlying_model}+SLSM({controller_model})"
    else:
        model_name = underlying_model

    def process_with_retry(conv) -> Tuple[str, bool]:
        messages = conv.conversation
        qid = conv.question_id

        for attempt in range(max_retries):
            try:
                if enable_slsm:
                    thread_controller_llm = get_openrouter_model(
                        controller_model, controller_temp,
                        seed=controller_seed, top_p=controller_top_p
                    )
                    thread_slsm_config = SLSMConfig(
                        disable_controller=slsm_cfg.get("disable_controller", False),
                        memory_mode=slsm_cfg.get("memory_mode", "structured"),
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

                if response and response.strip():
                    write_response_to_cache(cache_dir, qid, response, model_name)
                    return qid, True

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

            except Exception as e:
                if attempt == max_retries - 1:
                    write_response_to_cache(cache_dir, qid, f"[ERROR] {str(e)}", model_name)
                    return qid, False
                time.sleep(retry_delay)

        # All retries failed - keep as empty
        write_response_to_cache(cache_dir, qid, "", model_name)
        return qid, False

    # Run
    print(f"Parallel mode: {num_workers} workers, max_retries={max_retries}")
    success_count = 0

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_with_retry, conv): conv.question_id for conv in conversations}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Rerunning"):
                _, is_success = future.result()
                if is_success:
                    success_count += 1
    else:
        for conv in tqdm(conversations, desc="Rerunning"):
            _, is_success = process_with_retry(conv)
            if is_success:
                success_count += 1

    print(f"\nRerun complete: {success_count}/{len(empty_qids)} recovered")

    status = get_cache_status(cache_dir)
    print(f"Cache: {status['completed']} completed, {status['empty']} empty, {status['missing']} missing")


def cmd_merge(args, cfg: Dict[str, Any]):
    """Merge cache to single JSONL file."""
    run_cfg = cfg.get("run", {})
    paths_cfg = cfg.get("paths", {})

    tag = args.tag or run_cfg.get("tag")
    if not tag:
        print("ERROR: tag is required (via --tag or run.tag in config)")
        sys.exit(1)

    cache_dir = get_cache_dir(tag)
    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    # Output file
    output_dir = args.output_dir or paths_cfg.get("output_dir", "data/final_model_responses")
    output_file = args.output or os.path.join(output_dir, f"{tag}.jsonl")

    # Read all responses
    responses = read_cache_responses(cache_dir)

    if not responses:
        print("No responses found in cache")
        return

    # Sort by question_id for consistency
    responses.sort(key=lambda x: x.get("question_id", ""))

    # Write JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in responses:
            # Remove timestamp for final output (optional)
            output_record = {
                "question_id": r.get("question_id"),
                "model": r.get("model"),
                "response": r.get("response"),
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    # Stats
    completed = sum(1 for r in responses if r.get("response") and r.get("response").strip())
    empty = len(responses) - completed

    print(f"Merged {len(responses)} responses to: {output_file}")
    print(f"  Completed: {completed}")
    print(f"  Empty: {empty}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SLSM Multi-Challenge CLI - Split/Incremental Write",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to config YAML file")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run experiment with incremental write")
    run_parser.add_argument("--tag", "-t", type=str, help="Experiment tag (overrides config)")
    run_parser.add_argument("--benchmark", "-b", type=str, help="Benchmark file path")
    run_parser.add_argument("--num-samples", "-n", type=int, help="Number of samples")
    run_parser.add_argument("--underlying-model", type=str, help="Underlying model")
    run_parser.add_argument("--underlying-temp", type=float, help="Temperature")
    run_parser.add_argument("--controller-model", type=str, help="Controller model")
    run_parser.add_argument("--controller-temp", type=float, help="Controller temperature")
    run_parser.add_argument("--enable-slsm", type=lambda x: x.lower() == "true", help="Enable SLSM")
    run_parser.add_argument("--parallel", "-p", type=lambda x: x.lower() == "true", help="Enable parallel")
    run_parser.add_argument("--num-workers", "-w", type=int, help="Number of workers")
    run_parser.add_argument("--seed", "-s", type=int, help="Random seed")
    run_parser.add_argument("--top-p", type=float, help="Top-p value")

    # --- status command ---
    status_parser = subparsers.add_parser("status", help="Check cache status")
    status_parser.add_argument("--tag", "-t", type=str, help="Experiment tag")

    # --- rerun command ---
    rerun_parser = subparsers.add_parser("rerun", help="Rerun empty responses")
    rerun_parser.add_argument("--tag", "-t", type=str, help="Experiment tag")
    rerun_parser.add_argument("--benchmark", "-b", type=str, help="Benchmark file path")
    rerun_parser.add_argument("--underlying-model", type=str, help="Underlying model")
    rerun_parser.add_argument("--controller-model", type=str, help="Controller model")
    rerun_parser.add_argument("--num-workers", "-w", type=int, help="Number of workers")
    rerun_parser.add_argument("--max-retries", type=int, default=3, help="Max retries")
    rerun_parser.add_argument("--retry-delay", type=float, default=2.0, help="Retry delay")

    # --- merge command ---
    merge_parser = subparsers.add_parser("merge", help="Merge cache to JSONL")
    merge_parser.add_argument("--tag", "-t", type=str, help="Experiment tag")
    merge_parser.add_argument("--output", "-o", type=str, help="Output file path")
    merge_parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Load environment and config
    load_env()
    cfg = load_config(args.config)

    # Dispatch
    if args.command == "run":
        cmd_run(args, cfg)
    elif args.command == "status":
        cmd_status(args, cfg)
    elif args.command == "rerun":
        cmd_rerun(args, cfg)
    elif args.command == "merge":
        cmd_merge(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
