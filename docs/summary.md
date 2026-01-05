# Multi-Challenge Repository Summary

## Repository Overview

This repository implements a **benchmark evaluation framework for multi-turn conversational LLMs**. It evaluates how well language models handle complex conversational scenarios across four axes:
- **INSTRUCTION_RETENTION**: Ability to remember and follow user instructions throughout conversation
- **INFERENCE_MEMORY**: Ability to recall and use information provided earlier in conversation
- **RELIABLE_VERSION_EDITING**: Ability to handle user corrections and updates consistently
- **SELF_COHERENCE**: Ability to maintain internal consistency in responses

The repository also includes an implementation of **SLSM (Semantic-Level State Machine)**, a wrapper that tracks semantic state across conversation turns to improve model performance on these challenging scenarios.

---

## Python Scripts

### `main.py`
**Summary**: Main entry point for running the LLM benchmark. Loads benchmark questions, generates or loads model responses, evaluates them using a GPT-4o judge, and outputs scores.

| Function | Description |
|----------|-------------|
| `parse_provider_args(provider_args)` | Parses key-value pairs from command-line --provider-args into a dictionary. |
| `main()` | Orchestrates the entire benchmark pipeline: loads data, generates/loads responses, evaluates, calculates scores, and saves results. |

---

### `run_judge_eval.py`
**Summary**: Standalone script to run judge-based evaluation on pre-generated model responses. Reads responses from JSONL, evaluates using GPT-4o, and outputs scores and detailed CSV.

| Function | Description |
|----------|-------------|
| `load_responses_jsonl(path)` | Loads responses from a JSONL file with QUESTION_ID/RESPONSE format into a dictionary. |
| `main()` | Parses arguments, loads benchmark questions and responses, runs evaluation, computes scores, and saves outputs. |

---

### `run_slsm_multichallenge_gpt4o.py`
**Summary**: Runs the SLSM-wrapped GPT-4o model on the Multi-Challenge benchmark. Uses GPT-4o-mini as the controller and GPT-4o as the underlying model.

*No standalone functions defined - script-level execution only.*

---

### `run_slsm_multichallenge_gemini25pro.py`
**Summary**: Runs the SLSM-wrapped Gemini 2.5 Pro model on the Multi-Challenge benchmark. Uses GPT-4o-mini as the controller and Gemini 2.5 Pro as the underlying model.

*No standalone functions defined - script-level execution only.*

---

## Source Modules (`src/`)

### `src/conversation.py`
**Summary**: Defines the `Conversation` dataclass that represents a single benchmark conversation with its metadata (question_id, axis, messages, target_question, pass_criteria).

| Class/Function | Description |
|----------------|-------------|
| `Conversation` | Dataclass holding a benchmark conversation: question_id, axis, conversation messages, target_question, and pass_criteria. |

---

### `src/data_loader.py`
**Summary**: Handles loading benchmark questions from JSONL, loading pre-generated responses, and generating new responses using a model provider.

| Function | Description |
|----------|-------------|
| `DataLoader.__init__(input_file, response_file)` | Initializes the loader with paths to input questions and optional responses. |
| `DataLoader.load_data()` | Loads benchmark questions from JSONL and creates Conversation objects. |
| `DataLoader.load_responses(response_file)` | Loads pre-generated model responses from a JSONL file. |
| `DataLoader.generate_responses(model_provider, attempts, max_workers)` | Generates responses for each conversation using the provided model, with parallel execution support. |
| `DataLoader.get_conversations()` | Returns the list of loaded Conversation objects. |
| `DataLoader.get_responses()` | Returns the dictionary of responses keyed by question_id. |

---

### `src/evaluator.py`
**Summary**: Evaluates model responses using GPT-4o as a judge. Determines if each response meets the specified pass criteria (YES/NO verdict).

| Function | Description |
|----------|-------------|
| `JudgeResponse` | Pydantic model for structured judge output with reasoning and YES/NO verdict. |
| `Evaluator.__init__(conversations, responses)` | Initializes the evaluator with conversations and responses to evaluate. |
| `Evaluator.evaluate_helper(i, conversation, response)` | Evaluates a single response against pass criteria using the judge model. |
| `Evaluator.evaluate(max_workers)` | Evaluates all responses in parallel, computes pass/fail for each question across attempts. |

---

### `src/evaluator_bk.py`
**Summary**: Backup version of evaluator with added retry logic for handling rate limit (429) errors from the OpenAI API.

| Function | Description |
|----------|-------------|
| `call_with_retry(fn, max_retries)` | Wraps a function call with exponential backoff retry for rate limit errors. |
| `JudgeResponse` | Pydantic model for structured judge output with reasoning and YES/NO verdict. |
| `Evaluator.__init__(conversations, responses)` | Initializes the evaluator with conversations and responses. |
| `Evaluator.evaluate_helper(i, j, conversation, response)` | Evaluates a single response with retry logic, returns index, axis, reasoning, verdict, criteria. |
| `Evaluator.evaluate(max_workers)` | Evaluates all responses in parallel with retry support for rate limits. |

---

### `src/result_parser.py`
**Summary**: Parses evaluation results to calculate aggregate scores by axis and overall, and saves detailed raw output to CSV.

| Function | Description |
|----------|-------------|
| `ResultParser.__init__(evaluation_results)` | Initializes parser with evaluation results list. |
| `ResultParser.calculate_scores()` | Calculates per-axis pass rates and overall score (average of axis scores). |
| `ResultParser.save_raw_output(output_file, conversations, responses, attempts)` | Saves detailed evaluation results including conversation, responses, verdicts to CSV. |

---

### `src/slsm_wrapper.py`
**Summary**: Implements SLSM (Semantic-Level State Machine), a wrapper that tracks semantic state (facts, constraints, edits, unknowns) across conversation turns and injects memory notes to improve model consistency.

| Function/Class | Description |
|----------------|-------------|
| `LLM` | Protocol interface for model-agnostic LLM generate method. |
| `_normalize_messages_no_system(messages)` | Folds system messages into the first user message for APIs that don't support system role. |
| `_safe_generate(llm, messages, **kwargs)` | Safely calls llm.generate with filtered kwargs and handles system role normalization. |
| `_turn_text(turn)` | Converts a single message turn to "ROLE: content" string format. |
| `_evidence_grounded(evidence, turn_text)` | Checks if evidence string is a literal substring of the cited turn text (strict grounding). |
| `SemanticState` | Dataclass holding tracked facts, assumptions, constraints, unknowns, edits, and plan. |
| `SemanticState.to_compact_note(max_items)` | Renders a concise memory note from the state for injection into prompts. |
| `SLSMConfig` | Configuration dataclass for SLSM behavior (injection policy, temperature, token limits). |
| `_controller_prompt(history, new_msg, prev_state)` | Builds the prompt for the controller LLM to update semantic state. |
| `SLSMController.__init__(controller_llm, cfg)` | Initializes the controller with a cheap LLM for state tracking. |
| `SLSMController.update(history, new_msg, prev)` | Updates semantic state given new message, returns updated SemanticState. |
| `SLSMWrapper.__init__(controller, cfg)` | Initializes wrapper with controller and configuration. |
| `SLSMWrapper.track_state(conversation)` | Processes full conversation turn-by-turn to get final semantic state. |
| `SLSMWrapper.build_final_messages(original_conversation, state, system_prompt)` | Builds final messages for underlying LLM, optionally injecting memory note based on risk. |
| `SLSMWrapper.generate_last_turn(underlying_llm, original_conversation, system_prompt, **gen_kwargs)` | Main entry: tracks state, builds messages, generates response from underlying model. |

---

## Model Providers (`src/models/`)

### `src/models/base.py`
**Summary**: Abstract base class defining the interface for all model providers.

| Class | Description |
|-------|-------------|
| `ModelProvider` | ABC with abstract `generate(prompt)` method that all providers must implement. |

---

### `src/models/factory.py`
**Summary**: Factory pattern for creating model provider instances based on provider name.

| Function | Description |
|----------|-------------|
| `ModelFactory.register_provider(name, provider)` | Registers a new model provider class under a name. |
| `ModelFactory.get_provider(provider_name, **kwargs)` | Creates and returns an instance of the requested provider with given kwargs. |

---

### `src/models/openai.py`
**Summary**: OpenAI model provider using the OpenAI API for chat completions.

| Function | Description |
|----------|-------------|
| `OpenAIModel.__init__(model, temp, response_format)` | Initializes OpenAI client with API key, model name, temperature, and optional structured output format. |
| `OpenAIModel.generate(prompt)` | Generates a response using OpenAI chat completions API; supports string or message list prompts. |

---

### `src/models/gemini.py`
**Summary**: Google Gemini model provider supporting both google-genai and google-generativeai SDKs.

| Function | Description |
|----------|-------------|
| `GeminiModel.__init__(model, temp, api_key)` | Initializes Gemini client, preferring new google-genai SDK with fallback to google-generativeai. |
| `GeminiModel._to_gemini_contents(messages)` | Static method to convert OpenAI-style messages to Gemini contents format (user/model roles). |
| `GeminiModel.generate(messages, **kwargs)` | Generates response using Gemini API, handling both SDK variants. |
| `GeminiModel.chat(messages, **kwargs)` | Alias for generate() for compatibility with code using .chat() method. |

---

### `src/models/huggingface.py`
**Summary**: HuggingFace model provider using transformers pipeline for local model inference.

| Function | Description |
|----------|-------------|
| `HuggingFaceModel.__init__(model_path, temp, top_p)` | Initializes HuggingFace text-generation pipeline with specified model and sampling parameters. |
| `HuggingFaceModel.generate(chat)` | Generates a response from the local model given a chat message list. |

---

## Jupyter Notebooks (`scripts/`)

### `scripts/First_GPT_API_Test.ipynb`
**Summary**: Initial testing notebook for OpenAI API connectivity. Tests basic API calls, multi-turn conversation handling, and clones the multi-challenge repo.

---

### `scripts/Reproduce_MultiChallenge.ipynb`
**Summary**: Full reproduction notebook for Multi-Challenge benchmark. Clones repo, installs dependencies, runs judge evaluation on multiple model response files (GPT-4o, Claude, Gemini, Llama, Mistral, o1-preview), and generates summary comparison table.

---

### `scripts/SLSM_Wrapper.ipynb`
**Summary**: Development and testing notebook for SLSM wrapper. Demonstrates SLSM state tracking, compares baseline vs SLSM-controlled GPT-4o responses, runs evaluation on first 50 samples, and generates comparison statistics with LaTeX tables.

---

### `Untitled.ipynb`
**Summary**: Empty/placeholder notebook with no content.

---

## Data Files

- `data/benchmark_questions.jsonl`: 273 multi-turn conversation benchmark questions
- `data/final_model_responses/*.jsonl`: Pre-generated responses from various models
- `data/response_template.jsonl`: Template format for response files
