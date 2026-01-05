# MultiChallenge Baseline Models

Based on the paper "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs" (arXiv:2501.17399v2)

## Frontier Models (Human Evaluated - Table 2 & Auto Evaluated - Table 3)

| Model Name | Version/Date |
|------------|-------------|
| GPT-4o | August 2024 |
| Llama 3.1 405B Instruct | - |
| Mistral Large | - |
| Claude 3.5 Sonnet | June 2024 |
| Gemini 1.5 Pro | August 27, 2024 |
| o1-preview | - |

## Open Source Models (Auto Evaluated - Table 5)

| Model Name | Parameters |
|------------|------------|
| Llama-3.2-3B-Instruct | 3B |
| Llama-3.3-70B-Instruct | 70B |
| Qwen2-72B-Instruct | 72B |
| Qwen2.5-14B-Instruct | 14B |
| Qwen2.5-72B-Instruct | 72B |
| Mixtral-8x7B-Instruct-v0.1 | 8x7B |
| Mixtral-8x22B-Instruct-v0.1 | 8x22B |

## Responder Agents (Used for Synthetic Data Generation - Appendix A.4)

| Model Name | Version |
|------------|---------|
| GPT-4o | August 2024 |
| Llama 3.1 405B Instruct | - |
| Mistral Large | November 2024 |
| Claude 3.5 Sonnet | October 2024 |
| Gemini 1.5 Pro | August 27, 2024 |
| o1-preview | - |

## Key Results Summary

- Best performing model: **Claude 3.5 Sonnet (June 2024)** with 41.4% average accuracy
- Second best: **o1-preview** with 37.23% average accuracy
- All frontier models achieved less than 50% accuracy on MultiChallenge
