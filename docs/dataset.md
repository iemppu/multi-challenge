下面是各类多轮对话 / 对话能力评估基准（Benchmarks） 的官方论文链接或者权威公开链接，便于你进一步查阅与引用：

## 1. Multi-IF（多轮多语言指令跟随评估）

### 论文
Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following (arXiv)
链接: https://arxiv.org/abs/2410.15553

### 代码 / 数据集资源
Multi-IF 官方 GitHub（包含评估代码与数据说明）
https://github.com/facebookresearch/Multi-IF
 
## 2. MT-Eval（多轮能力评估指标）

### 论文
MT-Eval: A Multi-Turn Capabilities Evaluation Benchmark for Large Language Models (EMNLP 2024)
链接: https://aclanthology.org/2024.emnlp-main.1124/

### 数据集
Hugging Face 数据集（MT-Eval 多轮对话数据）
https://huggingface.co/datasets/wckwan/MT-Eval


## 3. MT-Bench（多轮开放对话 / Chatbot 评估）

### 论文 / 基准简介

MT-Bench 最早源于 Chatbot Arena 多轮对话评估方法，并作为 benchmark 在社区中广泛使用（GPT-judge / 多轮评分机制），你可以从以下学术与社区资源了解其设计与应用：
Chatbot Arena 背景文章（包含 MT-Bench Score 定义）：
https://lmsys.org/blog/2023-06-22-leaderboard/
 
多轮评估解释（MT-Bench 定义与对话评估目的）：
https://klu.ai/glossary/mt-bench-eval
 
注意: MT-Bench 本身并非单一 arXiv 论文，而是由 LMSys / FastChat / Chatbot Arena 联合发布的广义多轮对话评估 benchmark，并通过 GPT-judge（LLM 评估器）和 Elo 人类投票 leaderboard 实现模型比较。

## 4. 其他相关多轮对话 / 对话质量评估 Benchmarks（扩展参考）

MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues (ACL 2024)
链接: https://arxiv.org/abs/2402.14762

StructFlowBench: A Structured Flow Benchmark for Multi-Turn Instruction Following
链接: https://arxiv.org/abs/2502.14494

XIFBench: Evaluating Large Language Models on Multilingual Instruction Following（instruction-centric 多语评估方向）
链接: https://arxiv.org/abs/2503.07539

