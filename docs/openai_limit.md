# OpenAI API Rate Limits Analysis

## Sources
- [OpenAI Rate Limits Guide](https://platform.openai.com/docs/guides/rate-limits)
- [OpenAI API Rate Limits 2025 Update](https://www.scriptbyai.com/rate-limits-openai-api/)
- [How to Handle Rate Limits - OpenAI Cookbook](https://cookbook.openai.com/examples/how_to_handle_rate_limits)

## Rate Limits by Tier

### GPT-4o

| Tier | RPM (Requests/min) | TPM (Tokens/min) |
|------|-------------------|------------------|
| Tier 1 | 500 | 30,000 |
| Tier 2 | 5,000 | 450,000 |
| Tier 3 | 5,000 | 800,000 |
| Tier 4 | 10,000 | 2,000,000 |
| Tier 5 | 10,000 | 30,000,000 |

### GPT-4o-mini

| Tier | RPM (Requests/min) | TPM (Tokens/min) |
|------|-------------------|------------------|
| Tier 1 | 500 | 200,000 |
| Tier 2 | 5,000 | 2,000,000 |
| Tier 3 | 5,000 | 4,000,000 |
| Tier 4 | 10,000 | 10,000,000 |
| Tier 5 | 30,000 | 150,000,000 |

## Token Consumption Estimation (Multi-Challenge Benchmark)

Based on analysis of 10 sample conversations:

| Component | Estimated Tokens |
|-----------|-----------------|
| Base conversation (input) | ~1,676 |
| SLSM Controller call (gpt-4o-mini) | ~700 |
| Memory note injection | ~200 |
| Underlying LLM output | ~500 |
| **Total per GPT-4o request** | **~2,376** |
| **Total per GPT-4o-mini request** | **~700** |

## Parallel Workers Recommendation

Assumptions:
- Each API request takes ~5 seconds (network + processing latency)
- 1 worker can complete ~12 requests per minute
- 80% safety margin applied

### Tier 1 (New accounts, $5-$50 spent)

| Model | TPM Limit | RPM Limit | Max Workers |
|-------|-----------|-----------|-------------|
| GPT-4o | 4.2 | 41.7 | **3** |
| GPT-4o-mini | 95.2 | 41.7 | 33 |

**Bottleneck**: GPT-4o TPM

**Recommendation**: `parallel: true`, `num_workers: 3`

> Note: Based on actual test (each request ~20 seconds), not theoretical 5s latency.

### Tier 2 ($50-$100 spent)

| Model | TPM Limit | RPM Limit | Max Workers |
|-------|-----------|-----------|-------------|
| GPT-4o | 15.8 | 416.7 | **12** |
| GPT-4o-mini | 238.1 | 416.7 | 190 |

**Bottleneck**: GPT-4o TPM

**Recommendation**: `parallel: true`, `num_workers: 8-12`

### Tier 3+ ($100+ spent)

**Recommendation**: `parallel: true`, `num_workers: 15-20`

## Configuration Examples

### Tier 1
```yaml
run:
  parallel: true
  num_workers: 3
```

### Tier 2 (Balanced)
```yaml
run:
  parallel: true
  num_workers: 8
```

### Tier 3+ (Aggressive)
```yaml
run:
  parallel: true
  num_workers: 15
```

## Handling 429 Errors

If you encounter 429 errors:
1. Reduce `num_workers` by 50%
2. Add exponential backoff with jitter (already handled by OpenAI SDK)
3. Check your current tier at [OpenAI Limits Page](https://platform.openai.com/account/limits)

## Notes

- Rate limits are per-organization, not per-API-key
- Tiers automatically increase with successful payment history
- TPM limits typically more restrictive than RPM for long conversations
