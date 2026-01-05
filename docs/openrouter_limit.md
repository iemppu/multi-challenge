# OpenRouter API Limits and Pricing

## Sources
- [OpenRouter Rate Limits Documentation](https://openrouter.ai/docs/api/reference/limits)
- [OpenRouter Pricing](https://openrouter.ai/pricing)
- [OpenRouter Models Pricing Calculator](https://invertedstone.com/calculators/openrouter-pricing)
- [OpenRouter GPT-4o Comparison](https://openrouter.ai/compare/openai/chatgpt-4o-latest)

## Rate Limits

### Free Tier
| Limit Type | Value |
|------------|-------|
| Requests per day | 50 |
| Requests per minute | 20 RPM |

### Paid Tier ($10+ credits)
| Limit Type | Value |
|------------|-------|
| RPM | High global limits (not publicly specified) |
| TPM | Not publicly specified |
| Daily limits | None specified |

**Key difference from OpenAI**: OpenRouter does NOT publish fixed RPM/TPM limits for paid users. They use "global capacity governance" across all accounts.

## GPT-4o Pricing via OpenRouter

OpenRouter passes through provider pricing with no markup (except 5.5% platform fee for pay-as-you-go).

### GPT-4o (openai/gpt-4o-2024-08-06)
| Token Type | Price per 1K tokens | Price per 1M tokens |
|------------|---------------------|---------------------|
| Input | $0.0025 | $2.50 |
| Output | $0.0100 | $10.00 |

### GPT-4o-mini (openai/gpt-4o-mini)
| Token Type | Price per 1K tokens | Price per 1M tokens |
|------------|---------------------|---------------------|
| Input | $0.00015 | $0.15 |
| Output | $0.00060 | $0.60 |

## OpenRouter vs OpenAI Direct Comparison

| Feature | OpenAI Direct | OpenRouter |
|---------|---------------|------------|
| Rate limits | Tier-based, clearly defined | Global capacity, not specified |
| Pricing | Same base price | +5.5% platform fee |
| 429 handling | Your responsibility | Cloudflare DDoS protection |
| Multi-provider | No | Yes (can failover) |

## Cost Estimation for Multi-Challenge

Based on token consumption analysis (from openai_limit.md):
- Average input tokens per request: ~2,376 (GPT-4o)
- Average output tokens per request: ~500
- Controller tokens per request: ~700 (GPT-4o-mini)

### Per Request Cost
| Model | Input Cost | Output Cost | Total |
|-------|------------|-------------|-------|
| GPT-4o | $0.00594 | $0.00500 | $0.01094 |
| GPT-4o-mini | $0.000105 | $0.00042 | $0.000525 |
| **Combined** | | | **~$0.0115** |

### Full Benchmark (273 questions)
| Item | Cost |
|------|------|
| GPT-4o calls | $2.99 |
| GPT-4o-mini calls | $0.14 |
| Platform fee (5.5%) | $0.17 |
| **Total estimated** | **~$3.30** |

## Parallel Workers Recommendation for OpenRouter

Since OpenRouter doesn't publish specific RPM/TPM limits:

| Scenario | Recommendation |
|----------|----------------|
| Free tier | Sequential only (`parallel: false`) |
| Paid ($10+ credits) | Start with `num_workers: 3`, increase if no 429 |
| Conservative | `num_workers: 2-3` |
| Aggressive (test first) | `num_workers: 5-8` |

**Note**: OpenRouter may have lower effective limits than OpenAI direct since requests go through additional infrastructure.

## Monitoring Usage

Check your rate limit status:
```bash
curl https://openrouter.ai/api/v1/key \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

Response includes:
- `usage`: Total credits used
- `limit`: Credit limit (if set)
- Rate limit headers in responses

## Handling Rate Limits

1. OpenRouter uses Cloudflare protection that blocks excessive requests
2. No explicit retry-after header documented
3. Implement exponential backoff with jitter
4. Consider using multiple models as fallback (OpenRouter advantage)
