# Tasks

## 2026-01-04 14:30

### Task 1: Convert API calls to OpenRouter
**Status**: Pending

**Description**:
Replace current direct API calls (OpenAI, Gemini) with OpenRouter unified API endpoint.

**Affected files**:
- `src/models/openai.py`
- `src/models/gemini.py`
- Possibly create `src/models/openrouter.py`

**Key changes**:
- Use OpenRouter base URL: `https://openrouter.ai/api/v1`
- Use `OPENROUTER_API_KEY` environment variable
- Map model names to OpenRouter format (e.g., `openai/gpt-4o`, `google/gemini-2.5-pro`)

---

### Task 2: Verify input/output format compatibility with OpenRouter
**Status**: Pending

**Description**:
Check that current message format and response parsing are compatible with OpenRouter's API specification.

**Checklist**:
- [ ] Verify message format: `{"role": "user"|"assistant"|"system", "content": "..."}`
- [ ] Verify response parsing: `response.choices[0].message.content`
- [ ] Check if structured output (response_format) is supported
- [ ] Verify streaming compatibility (if needed)
- [ ] Check rate limit handling differences

**Reference**: OpenRouter API docs - https://openrouter.ai/docs

---

### Task 3: Add multi-threading support for API calls
**Status**: Pending

**Description**:
Implement proper multi-threaded API calls with rate limiting and error handling.

**Requirements**:
- Use `concurrent.futures.ThreadPoolExecutor`
- Add configurable `max_workers` parameter
- Implement retry logic with exponential backoff for rate limits
- Add proper error handling and logging
- Consider thread-safe response collection

**Affected files**:
- `src/data_loader.py` (already has basic threading)
- `src/evaluator.py` (already has basic threading)
- New: May need shared rate limiter across threads

---

### Task 4: Add token/price estimation
**Status**: Pending

**Description**:
Implement token counting and cost estimation for API calls to track usage and predict expenses.

**Requirements**:
- Count input/output tokens for each API call
- Calculate cost based on model pricing (per 1M tokens)
- Support different pricing for different models (GPT-4o, Gemini, etc.)
- Log cumulative token usage and cost during benchmark runs
- Provide summary report at end of run

**Implementation options**:
- Use `tiktoken` for OpenAI models token counting
- Use API response `usage` field when available
- Create pricing config file for different models

**Affected files**:
- `src/models/openai.py`
- `src/models/gemini.py`
- New: `src/utils/token_counter.py` or similar
