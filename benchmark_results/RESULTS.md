# Benchmark Results - 20 Messages

## Test Configuration
- API URL: http://localhost:58080/v1
- Parser Model: LFM2.5-1.2B-Instruct-Q8_0
- Chat Model: Qwen3-Coder-Next-Q4_K_M
- Excerpt Size: 20 messages

## Results

| Strategy | Context Size | Response Time | Notes |
|----------|--------------|---------------|-------|
| **Baseline** | 249 tokens | 0.03ms | Full history, fastest |
| **OpenAI Parser** | 1000 tokens | 25,892ms | LLM extraction, slowest |
| **Vector DB** | 500 tokens | 35.57ms | Similarity search, balanced |

## Performance Breakdown (OpenAI Parser)
- Total messages processed: 20
- Average time per message: 1,294ms
- Minimum: 901ms, Maximum: 1,909ms
- Total parsing time: 25,890ms

## Observations

1. **Baseline**: Fastest but largest context (no extraction overhead)
2. **Vector DB**: Balanced approach with moderate context reduction
3. **OpenAI Parser**: Most context reduction but significant latency from LLM calls

## Recommendations

- **For speed**: Use Baseline
- **For context efficiency**: Use Vector DB
- **For structured data extraction**: Use OpenAI Parser (accept latency tradeoff)

## Notes

- Parser latency is high due to local LLM inference
- With remote API (e.g., OpenAI), latency would be different
- Vector DB provides good balance of speed and context reduction
