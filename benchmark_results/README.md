# Benchmark Results

## 20 Messages Test

| Strategy | Context Size | Response Time | Notes |
|----------|--------------|---------------|-------|
| Baseline | 249 tokens | 0.03ms | Full history, fastest |
| OpenAI Parser | 1000 tokens | 25,892ms | LLM extraction, slowest |
| Vector DB | 500 tokens | 35.57ms | Similarity search, balanced |

See `benchmark_results/benchmark_20_messages.json` for detailed results.

## Next Steps

- Run with real LLM API
- Test with larger datasets
- Compare different vector store backends (ChromaDB, FAISS)
- Measure actual LLM response quality
