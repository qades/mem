# Next Steps

This document outlines the next steps to move from the current proof-of-concept to a production-ready system.

## High Priority

### 1. Real LLM API Integration

**Current**: Test server with mock data  
**Required**: Real OpenAI-compatible API

#### Tasks
- [ ] Replace test server with actual API endpoint
- [ ] Add `api_key` parameter to `OpenAICompatibleContextManager`
- [ ] Update `config/model.json` to use environment variables for API keys
- [ ] Add proper error handling for API failures

#### Implementation
```python
# context_managers/openai_parser.py
class OpenAICompatibleContextManager:
    def __init__(
        self,
        api_url: str = "http://localhost:58080/v1",
        api_key: str = None,  # ADD THIS
        model: str = "LFM2.5-1.2B-Instruct-Q8_0",
        k_retrieval: int = 5,
        enable_benchmarking: bool = True,
    ):
        self.parser = OpenAIContextParser(
            api_url=api_url,
            api_key=api_key,  # PASS THIS
            model=model,
        )
```

```python
# config/model.json
{
  "api_key": null  # Set from environment variable in production
}
```

#### Usage
```python
import os
from context_managers.openai_parser import OpenAICompatibleContextManager

parser = OpenAICompatibleContextManager(
    api_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    parser_model="gpt-4"
)
```

### 2. Real Conversation Datasets

**Current**: 20-message test dataset  
**Required**: Real conversation datasets for meaningful benchmarking

#### Options
- [ ] **Babilong**: Long-context benchmark dataset
- [ ] **ProLong**: Conversation dataset with multiple turns
- [ ] **Custom dataset**: Your own conversation data

#### Implementation
```python
# benchmark/dataset_loader.py
def load_babilong_dataset(path: str, max_messages: int = None):
    """Load Babilong dataset."""
    # Load from JSONL or other format
    # Return (messages, references)
    pass

def load_prodlong_dataset(path: str, max_messages: int = None):
    """Load ProLong dataset."""
    # Load from JSONL or other format
    # Return (messages, references)
    pass
```

#### Usage
```bash
# Run benchmark with real dataset
./run_benchmark.sh 1000 babilong
```

### 3. Actual LLM Response Generation

**Current**: Mock response generation  
**Required**: Call real LLM with context

#### Tasks
- [ ] Replace mock response in `benchmark/harness.py`
- [ ] Add response quality metrics
- [ ] Implement proper error handling

#### Implementation
```python
# benchmark/harness.py
def generate_response(
    context: str, 
    message: Dict[str, str],
    api_url: str,
    api_key: str,
    model: str
) -> str:
    """Generate response using real LLM."""
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context:\n{context}\n\nUser: {message['content']}"}
                ]
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return "[LLM Error]"
```

### 4. Real Vector Embeddings

**Current**: Hash-based placeholder embeddings  
**Required**: Real embedding model for meaningful similarity search

#### Tasks
- [ ] Replace hash-based embedding with real model
- [ ] Support multiple embedding models
- [ ] Add caching for embeddings

#### Implementation
```python
# memory_stores/vector_db.py
def generate_embedding(self, text: str) -> List[float]:
    """Generate embedding using real model."""
    try:
        response = requests.post(
            f"{self.api_url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={"input": text, "model": self.embedding_model}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding API error: {e}")
        return self._fallback_embedding(text)
```

### 5. Integration Testing

**Current**: Mock data tests  
**Required**: Tests with real APIs

#### Tasks
- [ ] Add tests for real API calls
- [ ] Test with real datasets
- [ ] End-to-end workflow tests

#### Implementation
```python
# test_new_system.py
def test_openai_parser_real():
    """Test OpenAI parser with real API."""
    parser = OpenAICompatibleContextManager(
        api_url="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        parser_model="gpt-4"
    )
    
    result = parser.parse_message("Hello, I love Python")
    assert "entities" in result
    assert len(result["entities"]) > 0
    assert result["confidence"] > 0.5
```

## Medium Priority

### 6. Async I/O

**Benefit**: Parallel API calls, faster processing  
**Action**: Measure before implementing

#### Tasks
- [ ] Measure current performance impact
- [ ] Implement async if beneficial
- [ ] Test with async vs sync

#### Implementation
```python
import asyncio
import aiohttp

async def batch_parse_messages_async(messages: List[Dict]) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [
            parse_message_async(session, msg) 
            for msg in messages
        ]
        return await asyncio.gather(*tasks)
```

### 7. Caching Layer

**Benefit**: Reduce API costs, faster processing  
**Note**: Will reduce significance of benchmark results

#### Tasks
- [ ] Implement in-memory caching
- [ ] Add cache invalidation
- [ ] Measure cost savings

#### Implementation
```python
from functools import lru_cache

class OpenAIContextParser:
    @lru_cache(maxsize=1000)
    def _cached_parse(self, content: str) -> Dict:
        return self._parse_content(content)
    
    def parse_message(self, content: str) -> Dict:
        return self._cached_parse(content)
```

### 8. Monitoring & Logging

**Benefit**: Production visibility, debugging

#### Tasks
- [ ] Add structured logging
- [ ] Implement metrics collection
- [ ] Add tracing for API calls

#### Implementation
```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_message(self, message: Dict[str, str]) -> None:
    start = time.time()
    
    parsed = self.parser.parse_message(...)
    
    elapsed = (time.time() - start) * 1000
    logger.info(f"Parsed message in {elapsed:.2f}ms")
    
    if elapsed > 1000:
        logger.warning(f"Slow parsing: {elapsed:.2f}ms")
```

### 9. Error Handling

**Benefit**: Robustness, user-friendly errors

#### Tasks
- [ ] Add retry logic with exponential backoff
- [ ] Implement fallback strategies
- [ ] Add graceful degradation

#### Implementation
```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAIContextParser:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def parse_message(self, content: str) -> Dict:
        response = requests.post(...)
        response.raise_for_status()
        return response.json()
```

### 10. Cost Tracking

**Benefit**: Understand API costs

#### Tasks
- [ ] Track token usage
- [ ] Calculate costs per strategy
- [ ] Generate cost reports

#### Implementation
```python
def track_cost(tokens: int, model: str) -> float:
    prices = {
        "gpt-3.5-turbo": 0.0005,  # $0.50 per 1M tokens
        "gpt-4": 0.03,  # $30 per 1M tokens
    }
    return prices.get(model, 0) * (tokens / 1_000_000)
```

## Low Priority

### 11. Advanced Retrieval

- [ ] Hybrid search (vector + keyword)
- [ ] Reranking with cross-encoder
- [ ] Context compression

### 12. Multi-Modal Support

- [ ] Image embeddings
- [ ] Audio processing
- [ ] Video context extraction

### 13. Distributed Processing

- [ ] Redis cache layer
- [ ] Message queues
- [ ] Worker pools

### 14. Web Interface

- [ ] Dashboard for results
- [ ] Real-time monitoring
- [ ] Configuration UI

### 15. Documentation

- [ ] API documentation (Sphinx/autodoc)
- [ ] Tutorial videos
- [ ] Code examples for common use cases

## Quick Start for Next Steps

### 1. Add Real API Key
```bash
export OPENAI_API_KEY="sk-..."
export LLM_API_URL="https://api.openai.com/v1"
```

### 2. Test with Real API
```python
from context_managers.openai_parser import OpenAICompatibleContextManager

parser = OpenAICompatibleContextManager(
    api_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    parser_model="gpt-4"
)

result = parser.parse_message("Hello, I love Python")
print(result)
```

### 3. Run Benchmark with Real API
```bash
./run_benchmark.sh 100
```

### 4. Compare Results
- Check token usage
- Measure response times
- Evaluate quality improvements

## Notes

- Start with **priority 1-5** for meaningful results
- Measure performance before implementing async
- Consider caching if API costs are high
- Add monitoring before production deployment
