# Multi-Turn & Long-Context Datasets

This directory contains scripts for downloading, processing, and loading multi-turn and long-context datasets for memory framework testing.

## Available Datasets

### 1. BABILong 📍
**Purpose**: Long-context reasoning with needle-in-haystack benchmark
- **Size**: Up to 1M tokens per sequence
- **Tasks**: 10 different reasoning tasks (single/fact, multi-fact, relations, etc.)
- **Best for**: Testing retrieval in extremely long contexts

**Subsets**:
- `0k`, `1k`, `2k`, `4k`, `8k`, `16k`, `32k`, `64k`, `128k`, `256k`, `512k`, `1m`
- Each subset has different context lengths

**Usage**:
```bash
# Download specific subset
python download_datasets.py --datasets babilong --subset 128k

# Download all subsets
python download_datasets.py --datasets babilong

# Convert to unified format
python download_datasets.py --datasets babilong --convert
```

### 2. MuTual 🗣️
**Purpose**: Multi-turn dialogue reasoning
- **Size**: ~8.8K samples
- **Tasks**: Dialogue completion with reasoning
- **Best for**: Testing conversation context and reasoning

**Usage**:
```bash
python download_datasets.py --datasets mutual
python download_datasets.py --datasets mutual --convert
```

### 3. AgentBench 🤖
**Purpose**: Agent-environment interactions
- **Size**: 138 tasks
- **Tasks**: OS, DB, web shopping interactions
- **Best for**: Testing action+observation sequences

**Usage**:
```bash
python download_datasets.py --datasets agentbench
python download_datasets.py --datasets agentbench --convert
```

### 4. ProLong 📚
**Purpose**: Long-context training data
- **Size**: 64K or 512K token sequences
- **Format**: Pre-tokenized (requires special loading)
- **Best for**: Training long-context models

**Usage**:
```bash
# 64K version
python download_datasets.py --datasets prolong --version 64K

# 512K version
python download_datasets.py --datasets prolong --version 512K
```

## Quick Start

### Download All Datasets
```bash
python download_datasets.py --datasets all
```

### Download and Convert
```bash
python download_datasets.py --datasets all --convert
```

### Specify Output Directory
```bash
python download_datasets.py --output-dir ./my_data --datasets babilong mutual
```

## Loading Data

### Using the Loader Utility
```python
from load_datasets import load_unified_dataset, format_for_memory_testing

# Load dataset
data = load_unified_dataset("data/datasets/babilong_unified.jsonl")

# Format for testing
formatted = format_for_memory_testing(
    data,
    max_turns=10,
    max_context_tokens=4096
)

# Extract QA pairs
qa_pairs = extract_qa_pairs(data)
```

### Direct HuggingFace Load
```python
from load_datasets import load_hf_dataset

data = load_hf_dataset("RMT-team/babilong", subset="128k")
```

## Dataset Format

All datasets are unified to this format:

```json
{
  "dataset": "babilong|mutual|agentbench|prolong",
  "type": "long_context_qa|multi_turn_dialogue|agent_interaction|long_context_text",
  "turns": [
    {
      "role": "user|assistant|system",
      "content": "message content"
    }
  ],
  "metadata": {
    "source": "split_name",
    "num_options": 4,
    "task_id": "some_id"
  }
}
```

## Choosing the Right Dataset

| Dataset | Context Length | Turn Count | Best For |
|---------|---------------|------------|----------|
| **BABILong** | 0k - 1M tokens | 1-2 turns | Long-context retrieval, needle-in-haystack |
| **MuTual** | ~1K tokens | 4-8 turns | Multi-turn dialogue, reasoning |
| **AgentBench** | ~2K tokens | 5-20 turns | Agent interactions, environment feedback |
| **ProLong** | 64K - 512K tokens | N/A | Training long-context models |

## Memory Framework Testing

### Test Scenarios

1. **Long-Context Retrieval** (BABILong)
   - Feed progressively longer contexts
   - Test if memory layer retrieves distant facts
   - Compare with full-context baseline

2. **Multi-Turn Consistency** (MuTual)
   - Track conversation state across turns
   - Test persona consistency
   - Verify information retention

3. **Agent-Environment Interaction** (AgentBench)
   - Test action+observation memory
   - Verify state tracking over many interactions
   - Check if memory loses critical feedback

### Evaluation Metrics

- **Accuracy**: Correct answers vs baseline
- **Context Window Efficiency**: Performance vs context size
- **Memory Retention**: Information recall after many turns
- **Retrieval Precision**: Correct fact retrieval from long context

## Troubleshooting

### Dataset Not Found
```bash
# Check if dataset exists on HuggingFace
# Some datasets may have been removed or renamed
```

### Memory Issues
```python
# For large datasets, use streaming
from load_datasets import load_hf_dataset

data = load_hf_dataset("RMT-team/babilong", subset="1m")
# Process in batches to avoid OOM
```

### Token Counting
The loader uses approximate token counting (words split by spaces). For precise counting:
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = len(enc.encode(text))
```

## References

- **BABILong**: [arXiv:2406.10149](https://arxiv.org/abs/2406.10149)
- **ProLong**: [arXiv:2410.02660](https://arxiv.org/abs/2410.02660)
- **MuTual**: lighteval benchmark collection
- **AgentBench**: ETH SRI agent benchmark collection

## License

Dataset licenses vary. Please check individual HuggingFace dataset pages for terms of use.
