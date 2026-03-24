"""
TrustGraph-AI based memory store with benchmarkable data structures.
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from .base import BaseMemoryStore

try:
    import requests
except ImportError:
    requests = None


class TrustGraphStore(BaseMemoryStore):
    """Memory store using TrustGraph-AI with benchmarkable data structures."""

    def __init__(
        self,
        api_url: str = "http://localhost:3000",
        api_key: str = None,
        vault: str = "default",
        embedding_model: str = "text-embedding-3-small",
        context_parser_model: str = "gpt-3.5-turbo",
        k_retrieval: int = 5,
        use_embeddings: bool = True,
        enable_benchmarking: bool = True,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.vault = vault
        self.embedding_model = embedding_model
        self.context_parser_model = context_parser_model
        self.k_retrieval = k_retrieval
        self.use_embeddings = use_embeddings
        self.enable_benchmarking = enable_benchmarking

        self.memory_counter = 0
        self.message_log: List[Dict[str, Any]] = []
        self.entity_index: Dict[str, List[str]] = defaultdict(list)
        self.benchmark_data: Dict[str, List[float]] = defaultdict(list)

        # Test connection
        self._connected = self._test_connection()

    def _test_connection(self) -> bool:
        """Test API connection."""
        try:
            import requests

            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _benchmark_operation(self, operation: str, func, *args, **kwargs) -> Any:
        """Benchmark an operation and record timing."""
        if not self.enable_benchmarking:
            return func(*args, **kwargs)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000  # ms
            self.benchmark_data[operation].append(elapsed)
            return result
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.benchmark_data[operation].append(elapsed)
            raise e

    def _call_api(
        self, endpoint: str, method: str = "POST", data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make API call to TrustGraph-AI."""
        import requests

        url = f"{self.api_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=data, timeout=10)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI-compatible API."""
        import requests

        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.embedding_model,
                    "input": text,
                },
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("data", [{}])[0].get("embedding", [])
        except Exception:
            # Fallback: simple hash-based embedding
            import hashlib

            hash_val = hashlib.md5(text.encode()).hexdigest()
            return [ord(hash_val[i % len(hash_val)]) / 255.0 - 0.5 for i in range(384)]

    def _parse_context(self, text: str, role: str = "user") -> Dict[str, Any]:
        """Parse context using OpenAI-compatible API."""
        import requests

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.context_parser_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": """Extract key information from this message.
Return JSON with: entities (list), relations (list of triples), facts (list), 
preferences (list), timestamps (list), confidence (0.0-1.0).""",
                        },
                        {"role": role, "content": text},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1000,
                },
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            content = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            )
            return json.loads(content)
        except Exception:
            return {
                "entities": self._extract_entities(text),
                "relations": [],
                "facts": [text[:200]] if text else [],
                "preferences": [],
                "timestamps": [],
                "confidence": 0.5,
            }

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction fallback."""
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and word.isalpha():
                if i < len(words) - 1 and words[i + 1][0].isupper():
                    entities.append(f"{word} {words[i + 1]}")
                else:
                    entities.append(word)
        return list(set(entities))[:10]

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert information into TrustGraph-AI."""

        def _insert_impl():
            memory_id = f"tg_{self.memory_counter}"
            self.memory_counter += 1

            timestamp = (
                metadata.get("timestamp", datetime.now().isoformat())
                if metadata
                else datetime.now().isoformat()
            )
            confidence = metadata.get("confidence", 1.0) if metadata else 1.0

            result = self._call_api(
                "/memories",
                data={
                    "content": f"{entity} {relation} {value}",
                    "vault": self.vault,
                    "type": "fact",
                    "entities": [
                        {"name": entity, "type": "entity"},
                        {"name": value, "type": "value"},
                    ],
                    "entity_relationships": [
                        {
                            "from_entity": entity,
                            "to_entity": value,
                            "rel_type": relation,
                        }
                    ],
                    "summary": f"{entity} {relation} {value}"[:100],
                    "created_at": timestamp,
                    "confidence": confidence,
                },
            )

            if result.get("success", False):
                memory_id = result.get("id", memory_id)
                self.entity_index[entity].append(memory_id)
                self.message_log.append(
                    {
                        "id": memory_id,
                        "entity": entity,
                        "relation": relation,
                        "value": value,
                        "timestamp": timestamp,
                        "confidence": confidence,
                    }
                )

            return memory_id

        return self._benchmark_operation("insert", _insert_impl)

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant information using semantic search."""

        def _retrieve_impl():
            results = []
            query_embedding = self._generate_embedding(query)

            search_result = self._call_api(
                "/search",
                data={
                    "query": query,
                    "embedding": query_embedding,
                    "vault": self.vault,
                    "limit": k,
                    "use_embeddings": use_embedding and self.use_embeddings,
                },
            )

            if search_result.get("success", False):
                results = search_result.get("results", [])
            else:
                for msg in self.message_log:
                    if query.lower() in str(msg).lower():
                        results.append(msg)
                        if len(results) >= k:
                            break

            return results[:k]

        return self._benchmark_operation("retrieve", _retrieve_impl)

    def update(
        self,
        memory_id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update existing memory."""

        def _update_impl():
            if not self._memory_exists(memory_id):
                return False

            result = self._call_api(
                "/memories/update",
                data={
                    "id": memory_id,
                    "vault": self.vault,
                    "entity": entity,
                    "relation": relation,
                    "value": value,
                },
            )
            return result.get("success", False)

        return self._benchmark_operation("update", _update_impl)

    def delete(self, memory_id: str) -> bool:
        """Delete memory from TrustGraph-AI."""

        def _delete_impl():
            result = self._call_api(
                "/memories/delete",
                data={"id": memory_id, "vault": self.vault},
            )

            if result.get("success", False):
                for entity, memory_ids in self.entity_index.items():
                    if memory_id in memory_ids:
                        memory_ids.remove(memory_id)

            return result.get("success", False)

        return self._benchmark_operation("delete", _delete_impl)

    def clear(self) -> None:
        """Clear all memories in vault."""
        self._benchmark_operation(
            "clear",
            lambda: self._call_api("/memories/clear", data={"vault": self.vault}),
        )
        self.message_log.clear()
        self.entity_index.clear()
        self.memory_counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get TrustGraph-AI statistics including benchmark data."""
        stats = self._call_api("/stats", data={"vault": self.vault})
        result = stats.get(
            "stats",
            {
                "total_memories": len(self.message_log),
                "unique_entities": len(self.entity_index),
                "memory_counter": self.memory_counter,
            },
        )

        if self.enable_benchmarking:
            result["benchmark_stats"] = {
                op: {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times) if times else 0,
                    "min_ms": min(times) if times else 0,
                    "max_ms": max(times) if times else 0,
                }
                for op, times in self.benchmark_data.items()
            }

        return result

    def _memory_exists(self, memory_id: str) -> bool:
        """Check if memory exists."""
        return any(m["id"] == memory_id for m in self.message_log)

    def store_message(
        self, message: Dict[str, str], message_id: str = None
    ) -> List[str]:
        """Store a message and extract information."""
        content = message.get("content", "")
        role = message.get("role", "user")

        parsed = self._parse_context(content, role)
        memory_ids = []

        for entity in parsed.get("entities", []):
            memory_id = self.insert(
                entity=entity,
                relation="mentioned_in",
                value=message_id or f"msg_{self.memory_counter}",
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.8,
                },
            )
            memory_ids.append(memory_id)

        for rel in parsed.get("relations", []):
            memory_id = self.insert(
                entity=rel.get("entity", "unknown"),
                relation=rel.get("relation", "related_to"),
                value=rel.get("value", ""),
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "confidence": rel.get("confidence", 0.7),
                },
            )
            memory_ids.append(memory_id)

        for fact in parsed.get("facts", []):
            memory_id = self.insert(
                entity="fact",
                relation="states",
                value=fact[:500],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "confidence": parsed.get("confidence", 0.7),
                },
            )
            memory_ids.append(memory_id)

        for pref in parsed.get("preferences", []):
            memory_id = self.insert(
                entity="user",
                relation="has_preference",
                value=pref,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.9,
                },
            )
            memory_ids.append(memory_id)

        return memory_ids

    def get_relevant_context(
        self, current_message: Dict[str, str], k: int = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for current message."""
        k = k or self.k_retrieval
        query = current_message.get("content", "")
        return self.retrieve(query, k=k, use_embedding=self.use_embeddings)

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get benchmark summary for all operations."""
        if not self.enable_benchmarking:
            return {"message": "Benchmarking not enabled"}

        summary = {}
        for op, times in self.benchmark_data.items():
            if times:
                summary[op] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "p95_ms": sorted(times)[int(len(times) * 0.95)]
                    if len(times) > 1
                    else times[0],
                }

        return summary
