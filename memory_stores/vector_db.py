"""
Vector DB-based memory store.
"""

import json
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

from .base import BaseMemoryStore


class VectorDBStore(BaseMemoryStore):
    """Memory store using vector embeddings for retrieval."""

    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.vector_dim: int = 768  # Default BERT dimension
        self.index: Dict[str, List[str]] = defaultdict(list)  # Inverted index
        self.vector_counter = 0

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a vector and its metadata."""
        vector_id = f"vec_{self.vector_counter}"
        self.vector_counter += 1

        # Generate embedding (placeholder - in real use LLM/embedding model)
        embedding = self._generate_embedding(f"{entity} {relation} {value}")

        self.vectors[vector_id] = embedding

        meta = {
            "entity": entity,
            "relation": relation,
            "value": value,
            "timestamp": metadata.get("timestamp", datetime.now().isoformat())
            if metadata
            else datetime.now().isoformat(),
            "confidence": metadata.get("confidence", 1.0) if metadata else 1.0,
            **(metadata or {}),
        }
        self.metadata[vector_id] = meta

        # Update inverted index
        for word in (entity + " " + value).lower().split():
            self.index[word].append(vector_id)

        return vector_id

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant vectors using similarity search."""
        if use_embedding:
            # Use vector similarity
            query_embedding = self._generate_embedding(query)
            scores = self._cosine_similarity(query_embedding)
        else:
            # Use inverted index + TF-IDF
            scores = self._bm25_score(query)

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return top-k results
        results = []
        for vector_id in sorted_ids[:k]:
            if vector_id in self.metadata:
                result = self.metadata[vector_id].copy()
                result["score"] = scores[vector_id]
                results.append(result)

        return results

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder)."""
        # In real implementation, use an embedding model
        # This is a simple hash-based placeholder
        import hashlib

        hash_val = hashlib.md5(text.encode()).hexdigest()
        embedding = []

        for i in range(self.vector_dim):
            val = ord(hash_val[i % len(hash_val)])
            embedding.append((val - 128) / 128.0)  # Normalize to [-1, 1]

        return embedding

    def _cosine_similarity(self, query_embedding: List[float]) -> Dict[str, float]:
        """Calculate cosine similarity between query and all vectors."""
        scores = {}

        for vector_id, vector in self.vectors.items():
            dot_product = sum(q * v for q, v in zip(query_embedding, vector))
            query_norm = math.sqrt(sum(q * q for q in query_embedding))
            vector_norm = math.sqrt(sum(v * v for v in vector))

            if query_norm > 0 and vector_norm > 0:
                scores[vector_id] = dot_product / (query_norm * vector_norm)

        return scores

    def _bm25_score(self, query: str) -> Dict[str, float]:
        """Calculate BM25 score for query (simplified)."""
        scores = defaultdict(float)

        query_terms = query.lower().split()

        # Simple scoring: count matching terms
        for term in query_terms:
            if term in self.index:
                for vector_id in self.index[term]:
                    # BM25 simplified: score based on term frequency and inverse document freq
                    doc_freq = len(self.index[term])
                    total_docs = len(self.metadata)
                    idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1
                    freq = 1  # Simplified
                    scores[vector_id] += idf * freq / (freq + 1.5 + 0.75)

        return dict(scores)

    def update(
        self,
        vector_id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update a vector's metadata."""
        if vector_id not in self.metadata:
            return False

        metadata = self.metadata[vector_id]

        if entity:
            old_entity = metadata["entity"]
            self.index[old_entity].remove(vector_id)
            metadata["entity"] = entity
            self.index[entity].append(vector_id)

        if relation:
            metadata["relation"] = relation

        if value:
            metadata["value"] = value
            # Regenerate embedding for value change
            new_text = f"{metadata['entity']} {metadata['relation']} {value}"
            metadata["_cached_embedding"] = self._generate_embedding(new_text)

        return True

    def delete(self, vector_id: str) -> bool:
        """Delete a vector and its metadata."""
        if vector_id not in self.vectors:
            return False

        metadata = self.metadata.pop(vector_id)
        self.vectors.pop(vector_id)

        # Remove from inverted index
        entity = metadata.get("entity", "")
        if entity in self.index:
            self.index[entity].remove(vector_id)

        return True

    def clear(self) -> None:
        """Clear all vectors and metadata."""
        self.vectors.clear()
        self.metadata.clear()
        self.index.clear()
        self.vector_counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get vector DB statistics."""
        return {
            "total_vectors": len(self.vectors),
            "vector_dimension": self.vector_dim,
            "unique_entities": len(self.index),
            "vector_counter": self.vector_counter,
        }
