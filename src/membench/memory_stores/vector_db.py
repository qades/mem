"""
Vector DB-based memory store with support for multiple backends (ChromaDB, FAISS, in-memory).
"""

import json
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

from membench.memory_stores.base import BaseMemoryStore


class VectorDBStore(BaseMemoryStore):
    """Memory store using vector embeddings for retrieval with multiple backend support."""

    def __init__(
        self,
        store_type: str = "in_memory",
        collection_name: str = "memory",
        dimension: int = 768,
        metric: str = "cosine",
        api_url: str = None,
        api_key: str = None,
    ):
        self.store_type = store_type
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric = metric
        self.api_url = api_url
        self.api_key = api_key

        # Initialize backend-specific storage
        if store_type == "in_memory":
            self.vectors: Dict[str, List[float]] = {}
            self.metadata: Dict[str, Dict[str, Any]] = {}
            self.vector_dim: int = dimension
            self.index: Dict[str, List[str]] = defaultdict(list)
            self.vector_counter = 0
        elif store_type == "chromadb":
            self._init_chromadb()
        elif store_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unknown store type: {store_type}")

    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.Client(
                Settings(
                    chroma_api_token=self.api_key,
                    chroma_server_host=self.api_url.replace("http://", "").replace(
                        "/v1", ""
                    )
                    if self.api_url
                    else None,
                )
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": self.metric}
            )
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Install with: pip install chromadb"
            )

    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            import numpy as np

            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = {}
            self.id_map = {}
            self.counter = 0
        except ImportError:
            raise ImportError("faiss is not installed. Install with: pip install faiss")

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a vector and its metadata."""
        if self.store_type == "in_memory":
            return self._insert_in_memory(entity, relation, value, metadata)
        elif self.store_type == "chromadb":
            return self._insert_chromadb(entity, relation, value, metadata)
        elif self.store_type == "faiss":
            return self._insert_faiss(entity, relation, value, metadata)
        else:
            raise ValueError(f"Unknown store type: {self.store_type}")

    def _insert_in_memory(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert into in-memory vector store."""
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

    def _insert_chromadb(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert into ChromaDB."""
        import numpy as np

        content = f"{entity} {relation} {value}"
        embedding = self._generate_embedding(content)

        vector_id = f"vec_{self.counter}"
        self.counter += 1

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

        self.collection.add(
            ids=[vector_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[content],
        )

        return vector_id

    def _insert_faiss(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert into FAISS."""
        import numpy as np

        content = f"{entity} {relation} {value}"
        embedding = self._generate_embedding(content)

        vector_id = f"vec_{self.counter}"
        self.counter += 1

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
        self.id_map[self.counter] = vector_id
        self.index.add(np.array([embedding], dtype=np.float32))

        return vector_id

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant vectors using similarity search."""
        if self.store_type == "in_memory":
            return self._retrieve_in_memory(query, k, use_embedding)
        elif self.store_type == "chromadb":
            return self._retrieve_chromadb(query, k, use_embedding)
        elif self.store_type == "faiss":
            return self._retrieve_faiss(query, k, use_embedding)
        else:
            raise ValueError(f"Unknown store type: {self.store_type}")

    def _retrieve_in_memory(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve from in-memory vector store."""
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

    def _retrieve_chromadb(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve from ChromaDB."""
        if use_embedding:
            query_embedding = self._generate_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
            )

        return self._format_chromadb_results(results)

    def _retrieve_faiss(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve from FAISS."""
        import numpy as np

        if use_embedding:
            query_embedding = self._generate_embedding(query)
            D, I = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        else:
            # Fallback to keyword search
            scores = self._bm25_score(query)
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
                :k
            ]
            return [self.metadata[vid] for vid in sorted_ids if vid in self.metadata]

        results = []
        for i in range(min(k, len(I[0]))):
            idx = I[0][i]
            if idx in self.id_map:
                vector_id = self.id_map[idx]
                if vector_id in self.metadata:
                    result = self.metadata[vector_id].copy()
                    result["score"] = float(D[0][i])
                    results.append(result)

        return results

    def _format_chromadb_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results."""
        formatted = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, (id_, meta, dist) in enumerate(zip(ids, metadatas, distances)):
            item = meta.copy() if meta else {}
            item["id"] = id_
            item["score"] = (
                1.0 - dist if isinstance(dist, (int, float)) else 0.5
            )  # Convert distance to similarity
            formatted.append(item)

        return formatted

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
        if self.store_type == "in_memory":
            return self._update_in_memory(vector_id, entity, relation, value)
        elif self.store_type == "chromadb":
            return self._update_chromadb(vector_id, entity, relation, value)
        elif self.store_type == "faiss":
            return self._update_faiss(vector_id, entity, relation, value)
        else:
            raise ValueError(f"Unknown store type: {self.store_type}")

    def _update_in_memory(
        self,
        vector_id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update in-memory vector."""
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

    def _update_chromadb(
        self,
        vector_id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update ChromaDB vector."""
        if vector_id not in self.collection.get(ids=[vector_id])["ids"]:
            return False

        metadata = self.collection.get(ids=[vector_id])["metadatas"][0]

        if entity:
            metadata["entity"] = entity
        if relation:
            metadata["relation"] = relation
        if value:
            metadata["value"] = value

        self.collection.update(
            ids=[vector_id],
            metadatas=[metadata],
        )

        return True

    def _update_faiss(
        self,
        vector_id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update FAISS vector."""
        if vector_id not in self.metadata:
            return False

        metadata = self.metadata[vector_id]

        if entity:
            metadata["entity"] = entity
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
        if self.store_type == "in_memory":
            return self._delete_in_memory(vector_id)
        elif self.store_type == "chromadb":
            return self._delete_chromadb(vector_id)
        elif self.store_type == "faiss":
            return self._delete_faiss(vector_id)
        else:
            raise ValueError(f"Unknown store type: {self.store_type}")

    def _delete_in_memory(self, vector_id: str) -> bool:
        """Delete from in-memory vector store."""
        if vector_id not in self.vectors:
            return False

        metadata = self.metadata.pop(vector_id)
        self.vectors.pop(vector_id)

        # Remove from inverted index
        entity = metadata.get("entity", "")
        if entity in self.index:
            self.index[entity].remove(vector_id)

        return True

    def _delete_chromadb(self, vector_id: str) -> bool:
        """Delete from ChromaDB."""
        try:
            self.collection.delete(ids=[vector_id])
            return True
        except Exception:
            return False

    def _delete_faiss(self, vector_id: str) -> bool:
        """Delete from FAISS."""
        if vector_id not in self.metadata:
            return False

        # FAISS doesn't support deletion, mark as deleted
        self.metadata[vector_id]["_deleted"] = True
        return True

    def clear(self) -> None:
        """Clear all vectors and metadata."""
        if self.store_type == "in_memory":
            self.vectors.clear()
            self.metadata.clear()
            self.index.clear()
            self.vector_counter = 0
        elif self.store_type == "chromadb":
            self.client.reset()
        elif self.store_type == "faiss":
            self.index.reset()
            self.metadata.clear()
            self.id_map.clear()
            self.counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get vector DB statistics."""
        if self.store_type == "in_memory":
            return {
                "total_vectors": len(self.vectors),
                "vector_dimension": self.vector_dim,
                "unique_entities": len(self.index),
                "vector_counter": self.vector_counter,
                "store_type": "in_memory",
            }
        elif self.store_type == "chromadb":
            try:
                count = self.collection.count()
                return {
                    "total_vectors": count,
                    "vector_dimension": self.dimension,
                    "collection_name": self.collection_name,
                    "store_type": "chromadb",
                }
            except Exception:
                return {
                    "error": "Could not get stats",
                    "store_type": "chromadb",
                }
        elif self.store_type == "faiss":
            return {
                "total_vectors": len(self.metadata),
                "vector_dimension": self.dimension,
                "store_type": "faiss",
            }
        return {"error": "Unknown store type", "store_type": self.store_type}
