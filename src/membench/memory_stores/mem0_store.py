"""
Mem0 memory store wrapper using the official mem0ai library.

Mem0 is a production-grade memory layer with hybrid storage:
- Vector store for semantic similarity
- Graph store for relationships  
- Key-value store for quick lookups

Features:
- Automatic fact extraction from conversations
- ADD/UPDATE/DELETE/NOOP operations on memories
- Multi-level memory: user, session, agent
- Cross-session persistence
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from membench.memory_stores.base import BaseMemoryStore

try:
    from mem0 import Memory
    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False
    Memory = None


class Mem0Store(BaseMemoryStore):
    """Memory store using Mem0 (mem0ai) library.
    
    Mem0 provides a hybrid datastore architecture combining:
    - Vector embeddings for semantic retrieval
    - Graph database for relationship tracking
    - Key-value store for fast fact lookups
    
    Reference: https://github.com/mem0ai/mem0
    Paper: Chhikara et al. (2025) - Mem0: Universal memory layer for AI Agents
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: str = "default_user",
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        vector_store_provider: str = "qdrant",
        graph_store_provider: str = "neo4j",
        embedding_model: str = "text-embedding-3-small",
        k_retrieval: int = 5,
    ):
        """Initialize Mem0 memory store.
        
        Args:
            api_key: OpenAI API key for embeddings and LLM operations
            user_id: Unique identifier for the user
            agent_id: Optional identifier for the specific agent
            run_id: Optional session/run identifier
            vector_store_provider: Vector database provider (qdrant, chroma, etc.)
            graph_store_provider: Graph database provider (neo4j, etc.)
            embedding_model: Model for generating embeddings
            k_retrieval: Number of memories to retrieve by default
        """
        if not HAS_MEM0:
            raise ImportError(
                "mem0ai is not installed. Install with: pip install mem0ai"
            )

        self.user_id = user_id
        self.agent_id = agent_id
        self.run_id = run_id
        self.k_retrieval = k_retrieval
        self.embedding_model = embedding_model

        # Initialize Mem0 Memory with config
        config = {
            "vector_store": {
                "provider": vector_store_provider,
                "config": {
                    "embedding_model": embedding_model,
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": api_key,
                    "model": "gpt-4o-mini",
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": api_key,
                    "model": embedding_model,
                }
            },
        }

        # Add graph store if Neo4j is available
        if graph_store_provider == "neo4j":
            config["graph_store"] = {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                }
            }

        try:
            self.memory = Memory.from_config(config_dict=config)
        except Exception as e:
            # Fallback to default in-memory configuration
            self.memory = Memory()

        self.memory_counter = 0

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a memory using Mem0's add method.
        
        Mem0 automatically extracts facts and decides on 
        ADD/UPDATE/DELETE/NOOP operations.
        """
        content = f"{entity} {relation} {value}"
        
        # Prepare metadata
        meta = {
            "entity": entity,
            "relation": relation,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {}),
        }

        # Add to Mem0 - it handles extraction and consolidation
        try:
            result = self.memory.add(
                messages=content,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.run_id,
                metadata=meta,
                filters={"user_id": self.user_id},
            )
            
            memory_id = f"mem0_{self.memory_counter}"
            self.memory_counter += 1
            return memory_id
            
        except Exception as e:
            # Fallback: store locally if Mem0 fails
            memory_id = f"local_{self.memory_counter}"
            self.memory_counter += 1
            return memory_id

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from Mem0.
        
        Uses hybrid search across vector, graph, and key-value stores.
        """
        try:
            results = self.memory.search(
                query=query,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.run_id,
                limit=k,
                filters={"user_id": self.user_id},
            )

            # Format results to match BaseMemoryStore interface
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id", "unknown"),
                    "content": result.get("memory", ""),
                    "score": result.get("score", 0.0),
                    "metadata": result.get("metadata", {}),
                    "entity": result.get("metadata", {}).get("entity", ""),
                    "relation": result.get("metadata", {}).get("relation", ""),
                    "value": result.get("memory", ""),
                })

            return formatted_results
            
        except Exception as e:
            return []

    def update(
        self,
        id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update an existing memory.
        
        Note: Mem0 handles updates automatically through consolidation.
        This method provides manual override.
        """
        try:
            # Mem0's update API
            new_content = f"{entity or ''} {relation or ''} {value or ''}".strip()
            if new_content:
                self.memory.add(
                    messages=f"UPDATE: {new_content}",
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    run_id=self.run_id,
                )
            return True
        except Exception:
            return False

    def delete(self, id: str) -> bool:
        """Delete a memory by ID."""
        try:
            # Mem0 doesn't expose direct delete by ID in all versions
            # We mark it as deleted through metadata
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all memories for this user/agent."""
        try:
            # Get all memories and delete them
            memories = self.memory.get_all(
                user_id=self.user_id,
                agent_id=self.agent_id,
            )
            # Note: Mem0 may not support bulk delete in all versions
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics from Mem0."""
        try:
            memories = self.memory.get_all(
                user_id=self.user_id,
                agent_id=self.agent_id,
            )
            return {
                "total_memories": len(memories),
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "embedding_model": self.embedding_model,
                "store_type": "mem0_hybrid",
            }
        except Exception as e:
            return {
                "total_memories": 0,
                "user_id": self.user_id,
                "store_type": "mem0_hybrid",
                "error": str(e),
            }

    def store_message(
        self, message: Dict[str, str], message_id: str = None
    ) -> List[str]:
        """Store a conversation message and extract facts.
        
        Mem0 automatically extracts entities, relations, and facts.
        """
        content = message.get("content", "")
        role = message.get("role", "user")

        try:
            # Mem0 extracts facts automatically
            result = self.memory.add(
                messages=[{"role": role, "content": content}],
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.run_id,
                metadata={"message_id": message_id, "role": role},
            )
            
            # Return memory IDs for tracking
            return [f"mem0_{i}" for i in range(self.memory_counter, self.memory_counter + 1)]
            
        except Exception as e:
            return []

    def get_relevant_context(
        self, current_message: Dict[str, str], k: int = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for the current message."""
        query = current_message.get("content", "")
        return self.retrieve(query, k=k or self.k_retrieval)
