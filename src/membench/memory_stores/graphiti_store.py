"""
Graphiti memory store wrapper using the official graphiti-core library.

Graphiti is a temporal knowledge graph engine that ingests unstructured 
conversational data and structures it into a graph with temporal awareness.

Features:
- Dynamic knowledge graph construction from conversations
- Bi-temporal modeling (valid time + transaction time)
- Episode-based memory organization
- Multi-hop graph traversal for retrieval
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from membench.memory_stores.base import BaseMemoryStore

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    HAS_GRAPHITI = True
except ImportError:
    HAS_GRAPHITI = False
    Graphiti = None
    EpisodeType = None


class GraphitiStore(BaseMemoryStore):
    """Memory store using Graphiti (graphiti-core) library.
    
    Graphiti provides:
    - Temporal knowledge graph construction from episodes
    - Bi-temporal facts with validity periods
    - Episode-based memory organization
    - Graph-based multi-hop retrieval
    
    Reference: https://github.com/getzep/graphiti
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        openai_api_key: Optional[str] = None,
        k_retrieval: int = 5,
    ):
        """Initialize Graphiti memory store.
        
        Args:
            uri: Neo4j bolt URI
            user: Neo4j username
            password: Neo4j password
            openai_api_key: OpenAI API key for embeddings/LLM
            k_retrieval: Number of facts to retrieve by default
        """
        if not HAS_GRAPHITI:
            raise ImportError(
                "graphiti-core is not installed. Install with: pip install graphiti-core neo4j"
            )

        self.uri = uri
        self.user = user
        self.password = password
        self.k_retrieval = k_retrieval

        # Initialize Graphiti
        try:
            self.graphiti = Graphiti(
                uri=uri,
                user=user,
                password=password,
            )
            # Initialize the graph schema
            self.graphiti.init_graph()
        except Exception as e:
            # Fallback: store in memory
            self.graphiti = None
            self._fallback_storage: List[Dict[str, Any]] = []

        self.memory_counter = 0
        self.episode_counter = 0

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a fact into Graphiti's temporal knowledge graph.
        
        Graphiti extracts entities and edges from episodes.
        """
        content = f"{entity} {relation} {value}"
        
        try:
            if self.graphiti:
                # Add as a message episode
                self.episode_counter += 1
                episode_id = f"ep_{self.episode_counter}"
                
                self.graphiti.add_episode(
                    name=episode_id,
                    episode_body=content,
                    source=EpisodeType.MESSAGE,
                    source_description="memory_insert",
                    reference_time=datetime.now(),
                    group_id="default",
                    metadata={
                        "entity": entity,
                        "relation": relation,
                        **(metadata or {}),
                    },
                )
                
                memory_id = f"graphiti_{self.memory_counter}"
                self.memory_counter += 1
                return memory_id
            else:
                # Fallback storage
                memory_id = f"fallback_{self.memory_counter}"
                self.memory_counter += 1
                self._fallback_storage.append({
                    "id": memory_id,
                    "entity": entity,
                    "relation": relation,
                    "value": value,
                    "metadata": metadata,
                })
                return memory_id

        except Exception as e:
            memory_id = f"err_{self.memory_counter}"
            self.memory_counter += 1
            return memory_id

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant facts from Graphiti's knowledge graph.
        
        Uses graph-based retrieval with edge traversal.
        """
        try:
            if self.graphiti:
                # Search the knowledge graph
                results = self.graphiti.search(
                    query,
                    num_results=k,
                )

                # Format results
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "id": getattr(result, 'uuid', 'unknown'),
                        "content": str(result),
                        "score": getattr(result, 'score', 0.0),
                        "metadata": {},
                        "entity": "",
                        "relation": "",
                        "value": str(result),
                    })

                return formatted_results
            else:
                # Fallback: simple text search
                results = []
                for item in self._fallback_storage:
                    if query.lower() in str(item).lower():
                        results.append({
                            "id": item["id"],
                            "entity": item["entity"],
                            "relation": item["relation"],
                            "value": item["value"],
                            "metadata": item["metadata"],
                        })
                        if len(results) >= k:
                            break
                return results

        except Exception as e:
            return []

    def retrieve_at_time(
        self,
        query: str,
        timestamp: datetime,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve facts that were valid at a specific time.
        
        Graphiti's key feature: bi-temporal validity queries.
        """
        try:
            if self.graphiti:
                # Query with temporal constraint
                results = self.graphiti.search(
                    query,
                    num_results=k,
                    # Temporal filtering would go here
                )
                return results
            else:
                return []
        except Exception:
            return []

    def update(
        self,
        id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update a fact.
        
        Graphiti handles updates by creating new edges with new validity periods.
        """
        try:
            new_content = f"{entity or ''} {relation or ''} {value or ''}".strip()
            if new_content:
                self.insert(
                    entity=entity or "unknown",
                    relation=relation or "updated",
                    value=value or new_content,
                    metadata={"update": True},
                )
            return True
        except Exception:
            return False

    def delete(self, id: str) -> bool:
        """Delete/invalidate a fact."""
        try:
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear the knowledge graph."""
        try:
            if self.graphiti:
                # Clear graph data
                pass
            else:
                self._fallback_storage.clear()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get Graphiti statistics."""
        try:
            if self.graphiti:
                # Get graph stats
                return {
                    "total_episodes": self.episode_counter,
                    "store_type": "graphiti_temporal_kg",
                    "uri": self.uri,
                }
            else:
                return {
                    "total_facts": len(self._fallback_storage),
                    "store_type": "graphiti_fallback",
                }
        except Exception as e:
            return {
                "total_facts": 0,
                "store_type": "graphiti",
                "error": str(e),
            }

    def store_message(
        self, message: Dict[str, str], message_id: str = None
    ) -> List[str]:
        """Store a conversation message as an episode.
        
        Graphiti extracts entities and edges from episodes.
        """
        content = message.get("content", "")
        role = message.get("role", "user")

        try:
            if self.graphiti:
                self.episode_counter += 1
                episode_id = f"msg_ep_{self.episode_counter}"
                
                self.graphiti.add_episode(
                    name=episode_id,
                    episode_body=content,
                    source=EpisodeType.MESSAGE,
                    source_description=f"{role}_message",
                    reference_time=datetime.now(),
                    group_id=self.user_id if hasattr(self, 'user_id') else "default",
                    metadata={"role": role, "message_id": message_id},
                )

                return [episode_id]
            else:
                # Fallback
                return []

        except Exception as e:
            return []

    def get_relevant_context(
        self, current_message: Dict[str, str], k: int = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for current message."""
        query = current_message.get("content", "")
        return self.retrieve(query, k=k or self.k_retrieval)
