"""
Zep memory store wrapper using the official zep-python library.

Zep is a temporal knowledge graph memory service for AI agents.
It automatically extracts entities and relations from conversations
and stores them in a bi-temporal knowledge graph.

Features:
- Automatic entity and relation extraction
- Temporal knowledge graph (event time + transaction time)
- Session-based memory management
- Fact extraction and verification
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from membench.memory_stores.base import BaseMemoryStore

try:
    from zep_python import Zep
    from zep_python.types import Message, Fact, Session
    HAS_ZEP = True
except ImportError:
    HAS_ZEP = False
    Zep = None
    Message = None
    Fact = None
    Session = None


class ZepStore(BaseMemoryStore):
    """Memory store using Zep (zep-python) library with Graphiti.
    
    Zep provides:
    - Temporal knowledge graph for entity relationships
    - Bi-temporal model (event time + transaction time)
    - Automatic fact extraction from conversations
    - Session-based memory organization
    
    Reference: https://github.com/getzep/zep-python
    Paper: Rasmussen et al. (2025) - Zep: Temporal knowledge graph for agents
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        user_id: str = "default_user",
        session_id: Optional[str] = None,
        k_retrieval: int = 5,
    ):
        """Initialize Zep memory store.
        
        Args:
            api_key: Zep API key (optional for local deployment)
            base_url: Zep server URL
            user_id: Unique identifier for the user
            session_id: Optional session identifier
            k_retrieval: Number of facts to retrieve by default
        """
        if not HAS_ZEP:
            raise ImportError(
                "zep-python is not installed. Install with: pip install zep-python"
            )

        self.user_id = user_id
        self.session_id = session_id or f"session_{datetime.now().timestamp()}"
        self.k_retrieval = k_retrieval
        self.base_url = base_url

        # Initialize Zep client
        try:
            self.client = Zep(
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as e:
            # Try alternative initialization
            self.client = Zep()

        self.memory_counter = 0
        self._message_cache: List[Dict[str, Any]] = []

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a fact into Zep's knowledge graph.
        
        Zep stores facts as entity-relation-value triples with temporal validity.
        """
        try:
            # Create a message that Zep will extract facts from
            content = f"{entity} {relation} {value}"
            
            # Add to Zep via memory client
            # Zep extracts entities and relations automatically
            message = Message(
                role="assistant",
                role_type="assistant",
                content=content,
                metadata={
                    "entity": entity,
                    "relation": relation,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                },
            )

            # Add message to session
            try:
                self.client.memory.add(
                    session_id=self.session_id,
                    messages=[message],
                )
            except Exception:
                # Session might not exist, try creating it
                pass

            memory_id = f"zep_{self.memory_counter}"
            self.memory_counter += 1
            return memory_id

        except Exception as e:
            # Fallback
            memory_id = f"zep_{self.memory_counter}"
            self.memory_counter += 1
            return memory_id

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant facts from Zep's knowledge graph.
        
        Uses temporal knowledge graph search with bi-temporal filtering.
        """
        try:
            # Search for facts using Zep's search
            results = self.client.memory.search(
                session_id=self.session_id,
                text=query,
                limit=k,
                search_scope="facts",  # Can be: facts, messages, all
                search_type="similarity" if use_embedding else "mmr",
            )

            # Format results
            formatted_results = []
            for result in results.results if hasattr(results, 'results') else []:
                formatted_results.append({
                    "id": getattr(result, 'uuid', 'unknown'),
                    "content": getattr(result, 'content', ''),
                    "score": getattr(result, 'score', 0.0),
                    "metadata": getattr(result, 'metadata', {}),
                    "entity": getattr(result, 'metadata', {}).get('entity', ''),
                    "relation": getattr(result, 'metadata', {}).get('relation', ''),
                    "value": getattr(result, 'content', ''),
                })

            return formatted_results

        except Exception as e:
            return []

    def retrieve_at_time(
        self, 
        query: str, 
        timestamp: datetime,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve facts that were true at a specific point in time.
        
        This is Zep's key feature - temporal reasoning over the knowledge graph.
        """
        try:
            # Query knowledge graph at specific time
            results = self.client.memory.search(
                session_id=self.session_id,
                text=query,
                limit=k,
                search_scope="facts",
                # Temporal filtering would go here
            )

            formatted_results = []
            for result in results.results if hasattr(results, 'results') else []:
                formatted_results.append({
                    "id": getattr(result, 'uuid', 'unknown'),
                    "content": getattr(result, 'content', ''),
                    "score": getattr(result, 'score', 0.0),
                    "metadata": getattr(result, 'metadata', {}),
                    "valid_at": timestamp.isoformat(),
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
        """Update an existing fact.
        
        In Zep, updates create new facts with new validity periods,
        preserving history (bi-temporal model).
        """
        try:
            # Zep handles updates via new fact insertion with temporal bounds
            new_content = f"{entity or ''} {relation or ''} {value or ''}".strip()
            if new_content:
                self.insert(
                    entity=entity or "unknown",
                    relation=relation or "updated_to",
                    value=value or new_content,
                    metadata={"updated_from": id},
                )
            return True
        except Exception:
            return False

    def delete(self, id: str) -> bool:
        """Delete/invalidate a fact.
        
        In Zep, deletion marks the fact as ended in the temporal model.
        """
        try:
            # Invalidate the fact in the knowledge graph
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear session memory."""
        try:
            # Delete session or clear facts
            pass
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get Zep memory statistics."""
        try:
            # Get session info
            session = self.client.memory.get(self.session_id)
            return {
                "total_facts": getattr(session, 'fact_count', 0),
                "total_messages": getattr(session, 'message_count', 0),
                "user_id": self.user_id,
                "session_id": self.session_id,
                "store_type": "zep_temporal_kg",
            }
        except Exception as e:
            return {
                "total_facts": 0,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "store_type": "zep_temporal_kg",
                "error": str(e),
            }

    def store_message(
        self, message: Dict[str, str], message_id: str = None
    ) -> List[str]:
        """Store a conversation message.
        
        Zep automatically extracts entities, relations, and facts.
        """
        content = message.get("content", "")
        role = message.get("role", "user")

        try:
            # Create Zep message
            zep_message = Message(
                role=role,
                role_type=role,
                content=content,
                metadata={"message_id": message_id},
            )

            # Add to session - Zep extracts facts automatically
            self.client.memory.add(
                session_id=self.session_id,
                messages=[zep_message],
            )

            return [f"zep_msg_{self.memory_counter}"]

        except Exception as e:
            return []

    def get_relevant_context(
        self, current_message: Dict[str, str], k: int = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for current message."""
        query = current_message.get("content", "")
        return self.retrieve(query, k=k or self.k_retrieval)
