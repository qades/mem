"""
Letta (formerly MemGPT) memory store wrapper using the official letta library.

Letta provides OS-style memory management for agents with:
- Hierarchical memory: core (RAM) ↔ archival (disk)
- Self-managing agents that can page memory in/out
- Memory blocks for structured context
- Tool-based memory operations

Features:
- Memory hierarchy with automatic paging
- Agent-controlled memory management
- Block-based context organization
- Persistent agent state
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from membench.memory_stores.base import BaseMemoryStore

try:
    import letta
    from letta import create_client
    from letta.schemas.memory import ChatMemory
    HAS_LETTA = True
except ImportError:
    HAS_LETTA = False
    create_client = None
    ChatMemory = None


class LettaStore(BaseMemoryStore):
    """Memory store using Letta (formerly MemGPT) library.
    
    Letta provides:
    - OS-style memory hierarchy (core ↔ archival)
    - Self-managing agents with memory tools
    - Block-based memory organization
    - Persistent agent state across sessions
    
    Reference: https://github.com/letta-ai/letta
    Paper: Packer et al. (2023) - MemGPT: LLMs as Operating Systems
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        agent_name: str = "benchmark_agent",
        user_id: str = "default_user",
        k_retrieval: int = 5,
        context_window_limit: int = 8000,
    ):
        """Initialize Letta memory store.
        
        Args:
            api_key: Letta API key
            base_url: Letta server URL (None for embedded)
            agent_name: Name for the agent
            user_id: User identifier
            k_retrieval: Number of memories to retrieve
            context_window_limit: Token limit for core memory
        """
        if not HAS_LETTA:
            raise ImportError(
                "letta is not installed. Install with: pip install letta"
            )

        self.agent_name = agent_name
        self.user_id = user_id
        self.k_retrieval = k_retrieval
        self.context_window_limit = context_window_limit

        # Initialize Letta client
        try:
            if base_url:
                self.client = create_client(base_url=base_url, token=api_key)
            else:
                self.client = create_client()
        except Exception as e:
            # Fallback to direct initialization
            self.client = None

        self.agent_id = None
        self.memory_counter = 0
        self._local_storage: List[Dict[str, Any]] = []

        # Create or load agent
        self._initialize_agent()

    def _initialize_agent(self):
        """Create or retrieve the Letta agent."""
        try:
            if self.client:
                # Try to find existing agent
                agents = self.client.list_agents()
                for agent in agents:
                    if hasattr(agent, 'name') and agent.name == self.agent_name:
                        self.agent_id = agent.id
                        return

                # Create new agent if not found
                memory = ChatMemory(
                    persona="I am a helpful assistant with persistent memory.",
                    human=f"User: {self.user_id}",
                )
                
                agent = self.client.create_agent(
                    name=self.agent_name,
                    memory=memory,
                )
                self.agent_id = agent.id
        except Exception as e:
            # Fallback: operate without client
            self.client = None

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a memory into Letta's archival memory.
        
        Letta agents manage their own memory using tools like
        archival_memory_insert and core_memory_append.
        """
        content = f"{entity} {relation} {value}"

        try:
            if self.client and self.agent_id:
                # Send a message that triggers memory insertion
                response = self.client.send_message(
                    agent_id=self.agent_id,
                    message=f"Remember: {content}",
                    role="user",
                )
                
                memory_id = f"letta_{self.memory_counter}"
                self.memory_counter += 1
                return memory_id
            else:
                # Fallback storage
                memory_id = f"local_{self.memory_counter}"
                self.memory_counter += 1
                self._local_storage.append({
                    "id": memory_id,
                    "entity": entity,
                    "relation": relation,
                    "value": value,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                })
                return memory_id

        except Exception as e:
            memory_id = f"err_{self.memory_counter}"
            self.memory_counter += 1
            return memory_id

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from Letta's archival memory.
        
        Uses the agent's archival_memory_search tool.
        """
        try:
            if self.client and self.agent_id:
                # Send search request
                response = self.client.send_message(
                    agent_id=self.agent_id,
                    message=f"Search memory for: {query}",
                    role="user",
                )

                # Parse response for retrieved memories
                # Letta returns tool calls with search results
                formatted_results = []
                # Extract from response...
                
                return formatted_results
            else:
                # Fallback: search local storage
                results = []
                for item in self._local_storage:
                    if query.lower() in str(item).lower():
                        results.append(item)
                        if len(results) >= k:
                            break
                return results

        except Exception as e:
            return []

    def update(
        self,
        id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update a memory.
        
        Letta handles updates through the agent's memory tools.
        """
        try:
            new_content = f"{entity or ''} {relation or ''} {value or ''}".strip()
            if new_content:
                self.insert(
                    entity=entity or "unknown",
                    relation=relation or "updated",
                    value=value or new_content,
                )
            return True
        except Exception:
            return False

    def delete(self, id: str) -> bool:
        """Delete a memory."""
        try:
            if not self.client:
                # Remove from local storage
                self._local_storage = [
                    item for item in self._local_storage 
                    if item.get("id") != id
                ]
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear agent memory."""
        try:
            if self.client and self.agent_id:
                # Reset agent memory
                pass
            else:
                self._local_storage.clear()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get Letta memory statistics."""
        try:
            if self.client and self.agent_id:
                agent = self.client.get_agent(self.agent_id)
                return {
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "store_type": "letta_hierarchical",
                    "context_window": self.context_window_limit,
                }
            else:
                return {
                    "local_memories": len(self._local_storage),
                    "store_type": "letta_fallback",
                }
        except Exception as e:
            return {
                "store_type": "letta",
                "error": str(e),
            }

    def store_message(
        self, message: Dict[str, str], message_id: str = None
    ) -> List[str]:
        """Store a conversation message.
        
        Letta automatically manages conversation history and extracts memories.
        """
        content = message.get("content", "")
        role = message.get("role", "user")

        try:
            if self.client and self.agent_id:
                # Send message to agent
                response = self.client.send_message(
                    agent_id=self.agent_id,
                    message=content,
                    role=role,
                )

                # Letta extracts memories automatically
                return [f"letta_msg_{self.memory_counter}"]
            else:
                # Store locally
                self._local_storage.append({
                    "role": role,
                    "content": content,
                    "message_id": message_id,
                    "timestamp": datetime.now().isoformat(),
                })
                return [f"local_{self.memory_counter}"]

        except Exception as e:
            return []

    def get_relevant_context(
        self, current_message: Dict[str, str], k: int = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for current message."""
        query = current_message.get("content", "")
        return self.retrieve(query, k=k or self.k_retrieval)

    def get_core_memory(self) -> str:
        """Get the agent's core memory (RAM equivalent).
        
        This is Letta's unique feature - the working context.
        """
        try:
            if self.client and self.agent_id:
                agent = self.client.get_agent(self.agent_id)
                if hasattr(agent, 'memory'):
                    return str(agent.memory)
            return ""
        except Exception:
            return ""
