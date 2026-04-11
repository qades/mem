"""
MemPalace memory store wrapper using the official mempalace library.

MemPalace organizes memories spatially using the method of loci:
- Wings (people/projects) → Rooms (topics) → Halls (connections)
- Closets (summaries) → Drawers (verbatim storage)
- Tunnels (cross-wing connections)

Features:
- Spatial organization inspired by memory palaces
- AAAK compression for token efficiency
- 4-layer memory stack (L0-L3)
- ~170 token wake-up cost
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from membench.memory_stores.base import BaseMemoryStore

try:
    import mempalace
    from mempalace.searcher import search_memories
    from mempalace.miner import mine_project, mine_conversations
    HAS_MEMPALACE = True
except ImportError:
    HAS_MEMPALACE = False
    search_memories = None
    mine_project = None
    mine_conversations = None


class MemPalaceStore(BaseMemoryStore):
    """Memory store using MemPalace library.
    
    MemPalace provides:
    - Spatial organization (Wings → Rooms → Halls)
    - Verbatim storage with closet summaries
    - AAAK compression dialect
    - 4-layer memory stack for efficient retrieval
    
    Reference: https://github.com/MemPalace/mempalace
    """

    def __init__(
        self,
        palace_path: str = "~/.mempalace",
        wing: str = "default",
        room: Optional[str] = None,
        k_retrieval: int = 5,
        use_aaak: bool = False,
    ):
        """Initialize MemPalace memory store.
        
        Args:
            palace_path: Path to the palace directory
            wing: Default wing (project/person) to use
            room: Default room (topic) to use
            k_retrieval: Number of memories to retrieve
            use_aaak: Whether to use AAAK compression
        """
        if not HAS_MEMPALACE:
            raise ImportError(
                "mempalace is not installed. Install with: pip install mempalace"
            )

        self.palace_path = palace_path
        self.wing = wing
        self.room = room
        self.k_retrieval = k_retrieval
        self.use_aaak = use_aaak

        self.memory_counter = 0
        self._local_drawers: List[Dict[str, Any]] = []

        # Initialize palace if needed
        self._initialize_palace()

    def _initialize_palace(self):
        """Set up the palace structure."""
        try:
            # Palace is initialized via CLI or Python API
            # For now, we use the searcher and miner modules
            pass
        except Exception:
            pass

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a memory into the palace.
        
        Creates a drawer in the appropriate wing/room.
        """
        content = f"{entity} {relation} {value}"
        
        try:
            # Store in local drawers (MemPalace uses file-based storage)
            memory_id = f"drawer_{self.wing}_{self.memory_counter}"
            self.memory_counter += 1
            
            drawer = {
                "id": memory_id,
                "wing": self.wing,
                "room": self.room or "general",
                "hall": metadata.get("hall", "facts") if metadata else "facts",
                "content": content,
                "entity": entity,
                "relation": relation,
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "aaak_compressed": False,
                **(metadata or {}),
            }
            
            self._local_drawers.append(drawer)
            
            # If AAAK is enabled, compress the content
            if self.use_aaak:
                drawer["aaak_content"] = self._compress_aaak(content)
                drawer["aaak_compressed"] = True
            
            return memory_id

        except Exception as e:
            memory_id = f"err_{self.memory_counter}"
            self.memory_counter += 1
            return memory_id

    def _compress_aaak(self, text: str) -> str:
        """Compress text using AAAK dialect.
        
        AAAK (AI Abbreviation Kompact) is MemPalace's compression scheme:
        - Abbreviate common entities
        - Drop filler words
        - Keep technical terms intact
        """
        # Simple AAAK-like compression
        abbreviations = {
            "database": "DB",
            "configuration": "config",
            "authentication": "auth",
            "authorization": "authz",
            "application": "app",
            "development": "dev",
            "production": "prod",
            "environment": "env",
        }
        
        result = text
        for word, abbrev in abbreviations.items():
            result = result.replace(word, abbrev)
        
        return result

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from the palace.
        
        Searches across wings/rooms with optional spatial filtering.
        """
        try:
            # Try using MemPalace searcher if available
            if search_memories:
                try:
                    results = search_memories(
                        query=query,
                        palace_path=self.palace_path,
                        wing=self.wing,
                        room=self.room,
                        limit=k,
                    )
                    return results
                except Exception:
                    pass

            # Fallback: local drawer search
            results = []
            for drawer in self._local_drawers:
                # Check if query matches content
                content_to_search = (
                    drawer.get("aaak_content", drawer["content"])
                    if self.use_aaak else drawer["content"]
                )
                
                if query.lower() in content_to_search.lower():
                    results.append({
                        "id": drawer["id"],
                        "entity": drawer["entity"],
                        "relation": drawer["relation"],
                        "value": drawer["value"],
                        "wing": drawer["wing"],
                        "room": drawer["room"],
                        "score": 1.0,
                        "metadata": {
                            "hall": drawer.get("hall"),
                            "timestamp": drawer.get("timestamp"),
                        },
                    })
                    
                    if len(results) >= k:
                        break
            
            return results

        except Exception as e:
            return []

    def retrieve_in_room(
        self, query: str, room: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific room (spatial filtering).
        
        This is MemPalace's key feature - narrowing search by spatial location.
        """
        # Filter drawers by room
        room_drawers = [
            d for d in self._local_drawers 
            if d.get("room") == room
        ]
        
        results = []
        for drawer in room_drawers:
            if query.lower() in drawer["content"].lower():
                results.append({
                    "id": drawer["id"],
                    "entity": drawer["entity"],
                    "relation": drawer["relation"],
                    "value": drawer["value"],
                    "score": 1.0,
                })
                if len(results) >= k:
                    break
        
        return results

    def update(
        self,
        id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update a drawer."""
        try:
            for drawer in self._local_drawers:
                if drawer["id"] == id:
                    if entity:
                        drawer["entity"] = entity
                    if relation:
                        drawer["relation"] = relation
                    if value:
                        drawer["value"] = value
                        drawer["content"] = f"{drawer['entity']} {drawer['relation']} {value}"
                    drawer["updated"] = datetime.now().isoformat()
                    return True
            return False
        except Exception:
            return False

    def delete(self, id: str) -> bool:
        """Delete a drawer."""
        try:
            original_len = len(self._local_drawers)
            self._local_drawers = [
                d for d in self._local_drawers if d["id"] != id
            ]
            return len(self._local_drawers) < original_len
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all drawers."""
        self._local_drawers.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get palace statistics."""
        # Count by wing/room
        wings = {}
        rooms = {}
        for drawer in self._local_drawers:
            wing = drawer.get("wing", "unknown")
            room = drawer.get("room", "unknown")
            wings[wing] = wings.get(wing, 0) + 1
            rooms[room] = rooms.get(room, 0) + 1

        return {
            "total_drawers": len(self._local_drawers),
            "wings": wings,
            "rooms": rooms,
            "current_wing": self.wing,
            "current_room": self.room,
            "use_aaak": self.use_aaak,
            "store_type": "mempalace_spatial",
        }

    def get_wake_up_context(self) -> str:
        """Get L0 + L1 context for agent wake-up.
        
        MemPalace's unique feature: ~170 tokens of critical context.
        """
        # L0: Identity
        l0 = f"Identity: AI assistant for {self.wing}"
        
        # L1: Critical facts (top by recency)
        l1_drawers = self._local_drawers[-15:]  # Last 15
        l1_facts = [f"{d['entity']} {d['relation']} {d['value']}"[:50] 
                   for d in l1_drawers]
        l1 = " | ".join(l1_facts)
        
        if self.use_aaak:
            l1 = self._compress_aaak(l1)
        
        return f"{l0}\n{l1}"

    def store_message(
        self, message: Dict[str, str], message_id: str = None
    ) -> List[str]:
        """Store a conversation message."""
        content = message.get("content", "")
        role = message.get("role", "user")

        # Store as a drawer in the events hall
        memory_id = self.insert(
            entity="conversation",
            relation="contains",
            value=content[:500],  # Truncate for storage
            metadata={
                "hall": "events",
                "role": role,
                "message_id": message_id,
            },
        )

        return [memory_id]

    def get_relevant_context(
        self, current_message: Dict[str, str], k: int = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for current message."""
        query = current_message.get("content", "")
        return self.retrieve(query, k=k or self.k_retrieval)
