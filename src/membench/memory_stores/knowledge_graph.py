"""
Knowledge Graph-based memory store.
"""

import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

from membench.memory_stores.base import BaseMemoryStore


class KnowledgeGraphStore(BaseMemoryStore):
    """Memory store using knowledge graph structure."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, str]] = []
        self.node_counter = 0
        self.entity_index: Dict[str, List[str]] = defaultdict(list)

    def insert(
        self,
        entity: str,
        relation: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a node and edge into the knowledge graph."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        node = {
            "id": node_id,
            "type": "info",
            "entity": entity,
            "relation": relation,
            "value": value,
            "timestamp": metadata.get("timestamp", datetime.now().isoformat())
            if metadata
            else datetime.now().isoformat(),
            "confidence": metadata.get("confidence", 1.0) if metadata else 1.0,
            **(metadata or {}),
        }

        self.nodes[node_id] = node
        self.entity_index[entity].append(node_id)

        # Create edge from entity to value
        edge = {
            "from": entity,
            "relation": relation,
            "to": value,
            "metadata": metadata or {},
        }
        self.edges.append(edge)

        return node_id

    def retrieve(
        self, query: str, k: int = 5, use_embedding: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant information using graph traversal and scoring."""
        results = []

        # Parse query to extract entities
        query_entities = self._extract_query_entities(query)

        # Find nodes matching query entities
        for entity in query_entities:
            node_ids = self._get_node_ids_for_entity(entity)
            for node_id in node_ids[:k]:
                if node_id in self.nodes:
                    results.append(self.nodes[node_id])

        # If not enough results, expand to related nodes
        if len(results) < k:
            results.extend(
                self._retrieve_related_nodes(query_entities, k - len(results))
            )

        # Sort by confidence and recency
        results.sort(
            key=lambda x: (x.get("confidence", 0), x.get("timestamp", "")), reverse=True
        )

        return results[:k]

    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query (simplified)."""
        # In real implementation, use NER or LLM
        words = query.lower().split()
        return list(set([w for w in words if len(w) > 3]))[:5]

    def _get_node_ids_for_entity(self, entity: str) -> List[str]:
        """Get node IDs for an entity (case-insensitive)."""
        entity_lower = entity.lower()
        for key in self.entity_index.keys():
            if key.lower() == entity_lower:
                return self.entity_index[key]
        return []

    def _retrieve_related_nodes(
        self, entities: List[str], k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve nodes related to given entities."""
        related = []

        for node_id, node in self.nodes.items():
            if node.get("entity") in entities:
                related.append(node)
                if len(related) >= k:
                    break

        return related[:k]

    def update(
        self,
        node_id: str,
        entity: Optional[str] = None,
        relation: Optional[str] = None,
        value: Optional[str] = None,
    ) -> bool:
        """Update a node in the knowledge graph."""
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]

        if entity:
            # Update entity index
            old_entity = node["entity"]
            self.entity_index[old_entity].remove(node_id)
            self.entity_index[entity].append(node_id)
            node["entity"] = entity

        if relation:
            node["relation"] = relation

        if value:
            node["value"] = value

        return True

    def delete(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        if node_id not in self.nodes:
            return False

        node = self.nodes.pop(node_id)

        # Update entity index
        entity = node.get("entity", "")
        if entity in self.entity_index:
            self.entity_index[entity].remove(node_id)

        # Remove related edges
        self.edges = [
            e
            for e in self.edges
            if e.get("from") != entity and e.get("to") != node.get("value", "")
        ]

        return True

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()
        self.entity_index.clear()
        self.node_counter = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "unique_entities": len(self.entity_index),
            "node_counter": self.node_counter,
        }

    def export_graph(self) -> Dict[str, Any]:
        """Export the knowledge graph as JSON."""
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "stats": self.get_stats(),
        }
