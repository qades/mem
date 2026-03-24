"""
Simple OpenAI-compatible API server for testing the new memory management system.
This provides a minimal implementation that mimics OpenAI's API for development/testing.
"""

import json
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any
from datetime import datetime


class SimpleMemoryStore:
    """Simple in-memory store for testing."""

    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.entities: Dict[str, List[Dict[str, Any]]] = {}
        self.memory_counter = 0

    def add_memory(
        self,
        content: str,
        vault: str = "default",
        entities: List[Dict[str, str]] = None,
        entity_relationships: List[Dict[str, str]] = None,
        summary: str = None,
        created_at: str = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Add a memory."""
        self.memory_counter += 1
        memory_id = f"mem_{self.memory_counter}"

        memory = {
            "id": memory_id,
            "content": content,
            "vault": vault,
            "entities": entities or [],
            "entity_relationships": entity_relationships or [],
            "summary": summary or content[:100],
            "created_at": created_at or datetime.now().isoformat(),
            "confidence": confidence,
        }

        self.memories.append(memory)

        # Index by entity
        for entity in entities or []:
            entity_name = entity.get("name", "")
            if entity_name not in self.entities:
                self.entities[entity_name] = []
            self.entities[entity_name].append(memory)

        return memory

    def search(
        self,
        query: str,
        vault: str = None,
        limit: int = 10,
        use_embeddings: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search memories."""
        if use_embeddings:
            # Simple embedding-based search (placeholder)
            query_hash = hashlib.md5(query.encode()).hexdigest()

            scored = []
            for mem in self.memories:
                mem_hash = hashlib.md5(mem["content"].encode()).hexdigest()
                # Simple similarity based on hash overlap
                similarity = sum(
                    1
                    for i, c in enumerate(query_hash)
                    if i < len(mem_hash) and c == mem_hash[i]
                ) / len(query_hash)
                if similarity > 0.3:
                    scored.append((mem, similarity))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [m for m, s in scored[:limit]]
        else:
            # Simple keyword search
            query_lower = query.lower()
            return [
                mem for mem in self.memories if query_lower in mem["content"].lower()
            ][:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "total_memories": len(self.memories),
            "unique_entities": len(self.entities),
            "memory_counter": self.memory_counter,
        }

    def clear(self, vault: str = None) -> int:
        """Clear memories."""
        if vault:
            count = len([m for m in self.memories if m["vault"] == vault])
            self.memories = [m for m in self.memories if m["vault"] != vault]
            return count
        else:
            count = len(self.memories)
            self.memories = []
            self.entities = {}
            return count


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible API."""

    memory_store = SimpleMemoryStore()

    def log_message(self, format, *args):
        """Suppress logging."""
        pass

    def send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self.send_json({"status": "ok"})
        elif self.path == "/stats":
            stats = self.memory_store.get_stats()
            self.send_json({"stats": stats})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        if self.path == "/chat/completions":
            self.handle_chat_completions(data)
        elif self.path == "/embeddings":
            self.handle_embeddings(data)
        elif self.path == "/memories":
            self.handle_memories(data)
        elif self.path == "/memories/update":
            self.handle_memories_update(data)
        elif self.path == "/memories/delete":
            self.handle_memories_delete(data)
        elif self.path == "/memories/clear":
            self.handle_memories_clear(data)
        elif self.path == "/search":
            self.handle_search(data)
        else:
            self.send_json({"error": "Not found"}, 404)

    def handle_chat_completions(self, data: Dict[str, Any]):
        """Handle chat completions endpoint."""
        messages = data.get("messages", [])

        # Extract user message for parsing
        user_message = next((m for m in messages if m.get("role") == "user"), {})
        content = user_message.get("content", "")

        # Return mock parsed result
        response = {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": data.get("model", "gpt-3.5-turbo"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "entities": self._extract_entities(content),
                                "relations": [],
                                "facts": [content[:200]] if content else [],
                                "preferences": [],
                                "timestamps": [],
                                "confidence": 0.8,
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(content.split()),
                "completion_tokens": 50,
                "total_tokens": len(content.split()) + 50,
            },
        }

        self.send_json(response)

    def handle_embeddings(self, data: Dict[str, Any]):
        """Handle embeddings endpoint."""
        input_text = data.get("input", "")

        # Generate simple embedding
        hash_val = hashlib.md5(input_text.encode()).hexdigest()
        embedding = [ord(hash_val[i % len(hash_val)]) / 255.0 - 0.5 for i in range(384)]

        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": 0,
                }
            ],
            "model": data.get("model", "text-embedding-3-small"),
            "usage": {
                "prompt_tokens": len(input_text.split()),
                "total_tokens": len(input_text.split()),
            },
        }

        self.send_json(response)

    def handle_memories(self, data: Dict[str, Any]):
        """Handle memories endpoint."""
        memory = self.memory_store.add_memory(
            content=data.get("content", ""),
            vault=data.get("vault", "default"),
            entities=data.get("entities", []),
            entity_relationships=data.get("entity_relationships", []),
            summary=data.get("summary"),
            created_at=data.get("created_at"),
            confidence=data.get("confidence", 1.0),
        )

        self.send_json({"success": True, "id": memory["id"]})

    def handle_memories_update(self, data: Dict[str, Any]):
        """Handle memories update endpoint."""
        memory_id = data.get("id")

        for mem in self.memory_store.memories:
            if mem["id"] == memory_id:
                if data.get("entity"):
                    mem["entity"] = data["entity"]
                if data.get("relation"):
                    mem["relation"] = data["relation"]
                if data.get("value"):
                    mem["value"] = data["value"]

                self.send_json({"success": True})
                return

        self.send_json({"success": False, "error": "Memory not found"}, 404)

    def handle_memories_delete(self, data: Dict[str, Any]):
        """Handle memories delete endpoint."""
        memory_id = data.get("id")

        for i, mem in enumerate(self.memory_store.memories):
            if mem["id"] == memory_id:
                self.memory_store.memories.pop(i)
                self.send_json({"success": True})
                return

        self.send_json({"success": False, "error": "Memory not found"}, 404)

    def handle_memories_clear(self, data: Dict[str, Any]):
        """Handle memories clear endpoint."""
        vault = data.get("vault")
        count = self.memory_store.clear(vault)
        self.send_json({"success": True, "cleared": count})

    def handle_search(self, data: Dict[str, Any]):
        """Handle search endpoint."""
        results = self.memory_store.search(
            query=data.get("query", ""),
            vault=data.get("vault"),
            limit=data.get("limit", 10),
            use_embeddings=data.get("use_embeddings", True),
        )

        self.send_json(
            {
                "success": True,
                "results": results,
                "count": len(results),
            }
        )

    def _extract_entities(self, content: str) -> List[str]:
        """Extract entities from content."""
        words = content.split()
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and word.isalpha():
                if i < len(words) - 1 and words[i + 1][0].isupper():
                    entities.append(f"{word} {words[i + 1]}")
                else:
                    entities.append(word)
        return list(set(entities))[:10]


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the OpenAI-compatible server."""
    server = HTTPServer((host, port), OpenAICompatibleHandler)
    print(f"OpenAI-compatible server running on http://{host}:{port}")
    print("Available endpoints:")
    print("  GET  /health")
    print("  POST /chat/completions")
    print("  POST /embeddings")
    print("  POST /memories")
    print("  POST /memories/update")
    print("  POST /memories/delete")
    print("  POST /memories/clear")
    print("  POST /search")
    print("  GET  /stats")
    print()
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI-compatible test server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()
    run_server(args.host, args.port)
