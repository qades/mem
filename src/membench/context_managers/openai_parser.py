"""
OpenAI-compatible API context parser for fast utility LLM that parses context from messages.
NOTE: This is NOT a memory store - it only parses and extracts structured information.
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from membench.context_managers.base import BaseContextManager

try:
    import requests
except ImportError:
    requests = None


class OpenAIContextParser:
    """Parse messages using OpenAI-compatible API for fast context extraction.

    This class does NOT store memory - it only extracts structured information
    from messages for use by other components.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:58080/v1",
        api_key: str = None,
        model: str = "LFM2.5-1.2B-Instruct-Q8_0",
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def parse_message(self, content: str, role: str = "user") -> Dict[str, Any]:
        """Parse a message and extract structured information."""
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_parser_system_prompt(),
                        },
                        {"role": role, "content": content},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            content = (
                result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            )
            return json.loads(content)
        except Exception as e:
            return {
                "error": str(e),
                "entities": self._extract_entities_fallback(content),
                "relations": [],
                "facts": [content[:200]] if content else [],
                "preferences": [],
                "timestamps": [],
                "confidence": 0.5,
            }

    def _get_parser_system_prompt(self) -> str:
        """Get system prompt for context parser."""
        return """You are a context parsing assistant. Your job is to extract structured information from user messages.

Extract and return JSON with:
- entities: List of important entities mentioned (names, objects, concepts)
- relations: List of entity-relation-value triples (e.g., {"entity": "user", "relation": "likes", "value": "python"})
- facts: List of factual statements made
- preferences: List of user preferences or opinions
- timestamps: List of mentioned times or dates
- confidence: Overall confidence in the extraction (0.0-1.0)

Be concise and accurate. Only return valid JSON."""

    def _extract_entities_fallback(self, content: str) -> List[str]:
        """Fallback entity extraction."""
        words = content.split()
        entities = []
        for i, word in enumerate(words):
            if word[0].isupper() and word.isalpha():
                if i < len(words) - 1 and words[i + 1][0].isupper():
                    entities.append(f"{word} {words[i + 1]}")
                else:
                    entities.append(word)
        return list(set(entities))[:10]

    def batch_parse_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Parse multiple messages efficiently."""
        results = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            results.append(self.parse_message(content, role))
        return results

    def extract_context_summary(
        self, messages: List[Dict[str, str]], k: int = 5
    ) -> str:
        """Extract a summary of relevant context from messages."""
        parsed = self.batch_parse_messages(messages[-k:])

        context_parts = []

        all_entities = set()
        all_facts = []
        all_preferences = []

        for p in parsed:
            all_entities.update(p.get("entities", []))
            all_facts.extend(p.get("facts", []))
            all_preferences.extend(p.get("preferences", []))

        if all_entities:
            context_parts.append(f"Entities: {', '.join(list(all_entities)[:10])}")

        if all_facts:
            context_parts.append(
                f"Key facts: {all_facts[0][:200] if all_facts else ''}"
            )

        if all_preferences:
            context_parts.append(
                f"Preferences: {all_preferences[0] if all_preferences else ''}"
            )

        return "\n".join(context_parts) if context_parts else ""


class OpenAICompatibleContextManager(BaseContextManager):
    """Context manager using OpenAI-compatible API for fast context parsing.

    NOTE: This is NOT a memory store - it only parses messages and extracts
    structured information. The parsed data is kept in memory only during
    the current session and is NOT persisted.

    Use with a memory store (VectorDB, KnowledgeGraph, etc.) for persistent storage.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:58080/v1",
        api_key: str = None,
        model: str = "LFM2.5-1.2B-Instruct-Q8_0",
        k_retrieval: int = 5,
        enable_benchmarking: bool = True,
    ):
        self.parser = OpenAIContextParser(
            api_url=api_url,
            api_key=api_key,
            model=model,
        )
        self.k_retrieval = k_retrieval
        self.enable_benchmarking = enable_benchmarking
        self.message_counter = 0
        self.message_log: List[Dict[str, Any]] = []
        self.parsed_messages: List[Dict[str, Any]] = []
        self.benchmark_times: List[float] = []

    def process_message(self, message: Dict[str, str]) -> None:
        """Process message: parse and store in temporary memory."""
        start_time = time.time() if self.enable_benchmarking else None

        content = message.get("content", "")
        role = message.get("role", "user")

        parsed = self.parser.parse_message(content, role)

        self.message_counter += 1
        self.message_log.append(
            {
                "id": self.message_counter,
                "content": content,
                "role": role,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.parsed_messages.append(parsed)

        if self.enable_benchmarking:
            elapsed = (time.time() - start_time) * 1000
            self.benchmark_times.append(elapsed)

    def get_context(self, current_message: Dict[str, str]) -> str:
        """Get context for current message from parsed history."""
        # Get relevant messages
        relevant = self._get_relevant_messages(current_message)

        # Build context from parsed data
        context_parts = []

        for msg in relevant:
            parsed = msg.get("parsed", {})
            if parsed:
                entities = parsed.get("entities", [])
                facts = parsed.get("facts", [])
                if entities:
                    context_parts.append(f"Entities: {', '.join(entities[:5])}")
                if facts:
                    context_parts.append(f"Fact: {facts[0][:100]}")

        return "\n".join(context_parts) if context_parts else ""

    def _get_relevant_messages(
        self, current_message: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Get relevant past messages based on parsed content."""
        current_content = current_message.get("content", "")

        # Simple relevance: check for keyword overlap with parsed entities
        current_words = set(current_content.lower().split())

        scored_messages = []
        for i, parsed in enumerate(self.parsed_messages):
            entities = set(e.lower() for e in parsed.get("entities", []))
            facts = " ".join(parsed.get("facts", [])).lower()

            overlap = len(current_words & entities)
            fact_overlap = sum(1 for w in current_words if w in facts)

            score = overlap + fact_overlap * 0.5
            if score > 0:
                scored_messages.append((i, score, parsed))

        scored_messages.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with parsed data
        return [
            {
                "id": i + 1,
                "parsed": parsed,
                "relevance_score": score,
            }
            for i, score, parsed in scored_messages[: self.k_retrieval]
        ]

    def get_context_size(self) -> int:
        """Estimate context size in tokens."""
        return len(self.parsed_messages) * 50  # Rough estimate

    def reset(self) -> None:
        """Reset state (clears temporary memory)."""
        self.message_counter = 0
        self.message_log.clear()
        self.parsed_messages.clear()
        self.benchmark_times.clear()

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.enable_benchmarking or not self.benchmark_times:
            return {"message": "No benchmark data"}

        return {
            "total_messages": len(self.benchmark_times),
            "avg_time_ms": sum(self.benchmark_times) / len(self.benchmark_times),
            "min_time_ms": min(self.benchmark_times),
            "max_time_ms": max(self.benchmark_times),
            "total_time_ms": sum(self.benchmark_times),
        }

    def get_parsed_history(self) -> List[Dict[str, Any]]:
        """Get full parsed message history."""
        return self.parsed_messages

    def get_entity_index(self) -> Dict[str, List[int]]:
        """Get index of which messages mention each entity."""
        entity_index = defaultdict(list)
        for i, parsed in enumerate(self.parsed_messages):
            for entity in parsed.get("entities", []):
                entity_index[entity].append(i)
        return dict(entity_index)
