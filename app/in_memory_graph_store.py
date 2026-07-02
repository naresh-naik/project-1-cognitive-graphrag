import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

_RE_REL_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_]")


def _normalize_relationship_type(relationship_type: str) -> str:
    """Return a safe relationship-type identifier."""
    if not relationship_type:
        return "RELATED_TO"

    normalized = _RE_REL_INVALID_CHARS.sub("_", relationship_type.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    if not normalized:
        return "RELATED_TO"

    if normalized[0].isdigit():
        normalized = f"_{normalized}"

    return normalized.upper()


class InMemoryCognitiveGraphStore:
    """In-memory fallback for CognitiveGraphStore when Neo4j is unavailable."""

    def __init__(self, *_args, **_kwargs):
        self.entities: Dict[str, Dict[str, str]] = {}
        self.relationships: List[Dict[str, str]] = []
        logger.info("Using in-memory graph store.")

    def close(self) -> None:
        pass

    def add_entity(self, name: str, entity_type: str, description: str) -> None:
        name = name.strip()
        self.entities[name] = {
            "type": entity_type.strip(),
            "description": description.strip(),
        }

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        description: str,
    ) -> None:
        safe_relationship_type = _normalize_relationship_type(relationship_type)
        self.relationships.append(
            {
                "source": source.strip(),
                "target": target.strip(),
                "relationship_type": safe_relationship_type,
                "description": description.strip(),
            }
        )
        # Ensure source/target entities exist minimally
        for name in (source.strip(), target.strip()):
            if name not in self.entities:
                self.entities[name] = {
                    "type": "Unknown",
                    "description": "No description available.",
                }

    def fuzzy_match_entities(self, query_text: str, limit: int = 5) -> List[str]:
        query_lower = query_text.lower()
        scored: List[Tuple[str, float]] = []

        for name, data in self.entities.items():
            score = 0.0
            name_lower = name.lower()
            desc_lower = data.get("description", "").lower()

            if query_lower in name_lower:
                score += 2.0
            if query_lower in desc_lower:
                score += 1.0

            for token in query_lower.split():
                if len(token) > 2:
                    if token in name_lower:
                        score += 0.5
                    if token in desc_lower:
                        score += 0.25

            if score > 0:
                scored.append((name, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in scored[:limit]]

    def get_entity_context(self, entities: List[str]) -> str:
        if not entities:
            return ""

        context_parts: Set[str] = set()
        entity_names = {name.strip() for name in entities}

        for name in entity_names:
            if name in self.entities:
                data = self.entities[name]
                context_parts.add(
                    f"Entity: {name} ({data.get('type', 'Unknown')}) - {data.get('description', 'No description available.')}"
                )

        for rel in self.relationships:
            if rel["source"] in entity_names or rel["target"] in entity_names:
                context_parts.add(
                    f"Relationship: ({rel['source']}) "
                    f"-[{rel['relationship_type']}: {rel['description']}]-> "
                    f"({rel['target']})"
                )

        if not context_parts:
            return "No matching graph context found."

        return "\n".join(sorted(context_parts))
