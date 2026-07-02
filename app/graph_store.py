import logging
import re
from typing import List

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


_RE_REL_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_]")


def _normalize_relationship_type(relationship_type: str) -> str:
    """Return a safe Neo4j relationship-type identifier."""
    if not relationship_type:
        return "RELATED_TO"

    normalized = _RE_REL_INVALID_CHARS.sub("_", relationship_type.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    if not normalized:
        return "RELATED_TO"

    if normalized[0].isdigit():
        normalized = f"_{normalized}"

    return normalized.upper()


class CognitiveGraphStore:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        self._create_indexes()

    def close(self) -> None:
        self.driver.close()

    def _create_indexes(self) -> None:
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT unique_entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            session.run(
                "CREATE FULLTEXT INDEX entity_search_idx IF NOT EXISTS "
                "FOR (e:Entity) ON EACH [e.name, e.description]"
            )

    def add_entity(self, name: str, entity_type: str, description: str) -> None:
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.created_at = timestamp()
        ON MATCH SET
            e.description = coalesce($description, e.description),
            e.updated_at = timestamp()
        """
        with self.driver.session() as session:
            session.run(
                query,
                name=name.strip(),
                type=entity_type.strip(),
                description=description.strip(),
            )

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        description: str,
    ) -> None:
        safe_relationship_type = _normalize_relationship_type(relationship_type)

        query = f"""
        MERGE (s:Entity {{name: $source}})
        MERGE (t:Entity {{name: $target}})
        MERGE (s)-[r:{safe_relationship_type}]->(t)
        ON CREATE SET
            r.description = $description,
            r.created_at = timestamp()
        ON MATCH SET
            r.description = coalesce($description, r.description),
            r.updated_at = timestamp()
        """

        with self.driver.session() as session:
            session.run(
                query,
                source=source.strip(),
                target=target.strip(),
                description=description.strip(),
            )

    def fuzzy_match_entities(self, query_text: str, limit: int = 5) -> List[str]:
        query = """
        CALL db.index.fulltext.queryNodes("entity_search_idx", $query_text)
        YIELD node, score
        RETURN node.name AS name, score
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                query_text=f"{query_text}~2",
                limit=limit,
            )
            return [record["name"] for record in result if record["score"] > 0.25]

    def get_entity_context(self, entities: List[str]) -> str:
        if not entities:
            return ""

        query = """
        MATCH (e:Entity)
        WHERE e.name IN $entities
        OPTIONAL MATCH (e)-[r]->(neighbor:Entity)
        RETURN
            e.name AS source,
            e.type AS source_type,
            e.description AS source_description,
            type(r) AS relationship_type,
            r.description AS relationship_description,
            neighbor.name AS target,
            neighbor.type AS target_type
        LIMIT 100
        """

        context_parts = []

        with self.driver.session() as session:
            result = session.run(query, entities=entities)

            for record in result:
                source = record["source"]
                source_type = record["source_type"] or "Unknown"
                source_description = record["source_description"] or "No description available."

                context_parts.append(
                    f"Entity: {source} ({source_type}) - {source_description}"
                )

                target = record["target"]
                relationship_type = record["relationship_type"]
                relationship_description = (
                    record["relationship_description"] or "related to"
                )

                if target and relationship_type:
                    context_parts.append(
                        f"Relationship: ({source}) "
                        f"-[{relationship_type}: {relationship_description}]-> "
                        f"({target})"
                    )

        if not context_parts:
            return "No matching graph context found."

        return "\n".join(sorted(set(context_parts)))
