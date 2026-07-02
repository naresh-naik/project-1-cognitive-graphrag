import logging
from typing import List, Literal

from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryRoutingDecision(BaseModel):
    reasoning: str = Field(description="Reasoning behind the routing decision.")
    route: Literal["vector_search", "graph_search", "hybrid"] = Field(
        description="Search route to use."
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Entities extracted from the query for graph lookup.",
    )


def query_router_node(
    state: AgentState,
    llm_client: LLMClient,
    model_name: str,
) -> dict:
    query = state["query"]

    prompt = f"""
Analyze the query and decide which retrieval strategy to use.

Routes:
- vector_search: broad conceptual questions
- graph_search: relationship/path/entity questions
- hybrid: entity-heavy questions that also need semantic background

Extract important entity names.

Query:
{query}
"""

    try:
        decision = llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a query router for a GraphRAG system. Return only valid JSON matching the schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=QueryRoutingDecision,
        )

        return {
            "route": decision.route,
            "extracted_entities": decision.entities,
            "loop_count": state.get("loop_count", 0) + 1,
        }

    except Exception as error:
        logger.exception("Router failed; defaulting to hybrid route: %s", error)

        return {
            "route": "hybrid",
            "extracted_entities": [],
            "loop_count": state.get("loop_count", 0) + 1,
        }
