import logging
from typing import List, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from app.agents.state import AgentState

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
    openai_client: OpenAI,
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
        completion = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a query router for a GraphRAG system.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=QueryRoutingDecision,
        )

        decision = completion.choices[0].message.parsed

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
