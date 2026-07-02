import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from app.agents.state import AgentState

logger = logging.getLogger(__name__)


class QueryRewriteDecision(BaseModel):
    reasoning: str = Field(description="Why the rewrite improves retrieval.")
    refined_query: str = Field(description="Improved search query.")


def query_rewriter_node(
    state: AgentState,
    openai_client: OpenAI,
    model_name: str,
) -> dict:
    original_query = state["query"]
    current_query = state.get("refined_query") or original_query

    prompt = f"""
Rewrite the query to improve retrieval quality without changing the original intent.

Original query:
{original_query}

Current query:
{current_query}

Retrieval score:
{state.get("retrieval_score")}

Hallucination score:
{state.get("hallucination_score")}

Answer score:
{state.get("answer_score")}
"""

    try:
        completion = openai_client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a search query optimization expert.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=QueryRewriteDecision,
        )

        decision = completion.choices[0].message.parsed

        return {
            "refined_query": decision.refined_query,
            "loop_count": state.get("loop_count", 0) + 1,
        }

    except Exception as error:
        logger.exception("Query rewrite failed; using original query: %s", error)

        return {
            "refined_query": original_query,
            "loop_count": state.get("loop_count", 0) + 1,
        }
