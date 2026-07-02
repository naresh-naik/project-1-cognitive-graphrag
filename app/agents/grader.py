import logging

from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ContextGraderDecision(BaseModel):
    reasoning: str = Field(description="Reasoning for context relevance.")
    relevant: bool = Field(description="Whether retrieved context is relevant.")


class HallucinationGraderDecision(BaseModel):
    reasoning: str = Field(description="Reasoning for grounding decision.")
    grounded: bool = Field(description="Whether answer is grounded in context.")


class AnswerGraderDecision(BaseModel):
    reasoning: str = Field(description="Reasoning for answer relevance.")
    answers_query: bool = Field(description="Whether answer addresses the query.")


def context_grader_node(
    state: AgentState,
    llm_client: LLMClient,
    model_name: str,
) -> dict:
    query = state.get("refined_query") or state["query"]

    vector_text = "\n".join(
        item.get("text", "") for item in state.get("vector_context", [])
    )

    graph_text = state.get("graph_context", "")

    combined_context = f"""
=== VECTOR CONTEXT ===
{vector_text}

=== GRAPH CONTEXT ===
{graph_text}
""".strip()

    prompt = f"""
Evaluate whether the retrieved context can help answer the query.

Query:
{query}

Context:
{combined_context}
"""

    try:
        decision = llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict retrieval quality grader. Return only valid JSON matching the schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=ContextGraderDecision,
        )

        score = "yes" if decision.relevant else "no"

        return {
            "retrieval_score": score,
            "combined_context": combined_context,
        }

    except Exception as error:
        logger.exception("Context grading failed; defaulting to relevant: %s", error)

        return {
            "retrieval_score": "yes",
            "combined_context": combined_context,
        }


def generation_grader_node(
    state: AgentState,
    llm_client: LLMClient,
    model_name: str,
) -> dict:
    query = state["query"]
    context = state.get("combined_context", "") or ""
    response = state.get("response", "") or ""

    if not response:
        return {
            "hallucination_score": "yes",
            "answer_score": "no",
        }

    hallucination_prompt = f"""
Check whether the answer is strictly grounded in the context.

Context:
{context}

Answer:
{response}
"""

    answer_prompt = f"""
Check whether the answer directly answers the user query.

Query:
{query}

Answer:
{response}
"""

    hallucination_score = "no"
    answer_score = "yes"

    try:
        decision = llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a hallucination detector. Return only valid JSON matching the schema.",
                },
                {"role": "user", "content": hallucination_prompt},
            ],
            response_format=HallucinationGraderDecision,
        )

        hallucination_score = "no" if decision.grounded else "yes"

    except Exception as error:
        logger.exception("Hallucination grading failed: %s", error)

    try:
        decision = llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an answer relevance grader. Return only valid JSON matching the schema.",
                },
                {"role": "user", "content": answer_prompt},
            ],
            response_format=AnswerGraderDecision,
        )

        answer_score = "yes" if decision.answers_query else "no"

    except Exception as error:
        logger.exception("Answer grading failed: %s", error)

    return {
        "hallucination_score": hallucination_score,
        "answer_score": answer_score,
    }
