import logging

from app.agents.state import AgentState
from app.llm_client import LLMClient

logger = logging.getLogger(__name__)


def synthesizer_node(
    state: AgentState,
    llm_client: LLMClient,
    model_name: str,
) -> dict:
    query = state["query"]
    context = state.get("combined_context") or ""

    prompt = f"""
Answer the query using ONLY the provided context.

Rules:
- Do not invent facts.
- If context is insufficient, say what is missing.
- Use clear markdown.
- Reference relationships if graph context is available.

Query:
{query}

Context:
{context}
"""

    try:
        response = llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded answer synthesizer.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        return {
            "response": response,
        }

    except Exception as error:
        logger.exception("Synthesis failed: %s", error)

        return {
            "response": "Synthesis failed. Please inspect server logs.",
        }
