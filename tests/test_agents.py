"""Unit tests for agent nodes with mocked LLM clients."""

from unittest.mock import MagicMock

from app.agents.grader import (
    AnswerGraderDecision,
    ContextGraderDecision,
    HallucinationGraderDecision,
    context_grader_node,
    generation_grader_node,
)
from app.agents.router import QueryRoutingDecision, query_router_node
from app.agents.rewriter import QueryRewriteDecision, query_rewriter_node
from app.agents.state import AgentState
from app.agents.synthesizer import synthesizer_node
from app.llm_client import LLMClient


def _make_state(**overrides) -> AgentState:
    """Return a baseline AgentState with optional overrides."""
    state: AgentState = {
        "query": "What is GraphRAG?",
        "refined_query": None,
        "extracted_entities": [],
        "vector_context": [],
        "graph_context": "",
        "combined_context": None,
        "response": None,
        "route": None,
        "retrieval_score": None,
        "hallucination_score": None,
        "answer_score": None,
        "loop_count": 0,
        "max_loops": 3,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _mock_parsed_client(parsed):
    """Return a MagicMock LLMClient whose chat_completion returns parsed."""
    client = MagicMock(spec=LLMClient)
    client.chat_completion.return_value = parsed
    return client


def _mock_parsed_client_sequence(decisions):
    """Return a mock LLMClient whose chat_completion yields each decision in order."""
    client = MagicMock(spec=LLMClient)
    client.chat_completion.side_effect = decisions
    return client


def _mock_chat_client(content: str):
    """Return a MagicMock LLMClient whose chat_completion returns content."""
    client = MagicMock(spec=LLMClient)
    client.chat_completion.return_value = content
    return client


def test_router_node_returns_vector_route():
    decision = QueryRoutingDecision(
        reasoning="Broad conceptual question.",
        route="vector_search",
        entities=["GraphRAG"],
    )
    client = _mock_parsed_client(decision)
    state = _make_state()

    result = query_router_node(state, client, "gpt-4o-mini")

    assert result["route"] == "vector_search"
    assert result["extracted_entities"] == ["GraphRAG"]
    assert result["loop_count"] == 1
    client.chat_completion.assert_called_once()


def test_router_node_defaults_to_hybrid_on_failure():
    client = MagicMock(spec=LLMClient)
    client.chat_completion.side_effect = RuntimeError("API error")
    state = _make_state(loop_count=2)

    result = query_router_node(state, client, "gpt-4o-mini")

    assert result["route"] == "hybrid"
    assert result["extracted_entities"] == []
    assert result["loop_count"] == 3


def test_context_grader_node_returns_relevant():
    decision = ContextGraderDecision(
        reasoning="Context directly answers the query.",
        relevant=True,
    )
    client = _mock_parsed_client(decision)
    state = _make_state(
        refined_query="Explain GraphRAG.",
        vector_context=[{"text": "GraphRAG combines graphs and RAG."}],
        graph_context="GraphRAG -> improves -> retrieval",
    )

    result = context_grader_node(state, client, "gpt-4o-mini")

    assert result["retrieval_score"] == "yes"
    assert "GraphRAG combines graphs and RAG." in result["combined_context"]
    assert "GraphRAG -> improves -> retrieval" in result["combined_context"]


def test_context_grader_node_returns_irrelevant():
    decision = ContextGraderDecision(
        reasoning="Context does not match the query.",
        relevant=False,
    )
    client = _mock_parsed_client(decision)
    state = _make_state(
        query="What is the weather?",
        vector_context=[{"text": "GraphRAG combines graphs and RAG."}],
    )

    result = context_grader_node(state, client, "gpt-4o-mini")

    assert result["retrieval_score"] == "no"


def test_context_grader_node_defaults_to_relevant_on_failure():
    client = MagicMock(spec=LLMClient)
    client.chat_completion.side_effect = RuntimeError("API error")
    state = _make_state(vector_context=[{"text": "Some context."}])

    result = context_grader_node(state, client, "gpt-4o-mini")

    assert result["retrieval_score"] == "yes"
    assert "Some context." in result["combined_context"]


def test_generation_grader_node_grounded_and_useful():
    hallucination_decision = HallucinationGraderDecision(
        reasoning="Answer is supported by context.",
        grounded=True,
    )
    answer_decision = AnswerGraderDecision(
        reasoning="Answer addresses the query.",
        answers_query=True,
    )
    client = _mock_parsed_client_sequence([hallucination_decision, answer_decision])

    state = _make_state(
        query="What is GraphRAG?",
        response="GraphRAG is a retrieval technique.",
        combined_context="GraphRAG is a retrieval technique.",
    )

    result = generation_grader_node(state, client, "gpt-4o-mini")

    assert result["hallucination_score"] == "no"
    assert result["answer_score"] == "yes"
    assert client.chat_completion.call_count == 2


def test_generation_grader_node_not_grounded_not_useful():
    hallucination_decision = HallucinationGraderDecision(
        reasoning="Answer invents facts.",
        grounded=False,
    )
    answer_decision = AnswerGraderDecision(
        reasoning="Answer does not address the query.",
        answers_query=False,
    )
    client = _mock_parsed_client_sequence([hallucination_decision, answer_decision])

    state = _make_state(
        query="What is GraphRAG?",
        response="It is a type of sandwich.",
        combined_context="GraphRAG is a retrieval technique.",
    )

    result = generation_grader_node(state, client, "gpt-4o-mini")

    assert result["hallucination_score"] == "yes"
    assert result["answer_score"] == "no"
    assert client.chat_completion.call_count == 2


def test_generation_grader_node_empty_response():
    client = MagicMock(spec=LLMClient)
    state = _make_state(query="What is GraphRAG?", response="")

    result = generation_grader_node(state, client, "gpt-4o-mini")

    assert result["hallucination_score"] == "yes"
    assert result["answer_score"] == "no"
    client.chat_completion.assert_not_called()


def test_query_rewriter_node_returns_rewritten_query():
    decision = QueryRewriteDecision(
        reasoning="Adds synonyms for better retrieval.",
        refined_query="What is GraphRAG and how does it work?",
    )
    client = _mock_parsed_client(decision)
    state = _make_state(loop_count=1)

    result = query_rewriter_node(state, client, "gpt-4o-mini")

    assert result["refined_query"] == "What is GraphRAG and how does it work?"
    assert result["loop_count"] == 2


def test_query_rewriter_node_defaults_to_original_on_failure():
    client = MagicMock(spec=LLMClient)
    client.chat_completion.side_effect = RuntimeError("API error")
    state = _make_state(query="GraphRAG benefits", loop_count=0)

    result = query_rewriter_node(state, client, "gpt-4o-mini")

    assert result["refined_query"] == "GraphRAG benefits"
    assert result["loop_count"] == 1


def test_synthesizer_node_returns_answer():
    client = _mock_chat_client("GraphRAG combines knowledge graphs with retrieval-augmented generation.")
    state = _make_state(
        query="What is GraphRAG?",
        combined_context="GraphRAG combines knowledge graphs with RAG.",
    )

    result = synthesizer_node(state, client, "gpt-4o-mini")

    assert result["response"] == "GraphRAG combines knowledge graphs with retrieval-augmented generation."
    client.chat_completion.assert_called_once()


def test_synthesizer_node_returns_fallback_on_failure():
    client = MagicMock(spec=LLMClient)
    client.chat_completion.side_effect = RuntimeError("API error")
    state = _make_state(
        query="What is GraphRAG?",
        combined_context="GraphRAG combines knowledge graphs with RAG.",
    )

    result = synthesizer_node(state, client, "gpt-4o-mini")

    assert "LLM synthesis unavailable" in result["response"]
    assert "GraphRAG combines knowledge graphs with RAG." in result["response"]
