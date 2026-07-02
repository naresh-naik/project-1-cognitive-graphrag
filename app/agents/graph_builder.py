import logging
from functools import partial

from langgraph.graph import END, START, StateGraph

from app.agents.grader import context_grader_node, generation_grader_node
from app.agents.rewriter import query_rewriter_node
from app.agents.router import query_router_node
from app.agents.state import AgentState
from app.agents.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def retrieve_context_node(
    state: AgentState,
    vector_store,
    graph_store,
) -> dict:
    query = state.get("refined_query") or state["query"]
    route = state.get("route") or "hybrid"
    entities = state.get("extracted_entities", [])

    vector_context = []
    graph_context = ""

    if route in ("vector_search", "hybrid"):
        vector_context = vector_store.similarity_search(query, limit=5)

    if route in ("graph_search", "hybrid"):
        resolved_entities = []

        for entity in entities:
            matches = graph_store.fuzzy_match_entities(entity, limit=1)
            resolved_entities.extend(matches or [entity])

        graph_context = graph_store.get_entity_context(
            list(sorted(set(resolved_entities)))
        )

    return {
        "vector_context": vector_context,
        "graph_context": graph_context,
    }


def route_after_context_grade(state: AgentState) -> str:
    if state.get("retrieval_score") == "yes":
        return "synthesize"

    if state.get("loop_count", 0) >= state.get("max_loops", 3):
        return "synthesize"

    return "rewrite_query"


def route_after_generation_grade(state: AgentState) -> str:
    hallucination_score = state.get("hallucination_score")
    answer_score = state.get("answer_score")

    if hallucination_score == "no" and answer_score == "yes":
        return END

    if state.get("loop_count", 0) >= state.get("max_loops", 3):
        return END

    return "rewrite_query"


def build_workflow(llm_client, model_name: str, vector_store, graph_store):
    workflow = StateGraph(AgentState)

    workflow.add_node(
        "router",
        partial(
            query_router_node,
            llm_client=llm_client,
            model_name=model_name,
        ),
    )

    workflow.add_node(
        "retrieve",
        partial(
            retrieve_context_node,
            vector_store=vector_store,
            graph_store=graph_store,
        ),
    )

    workflow.add_node(
        "grade_context",
        partial(
            context_grader_node,
            llm_client=llm_client,
            model_name=model_name,
        ),
    )

    workflow.add_node(
        "rewrite_query",
        partial(
            query_rewriter_node,
            llm_client=llm_client,
            model_name=model_name,
        ),
    )

    workflow.add_node(
        "synthesize",
        partial(
            synthesizer_node,
            llm_client=llm_client,
            model_name=model_name,
        ),
    )

    workflow.add_node(
        "grade_generation",
        partial(
            generation_grader_node,
            llm_client=llm_client,
            model_name=model_name,
        ),
    )

    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retrieve")
    workflow.add_edge("retrieve", "grade_context")

    workflow.add_conditional_edges(
        "grade_context",
        route_after_context_grade,
        {
            "synthesize": "synthesize",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "router")
    workflow.add_edge("synthesize", "grade_generation")

    workflow.add_conditional_edges(
        "grade_generation",
        route_after_generation_grade,
        {
            END: END,
            "rewrite_query": "rewrite_query",
        },
    )

    return workflow.compile()
