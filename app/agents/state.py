from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    query: str
    refined_query: Optional[str]

    extracted_entities: List[str]

    vector_context: List[Dict[str, Any]]
    graph_context: str
    combined_context: Optional[str]

    response: Optional[str]

    route: Optional[str]
    retrieval_score: Optional[str]
    hallucination_score: Optional[str]
    answer_score: Optional[str]

    loop_count: int
    max_loops: int
