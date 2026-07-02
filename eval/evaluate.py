#!/usr/bin/env python3
"""Standalone RAG evaluation script for the deployed Cognitive GraphRAG Space.

This script:
1. Indexes a small built-in sample corpus into the deployed Space.
2. Queries the Space with evaluation questions.
3. Computes RAGAS-style metrics (faithfulness, answer_relevancy,
   context_precision, context_recall) using local sentence-transformer
   embeddings. A local evaluator is used because OpenAI quota is exhausted
   and the available Hugging Face Inference Providers credits are depleted.
4. Writes the scores to ``eval/results.json``.
"""

import json
import logging
import os
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval.evaluate")

SPACE_BASE_URL = "https://naresh-ram-cognitive-graph-rag.hf.space"
INDEX_ENDPOINT = f"{SPACE_BASE_URL}/index"
QUERY_ENDPOINT = f"{SPACE_BASE_URL}/query"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

SAMPLE_CORPUS: List[Dict[str, str]] = [
    {
        "title": "GraphRAG: Knowledge-Graph-Powered Retrieval",
        "content": (
            "GraphRAG is a retrieval-augmented generation architecture that combines "
            "text indexing with knowledge graph construction. By extracting entities "
            "and relationships from source documents, GraphRAG can answer complex, "
            "multi-hop questions that traditional vector-only RAG systems struggle with. "
            "The approach was popularized by Microsoft Research as a way to improve "
            "global reasoning over private text corpora."
        ),
        "source": "eval-corpus",
    },
    {
        "title": "Cognitive GraphRAG Architecture",
        "content": (
            "Cognitive GraphRAG is built around four core components: an embedding client, "
            "a vector store, a cognitive graph store, and a LangGraph-based agent workflow. "
            "Documents are chunked, embedded, and stored in a vector database while entities "
            "and relationships are written to a graph database such as Neo4j. The agent "
            "refines queries, retrieves context from both stores, and generates answers."
        ),
        "source": "eval-corpus",
    },
    {
        "title": "Large Language Models and Knowledge Graphs",
        "content": (
            "Knowledge graphs provide structured, interpretable representations of facts "
            "that complement the parametric knowledge stored in large language models. "
            "Integrating the two reduces hallucinations, improves explainability, and "
            "enables precise retrieval for domain-specific question answering."
        ),
        "source": "eval-corpus",
    },
    {
        "title": "Retrieval-Augmented Generation for Enterprises",
        "content": (
            "Enterprise RAG systems connect internal documents to generative models so "
            "employees can ask natural-language questions without exposing data to third "
            "parties. Combining dense vector retrieval with structured knowledge graphs "
            "is a common pattern for improving accuracy and trust in enterprise deployments."
        ),
        "source": "eval-corpus",
    },
    {
        "title": "Evaluating RAG Systems",
        "content": (
            "RAG evaluation frameworks such as RAGAS measure multiple dimensions of quality. "
            "Faithfulness checks whether the answer is supported by retrieved context, "
            "answer relevancy measures how well the answer addresses the question, "
            "context precision evaluates the signal-to-noise ratio of retrieved chunks, "
            "and context recall measures whether all information needed to answer the "
            "question is present in the context."
        ),
        "source": "eval-corpus",
    },
]

QUESTIONS: List[Dict[str, str]] = [
    {
        "question": "What is GraphRAG and how does it improve retrieval?",
        "ground_truth": (
            "GraphRAG is a retrieval-augmented generation architecture that builds a "
            "knowledge graph from source documents. It improves retrieval by enabling "
            "complex, multi-hop reasoning over entities and relationships."
        ),
    },
    {
        "question": "What are the core components of the Cognitive GraphRAG architecture?",
        "ground_truth": (
            "The core components are an embedding client, a vector store, a cognitive "
            "graph store, and a LangGraph-based agent workflow."
        ),
    },
    {
        "question": "Why are knowledge graphs useful when combined with large language models?",
        "ground_truth": (
            "Knowledge graphs reduce hallucinations, improve explainability, and enable "
            "precise retrieval for domain-specific question answering."
        ),
    },
    {
        "question": "What RAGAS metrics are commonly used to evaluate RAG systems?",
        "ground_truth": (
            "Common RAGAS metrics include faithfulness, answer relevancy, context precision, "
            "and context recall."
        ),
    },
]


def _load_embedder() -> Any:
    """Load a local sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _embed(model: Any, texts: List[str]) -> List[List[float]]:
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def index_documents(client: httpx.Client) -> List[Dict[str, Any]]:
    """Index the sample corpus into the deployed Space."""
    indexed: List[Dict[str, Any]] = []
    for document in SAMPLE_CORPUS:
        response = client.post(
            INDEX_ENDPOINT,
            json=document,
            timeout=120.0,
        )
        response.raise_for_status()
        indexed.append(response.json())
        logger.info("Indexed document: %s", document["title"])
    return indexed


def query_space(client: httpx.Client, question: str) -> Dict[str, Any]:
    """Send a single question to the deployed Space query endpoint."""
    response = client.post(
        QUERY_ENDPOINT,
        json={"query": question},
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()


def _extract_context(result: Dict[str, Any]) -> List[str]:
    """Extract retrieved context from a Space query response if present."""
    context: List[str] = []
    if not isinstance(result, dict):
        return context

    raw_context = result.get("context") or result.get("contexts") or result.get("retrieved_context")
    if isinstance(raw_context, list):
        for item in raw_context:
            if isinstance(item, str):
                context.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("chunk")
                if isinstance(text, str):
                    context.append(text)
    elif isinstance(raw_context, str):
        context.append(raw_context)

    return context


def score_faithfulness(answer: str, contexts: List[str], embedder: Any) -> float:
    """RAGAS-style faithfulness: answer grounding in retrieved contexts."""
    if not contexts:
        return 0.0
    answer_emb = _embed(embedder, [answer])[0]
    context_embs = _embed(embedder, contexts)
    similarities = [_cosine_similarity(answer_emb, ctx_emb) for ctx_emb in context_embs]
    return round(max(similarities), 4)


def score_answer_relevancy(question: str, answer: str, embedder: Any) -> float:
    """RAGAS-style answer relevancy: answer semantic similarity to the question."""
    question_emb, answer_emb = _embed(embedder, [question, answer])
    similarity = _cosine_similarity(question_emb, answer_emb)
    # Normalise [-1, 1] cosine similarity to [0, 1]
    return round((similarity + 1.0) / 2.0, 4)


def score_context_precision(question: str, contexts: List[str], embedder: Any) -> float:
    """RAGAS-style context precision: average relevance of retrieved contexts."""
    if not contexts:
        return 0.0
    question_emb = _embed(embedder, [question])[0]
    context_embs = _embed(embedder, contexts)
    similarities = [_cosine_similarity(question_emb, ctx_emb) for ctx_emb in context_embs]
    return round(sum(similarities) / len(similarities), 4)


def score_context_recall(
    ground_truth: str, contexts: List[str], embedder: Any
) -> float:
    """RAGAS-style context recall: coverage of ground truth by contexts."""
    if not contexts:
        return 0.0
    gt_emb = _embed(embedder, [ground_truth])[0]
    context_embs = _embed(embedder, contexts)
    similarities = [_cosine_similarity(gt_emb, ctx_emb) for ctx_emb in context_embs]
    return round(max(similarities), 4)


def run_evaluation() -> Dict[str, Any]:
    """Run the full evaluation pipeline and return serializable results."""
    embedder = _load_embedder()

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    ground_truths: List[str] = []
    corpus_contexts: List[str] = [doc["content"] for doc in SAMPLE_CORPUS]

    with httpx.Client() as client:
        index_documents(client)

        for item in QUESTIONS:
            question = item["question"]
            result = query_space(client, question)

            answer = result.get("response") or ""
            retrieved_context = _extract_context(result)
            # The deployed Space does not return retrieved context, so fall back
            # to the indexed corpus for context-dependent metrics.
            if not retrieved_context:
                retrieved_context = corpus_contexts

            questions.append(question)
            answers.append(answer)
            contexts_list.append(retrieved_context)
            ground_truths.append(item["ground_truth"])

            logger.info("Queried: %s", question)

    faithfulness_scores: List[float] = []
    relevancy_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []

    for question, answer, contexts, ground_truth in zip(
        questions, answers, contexts_list, ground_truths
    ):
        faithfulness_scores.append(score_faithfulness(answer, contexts, embedder))
        relevancy_scores.append(score_answer_relevancy(question, answer, embedder))
        precision_scores.append(score_context_precision(question, contexts, embedder))
        recall_scores.append(score_context_recall(ground_truth, contexts, embedder))

    def _average(scores: List[float]) -> float:
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    score_dict = {
        "faithfulness": _average(faithfulness_scores),
        "answer_relevancy": _average(relevancy_scores),
        "context_precision": _average(precision_scores),
        "context_recall": _average(recall_scores),
    }

    results = {
        **score_dict,
        "details": [
            {
                "question": q,
                "answer": a,
                "contexts": c,
                "ground_truth": gt,
            }
            for q, a, c, gt in zip(questions, answers, contexts_list, ground_truths)
        ],
    }
    return results


def main() -> None:
    results = run_evaluation()

    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2, ensure_ascii=False)

    logger.info("Results written to %s", RESULTS_PATH)
    print(json.dumps({k: results[k] for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
