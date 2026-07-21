#!/usr/bin/env python3
"""
RAG evaluation runner — precision@k, faithfulness, answer relevance.

Scores retrieval before and after re-ranking, plus LLM-as-judge graders on
the generated answer. Re-run after chunking, re-ranker, or prompt changes to
compare before/after scores.

Usage (from backend/):
  python -m eval.run_eval --user-id <USER_UUID> [--k 4] [--dataset eval/dataset.json]
  python -m eval.run_eval --user-id <USER_UUID> --upload-langsmith

Requires:
  - Indexed documents for the user in Neon pgvector
  - MISTRAL_API_KEY in .env
  - Hand-verified expected_doc_ids in eval/dataset.json (replace placeholders)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langsmith import Client, evaluate
from langsmith.schemas import Example, Run

# Ensure backend app is importable
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.services import langchain_rag as lc
from app.services import langgraph_rag
from app.services import retrieval
from app.services.observability import configure_observability

logger = logging.getLogger(__name__)
DEFAULT_DATASET = Path(__file__).parent / "dataset.json"


def _precision_at_k(retrieved_doc_ids: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    top = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top if doc_id in expected)
    return hits / min(k, len(expected))


def _collect_context(chunks) -> str:
    return "\n\n---\n\n".join(c.text for c in chunks)


def _run_example(inputs: dict, user_id: str, k: int) -> dict:
    question = inputs["question"]
    expected_doc_ids = set(inputs.get("expected_doc_ids") or [])

    pre = retrieval.retrieve_hybrid(question, user_id, top_k=k)
    post = retrieval.retrieve(question, user_id, top_k=k)

    pre_ids = [c.doc_id for c in pre]
    post_ids = [c.doc_id for c in post]

    answer, sources, meta = langgraph_rag.answer_with_langgraph(question, user_id)
    context = _collect_context(post)

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "expected_answer": inputs.get("expected_answer", ""),
        "expected_doc_ids": list(expected_doc_ids),
        "retrieved_doc_ids_pre_rerank": pre_ids,
        "retrieved_doc_ids_post_rerank": post_ids,
        "precision_at_k_pre_rerank": _precision_at_k(pre_ids, expected_doc_ids, k),
        "precision_at_k_post_rerank": _precision_at_k(post_ids, expected_doc_ids, k),
        "sources": [s.model_dump() for s in sources],
        "retrieval_query": meta.get("retrieval_query", question),
    }


def _llm_judge(prompt: str) -> float:
    llm = lc.get_chat_llm(streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])
    text = (response.content or "").strip().lower()
    if "yes" in text[:10]:
        return 1.0
    if "partial" in text[:15]:
        return 0.5
    return 0.0


def faithfulness_evaluator(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    reference = example.outputs or {}
    answer = outputs.get("answer", "")
    context = outputs.get("context", "")
    if not answer or not context:
        return {"key": "faithfulness", "score": 0.0}

    prompt = (
        "You are grading RAG faithfulness. "
        "Does EVERY factual claim in the answer trace back to the retrieved context? "
        "Reply with exactly one word: yes, partial, or no.\n\n"
        f"Context:\n{context[:6000]}\n\n"
        f"Answer:\n{answer}\n\n"
        "Verdict:"
    )
    score = _llm_judge(prompt)
    return {"key": "faithfulness", "score": score}


def answer_relevance_evaluator(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    reference = example.outputs or {}
    answer = outputs.get("answer", "")
    question = (example.inputs or {}).get("question", "")
    expected = reference.get("expected_answer", "")
    if not answer:
        return {"key": "answer_relevance", "score": 0.0}

    prompt = (
        "You are grading answer relevance. "
        "Does the answer appropriately address the question and align with the expected answer? "
        "Reply with exactly one word: yes, partial, or no.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Generated answer: {answer}\n\n"
        "Verdict:"
    )
    score = _llm_judge(prompt)
    return {"key": "answer_relevance", "score": score}


def precision_pre_evaluator(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    score = outputs.get("precision_at_k_pre_rerank", 0.0)
    return {"key": "precision_at_k_pre_rerank", "score": score}


def precision_post_evaluator(run: Run, example: Example) -> dict:
    outputs = run.outputs or {}
    score = outputs.get("precision_at_k_post_rerank", 0.0)
    return {"key": "precision_at_k_post_rerank", "score": score}


def load_dataset(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("examples") or data


def run_local(dataset_path: Path, user_id: str, k: int) -> dict:
    examples = load_dataset(dataset_path)
    results = []
    for ex in examples:
        if not ex.get("expected_doc_ids"):
            logger.warning("Skipping %s — set expected_doc_ids after verification", ex.get("id"))
            continue
        out = _run_example(ex, user_id, k)
        out["id"] = ex.get("id")
        results.append(out)

    if not results:
        raise SystemExit(
            "No eval examples with expected_doc_ids. "
            "Edit eval/dataset.json and replace placeholder doc UUIDs."
        )

    def avg(key: str) -> float:
        return round(sum(r[key] for r in results) / len(results), 4)

    summary = {
        "n": len(results),
        "precision_at_k_pre_rerank": avg("precision_at_k_pre_rerank"),
        "precision_at_k_post_rerank": avg("precision_at_k_post_rerank"),
        "k": k,
    }
    return {"summary": summary, "results": results}


def upload_and_evaluate(dataset_path: Path, user_id: str, k: int, project: str) -> None:
    settings = get_settings()
    client = Client(api_key=settings.langsmith_api_key)
    examples = load_dataset(dataset_path)
    dataset_name = f"rag-eval-{settings.langsmith_project}"

    ds = client.create_dataset(dataset_name, description="RAG retrieval + generation eval")
    for ex in examples:
        if not ex.get("expected_doc_ids"):
            continue
        client.create_example(
            inputs={
                "question": ex["question"],
                "user_id": user_id,
                "k": k,
                "expected_doc_ids": ex["expected_doc_ids"],
                "expected_answer": ex.get("expected_answer", ""),
            },
            outputs={
                "expected_doc_ids": ex["expected_doc_ids"],
                "expected_answer": ex.get("expected_answer", ""),
            },
            dataset_id=ds.id,
        )

    def target(inputs: dict) -> dict:
        return _run_example(inputs, inputs.get("user_id", user_id), inputs.get("k", k))

    evaluate(
        target,
        data=ds.name,
        evaluators=[
            precision_pre_evaluator,
            precision_post_evaluator,
            faithfulness_evaluator,
            answer_relevance_evaluator,
        ],
        experiment_prefix="rag-eval",
        metadata={"k": k, "user_id": user_id},
        client=client,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument("--user-id", required=True, help="User UUID with indexed documents")
    parser.add_argument("--k", type=int, default=4, help="Precision@k cutoff")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--upload-langsmith", action="store_true")
    parser.add_argument("--project", default=None, help="LangSmith project override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    configure_observability()

    if args.upload_langsmith:
        settings = get_settings()
        project = args.project or settings.langsmith_project
        upload_and_evaluate(args.dataset, args.user_id, args.k, project)
        print(f"LangSmith evaluation uploaded to project '{project}'")
        return

    report = run_local(args.dataset, args.user_id, args.k)
    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    print(
        f"\nRetrieval precision@{summary['k']}: "
        f"pre-rerank={summary['precision_at_k_pre_rerank']:.2%} → "
        f"post-rerank={summary['precision_at_k_post_rerank']:.2%}"
    )


if __name__ == "__main__":
    main()
