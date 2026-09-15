"""## Executive summary (read this first)

The reasoning agent selects only as-of-safe documents using a deterministic, auditable blend of
target relevance and recency. These tests exercise selection alone; forecasting stays untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from baselines.reasoning_agent import rank_documents, read_corpus


def _doc(doc_id: str, timestamp: str, text: str) -> dict[str, str]:
    return {
        "doc_id": doc_id,
        "timestamp": timestamp,
        "doc_type": "synthetic",
        "source": "synthetic",
        "text": text,
    }


def test_relevant_older_document_beats_unrelated_recent_document() -> None:
    docs = [
        _doc("recent", "2024-05-31", "An unrelated announcement."),
        _doc("relevant", "2024-04-01", "Rates moved as the UST 10Y outlook changed."),
    ]
    ranked = rank_documents(docs, ["UST_10Y"], ["rates_daily"], "2024-05-31", 2)
    assert [doc["doc_id"] for doc in ranked] == ["relevant", "recent"]
    assert ranked[0]["selection_keywords"] == ["10y", "rates", "ust"]


def test_recency_breaks_equal_relevance_toward_the_newer_document() -> None:
    docs = [
        _doc("older", "2024-04-01", "The rates outlook."),
        _doc("newer", "2024-05-01", "The rates outlook."),
    ]
    ranked = rank_documents(docs, [], ["rates_daily"], "2024-05-31", 2)
    assert [doc["doc_id"] for doc in ranked] == ["newer", "older"]


def test_doc_id_makes_exact_ties_deterministic() -> None:
    docs = [
        _doc("z-document", "2024-05-01", "Same text."),
        _doc("a-document", "2024-05-01", "Same text."),
    ]
    ranked = rank_documents(docs, ["UST_10Y"], [], "2024-05-31", 2)
    assert [doc["doc_id"] for doc in ranked] == ["a-document", "z-document"]


def test_read_corpus_filters_after_asof_before_ranking_and_honours_budget(
    tmp_path: Path,
) -> None:
    records = [
        ("future", "2024-06-01", "UST 10Y rates"),
        ("relevant", "2024-04-01", "UST 10Y rates"),
        ("recent", "2024-05-31", "Unrelated announcement"),
    ]
    documents = []
    for doc_id, timestamp, content in records:
        filename = f"{doc_id}.txt"
        (tmp_path / filename).write_text(content, encoding="utf-8")
        documents.append(
            {
                "doc_id": doc_id,
                "timestamp": timestamp,
                "file": filename,
                "doc_type": "synthetic",
                "source": "synthetic",
            }
        )
    (tmp_path / "corpus_index.json").write_text(
        json.dumps({"documents": documents}), encoding="utf-8"
    )

    selected, excluded, over_budget = read_corpus(
        tmp_path, "2024-05-31", ["UST_10Y"], ["rates_daily"], max_docs=1
    )
    assert [doc["doc_id"] for doc in selected] == ["relevant"]
    assert excluded == 1
    assert over_budget == 1


def test_zero_document_budget_selects_nothing() -> None:
    docs = [_doc("relevant", "2024-05-01", "UST 10Y rates")]
    assert rank_documents(docs, ["UST_10Y"], ["rates_daily"], "2024-05-31", 0) == []
