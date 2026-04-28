"""
RAGAS evaluation — Groq-backed (no OpenAI).

This script is intentionally lightweight and safe to import in CI.
Run manually when you have a dataset prepared.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def run_ragas_eval(dataset: Any, metrics: Optional[list] = None) -> Dict[str, Any]:
    """
    Evaluate a RAG dataset with RAGAS using Groq LLM.

    Parameters
    ----------
    dataset:
        A RAGAS-compatible dataset (e.g., from `ragas.dataset.Schema` / HF dataset).
    metrics:
        Optional list of ragas metrics. If None, ragas defaults are used.
    """
    from langchain_groq import ChatGroq
    from ragas import evaluate

    llm = ChatGroq(model="llama3-70b-8192")
    if metrics is None:
        return evaluate(dataset, llm=llm)  # type: ignore[no-any-return]
    return evaluate(dataset, metrics=metrics, llm=llm)  # type: ignore[no-any-return]


if __name__ == "__main__":
    raise SystemExit(
        "Provide a RAGAS dataset object and call run_ragas_eval(dataset). "
        "This module is intentionally not wired to any live data by default."
    )

