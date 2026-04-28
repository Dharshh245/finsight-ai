"""
FinSight AI — Chainlit UI

Single integration point:
  - orchestrator.graph.run_graph
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Optional

import chainlit as cl

from orchestrator.graph import run_graph
from ui.trace_panel import format_trace


async def run_pipeline(query: str, image_path: Optional[str] = None) -> dict:
    return await asyncio.to_thread(run_graph, {"query": query, "image_data": image_path})


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("ready", True)
    await cl.Message(
        content=(
            "## Welcome to **FinSight AI**\n\n"
            "Ask about a stock or financial event. I’ll run the full pipeline (RAG/SQL/Chart/Sentiment/Forecast) "
            "and show the results here."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a query.").send()
        return

    image_path: Optional[str] = None
    for element in message.elements:
        if hasattr(element, "path") and element.path:
            suffix = Path(element.path).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                image_path = element.path
                break

    async with cl.Step(name="Running FinSight pipeline…", type="run") as step:
        step.input = query
        try:
            result = await run_pipeline(query, image_path)
            step.output = "Pipeline completed."
        except Exception as exc:  # noqa: BLE001
            step.output = f"Pipeline error: {exc}"
            await cl.Message(
                content=f"❌ Pipeline failed.\n\n```\n{traceback.format_exc()}\n```"
            ).send()
            return

    sentiment = result.get("sentiment_result", {}) or {}
    sql = result.get("sql_result", None)
    chart = result.get("chart_path", None)
    trace = result.get("trace_log", []) or []

    # ✔ Sentiment section (trend, score, summary)
    trend = "neutral"
    score = 0.0
    summary = ""
    if isinstance(sentiment, dict):
        trend = str(sentiment.get("trend", "neutral")).lower()
        try:
            score = float(sentiment.get("score", 0.0))
        except Exception:  # noqa: BLE001
            score = 0.0
        summary = str(sentiment.get("summary", "") or "")

    await cl.Message(
        content=(
            "## 📊 Sentiment\n\n"
            f"**Trend:** `{trend.upper()}`\n"
            f"**Score:** `{score:+.3f}`\n\n"
            f"**Summary:** {summary if summary else '_No summary available_'}"
        )
    ).send()

    # ✔ SQL section
    if sql is None or sql == "":
        await cl.Message(content="## 🗄️ SQL\n\nNo SQL data").send()
    else:
        await cl.Message(content=f"## 🗄️ SQL\n\n{sql}").send()

    # ✔ Chart (only if exists)
    if isinstance(chart, str) and chart.strip() and Path(chart.strip()).exists():
        await cl.Message(
            content="## 📈 Chart",
            elements=[cl.Image(path=chart.strip())],
        ).send()
    else:
        await cl.Message(content="## 📈 Chart\n\nChart not available").send()

    # ✔ Trace panel (code block, newline-joined)
    trace_text = format_trace(trace if isinstance(trace, list) else [str(trace)])
    await cl.Message(content=f"## 🔍 Trace\n\n```\n{trace_text}\n```").send()

