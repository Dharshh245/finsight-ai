"""FinSight AI — Sentiment Agent (Member 4).

Yahoo Finance RSS headlines → FinBERT inference → aggregation.

Returns ONLY:
  {
    "sentiment_result": dict,
    "trace_log": List[str]
  }
"""

from __future__ import annotations

import logging
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Dict, List

import feedparser

from models.sentiment_model import predict_sentiment
from state import AgentState

logger = logging.getLogger(__name__)

# Prevent RSS fetch from hanging indefinitely.
socket.setdefaulttimeout(10)

RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
MAX_HEADLINES = 20
DEDUP_THRESHOLD = 0.72

TICKER_ALIASES: Dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
}


@dataclass
class ScoredHeadline:
    headline: str
    label: str  # positive | neutral | negative
    score: float  # confidence [0,1]
    signed_score: float = field(init=False)

    def __post_init__(self) -> None:
        sign = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(self.label, 0.0)
        self.signed_score = sign * self.score


def extract_ticker(query: str) -> str:
    q = (query or "").strip()

    m = re.search(r"\(([A-Z\^]{1,6})\)", q)
    if m:
        return m.group(1)

    caps = re.findall(r"\b\^?[A-Z]{2,6}\b", q)
    if caps:
        return caps[0]

    ql = q.lower()
    for alias, ticker in TICKER_ALIASES.items():
        if alias in ql:
            return ticker

    return "AAPL"


def _jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate_headlines(headlines: List[str], threshold: float = DEDUP_THRESHOLD) -> List[str]:
    unique: List[str] = []
    for candidate in headlines:
        c = (candidate or "").strip()
        if not c:
            continue
        if not any(_jaccard(c, kept) >= threshold for kept in unique):
            unique.append(c)
        if len(unique) >= MAX_HEADLINES:
            break
    return unique


def fetch_headlines(ticker: str) -> List[str]:
    try:
        feed = feedparser.parse(RSS_URL.format(ticker=ticker))
        titles: List[str] = []
        for entry in getattr(feed, "entries", []) or []:
            title = (entry.get("title") or "").strip()
            if title and len(title) > 10:
                titles.append(title)
        return titles[:MAX_HEADLINES]
    except Exception as exc:  # noqa: BLE001
        logger.warning("RSS fetch failed for %s: %s", ticker, exc)
        return []


def analyze_sentiment(headlines: List[str]) -> List[ScoredHeadline]:
    results: List[ScoredHeadline] = []
    for h in headlines[:MAX_HEADLINES]:
        try:
            pred = predict_sentiment(h)
            label = str(pred.get("label", "neutral")).lower().strip()
            score = float(pred.get("score", 0.5))
        except Exception:  # noqa: BLE001
            label, score = "neutral", 0.5

        if label not in {"positive", "neutral", "negative"}:
            label = "neutral"
        score = max(0.0, min(1.0, score))
        results.append(ScoredHeadline(headline=h, label=label, score=score))
    return results


def aggregate(scored: List[ScoredHeadline]) -> Dict[str, object]:
    if not scored:
        return {
            "score": 0.0,
            "trend": "neutral",
            "confidence": "low",
            "distribution": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
        }

    n = len(scored)
    total_weight = sum(s.score for s in scored)

    weighted_score = (
        sum(s.signed_score * s.score for s in scored) / total_weight
        if total_weight > 0
        else 0.0
    )

    counts: Dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    for s in scored:
        counts[s.label] = counts.get(s.label, 0) + 1
    distribution = {k: round(v / n, 3) for k, v in counts.items()}

    mean_conf = (total_weight / n) if n else 0.0
    if mean_conf >= 0.75:
        confidence = "high"
    elif mean_conf >= 0.55:
        confidence = "medium"
    else:
        confidence = "low"

    trend = "mixed"
    if weighted_score >= 0.25 and distribution.get("positive", 0.0) >= 0.45:
        trend = "bullish"
    elif weighted_score <= -0.25 and distribution.get("negative", 0.0) >= 0.45:
        trend = "bearish"
    elif abs(weighted_score) < 0.15 and distribution.get("neutral", 0.0) >= 0.50:
        trend = "neutral"

    return {
        "score": round(float(weighted_score), 4),
        "trend": trend,
        "confidence": confidence,
        "distribution": distribution,
    }


def run(state: AgentState) -> Dict[str, object]:
    trace_log: List[str] = []
    t0 = time.perf_counter()

    query = (state.get("query", "") or "").strip()
    trace_log.append(f"SentimentAgent | start | query='{query}'")

    if not query:
        return {
            "sentiment_result": {
                "score": 0.0,
                "trend": "neutral",
                "confidence": "low",
                "distribution": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                "summary": "No query provided",
                "top_headlines": [],
                "headline_count": 0,
                "ticker": "UNKNOWN",
                "elapsed_ms": 0.0,
            },
            "trace_log": trace_log,
        }

    ticker = extract_ticker(query)
    trace_log.append(f"SentimentAgent | ticker_extracted | ticker={ticker}")

    raw_headlines = fetch_headlines(ticker)
    trace_log.append(f"SentimentAgent | headlines_fetched | raw_count={len(raw_headlines)}")

    headlines = deduplicate_headlines(raw_headlines)
    trace_log.append(f"SentimentAgent | deduplication_complete | unique_count={len(headlines)}")

    if not headlines:
        return {
            "sentiment_result": {
                "score": 0.0,
                "trend": "neutral",
                "confidence": "low",
                "distribution": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                "summary": "No headlines found",
                "top_headlines": [],
                "headline_count": 0,
                "ticker": ticker,
                "elapsed_ms": 0.0,
            },
            "trace_log": trace_log,
        }

    scored = analyze_sentiment(headlines)
    trace_log.append(f"SentimentAgent | sentiment_scored | scored={len(scored)}")

    agg = aggregate(scored)
    trend = str(agg["trend"])
    score = float(agg["score"])
    confidence = str(agg["confidence"])
    distribution = agg["distribution"]

    top_headlines = [s.headline for s in sorted(scored, key=lambda x: x.score, reverse=True)[:5]]
    summary = f"Market sentiment for {ticker.upper()} is {trend} (confidence: {confidence}, score: {score:+.2f})."

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace_log.append(f"SentimentAgent | complete | elapsed={elapsed_ms}ms")

    return {
        "sentiment_result": {
            "score": round(score, 3),
            "trend": trend,
            "confidence": confidence,
            "distribution": distribution,
            "summary": summary,
            "top_headlines": top_headlines,
            "headline_count": len(scored),
            "ticker": ticker,
            "elapsed_ms": elapsed_ms,
        },
        "trace_log": trace_log,
    }


if __name__ == "__main__":
    print(run({"query": "NVDA news", "trace_log": []}))
