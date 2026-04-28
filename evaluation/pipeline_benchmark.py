"""
Pipeline benchmark — measures execution time of FinSight pipeline.
"""

import time
from orchestrator.graph import run_graph


def run_benchmark(query: str) -> None:
    print("\n📊 Benchmark Results:\n")

    start = time.perf_counter()

    result = run_graph({"query": query, "image_data": None})

    end = time.perf_counter()

    total_time = (end - start) * 1000  # ms

    # Extract optional timings
    sentiment_time = result.get("sentiment_result", {}).get("elapsed_ms", "N/A")
    sql_time = result.get("sql_result", {}).get("elapsed_ms", "N/A") if isinstance(result.get("sql_result"), dict) else "N/A"
    chart_time = result.get("chart_result", {}).get("elapsed_ms", "N/A") if isinstance(result.get("chart_result"), dict) else "N/A"

    print(f"Total pipeline: {round(total_time, 2)} ms")
    print(f"Sentiment: {sentiment_time} ms")
    print(f"SQL: {sql_time} ms")
    print(f"Chart: {chart_time} ms")


if __name__ == "__main__":
    run_benchmark("AAPL stock analysis")