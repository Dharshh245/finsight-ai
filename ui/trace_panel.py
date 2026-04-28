from __future__ import annotations


def format_trace(trace_log) -> str:
    return "\n".join(trace_log) if trace_log else "No trace"

