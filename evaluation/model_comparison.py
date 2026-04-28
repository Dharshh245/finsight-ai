"""
Model comparison summary table (structured placeholders).

Prints:
  - Fraud model metrics (AUC, F1, Recall)
  - FinBERT base vs fine-tuned accuracy
  - Price forecaster accuracy
"""

from __future__ import annotations

from typing import Dict, Tuple


def _fmt_row(name: str, metrics: Dict[str, float]) -> Tuple[str, str, str, str]:
    def f(x: float) -> str:
        return f"{x:.3f}"

    return (
        name,
        f(metrics.get("auc", float("nan"))),
        f(metrics.get("f1", float("nan"))),
        f(metrics.get("recall", float("nan"))),
    )


def main() -> None:
    # Placeholders — replace with real values when available.
    fraud_metrics = {"auc": 0.910, "f1": 0.780, "recall": 0.740}

    finbert_base_acc = 0.840
    finbert_finetuned_acc = 0.890

    forecaster_accuracy = 0.620

    rows = []
    rows.append(_fmt_row("Fraud model", fraud_metrics))
    rows.append(("FinBERT (base)", f"{finbert_base_acc:.3f}", "-", "-",))
    rows.append(("FinBERT (fine-tuned)", f"{finbert_finetuned_acc:.3f}", "-", "-",))
    rows.append(("Price forecaster", f"{forecaster_accuracy:.3f}", "-", "-",))

    headers = ("Model", "AUC / Acc", "F1", "Recall")
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(4)]

    def line(sep: str = "-") -> str:
        return "+".join(sep * (w + 2) for w in widths).join(["+", "+"])

    print(line("-"))
    print(
        "| "
        + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(4))
        + " |"
    )
    print(line("="))
    for r in rows:
        print("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(4)) + " |")
    print(line("-"))


if __name__ == "__main__":
    main()

