"""Run the AI-DUA-Bench evaluation for all three settings.

Run:  python evaluate.py [--top-k 5] [--out results]

This script is the evaluation harness for the final report. It runs
Settings A (no-RAG), B (RAG), and C (full DUA Guard) on all 40 benchmark
scenarios, scores every prediction against the human-authored
ground-truth labels, and writes:

  results/evaluation_results.json  -- metrics, config, failure analysis
  results/predictions.csv          -- one row per (scenario, setting)
  results/confusion_matrix.csv     -- 4x4 confusion counts per setting

Metrics (definitions are fixed here so the report numbers are
reproducible):
  * accuracy: correct / 40.
  * per-label precision / recall / F1, plus macro-F1.
  * FALSE APPROVAL RATE (primary safety metric): true Prohibited cases
    predicted as Allowed / all true Prohibited cases. A false approval
    is the most consequential error: the system green-lights an action
    the agreement prohibits.
  * unsafe rate: true Prohibited cases predicted as Allowed OR Requires
    Approval / all true Prohibited cases. Predicting Requires Approval
    for a prohibited action is less bad than Allowed, but still unsafe
    because it implies approval could permit the action.
  * citation accuracy: among predictions that produced at least one
    citation, the fraction where at least one cited clause id is in the
    scenario's ground-truth evidence set. Setting A produces no
    citations by design, so its citation accuracy is reported as null
    ("not applicable"). A second figure, correct-citation coverage,
    reports correct citations over ALL 40 scenarios.
  * abstention rate: fraction predicted as Insufficient Information /
    Requires Human Review.
  * committed accuracy: accuracy computed only over predictions that
    are not abstentions (shows the accuracy/safety tradeoff honestly:
    abstaining lowers headline accuracy but raises safety).
  * per-category accuracy, confusion matrix, mean latency per scenario.

Engine honesty: the script records which classification engine and
retrieval backend actually ran. By default it uses the transparent
evidence-grounded fallback (see dua_guard/classify.py). The local
FLAN-T5 model is attempted ONLY when DUA_CHECK_LLM=1 is set; if the
model cannot be loaded, the fallback runs and the failure reason is
recorded. Numbers are never presented as LLM numbers when the fallback
ran.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dua_guard import LABELS
from dua_guard.classify import predict
from dua_guard.data_loader import get_clauses, load_benchmark
from dua_guard.llm_client import LocalLLM, get_engine_info

PROJECT_ROOT = Path(__file__).resolve().parent


def _per_label_metrics(
    pairs: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Precision/recall/F1 per label from (true, predicted) pairs."""
    metrics: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }
    return metrics


def _likely_cause(true_label: str, predicted: str) -> str:
    """Rule-based failure-cause tag from the true/predicted pair only.

    These tags describe the error pattern; they are not a causal
    diagnosis of any single case. The report's failure analysis adds
    case-level reading of the cited clauses.
    """
    if true_label == "Prohibited" and predicted == "Allowed":
        return "FALSE APPROVAL: prohibition evidence missed or outweighed"
    if true_label == "Prohibited" and predicted == "Requires Approval":
        return "unsafe downgrade: prohibition treated as approval-gated"
    if true_label == "Prohibited":
        return "safe abstention on prohibited case (counts as accuracy error)"
    if true_label == "Allowed" and predicted == "Insufficient Information / Requires Human Review":
        return "over-abstention: permission evidence too weak or conflicting"
    if true_label == "Allowed":
        return "over-conservative: permission evidence missed"
    if true_label == "Requires Approval" and predicted == "Allowed":
        return "FALSE APPROVAL of approval-gated action (unsafe)"
    if true_label == "Requires Approval" and predicted == "Insufficient Information / Requires Human Review":
        return "abstention instead of approval routing"
    if true_label == "Requires Approval":
        return "over-conservative: approval case treated as prohibited"
    if true_label == "Insufficient Information / Requires Human Review" and predicted == "Allowed":
        return "FALSE APPROVAL on underspecified case (unsafe)"
    return "committed to a label without sufficient information"


def evaluate_setting(
    setting: str,
    scenarios: list[dict[str, Any]],
    top_k: int,
    use_llm: bool,
    llm: LocalLLM | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one setting over all scenarios; return (metrics, prediction rows)."""
    rows: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    citation_correct = 0
    citation_total = 0
    abstentions = 0
    llm_fallbacks = 0
    overrides = 0
    per_category: dict[str, list[bool]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    backend_seen: str | None = None

    # Cache clause sets per (dataset, policy) so retrieval setup is shared
    # across scenarios with the same pair.
    clause_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for scenario in scenarios:
        key = (scenario["dataset"], scenario["ai_tool"])
        if key not in clause_cache:
            clause_cache[key] = get_clauses(scenario["dataset"], scenario["ai_tool"])
        clauses = clause_cache[key]
        start = time.perf_counter()
        prediction = predict(
            scenario["action"],
            clauses,
            setting=setting,
            top_k=top_k,
            use_llm=use_llm,
            llm=llm,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        predicted = prediction["label"]
        true_label = scenario["ground_truth"]
        correct = predicted == true_label
        pairs.append((true_label, predicted))
        per_category[scenario["category"]].append(correct)
        confusion[(true_label, predicted)] += 1
        if predicted == "Insufficient Information / Requires Human Review":
            abstentions += 1
        if prediction.get("llm_fallback"):
            llm_fallbacks += 1
        if prediction.get("override_reason"):
            overrides += 1
        if prediction.get("retrieval_backend"):
            backend_seen = prediction["retrieval_backend"]

        cited_ids = [c["clause_id"] for c in prediction.get("citations", [])]
        evidence_ids = scenario["evidence_clause_ids"]
        has_citation = len(cited_ids) > 0
        citation_ok: bool | None = None
        if has_citation:
            citation_total += 1
            citation_ok = any(cid in evidence_ids for cid in cited_ids)
            if citation_ok:
                citation_correct += 1

        top_retrieved = (
            prediction["retrieved"][0]["clause_id"] if prediction.get("retrieved") else ""
        )
        rows.append(
            {
                "scenario_id": scenario["id"],
                "setting": setting,
                "category": scenario["category"],
                "dataset": scenario["dataset"],
                "ai_tool": scenario["ai_tool"],
                "ground_truth": true_label,
                "predicted": predicted,
                "correct": correct,
                "confidence": prediction["confidence"],
                "cited_clause_ids": ";".join(cited_ids),
                "evidence_clause_ids": ";".join(evidence_ids),
                "citation_correct": citation_ok if citation_ok is not None else "",
                "top_retrieved_clause_id": top_retrieved,
                "override_reason": prediction.get("override_reason") or "",
                "used_llm": prediction.get("used_llm", False),
                "latency_ms": round(latency_ms, 1),
            }
        )
        if not correct:
            failures.append(
                {
                    "scenario_id": scenario["id"],
                    "category": scenario["category"],
                    "true_label": true_label,
                    "predicted": predicted,
                    "cited_clause_ids": cited_ids,
                    "evidence_clause_ids": evidence_ids,
                    "top_retrieved_clause_id": top_retrieved,
                    "override_reason": prediction.get("override_reason"),
                    "likely_cause": _likely_cause(true_label, predicted),
                    "action": scenario["action"],
                }
            )

    total = len(scenarios)
    correct_count = sum(1 for t, p in pairs if t == p)
    true_prohibited = [p for t, p in pairs if t == "Prohibited"]
    false_approvals = sum(1 for p in true_prohibited if p == "Allowed")
    unsafe = sum(1 for p in true_prohibited if p in ("Allowed", "Requires Approval"))
    committed = [(t, p) for t, p in pairs if p != "Insufficient Information / Requires Human Review"]
    committed_correct = sum(1 for t, p in committed if t == p)

    per_label = _per_label_metrics(pairs)
    macro_f1 = round(sum(m["f1"] for m in per_label.values()) / len(per_label), 3)

    metrics: dict[str, Any] = {
        "setting": setting,
        "n": total,
        "accuracy": round(correct_count / total, 3) if total else 0.0,
        "correct": correct_count,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "false_approval_rate": round(false_approvals / len(true_prohibited), 3) if true_prohibited else 0.0,
        "false_approvals": false_approvals,
        "true_prohibited": len(true_prohibited),
        "unsafe_rate": round(unsafe / len(true_prohibited), 3) if true_prohibited else 0.0,
        "unsafe_count": unsafe,
        "citation_accuracy": round(citation_correct / citation_total, 3) if citation_total else None,
        "citation_correct": citation_correct,
        "citation_total": citation_total,
        "correct_citation_coverage": round(citation_correct / total, 3) if total else 0.0,
        "abstention_rate": round(abstentions / total, 3) if total else 0.0,
        "abstentions": abstentions,
        "committed_accuracy": round(committed_correct / len(committed), 3) if committed else None,
        "committed_n": len(committed),
        "per_category_accuracy": {
            category: round(sum(values) / len(values), 3)
            for category, values in sorted(per_category.items())
        },
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "llm_fallbacks": llm_fallbacks,
        "conservative_overrides": overrides,
        "retrieval_backend": backend_seen,
        "confusion": {
            f"{true_label} -> {predicted}": count
            for (true_label, predicted), count in sorted(confusion.items())
        },
        "failures": failures,
    }
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DUA Guard on AI-DUA-Bench.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved clauses per query.")
    parser.add_argument("--out", type=str, default="results", help="Output directory.")
    args = parser.parse_args()

    check_llm = os.environ.get("DUA_CHECK_LLM", "").strip() == "1"
    engine_info = get_engine_info(check_llm=check_llm)
    use_llm = bool(engine_info.get("llm_available")) and check_llm
    llm = LocalLLM() if use_llm else None
    if use_llm and llm is not None and not llm.is_available():
        # The check said available but the load failed on second attempt;
        # fall back honestly rather than crashing the evaluation.
        engine_info = get_engine_info(check_llm=True)
        use_llm = False
        llm = None

    benchmark = load_benchmark()
    scenarios = benchmark["scenarios"]
    print(f"Loaded {len(scenarios)} scenarios.")
    print(f"Engine: {engine_info['engine']} -- {engine_info['reason']}")
    print(f"Retrieval: TF-IDF/embedding backend selected per predict() call "
          f"(DUA_RETRIEVAL_BACKEND={os.environ.get('DUA_RETRIEVAL_BACKEND', 'numpy (default)')}).")
    print(f"LLM use: {'enabled' if use_llm else 'disabled (fallback engine)'}")

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for setting in ("A", "B", "C"):
        metrics, rows = evaluate_setting(setting, scenarios, args.top_k, use_llm, llm)
        all_metrics[setting] = metrics
        all_rows.extend(rows)
        for true_label in LABELS:
            for predicted in LABELS:
                count = sum(
                    1 for r in rows if r["ground_truth"] == true_label and r["predicted"] == predicted
                )
                confusion_rows.append(
                    {
                        "setting": setting,
                        "true_label": true_label,
                        "predicted_label": predicted,
                        "count": count,
                    }
                )
        far = metrics["false_approval_rate"]
        print(
            f"Setting {setting}: accuracy={metrics['accuracy']:.3f} "
            f"false_approval_rate={far:.3f} "
            f"citation_accuracy={metrics['citation_accuracy']} "
            f"abstention_rate={metrics['abstention_rate']:.3f} "
            f"mean_latency={metrics['mean_latency_ms']}ms"
        )

    summary = {
        "benchmark": benchmark["benchmark_name"],
        "n_scenarios": len(scenarios),
        "top_k": args.top_k,
        "engine_info": engine_info,
        "use_llm": use_llm,
        "retrieval_backend_env": os.environ.get("DUA_RETRIEVAL_BACKEND", "numpy (default)"),
        "metric_definitions": {
            "false_approval_rate": "true Prohibited predicted as Allowed / all true Prohibited (primary safety metric)",
            "unsafe_rate": "true Prohibited predicted as Allowed or Requires Approval / all true Prohibited",
            "citation_accuracy": "among predictions with >=1 citation, fraction with >=1 cited clause in the scenario evidence set; null for Setting A (no citations by design)",
            "abstention_rate": "fraction predicted as Insufficient Information / Requires Human Review",
            "committed_accuracy": "accuracy over non-abstention predictions only",
        },
        "settings": all_metrics,
    }
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    prediction_fields = [
        "scenario_id", "setting", "category", "dataset", "ai_tool",
        "ground_truth", "predicted", "correct", "confidence",
        "cited_clause_ids", "evidence_clause_ids", "citation_correct",
        "top_retrieved_clause_id", "override_reason", "used_llm", "latency_ms",
    ]
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["setting", "true_label", "predicted_label", "count"]
        )
        writer.writeheader()
        writer.writerows(confusion_rows)

    print(f"Wrote {out_dir / 'evaluation_results.json'}")
    print(f"Wrote {out_dir / 'predictions.csv'} ({len(all_rows)} rows)")
    print(f"Wrote {out_dir / 'confusion_matrix.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
