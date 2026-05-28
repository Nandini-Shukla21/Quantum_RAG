"""Benchmark the local Research RAG pipeline on at least 20 QML queries.

The benchmark builds weakly supervised evaluation queries from
metadata/paper_metadata.csv. Each query targets one paper title and uses that
paper's PDF filename as the relevant source for Recall@K and MRR.
"""

import csv
from pathlib import Path
from typing import Dict, List

from config import (
    BENCHMARK_REPORT_PATH,
    BENCHMARK_RESULTS_PATH,
    EVALUATION_RESULTS_PATH,
    PAPER_METADATA_PATH,
    PAPERS_DIR,
)
from evaluator import RAGEvaluator
from rag_pipeline import ResearchRAGPipeline, run_startup_diagnostics


def load_benchmark_queries(limit: int = 20) -> List[Dict]:
    """Create benchmark queries from metadata for the indexed papers."""

    if not PAPER_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {PAPER_METADATA_PATH}")

    rows: List[Dict] = []
    with open(PAPER_METADATA_PATH, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            title = (row.get("title") or "").strip()
            pdf_file = (row.get("pdf_file") or "").strip()
            summary = (row.get("summary") or "").strip()
            if not title or not pdf_file:
                continue
            if not (PAPERS_DIR / pdf_file).exists():
                continue

            rows.append(
                {
                    "query": (
                        f"What are the main research contribution, method, and limitation of the paper titled "
                        f"{title}? Answer using the retrieved context."
                    ),
                    "relevant_sources": pdf_file,
                    "title": title,
                    "summary_preview": summary[:240],
                }
            )

            if len(rows) >= limit:
                break

    if len(rows) < limit:
        raise ValueError(
            f"Only found {len(rows)} benchmarkable papers with local PDFs; expected at least {limit}."
        )

    return rows


def write_summary_report(results: List[Dict], summary: Dict, output_path: Path = BENCHMARK_REPORT_PATH) -> None:
    """Write a compact text report for benchmark runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Quantum Machine Learning RAG Benchmark Summary",
        "=" * 52,
        f"Queries evaluated: {summary['num_queries']}",
        f"Mean Recall@5: {summary['mean_recall_at_5']:.4f}",
        f"Mean Recall@10: {summary['mean_recall_at_10']:.4f}",
        f"Mean MRR: {summary['mean_mrr']:.4f}",
        f"Mean Faithfulness: {summary['mean_faithfulness']:.4f}",
        f"Mean Context Relevance: {summary['mean_context_relevance']:.4f}",
        f"Mean Answer Relevance: {summary['mean_answer_relevance']:.4f}",
        f"Mean Extractiveness Score: {summary['mean_extractiveness_score']:.4f}",
        f"Mean Response Time Seconds: {summary['mean_response_time_seconds']:.2f}",
        "",
        "Per-query results:",
    ]

    for index, row in enumerate(results, start=1):
        lines.extend(
            [
                f"{index}. {row['query']}",
                f"   Recall@5={float(row['recall_at_5']):.4f}, "
                f"Recall@10={float(row['recall_at_10']):.4f}, "
                f"MRR={float(row['mrr']):.4f}, "
                f"Extractiveness={float(row['extractiveness_score']):.4f}",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    diagnostics = run_startup_diagnostics(load_model=False)
    for warning in diagnostics["warnings"]:
        print(f"Warning: {warning}")

    pdf_count = len(list(PAPERS_DIR.glob("*.pdf")))
    print(f"Local PDFs found: {pdf_count}")

    benchmark_queries = load_benchmark_queries(limit=20)
    print(f"Benchmark queries created: {len(benchmark_queries)}")

    pipeline = ResearchRAGPipeline(fail_soft_llm=True)
    evaluator = RAGEvaluator(pipeline=pipeline)

    results = evaluator.evaluate_rows(
        benchmark_queries,
        output_path=BENCHMARK_RESULTS_PATH,
    )
    evaluator.save_results(results, EVALUATION_RESULTS_PATH)

    summary = evaluator.summarize(results)
    write_summary_report(results, summary)

    print(f"Saved benchmark results to {BENCHMARK_RESULTS_PATH}")
    print(f"Saved evaluation results to {EVALUATION_RESULTS_PATH}")
    print(f"Saved summary report to {BENCHMARK_REPORT_PATH}")
    print(summary)


if __name__ == "__main__":
    main()

