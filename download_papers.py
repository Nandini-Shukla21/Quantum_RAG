"""Build a curated Quantum Machine Learning paper corpus for RAG.

This script deliberately downloads a small, high-signal pilot corpus before
scaling. It searches targeted QML topics, applies strict relevance filtering,
deduplicates by arXiv id and title, handles arXiv rate limits with backoff, and
writes clean metadata plus dataset statistics.

Run:
    python download_papers.py
"""

from __future__ import annotations

import csv
import re
import ssl
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError

import arxiv
import certifi


# ---------------------------------------------------------------------------
# SSL
# ---------------------------------------------------------------------------
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAPERS_DIR = Path("papers")
METADATA_DIR = Path("metadata")
METADATA_CSV_PATH = METADATA_DIR / "paper_metadata.csv"
DATASET_SUMMARY_PATH = METADATA_DIR / "dataset_summary.txt"

QUERIES = [
    "quantum kernel learning",
    "quantum support vector machine",
    "variational quantum classifier",
    "quantum neural network",
    "quantum feature map",
]

MAX_PAPERS_PER_TOPIC = 6
MAX_SEARCH_RESULTS_PER_TOPIC = 60
TARGET_TOTAL_PAPERS = 30

ARXIV_CLIENT_CONFIG = {
    "page_size": 5,
    "delay_seconds": 10,
    "num_retries": 5,
}

BACKOFF_SECONDS = [30, 60, 90, 120, 150]
SLEEP_AFTER_SUCCESSFUL_DOWNLOAD_SECONDS = 2
SLEEP_AFTER_TOPIC_SECONDS = 15

RELEVANCE_TERMS = [
    "quantum kernel",
    "kernel learning",
    "support vector",
    "classifier",
    "classification",
    "variational",
    "quantum neural network",
    "feature map",
    "machine learning",
    "quantum learning",
    "quantum classifier",
    "hybrid quantum",
    "variational circuit",
    "quantum feature",
]

REJECT_TERMS = [
    "video generation",
    "vision language",
    "astronomy",
    "gravitational physics",
    "distributed simulation",
    "hardware benchmarking",
    "large language model",
]

METADATA_COLUMNS = [
    "paper_id",
    "title",
    "authors",
    "published",
    "updated",
    "summary",
    "categories",
    "primary_category",
    "pdf_url",
    "search_keyword",
    "pdf_file",
]


def log(message: str) -> None:
    print(message, flush=True)


def ensure_directories() -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(title: object) -> str:
    return normalize_text(title)


def normalize_arxiv_id(paper_id: object) -> str:
    """Deduplicate v1/v2/v3 variants as the same paper."""

    return re.sub(r"v\d+$", "", str(paper_id or "").strip())


def is_http_429(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return (
        "429" in text
        or "too many requests" in text
        or isinstance(error, HTTPError)
        and getattr(error, "code", None) == 429
    )


def is_network_error(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return (
        isinstance(error, (HTTPError, URLError, TimeoutError, ssl.SSLError))
        or "connection" in text
        or "timeout" in text
        or "network" in text
        or "ssl" in text
        or "temporarily unavailable" in text
    )


def terms_found(text: str, terms: Iterable[str]) -> list[str]:
    normalized = normalize_text(text)
    return [term for term in terms if term in normalized]


def relevance_decision(title: object, summary: object) -> tuple[bool, str]:
    searchable_text = f"{title or ''}\n{summary or ''}"

    matched_relevance_terms = terms_found(searchable_text, RELEVANCE_TERMS)
    if not matched_relevance_terms:
        return False, "no required QML relevance term"

    matched_reject_terms = terms_found(searchable_text, REJECT_TERMS)
    if matched_reject_terms:
        return False, f"rejected noisy topic: {', '.join(matched_reject_terms)}"

    return True, f"matched: {', '.join(matched_relevance_terms)}"


def safe_short_id(result: arxiv.Result) -> str:
    try:
        paper_id = result.get_short_id()
    except Exception:
        paper_id = ""
    return str(paper_id or "").strip()


def safe_text(value: object) -> str:
    return "" if value is None else str(value).replace("\r", " ").strip()


def safe_isoformat(value: object) -> str:
    try:
        return value.isoformat() if value else ""
    except Exception:
        return safe_text(value)


def metadata_from_result(result: arxiv.Result, search_keyword: str, pdf_file: str) -> dict[str, str]:
    """Create robust CSV-safe metadata from an arXiv result."""

    try:
        authors = ", ".join(author.name for author in getattr(result, "authors", []) if author.name)
    except Exception:
        authors = ""

    try:
        categories = ", ".join(getattr(result, "categories", []) or [])
    except Exception:
        categories = ""

    return {
        "paper_id": safe_short_id(result),
        "title": safe_text(getattr(result, "title", "")),
        "authors": authors,
        "published": safe_isoformat(getattr(result, "published", "")),
        "updated": safe_isoformat(getattr(result, "updated", "")),
        "summary": safe_text(getattr(result, "summary", "")),
        "categories": categories,
        "primary_category": safe_text(getattr(result, "primary_category", "")),
        "pdf_url": safe_text(getattr(result, "pdf_url", "")),
        "search_keyword": search_keyword,
        "pdf_file": pdf_file,
    }


def is_valid_metadata(row: dict[str, str]) -> tuple[bool, str]:
    if not row.get("paper_id"):
        return False, "missing paper_id"
    if not row.get("title"):
        return False, "missing title"
    if not row.get("pdf_url"):
        return False, "missing pdf_url"
    if not row.get("pdf_file"):
        return False, "missing pdf_file"
    return True, "ok"


def make_search(query: str) -> arxiv.Search:
    return arxiv.Search(
        query=query,
        max_results=MAX_SEARCH_RESULTS_PER_TOPIC,
        sort_by=arxiv.SortCriterion.Relevance,
    )


def make_client() -> arxiv.Client:
    return arxiv.Client(**ARXIV_CLIENT_CONFIG)


def fetch_topic_results(client: arxiv.Client, query: str) -> list[arxiv.Result] | None:
    """Fetch arXiv results with explicit 429 exponential backoff.

    Returns None when the topic should be skipped after repeated failures.
    """

    search = make_search(query)

    for attempt, wait_seconds in enumerate([0] + BACKOFF_SECONDS, start=1):
        if wait_seconds:
            log(f"Rate limit/network retry for topic '{query}'. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)

        try:
            results = list(client.results(search))
            if not results:
                log(f"Skipped topic '{query}': empty search results.")
            return results

        except Exception as exc:
            if is_http_429(exc):
                log(f"HTTP 429 while searching '{query}' (attempt {attempt}/6).")
            elif is_network_error(exc):
                log(f"Network error while searching '{query}' (attempt {attempt}/6): {exc}")
            else:
                log(f"Unexpected arXiv error while searching '{query}': {exc}")
                log(traceback.format_exc(limit=1).strip())
                return None

    log(f"Skipped topic '{query}': failed after HTTP/network retries.")
    return None


def download_pdf_if_needed(result: arxiv.Result, pdf_file: str) -> tuple[bool, str]:
    """Download PDF unless already present.

    Returns (downloaded_now, status_message).
    """

    pdf_path = PAPERS_DIR / pdf_file
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return False, f"Skipped existing PDF: {pdf_path}"

    try:
        result.download_pdf(dirpath=str(PAPERS_DIR), filename=pdf_file)
        time.sleep(SLEEP_AFTER_SUCCESSFUL_DOWNLOAD_SECONDS)
        return True, f"Downloaded: {pdf_path}"
    except Exception as exc:
        return False, f"download failure: {exc}"


def write_metadata(rows: list[dict[str, str]]) -> None:
    with open(METADATA_CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_summary(
    rows: list[dict[str, str]],
    papers_downloaded: int,
    duplicates_removed: int,
    papers_per_topic: dict[str, int],
    rejection_counts: Counter,
) -> None:
    lines = [
        "Quantum Machine Learning Corpus Summary",
        "=" * 48,
        f"papers downloaded: {papers_downloaded}",
        f"duplicates removed: {duplicates_removed}",
        "",
        "papers per topic:",
    ]

    for query in QUERIES:
        lines.append(f"- {query}: {papers_per_topic.get(query, 0)}")

    lines.extend(["", "rejections:"])
    if rejection_counts:
        for reason, count in rejection_counts.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", f"final paper count: {len(rows)}"])
    DATASET_SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def process_topic(
    client: arxiv.Client,
    query: str,
    rows: list[dict[str, str]],
    seen_arxiv_ids: set[str],
    seen_titles: set[str],
    papers_per_topic: dict[str, int],
    rejection_counts: Counter,
) -> tuple[int, int]:
    """Process one query topic.

    Returns:
        (papers_downloaded_now, duplicates_removed_now)
    """

    log("")
    log("=" * 72)
    log(f"Searching topic: {query}")
    log("=" * 72)

    results = fetch_topic_results(client, query)
    if results is None:
        rejection_counts["topic skipped after arXiv failures"] += 1
        return 0, 0
    if not results:
        rejection_counts["empty search results"] += 1
        return 0, 0

    downloaded_count = 0
    duplicates_removed = 0

    for result in results:
        if papers_per_topic[query] >= MAX_PAPERS_PER_TOPIC:
            log(f"Skipped remaining results for '{query}': topic quota reached.")
            break
        if len(rows) >= TARGET_TOTAL_PAPERS:
            log("Skipped remaining results: target corpus size reached.")
            break

        paper_id = safe_short_id(result)
        base_paper_id = normalize_arxiv_id(paper_id)
        title = safe_text(getattr(result, "title", ""))
        summary = safe_text(getattr(result, "summary", ""))
        normalized_title = normalize_title(title)

        if not paper_id or not title:
            rejection_counts["corrupted metadata"] += 1
            log(f"Rejected: corrupted metadata for result with id '{paper_id or 'unknown'}'.")
            continue

        if base_paper_id in seen_arxiv_ids:
            duplicates_removed += 1
            log(f"Duplicate: arXiv id {paper_id} | {title}")
            continue

        if normalized_title in seen_titles:
            duplicates_removed += 1
            log(f"Duplicate: title match {paper_id} | {title}")
            continue

        accepted, reason = relevance_decision(title, summary)
        if not accepted:
            rejection_counts[reason] += 1
            log(f"Rejected: {paper_id} | {title} | {reason}")
            continue

        pdf_file = f"{paper_id}.pdf"
        row = metadata_from_result(result, query, pdf_file)
        valid_metadata, metadata_reason = is_valid_metadata(row)
        if not valid_metadata:
            rejection_counts[f"corrupted metadata: {metadata_reason}"] += 1
            log(f"Rejected: {paper_id} | {title} | corrupted metadata: {metadata_reason}")
            continue

        log(f"Accepted: {paper_id} | {title} | {reason}")
        downloaded_now, download_message = download_pdf_if_needed(result, pdf_file)
        if download_message.startswith("download failure"):
            rejection_counts["download failure"] += 1
            log(f"Skipped: {paper_id} | {download_message}")
            continue

        if downloaded_now:
            downloaded_count += 1
        log(download_message)

        seen_arxiv_ids.add(base_paper_id)
        seen_titles.add(normalized_title)
        papers_per_topic[query] += 1
        rows.append(row)

    log(f"Summary for topic '{query}': accepted={papers_per_topic[query]}")
    return downloaded_count, duplicates_removed


def build_corpus() -> None:
    ensure_directories()

    rows: list[dict[str, str]] = []
    seen_arxiv_ids: set[str] = set()
    seen_titles: set[str] = set()
    papers_per_topic: dict[str, int] = defaultdict(int)
    rejection_counts: Counter = Counter()
    papers_downloaded = 0
    duplicates_removed = 0

    client = make_client()

    for query_index, query in enumerate(QUERIES, start=1):
        downloaded_now, duplicates_now = process_topic(
            client=client,
            query=query,
            rows=rows,
            seen_arxiv_ids=seen_arxiv_ids,
            seen_titles=seen_titles,
            papers_per_topic=papers_per_topic,
            rejection_counts=rejection_counts,
        )
        papers_downloaded += downloaded_now
        duplicates_removed += duplicates_now

        if len(rows) >= TARGET_TOTAL_PAPERS:
            log("Target corpus size reached. Stopping topic loop.")
            break

        if query_index < len(QUERIES):
            log(f"Waiting {SLEEP_AFTER_TOPIC_SECONDS}s before next topic...")
            time.sleep(SLEEP_AFTER_TOPIC_SECONDS)

    write_metadata(rows)
    write_dataset_summary(
        rows=rows,
        papers_downloaded=papers_downloaded,
        duplicates_removed=duplicates_removed,
        papers_per_topic=papers_per_topic,
        rejection_counts=rejection_counts,
    )

    log("")
    log("=" * 72)
    log("Summary")
    log("=" * 72)
    log(f"Metadata saved: {METADATA_CSV_PATH}")
    log(f"Dataset summary saved: {DATASET_SUMMARY_PATH}")
    log(f"Papers downloaded: {papers_downloaded}")
    log(f"Duplicates removed: {duplicates_removed}")
    log("Papers per topic:")
    for query in QUERIES:
        log(f"  - {query}: {papers_per_topic.get(query, 0)}")
    log("Rejections:")
    if rejection_counts:
        for reason, count in rejection_counts.most_common():
            log(f"  - {reason}: {count}")
    else:
        log("  - none")
    log(f"Final paper count: {len(rows)}")


if __name__ == "__main__":
    try:
        build_corpus()
    except KeyboardInterrupt:
        log("\nStopped by user.")
    except Exception as exc:
        log(f"Fatal setup error: {exc}")
        log(traceback.format_exc().strip())
