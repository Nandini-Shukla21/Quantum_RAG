"""Automatic evaluation for the local Research RAG system."""

import csv
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Set

from sentence_transformers import SentenceTransformer, util

from config import EMBEDDING_MODEL_NAME, EVALUATION_RESULTS_PATH
from rag_pipeline import ResearchRAGPipeline


class RAGEvaluator:
    """Compute retrieval and generation metrics and save them to CSV."""

    def __init__(
        self,
        pipeline: ResearchRAGPipeline | None = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.pipeline = pipeline or ResearchRAGPipeline(fail_soft_llm=True)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    @staticmethod
    def _normalize_source(value: str) -> str:
        return Path(str(value).strip()).name.lower()

    @staticmethod
    def parse_relevant_sources(value: str | Sequence[str]) -> Set[str]:
        if isinstance(value, str):
            parts = re.split(r"[;,\n|]+", value)
            return {Path(part.strip()).name.lower() for part in parts if part.strip()}
        return {Path(str(item).strip()).name.lower() for item in value if str(item).strip()}

    def recall_at_k(self, retrieved_chunks: List[Dict], relevant_sources: Set[str], k: int) -> float:
        """Fraction of gold papers present in the first k retrieved chunks."""

        if not relevant_sources:
            return 0.0
        retrieved_sources = {
            self._normalize_source(chunk["source"])
            for chunk in retrieved_chunks[:k]
        }
        return len(retrieved_sources & relevant_sources) / len(relevant_sources)

    def mrr(self, retrieved_chunks: List[Dict], relevant_sources: Set[str]) -> float:
        """Reciprocal rank of the first relevant retrieved paper."""

        if not relevant_sources:
            return 0.0
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if self._normalize_source(chunk["source"]) in relevant_sources:
                return 1.0 / rank
        return 0.0

    def context_relevance(self, query: str, contexts: Iterable[str]) -> float:
        joined_context = " ".join(contexts)
        if not joined_context.strip():
            return 0.0
        embeddings = self.embedding_model.encode(
            [query, joined_context],
            normalize_embeddings=True,
        )
        return float(util.cos_sim(embeddings[0], embeddings[1]))

    def answer_relevance(self, query: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        embeddings = self.embedding_model.encode(
            [query, answer],
            normalize_embeddings=True,
        )
        return float(util.cos_sim(embeddings[0], embeddings[1]))

    @staticmethod
    def extractiveness_score(answer: str, contexts: Iterable[str], ngram_size: int = 8) -> float:
        """Share of answer n-grams copied from retrieved context."""

        def ngrams(text: str) -> Set[str]:
            tokens = re.findall(r"\w+", text.lower())
            return {
                " ".join(tokens[index : index + ngram_size])
                for index in range(max(0, len(tokens) - ngram_size + 1))
            }

        answer_ngrams = ngrams(answer)
        if not answer_ngrams:
            return 0.0
        context_ngrams = ngrams(" ".join(contexts))
        return len(answer_ngrams & context_ngrams) / len(answer_ngrams)

    @staticmethod
    def faithfulness(answer: str, contexts: Iterable[str]) -> float:
        """Lightweight support score for answer sentences against context."""

        context_words = set(re.findall(r"\w+", " ".join(contexts).lower()))
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer)
            if sentence.strip()
        ]
        if not sentences:
            return 0.0

        supported = 0
        scored = 0
        for sentence in sentences:
            words = {
                word
                for word in re.findall(r"\w+", sentence.lower())
                if len(word) > 3
            }
            if not words:
                continue
            scored += 1
            if len(words & context_words) / len(words) >= 0.5:
                supported += 1

        return supported / scored if scored else 0.0

    def evaluate_case(self, query: str, relevant_sources: Set[str]) -> Dict:
        result = self.pipeline.answer(query)
        retrieved_chunks = result["retrieved_chunks"]
        context_texts = [chunk["text"] for chunk in result["compressed_chunks"]]
        answer = result["answer"]

        return {
            "query": query,
            "relevant_sources": "; ".join(sorted(relevant_sources)),
            "recall_at_5": self.recall_at_k(retrieved_chunks, relevant_sources, 5),
            "recall_at_10": self.recall_at_k(retrieved_chunks, relevant_sources, 10),
            "mrr": self.mrr(retrieved_chunks, relevant_sources),
            "faithfulness": self.faithfulness(answer, context_texts),
            "context_relevance": self.context_relevance(query, context_texts),
            "answer_relevance": self.answer_relevance(query, answer),
            "extractiveness_score": self.extractiveness_score(answer, context_texts),
            "response_time_seconds": result["response_time_seconds"],
            "llm_model": result["llm_model"],
            "generation_error": result["generation_error"],
            "answer": answer,
            "retrieved_chunks": "; ".join(
                f"{chunk['source']} p.{chunk['page']} faiss={chunk['faiss_score']:.4f} reranker={chunk['rerank_score']:.4f}"
                for chunk in retrieved_chunks
            ),
        }

    def evaluate_rows(
        self,
        rows: List[Dict],
        output_path: str | Path = EVALUATION_RESULTS_PATH,
    ) -> List[Dict]:
        results: List[Dict] = []
        for row in rows:
            query = row["query"]
            relevant_sources = self.parse_relevant_sources(row.get("relevant_sources", ""))
            results.append(self.evaluate_case(query, relevant_sources))
        self.save_results(results, output_path)
        return results

    def evaluate_csv(
        self,
        dataset_path: str | Path,
        output_path: str | Path = EVALUATION_RESULTS_PATH,
    ) -> List[Dict]:
        with open(dataset_path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        return self.evaluate_rows(rows, output_path)

    @staticmethod
    def save_results(rows: List[Dict], output_path: str | Path = EVALUATION_RESULTS_PATH) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def summarize(rows: List[Dict]) -> Dict:
        metric_names = [
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "faithfulness",
            "context_relevance",
            "answer_relevance",
            "extractiveness_score",
            "response_time_seconds",
        ]
        summary = {"num_queries": len(rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if row.get(metric) != ""]
            summary[f"mean_{metric}"] = mean(values) if values else 0.0
        return summary


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    dataset = input("Path to evaluation CSV with query,relevant_sources columns: ")
    results = evaluator.evaluate_csv(dataset)
    print(f"Saved {len(results)} rows to {EVALUATION_RESULTS_PATH}")
    print(RAGEvaluator.summarize(results))

