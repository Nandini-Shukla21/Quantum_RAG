"""Hybrid FAISS, keyword, title-boosted retrieval plus cross-encoder reranking."""

import csv
import pickle
import re
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_CANDIDATE_POOL_K,
    FAISS_INDEX_PATH,
    FINAL_RETRIEVAL_K,
    INITIAL_RETRIEVAL_K,
    KEYWORD_SCORE_WEIGHT,
    PAPER_METADATA_PATH,
    QUERY_EXPANSION_TERMS,
    SEMANTIC_SCORE_WEIGHT,
    TITLE_BOOST_WEIGHT,
)
from reranker import CrossEncoderReranker, RetrievedChunk


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
    "with",
}


class Retriever:
    """Load the vector store and retrieve reranked scientific chunks."""

    def __init__(
        self,
        index_path=FAISS_INDEX_PATH,
        chunks_path=CHUNKS_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as file:
            self.chunks = pickle.load(file)

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.paper_titles = self._load_paper_titles()
        self.reranker = reranker or CrossEncoderReranker()

    @staticmethod
    def _load_paper_titles() -> dict[str, str]:
        if not PAPER_METADATA_PATH.exists():
            return {}
        with open(PAPER_METADATA_PATH, newline="", encoding="utf-8") as file:
            return {
                row["pdf_file"].strip(): row["title"].strip()
                for row in csv.DictReader(file)
                if row.get("pdf_file") and row.get("title")
            }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
            if len(token) > 2 and token not in STOPWORDS
        ]

    @classmethod
    def _keywords(cls, text: str) -> set[str]:
        return set(cls._tokenize(text))

    @staticmethod
    def _semantic_similarity(score: float) -> float:
        return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """Add domain-specific aliases before retrieval."""

        normalized_query = " ".join(self._tokenize(query))
        expansions: List[str] = []

        for trigger, terms in QUERY_EXPANSION_TERMS.items():
            trigger_tokens = set(self._tokenize(trigger))
            if trigger.lower() in query.lower() or trigger_tokens.issubset(set(normalized_query.split())):
                expansions.extend(terms)

        deduped_expansions = list(dict.fromkeys(expansions))
        if not deduped_expansions:
            return query, []
        return f"{query}\nRelated concepts: {', '.join(deduped_expansions)}", deduped_expansions

    def _encode_query(self, query: str) -> np.ndarray:
        query_text = query
        if "bge-" in self.embedding_model_name.lower():
            query_text = f"Represent this sentence for searching relevant passages: {query}"

        normalize_query = self.index.metric_type == faiss.METRIC_INNER_PRODUCT
        embedding = self.embedding_model.encode(
            [query_text],
            normalize_embeddings=normalize_query,
        ).astype("float32")

        if embedding.shape[1] != self.index.d:
            raise ValueError(
                "Embedding dimension mismatch. The FAISS index was built with "
                f"dimension {self.index.d}, but {self.embedding_model_name} produced "
                f"dimension {embedding.shape[1]}. Rebuild the vector store with "
                "`python create_embeddings.py` after changing EMBEDDING_MODEL_NAME."
            )
        return embedding

    def _keyword_overlap(self, query_terms: set[str], text: str, title: str) -> float:
        if not query_terms:
            return 0.0
        target_terms = self._keywords(f"{title} {text}")
        if not target_terms:
            return 0.0
        return len(query_terms & target_terms) / len(query_terms)

    def _title_boost(self, query_terms: set[str], title: str) -> float:
        if not query_terms or not title:
            return 0.0
        title_terms = self._keywords(title)
        if not title_terms:
            return 0.0
        return TITLE_BOOST_WEIGHT * (len(query_terms & title_terms) / len(query_terms))

    @staticmethod
    def _reason(
        semantic_score: float,
        keyword_score: float,
        title_boost: float,
        matched_terms: set[str],
        expansions: List[str],
    ) -> str:
        parts = [
            f"semantic={semantic_score:.3f}",
            f"keyword_overlap={keyword_score:.3f}",
        ]
        if title_boost > 0:
            parts.append(f"title_boost={title_boost:.3f}")
        if matched_terms:
            parts.append(f"matched_terms={', '.join(sorted(matched_terms)[:8])}")
        if expansions:
            parts.append(f"query_expansion={', '.join(expansions[:4])}")
        return "; ".join(parts)

    def retrieve_candidates(self, query: str, top_k: int = INITIAL_RETRIEVAL_K) -> List[RetrievedChunk]:
        """Retrieve top-k hybrid candidates before cross-encoder reranking."""

        expanded_query, expansions = self.expand_query(query)
        query_terms = self._keywords(f"{query} {' '.join(expansions)}")
        query_embedding = self._encode_query(expanded_query)

        pool_k = min(max(FAISS_CANDIDATE_POOL_K, top_k), self.index.ntotal)
        scores, indices = self.index.search(np.array(query_embedding), k=pool_k)
        candidates: List[RetrievedChunk] = []

        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue

            chunk = self.chunks[int(index_id)]
            metadata = getattr(chunk, "metadata", {}) or {}
            source = metadata.get("paper") or metadata.get("source") or "unknown"
            source_name = str(source).split("\\")[-1].split("/")[-1]
            page = metadata.get("page", "unknown")
            chunk_id = metadata.get("chunk_id", int(index_id))
            title = metadata.get("title") or self.paper_titles.get(source_name, "")
            semantic_score = self._semantic_similarity(float(score))
            keyword_score = self._keyword_overlap(query_terms, chunk.page_content, title)
            title_boost = self._title_boost(query_terms, title)
            retrieval_score = (
                SEMANTIC_SCORE_WEIGHT * semantic_score
                + KEYWORD_SCORE_WEIGHT * keyword_score
                + title_boost
            )
            matched_terms = query_terms & self._keywords(f"{title} {chunk.page_content}")

            candidates.append(
                RetrievedChunk(
                    text=chunk.page_content,
                    source=source_name,
                    page=page,
                    chunk_id=chunk_id,
                    faiss_score=float(score),
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    title_boost=title_boost,
                    retrieval_score=retrieval_score,
                    title=title,
                    expanded_query=expanded_query,
                    retrieval_reason=self._reason(
                        semantic_score,
                        keyword_score,
                        title_boost,
                        matched_terms,
                        expansions,
                    ),
                )
            )

        candidates.sort(
            key=lambda candidate: candidate.retrieval_score
            if candidate.retrieval_score is not None
            else candidate.faiss_score,
            reverse=True,
        )
        return candidates[:top_k]

    def retrieve(
        self,
        query: str,
        initial_k: int = INITIAL_RETRIEVAL_K,
        final_k: int = FINAL_RETRIEVAL_K,
    ) -> List[RetrievedChunk]:
        """Retrieve top 20 hybrid candidates, rerank them, and keep the best 10."""

        _, final_chunks, _ = self.retrieve_with_candidates(query, initial_k, final_k)
        return final_chunks

    def retrieve_with_candidates(
        self,
        query: str,
        initial_k: int = INITIAL_RETRIEVAL_K,
        final_k: int = FINAL_RETRIEVAL_K,
    ) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], List[RetrievedChunk]]:
        """Return FAISS candidates, final context chunks, and all reranked candidates."""

        candidates = self.retrieve_candidates(query, top_k=initial_k)
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=len(candidates),
        )
        return candidates, reranked_candidates[:final_k], reranked_candidates


if __name__ == "__main__":
    retriever = Retriever()
    user_query = input("Enter your scientific query: ")
    results = retriever.retrieve(user_query)

    for rank, result in enumerate(results, start=1):
        print("=" * 100)
        print(f"Rank: {rank}")
        print(f"FAISS score: {result.faiss_score:.4f}")
        print(f"Rerank score: {result.rerank_score:.4f}")
        print(result.citation)
        print(result.text[:1500])
