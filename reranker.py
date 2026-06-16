"""Cross-encoder reranking for retrieved FAISS candidates."""

from dataclasses import dataclass
from typing import List

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME


@dataclass
class RetrievedChunk:
    """A retrieved chunk with all scoring and citation metadata."""

    text: str
    source: str
    page: int | str
    chunk_id: int | None
    faiss_score: float
    semantic_score: float | None = None
    keyword_score: float = 0.0
    title_boost: float = 0.0
    retrieval_score: float | None = None
    title: str = ""
    expanded_query: str = ""
    retrieval_reason: str = ""
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        return f"Source: {self.source} (Page {self.page})"


class CrossEncoderReranker:
    """Rerank FAISS candidates using a query-document cross encoder."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Return the highest scoring chunks after cross-encoder reranking."""

        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        scored_candidates: List[RetrievedChunk] = []
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
            scored_candidates.append(candidate)

        scored_candidates.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else -999,
            reverse=True,
        )
        return scored_candidates[:top_k]
