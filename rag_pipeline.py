"""End-to-end local Research RAG pipeline.

The pipeline uses FAISS retrieval, cross-encoder reranking, context compression,
Hugging Face generation, answer cleanup, source citations, and CSV logging.
Ollama is intentionally not required.
"""

import csv
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

from sentence_transformers import SentenceTransformer, util

from config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    HF_DEVICE_MAP,
    HF_FALLBACK_MODEL_NAME,
    HF_LOW_CPU_MEM_USAGE,
    HF_MODEL_NAME,
    HF_MODEL_PRIORITY,
    HF_TORCH_DTYPE,
    LEXICAL_OVERLAP_THRESHOLD,
    LLM_PROVIDER,
    MAX_CONTEXT_CHARS,
    MAX_NEW_TOKENS,
    QUERY_LOG_PATH,
    REPETITION_PENALTY,
    EXPANSION_PROMPT_TEMPLATE,
    REFINEMENT_PROMPT_TEMPLATE,
    SIMILARITY_MERGE_THRESHOLD,
    SYNTHESIS_PROMPT_TEMPLATE,
    TEMPERATURE,
    TOP_P,
)
from retrieval import Retriever
from reranker import RetrievedChunk


class ModelLoadError(RuntimeError):
    """Raised when no configured Hugging Face model can be loaded."""


class BaseLLM(ABC):
    """Common interface for local generation backends."""

    model_name: str
    load_warnings: List[str]

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class UnavailableLLM(BaseLLM):
    """Non-crashing LLM placeholder used when startup diagnostics fail."""

    def __init__(self, error: str):
        self.model_name = "unavailable"
        self.load_warnings = [error]

    def generate(self, prompt: str) -> str:
        return (
            "The local Hugging Face model is not available, so an answer could "
            "not be generated. Check the startup diagnostics for the loading "
            "error, install the required dependencies, or use the fallback model."
        )


class HuggingFaceLocalLLM(BaseLLM):
    """Local Hugging Face generator with Phi-3 first and FLAN-T5 fallback."""

    def __init__(
        self,
        primary_model: str = HF_MODEL_NAME,
        fallback_model: str = HF_FALLBACK_MODEL_NAME,
        model_priority: List[str] | None = None,
        device_map: str = HF_DEVICE_MAP,
        torch_dtype: str = HF_TORCH_DTYPE,
        low_cpu_mem_usage: bool = HF_LOW_CPU_MEM_USAGE,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.model_priority = model_priority or HF_MODEL_PRIORITY or [primary_model, fallback_model]
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.load_warnings: List[str] = []
        self.model_name = ""
        self.model_kind = ""
        self.tokenizer = None
        self.model = None

        self._load_with_fallback()

    @staticmethod
    def clear_accelerator_cache() -> None:
        """Release GPU memory if the environment has CUDA available."""

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            # Cache cleanup is best-effort and should never prevent startup.
            pass

    @staticmethod
    def _is_memory_error(error: Exception) -> bool:
        text = str(error).lower()
        return (
            isinstance(error, MemoryError)
            or "out of memory" in text
            or "cuda out of memory" in text
            or "not enough memory" in text
            or "unable to allocate" in text
        )

    def _load_with_fallback(self) -> None:
        errors: List[str] = []
        model_names = list(dict.fromkeys(self.model_priority + [self.primary_model, self.fallback_model]))

        for model_name in model_names:
            model_kind = "seq2seq" if "flan-t5" in model_name.lower() else "causal"
            try:
                self._load_model(model_name=model_name, model_kind=model_kind)
                self.model_name = model_name
                self.model_kind = model_kind
                return
            except Exception as exc:
                self.clear_accelerator_cache()
                message = f"{model_name} failed to load: {type(exc).__name__}: {exc}"
                errors.append(message)
                self.load_warnings.append(message)

                if model_name == model_names[0] and self._is_memory_error(exc):
                    self.load_warnings.append(
                        f"Falling back because {model_name} exceeded available memory."
                    )

        raise ModelLoadError("No Hugging Face generation model could be loaded. " + " | ".join(errors))

    def _load_model(self, model_name: str, model_kind: str) -> None:
        """Load tokenizer and model with laptop-friendly memory settings."""

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except Exception as exc:
            raise ModelLoadError(
                "transformers could not be imported. Install transformers, torch, and accelerate."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception as exc:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=False,
                )
            except Exception as slow_exc:
                raise ModelLoadError(
                    f"Tokenizer load error for {model_name}: fast={exc}; slow={slow_exc}"
                ) from slow_exc

        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": True,
        }

        try:
            if model_kind == "causal":
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        except Exception as exc:
            message = str(exc)
            if "'type'" in message or '"type"' in message:
                message = (
                    f"{message}. This commonly indicates that the installed transformers "
                    "version cannot parse this model's custom Phi configuration. Upgrade "
                    "transformers and accelerate, or use the next configured fallback."
                )
            raise ModelLoadError(f"Model load error for {model_name}: {message}") from exc

        self.tokenizer = tokenizer
        self.model = model

    def generate(self, prompt: str) -> str:
        """Generate an answer with robust error handling."""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Hugging Face model is not loaded.")

        try:
            import torch

            if self.model_kind == "causal" and hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass

            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=min(getattr(self.tokenizer, "model_max_length", 4096), 4096),
            )
            input_device = getattr(self.model, "device", None)
            if input_device is not None:
                encoded = {key: value.to(input_device) for key, value in encoded.items()}

            generation_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "repetition_penalty": REPETITION_PENALTY,
                "no_repeat_ngram_size": 5,
                "do_sample": TEMPERATURE > 0,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }

            with torch.inference_mode():
                output_ids = self.model.generate(**encoded, **generation_kwargs)

            if self.model_kind == "causal":
                generated_ids = output_ids[0][encoded["input_ids"].shape[-1] :]
            else:
                generated_ids = output_ids[0]

            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        except Exception as exc:
            self.clear_accelerator_cache()
            raise RuntimeError(f"Generation error with {self.model_name}: {exc}") from exc


def build_llm(provider: str = LLM_PROVIDER, fail_soft: bool = False) -> BaseLLM:
    """Create the local LLM.

    `huggingface` is the default and only required provider. If a legacy config
    still says `ollama`, this function records a warning and uses Hugging Face
    instead, so a missing Ollama server can never crash the app.
    """

    warnings: List[str] = []
    if provider.lower() != "huggingface":
        warnings.append(
            f"LLM_PROVIDER={provider!r} is not used. Falling back to local Hugging Face generation."
        )

    try:
        llm = HuggingFaceLocalLLM()
        llm.load_warnings = warnings + llm.load_warnings
        return llm
    except Exception as exc:
        if fail_soft:
            return UnavailableLLM(str(exc))
        raise


class AnswerPostProcessor:
    """Clean model output without changing the factual content."""

    @staticmethod
    def _citation_text(citation: str) -> str:
        return citation.replace("[Paper:", "Source:").replace("]", "")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

    @staticmethod
    def remove_duplicate_sentences(text: str) -> str:
        seen: set[str] = set()
        cleaned_paragraphs: List[str] = []
        for paragraph in re.split(r"\n\s*\n", text.strip()):
            kept: List[str] = []
            for sentence in AnswerPostProcessor._split_sentences(paragraph):
                normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    kept.append(sentence)
            if kept:
                cleaned_paragraphs.append(" ".join(kept))
        return "\n\n".join(cleaned_paragraphs)

    @staticmethod
    def remove_repeated_citations(text: str) -> str:
        pattern = r"(Source:\s*[^;\n]+\(Page\s*[^\)]+\))(?:\s*\1)+"
        previous = None
        while previous != text:
            previous = text
            text = re.sub(pattern, r"\1", text)
        return text

    @staticmethod
    def reduce_excessive_quoting(text: str) -> str:
        text = re.sub(r'"""(.+?)"""', r"\1", text, flags=re.DOTALL)
        text = re.sub(r'"([^"]{80,})"', r"\1", text)
        return text.replace("“", "").replace("”", "")

    @staticmethod
    def improve_readability(text: str) -> str:
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", text.strip())
            if paragraph.strip()
        ]
        text = "\n\n".join(paragraphs)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(
            r"(?:Answer:|Detailed Research Answer:|Refined Scientific Answer:|Expanded Scientific Answer:)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return text

    @staticmethod
    def strip_prompt_tail(text: str) -> str:
        markers = [
            "Return only the refined answer.",
            "Retrieved Context:",
            "Question:",
            "Draft Answer:",
            "Current Answer:",
        ]
        for marker in markers:
            if marker in text:
                text = text.split(marker, 1)[0].strip()
        return text

    @staticmethod
    def group_citations_by_paragraph(text: str, citations: List[str]) -> str:
        if not citations:
            return text

        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []

        cleaned_citations = [AnswerPostProcessor._citation_text(citation) for citation in citations]
        citation_line = "; ".join(
            citation.replace("Source: ", "", 1) for citation in cleaned_citations[:4]
        )

        cited_paragraphs: List[str] = []
        for paragraph in paragraphs:
            paragraph = re.sub(r"\[Paper:\s*([^,\]]+),\s*Page\s*([^\]]+)\]", r"Source: \1 (Page \2)", paragraph)
            if "Source:" not in paragraph:
                paragraph = f"{paragraph}\nSource: {citation_line}"
            cited_paragraphs.append(paragraph)

        return "\n\n".join(cited_paragraphs)

    def process(self, answer: str, citations: List[str]) -> str:
        answer = self.strip_prompt_tail(answer)
        answer = self.reduce_excessive_quoting(answer)
        answer = self.remove_duplicate_sentences(answer)
        answer = self.remove_repeated_citations(answer)
        answer = self.improve_readability(answer)
        return self.group_citations_by_paragraph(answer, citations)


class ContextCompressor:
    """Lightly deduplicate and trim retrieved chunks before prompting."""

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        lexical_overlap_threshold: float = LEXICAL_OVERLAP_THRESHOLD,
        similarity_merge_threshold: float = SIMILARITY_MERGE_THRESHOLD,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.lexical_overlap_threshold = lexical_overlap_threshold
        self.similarity_merge_threshold = similarity_merge_threshold
        self.max_context_chars = max_context_chars

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _word_overlap(text_a: str, text_b: str) -> float:
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / min(len(words_a), len(words_b))

    def remove_duplicates(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        kept: List[RetrievedChunk] = []
        seen_texts: set[str] = set()

        for chunk in chunks:
            normalized = self._normalize_text(chunk.text)
            if normalized in seen_texts:
                continue
            if any(
                self._word_overlap(normalized, self._normalize_text(existing.text))
                >= self.lexical_overlap_threshold
                for existing in kept
            ):
                continue
            seen_texts.add(normalized)
            kept.append(chunk)

        return kept

    def merge_similar_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if len(chunks) <= 1:
            return chunks

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        embeddings = self.embedding_model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
        )
        assigned: set[int] = set()
        merged: List[RetrievedChunk] = []

        for index, chunk in enumerate(chunks):
            if index in assigned:
                continue

            group = [chunk]
            assigned.add(index)
            similarities = util.cos_sim(embeddings[index], embeddings)[0]

            for other_index, similarity in enumerate(similarities):
                if other_index == index or other_index in assigned:
                    continue
                if float(similarity) >= self.similarity_merge_threshold:
                    group.append(chunks[other_index])
                    assigned.add(other_index)

            if len(group) == 1:
                merged.append(chunk)
            else:
                best = group[0]
                merged_text = "\n".join(
                    f"{item.text.strip()} {item.citation}" for item in group
                )
                merged.append(
                    RetrievedChunk(
                        text=merged_text,
                        source=best.source,
                        page=best.page,
                        chunk_id=best.chunk_id,
                        faiss_score=best.faiss_score,
                        rerank_score=best.rerank_score,
                    )
                )

        return merged

    def fit_to_context_window(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        selected: List[RetrievedChunk] = []
        used_chars = 0

        for chunk in chunks:
            block = self.format_chunk(chunk)
            if used_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining > 600:
                    selected.append(
                        RetrievedChunk(
                            text=chunk.text[:remaining],
                            source=chunk.source,
                            page=chunk.page,
                            chunk_id=chunk.chunk_id,
                            faiss_score=chunk.faiss_score,
                            rerank_score=chunk.rerank_score,
                        )
                    )
                break
            selected.append(chunk)
            used_chars += len(block)

        return selected

    @staticmethod
    def format_chunk(chunk: RetrievedChunk) -> str:
        return (
            f"{chunk.citation}\n"
            f"Title: {chunk.title}\n"
            f"FAISS score: {chunk.faiss_score:.4f}; "
            f"Semantic score: {(chunk.semantic_score or 0.0):.4f}; "
            f"Keyword score: {chunk.keyword_score:.4f}; "
            f"Title boost: {chunk.title_boost:.4f}; "
            f"Retrieval score: {(chunk.retrieval_score or 0.0):.4f}; "
            f"Reranker score: {chunk.rerank_score:.4f}\n"
            f"Why retrieved: {chunk.retrieval_reason}\n"
            f"{chunk.text.strip()}"
        )

    def compress(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        return self.fit_to_context_window(self.remove_duplicates(chunks))

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        return "\n\n---\n\n".join(self.format_chunk(chunk) for chunk in chunks)


def run_startup_diagnostics(load_model: bool = False) -> Dict:
    """Check local files and optionally verify the Hugging Face model."""

    diagnostics = {
        "ok": True,
        "warnings": [],
        "faiss_exists": FAISS_INDEX_PATH.exists(),
        "chunks_exists": CHUNKS_PATH.exists(),
        "model_name": None,
    }

    if not diagnostics["faiss_exists"]:
        diagnostics["ok"] = False
        diagnostics["warnings"].append(f"Missing FAISS index: {FAISS_INDEX_PATH}")
    if not diagnostics["chunks_exists"]:
        diagnostics["ok"] = False
        diagnostics["warnings"].append(f"Missing chunks file: {CHUNKS_PATH}")

    if load_model:
        llm = build_llm(fail_soft=True)
        diagnostics["model_name"] = llm.model_name
        diagnostics["warnings"].extend(llm.load_warnings)
        if isinstance(llm, UnavailableLLM):
            diagnostics["ok"] = False

    return diagnostics


class ResearchRAGPipeline:
    """Production-oriented local RAG pipeline for QML research papers."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        compressor: ContextCompressor | None = None,
        llm: BaseLLM | None = None,
        fail_soft_llm: bool = False,
    ):
        self.retriever = retriever or Retriever()
        self.compressor = compressor or ContextCompressor()
        self.llm = llm or build_llm(fail_soft=fail_soft_llm)
        self.post_processor = AnswerPostProcessor()

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\b\w+\b", text))

    def _generate_draft(self, query: str, context: str) -> str:
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(context=context, query=query)
        return self.llm.generate(prompt)

    def _refine_answer(self, draft: str) -> str:
        prompt = REFINEMENT_PROMPT_TEMPLATE.format(draft=draft)
        return self.llm.generate(prompt)

    def _expand_answer(self, query: str, context: str, answer: str) -> str:
        prompt = EXPANSION_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
            answer=answer,
        )
        return self.llm.generate(prompt)

    def answer(self, query: str) -> Dict:
        start_time = time.perf_counter()
        faiss_candidates, retrieved_chunks, reranked_candidates = self.retriever.retrieve_with_candidates(query)
        compressed_chunks = self.compressor.compress(retrieved_chunks)
        context = self.compressor.build_context(compressed_chunks)
        context_length = len(context)

        citations = []
        for chunk in compressed_chunks:
            if chunk.citation not in citations:
                citations.append(chunk.citation)

        draft_answer = ""
        refined_answer = ""
        expanded_answer = ""
        try:
            draft_answer = self.post_processor.process(self._generate_draft(query, context), citations)
            refined_answer = self.post_processor.process(self._refine_answer(draft_answer), citations)
            answer = refined_answer or draft_answer
            if self._word_count(answer) < 250:
                expanded_answer = self.post_processor.process(
                    self._expand_answer(query, context, answer),
                    citations,
                )
                if self._word_count(expanded_answer) > self._word_count(answer):
                    answer = expanded_answer
            generation_error = ""
        except Exception as exc:
            answer = f"Generation failed: {exc}"
            generation_error = str(exc)

        response_time = time.perf_counter() - start_time
        result = {
            "query": query,
            "answer": answer,
            "draft_answer": draft_answer,
            "refined_answer": refined_answer,
            "expanded_answer": expanded_answer,
            "generation_error": generation_error,
            "llm_model": self.llm.model_name,
            "faiss_candidates": [asdict(chunk) | {"citation": chunk.citation} for chunk in faiss_candidates],
            "reranked_candidates": [asdict(chunk) | {"citation": chunk.citation} for chunk in reranked_candidates],
            "retrieved_chunks": [asdict(chunk) | {"citation": chunk.citation} for chunk in retrieved_chunks],
            "compressed_chunks": [asdict(chunk) | {"citation": chunk.citation} for chunk in compressed_chunks],
            "citations": citations,
            "context_length_chars": context_length,
            "response_time_seconds": response_time,
        }
        self.log_query(result)
        return result

    def log_query(self, result: Dict) -> None:
        QUERY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": result["query"],
            "llm_model": result["llm_model"],
            "retrieved_papers": "; ".join(
                chunk["source"] for chunk in result["retrieved_chunks"]
            ),
            "retrieval_scores": "; ".join(
                f"{chunk['source']} p.{chunk['page']} title={chunk.get('title', '')} "
                f"faiss={chunk['faiss_score']:.4f} "
                f"semantic={chunk.get('semantic_score') or 0.0:.4f} "
                f"keyword={chunk.get('keyword_score') or 0.0:.4f} "
                f"title_boost={chunk.get('title_boost') or 0.0:.4f} "
                f"retrieval={chunk.get('retrieval_score') or 0.0:.4f} "
                f"reranker={chunk['rerank_score']:.4f}"
                for chunk in result["retrieved_chunks"]
            ),
            "response_time_seconds": f"{result['response_time_seconds']:.4f}",
            "context_length_chars": result["context_length_chars"],
            "generation_error": result["generation_error"],
            "draft_answer": result["draft_answer"],
            "refined_answer": result["refined_answer"],
            "expanded_answer": result["expanded_answer"],
            "final_answer": result["answer"],
        }

        write_header = not QUERY_LOG_PATH.exists()
        with open(QUERY_LOG_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    diagnostics = run_startup_diagnostics(load_model=False)
    for warning in diagnostics["warnings"]:
        print(f"Warning: {warning}")

    pipeline = ResearchRAGPipeline(fail_soft_llm=True)
    question = input("Question: ")
    output = pipeline.answer(question)
    print(output["answer"])
    print("\nSources:")
    for citation in output["citations"]:
        print(citation)
