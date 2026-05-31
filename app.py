"""Streamlit UI for the local Quantum Machine Learning research RAG system."""

import streamlit as st

from config import HF_MODEL_PRIORITY
from rag_pipeline import ResearchRAGPipeline, run_startup_diagnostics


st.set_page_config(
    page_title="Quantum ML Research RAG",
    page_icon="QML",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading FAISS, reranker, compressor, and local Hugging Face model...")
def load_pipeline() -> ResearchRAGPipeline:
    return ResearchRAGPipeline(fail_soft_llm=True)


@st.cache_data(show_spinner=False)
def get_file_diagnostics() -> dict:
    return run_startup_diagnostics(load_model=False)


st.title("Quantum Machine Learning Research RAG")

with st.sidebar:
    st.header("Startup Diagnostics")
    diagnostics = get_file_diagnostics()

    if diagnostics["faiss_exists"]:
        st.success("FAISS index found")
    else:
        st.warning("FAISS index missing")

    if diagnostics["chunks_exists"]:
        st.success("Chunk store found")
    else:
        st.warning("Chunk store missing")

    st.caption("Model priority:")
    for model_name in HF_MODEL_PRIORITY:
        st.caption(f"- {model_name}")

    for warning in diagnostics["warnings"]:
        st.warning(warning)


pipeline = None
try:
    pipeline = load_pipeline()
    if pipeline.llm.model_name == "unavailable":
        st.warning("Hugging Face generation model could not be loaded. Retrieval diagnostics still work.")
    else:
        st.success(f"Loaded local generation model: {pipeline.llm.model_name}")

    for warning in pipeline.llm.load_warnings:
        st.warning(warning)
except Exception as exc:
    st.error(f"Pipeline startup failed: {exc}")


question = st.text_area(
    "Question",
    placeholder="Ask a technical question about the indexed QML papers...",
    height=110,
)

run_query = st.button(
    "Generate answer",
    type="primary",
    disabled=not question.strip() or pipeline is None,
)

if run_query and pipeline is not None:
    with st.spinner("Retrieving, reranking, compressing context, and generating answer..."):
        result = pipeline.answer(question.strip())

    if result["generation_error"]:
        st.warning(result["generation_error"])

    st.subheader("Generated Answer")
    st.write(result["answer"])

    metric_columns = st.columns(4)
    metric_columns[0].metric("Response time", f"{result['response_time_seconds']:.2f}s")
    metric_columns[1].metric("Retrieved chunks", len(result["retrieved_chunks"]))
    metric_columns[2].metric("Compressed chunks", len(result["compressed_chunks"]))
    metric_columns[3].metric("Context length", f"{result['context_length_chars']:,} chars")

    st.subheader("Draft Answer")
    st.write(result["draft_answer"] or "No draft answer generated.")

    st.subheader("Refined Answer")
    st.write(result["refined_answer"] or "No refined answer generated.")

    if result.get("expanded_answer"):
        st.subheader("Expanded Answer")
        st.write(result["expanded_answer"])

    st.subheader("Retrieval Diagnostics")
    diagnostics_rows = [
        {
            "rank": rank,
            "paper": chunk["source"],
            "title": chunk.get("title", ""),
            "page": chunk["page"],
            "faiss_score": round(chunk["faiss_score"], 4),
            "semantic_score": round(chunk.get("semantic_score") or 0.0, 4),
            "keyword_score": round(chunk.get("keyword_score") or 0.0, 4),
            "title_boost": round(chunk.get("title_boost") or 0.0, 4),
            "retrieval_score": round(chunk.get("retrieval_score") or 0.0, 4),
            "reranker_score": round(chunk["rerank_score"], 4),
            "why_retrieved": chunk.get("retrieval_reason", ""),
        }
        for rank, chunk in enumerate(result["retrieved_chunks"], start=1)
    ]
    st.dataframe(diagnostics_rows, use_container_width=True)

    with st.expander("Top 20 candidates before reranking"):
        faiss_rows = [
            {
                "rank": rank,
                "paper": chunk["source"],
                "title": chunk.get("title", ""),
                "page": chunk["page"],
                "faiss_score": round(chunk["faiss_score"], 4),
                "semantic_score": round(chunk.get("semantic_score") or 0.0, 4),
                "keyword_score": round(chunk.get("keyword_score") or 0.0, 4),
                "title_boost": round(chunk.get("title_boost") or 0.0, 4),
                "retrieval_score": round(chunk.get("retrieval_score") or 0.0, 4),
                "why_retrieved": chunk.get("retrieval_reason", ""),
            }
            for rank, chunk in enumerate(result.get("faiss_candidates", []), start=1)
        ]
        st.dataframe(faiss_rows, use_container_width=True)

    with st.expander("Reranked top 20 candidates"):
        reranked_rows = [
            {
                "rank": rank,
                "paper": chunk["source"],
                "title": chunk.get("title", ""),
                "page": chunk["page"],
                "faiss_score": round(chunk["faiss_score"], 4),
                "semantic_score": round(chunk.get("semantic_score") or 0.0, 4),
                "keyword_score": round(chunk.get("keyword_score") or 0.0, 4),
                "title_boost": round(chunk.get("title_boost") or 0.0, 4),
                "retrieval_score": round(chunk.get("retrieval_score") or 0.0, 4),
                "reranker_score": round(chunk["rerank_score"], 4),
                "why_retrieved": chunk.get("retrieval_reason", ""),
            }
            for rank, chunk in enumerate(result.get("reranked_candidates", []), start=1)
        ]
        st.dataframe(reranked_rows, use_container_width=True)

    st.subheader("Retrieved Papers")
    papers = []
    seen_papers = set()
    for chunk in result["retrieved_chunks"]:
        paper_key = (chunk["source"], chunk["page"])
        if paper_key in seen_papers:
            continue
        seen_papers.add(paper_key)
        papers.append(
            {
                "paper": chunk["source"],
                "title": chunk.get("title", ""),
                "page": chunk["page"],
                "faiss_score": round(chunk["faiss_score"], 4),
                "retrieval_score": round(chunk.get("retrieval_score") or 0.0, 4),
                "reranker_score": round(chunk["rerank_score"], 4),
                "why_retrieved": chunk.get("retrieval_reason", ""),
            }
        )
    st.dataframe(papers, use_container_width=True)

    st.subheader("Source Citations")
    if result["citations"]:
        for citation in result["citations"]:
            st.markdown(f"- {citation}")
    else:
        st.info("No citations returned.")

    st.subheader("Retrieved Chunks")
    for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
        label = (
            f"Rank {rank}: {chunk['source']} p.{chunk['page']} | "
            f"{chunk.get('title', '')} | "
            f"Retrieval {chunk.get('retrieval_score') or 0.0:.4f} | "
            f"Reranker {chunk['rerank_score']:.4f}"
        )
        with st.expander(label):
            st.caption(chunk.get("retrieval_reason", "No retrieval reason recorded."))
            st.write(chunk["text"])

    st.subheader("Compressed Chunks Used for Generation")
    for rank, chunk in enumerate(result["compressed_chunks"], start=1):
        label = (
            f"Context {rank}: {chunk['source']} p.{chunk['page']} | "
            f"{chunk.get('title', '')} | "
            f"Retrieval {chunk.get('retrieval_score') or 0.0:.4f} | "
            f"Reranker {chunk['rerank_score']:.4f}"
        )
        with st.expander(label):
            st.caption(chunk.get("retrieval_reason", "No retrieval reason recorded."))
            st.write(chunk["text"])
