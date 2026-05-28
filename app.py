"""Streamlit UI for the local Quantum Machine Learning research RAG system."""

import streamlit as st

from config import HF_FALLBACK_MODEL_NAME, HF_MODEL_NAME
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

    st.caption(f"Primary model: {HF_MODEL_NAME}")
    st.caption(f"Fallback model: {HF_FALLBACK_MODEL_NAME}")

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

    metric_columns = st.columns(3)
    metric_columns[0].metric("Response time", f"{result['response_time_seconds']:.2f}s")
    metric_columns[1].metric("Retrieved chunks", len(result["retrieved_chunks"]))
    metric_columns[2].metric("Compressed chunks", len(result["compressed_chunks"]))

    st.subheader("Retrieval Diagnostics")
    diagnostics_rows = [
        {
            "rank": rank,
            "paper": chunk["source"],
            "page": chunk["page"],
            "faiss_score": round(chunk["faiss_score"], 4),
            "reranker_score": round(chunk["rerank_score"], 4),
        }
        for rank, chunk in enumerate(result["retrieved_chunks"], start=1)
    ]
    st.dataframe(diagnostics_rows, use_container_width=True)

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
            f"FAISS {chunk['faiss_score']:.4f} | "
            f"Reranker {chunk['rerank_score']:.4f}"
        )
        with st.expander(label):
            st.write(chunk["text"])

    st.subheader("Compressed Chunks Used for Generation")
    for rank, chunk in enumerate(result["compressed_chunks"], start=1):
        label = (
            f"Context {rank}: {chunk['source']} p.{chunk['page']} | "
            f"FAISS {chunk['faiss_score']:.4f} | "
            f"Reranker {chunk['rerank_score']:.4f}"
        )
        with st.expander(label):
            st.write(chunk["text"])

