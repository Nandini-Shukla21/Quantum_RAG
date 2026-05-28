"""Configuration for the local Quantum Machine Learning RAG system."""

from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"
METADATA_DIR = BASE_DIR / "metadata"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "quantum_index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.pkl"
PAPER_METADATA_PATH = METADATA_DIR / "paper_metadata.csv"
QUERY_LOG_PATH = LOG_DIR / "query_logs.csv"
EVALUATION_RESULTS_PATH = BASE_DIR / "evaluation_results.csv"
BENCHMARK_RESULTS_PATH = BASE_DIR / "benchmark_results.csv"
BENCHMARK_REPORT_PATH = BASE_DIR / "benchmark_summary_report.txt"


# Embedding and chunking
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 32


# Retrieval and reranking
INITIAL_RETRIEVAL_K = 20
FINAL_RETRIEVAL_K = 5
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# Context compression
SIMILARITY_MERGE_THRESHOLD = 0.86
LEXICAL_OVERLAP_THRESHOLD = 0.82
MAX_CONTEXT_CHARS = 9000


# Local Hugging Face generation
LLM_PROVIDER = "huggingface"
HF_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
HF_FALLBACK_MODEL_NAME = "google/flan-t5-base"
HF_DEVICE_MAP = "auto"
HF_TORCH_DTYPE = "auto"
HF_LOW_CPU_MEM_USAGE = True

MAX_NEW_TOKENS = 450
TEMPERATURE = 0.15
TOP_P = 0.9
REPETITION_PENALTY = 1.12


# Prompt used by the final answer generator.
SYNTHESIS_PROMPT_TEMPLATE = """You are a Quantum Machine Learning research assistant.

Your job is to answer using the retrieved scientific context.

Rules:

* Synthesize information from multiple papers.
* Do not copy sentences directly.
* Rewrite information in your own words.
* Produce concise technical explanations.
* Mention disagreements between papers if present.
* Include citations.
* If the answer is not present in the context, explicitly state that.

Retrieved Context:
{context}

Question:
{query}

Answer:"""

