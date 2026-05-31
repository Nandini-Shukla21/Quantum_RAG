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
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
BATCH_SIZE = 32


# Retrieval and reranking
INITIAL_RETRIEVAL_K = 20
FINAL_RETRIEVAL_K = 10
FAISS_CANDIDATE_POOL_K = 80
SEMANTIC_SCORE_WEIGHT = 0.7
KEYWORD_SCORE_WEIGHT = 0.3
TITLE_BOOST_WEIGHT = 0.12
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

QUERY_EXPANSION_TERMS = {
    "quantum kernel learning": [
        "Quantum Kernel Method",
        "Quantum Feature Map",
        "Quantum Support Vector Machine",
        "Quantum Kernel Estimation",
    ],
    "quantum kernel": [
        "Quantum Kernel Method",
        "Quantum Feature Map",
        "Quantum Support Vector Machine",
        "Quantum Kernel Estimation",
    ],
    "variational quantum circuit": [
        "Parameterized Quantum Circuit",
        "Variational Quantum Algorithm",
        "Quantum Neural Network",
        "Hybrid Quantum Classical Optimization",
    ],
    "quantum neural network": [
        "Parameterized Quantum Circuit",
        "Variational Quantum Circuit",
        "Quantum Circuit Learning",
        "Hybrid Quantum Classical Neural Network",
    ],
}


# Context compression
SIMILARITY_MERGE_THRESHOLD = 0.94
LEXICAL_OVERLAP_THRESHOLD = 0.95
MAX_CONTEXT_CHARS = 15000


# Local Hugging Face generation
LLM_PROVIDER = "huggingface"
HF_MODEL_NAME = "google/flan-t5-base"
HF_MODEL_PRIORITY = [
    "google/flan-t5-base"
]
HF_FALLBACK_MODEL_NAME = "google/flan-t5-base"
HF_DEVICE_MAP = "auto"
HF_TORCH_DTYPE = "auto"
HF_LOW_CPU_MEM_USAGE = True

MAX_NEW_TOKENS = 900
TEMPERATURE = 0.15
TOP_P = 0.9
REPETITION_PENALTY = 1.12


# Prompt used by the draft answer generator.
SYNTHESIS_PROMPT_TEMPLATE = """You are a senior Quantum Machine Learning researcher.

Your task is to answer scientific questions using the retrieved research papers.

Requirements:

1. Synthesize information from multiple papers.
2. Never copy sentences directly.
3. Rewrite information in your own words.
4. Produce detailed technical explanations.
5. Mention:

   * definition
   * working principle
   * advantages
   * limitations
   * applications
6. If multiple papers disagree, summarize the differences.
7. Use scientific language.
8. Generate 3-6 paragraphs.
9. Include citations.
10. If context is insufficient, explicitly say so.

Retrieved Context:
{context}

Question:
{query}

Detailed Research Answer:"""

REFINEMENT_PROMPT_TEMPLATE = """Rewrite the following scientific answer.

Requirements:

* More technical
* More coherent
* Better structured
* Remove repetition
* Improve readability
* Merge duplicate information
* Keep citations
* Maintain factual correctness

Draft Answer:
{draft}

Refined Scientific Answer:

Return only the refined answer."""

EXPANSION_PROMPT_TEMPLATE = """Expand the following scientific answer using only the retrieved context.

Requirements:

* Add theoretical background
* Add methodology
* Add practical applications
* Add challenges
* Add future directions
* Keep citations
* Do not introduce information that is absent from the retrieved context
* Return 3-6 coherent paragraphs

Retrieved Context:
{context}

Question:
{query}

Current Answer:
{answer}

Expanded Scientific Answer:"""
