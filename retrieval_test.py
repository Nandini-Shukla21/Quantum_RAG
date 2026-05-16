import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

# ==========================================
# LOAD FAISS INDEX
# ==========================================

print("Loading FAISS index...")

index = faiss.read_index(
    "vector_store/quantum_index.faiss"
)

print("FAISS index loaded")


# ==========================================
# LOAD CHUNKS
# ==========================================

print("Loading chunks...")

with open("vector_store/chunks.pkl", "rb") as f:

    chunks = pickle.load(f)

print("Chunks loaded")


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("Embedding model loaded")


# ==========================================
# USER QUERY
# ==========================================

query = input("\nEnter your scientific query: ")


# ==========================================
# QUERY EMBEDDING
# ==========================================

query_embedding = embedding_model.encode(
    [query]
)


# ==========================================
# SEARCH VECTOR DATABASE
# ==========================================

D, I = index.search(
    np.array(query_embedding),
    k=5
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nTop Retrieved Chunks:\n")

for idx in I[0]:

    print("=" * 100)

    print(chunks[idx].page_content[:1500])

    print("\n")