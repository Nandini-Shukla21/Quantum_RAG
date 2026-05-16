from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import pickle
import os

# ==========================================
# LOAD ALL PDF DOCUMENTS
# ==========================================

documents = []

paper_folder = "papers"

for file in os.listdir(paper_folder):

    if file.endswith(".pdf"):

        path = os.path.join(paper_folder, file)

        try:

            loader = PyPDFLoader(path)

            docs = loader.load()

            documents.extend(docs)

            print(f"Loaded: {file}")

        except Exception as e:

            print(f"Error loading {file}: {e}")

print(f"\nTotal Pages Loaded: {len(documents)}")


# ==========================================
# CHUNK DOCUMENTS
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"\nTotal Chunks Created: {len(chunks)}")


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("Embedding model loaded")


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

texts = [chunk.page_content for chunk in chunks]

print("\nGenerating embeddings...")

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

print("Embeddings generated successfully")


# ==========================================
# CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("\nFAISS index created")


# ==========================================
# SAVE FAISS INDEX
# ==========================================

faiss.write_index(
    index,
    "vector_store/quantum_index.faiss"
)

print("FAISS index saved")


# ==========================================
# SAVE CHUNKS
# ==========================================

with open("vector_store/chunks.pkl", "wb") as f:

    pickle.dump(chunks, f)

print("Chunks saved")


print("\nPROCESS COMPLETED SUCCESSFULLY")