import ssl
import certifi
import arxiv
import pandas as pd
import os
import re

# =========================
# SSL FIX
# =========================
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("papers", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

# =========================
# SEARCH KEYWORDS
# =========================
queries = [
    "quantum machine learning",
    "quantum neural network",
    "variational quantum circuit",
    "quantum reinforcement learning",
    "quantum kernel methods"
]

# =========================
# METADATA STORAGE
# =========================
paper_metadata = []

# To avoid duplicate downloads
downloaded_ids = set()

# =========================
# ARXIV CLIENT
# =========================
client = arxiv.Client()

# =========================
# DOWNLOAD LOOP
# =========================
for query in queries:

    print(f"\n{'='*60}")
    print(f"SEARCHING QUERY: {query}")
    print(f"{'='*60}")

    search = arxiv.Search(
        query=query,
        max_results=20,   # 20 papers per keyword
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for i, result in enumerate(client.results(search)):

        try:

            paper_id = result.get_short_id()

            # Skip duplicates
            if paper_id in downloaded_ids:
                continue

            downloaded_ids.add(paper_id)

            print(f"\nDownloading Paper")
            print("Title:", result.title)

            # =========================
            # CLEAN FILENAME
            # =========================
            safe_title = re.sub(
                r'[\\/*?:"<>|]',
                "",
                result.title
            )

            filename = f"{paper_id}.pdf"

            # =========================
            # DOWNLOAD PDF
            # =========================
            result.download_pdf(
                dirpath="papers",
                filename=filename
            )

            # =========================
            # STORE METADATA
            # =========================
            paper_metadata.append({

                "paper_id": paper_id,

                "title": result.title,

                "authors": ", ".join(
                    [author.name for author in result.authors]
                ),

                "published": result.published,

                "updated": result.updated,

                "summary": result.summary,

                "categories": ", ".join(result.categories),

                "primary_category": result.primary_category,

                "pdf_url": result.pdf_url,

                "entry_id": result.entry_id,

                "comment": result.comment,

                "journal_ref": result.journal_ref,

                "doi": result.doi,

                "search_keyword": query,

                "pdf_file": filename
            })

            print("Downloaded Successfully")

        except Exception as e:

            print("Error:", e)

# =========================
# SAVE METADATA CSV
# =========================
df = pd.DataFrame(paper_metadata)

csv_path = "metadata/paper_metadata.csv"

df.to_csv(
    csv_path,
    index=False
)

print(f"\nMetadata saved at: {csv_path}")

print("\nAll Downloads Completed Successfully")