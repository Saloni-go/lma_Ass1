"""Central configuration: paths and parameters shared by every stage.

Values here are the single source of truth so that the scripts and the final
notebook produce identical results. Secrets (GEMINI_API_KEY) are read from a
local .env file, never hard-coded.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- data -----------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Knowledge source: IIT Madras B.Tech / B.Tech (Honours) Ordinances and
# Regulations (2010-2014 batches), 25 pages, born-digital (text layer present).
SOURCE_PDF = DATA_DIR / "document.pdf"

PAGES_JSON = PROCESSED_DIR / "pages.json"
CHUNKS_JSON = PROCESSED_DIR / "chunks.json"

# ---- extraction -------------------------------------------------------------
# Render detected tables as Markdown rows instead of one-cell-per-line text.
EXTRACT_TABLES = True
# A page with fewer characters than this (after cleaning) is flagged as suspicious.
MIN_PAGE_CHARS = 200

# ---- chunking (RecursiveCharacterTextSplitter) ------------------------------
CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 200    # characters
# Tried in order: paragraph, line, sentence, word, character.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ---- retrieval ----------------------------------------------------------------
BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 10    # candidates from the vector store (shared by both pipelines)
TOP_K_CONTEXT = 3      # chunks passed to the LLM
INDEX_DIR = DATA_DIR / "index"   # cached embeddings + FAISS index

# ---- generation -----------------------------------------------------------------
# Override with GEMINI_MODEL in .env; the key is read from GEMINI_API_KEY.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

BASELINE_RESULTS_JSON = RESULTS_DIR / "baseline_results.json"
COMPARISON_RESULTS_JSON = RESULTS_DIR / "comparison_results.json"

SEED = 42
