"""Central configuration for the RAG pipeline.

Everything tunable lives here so the pipeline stages stay readable and the
planning.md / README.md numbers have a single source of truth.
"""
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a local .env (e.g. GROQ_API_KEY) so every
# pipeline stage that imports config picks the key up automatically.
# override=True so the .env value wins over any stale/placeholder value already
# exported in the shell (e.g. a leftover GROQ_API_KEY=your_key_here).
load_dotenv(override=True)

# --- Paths ---
ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT / "documents"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "usf_cs_guide"

# --- Chunking ---
# Documents are short, conversational sources (Reddit comments, RMP reviews,
# study-guide sections). 800 chars (~150-200 words) keeps a single comment or
# review whole while staying small enough that retrieval stays focused.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Embedding ---
# all-MiniLM-L6-v2: 384-dim, fast, strong on short-text semantic similarity.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 5
# Chroma returns cosine *distance* (0 = identical). Drop anything looser than
# this so off-topic chunks never reach the model.
MAX_DISTANCE = 1.15

# --- Generation ---
# Primary path is Groq (per the project spec). If GROQ_API_KEY is missing we
# fall back to a local Ollama model so the pipeline is runnable offline.
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "llama3.2"
TEMPERATURE = 0.1
MAX_TOKENS = 700
