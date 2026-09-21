# FrenchTutor Application

This project is a Gradio-based interactive web application for learning French, tailored for Spanish speakers. It utilizes a state-of-the-art Natural Language Processing (NLP) architecture to map translations and score the student's input using three distinct metrics.

## Architecture

The system evaluates a given French translation based on the following pipeline:

1.  **Language Identification (LID) Gatekeeper:**
    *   **Model:** `cis-lmu/glotlid` (via FastText).
    *   **Purpose:** Quickly analyzes the input to ensure the language is actually French before passing it to heavier neural models.
2.  **Semantic Adequacy ($S_{\text{sem}}$):**
    *   **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
    *   **Purpose:** Computes cosine similarity between the student's input and the reference translation to evaluate whether the meaning was preserved, regardless of syntax.
3.  **NMT Conditional Perplexity ($S_{\text{NMT}}$) & Word Mapping:**
    *   **Model:** `facebook/nllb-200-distilled-600M` (No Language Left Behind).
    *   **Purpose:** Calculates the cross-entropy perplexity of the generated tokens given the Spanish source. Furthermore, **cross-attention weights** are extracted to dynamically map which Spanish words correspond to which French words, providing interactive visual feedback.
4.  **Monolingual Fluency ($S_{\text{fluency}}$):**
    *   **Model:** `asi/gpt-fr-cased-base` (Autoregressive causal LM).
    *   **Purpose:** Runs a single pass over the student's input to calculate the structural fluency and naturalness of the sentence strictly in French, ignoring the source text.

## Project Structure

*   `backend/`: Python backend application code:
    *   `server.py`: FastAPI application serving the API and React frontend.
    *   `schemas.py`: Pydantic request and response models.
    *   `data_manager.py`: SQLite database operations and repository CRUD.
    *   `models.py`: Natural Language Processing and translation evaluation engine (`MLEngine`).
    *   `model_registry.py`: Modular model wrappers and registry for semantic, NMT, fluency, and CEFR models.
    *   `scoring_engine.py`: Adaptive CEFR score calculation and transcription diff evaluation.
    *   `tts_engine.py`: Neural French Text-to-Speech synthesis and caching.
    *   `vocab_engine.py`: CEFRLex / FLELex + ML vocabulary engine for CEFR level estimation.
    *   `db/`: SQLite connection management and modular domain mixins (`sentences.py`, `users.py`, `history.py`, `audio.py`, `vocabulary.py`).
    *   `ingestion/`: Offline sentence harvesting, mining, translation, and database ingestion pipeline (`orchestrator.py`, `pipeline/`).
*   `frontend/`: React + Vite single page application.
*   `curriculum/`: CEFR levels curriculum definitions, module units, and parsing tools.
*   `data/`: Processed SQLite database (`app.db`), lexicons (`flelex_beacco.tsv`), and audio cache.
*   `Makefile`: Standard developer automation commands (`make dev`, `make start`, `make test`, `make build`, `make install`).
*   `tests/`: Standardized Pytest test suite with isolated fixtures.
*   `run_tests.py`: Fast test runner script with automated categorization.
*   `pyproject.toml`: Modern project specification and dependencies managed by `uv`.
*   `uv.lock`: Deterministic lockfile for dependencies.

## Installation & Execution

### 1. Environment Setup
```bash
# Option A: Lightweight CPU-only PyTorch (~180 MB, fast download & ideal for testing / CPU inference)
uv sync --extra cpu

# Option B: GPU PyTorch with CUDA 12.4 acceleration (~2.5 GB+)
uv sync --extra cu124
```

### 2. Development Mode (Vite HMR + FastAPI Live-Reload)
Start both backend and frontend development servers with a single command:
```bash
make dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser. Any change made to React components updates instantly via Vite HMR without rebuilds!

#### Parallel Git Worktrees
When working across multiple git worktrees on the same machine, customize ports to avoid collisions:
```bash
# Main worktree
make dev

# Feature worktree (custom ports)
make dev BACKEND_PORT=8001 FRONTEND_PORT=5174
```

#### Troubleshooting Busy Ports
If a port (e.g. `8000` or `5173`) is already in use:
```bash
# Find processes listening on a port
lsof -i :8000
# or
ss -tulpn | grep :8000

# Kill process by PID
kill -9 <PID>

# Or kill process directly by port
fuser -k 8000/tcp
# or
kill -9 $(lsof -t -i:8000)
```

### 3. Production Build & Execution
```bash
# Build frontend and serve from FastAPI (port 8000)
make start

# Or custom port:
make start PORT=8080
```

### 4. Running the Test Suite
```bash
# Run fast unit & API tests (<2s)
make test
# or
python run_tests.py

# Run all tests including heavy ML models
python run_tests.py --all
```
3. *Note: The first run with ML models will take several minutes to download the neural models from the Hugging Face Hub (approx. ~3-4 GB).*
