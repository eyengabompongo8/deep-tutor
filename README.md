# 🇫🇷 DeepTutor

<p align="center">
  <strong>Intelligent, Adaptive French Acquisition Platform Powered by Multi-Model Neural NLP, Psycholinguistic Lexical Modeling, and Comprehensible Input Principles.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React 18" />
  <img src="https://img.shields.io/badge/Vite-6-646CFF?style=flat-square&logo=vite&logoColor=white" alt="Vite" />
  <img src="https://img.shields.io/badge/TailwindCSS-3.4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/PyTorch-2.6%20(CPU%20%2F%20CUDA%2012.4)-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/SQLite-WAL%20Mode-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/CEFR-A1%20to%20C2-4CAF50?style=flat-square" alt="CEFR A1-C2" />
  <img src="https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen?style=flat-square&logo=pytest&logoColor=white" alt="Pytest" />
</p>

---

## 🎬 Application Demo

Watch DeepTutor in action—from real-time Spanish-to-French translation exercises and neural audio playback to pure NMT cross-attention token alignment, in-sentence error classification, and dynamic CEFR progression:

<p align="center">
  <a href="docs/assets/demo_frenchtutor.mp4" title="Click to open and watch full HD demo video with audio">
    <img src="docs/assets/demo_preview.gif" alt="DeepTutor Walkthrough Demo Preview" width="100%" style="max-width: 860px; border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);" />
  </a>
  <br />
  <em>Click the preview above or <a href="docs/assets/demo_frenchtutor.mp4">▶️ <strong>click here to watch the full demo video with sound</strong></a>.</em>
</p>

### What You'll See in the Demo:
- **Interactive Translation Practice**: Instant evaluation of Spanish-to-French submissions with real-time score feedback.
- **Cross-Attention Alignment & Token Hints**: Hover over Spanish words to inspect bilingual alignments derived directly from NLLB-200 encoder-decoder attention.
- **Transition Matrix Error Classification**: Visual distinction between **Type A (Mapped)** substitutions (orange) and **Type B (Unmapped)** extraneous words (rose) with in-sentence highlights.
- **Audio Transcription Mode**: Native French speech synthesis powered by Edge-TTS with phonetic diffs and accent-sensitive grading.
- **Adaptive Level Progression**: Live CEFR score deltas based on Stephen Krashen's $i+1$ comprehensible input constraints.
- **Seamless Theming**: Full dark and light mode UI designed with Tailwind CSS.

---

## 📑 Table of Contents

- [Application Demo](#-application-demo)
- [Technical Documentation Hub](#-technical-documentation-hub)
- [Key Features](#-key-features)
- [NLP Evaluation Architecture](#-nlp-evaluation-architecture)
- [System Architecture & Concurrency](#-system-architecture--concurrency)
- [Project Directory Structure](#-project-directory-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Docker Deployment](#-docker-deployment)
- [Development Workflows](#-development-workflows)
- [CLI Commands & Task Automation](#-cli-commands--task-automation)
  - [Makefile Targets Reference](#makefile-targets-reference)
  - [Data Ingestion & Mining Pipeline CLI](#data-ingestion--mining-pipeline-cli)
  - [Running the Application Manually](#running-the-application-manually)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Environment Configuration](#-environment-configuration)
- [Troubleshooting](#-troubleshooting)

---

## 📚 Technical Documentation Hub

For in-depth architectural specifications, mathematical formulations, database schemas, and engineering guides, refer to the technical chapters in the [`docs/`](docs/README.md) hub:

| Chapter | Document | Core Topics Covered |
| :--- | :--- | :--- |
| **Index** | **[`docs/README.md`](docs/README.md)** | Technical documentation hub, reading pathways, system map, and glossary. |
| **01** | **[`01. Architecture Overview`](docs/01_architecture_overview.md)** | Full-stack topology, submission lifecycle, asynchronous precomputation, and concurrency locks. |
| **02** | **[`02. Database & Data Models`](docs/02_database_and_models.md)** | Two-tier lexical schema (`lemmas` vs `words`), dual user tracking, ER diagrams, and SQLite WAL configuration. |
| **03** | **[`03. NLP & Scoring Pipeline`](docs/03_nlp_and_scoring_pipeline.md)** | Multi-stage evaluation (GlotLID, MiniLM semantic adequacy, NLLB perplexity & cross-attention, French GPT-2 fluency, LanguageTool, transition matrix error classification, and transcription diffs). |
| **04** | **[`04. Vocabulary & Progression`](docs/04_vocabulary_and_progression.md)** | Lexical math, Zipf score formula, cumulative rank CEFR cutoffs, Krashen's $i+1$ algorithm, and candidate buffering queues. |
| **05** | **[`05. Ingestion & Mining Pipeline`](docs/05_ingestion_and_mining.md)** | CLI orchestrator, Lexique 4 seeding, multi-source corpus miners (UniversalCEFR, Tatoeba, Lingua, CATIE-AQ, PDFs), LSH deduplication, and strict relational linking. |
| **06** | **[`06. Deployment & Operations`](docs/06_deployment_and_operations.md)** | Dependency management via `uv`, CPU vs CUDA 12.4 GPU PyTorch environments (`UV_NO_SYNC=1`), Docker setups, pytest test markers, and Makefile automation. |

---

## ✨ Key Features

### 1. Dual Practice Modes
- **Spanish-to-French Translation**: Real-time evaluation of user translations against reference targets with fine-grained error feedback.
- **Audio Transcription Training**: High-fidelity listening comprehension exercises where learners transcribe spoken French with character-level diffs, phonetic scoring, and accent verification.

### 2. Pure NMT Cross-Attention & Transition Matrix Error Classification
- **Subtoken Cross-Attention Pooling**: Whole-word bilingual alignment extracted directly from the cross-attention tensors of `facebook/nllb-200-distilled-600M`.
- **Attention Transition Matrix**: Computes the linear algebraic overlap $T = M_{\text{stu}} M_{\text{ref}}^T \in \mathbb{R}^{|W_{\text{stu}}| \times |W_{\text{ref}}|}$ to pair student words with reference counterparts through mutual competitive alignment.
- **Categorized Error Badges & In-Sentence Highlights**:
  - **Type A (Mapped)**: Identifies incorrect word substitutions (e.g., *lui* for *toi*), highlighted in **orange/amber** with replacement targets.
  - **Type B (Unmapped)**: Identifies extraneous or unexpected words without reference counterparts (e.g., adding an unprompted adverb), highlighted in **rose/pink**.

### 3. Multi-Metric Neural Evaluation Pipeline
- Evaluates student input across independent linguistic dimensions: FastText language verification (`glotlid`), SentenceTransformers semantic similarity ($S_{\text{sem}}$), NMT translation conditional perplexity ($S_{\text{NMT}}$), causal French GPT-2 fluency ($S_{\text{fluency}}$), and rule-based grammar checking (`LanguageTool`).

### 4. Psycholinguistic Lexical Modeling & CEFR Progression
- **Two-Tier Lexical Graph**: Disentangles 69,728 canonical dictionary headwords (`lemmas`) from 189,863 surface inflected tokens (`words`), grounded in **Lexique 4**.
- **Continuous Zipf Scoring & Rank Cutoffs**: Standardized logarithmic word frequency $\text{Zipf} = \log_{10}(\text{FreqLemme}) + 3$ mapped to rank thresholds for CEFR levels A1 through C2.
- **Krashen's $i+1$ Comprehensible Input Engine**: Dynamically queries sentences containing **exactly 1 unknown lemma** for progression, balanced with reinforcement sentences containing **0 unknown lemmas**.
- **In-Memory Candidate Buffering**: Pre-samples progression and reinforcement cards to eliminate database scan latencies during active study sessions.

### 5. Neural Speech Synthesis & Caching
- **Edge-TTS Neural Audio**: High-clarity native French neural voices (`fr-FR-HenriNeural`, `fr-FR-DeniseNeural`) generated on-demand with MD5-keyed persistent audio caching.

### 6. Multi-Source Ingestion & Mining Pipeline
- Extract, validate, deduplicate (via MinHash LSH), score, and link sentences from **HuggingFace UniversalCEFR** (`UniversalCEFR/readme_fr`), **Tatoeba**, **Lingua.com**, **CATIE-AQ dialogues**, and **monolingual educational French PDFs** (with automatic MarianMT translation).

### 7. Modern Reactive Frontend
- Built with **React 18 + Vite**, **Tailwind CSS**, and **Heroicons**, featuring full **Dark / Light mode** toggling, responsive layouts, interactive token hint popovers, and score delta banners.

---

## 🧠 NLP Evaluation Architecture

When a student submits a translation, the input is processed through a sequential and parallel multi-stage evaluation pipeline:

```mermaid
flowchart TD
    Input["Student French Submission"] --> Typo["Typographic Normalization & Accent Autocorrect"]
    Typo --> LID{"1. GlotLID FastText Gatekeeper\n(cis-lmu/glotlid)"}
    
    LID -->|Rejected: Non-French| Reject["Immediate Helpful Feedback\n(Bypasses Heavy Neural Scoring)"]
    LID -->|Accepted: Valid French| EvalParallel["Parallel Multi-Metric Scoring"]

    subgraph EvalParallel ["Parallel Neural & Rule Evaluation"]
        direction TB
        S_sem["2. Semantic Adequacy (S_sem)\nSentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)\nCosine similarity with reference embedding"]
        S_nmt["3. NMT Perplexity & Cross-Attention (S_nmt)\nNLLB-200 (facebook/nllb-200-distilled-600M)\nCross-entropy loss + Whole-word attention pooling"]
        S_fl["4. Monolingual Fluency (S_fluency)\nFrench GPT-2 (asi/gpt-fr-cased-base)\nCausal LM shifted cross-entropy on French syntax"]
        S_spell["5. Orthography & Syntax Rules\nLanguageTool (Grammar, agreements & conjugations)"]
    end

    EvalParallel --> TransMatrix["6. Attention Transition Matrix Error Classifier\nT = M_stu @ M_ref.T\nMutual competitive alignment (Type A Mapped vs Type B Unmapped)"]
    TransMatrix --> Aggregator["7. Scoring Engine & CEFR Progression\nComposite weighted score, lemma exposure & unlock commit"]
    Aggregator --> Output["Interactive Feedback, Word Alignment & Delta Banner"]
```

### Detailed Pipeline Stages:
1. **Typographic Normalization**: Automatically normalizes quotes, apostrophes, and spacing around French high punctuation (`:`, `;`, `?`, `!`), providing non-punitive autocorrect hints.
2. **GlotLID Gatekeeper**: Evaluates input using `cis-lmu/glotlid`. Non-French submissions (e.g., Spanish, English, or keyboard spam) fail immediately with pedagogical guidance without wasting GPU/CPU inference on deeper models.
3. **Semantic Adequacy ($S_{\text{sem}}$)**: Measures conceptual fidelity by computing cosine similarity between sentence embeddings:
   $$S_{\text{sem}} = \max(0, \min(100, \cos(\mathbf{e}_{\text{student}}, \mathbf{e}_{\text{ref}}) \times 100))$$
4. **NMT Perplexity ($S_{\text{NMT}}$)**: Evaluates translation likelihood given the Spanish prompt using `facebook/nllb-200-distilled-600M`:
   $$S_{\text{NMT}} = \max\left(0, \min\left(100, \frac{\text{PPL}_{\text{ref}}}{\max(\text{PPL}_{\text{student}}, 10^{-5})} \times 100\right)\right)$$
5. **Monolingual Fluency ($S_{\text{fluency}}$)**: Uses `asi/gpt-fr-cased-base` to evaluate grammatical flow and natural phrasing native to French:
   $$S_{\text{fluency}} = \max\left(0, \min\left(100, \frac{\text{PPL}_{\text{ref}}^{\text{LM}}}{\max(\text{PPL}_{\text{student}}^{\text{LM}}, 10^{-5})} \times 100\right)\right)$$
6. **LanguageTool Syntax & Agreement Engine**: Local grammar checking tuned to ignore stylistic trivialities while catching genuine agreement, gender/number, and verb conjugation errors.
7. **Transition Matrix Error Classification**: Analyzes cross-attention overlap to distinguish misselected words from completely unexpected tokens, driving the inline error highlights and badges in the UI.

---

## ⚡ System Architecture & Concurrency

DeepTutor pairs FastAPI's asynchronous event loop with a multi-threaded ML execution layer to guarantee sub-second user response times:

```mermaid
flowchart TD
    subgraph AsyncFastAPI ["FastAPI Async Server (Uvicorn)"]
        ReqBatch["GET /api/exercises/batch"]
        ReqHint["POST /api/exercises/hint"]
        ReqEval["POST /api/exercises/evaluate-single"]
    end

    subgraph MLController ["ML Concurrency Controller (precompute.py)"]
        T1["Thread 1: Eager Worker<br/>(Background Precomputation)"]
        T2["Thread 2: Interactive Evaluator<br/>(Request-Time Inference)"]
        BundleCache[("In-Memory Session Cache<br/>ReferenceBundle LRU")]
    end

    subgraph DataStore ["Persistence Layer"]
        DB[(SQLite WAL Mode)]
        AudioCache[("TTS Audio Cache<br/>data/audio_cache/")]
    end

    ReqBatch -->|Enqueue next exercise batch| T1
    T1 -->|Precompute ref tensors, PPL & alignments| BundleCache
    ReqHint -->|Instant lookup from bundle| BundleCache
    ReqEval -->|Acquires _eval_lock| T2
    BundleCache -->|Reuse precomputed tensors| T2
    T2 --> DB
    T2 --> AudioCache
```

- **Thread 1 (Eager Precomputation Worker)**: When a batch of sentences is sampled, Thread 1 precomputes reference embeddings, reference perplexity, and source-to-reference attention in the background. Token hints are served instantly from memory.
- **Thread 2 (Interactive Student Evaluator)**: Dedicated to evaluating the student submission against the cached `ReferenceBundle`, avoiding redundant forward passes on reference sentences and eliminating lock contention.

---

## 📁 Project Directory Structure

```
DeepTutor/
├── backend/                        # Modular Python backend application & API
│   ├── api/                        # FastAPI routers & request/response schemas
│   │   ├── audio.py                # Audio TTS streaming & generation endpoints
│   │   ├── deps.py                 # Dependency injection & singleton references
│   │   ├── exercises.py            # Translation & transcription session endpoints
│   │   ├── schemas.py              # Pydantic data contract schemas
│   │   ├── stats.py                # User stats & vocabulary summary endpoints
│   │   └── users.py                # User profile CRUD & level configuration
│   ├── db/                         # Modular SQLite database management (WAL mode)
│   │   ├── connection.py           # SQLite connection pooling & schema initialization
│   │   ├── manager.py              # Unified DataManager repository aggregator
│   │   ├── sentences.py            # Sentence querying, filtering & updates
│   │   ├── users.py                # User profiles, CEFR levels & settings
│   │   ├── history.py              # User exercise attempts & evaluation history
│   │   ├── audio.py                # Cached TTS audio file mapping
│   │   └── vocabulary.py           # User vocabulary state & candidate buffering
│   ├── ml/                         # Machine learning & neural model inference
│   │   ├── engine.py               # MLEngine: neural inference execution & transition matrix
│   │   ├── registry.py             # Singleton model loading & caching registry
│   │   ├── precompute.py           # Background eager loading & ReferenceBundle cache
│   │   └── token_utils.py          # Pure NMT cross-attention pooling & token alignment
│   ├── nlp/                        # Natural language processing & linguistic utilities
│   │   ├── text_utils.py           # Text normalization, tokenization & diff helpers
│   │   ├── tts_engine.py           # Edge-TTS neural speech synthesis & cache
│   │   └── vocab_engine.py         # CEFR vocabulary classifier & lemma extractor
│   ├── scoring/                    # Linguistic evaluation & scoring engines
│   │   ├── engine.py               # Composite scoring aggregation & feedback
│   │   ├── cefr.py                 # Dynamic CEFR level delta computation
│   │   └── transcription.py        # Phonetic, typo, & word diff transcription scoring
│   ├── ingestion/                  # Data mining and dataset ingestion pipeline
│   │   ├── pipeline/               # Mining modules (HuggingFace, Tatoeba, PDF, Lexique 4)
│   │   │   ├── mine_universal_cefr.py # UniversalCEFR HuggingFace mining module
│   │   │   ├── mine_tatoeba.py     # Tatoeba corpus sentence miner
│   │   │   ├── mine_pdf.py         # Monolingual PDF extractor and MarianMT translator
│   │   │   ├── seed_lemmas.py      # Lexique 4 lemmas & words catalog seeder
│   │   │   └── ingest_and_link.py  # Sentence linker with strict lexeme validation
│   │   ├── orchestrator.py         # Ingestion CLI orchestrator entry point
│   │   └── rescore_dataset.py      # Batch rescoring of sentence corpus
│   ├── config.py                   # Centralized application configuration & defaults
│   └── server.py                   # FastAPI application initialization & static mount
├── frontend/                       # Modern React Single Page Application (SPA)
│   ├── src/
│   │   ├── components/             # Reusable UI components with full Dark Mode support
│   │   │   ├── ExerciseSession.jsx # Translation & transcription exercise container
│   │   │   ├── SentenceCard.jsx    # Translation card with transition matrix highlights
│   │   │   ├── TranscriptionCard.jsx # Audio transcription practice card
│   │   │   ├── WordAlignmentView.jsx # Attention alignment visualization
│   │   │   ├── ScoreDeltaBanner.jsx# Level progression and delta feedback banner
│   │   │   ├── Header.jsx          # App navbar, stats, and theme toggler
│   │   │   ├── UserMenu.jsx        # User selection and CEFR level switcher
│   │   │   ├── SettingsModal.jsx   # Exercise settings & preferences modal
│   │   │   └── LoginScreen.jsx     # Profile selection & onboarding screen
│   │   ├── App.jsx                 # Core application controller & exercise state
│   │   ├── main.jsx                # React DOM root entry point
│   │   └── index.css               # Tailwind CSS design system & dark theme tokens
│   ├── package.json                # Frontend npm dependencies & scripts
│   ├── vite.config.js              # Vite bundler configuration & backend proxy
│   └── tailwind.config.js          # Tailwind CSS design system rules
├── data/                           # Application datasets & persistent stores (runtime/ignored)
│   ├── app.db                      # Primary SQLite database (generated)
│   ├── audio_cache/                # Cached synthetic MP3 audio files (generated)
│   ├── lexique/                    # Source Lexique 4 database TSV
│   └── books/                      # Source educational PDFs for sentence mining
├── docs/                           # Architectural and design documentation hub
│   ├── README.md                   # Documentation index & system navigation map
│   ├── assets/                     # Media and visual assets
│   │   ├── demo_preview.gif        # Animated walkthrough GIF preview
│   │   └── demo_frenchtutor.mp4    # Full walkthrough demonstration video (with audio)
│   ├── 01_architecture_overview.md # Chapter 1: System architecture & lifecycle
│   ├── 02_database_and_models.md   # Chapter 2: Database schema & 2-tier lexical model
│   ├── 03_nlp_and_scoring_pipeline.md # Chapter 3: Multi-stage scoring & transition matrix
│   ├── 04_vocabulary_and_progression.md # Chapter 4: Zipf math, CEFR rank & candidate buffer
│   ├── 05_ingestion_and_mining.md  # Chapter 5: Ingestion, miners & relational linking
│   └── 06_deployment_and_operations.md # Chapter 6: Docker, uv, PyTorch & testing
├── tests/                          # Pytest automated test suite
│   ├── conftest.py                 # Shared fixtures & test database isolation
│   ├── test_server.py              # FastAPI endpoint integration tests
│   ├── test_scoring.py             # Scoring & evaluation logic tests
│   ├── test_data_manager.py        # Database operations unit tests
│   ├── test_ml_models.py           # Neural ML model & transition matrix tests
│   ├── test_progressive_vocab.py   # Vocabulary progression unit tests
│   ├── test_transcription.py       # Speech & transcription verification tests
│   └── ...
├── .env.example                    # Environment variable template
├── Makefile                        # Central developer automation tasks
├── pyproject.toml                  # Python package configuration (uv managed)
└── uv.lock                         # Deterministic package dependency pins
```

---

## 📦 Prerequisites

Ensure the following tools are available on your system:
- **Python**: `>= 3.11` (Python 3.11 or 3.12 recommended, `< 3.14`)
- **uv**: Ultra-fast Python package and project manager ([Installation Guide](https://github.com/astral-sh/uv))
- **Node.js**: `>= 18.0` & **npm**: `>= 9.0`
- **Docker & Docker Compose** (Optional, for containerized deployments)
- **NVIDIA GPU with CUDA 12.4+** (Optional, for GPU-accelerated inference)

---

## 🚀 Installation & Setup

### 1. Clone & Configure Environment

```bash
git clone <repo-url> DeepTutor
cd DeepTutor

# Create local environment config
cp .env.example .env
```

> [!NOTE]
> `.env` sets `UV_NO_SYNC=1`. This prevents `uv run` from unintentionally reverting or switching your chosen PyTorch variant (CPU vs CUDA) during command invocations.

### 2. Install Dependencies

Use the project `Makefile` to install both Python dependencies and frontend packages in a single step:

```bash
# Option A: Standard CPU setup (Fast download, ~180MB PyTorch wheel)
make install

# Option B: GPU setup with CUDA 12.4 acceleration (~2.5GB PyTorch wheel)
make install-gpu
```

*(Under the hood, `make install` runs `uv sync --extra cpu` followed by `cd frontend && npm install`)*.

### 3. Load Database Snapshot (Skip Ingestion)

To immediately use the app without mining datasets from scratch (~30,000+ pre-scored sentences and complete Lexique 4 lemmas):

```bash
# Cleanly initialize the database with pre-mined linguistic data
make load-db
```

*(Alternatively, you can run the full ingestion pipeline from scratch using `make ingest ARGS="--all"`)*.

---

## 🐳 Docker Deployment

The application is containerized using Docker, allowing you to run the complete stack (FastAPI backend + React frontend) in unified containers without configuring Python or Node on the host:

### CPU-Only Deployment (Default)
```bash
# Build and start container in detached mode
docker-compose up backend-cpu -d --build

# Inspect container logs
docker-compose logs -f backend-cpu
```

### GPU-Accelerated Deployment (CUDA 12.4)
Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
# Build and start GPU container in detached mode
docker-compose up backend-gpu -d --build

# Inspect container logs
docker-compose logs -f backend-gpu
```

### Development with Live Reloading (Docker Compose)
Mounts source code directly into containers so that edits hot-reload automatically:

```bash
# CPU development (FastAPI reload + Vite HMR)
docker compose -f docker-compose.dev.yml up backend-dev frontend-dev --build

# GPU development (CUDA accelerated backend + Vite HMR)
docker compose -f docker-compose.dev.yml up backend-dev-gpu frontend-dev --build
```

### Accessing the Application
- **Production Web App:** [http://localhost:8000](http://localhost:8000) (or `8001` for GPU deployment)
- **Development Web App (Vite HMR):** [http://localhost:5173](http://localhost:5173)
- **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💻 Development Workflows

### Full-Stack Development (Recommended)

To start both the **FastAPI backend** (with auto-reload) and the **Vite React frontend** (with Hot Module Replacement) concurrently:

```bash
make dev
```

- **Frontend App**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Parallel Worktree Execution (Custom Ports)

When running multiple Git worktrees or feature branches simultaneously, override port assignments to avoid port conflicts:

```bash
make dev BACKEND_PORT=8001 FRONTEND_PORT=5174
```

### Production Build & Launch

To compile the React production bundle into `frontend/dist` and serve all assets through FastAPI:

```bash
# Compile frontend bundle and launch production server
make start

# Or specify a custom production port
make start PORT=8080
```

---

## 🛠️ CLI Commands & Task Automation

### Makefile Targets Reference

| Target | Command | Description |
| :--- | :--- | :--- |
| `make help` | `make help` | Displays list of all available Makefile commands |
| `make install` | `uv sync --extra cpu && cd frontend && npm install` | Installs Python CPU environment and frontend npm dependencies |
| `make install-gpu` | `uv sync --extra cu124 && cd frontend && npm install` | Installs Python GPU environment with CUDA 12.4 and frontend npm dependencies |
| `make dev` | *Spawns FastAPI + Vite HMR* | Runs development servers concurrently with cleanup signal trapping |
| `make build` | `cd frontend && npm run build` | Compiles the production frontend bundle into `frontend/dist` |
| `make start` | `uv run ... python -m backend.server` | Launches production server serving API + static frontend bundle |
| `make test` | `uv run --no-sync pytest` | Runs fast unit, API, and scoring test suite (`<2s`) |
| `make ingest` | `uv run ... backend.ingestion.orchestrator` | Runs database ingestion pipeline (pass `ARGS="..."`) |
| `make load-db` | `uv run ... backend.db.snapshot restore` | Restores default database snapshot into `data/app.db` (skips ingestion) |
| `make dump-db` | `uv run ... backend.db.snapshot dump` | Creates clean database snapshot (excluding user data) in `data/snapshots/` |
| `make clean` | `rm -rf frontend/dist .pytest_cache ...` | Cleans build artifacts and Python `__pycache__` directories |

---

### Data Ingestion & Mining Pipeline CLI

The application includes an ingestion orchestrator to seed the Lexique 4 lexical catalog, mine sentences from multiple corpora, and populate the SQLite database.

Invoke the orchestrator using `uv run` or via `make ingest ARGS="..."`:

```bash
# Display full orchestrator options
uv run python -m backend.ingestion.orchestrator --help
# or: make ingest ARGS="--help"
```

#### Common Ingestion Workflows:

```bash
# 1. Complete default setup (Init DB -> Seed Lexique Lemmas -> Mine Tatoeba -> Ingest & Link)
uv run python -m backend.ingestion.orchestrator --all

# 2. Mine sentences from HuggingFace UniversalCEFR (UniversalCEFR/readme_fr)
uv run python -m backend.ingestion.orchestrator --mine-universal-cefr

# 3. Reset and initialize fresh database schema
uv run python -m backend.ingestion.orchestrator --reset-db

# 4. Seed 69,728 lemmas & 189,863 words from Lexique 4
uv run python -m backend.ingestion.orchestrator --seed-lemmas

# 5. Mine and score bilingual sentences from Tatoeba
uv run python -m backend.ingestion.orchestrator --mine-tatoeba

# 6. Mine, translate (MarianMT), and score sentences from an educational PDF
uv run python -m backend.ingestion.orchestrator --mine-pdf data/books/sample.pdf

# 7. Ingest mined sentences with strict relational lexeme validation
uv run python -m backend.ingestion.orchestrator --ingest

# 8. Clear all sentences and links (maintenance)
uv run python -m backend.ingestion.orchestrator --clear-db
```

---

### Running the Application Manually

You can also start components independently without `make`:

#### Backend FastAPI Server
```bash
# Development mode with hot-reload
uv run python -m backend.server --port 8000 --reload

# Production mode
uv run python -m backend.server --port 8000
```

#### Frontend Vite Server
```bash
cd frontend
npm run dev        # Development HMR server (http://localhost:5173)
npm run build      # Production build to dist/
npm run preview    # Preview production bundle locally
```

---

## 🧪 Testing & Quality Assurance

The test suite uses `pytest` with markers to separate fast unit/API tests from heavyweight neural model tests:

```bash
# 1. Run fast unit and API tests (<2s, skips heavy neural downloads)
make test
# or:
uv run pytest

# 2. Run the entire test suite including heavy ML models
uv run pytest -m ""

# 3. Run only ML model inference & transition matrix tests
uv run pytest -m ml

# 4. Run tests with coverage reporting
uv run pytest --cov=backend --cov-report=term-missing
```

> [!TIP]
> ML tests automatically download weights on the first invocation (~3-4 GB). Fast unit tests mock these layers or use lightweight surrogates to ensure rapid developer iteration.

---

## ⚙️ Environment Configuration

Configuration variables can be customized in a `.env` file at the root of the repository:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `UV_NO_SYNC` | `1` | Prevents `uv` from re-syncing dependencies dynamically on `run` |
| `HOST` | `0.0.0.0` | Bind host address for the FastAPI server |
| `PORT` / `BACKEND_PORT` | `8000` | Port for the FastAPI backend server |
| `FRONTEND_PORT` | `5173` | Port for the Vite frontend development server |
| `RELOAD` | `false` | Enable/disable FastAPI uvicorn auto-reload |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | Hugging Face model cache directory |
| `DATABASE_URL` | `sqlite:///data/app.db` | Path to SQLite database file |

---

## 🔧 Troubleshooting

### Port Already in Use (`8000` or `5173`)
If a background server process was left running:
```bash
# Check process occupying the port
lsof -i :8000 || ss -tulpn | grep 8000

# Terminate process by PID
kill -9 <PID>

# Or terminate directly by port
fuser -k 8000/tcp
```

### PyTorch Flavor Desynchronization
If `uv` inadvertently switches your PyTorch installation between CPU and CUDA:
1. Verify `UV_NO_SYNC=1` is present in `.env`.
2. Reinstall your desired flavor:
   ```bash
   make install       # for CPU
   make install-gpu   # for CUDA 12.4
   ```

### Virtual Environment Reset
To completely reinstall the virtual environment from scratch:
```bash
rm -rf .venv
uv sync --extra cpu   # or --extra cu124
```

---

<p align="center">
  Built with ❤️ for natural, data-driven French language acquisition.
</p>
