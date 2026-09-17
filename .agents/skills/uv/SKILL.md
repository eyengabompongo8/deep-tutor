---
name: uv
description: >-
  Manage Python dependencies, virtual environments, packaging, and command execution using uv in this project. Use when running Python scripts, running tests, adding or removing dependencies, syncing the virtual environment, or modifying pyproject.toml.
---

# UV Skill for FrenchTutor Project

This skill guides AI agents on using `uv`—the ultra-fast Python package and project manager—in the `FrenchTutor` project.

---

## 1. Project Context & Configuration

* **Project Mode**: Configured as an application (`[tool.uv] package = false` in `pyproject.toml`).
* **Python Requirement**: `>=3.11` (typically using Python 3.11 or 3.12).
* **Key Constraints**: `numpy<2` (enforced for compatibility with PyTorch, SentencePiece, FastText).
* **Virtual Environment**: Located at `.venv/` at the workspace root.
* **Lockfile**: `uv.lock` maintains deterministic package pins.

---

## 2. Common Workflows & Command Reference

### Running Applications & Scripts

Always run Python commands through `uv run` to ensure the isolated virtual environment is used with proper dependencies:

```bash
# Run the FastAPI web server with frontend
uv run python -m backend.server

# Run the Gradio application
uv run python -m backend.app

# Run the dedicated test suite
uv run pytest                  # Fast unit & API tests (<2s)
uv run python run_tests.py     # Fast unit & API tests
uv run pytest -m ""            # All tests including heavy ML models
uv run python run_tests.py --all

# Run arbitrary scripts
uv run python scripts/translate_sentences.py
```

> [!TIP]
> If `uv run` is invoked and dependencies in `pyproject.toml` or `uv.lock` are out of sync, `uv` will automatically synchronize `.venv` before executing.

### Adding & Removing Dependencies

Modify `pyproject.toml` and lockfile consistently using `uv add` and `uv remove`:

```bash
# Add a new runtime dependency
uv add <package_name>

# Add with version constraint
uv add "fasttext>=0.9.3"

# Add development/test dependencies
uv add --dev pytest pytest-cov

# Remove a dependency
uv remove <package_name>
```

### Syncing the Virtual Environment

Ensure the `.venv` matches `uv.lock` exactly with the desired PyTorch hardware accelerator:

```bash
# Sync with CPU-only PyTorch (~180 MB)
uv sync --extra cpu

# Sync with GPU CUDA 12.4 PyTorch (~2.5 GB+)
uv sync --extra cu124

# Sync including development test dependencies
uv sync --extra cpu --dev

# Sync strictly from existing uv.lock without updating it
uv sync --frozen --extra cpu
```

### Upgrading Dependencies

```bash
# Upgrade all packages in the lockfile within pyproject.toml constraints
uv lock --upgrade

# Upgrade a specific package
uv lock --upgrade-package transformers
```

---

## 3. Project-Specific Nuances

### NLP Models & Data Downloads
- The application relies on Hugging Face models (`cis-lmu/glotlid`, `facebook/nllb-200-distilled-600M`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `asi/gpt-fr-cased-base`).
- When running tests or the app for the first time, initial model caching will occur.
- When working with SpaCy models or LanguageTool within `uv`:
  ```bash
  # Download a SpaCy language model into the environment if needed
  uv run python -m spacy download fr_core_news_sm
  ```

### Direct Venv Binary Fallback
If direct binary execution is needed:
- Python interpreter: `.venv/bin/python`
- Executable tools installed in environment: `.venv/bin/<tool_name>` (e.g. `.venv/bin/gradio`)

---

## 4. Troubleshooting & Verification

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError` | Missing package or unsynced `.venv` | Run `uv sync` or `uv add <package_name>` |
| NumPy 2.x ABI conflict | Incompatible newer NumPy installed | Verify `numpy<2` in `pyproject.toml` and run `uv sync` |
| Lockfile desynchronization | Manual edits to `pyproject.toml` | Run `uv lock` followed by `uv sync` |
| Corrupt environment | Broken `.venv` state | Recreate: `rm -rf .venv && uv venv && uv sync` |
