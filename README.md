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

*   `app.py`: Main Gradio application. Contains the UI layout, state management, and custom HTML mapping visualization.
*   `models.py`: The `MLEngine` class. Handles model loading, tokenizer management, and implements the math behind the evaluation metrics and attention extraction.
*   `data_manager.py`: Handles CRUD operations for the sentence dataset.
*   `data/sentences.json`: Preloaded dataset of 20 Spanish-French sentence pairs.
*   `app.css`: Custom styling to render the sentence mapping tokens as inline elements instead of standard block elements.
*   `pyproject.toml`: Modern project specification and dependencies managed by `uv`.
*   `requirements.txt`: Python dependencies fallback.
*   `uv.lock`: Deterministic lockfile for dependencies.

## Installation & Execution

1. Sync the environment and install all required dependencies using `uv`:
    ```bash
    uv sync
    ```
2. Run the Gradio application:
    ```bash
    uv run python app.py
    ```
    or run the test suite:
    ```bash
    uv run python test_app.py
    ```
3. *Note: The first run will take several minutes to download the PyTorch models from the Hugging Face Hub (approx. ~3-4 GB).*
