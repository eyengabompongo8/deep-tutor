<!-- Navigation Header -->
[← 02 Database Schema & Models](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/02_database_and_models.md) | [Documentation Index](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/README.md) | [04 Vocabulary & Progression →](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/04_vocabulary_and_progression.md)

---

# Chapter 3: Multi-Stage NLP & Scoring Pipeline

This chapter provides an exhaustive architectural, mathematical, and algorithmic reference for the **linguistic evaluation and scoring pipeline** in **FrenchTutor**.

Rather than relying on brittle string matching or opaque, non-deterministic commercial LLM APIs, FrenchTutor employs a **multi-stage, multi-model evaluation pipeline** composed of fast local neural models and linguistic heuristics:

1. **Gatekeeper**: FastText-based language identification ([`cis-lmu/glotlid`](https://huggingface.co/cis-lmu/glotlid)) with short-utterance French lexicon fallback.
2. **Semantic Adequacy ($S_{\text{sem}}$)**: Dense multilingual bi-encoder embeddings ([`dangvantuan/sentence-camembert-base`](https://huggingface.co/dangvantuan/sentence-camembert-base) or [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)) measuring semantic cosine similarity against reference translations.
3. **NMT Perplexity ($S_{\text{NMT}}$) & Cross-Attention Alignment**: Sequence-to-sequence translation models ([`Helsinki-NLP/opus-mt-es-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-es-fr) and [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)) providing conditional cross-entropy loss, token surprise ($\Delta L$), and cross-attention matrix extraction for token-level Spanish-to-French word alignment.
4. **Monolingual Fluency ($S_{\text{fluency}}$)**: Causal autoregressive French language modeling ([`asi/gpt-fr-cased-small`](https://huggingface.co/asi/gpt-fr-cased-small) / [`asi/gpt-fr-cased-base`](https://huggingface.co/asi/gpt-fr-cased-base)) calculating shifted token-level negative log-likelihood and relative perplexity.
5. **Orthography, Grammar & Phonetic Evaluation**: Two-tier LanguageTool rule matching, typographical normalization, Levenshtein distance, and phonetic sound matching.
6. **Audio Transcription & Pronunciation Scoring**: Word-level diff sequence matching, accent error classification, and acoustic transcription feedback.

```mermaid
flowchart TD
    subgraph InputStage ["Student Submission & Normalization"]
        Sub["Student Input (Text or STT Transcription)"] --> Norm["normalize_french_typography(text)"]
        Norm --> Punct["autocorrect_text(..., lang_tool)"]
        Punct --> Corrected["Normalized & Punctuation-Reconciled Input"]
    end

    subgraph Stage1 ["Stage 1: Gatekeeper"]
        Corrected --> LID{"LID Gatekeeper<br/>FastText (GlotLID)"}
        LID -- "< 6 words in COMMON_FRENCH_WORDS" --> ValidLID["Accept as French"]
        LID -- "pred_lang in [fra, fr, pcd, gcf, gcr] & p >= 0.40" --> ValidLID
        LID -- "Non-French / Unknown" --> RejectLID["Reject Submission<br/>quality = 0.0, is_correct = False"]
    end

    subgraph Stage234 ["Stages 2, 3 & 4: Neural Inference"]
        ValidLID --> ParallelML

        subgraph ParallelML ["Parallel Neural Evaluators"]
            direction TB
            subgraph S2 ["Stage 2: Semantic Adequacy"]
                SemModel["Bi-Encoder (CamemBERT / MiniLM)"]
                SemModel --> CosSim["Cosine Similarity vs Precomputed Ref Emb"]
                CosSim --> SsemScore["S_sem = max(0, min(100, R_cos * 100))"]
            end

            subgraph S3 ["Stage 3: NMT Perplexity & Alignment"]
                NMTModel["Seq2Seq LM (Opus-MT / NLLB-200)"]
                NMTModel --> NMTLoss["Cross-Entropy Loss & PPL_stu"]
                NMTModel --> AttnMatrix["Cross-Attention Layers [-4:]"]
                AttnMatrix --> SimAlign["SimAlign Embedding Fusion"]
                SimAlign --> HybridMatrix["Hybrid Alignment Matrix (M_hybrid)"]
                NMTLoss --> SnmtScore["S_nmt = min(100, (PPL_ref / PPL_stu) * 100)"]
            end

            subgraph S4 ["Stage 4: Monolingual Fluency"]
                GPTModel["Causal LM (asi/gpt-fr-cased)"]
                GPTModel --> GPTLoss["Shifted Cross-Entropy Loss & PPL_stu"]
                GPTLoss --> SfluencyScore["S_fluency = min(100, (PPL_ref / PPL_stu) * 100)"]
            end
        end
    end

    subgraph Stage5 ["Stage 5: Grammar, Orthography & Sound Matching"]
        Corrected --> LT["LanguageTool (fr) Rules"]
        LT --> SyntacticErr["Filter Style / Whitespace / Punctuation Errors"]
        HybridMatrix --> AlignCheck["align_and_filter_unexpected_words(...)"]
        AlignCheck --> SoundCheck{"Phonetic & Functional Sound Matching"}
        SoundCheck --> FlaggedTokens["Token & Word Surprise Breakdowns (Delta_L > 8.0)"]
    end

    subgraph Aggregator ["Composite Score Aggregator & Progression"]
        SsemScore & SnmtScore & SfluencyScore & SyntacticErr & FlaggedTokens --> Engine["ScoringEngine.compute_sentence_quality(...)"]
        Engine --> Decision{"S_sem >= 75 &<br/>S_nmt >= 75 &<br/>S_fluency >= 75"}
        Decision -- Yes --> Pass["is_correct = True<br/>quality = 1.0"]
        Decision -- No --> Fail["is_correct = False<br/>quality = 0.0"]
        Pass & Fail --> Mastery["Record user_lemmas & user_words Appearances"]
        Mastery --> CEFRProg["Update Streak & CEFR Level Ranges"]
    end
```

---

## 1. Pipeline Architecture Overview & Lifecycle

The scoring pipeline runs within the FastAPI execution lifecycle defined in [`backend/api/exercises.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/api/exercises.py), coordinated by [`backend/ml/engine.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/engine.py), [`backend/ml/precompute.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/precompute.py), and [`backend/scoring/engine.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/scoring/engine.py).

### 1.1 Dual-Threaded Concurrency Pattern
Running four neural network forward passes for every submission would introduce unacceptable latency (800ms–1.5s on CPU). To achieve interactive response times (<150ms on CPU, <40ms on GPU), FrenchTutor uses an **eager dual-threaded architecture** implemented in [`backend/ml/precompute.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/precompute.py):

- **Thread 1 (`EagerPrecomputeManager` Background Worker)**:
  When an exercise batch is requested via `GET /api/exercises/batch`, Thread 1 immediately processes the reference sentences in the background. It generates a [`ReferenceBundle`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/precompute.py#L17-L33) containing:
  - $\text{PPL}_{\text{ref}}^{\text{NMT}}$: Reference sentence NMT perplexity.
  - $\text{PPL}_{\text{ref}}^{\text{fluency}}$: Reference sentence causal GPT-2 perplexity.
  - $\mathbf{e}_{\text{ref}}$: Precomputed normalized semantic embedding vector.
  - $\mathbf{A}_{\text{ref}}$: Source-to-reference cross-attention alignment matrix.
  - `first_time_src_indices` and `first_time_tgt_indices`: Indices corresponding to target $i+1$ progression vocabulary.

- **Thread 2 (Student Evaluation Worker)**:
  When the learner submits a translation (`POST /api/exercises/evaluate-single`), the precomputed bundle is retrieved from memory (`_cache`). Neural inference is executed **only on the student's text**. The student metrics are directly compared against the cached baseline.

```
Total Student Latency = Latency(Student Passes) + 0 ms (Reference Passes)
```

---

## 2. Text Normalization & Punctuation Reconciliation

Before any neural model processes user input, the text undergoes strict typographic normalization and reconciliation in [`backend/nlp/text_utils.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/nlp/text_utils.py):

### 2.1 Typographic Normalization: [`normalize_french_typography`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/nlp/text_utils.py#L26-L59)
French text submitted through web inputs frequently contains mismatched apostrophes, inconsistent hyphens, and irregular non-breaking spaces. The function applies deterministic normalization:
1. **Apostrophes**: Normalizes Unicode right/left single quotes (`’`, `‘`, `` ` ``, `´`, `′`) to standard ASCII single quote `'`.
2. **Hyphens & Dashes**: Normalizes non-breaking hyphens, en-dashes, em-dashes, and minus signs (`‑`, `–`, `—`, `−`) to standard ASCII `-`.
3. **Space Sanitization**: Replaces non-breaking spaces (`\u00a0`), narrow non-breaking spaces (`\u202f`), and thin spaces (`\u2009`) with standard spaces, stripping zero-width spaces (`\u200b`, `\ufeff`).
4. **Punctuation Spacing**: Enforces standard French typographic rules by inserting a space before high punctuation marks (`?`, `!`, `:`, `;`) using regex lookbehinds and collapses redundant whitespace.

### 2.2 Terminal Punctuation Reconciliation: [`autocorrect_text`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/nlp/text_utils.py#L61-L150)
Language learners often omit the final period or question mark. If a student writes `"Je vais à la gare"` for a reference sentence ending in `"."` or `"?"`, the terminal punctuation is reconciled automatically:
- Terminal marks (`.`, `?`, `!`) from the reference are appended or harmonized.
- **Tier 1 Typo Correction**: LanguageTool is invoked strictly for spelling typos where edit distance $\le 2$ or casing errors. Genuine grammatical errors, syntax issues, and confused words are explicitly preserved so the scoring engine can penalize them.

---

## 3. Stage 1: FastText Language Identification Gatekeeper

### 3.1 Model & Mechanism
Learners may mistakenly submit answers in the source language (Spanish), native English, or random gibberish. The Gatekeeper in [`MLEngine.check_language`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/engine.py#L83-L106) validates that the candidate is authentic French before triggering heavy neural evaluation.

The system uses **GlotLID** ([`cis-lmu/glotlid`](https://huggingface.co/cis-lmu/glotlid)), an open-source fasttext neural classifier trained on over 2,000 languages:
- **Model File**: Downloaded automatically via `hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")`.
- **Inference**: `predictions, probabilities = self.lid_model.predict(text_clean)`.
- **Validation Criteria**: The top predicted language tag must belong to the French dialect family (`fra`, `fr`, Picard `pcd`, Guadeloupean Creole `gcf`, Guianese Creole `gcr`) with a confidence probability:
  $$P(\text{lang} \in \text{French}) \ge 0.40$$

### 3.2 Short-Utterance False-Negative Bypass
FastText models rely on character n-grams ($n \in [3, 6]$). Extremely short utterances (e.g., *"Oui"*, *"Merci"*, *"C'est bon"*, *"Il va"*) lack sufficient n-gram entropy, causing standard fasttext classifiers to misclassify them or produce low confidence scores.

To prevent false rejections, `MLEngine` implements a short-utterance fallback:
```python
words = re.findall(r"\b[^\W\d_]+\b", text_clean.lower(), re.UNICODE)
if len(words) <= 5 and any(w in COMMON_FRENCH_WORDS for w in words):
    return True, 1.0
```
[`COMMON_FRENCH_WORDS`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/config.py#L46-L54) contains high-frequency foundational French tokens (`le`, `la`, `les`, `est`, `ai`, `suis`, `bonjour`, `merci`, `oui`, `non`, `bien`, etc.). If an utterance is 5 words or fewer and contains at least one recognized word, it bypasses FastText classification.

If language validation fails:
$$\text{is\_correct} \leftarrow \text{False}, \quad \text{quality} \leftarrow 0.0$$
The request short-circuits immediately, conserving CPU/GPU cycles.

---

## 4. Stage 2: Semantic Adequacy ($S_{\text{sem}}$)

### 4.1 Bi-Encoder Dense Embedding
The second stage measures whether the student’s translation preserves the meaning of the source sentence, accommodating valid synonyms (e.g., *voiture* vs. *automobile*, *vélo* vs. *bicyclette*).

Evaluation is performed via dense sentence representation using models registered in [`backend/ml/registry.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/registry.py):
- **Default Production Model**: [`dangvantuan/sentence-camembert-base`](https://huggingface.co/dangvantuan/sentence-camembert-base), a French RoBERTa-based sentence transformer optimized for French semantic similarity.
- **Multilingual Fallback**: [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), a 12-layer distilled cross-lingual MiniLM mapping French and Spanish into a shared 384-dimensional vector space.

### 4.2 Cosine Similarity Formulation
Let $\mathbf{u} = f_{\theta}(s_{\text{student}})$ and $\mathbf{v} = f_{\theta}(s_{\text{ref}})$ be the mean-pooled, $L_2$-normalized dense embeddings:
$$\|\mathbf{u}\|_2 = 1, \quad \|\mathbf{v}\|_2 = 1$$

The raw semantic cosine similarity $R_{\text{cos}}$ is computed across all acceptable reference translations $\{s_{\text{ref}}^{(1)}, \dots, s_{\text{ref}}^{(K)}\}$:
$$R_{\text{cos}} = \max_{k \in [1, K]} \left( \mathbf{u} \cdot \mathbf{v}^{(k)} \right) = \max_{k \in [1, K]} \left( \frac{\mathbf{u} \cdot \mathbf{v}^{(k)}}{\|\mathbf{u}\|_2 \|\mathbf{v}^{(k)}\|_2} \right)$$

The semantic adequacy score $S_{\text{sem}}$ is scaled to a continuous percentage $[0.0, 100.0]$:
$$S_{\text{sem}} = \max\left(0.0, \min\left(100.0, R_{\text{cos}} \times 100.0\right)\right)$$

If precomputed reference embeddings are available in `reference_bundle.ref_sem_emb`, only $\mathbf{u}$ is encoded at runtime.

---

## 5. Stage 3: NMT Perplexity ($S_{\text{NMT}}$) & Cross-Attention Word Alignment

### 5.1 Conditional Sequence-to-Sequence Modeling
Semantic embeddings alone cannot verify whether every individual word in the source Spanish sentence was translated into French. A student could write a grammatically correct French sentence that only translates half the prompt.

Stage 3 uses a neural machine translation (NMT) model:
- **Default Production Model**: [`Helsinki-NLP/opus-mt-es-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-es-fr) (MarianSeq2Seq architecture).
- **Multilingual Model**: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) (No Language Left Behind, 200 languages).

The source text $X = (x_1, \dots, x_M)$ is passed to the encoder, and the candidate French text $Y = (y_1, \dots, y_N)$ is passed as labels to the decoder.

### 5.2 NMT Perplexity & Score Calculation
The model computes the conditional cross-entropy loss over target tokens:
$$\mathcal{L}_{\text{NMT}}(Y \mid X) = -\frac{1}{N} \sum_{i=1}^N \log P(y_i \mid y_{<i}, X; \theta_{\text{NMT}})$$

The sequence perplexity $\text{PPL}_{\text{NMT}}$ represents the model's surprise at generating the candidate tokens:
$$\text{PPL}_{\text{NMT}} = \exp\left( \mathcal{L}_{\text{NMT}}(Y \mid X) \right)$$

The NMT adequacy score $S_{\text{NMT}}$ normalizes the student's perplexity relative to the gold-standard reference sentence perplexity ($\text{PPL}_{\text{ref}}$):
$$S_{\text{NMT}} = \max\left(0.0, \min\left(100.0, \frac{\text{PPL}_{\text{ref}}}{\max(\text{PPL}_{\text{student}}, 10^{-5})} \times 100.0\right)\right)$$

If the student's translation is as probable as or more probable than the reference translation, $S_{\text{NMT}} \to 100.0$. If the student makes translation errors, hallucinating unrelated words or omitting clauses, $\text{PPL}_{\text{student}}$ spikes, driving $S_{\text{NMT}}$ down.

### 5.3 Token-Level Surprise Metric ($\Delta L$)
To identify the specific token that caused an error, [`_build_nmt_breakdowns`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/token_utils.py#L79-L157) calculates the token-level cross-entropy loss $\ell_i = -\log P(y_i \mid y_{<i}, X)$ and compares it to the loss of the model's top-1 greedy prediction:
$$\ell_i^* = -\log P(\hat{y}_i \mid y_{<i}, X) \quad \text{where} \quad \hat{y}_i = \arg\max_v P(v \mid y_{<i}, X)$$
$$\Delta L_i = \max\left(0.0, \ell_i - \ell_i^*\right)$$

When $\Delta L_i > 8.0$ ([`NMT_DELTA_L_THRESHOLD`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/config.py#L20)), the token is flagged as an anomaly (`is_error = True`), and the model's expected token $\hat{y}_i$ is surfaced as `top_prediction`.

### 5.4 Cross-Attention Alignment Matrix Extraction
Transformer seq2seq models contain multi-head cross-attention mechanisms connecting the decoder to encoder representations. In [`HelsinkiNMTModel.infer`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/registry.py#L93-L135), the cross-attention tensors are captured with `output_attentions=True`.

Cross-attention weights from the final 4 decoder layers are stacked and averaged across heads and layers:
$$\mathbf{A}_{\text{subtokens}} = \frac{1}{4 H} \sum_{l=L-3}^L \sum_{h=1}^H \mathbf{A}^{(l, h)} \in \mathbb{R}^{N_{\text{sub}} \times M_{\text{sub}}}$$

Because Marian and NLLB tokenizers use SentencePiece BPE sub-tokens (prefixed with `\u2581`), sub-tokens are mapped back to whole words via [`aggregate_sentencepiece_tokens_to_words`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/token_utils.py#L34-L77), which handles French contraction splitting (e.g., *"d'un"* $\to$ `["d'", "un"]`, *"c'est"* $\to$ `["c'", "est"]`).

### 5.5 Hybrid Word Alignment: Cross-Attention + SimAlign
Raw cross-attention matrices can be noisy for function words and prepositions. FrenchTutor uses a **hybrid alignment algorithm** combining attention pooling with contextual multilingual cosine similarity ([`_compute_hybrid_alignment_matrix`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/token_utils.py#L222-L262)):

1. **Row Normalization**:
   $$\tilde{\mathbf{A}}_{t, s} = \frac{\mathbf{A}_{t, s}}{\sum_{s'} \mathbf{A}_{t, s'} + 10^{-8}}$$
2. **Word-Level Attention Pooling**:
   For target word $W_t^{\text{tgt}}$ consisting of sub-tokens $\mathcal{T}_t$ and source word $W_s^{\text{src}}$ consisting of sub-tokens $\mathcal{S}_s$:
   $$\mathbf{M}_{\text{attn}}(t, s) = \sum_{i \in \mathcal{T}_t} \sum_{j \in \mathcal{S}_s} \tilde{\mathbf{A}}_{i, j}$$
3. **SimAlign Semantic Similarity**:
   Embeddings for source and target words are computed using the semantic model:
   $$\mathbf{M}_{\text{sim}}(t, s) = \max\left(0, \cos\left(f(W_t^{\text{tgt}}), f(W_s^{\text{src}})\right)\right)$$
4. **Hybrid Combination**:
   $$\mathbf{M}_{\text{hybrid}} = 0.5 \times \left(\frac{\mathbf{M}_{\text{attn}}}{\max(\mathbf{M}_{\text{attn}}) + 10^{-8}}\right) + 0.5 \times \mathbf{M}_{\text{sim}}$$

This produces a continuous alignment matrix $\mathbf{M}_{\text{hybrid}} \in [0.0, 1.0]^{|W^{\text{tgt}}| \times |W^{\text{src}}|}$. 

When the user hovers over a Spanish source word $s$ in the UI, [`get_aligned_targets`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/nlp/text_utils.py#L151-L171) resolves the corresponding French targets:
$$\hat{t} = \arg\max_t \mathbf{M}_{t, s}$$
Any other target $t'$ satisfying $\mathbf{M}_{t', s} \ge 0.65 \times \mathbf{M}_{\hat{t}, s}$ and whose row argmax aligns back to $s$ is also included. This enables multi-word expression highlighting (e.g., Spanish *"desayunar"* $\to$ French *"prendre le petit déjeuner"*).

---

## 6. Stage 4: Monolingual Fluency ($S_{\text{fluency}}$)

### 6.1 Autoregressive Causal Language Model
Even if a translation is semantically accurate and maps each source word, it may sound unnatural or unidiomatic to a native speaker.

Stage 4 evaluates monolingual French naturalness using a French autoregressive language model:
- **Default Production Model**: [`asi/gpt-fr-cased-small`](https://huggingface.co/asi/gpt-fr-cased-small) (or [`asi/gpt-fr-cased-base`](https://huggingface.co/asi/gpt-fr-cased-base)), trained on 100GB+ of French text.

### 6.2 Causal Perplexity & Shifted Cross-Entropy
Given student token sequence $Y = (y_1, y_2, \dots, y_N)$, the causal LM predicts token $y_{i+1}$ given history $y_{\le i}$. The logits are shifted by 1 position:
$$\text{logits}_{\text{shift}} = \text{logits}_{[:-1, :]}, \quad \text{labels}_{\text{shift}} = Y_{[1:]}$$

The shifted cross-entropy loss is calculated:
$$\mathcal{L}_{\text{fluency}}(Y) = -\frac{1}{N - 1} \sum_{i=1}^{N - 1} \log P(y_{i+1} \mid y_{\le i}; \theta_{\text{GPT}})$$
$$\text{PPL}_{\text{fluency}} = \exp\left( \mathcal{L}_{\text{fluency}}(Y) \right)$$

The relative fluency score $S_{\text{fluency}}$ compares student naturalness against the native reference:
$$S_{\text{fluency}} = \max\left(0.0, \min\left(100.0, \frac{\text{PPL}_{\text{ref}}^{\text{fluency}}}{\max(\text{PPL}_{\text{student}}^{\text{fluency}}, 10^{-5})} \times 100.0\right)\right)$$

If the student uses unnatural word orders or ungrammatical collocations, causal perplexity spikes, reducing $S_{\text{fluency}}$.

---

## 7. Stage 5: Orthography, Grammar & Phonetic Sound Matching

### 7.1 LanguageTool Rule-Based Engine
Neural language models can overlook subtle agreement rules (e.g., feminine past participle agreement: *"Elle est parti"* vs. *"Elle est partie"*). Stage 5 integrates a local rule-based engine:
- **Engine**: [`language_tool_python.LanguageTool('fr')`](https://pypi.org/project/language-tool-python/).
- **Filtered Categories**: FrenchTutor filters out non-pedagogical stylistic suggestions in [`MLEngine.evaluate_syntax`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/engine.py#L155-L178):
  - **Ignored Categories**: `STYLE`, `TYPOGRAPHY`, `PUNCTUATION`, `COLLOQUIALISMS`, `REDUNDANCY`.
  - **Ignored Issue Types**: `style`, `register`, `typographical`, `whitespace`, `locale-violation`.
- **Targeted Rules**: Retains genuine syntax errors, verb conjugations, gender/number disagreements, and confused homophones (e.g., *a/à*, *ou/où*, *son/sont*, *ce/se*).

### 7.2 Soundness & Phonetic Matching Heuristics
When a student word triggers a high surprise score ($\Delta L > 8.0$), the system verifies whether the substitution is linguistically acceptable before marking it as an error:

Implemented in [`align_and_filter_unexpected_words`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/ml/engine.py#L179-L264):
1. **Triangular Alignment Lookup**:
   Using the student alignment matrix $\mathbf{M}_{\text{stu}}$ and reference alignment matrix $\mathbf{M}_{\text{ref}}$, the system resolves which source word $s$ the student word $w_{\text{stu}}$ was intended to translate, and which reference word $w_{\text{ref}}$ was expected:
   $$s^* = \arg\max_s \mathbf{M}_{\text{stu}}[w_{\text{stu}}, s], \quad w_{\text{ref}}^* = \arg\max_r \mathbf{M}_{\text{ref}}[r, s^*]$$

2. **Soundness Verification**:
   The word is marked as valid (`is_error = False`) if any of these conditions hold:
   - **Exact Normalized Match**: `clean(w_stu) == clean(w_ref*)`.
   - **Functional Word Equivalence**: Both words belong to [`FUNCTION_WORDS`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/config.py#L39-L43) (e.g., valid alternations like *"un"* vs. *"le"*, *"ce"* vs. *"cet"*).
   - **Semantic Near-Synonym**: Embedding cosine similarity between isolated words $\ge 0.70$.
   - **Phonetic Equivalence & Typo Proximity**: Edit distance $\le 2$ or identical phonetic representations without accents (`remove_accents`).

If the word fails all soundness checks, it is flagged as an error, with the expected reference token presented to the user.

---

## 8. Composite Score Aggregation & CEFR Progression

### 8.1 Binary Correctness & Quality Computation
FrenchTutor evaluates translations through a unified criteria function: [`ScoringEngine.compute_sentence_quality`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/scoring/engine.py#L269-L298).

A submission is marked correct (`is_correct = True`, `quality = 1.0`) if and only if **all three neural metrics meet or exceed the threshold** ($\tau = 75.0$, defined by [`DEFAULT_CORRECTION_THRESHOLD`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/config.py#L19)):

$$\text{is\_correct} = \begin{cases} 
\text{True} & \text{if } \text{is\_french} \land (S_{\text{sem}} \ge \tau) \land (S_{\text{NMT}} \ge \tau) \land (S_{\text{fluency}} \ge \tau) \\
\text{False} & \text{otherwise}
\end{cases}$$

$$\text{quality} = \begin{cases} 1.0 & \text{if } \text{is\_correct} \\ 0.0 & \text{otherwise} \end{cases}$$

```python
is_correct = bool(s_sem >= threshold and s_nmt >= threshold and s_fluency >= threshold)
return {
    "is_correct": is_correct,
    "quality": 1.0 if is_correct else 0.0
}
```

This strict multi-metric conjunction prevents partial errors from passing:
- A sentence with high semantic similarity but broken grammar fails $S_{\text{fluency}}$.
- A grammatically fluent sentence that alters the intended meaning fails $S_{\text{sem}}$ or $S_{\text{NMT}}$.
- A non-French sentence fails the Gatekeeper immediately.

### 8.2 Streak & CEFR Level Progression
When an exercise session completes, [`ScoringEngine.calculate_score_adjustment`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/scoring/engine.py#L299-L424) updates the user's progress:
1. **Batch Accuracy Rate**:
   $$\text{accuracy\_rate} = \frac{\sum_{i=1}^B \mathbb{I}(\text{is\_correct}_i)}{B}$$
2. **Streak Adjustments**:
   - If $\text{accuracy\_rate} \ge 0.66$ (66%), the active mode streak increments: $\text{streak} \leftarrow \text{streak} + 1$.
   - If $\text{accuracy\_rate} < 0.33$ (33%), the streak resets: $\text{streak} \leftarrow 0$.
   - Intermediate accuracy ($0.33 \le \text{rate} < 0.66$) maintains the current streak.
3. **Database Synchronization**:
   Stats are written to the `user_modes` and `exercise_history` SQLite tables via `DataManager.update_user_mode` and `DataManager.record_exercise_history`.

### 8.3 CEFR Continuous Mapping
For continuous calibration, [`backend/scoring/cefr.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/scoring/cefr.py) provides mapping between continuous scores $[1.0, 6.0]$ and CEFR levels:

| Score Interval | CEFR Level | Target Proficiency Description |
| :--- | :---: | :--- |
| $[1.0, 2.0)$ | **A1** | Breakthrough: basic everyday expressions and phrases |
| $[2.0, 3.0)$ | **A2** | Waystage: routine tasks and direct exchanges |
| $[3.0, 4.0)$ | **B1** | Threshold: main points of clear standard input |
| $[4.0, 5.0)$ | **B2** | Vantage: complex texts and spontaneous interaction |
| $[5.0, 6.0)$ | **C1** | Effective Operational: flexible language for complex subjects |
| $6.0+$ | **C2** | Mastery: native-equivalent precision and nuance |

```python
LEVEL_RANGES = [
    (1.0, 2.0, 'A1'),
    (2.0, 3.0, 'A2'),
    (3.0, 4.0, 'B1'),
    (4.0, 5.0, 'B2'),
    (5.0, 6.0, 'C1'),
    (6.0, 6.01, 'C2'),
]
```

---

## 9. Audio Transcription & Pronunciation Scoring

FrenchTutor supports an audio transcription training mode (`mode = "transcription"`), evaluated through [`backend/scoring/transcription.py`](file:///home/eyenga/Disk2/Projects/FrenchTutor/backend/scoring/transcription.py).

### 9.1 Evaluation Algorithm
In transcription mode, the student listens to native audio and transcribes what they heard. The transcription engine evaluates input against the reference text using token-level and character-level alignment:

```mermaid
flowchart TD
    RawAudio["Native Audio (.wav / .mp3)"] --> StudentTrans["Student Typed Transcription"]
    StudentTrans --> Norm["normalize_french_typography()"]
    Norm --> WordTokenize["Regex Token Extraction: WORD_RE"]
    WordTokenize --> AccentStrip["remove_accents(word.lower())"]
    AccentStrip --> DiffMatch["difflib.SequenceMatcher(ref_norm, stu_norm)"]
    DiffMatch --> OpCodes{"Opcode Evaluation"}

    OpCodes -- "equal" --> CheckCaseAcc{"Case / Accent Check"}
    CheckCaseAcc -- "r.lower() == s.lower()" --> Exact["type: 'exact'"]
    CheckCaseAcc -- "Different accents" --> AccentErr["type: 'accent'<br/>Note: 's' -> 'r'"]

    OpCodes -- "replace" --> EditDist{"Levenshtein Distance"}
    EditDist -- "Same sans accents" --> AccentErr
    EditDist -- "dist <= 2" --> Typo["type: 'typo'<br/>Note: 's' -> 'r'"]
    EditDist -- "dist > 2" --> Mismatch["type: 'mismatch'"]

    OpCodes -- "delete" --> Missing["type: 'missing' (in reference, omitted by student)"]
    OpCodes -- "insert" --> Extra["type: 'extra' (hallucinated by student)"]

    Exact & AccentErr & Typo & Mismatch & Missing & Extra --> GradeSummary["Binary Correctness:<br/>all tokens exact & count matches ref"]
```

### 9.2 Token Extraction & Matching
Tokens are extracted using a Unicode regex that preserves intra-word apostrophes and hyphens:
```python
WORD_RE = re.compile(r"\b[^\W\d_]+(?:[-'][^\W\d_]+)*\b", flags=re.UNICODE)
```
Reference tokens and student tokens are normalized without accents (`remove_accents(w.lower())`), and aligned using `difflib.SequenceMatcher`.

### 9.3 Error Categorization & Scoring Output
The sequence matcher opcodes are classified into five diagnostic categories:
- **`exact`**: Identical token (including accents and casing).
- **`accent`**: Base word matches, but accentuation is missing or incorrect (e.g., *a* vs. *à*, *ete* vs. *été*, *cote* vs. *côté*).
- **`typo`**: Levenshtein distance $\le 2$ between student and reference tokens.
- **`mismatch`**: Unrelated word substitution (Levenshtein distance $> 2$).
- **`missing`**: Reference word omitted in the student transcription.
- **`extra`**: Extraneous word added by the student.

Binary transcription correctness requires all reference words to be present with zero errors:
$$\text{is\_correct} = \left( |\text{diff\_items}| == |W_{\text{ref}}| \right) \land \forall d \in \text{diff\_items}, \, d.\text{type} == \text{"exact"}$$
$$\text{quality} = \begin{cases} 1.0 & \text{if } \text{is\_correct} \\ 0.0 & \text{otherwise} \end{cases}$$
$$\text{composite\_score} = \begin{cases} 100.0 & \text{if } \text{is\_correct} \\ 0.0 & \text{otherwise} \end{cases}$$

---

## 10. End-to-End Evaluation Request Flow

The sequence diagram below traces an evaluation request through the multi-stage pipeline:

```mermaid
sequenceDiagram
    autonumber
    actor Learner as User Interface (React)
    participant API as FastAPI Router (/api/exercises)
    participant Pre as EagerPrecomputeManager
    participant ML as MLEngine
    participant Score as ScoringEngine
    participant DM as DataManager & DB

    Learner->>API: POST /api/exercises/evaluate-single<br/>{sentence_id, source_text, student_text, reference_text}
    
    API->>API: autocorrect_text(student_text, reference_text)<br/>(Typography, trailing punctuation, tier-1 typos)

    API->>Pre: evaluate_student(source_text, corrected_text, reference_text)
    
    activate Pre
    Pre->>Pre: get_or_compute_bundle()<br/>Retrieve cached ReferenceBundle
    Note over Pre: Reuses ref_sem_emb, nmt_ppl_ref, fluency_ppl_ref!
    
    Pre->>ML: evaluate_student_against_bundle(source, student, bundle)
    activate ML
    
    ML->>ML: check_language(student_text) [FastText GlotLID]
    alt Not French
        ML-->>Pre: is_french = False
    else Is French
        par Neural Evaluation
            ML->>ML: evaluate_semantics(student_text, bundle.ref_sem_emb)<br/>-> S_sem, raw_cos
        and
            ML->>ML: evaluate_nmt_and_map(source_text, student_text)<br/>-> nmt_ppl_stu, hybrid matrix, token breakdowns
        and
            ML->>ML: evaluate_fluency(student_text)<br/>-> fluency_ppl_stu, token breakdowns
        and
            ML->>ML: evaluate_syntax(student_text)<br/>LanguageTool rule matching
        end
        ML->>ML: align_and_filter_unexpected_words(...)
    end
    ML-->>Pre: Evaluation metrics dict
    deactivate ML
    Pre-->>API: Raw evaluation payload
    deactivate Pre

    API->>Score: compute_sentence_quality(result, threshold=75.0)
    Score-->>API: {is_correct, quality (1.0 or 0.0)}

    API->>DM: record_token_appearances(user_id, sentence_id, is_success)
    DM->>DM: Update user_lemmas & user_words mastery counters
    DM-->>API: Updated vocabulary progress

    API-->>Learner: 200 OK Response<br/>{is_correct, quality, breakdowns, alignments, vocab_progress}
```

---

<!-- Navigation Footer -->
[← 02 Database Schema & Models](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/02_database_and_models.md) | [Documentation Index](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/README.md) | [04 Vocabulary & Progression →](file:///home/eyenga/Disk2/Projects/FrenchTutor/docs/04_vocabulary_and_progression.md)
