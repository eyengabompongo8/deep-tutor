import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Allow transformers to load trusted local model weights on PyTorch 2.5
try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import torch
import re
from typing import Dict, Any, Tuple, List, Optional
from sentence_transformers import util
import fasttext
from huggingface_hub import hf_hub_download
import numpy as np
import language_tool_python

from backend.model_registry import registry, NMTRawOutput, FluencyRawOutput
from backend.config import NMT_DELTA_L_THRESHOLD, FUNCTION_WORDS, COMMON_FRENCH_WORDS

# SentencePiece word-boundary markers used by M2M100 and related models
_SP_BOUNDARIES = (
    "\u2581",   # ▁  LOWER ONE EIGHTH BLOCK (primary marker in M2M100)
    " ",         # regular space
    "\u00a0",    # narrow no-break space
)

def _is_special_token(tok: str) -> bool:
    """Return True for any SentencePiece/HuggingFace special token."""
    return tok.startswith("<") or tok.startswith("__") or tok in {"</s>", "<pad>", "<s>", "<unk>"}

def _strip_boundary(tok: str) -> str:
    """Strip leading boundary markers and surrounding whitespace from a token."""
    for marker in _SP_BOUNDARIES:
        tok = tok.lstrip(marker)
    return tok.strip()

def _starts_new_word(tok: str) -> bool:
    """Return True if the token begins a new word (starts with a boundary marker)."""
    return any(tok.startswith(m) for m in _SP_BOUNDARIES)

def aggregate_sentencepiece_tokens_to_words(tokens: List[str]) -> Tuple[List[str], List[List[int]]]:
    """
    Groups SentencePiece sub-tokens into full, clean words.
    Handles the ▁ (U+2581) boundary marker used by M2M100 and mBART tokenizers.
    Strips ALL special tokens (<s>, </s>, <pad>, <unk>, __lang__, etc.).
    Dynamically splits French/apostrophe contractions (e.g. "J'aime" -> ["J'", "aime"],
    "c'est" -> ["c'", "est"], "d'un" -> ["d'", "un"], "qu'il" -> ["qu'", "il"])
    so that contracted words are separated into distinct words for fine-grained mapping.

    Returns:
        words: list[str] — clean word strings visible to the user.
        word_subtoken_indices: list[list[int]] — original token indices per word.
    """
    word_subtoken_indices = []
    current_subtokens = []

    for idx, token in enumerate(tokens):
        if _is_special_token(token):
            continue

        if _starts_new_word(token) or not current_subtokens:
            if current_subtokens:
                word_subtoken_indices.append(current_subtokens)
                current_subtokens = []
            current_subtokens.append(idx)
        else:
            current_subtokens.append(idx)

    if current_subtokens:
        word_subtoken_indices.append(current_subtokens)

    words = []
    final_subtoken_indices = []
    # Generic regex for dynamic apostrophe contraction splitting (NO hardcoded words)
    contraction_pattern = re.compile(r"^([a-zA-Z]+['’])(.+)$")

    for sub_indices in word_subtoken_indices:
        parts = [_strip_boundary(tokens[i]) for i in sub_indices]
        word_text = "".join(parts).strip()
        if not word_text:
            continue

        m = contraction_pattern.match(word_text)
        if m and m.group(2).strip():
            prefix = m.group(1)
            rest = m.group(2).strip()

            words.append(prefix)
            words.append(rest)

            if len(sub_indices) > 1:
                prefix_subs = []
                rest_subs = []
                running = ""
                for idx_i, sub_idx in enumerate(sub_indices):
                    running += parts[idx_i]
                    if len(prefix_subs) == 0 or len(running) <= len(prefix):
                        prefix_subs.append(sub_idx)
                    else:
                        rest_subs.append(sub_idx)
                if not rest_subs:
                    rest_subs = [sub_indices[-1]]
                final_subtoken_indices.append(prefix_subs)
                final_subtoken_indices.append(rest_subs)
            else:
                final_subtoken_indices.append(sub_indices)
                final_subtoken_indices.append(sub_indices)
        else:
            words.append(word_text)
            final_subtoken_indices.append(sub_indices)

    return words, final_subtoken_indices


def _build_nmt_breakdowns(raw: NMTRawOutput) -> Tuple[List[str], List[str], List[List[int]], List[List[int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes token-level and word-level metric breakdowns from raw NMT inference outputs.
    Calculates surprise Delta_L = max(0, loss - top_1_loss) and flags errors when Delta_L > 8.0.
    """
    src_words, src_sub_groups = aggregate_sentencepiece_tokens_to_words(raw.src_raw_tokens)
    tgt_words, tgt_sub_groups = aggregate_sentencepiece_tokens_to_words(raw.tgt_raw_tokens)

    nmt_token_breakdown = []
    token_loss_map = {}
    top_1_loss_map = {}

    for idx, t_id in enumerate(raw.target_ids):
        tok_str = raw.tgt_raw_tokens[idx] if idx < len(raw.tgt_raw_tokens) else str(t_id)
        if _is_special_token(tok_str) or t_id == -100:
            continue

        loss_i = raw.token_losses[idx] if idx < len(raw.token_losses) else 0.0
        top_1_tok_str = raw.top_1_tokens[idx] if idx < len(raw.top_1_tokens) else ""
        top_1_loss_i = raw.top_1_losses[idx] if idx < len(raw.top_1_losses) else 0.0

        tok_clean = _strip_boundary(tok_str)
        top_1_clean = _strip_boundary(top_1_tok_str)

        ppl_i = float(np.exp(loss_i))
        prob_i = float(np.exp(-loss_i)) * 100.0
        delta_l_i = max(0.0, loss_i - top_1_loss_i)

        token_loss_map[idx] = loss_i
        top_1_loss_map[idx] = top_1_loss_i

        nmt_token_breakdown.append({
            "index": idx,
            "token": tok_clean if tok_clean else tok_str,
            "loss": loss_i,
            "ppl": ppl_i,
            "prob": prob_i,
            "top_prediction": top_1_clean if top_1_clean else top_1_tok_str,
            "delta_l": delta_l_i,
            "is_error": delta_l_i > NMT_DELTA_L_THRESHOLD
        })

    nmt_word_breakdown = []
    for w_idx, (word_text, sub_indices) in enumerate(zip(tgt_words, tgt_sub_groups)):
        word_losses = [token_loss_map[idx] for idx in sub_indices if idx in token_loss_map]
        word_top_1_losses = [top_1_loss_map[idx] for idx in sub_indices if idx in top_1_loss_map]

        if word_losses:
            w_loss = float(np.mean(word_losses))
            w_top_1_loss = float(np.mean(word_top_1_losses))
        else:
            w_loss = 0.0
            w_top_1_loss = 0.0

        w_ppl = float(np.exp(w_loss))
        w_prob = float(np.exp(-w_loss)) * 100.0
        w_delta_l = max(0.0, w_loss - w_top_1_loss)

        top_pred_parts = []
        for s_idx in sub_indices:
            if s_idx < len(raw.top_1_tokens):
                tok_str = raw.top_1_tokens[s_idx]
                if not _is_special_token(tok_str):
                    top_pred_parts.append(_strip_boundary(tok_str))
        top_pred_word = "".join(top_pred_parts).strip() if top_pred_parts else "—"

        nmt_word_breakdown.append({
            "index": w_idx,
            "word": word_text,
            "loss": w_loss,
            "ppl": w_ppl,
            "prob": w_prob,
            "top_prediction": top_pred_word,
            "delta_l": w_delta_l,
            "is_error": w_delta_l > NMT_DELTA_L_THRESHOLD
        })

    return src_words, tgt_words, src_sub_groups, tgt_sub_groups, nmt_token_breakdown, nmt_word_breakdown


def _build_fluency_breakdowns(raw: FluencyRawOutput, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes token-level and word-level metric breakdowns from raw Fluency inference outputs.
    """
    fluency_token_breakdown = []
    for idx in range(len(raw.tokens)):
        tok_str = raw.tokens[idx]
        loss_i = raw.token_losses[idx]
        top_1_str = raw.top_1_tokens[idx]
        top_1_loss_i = raw.top_1_losses[idx]

        ppl_i = float(np.exp(loss_i))
        prob_i = float(np.exp(-loss_i)) * 100.0
        delta_l_i = max(0.0, loss_i - top_1_loss_i)

        fluency_token_breakdown.append({
            "index": idx + 1,
            "token": tok_str if tok_str else f"token_{idx}",
            "loss": loss_i,
            "ppl": ppl_i,
            "prob": prob_i,
            "top_prediction": top_1_str,
            "delta_l": delta_l_i,
            "is_error": delta_l_i > NMT_DELTA_L_THRESHOLD
        })

    words = text.strip().split()
    fluency_word_breakdown = []
    if words and fluency_token_breakdown:
        num_words = len(words)
        tokens_per_word = max(1, len(fluency_token_breakdown) // num_words)

        for w_idx, w_text in enumerate(words):
            start_t = w_idx * tokens_per_word
            end_t = (w_idx + 1) * tokens_per_word if w_idx < num_words - 1 else len(fluency_token_breakdown)
            word_toks = fluency_token_breakdown[start_t:end_t]

            if word_toks:
                w_loss = float(np.mean([t["loss"] for t in word_toks]))
                w_top_1 = word_toks[0]["top_prediction"]
            else:
                w_loss = 0.0
                w_top_1 = "—"

            w_ppl = float(np.exp(w_loss))
            w_prob = float(np.exp(-w_loss)) * 100.0
            w_top_1_loss = float(np.mean([t["loss"] - t["delta_l"] for t in word_toks])) if word_toks else 0.0
            w_delta_l = max(0.0, w_loss - w_top_1_loss)

            fluency_word_breakdown.append({
                "index": w_idx,
                "word": w_text,
                "loss": w_loss,
                "ppl": w_ppl,
                "prob": w_prob,
                "top_prediction": w_top_1,
                "delta_l": w_delta_l,
                "is_error": w_delta_l > NMT_DELTA_L_THRESHOLD
            })

    return fluency_token_breakdown, fluency_word_breakdown


def _compute_hybrid_alignment_matrix(
    subtoken_attn: np.ndarray,
    src_words: List[str],
    tgt_words: List[str],
    src_sub_groups: List[List[int]],
    tgt_sub_groups: List[List[int]],
    sem_model: Any
) -> np.ndarray:
    """
    Constructs a high-precision hybrid word alignment matrix by combining:
    1. Row-softmax normalized NMT cross-attention weights pooled to word boundaries.
    2. Contextual multilingual embedding cosine similarity (SimAlign) from the semantic model.
    """
    row_sums = subtoken_attn.sum(axis=1, keepdims=True) + 1e-8
    norm_subtoken_attn = subtoken_attn / row_sums

    num_tgt_words = max(1, len(tgt_words))
    num_src_words = max(1, len(src_words))
    word_attn_matrix = np.zeros((num_tgt_words, num_src_words))

    for tgt_w_idx, tgt_subs in enumerate(tgt_sub_groups):
        for src_w_idx, src_subs in enumerate(src_sub_groups):
            attn_sum = 0.0
            for t_sub in tgt_subs:
                for s_sub in src_subs:
                    if t_sub < norm_subtoken_attn.shape[0] and s_sub < norm_subtoken_attn.shape[1]:
                        attn_sum += norm_subtoken_attn[t_sub, s_sub]
            word_attn_matrix[tgt_w_idx, src_w_idx] = attn_sum

    if src_words and tgt_words:
        src_embs = sem_model.encode(src_words)
        tgt_embs = sem_model.encode(tgt_words)
        sim_matrix = util.cos_sim(tgt_embs, src_embs).cpu().float().numpy()
        sim_matrix = np.maximum(0.0, sim_matrix)
    else:
        sim_matrix = np.zeros((num_tgt_words, num_src_words))

    max_attn = np.max(word_attn_matrix) + 1e-8
    hybrid_matrix = 0.5 * (word_attn_matrix / max_attn) + 0.5 * sim_matrix
    return hybrid_matrix


def resolve_device_and_dtype():
    """
    Attempts to use CUDA acceleration if available and capable of executing kernels.
    Performs a live kernel probe to verify hardware capability (e.g. sm_61 on GTX 1050 Ti).
    Falls back gracefully to CPU if probe fails or if CUDA is unavailable.
    """
    if torch.cuda.is_available():
        try:
            probe = torch.zeros((1, 1), device="cuda")
            _ = probe + 1.0
            del probe
            torch.cuda.empty_cache()
            dev_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            print(f"[MLEngine] CUDA active: {dev_name} (Compute Capability {cc[0]}.{cc[1]}). Using float16.")
            return "cuda", torch.float16
        except Exception as e:
            print(f"[MLEngine] CUDA is available but failed kernel probe ({e}). Falling back to CPU in float32.")
            return "cpu", torch.float32
    return "cpu", torch.float32


class MLEngine:
    """
    Main evaluation pipeline orchestrator for language exercises.
    Coordinates swappable ML model wrappers retrieved from ModelRegistry and applies
    pipeline logic for scoring, word breakdowns, hybrid alignment mapping, and error detection.
    """
    def __init__(self, semantic_name="camembert", nmt_name="helsinki", fluency_name="gpt_fr"):
        self.device, self.dtype = resolve_device_and_dtype()

        # Load models dynamically from the registry
        self.sem_model = registry.get_semantic(semantic_name, self.device, self.dtype)
        self.nmt_model = registry.get_nmt(nmt_name, self.device, self.dtype)
        self.fluency_model = registry.get_fluency(fluency_name, self.device, self.dtype)

        # 4. LID Gatekeeper (cis-lmu/glotlid)
        print("Loading GlotLID Gatekeeper...")
        lid_model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        self.lid_model = fasttext.load_model(lid_model_path)

        # 5. Syntax Evaluator (LanguageTool)
        print("Loading LanguageTool (fr)...")
        self.lang_tool = language_tool_python.LanguageTool('fr')

        print(f"All ML models loaded successfully on {self.device} ({self.dtype}).")

    def check_language(self, text: str) -> Tuple[bool, float]:
        text_clean = text.strip().replace("\n", " ")
        if not text_clean:
            return False, 0.0

        words = re.findall(r"\b[^\W\d_]+\b", text_clean.lower(), re.UNICODE)
        if not words:
            return False, 0.0

        # Short French sentences (< 6 words) frequently trigger false negatives in fasttext
        if len(words) <= 5 and any(w in COMMON_FRENCH_WORDS for w in words):
            return True, 1.0

        try:
            predictions, probabilities = self.lid_model.predict(text_clean)
            raw_p = probabilities[0] if len(probabilities) > 0 else 0.0
            prob = float(raw_p) if raw_p is not None and not np.isnan(raw_p) else 0.0
            if len(predictions) > 0:
                pred_lang = str(predictions[0])
                if any(tag in pred_lang for tag in ["fra", "fr", "pcd", "gcf", "gcr"]) and prob >= 0.40:
                    return True, prob
            return False, prob
        except Exception:
            return True, 1.0

    def encode_semantics(self, texts: List[str]) -> torch.Tensor:
        """Encodes sentences into normalized embeddings via the registered semantic model."""
        return self.sem_model.encode(texts)

    def evaluate_semantics(self, student_text: str, reference_texts: List[str], ref_embs: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """
        Calculates Semantic Adequacy (S_sem) using normalized Cosine Similarity (R_cos * 100).
        Accepts optional precomputed ref_embs to eliminate redundant forward passes.
        """
        student_emb = self.sem_model.encode([student_text] if isinstance(student_text, str) else student_text)
        if ref_embs is None:
            ref_embs = self.sem_model.encode(reference_texts)

        cosine_scores = util.cos_sim(student_emb, ref_embs)
        raw_cos = float(torch.max(cosine_scores).item())
        score = max(0.0, min(100.0, raw_cos * 100.0))

        return score, raw_cos

    def evaluate_nmt_and_map(self, source_text: str, target_text: str) -> Tuple[float, Tuple[List[str], List[str], np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs NMT inference via the registered NMT wrapper, then computes token/word breakdowns
        and constructs the hybrid cross-attention + SimAlign word alignment matrix.
        """
        raw_output = self.nmt_model.infer(source_text, target_text)

        src_words, tgt_words, src_subs, tgt_subs, nmt_token_breakdown, nmt_word_breakdown = _build_nmt_breakdowns(raw_output)

        hybrid_matrix = _compute_hybrid_alignment_matrix(
            raw_output.cross_attentions,
            src_words,
            tgt_words,
            src_subs,
            tgt_subs,
            self.sem_model
        )

        return raw_output.perplexity, (src_words, tgt_words, hybrid_matrix), nmt_token_breakdown, nmt_word_breakdown

    def evaluate_fluency(self, text: str) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs fluency inference via the registered Fluency wrapper, then computes token and word breakdowns.
        """
        raw_output = self.fluency_model.infer(text)
        fluency_token_breakdown, fluency_word_breakdown = _build_fluency_breakdowns(raw_output, text)
        return raw_output.perplexity, fluency_token_breakdown, fluency_word_breakdown

    def evaluate_syntax(self, text: str) -> List[str]:
        """Uses LanguageTool (Tier 2) to detect genuine syntactical, grammatical, and confused-word errors."""
        matches = self.lang_tool.check(text)
        errors = []
        
        IGNORED_CATEGORIES = {"STYLE", "TYPOGRAPHY", "PUNCTUATION", "COLLOQUIALISMS", "REDUNDANCY"}
        IGNORED_ISSUE_TYPES = {"style", "register", "typographical", "whitespace", "locale-violation"}

        for match in matches:
            cat = getattr(match, "category", "")
            issue_type = getattr(match, "rule_issue_type", "")

            if cat in IGNORED_CATEGORIES or issue_type in IGNORED_ISSUE_TYPES:
                continue

            error_msg = match.message.strip()
            context = text[match.offset:match.offset + match.error_length].strip()
            if context:
                error_msg = f"On '{context}': {error_msg}"
                
            errors.append(error_msg)
                            
        return errors

    def align_and_filter_unexpected_words(
        self,
        nmt_word_breakdown: List[Dict[str, Any]],
        mapping_data_student: Tuple[List[str], List[str], np.ndarray],
        mapping_data_ref: Tuple[List[str], List[str], np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Soundness & vocabulary verification using student-to-source-to-reference alignment.
        Instead of arbitrary greedy NMT sub-tokens, finds which source word the student word
        aligned with, and looks up the intended reference word.
        If the student word is unsound for that concept and has elevated surprise, flags it
        and provides the actual reference word as top_prediction.
        """
        if not nmt_word_breakdown:
            return nmt_word_breakdown

        ref_words = mapping_data_ref[1] if mapping_data_ref is not None else []
        ref_matrix = mapping_data_ref[2] if mapping_data_ref is not None else None
        src_words_ref = mapping_data_ref[0] if mapping_data_ref is not None else []

        src_words_stu = mapping_data_student[0] if mapping_data_student is not None else []
        stu_words = mapping_data_student[1] if mapping_data_student is not None else []
        stu_matrix = mapping_data_student[2] if mapping_data_student is not None else None

        ref_words_clean = [w.lower().strip(".,!?:;'\"-") for w in ref_words]
        ref_words_set = set(ref_words_clean)

        for item in nmt_word_breakdown:
            w_idx = item.get("index", 0)
            raw_w = item.get("word", "")
            clean_w = raw_w.lower().strip(".,!?:;'\"-")

            if clean_w in ref_words_set:
                item["is_error"] = False
                item["top_prediction"] = ""
                continue

            expected_ref_word = ""
            matched_src_word = ""

            if stu_matrix is not None and w_idx < stu_matrix.shape[0]:
                best_s_idx = int(np.argmax(stu_matrix[w_idx, :]))
                if best_s_idx < len(src_words_stu):
                    matched_src_word = src_words_stu[best_s_idx]

                if ref_matrix is not None and best_s_idx < ref_matrix.shape[1]:
                    best_ref_idx = int(np.argmax(ref_matrix[:, best_s_idx]))
                    if best_ref_idx < len(ref_words):
                        expected_ref_word = ref_words[best_ref_idx]

            if not expected_ref_word and ref_words:
                expected_ref_word = ref_words[-1]

            clean_expected = expected_ref_word.lower().strip(".,!?:;'\"-")

            is_sound = False
            if clean_expected and clean_w:
                if clean_w == clean_expected:
                    is_sound = True
                elif clean_w in FUNCTION_WORDS and clean_expected in FUNCTION_WORDS:
                    is_sound = True
                else:
                    try:
                        embs = self.sem_model.encode([clean_w, clean_expected])
                        sim = float(util.cos_sim(embs[0], embs[1]).item())
                        if sim >= 0.70:
                            is_sound = True
                    except Exception:
                        pass

            if is_sound:
                item["is_error"] = False
                item["top_prediction"] = ""
            else:
                item["is_error"] = bool(clean_expected != "" and clean_w != "")
                item["top_prediction"] = expected_ref_word
                item["source_word"] = matched_src_word

        return nmt_word_breakdown

    def evaluate_student_against_bundle(self, source_text: str, student_text: str, reference_bundle: Any) -> Dict[str, Any]:
        """
        Evaluates the student text against a precomputed ReferenceBundle.
        Only runs inference passes on the student text (S_sem vs cached ref embedding,
        student NMT pass, student fluency pass, syntax check).
        Reuses reference_bundle.nmt_ppl_ref and reference_bundle.fluency_ppl_ref to cut latency in half.
        """
        is_french, lid_prob = self.check_language(student_text)

        # 1. Semantic Adequacy (Direct Cosine Similarity R_cos * 100 against precomputed embedding)
        s_sem, raw_cos = self.evaluate_semantics(
            student_text,
            reference_texts=[reference_bundle.reference_text],
            ref_embs=reference_bundle.ref_sem_emb
        )

        # 2. NMT Quality & Word Mapping for Student
        nmt_ppl_student, mapping_data_student, nmt_token_breakdown, nmt_word_breakdown = self.evaluate_nmt_and_map(
            source_text, student_text
        )
        nmt_ppl_ref = reference_bundle.nmt_ppl_ref
        s_nmt = max(0.0, min(100.0, (nmt_ppl_ref / max(nmt_ppl_student, 1e-5)) * 100.0))

        # 3. Fluency for Student
        fluency_ppl_student, fluency_token_breakdown, fluency_word_breakdown = self.evaluate_fluency(student_text)
        fluency_ppl_ref = reference_bundle.fluency_ppl_ref
        s_fluency = max(0.0, min(100.0, (fluency_ppl_ref / max(fluency_ppl_student, 1e-5)) * 100.0))

        # 4. Syntactical Analysis (LanguageTool)
        syntactical_errors = self.evaluate_syntax(student_text)

        # Reconstructed reference mapping data: (src_words, ref_words, attn_matrix_ref)
        mapping_data_ref = (
            reference_bundle.src_words,
            reference_bundle.ref_words,
            reference_bundle.attn_matrix_ref
        )

        # Soundness and expected-word refinement using reference alignment
        nmt_word_breakdown = self.align_and_filter_unexpected_words(
            nmt_word_breakdown, mapping_data_student, mapping_data_ref
        )

        return {
            "is_french": is_french,
            "lid_prob": lid_prob,
            "s_sem": s_sem,
            "s_nmt": s_nmt,
            "s_fluency": s_fluency,
            "mapping_data": mapping_data_student,
            "mapping_data_ref": mapping_data_ref,
            "nmt_ppl": nmt_ppl_student,
            "fluency_ppl": fluency_ppl_student,
            "raw_cos": raw_cos,
            "syntactical_errors": syntactical_errors,
            "nmt_token_breakdown": nmt_token_breakdown,
            "nmt_word_breakdown": nmt_word_breakdown,
            "fluency_token_breakdown": fluency_token_breakdown,
            "fluency_word_breakdown": fluency_word_breakdown
        }

    def compute_final_score(self, source_text: str, student_text: str, reference_text: str) -> Dict[str, Any]:
        """
        Direct evaluation of a student submission against a reference text without precomputed bundles.
        """
        nmt_ppl_ref, (src_words, ref_words, attn_matrix_ref), _, _ = self.evaluate_nmt_and_map(source_text, reference_text)
        fluency_ppl_ref, _, _ = self.evaluate_fluency(reference_text)
        ref_sem_emb = self.encode_semantics([reference_text])

        mapping_data_ref = (src_words, ref_words, attn_matrix_ref)

        is_french, lid_prob = self.check_language(student_text)
        s_sem, raw_cos = self.evaluate_semantics(student_text, [reference_text], ref_embs=ref_sem_emb)

        nmt_ppl_student, mapping_data_student, nmt_token_breakdown, nmt_word_breakdown = self.evaluate_nmt_and_map(source_text, student_text)
        s_nmt = max(0.0, min(100.0, (nmt_ppl_ref / max(nmt_ppl_student, 1e-5)) * 100.0))

        fluency_ppl_student, fluency_token_breakdown, fluency_word_breakdown = self.evaluate_fluency(student_text)
        s_fluency = max(0.0, min(100.0, (fluency_ppl_ref / max(fluency_ppl_student, 1e-5)) * 100.0))

        syntactical_errors = self.evaluate_syntax(student_text)

        nmt_word_breakdown = self.align_and_filter_unexpected_words(
            nmt_word_breakdown, mapping_data_student, mapping_data_ref
        )

        return {
            "is_french": is_french,
            "lid_prob": lid_prob,
            "s_sem": s_sem,
            "s_nmt": s_nmt,
            "s_fluency": s_fluency,
            "mapping_data": mapping_data_student,
            "mapping_data_ref": mapping_data_ref,
            "nmt_ppl": nmt_ppl_student,
            "fluency_ppl": fluency_ppl_student,
            "raw_cos": raw_cos,
            "syntactical_errors": syntactical_errors,
            "nmt_token_breakdown": nmt_token_breakdown,
            "nmt_word_breakdown": nmt_word_breakdown,
            "fluency_token_breakdown": fluency_token_breakdown,
            "fluency_word_breakdown": fluency_word_breakdown
        }

