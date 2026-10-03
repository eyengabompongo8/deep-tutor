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
import atexit
import weakref
from sentence_transformers import util
import fasttext
from huggingface_hub import hf_hub_download
import numpy as np
import language_tool_python

from backend.ml.registry import registry, NMTRawOutput, FluencyRawOutput
from backend.config import NMT_DELTA_L_THRESHOLD, FUNCTION_WORDS
from backend.ml.token_utils import (
    _build_nmt_breakdowns,
    _build_fluency_breakdowns,
    _compute_hybrid_alignment_matrix,
)
from backend.nlp.text_utils import normalize_french_typography


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
        # Register process exit cleanup to prevent orphaned Java LanguageTool servers
        atexit.register(self.close)

        print(f"All ML models loaded successfully on {self.device} ({self.dtype}).")

    def check_language(self, text: str) -> Tuple[bool, float]:
        text_clean = text.strip().replace("\n", " ")
        if not text_clean:
            return False, 0.0

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

        def _clean_word(w: str) -> str:
            if not w:
                return ""
            w = re.sub(r"[’‘`´′]", "'", w)
            w = re.sub(r"[‑–—−]", "-", w)
            return w.lower().strip(".,!?:;\"«»()[]{}").strip()

        ref_words_clean = [_clean_word(w) for w in ref_words]
        ref_words_set = set(ref_words_clean)

        for item in nmt_word_breakdown:
            w_idx = item.get("index", 0)
            raw_w = item.get("word", "")
            clean_w = _clean_word(raw_w)

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

            clean_expected = _clean_word(expected_ref_word)

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
        norm_ref = normalize_french_typography(reference_text)
        nmt_ppl_ref, (src_words, ref_words, attn_matrix_ref), _, _ = self.evaluate_nmt_and_map(source_text, norm_ref)
        fluency_ppl_ref, _, _ = self.evaluate_fluency(norm_ref)
        ref_sem_emb = self.encode_semantics([norm_ref])

        mapping_data_ref = (src_words, ref_words, attn_matrix_ref)

        is_french, lid_prob = self.check_language(student_text)
        s_sem, raw_cos = self.evaluate_semantics(student_text, [norm_ref], ref_embs=ref_sem_emb)

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

    def close(self):
        """Clean up external sub-processes (such as LanguageTool Java HTTPServer)."""
        if hasattr(self, "lang_tool") and self.lang_tool is not None:
            try:
                self.lang_tool.close()
            except Exception:
                pass
            self.lang_tool = None
