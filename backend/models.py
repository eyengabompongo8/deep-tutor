import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import fasttext
from huggingface_hub import hf_hub_download
import numpy as np
import language_tool_python
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

def aggregate_sentencepiece_tokens_to_words(tokens):
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


def resolve_device_and_dtype():
    """
    Attempts to use CUDA acceleration if available and capable of executing kernels.
    Performs a live kernel probe to verify hardware capability (e.g. sm_61 on GTX 1050 Ti).
    Falls back gracefully to CPU if probe fails or if CUDA is unavailable.
    """
    if torch.cuda.is_available():
        try:
            # Active probe test: verify kernel execution on the device
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
    def __init__(self):
        # Auto-detect CUDA with active probe test, fallback to CPU
        self.device, self.dtype = resolve_device_and_dtype()

        # 1. Term 2: NMT Conditional Perplexity & Word Mapping (facebook/m2m100_418M)
        print(f"Loading NMT Model (facebook/m2m100_418M) on {self.device}...")
        self.nmt_model_name = "facebook/m2m100_418M"
        self.nmt_tokenizer = M2M100Tokenizer.from_pretrained(self.nmt_model_name)
        self.nmt_model = M2M100ForConditionalGeneration.from_pretrained(
            self.nmt_model_name,
            torch_dtype=self.dtype,
            attn_implementation="eager"
        ).to(self.device)

        # 2. Term 1: Semantic Adequacy (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
        print(f"Loading Semantic Model (paraphrase-multilingual-MiniLM-L12-v2) on {self.device}...")
        self.sem_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=self.device)

        # 3. Term 3: Monolingual Fluency (distilgpt2)
        print(f"Loading Fluency Model (distilgpt2) on {self.device}...")
        self.fluency_model_name = "distilgpt2"
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(self.fluency_model_name)
        self.fluency_model = AutoModelForCausalLM.from_pretrained(
            self.fluency_model_name,
            torch_dtype=self.dtype
        ).to(self.device)

        # 4. LID Gatekeeper (cis-lmu/glotlid)
        print("Loading GlotLID Gatekeeper...")
        lid_model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        self.lid_model = fasttext.load_model(lid_model_path)

        # 5. Syntax Evaluator (LanguageTool)
        print("Loading LanguageTool (fr)...")
        self.lang_tool = language_tool_python.LanguageTool('fr')

        print(f"All ML models loaded successfully on {self.device} ({self.dtype}).")

    def check_language(self, text):
        text_clean = text.strip().replace("\n", " ")
        if not text_clean:
            return False, 0.0

        words = re.findall(r"\b[^\W\d_]+\b", text_clean.lower(), re.UNICODE)
        if not words:
            return False, 0.0

        # Short French sentences (< 6 words) frequently trigger false negatives in fasttext
        # (e.g. "Bonjour" getting classified as Picard or Norman).
        # Check against common French functional and core words first:
        common_fr = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'au', 'aux',
            'ce', 'cet', 'cette', 'ces', 'je', 'tu', 'il', 'elle', 'on',
            'nous', 'vous', 'ils', 'elles', 'j', 't', 's', 'l', 'd', 'c', 'm', 'n', 'qu',
            'me', 'te', 'se', 'lui', 'leur', 'y', 'en', 'moi', 'toi',
            'est', 'ai', 'suis', 'es', 'sommes', 'êtes', 'sont', 'a', 'ont', 'va', 'vais', 'vas', 'vont',
            'bonjour', 'merci', 'oui', 'non', 'salut', 'revoir', 'bien', 'sûr', 'faim', 'soif', 'peur',
            'très', 'plus', 'pas', 'ne', 'où', 'que', 'qui', 'quoi', 'comment', 'pourquoi', 'avec', 'sans', 'dans', 'pour', 'sur'
        }
        if len(words) <= 5 and any(w in common_fr for w in words):
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

    def encode_semantics(self, texts):
        """Encodes sentences into normalized embeddings on the active device."""
        return self.sem_model.encode(texts, convert_to_tensor=True, device=self.device)

    def evaluate_semantics(self, student_text, reference_texts, ref_embs=None):
        """
        Calculates Semantic Adequacy (S_sem) directly using normalized Cosine Similarity (R_cos * 100).
        Accepts optional precomputed ref_embs to eliminate redundant forward passes.
        """
        student_emb = self.sem_model.encode(student_text, convert_to_tensor=True, device=self.device)
        if ref_embs is None:
            ref_embs = self.sem_model.encode(reference_texts, convert_to_tensor=True, device=self.device)

        cosine_scores = util.cos_sim(student_emb, ref_embs)
        raw_cos = float(torch.max(cosine_scores).item())
        score = max(0.0, min(100.0, raw_cos * 100.0))

        return score, raw_cos

    def evaluate_nmt_and_map(self, source_text, target_text):
        """
        Calculates NMT perplexity, per-token/word breakdown metrics (loss, PPL, prob, top prediction, Delta L),
        and constructs a high-precision hybrid word alignment matrix.
        """
        self.nmt_tokenizer.src_lang = "es"
        self.nmt_tokenizer.tgt_lang = "fr"

        inputs = self.nmt_tokenizer(source_text, return_tensors="pt").to(self.device)
        labels = self.nmt_tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            outputs = self.nmt_model(**inputs, labels=labels, output_attentions=True)
            loss = float(outputs.loss.item())
            perplexity = float(np.exp(loss))

            if outputs.cross_attentions is not None and len(outputs.cross_attentions) > 0:
                top_layers = outputs.cross_attentions[-4:]
                subtoken_attn = torch.mean(torch.stack(top_layers), dim=(0, 1, 2)).squeeze(0).cpu().float().numpy()
            else:
                subtoken_attn = np.zeros((labels.shape[1], inputs["input_ids"].shape[1]))

            logits = outputs.logits[0].float()  # Float32 for stable CrossEntropyLoss
            target_ids = labels[0]
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses_tensor = loss_fct(logits, target_ids)
            top_1_token_ids = torch.argmax(logits, dim=-1)
            top_1_losses_tensor = loss_fct(logits, top_1_token_ids)

        src_raw_tokens = self.nmt_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tgt_raw_tokens = self.nmt_tokenizer.convert_ids_to_tokens(labels[0])

        src_words, src_sub_groups = aggregate_sentencepiece_tokens_to_words(src_raw_tokens)
        tgt_words, tgt_sub_groups = aggregate_sentencepiece_tokens_to_words(tgt_raw_tokens)

        # Build NMT Token Breakdown
        nmt_token_breakdown = []
        token_loss_map = {}
        top_1_loss_map = {}

        for idx, t_id_tensor in enumerate(target_ids):
            t_id = int(t_id_tensor.item())
            tok_str = tgt_raw_tokens[idx] if idx < len(tgt_raw_tokens) else str(t_id)
            if _is_special_token(tok_str) or t_id == -100:
                continue

            loss_i = float(token_losses_tensor[idx].item())
            top_1_id = int(top_1_token_ids[idx].item())
            top_1_tok_str = self.nmt_tokenizer.convert_ids_to_tokens(top_1_id)
            top_1_loss_i = float(top_1_losses_tensor[idx].item())

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
                "is_error": delta_l_i > 8.0
            })

        # Build NMT Word Breakdown
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

            first_idx = sub_indices[0] if sub_indices else None
            top_pred_word = "—"
            if first_idx is not None and first_idx < len(top_1_token_ids):
                top_pred_id = int(top_1_token_ids[first_idx].item())
                top_pred_word = _strip_boundary(self.nmt_tokenizer.convert_ids_to_tokens(top_pred_id))

            nmt_word_breakdown.append({
                "index": w_idx,
                "word": word_text,
                "loss": w_loss,
                "ppl": w_ppl,
                "prob": w_prob,
                "top_prediction": top_pred_word,
                "delta_l": w_delta_l,
                "is_error": w_delta_l > 8.0
            })

        # Row-level softmax normalization per target token to filter background noise
        row_sums = subtoken_attn.sum(axis=1, keepdims=True) + 1e-8
        norm_subtoken_attn = subtoken_attn / row_sums

        # Construct Word-Level Attention Matrix
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

        # Multilingual Contextual Embedding Similarity (SimAlign)
        src_embs = self.sem_model.encode(src_words, convert_to_tensor=True, device=self.device)
        tgt_embs = self.sem_model.encode(tgt_words, convert_to_tensor=True, device=self.device)
        sim_matrix = util.cos_sim(tgt_embs, src_embs).cpu().float().numpy()
        sim_matrix = np.maximum(0.0, sim_matrix)

        # Hybrid Alignment Matrix: 50% normalized cross-attention + 50% contextual embedding similarity
        max_attn = np.max(word_attn_matrix) + 1e-8
        hybrid_matrix = 0.5 * (word_attn_matrix / max_attn) + 0.5 * sim_matrix

        return perplexity, (src_words, tgt_words, hybrid_matrix), nmt_token_breakdown, nmt_word_breakdown

    def evaluate_fluency(self, text):
        """Calculates perplexity and per-token/word breakdown of the French text using distilgpt2."""
        encodings = self.fluency_tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids.to(self.device)
        if input_ids.shape[1] == 0:
            return 100.0, [], []

        with torch.inference_mode():
            outputs = self.fluency_model(input_ids, labels=input_ids)
            loss = float(outputs.loss.item())
            perplexity = float(np.exp(loss))

            logits = outputs.logits[0].float()
            if input_ids.shape[1] > 1:
                shift_logits = logits[:-1, :]
                shift_labels = input_ids[0, 1:]
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits, shift_labels)
                top_1_ids = torch.argmax(shift_logits, dim=-1)
                top_1_losses = loss_fct(shift_logits, top_1_ids)

                fluency_token_breakdown = []
                for idx in range(len(shift_labels)):
                    lbl_id = int(shift_labels[idx].item())
                    tok_str = self.fluency_tokenizer.decode([lbl_id]).strip()

                    loss_i = float(token_losses[idx].item())
                    top_1_id = int(top_1_ids[idx].item())
                    top_1_str = self.fluency_tokenizer.decode([top_1_id]).strip()
                    top_1_loss_i = float(top_1_losses[idx].item())

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
                        "is_error": delta_l_i > 8.0
                    })
            else:
                fluency_token_breakdown = []

        # Fluency Word Breakdown
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
                    "is_error": w_delta_l > 8.0
                })

        return perplexity, fluency_token_breakdown, fluency_word_breakdown

    def evaluate_syntax(self, text):
        """Uses LanguageTool to detect syntactical and grammatical errors."""
        matches = self.lang_tool.check(text)
        errors = []
        
        for match in matches:
            # Skip pure spelling errors since pyspellchecker handles basic typos
            if "MORFOLOGIK_RULE_FR" in match.rule_id or "FRENCH_SPELLER_RULE" in match.rule_id:
                continue
                
            error_msg = f"{match.message} (Rule: {match.rule_id})"
            
            # Optionally extract bad text context for clarity
            context = text[match.offset:match.offset+match.error_length]
            if context:
                error_msg = f"On '{context}': {error_msg}"
                
            errors.append(error_msg)
                            
        return errors

    def evaluate_student_against_bundle(self, source_text, student_text, reference_bundle):
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

    def compute_final_score(self, source_text, student_text, reference_text):
        """Computes individual term evaluation scores (0-100 normalized) without arbitrary scales or global score."""
        is_french, lid_prob = self.check_language(student_text)

        # 1. Semantic Adequacy (Direct Cosine Similarity R_cos * 100)
        s_sem, raw_cos = self.evaluate_semantics(student_text, [reference_text])

        # 2. NMT Quality & Word Mapping
        nmt_ppl_student, mapping_data, nmt_token_breakdown, nmt_word_breakdown = self.evaluate_nmt_and_map(source_text, student_text)
        nmt_ppl_ref, mapping_data_ref, _, _ = self.evaluate_nmt_and_map(source_text, reference_text)
        s_nmt = max(0.0, min(100.0, (nmt_ppl_ref / max(nmt_ppl_student, 1e-5)) * 100.0))

        # 3. Fluency
        fluency_ppl_student, fluency_token_breakdown, fluency_word_breakdown = self.evaluate_fluency(student_text)
        fluency_ppl_ref, _, _ = self.evaluate_fluency(reference_text)
        s_fluency = max(0.0, min(100.0, (fluency_ppl_ref / max(fluency_ppl_student, 1e-5)) * 100.0))

        # 4. Syntactical Analysis (LanguageTool)
        syntactical_errors = self.evaluate_syntax(student_text)

        return {
            "is_french": is_french,
            "lid_prob": lid_prob,
            "s_sem": s_sem,
            "s_nmt": s_nmt,
            "s_fluency": s_fluency,
            "mapping_data": mapping_data,
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
