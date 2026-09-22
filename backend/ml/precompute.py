import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.ml.engine import MLEngine

from backend.nlp.text_utils import get_aligned_targets, normalize_french_typography
from backend.nlp.vocab_engine import get_vocab_engine


@dataclass
class ReferenceBundle:
    """Precomputed artifacts for a source-reference sentence pair."""
    sentence_id: Optional[int]
    source_text: str
    reference_text: str
    nmt_ppl_ref: float
    fluency_ppl_ref: float
    ref_sem_emb: Any  # torch.Tensor or numpy.ndarray
    src_words: List[str]
    ref_words: List[str]
    attn_matrix_ref: np.ndarray
    aligned_ref_map: Dict[int, List[int]]
    first_time_src_indices: List[int] = field(default_factory=list)
    first_time_tgt_indices: List[int] = field(default_factory=list)
    first_time_lemmas: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class EagerPrecomputeManager:
    """
    Manages dual-threaded ML execution during exercises:
    - Thread 1 (Eager Worker): Runs static reference sentence passes in the background
      immediately when an exercise batch is fetched (NMT reference PPL, word alignments,
      fluency reference PPL, reference semantic embedding).
    - Thread 2 (Evaluation Worker): Evaluates student submissions one-by-one, reusing
      precomputed ReferenceBundles to avoid running duplicate passes on the reference text.
    """

    def __init__(self, get_engine_fn, data_manager: Optional[Any] = None, max_cache_size: int = 250):
        self.get_engine = get_engine_fn
        self.data_manager = data_manager
        self.max_cache_size = max_cache_size

        # In-memory session cache: (source_text, reference_text) -> ReferenceBundle
        self._cache: Dict[Tuple[str, str], ReferenceBundle] = {}
        self._cache_lock = threading.Lock()

        # Sentence ID alias lookup: sentence_id -> (source_text, reference_text)
        self._id_to_key: Dict[int, Tuple[str, str]] = {}

        # In-flight events: key -> threading.Event (allows requests to await active precomputations)
        self._inflight_events: Dict[Tuple[str, str], threading.Event] = {}

        # Event-driven worker pool with max_workers=1 for strictly sequential processing
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EagerMLWorker")
        self._is_shutdown = False

        # Thread 2: Evaluation lock to serialize active student sentence evaluations
        self._eval_lock = threading.Lock()

    def shutdown(self, wait: bool = False) -> None:
        """Shuts down the background executor."""
        with self._cache_lock:
            self._is_shutdown = True
        self._executor.shutdown(wait=wait)

    def _normalize_key(self, source_text: str, reference_text: str) -> Tuple[str, str]:
        return (source_text.strip(), reference_text.strip())

    def enqueue_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Enqueues an entire exercise batch for eager background reference computation.
        Invoked by GET /api/exercises/next / batch.
        """
        if not batch:
            return

        for item in batch:
            src = item.get("es") or item.get("source_text") or item.get("en") or ""
            ref = item.get("fr") or item.get("reference_text") or ""
            s_id = item.get("id") or item.get("sentence_id")
            first_time_lexemas = item.get("first_time_lexemas") or item.get("new_lexemas") or []

            if not src or not ref:
                continue

            key = self._normalize_key(src, ref)
            with self._cache_lock:
                if s_id:
                    self._id_to_key[s_id] = key
                if key in self._cache:
                    continue
                if key not in self._inflight_events:
                    if self._is_shutdown:
                        continue
                    event = threading.Event()
                    self._inflight_events[key] = event
                    try:
                        self._executor.submit(self._process_single_task, {
                            "key": key,
                            "sentence_id": s_id,
                            "source_text": src,
                            "reference_text": ref,
                            "first_time_lexemas": first_time_lexemas
                        })
                    except RuntimeError:
                        self._inflight_events.pop(key, None)
                        event.set()

    def _process_single_task(self, task: Dict[str, Any]) -> None:
        """Processes reference passes for an enqueued sentence task."""
        key = task["key"]
        # Skip if already computed by an eager or on-demand pass
        with self._cache_lock:
            if key in self._cache:
                event = self._inflight_events.pop(key, None)
                if event:
                    event.set()
                return

        try:
            bundle = self._compute_reference_bundle(
                sentence_id=task.get("sentence_id"),
                source_text=task["source_text"],
                reference_text=task["reference_text"],
                first_time_lemmas=task.get("first_time_lexemas") or []
            )
            with self._cache_lock:
                # Manage cache size (evict oldest if full)
                if len(self._cache) >= self.max_cache_size:
                    oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
                    del self._cache[oldest_key]

                self._cache[key] = bundle
                event = self._inflight_events.pop(key, None)
                if event:
                    event.set()
        except Exception as ex:
            if not self._is_shutdown and "interpreter shutdown" not in str(ex):
                print(f"[EagerMLWorker] Error precomputing reference bundle for {key}: {ex}")
            with self._cache_lock:
                event = self._inflight_events.pop(key, None)
                if event:
                    event.set()

    def _compute_reference_bundle(
        self,
        sentence_id: Optional[int],
        source_text: str,
        reference_text: str,
        first_time_lemmas: List[str]
    ) -> ReferenceBundle:
        """Computes all source-reference passes using MLEngine."""
        engine = self.get_engine()

        # Symmetrically normalize reference text typography and punctuation spacing
        norm_ref = normalize_french_typography(reference_text)

        # 1. NMT perplexity and cross-attention alignment matrix on reference sentence
        nmt_ppl_ref, (src_words, ref_words, attn_matrix_ref), _, _ = engine.evaluate_nmt_and_map(
            source_text, norm_ref
        )

        num_tgt = len(ref_words)
        aligned_ref_map = {
            i: get_aligned_targets([i], attn_matrix_ref, num_tgt)
            for i in range(len(src_words))
        }

        # 2. Fluency perplexity on reference sentence
        fluency_ppl_ref, _, _ = engine.evaluate_fluency(norm_ref)

        # 3. Semantic embedding on reference sentence
        if hasattr(engine, "encode_semantics"):
            ref_sem_emb = engine.encode_semantics([norm_ref])
        else:
            ref_sem_emb = engine.sem_model.encode([norm_ref], convert_to_tensor=True)

        # 4. First-time lemmas mapping (if database is present)
        if not first_time_lemmas and sentence_id and self.data_manager:
            try:
                sent_lex = self.data_manager.get_sentence_lexemas(sentence_id)
                known_ids = self.data_manager.get_user_known_lexema_ids("default_user")
                first_time_lemmas = [
                    l["lemma"] for l in sent_lex
                    if not l.get("is_base") and l["id"] not in known_ids
                ]
            except Exception:
                pass

        first_time_src_indices = set()
        first_time_tgt_indices = set()

        if first_time_lemmas:
            pairs = []
            if get_vocab_engine is not None:
                try:
                    vocab_engine = get_vocab_engine()
                    pairs = vocab_engine.extract_sentence_token_pairs(reference_text, first_time_lemmas)
                except Exception:
                    pairs = []

            matched_words = {p['word'].lower().strip() for p in pairs if p.get('word')}
            clean_lemmas = {l.lower().strip() for l in first_time_lemmas if l}

            for t_idx, t_word in enumerate(ref_words):
                clean_t = re.sub(r'^[^\w\u00C0-\u017F]+|[^\w\u00C0-\u017F]+$', '', t_word.lower())
                clean_t_no_prefix = re.sub(r"^[ldjcmnstLDJCMNST]['’]", "", clean_t)
                if not clean_t:
                    continue
                is_match = (
                    clean_t in matched_words or
                    clean_t_no_prefix in matched_words or
                    clean_t in clean_lemmas or
                    clean_t_no_prefix in clean_lemmas
                )
                if not is_match and not matched_words:
                    for lem in clean_lemmas:
                        if clean_t == lem or clean_t_no_prefix == lem:
                            is_match = True
                            break

                if is_match:
                    first_time_tgt_indices.add(t_idx)
                    if attn_matrix_ref is not None and attn_matrix_ref.size > 0 and t_idx < attn_matrix_ref.shape[0]:
                        best_s = int(np.argmax(attn_matrix_ref[t_idx, :]))
                        first_time_src_indices.add(best_s)
                    for s_idx, t_list in aligned_ref_map.items():
                        if t_idx in t_list:
                            first_time_src_indices.add(int(s_idx))

        return ReferenceBundle(
            sentence_id=sentence_id,
            source_text=source_text,
            reference_text=reference_text,
            nmt_ppl_ref=nmt_ppl_ref,
            fluency_ppl_ref=fluency_ppl_ref,
            ref_sem_emb=ref_sem_emb,
            src_words=src_words,
            ref_words=ref_words,
            attn_matrix_ref=attn_matrix_ref,
            aligned_ref_map=aligned_ref_map,
            first_time_src_indices=sorted(list(first_time_src_indices)),
            first_time_tgt_indices=sorted(list(first_time_tgt_indices)),
            first_time_lemmas=sorted(list(set(first_time_lemmas)))
        )

    def get_or_compute_bundle(
        self,
        source_text: str,
        reference_text: str,
        sentence_id: Optional[int] = None,
        first_time_lemmas: Optional[List[str]] = None,
        timeout: float = 12.0
    ) -> ReferenceBundle:
        """
        Retrieves a ReferenceBundle from memory.
        If Thread 1 is currently processing it, awaits completion up to timeout.
        If not in cache or queue, computes synchronously.
        """
        key = self._normalize_key(source_text, reference_text)
        event = None

        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
            if key in self._inflight_events:
                event = self._inflight_events[key]

        if event is not None:
            # Await background completion from Thread 1
            completed = event.wait(timeout=timeout)
            with self._cache_lock:
                if key in self._cache:
                    return self._cache[key]

        # If still not ready, compute on demand
        bundle = self._compute_reference_bundle(
            sentence_id=sentence_id,
            source_text=source_text,
            reference_text=reference_text,
            first_time_lemmas=first_time_lemmas or []
        )
        with self._cache_lock:
            self._cache[key] = bundle
            if sentence_id:
                self._id_to_key[sentence_id] = key
        return bundle

    def get_hint(
        self,
        source_text: str,
        reference_text: str,
        selected_indices: Optional[List[int]] = None,
        sentence_id: Optional[int] = None,
        first_time_lemmas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Returns hint mapping data. If already precomputed by Thread 1,
        returns in <1ms without running any neural models.
        """
        bundle = self.get_or_compute_bundle(
            source_text=source_text,
            reference_text=reference_text,
            sentence_id=sentence_id,
            first_time_lemmas=first_time_lemmas
        )

        selected_indices = selected_indices or []
        target_indices = set()
        for s_idx in selected_indices:
            for t_idx in bundle.aligned_ref_map.get(s_idx, []):
                target_indices.add(t_idx)

        mapped_target_words = [
            bundle.ref_words[t] for t in sorted(list(target_indices))
            if t < len(bundle.ref_words)
        ]
        selected_source_words = [
            bundle.src_words[s] for s in sorted(selected_indices)
            if s < len(bundle.src_words)
        ]

        return {
            "src_words": bundle.src_words,
            "tgt_words": bundle.ref_words,
            "aligned_targets_map": bundle.aligned_ref_map,
            "first_time_src_indices": bundle.first_time_src_indices,
            "first_time_tgt_indices": bundle.first_time_tgt_indices,
            "first_time_lemmas": bundle.first_time_lemmas,
            "selected_source_words": selected_source_words,
            "mapped_target_words": mapped_target_words,
            "hint_translation": " ".join(mapped_target_words) if mapped_target_words else ""
        }

    def evaluate_student(
        self,
        source_text: str,
        student_text: str,
        reference_text: str,
        sentence_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Thread 2 Evaluation Worker:
        Evaluates a user sentence one-by-one under self._eval_lock.
        Reuses the precomputed ReferenceBundle so NO forward passes are run on the reference text!
        """
        with self._eval_lock:
            # 1. Fetch precomputed reference metrics
            bundle = self.get_or_compute_bundle(
                source_text=source_text,
                reference_text=reference_text,
                sentence_id=sentence_id
            )

            engine = self.get_engine()

            # 2. Evaluate student text against bundle
            if hasattr(engine, "evaluate_student_against_bundle"):
                result = engine.evaluate_student_against_bundle(
                    source_text=source_text,
                    student_text=student_text,
                    reference_bundle=bundle
                )
            else:
                # Fallback to compute_final_score if method not yet bound
                result = engine.compute_final_score(
                    source_text=source_text,
                    student_text=student_text,
                    reference_text=reference_text
                )

            return result
