import os
import hashlib
import threading
import asyncio
from typing import Optional, List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_CACHE_DIR = os.path.join(ROOT_DIR, "data", "audio_cache")

# Preferred high-fidelity Parisian French voices
DEFAULT_VOICE = "fr-FR-VivienneMultilingualNeural"

_tts_instance: Optional["TTSEngine"] = None
_instance_lock = threading.Lock()


class TTSEngine:
    """
    High-fidelity French Text-to-Speech engine.
    Primary: Microsoft Edge Neural TTS (edge-tts) for studio-quality 24kHz natural prosody.
    Fallback: Local Hugging Face VITS model (facebook/mms-tts-fra) for offline resilience.
    """

    def __init__(
        self,
        data_manager: Optional[Any] = None,
        cache_dir: str = AUDIO_CACHE_DIR,
        default_voice: str = DEFAULT_VOICE,
    ):
        self.data_manager = data_manager
        self.cache_dir = cache_dir
        self.default_voice = default_voice
        os.makedirs(self.cache_dir, exist_ok=True)

        # Fallback model state (lazy loaded only if needed)
        self.fallback_model_id = "facebook/mms-tts-fra"
        self._fallback_tokenizer = None
        self._fallback_model = None
        self._fallback_device = "cpu"
        self._lock = threading.Lock()

    def _synth_edge_tts(self, text: str, output_path: str, voice: str) -> None:
        """Synthesizes speech using edge-tts with event-loop safety across worker threads."""
        import edge_tts

        async def _run():
            communicate = edge_tts.Communicate(text, voice=voice)
            await communicate.save(output_path)

        asyncio.run(_run())

    def _ensure_fallback_loaded(self):
        """Lazy-loads local MMS-TTS model if offline."""
        if self._fallback_model is not None:
            return

        with self._lock:
            if self._fallback_model is not None:
                return
            import torch
            from transformers import VitsModel, AutoTokenizer

            if torch.cuda.is_available():
                try:
                    major, minor = torch.cuda.get_device_capability()
                    if (major, minor) >= (7, 5):
                        self._fallback_device = "cuda"
                except Exception:
                    self._fallback_device = "cpu"

            print(f"[TTSEngine] Loading fallback model {self.fallback_model_id} on {self._fallback_device}...")
            self._fallback_tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_id)
            self._fallback_model = VitsModel.from_pretrained(self.fallback_model_id).to(self._fallback_device)
            self._fallback_model.eval()

    def _synth_local_fallback(self, text: str, output_path: str) -> None:
        """Synthesizes speech using the local MMS PyTorch model."""
        import torch
        import scipy.io.wavfile as wavfile

        self._ensure_fallback_loaded()
        with self._lock:
            inputs = self._fallback_tokenizer(text, return_tensors="pt").to(self._fallback_device)
            with torch.no_grad():
                output = self._fallback_model(**inputs).waveform

            audio_data = output.squeeze().cpu().numpy()
            sample_rate = int(self._fallback_model.config.sampling_rate)
            wavfile.write(output_path, rate=sample_rate, data=audio_data)

    def synthesize(
        self,
        text: str,
        sentence_id: Optional[int] = None,
        voice: Optional[str] = None,
    ) -> str:
        """
        Synthesizes speech for the provided French text.
        Returns the absolute file path to the generated or cached audio file (.mp3 or .wav).
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        active_voice = voice or self.default_voice
        # Voice-keyed hash prevents collision with old MMS wav files
        cache_key = f"{active_voice}:{clean_text}"
        text_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
        mp3_path = os.path.join(self.cache_dir, f"{text_hash}.mp3")

        # 1. Check SQLite metadata cache
        if self.data_manager is not None:
            try:
                cached = self.data_manager.get_audio_cache(text_hash)
                if cached and os.path.exists(cached.get("file_path", "")):
                    return cached["file_path"]
            except Exception as e:
                print(f"[TTSEngine] Cache check warning: {e}")

        # 2. Check disk cache
        if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            if self.data_manager is not None:
                try:
                    self.data_manager.save_audio_cache(
                        text_hash=text_hash,
                        text=clean_text,
                        file_path=mp3_path,
                        sentence_id=sentence_id
                    )
                except Exception:
                    pass
            return mp3_path

        # 3. Primary Synthesis (edge-tts)
        try:
            self._synth_edge_tts(clean_text, mp3_path, active_voice)
            final_path = mp3_path
        except Exception as edge_err:
            print(f"[TTSEngine] edge-tts unavailable ({edge_err}). Falling back to local MMS model...")
            wav_path = os.path.join(self.cache_dir, f"{text_hash}_fallback.wav")
            if not (os.path.exists(wav_path) and os.path.getsize(wav_path) > 0):
                self._synth_local_fallback(clean_text, wav_path)
            final_path = wav_path

        # 4. Store in SQLite cache
        if self.data_manager is not None:
            try:
                self.data_manager.save_audio_cache(
                    text_hash=text_hash,
                    text=clean_text,
                    file_path=final_path,
                    sentence_id=sentence_id
                )
            except Exception as e:
                print(f"[TTSEngine] Cache save warning: {e}")

        return final_path

    def prewarm_batch(self, sentences: List[Dict[str, Any]], voice: Optional[str] = None):
        """Pre-generates audio for upcoming sentences in the background."""
        def _prewarm_worker():
            for item in sentences:
                fr_text = item.get("fr", "").strip()
                s_id = item.get("id")
                if not fr_text:
                    continue
                try:
                    self.synthesize(fr_text, sentence_id=s_id, voice=voice)
                except Exception as e:
                    print(f"[TTSEngine] Prewarming failed for sentence {s_id}: {e}")

        thread = threading.Thread(target=_prewarm_worker, daemon=True)
        thread.start()


def get_tts_engine(data_manager: Optional[Any] = None) -> TTSEngine:
    global _tts_instance
    if _tts_instance is None:
        with _instance_lock:
            if _tts_instance is None:
                _tts_instance = TTSEngine(data_manager=data_manager)
    elif data_manager is not None and _tts_instance.data_manager is None:
        _tts_instance.data_manager = data_manager
    return _tts_instance
