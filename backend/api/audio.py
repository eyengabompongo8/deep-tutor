import os
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.api import deps

logger = logging.getLogger("backend.api.audio")
router = APIRouter()


@router.get("/api/audio/synthesize")
def synthesize_audio(text: str, sentence_id: Optional[int] = None, voice: Optional[str] = None):
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")
    if deps.tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS Engine is not initialized.")
    try:
        audio_path = deps.tts_engine.synthesize(clean_text, sentence_id=sentence_id, voice=voice)
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Failed to generate audio file.")

        media_type = "audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav"
        return FileResponse(
            audio_path,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Audio synthesis error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")
