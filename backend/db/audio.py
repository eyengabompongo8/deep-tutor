import os
from typing import Dict, Any, Optional

class AudioMixin:
    def get_audio_cache(self, text_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached audio record by text_hash if it exists on disk."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM audio_cache WHERE text_hash = ?;", (text_hash,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if os.path.exists(d.get("file_path", "")):
                    return d
            return None

    def save_audio_cache(self, text_hash: str, text: str, file_path: str, sentence_id: Optional[int] = None):
        """Saves a synthesized audio file mapping into the audio_cache table."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO audio_cache (sentence_id, text_hash, text, file_path)
                VALUES (?, ?, ?, ?);
            """, (sentence_id, text_hash, text, file_path))
