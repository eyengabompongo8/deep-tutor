import os
import sqlite3
import pytest
from backend.db.snapshot import dump_snapshot, restore_snapshot, DEFAULT_SNAPSHOT_PATH


def test_snapshot_restore_lifecycle(tmp_path):
    # Verify snapshot file exists
    assert os.path.exists(DEFAULT_SNAPSHOT_PATH), "Default snapshot does not exist"

    test_db = str(tmp_path / "restored.db")

    # Restore snapshot into test DB
    assert restore_snapshot(snapshot_path=DEFAULT_SNAPSHOT_PATH, dest_db=test_db, backup_existing=False)
    assert os.path.exists(test_db)

    # Check contents
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()

    # Linguistic tables must be fully populated
    sentences = cursor.execute("SELECT COUNT(*) FROM sentences;").fetchone()[0]
    lemmas = cursor.execute("SELECT COUNT(*) FROM lemmas;").fetchone()[0]
    words = cursor.execute("SELECT COUNT(*) FROM words;").fetchone()[0]
    sentence_lemmas = cursor.execute("SELECT COUNT(*) FROM sentence_lemmas;").fetchone()[0]
    sentence_words = cursor.execute("SELECT COUNT(*) FROM sentence_words;").fetchone()[0]

    assert sentences > 30000
    assert lemmas > 65000
    assert words > 180000
    assert sentence_lemmas > 250000
    assert sentence_words > 250000

    # User tables must be completely clean (only default_user seeded)
    users = cursor.execute("SELECT id, name FROM users;").fetchall()
    assert users == [("default_user", "Default User")]

    history_cnt = cursor.execute("SELECT COUNT(*) FROM exercise_history;").fetchone()[0]
    assert history_cnt == 0

    audio_cnt = cursor.execute("SELECT COUNT(*) FROM audio_cache;").fetchone()[0]
    assert audio_cnt == 0

    user_lemmas = cursor.execute("SELECT COUNT(*) FROM user_lemmas;").fetchone()[0]
    assert user_lemmas == 0

    conn.close()
