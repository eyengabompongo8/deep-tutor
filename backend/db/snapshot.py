import os
import sys
import gzip
import shutil
import sqlite3
import argparse
from typing import Optional

from backend.db.connection import DB_PATH, ROOT_DIR

DEFAULT_SNAPSHOT_DIR = os.path.join(ROOT_DIR, "data", "snapshots")
DEFAULT_SNAPSHOT_PATH = os.path.join(DEFAULT_SNAPSHOT_DIR, "default_snapshot.db.gz")

USER_TABLES = [
    "user_lemmas",
    "user_words",
    "user_modes",
    "exercise_history",
    "audio_cache",
    "user_settings",
    "users",
]


def dump_snapshot(
    src_db: Optional[str] = None,
    out_snapshot: Optional[str] = None,
    compress_level: int = 9,
) -> bool:
    src_db = src_db or (os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH)
    out_snapshot = out_snapshot or DEFAULT_SNAPSHOT_PATH

    if not os.path.exists(src_db):
        print(f"Error: Source database does not exist at {src_db}")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(out_snapshot)), exist_ok=True)
    temp_db_path = out_snapshot + ".tmp.db"
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

    print(f"1. Creating clean in-memory / temporary copy of {src_db}...")
    try:
        src_conn = sqlite3.connect(f"file:{os.path.abspath(src_db)}?mode=ro", uri=True)
        dest_conn = sqlite3.connect(temp_db_path, autocommit=True)
        src_conn.backup(dest_conn)
        src_conn.close()

        print("2. Stripping user-specific tables and runtime state...")
        for tbl in USER_TABLES:
            try:
                dest_conn.execute(f"DELETE FROM {tbl};")
            except sqlite3.OperationalError:
                pass

        try:
            dest_conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('exercise_history', 'audio_cache');")
        except sqlite3.OperationalError:
            pass

        print("3. Vacuuming and optimizing database file...")
        dest_conn.execute("VACUUM;")
        dest_conn.close()

        uncompressed_size = os.path.getsize(temp_db_path) / (1024 * 1024)
        print(f"   Clean database size: {uncompressed_size:.2f} MB")

        print(f"4. Compressing snapshot to {out_snapshot} (level={compress_level})...")
        with open(temp_db_path, "rb") as f_in, gzip.open(out_snapshot, "wb", compresslevel=compress_level) as f_out:
            shutil.copyfileobj(f_in, f_out)

        compressed_size = os.path.getsize(out_snapshot) / (1024 * 1024)
        print(f"✅ Successfully created snapshot: {out_snapshot} ({compressed_size:.2f} MB)")
        return True
    except Exception as e:
        print(f"Error dumping snapshot: {e}")
        return False
    finally:
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass


def restore_snapshot(
    snapshot_path: Optional[str] = None,
    dest_db: Optional[str] = None,
    backup_existing: bool = True,
) -> bool:
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT_PATH
    dest_db = dest_db or (os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH)

    if not os.path.exists(snapshot_path):
        print(f"Error: Snapshot file does not exist at {snapshot_path}")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(dest_db)), exist_ok=True)

    if os.path.exists(dest_db) and backup_existing:
        backup_path = dest_db + ".bak"
        print(f"Backing up existing database to {backup_path}...")
        try:
            shutil.copy2(dest_db, backup_path)
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")

    # Remove existing db and any SQLite WAL/SHM files for clean restart
    for suffix in ("", "-wal", "-shm"):
        target_file = dest_db + suffix
        if os.path.exists(target_file):
            try:
                os.remove(target_file)
            except OSError as e:
                print(f"Warning: Failed to remove {target_file}: {e}")

    print(f"Extracting snapshot {snapshot_path} -> {dest_db}...")
    temp_extract = dest_db + ".restore.tmp"
    if os.path.exists(temp_extract):
        os.remove(temp_extract)

    try:
        with gzip.open(snapshot_path, "rb") as f_in, open(temp_extract, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(temp_extract, dest_db)

        # Seed defaults for fresh default_user (using pure standard library SQL)
        from backend.config import (
            DEFAULT_CEFR_LEVEL,
            DEFAULT_CORRECTION_THRESHOLD,
            DEFAULT_CARDS_PER_EXERCISE,
            DEFAULT_MIXED_TRANSLATION_RATIO,
        )

        with sqlite3.connect(dest_db, timeout=15.0) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO users (id, name)
                VALUES ('default_user', 'Default User');
            """)
            trans_mode = conn.execute("SELECT mode FROM user_modes WHERE user_id = 'default_user' AND mode = 'translation';").fetchone()
            if not trans_mode:
                conn.execute("""
                    INSERT OR IGNORE INTO user_modes (user_id, mode, level, streak, total_completed, total_skipped)
                    VALUES ('default_user', 'translation', ?, 0, 0, 0);
                """, (DEFAULT_CEFR_LEVEL,))

            audio_mode = conn.execute("SELECT mode FROM user_modes WHERE user_id = 'default_user' AND mode = 'transcription';").fetchone()
            if not audio_mode:
                conn.execute("""
                    INSERT OR IGNORE INTO user_modes (user_id, mode, level, streak, total_completed, total_skipped)
                    VALUES ('default_user', 'transcription', ?, 0, 0, 0);
                """, (DEFAULT_CEFR_LEVEL,))

            conn.execute("""
                INSERT OR IGNORE INTO user_settings (user_id, theme, preferred_voice, correction_threshold, cards_per_exercise, auto_jump, auto_play_audio, transcription_retry, blind_retry, active_mode, mixed_translation_ratio)
                VALUES ('default_user', 'system', 'default', ?, ?, 1, 1, 1, 0, 'translation', ?);
            """, (DEFAULT_CORRECTION_THRESHOLD, DEFAULT_CARDS_PER_EXERCISE, DEFAULT_MIXED_TRANSLATION_RATIO))
            conn.commit()

        print(f"✅ Database cleanly restored and initialized at {dest_db}!")
        return True
    except Exception as e:
        print(f"Error restoring snapshot: {e}")
        if os.path.exists(temp_extract):
            try:
                os.remove(temp_extract)
            except OSError:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(description="FrenchTutor SQLite Snapshot Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Dump command
    dump_parser = subparsers.add_parser("dump", help="Create a clean snapshot from current database")
    dump_parser.add_argument("--src", type=str, default=None, help="Source database path")
    dump_parser.add_argument("--out", type=str, default=None, help="Output compressed snapshot path")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore database from snapshot (clean restart)")
    restore_parser.add_argument("--snapshot", type=str, default=None, help="Snapshot file path to restore from")
    restore_parser.add_argument("--dest", type=str, default=None, help="Destination database path")
    restore_parser.add_argument("--no-backup", action="store_true", help="Do not create a .bak file of existing database")

    args = parser.parse_args()

    if args.command == "dump":
        success = dump_snapshot(src_db=args.src, out_snapshot=args.out)
        sys.exit(0 if success else 1)
    elif args.command == "restore":
        success = restore_snapshot(
            snapshot_path=args.snapshot,
            dest_db=args.dest,
            backup_existing=not args.no_backup,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
