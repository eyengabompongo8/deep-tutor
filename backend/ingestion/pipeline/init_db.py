import os
import argparse
from backend.db import DataManager
from backend.db.connection import DB_PATH, ROOT_DIR


def run_init_db(reset=False, db_path=None):
    if db_path is None:
        db_path = os.environ.get("FRENCHTUTOR_DB_PATH") or DB_PATH
    
    if reset and os.path.exists(db_path):
        print(f"Resetting database at {db_path} (deleting existing file)...")
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"Error removing existing database: {e}")
            return False
            
    print(f"Initializing database schema at {db_path}...")
    
    dm = DataManager(db_path=db_path)
    
    with dm._get_connection() as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        table_names = [t[0] for t in tables]
        
    print(f"Successfully initialized tables: {', '.join(table_names)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the FrenchTutor database.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database and start fresh.")
    args = parser.parse_args()
    
    run_init_db(reset=args.reset)
