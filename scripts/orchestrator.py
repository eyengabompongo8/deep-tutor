#!/usr/bin/env python3
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from scripts.pipeline import (
    run_init_db,
    run_seed_lexemas,
    run_mine_tatoeba,
    run_ingest_and_link
)

def main():
    parser = argparse.ArgumentParser(description="FrenchTutor Database Population Orchestrator")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline end-to-end")
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema")
    parser.add_argument("--reset-db", action="store_true", help="Delete existing database before initializing")
    parser.add_argument("--seed-lexemas", action="store_true", help="Seed the lexemas table from flelex_beacco.tsv")
    parser.add_argument("--mine", action="store_true", help="Mine sentences from Tatoeba")
    parser.add_argument("--ingest", action="store_true", help="Ingest sentences and link lexemas (requires --mine to be run, or --all)")
    
    args = parser.parse_args()
    
    if not any([args.all, args.init_db, args.seed_lexemas, args.mine, args.ingest]):
        parser.print_help()
        sys.exit(0)

    if args.all or args.init_db:
        print("\n" + "="*50)
        print("STEP 1: Initializing Database Schema")
        print("="*50)
        reset = args.reset_db if args.init_db else (args.all and getattr(args, 'reset_db', False))
        success = run_init_db(reset=reset)
        if not success:
            print("Failed to initialize database. Aborting pipeline.")
            sys.exit(1)
            
    if args.all or args.seed_lexemas:
        print("\n" + "="*50)
        print("STEP 2: Seeding Lexemas Catalog")
        print("="*50)
        success = run_seed_lexemas()
        if not success:
            print("Failed to seed lexemas. Aborting pipeline.")
            sys.exit(1)
            
    mined_sentences = None
    if args.all or args.mine:
        print("\n" + "="*50)
        print("STEP 3: Mining Tatoeba Sentences")
        print("="*50)
        mined_sentences = run_mine_tatoeba(max_per_level=1200)
        
        if not args.all and args.mine:
            import json
            out_path = os.path.join(BASE_DIR, 'data', 'mined_sentences.json')
            print(f"\nSaving output to {out_path} for inspection or standalone ingestion...")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(mined_sentences, f, ensure_ascii=False, indent=2)

    if args.all or args.ingest:
        print("\n" + "="*50)
        print("STEP 4: Ingesting Sentences and Relational Linking")
        print("="*50)
        
        if mined_sentences is None:
            import json
            in_path = os.path.join(BASE_DIR, 'data', 'mined_sentences.json')
            if not os.path.exists(in_path):
                print(f"Error: {in_path} not found.")
                print("You must run with --mine first, or use --all to run everything in memory.")
                sys.exit(1)
            print(f"Loading mined sentences from {in_path}...")
            with open(in_path, 'r', encoding='utf-8') as f:
                mined_sentences = json.load(f)
                
        success = run_ingest_and_link(mined_sentences)
        if not success:
            print("Failed to ingest and link sentences.")
            sys.exit(1)
            
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
