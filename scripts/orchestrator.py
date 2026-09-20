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
    run_mine_pdf,
    run_ingest_and_link,
    run_clear_db
)

def main():
    parser = argparse.ArgumentParser(description="FrenchTutor Database Ingestion & Mining Orchestrator")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: init DB -> seed lexemas -> mine Tatoeba -> ingest & link")
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema")
    parser.add_argument("--reset-db", action="store_true", help="Delete existing database file before initializing schema")
    parser.add_argument("--clear-db", action="store_true", help="Clear all sentences and lexema links from the database")
    parser.add_argument("--seed-lexemas", action="store_true", help="Seed the lexemas catalog table from flelex_beacco.tsv")
    parser.add_argument("--mine-tatoeba", action="store_true", help="Mine and score sentences from Tatoeba dataset")
    parser.add_argument("--mine-pdf", type=str, metavar="PATH", help="Mine, score, and translate sentences from a monolingual French PDF")
    parser.add_argument("--ingest", action="store_true", help="Ingest mined sentences into the database and build relational lexema links")
    
    args = parser.parse_args()
    
    actions = [args.all, args.init_db, args.reset_db, args.clear_db, args.seed_lexemas, args.mine_tatoeba, args.mine_pdf, args.ingest]
    if not any(actions):
        parser.print_help()
        sys.exit(0)

    # 1. Reset / Init DB
    if args.all or args.init_db or args.reset_db:
        print("\n" + "="*50)
        print("STEP: Initializing Database Schema")
        print("="*50)
        reset = args.reset_db or (args.all and getattr(args, 'reset_db', False))
        success = run_init_db(reset=reset)
        if not success:
            print("Failed to initialize database. Aborting.")
            sys.exit(1)
            
    # 2. Clear DB (Standalone maintenance)
    if args.clear_db:
        print("\n" + "="*50)
        print("STEP: Clearing Database Sentences & Links")
        print("="*50)
        success = run_clear_db()
        if not success:
            print("Failed to clear database.")
            sys.exit(1)

    # 3. Seed Lexemas
    if args.all or args.seed_lexemas:
        print("\n" + "="*50)
        print("STEP: Seeding Lexemas Catalog")
        print("="*50)
        success = run_seed_lexemas()
        if not success:
            print("Failed to seed lexemas. Aborting.")
            sys.exit(1)
            
    # Sentence pool for ingestion
    mined_sentences = []

    # 4. Mine Tatoeba
    if args.all or args.mine_tatoeba:
        print("\n" + "="*50)
        print("STEP: Mining Tatoeba Sentences")
        print("="*50)
        tatoeba_sentences = run_mine_tatoeba(max_per_level=1200)
        mined_sentences.extend(tatoeba_sentences)

    # 5. Mine PDF
    if args.mine_pdf:
        print("\n" + "="*50)
        print(f"STEP: Mining PDF ({args.mine_pdf})")
        print("="*50)
        pdf_sentences = run_mine_pdf(args.mine_pdf)
        mined_sentences.extend(pdf_sentences)

    # 6. Ingest & Link
    if args.all or args.ingest:
        print("\n" + "="*50)
        print("STEP: Ingesting Sentences and Relational Linking")
        print("="*50)
        
        if not mined_sentences:
            print("Error: No sentences were mined to ingest.")
            print("Please specify a mining source (e.g. --mine-tatoeba or --mine-pdf <path>) along with --ingest.")
            sys.exit(1)
            
        success = run_ingest_and_link(mined_sentences)
        if not success:
            print("Failed to ingest and link sentences.")
            sys.exit(1)
            
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
