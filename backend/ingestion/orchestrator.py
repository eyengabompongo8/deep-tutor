#!/usr/bin/env python3
import sys
import argparse

from backend.ingestion.pipeline import (
    run_init_db,
    run_seed_lexemas,
    run_mine_tatoeba,
    run_mine_pdf,
    run_mine_lingua,
    run_mine_vekkt_cefr,
    run_mine_catie_conversations,
    run_ingest_and_link,
    run_clear_db
)
from backend.ingestion.pipeline.mine_hf_dataset import run_mine_hf_dataset


def main():
    parser = argparse.ArgumentParser(description="FrenchTutor Database Ingestion & Mining Orchestrator")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: init DB -> seed lexemas -> mine Tatoeba -> ingest & link")
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema")
    parser.add_argument("--reset-db", action="store_true", help="Delete existing database file before initializing schema")
    parser.add_argument("--clear-db", action="store_true", help="Clear all sentences and lexema links from the database")
    parser.add_argument("--seed-lexemas", action="store_true", help="Seed the lexemas catalog table from flelex_beacco.tsv")
    parser.add_argument("--mine-tatoeba", action="store_true", help="Mine and score sentences from Tatoeba dataset")
    parser.add_argument("--mine-pdf", type=str, metavar="PATH", help="Mine, score, and translate sentences from a monolingual French PDF")
    parser.add_argument("--mine-hf", action="store_true", help="Mine, score, and translate sentences from UniversalCEFR HuggingFace dataset")
    parser.add_argument("--mine-lingua", action="store_true", help="Download and mine sentences from Lingua.com learner PDFs")
    parser.add_argument("--lingua-limit", type=int, default=None, metavar="N", help="Limit number of Lingua PDFs to download/mine")
    parser.add_argument("--mine-vekkt", action="store_true", help="Mine sentences from vekkt/french_CEFR HuggingFace dataset")
    parser.add_argument("--vekkt-limit", type=int, default=None, metavar="N", help="Limit number of sentences to mine from vekkt/french_CEFR")
    parser.add_argument("--mine-conversations", action="store_true", help="Mine everyday spoken sentences from CATIE-AQ conversations")
    parser.add_argument("--conversations-limit", type=int, default=None, metavar="N", help="Limit number of sentences to mine from conversations")
    parser.add_argument("--ingest", action="store_true", help="Ingest mined sentences into the database and build relational lexema links")
    
    args = parser.parse_args()
    
    actions = [
        args.all,
        args.init_db,
        args.reset_db,
        args.clear_db,
        args.seed_lexemas,
        args.mine_tatoeba,
        args.mine_pdf,
        args.mine_hf,
        args.mine_lingua,
        args.mine_vekkt,
        args.mine_conversations,
        args.ingest
    ]
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

    # 5b. Mine HuggingFace
    if args.all or args.mine_hf:
        print("\n" + "="*50)
        print("STEP: Mining HuggingFace Dataset (UniversalCEFR)")
        print("="*50)
        hf_sentences = run_mine_hf_dataset()
        mined_sentences.extend(hf_sentences)

    # 5c. Mine Lingua.com Learner PDFs
    if args.mine_lingua:
        print("\n" + "="*50)
        print("STEP: Mining Lingua.com Learner PDFs")
        print("="*50)
        lingua_sentences = run_mine_lingua(limit=args.lingua_limit, interactive=False)
        mined_sentences.extend(lingua_sentences)

    # 5d. Mine vekkt/french_CEFR
    if args.mine_vekkt:
        print("\n" + "="*50)
        print("STEP: Mining vekkt/french_CEFR Dataset")
        print("="*50)
        vekkt_sentences = run_mine_vekkt_cefr(limit=args.vekkt_limit)
        mined_sentences.extend(vekkt_sentences)

    # 5e. Mine CATIE-AQ Everyday Conversations
    if args.mine_conversations:
        print("\n" + "="*50)
        print("STEP: Mining Spoken French (CATIE-AQ Conversations)")
        print("="*50)
        catie_sentences = run_mine_catie_conversations(limit=args.conversations_limit)
        mined_sentences.extend(catie_sentences)

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
