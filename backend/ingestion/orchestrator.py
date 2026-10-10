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
    run_mine_universal_cefr,
    run_ingest_and_link,
    run_clear_db
)
from backend.db.snapshot import dump_snapshot, restore_snapshot


def main():
    parser = argparse.ArgumentParser(description="FrenchTutor Database Ingestion & Mining Orchestrator")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: init DB -> seed lexemas -> mine Tatoeba -> ingest & link")
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema")
    parser.add_argument("--reset-db", action="store_true", help="Delete existing database file before initializing schema")
    parser.add_argument("--clear-db", action="store_true", help="Clear all sentences and lexema links from the database")
    parser.add_argument("--dump-snapshot", action="store_true", help="Create a clean database snapshot (excluding user data) into data/snapshots/")
    parser.add_argument("--load-snapshot", action="store_true", help="Cleanly restore database from data/snapshots/default_snapshot.db.gz")
    parser.add_argument("--seed-lexemas", "--seed-lemmas", dest="seed_lexemas", action="store_true", help="Seed the lemmas and words catalog tables from Lexique 4")
    parser.add_argument("--mine-tatoeba", action="store_true", help="Mine and score sentences from Tatoeba dataset")
    parser.add_argument("--tatoeba-mode", type=str, choices=["parallel", "monolingual", "both"], default="both", help="Tatoeba mining mode: parallel (fra-spa), monolingual (fra + MarianMT), or both (default: both)")
    parser.add_argument("--tatoeba-limit", type=int, default=None, metavar="N", help="Limit number of sentences to mine/extract from Tatoeba")
    parser.add_argument("--mine-pdf", type=str, metavar="PATH", help="Mine, score, and translate sentences from a monolingual French PDF")
    parser.add_argument("--mine-universal-cefr", action="store_true", help="Mine, score, and translate sentences from UniversalCEFR HuggingFace dataset")
    parser.add_argument("--universal-cefr-limit", type=int, default=None, metavar="N", help="Limit number of sentences to mine from UniversalCEFR")
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
        args.dump_snapshot,
        args.load_snapshot,
        args.seed_lexemas,
        args.mine_tatoeba,
        args.mine_pdf,
        args.mine_universal_cefr,
        args.mine_lingua,
        args.mine_vekkt,
        args.mine_conversations,
        args.ingest
    ]
    if not any(actions):
        parser.print_help()
        sys.exit(0)

    # Snapshot operations
    if args.load_snapshot:
        print("\n" + "="*50)
        print("STEP: Restoring Database Snapshot")
        print("="*50)
        success = restore_snapshot()
        sys.exit(0 if success else 1)

    if args.dump_snapshot:
        print("\n" + "="*50)
        print("STEP: Creating Clean Database Snapshot")
        print("="*50)
        success = dump_snapshot()
        sys.exit(0 if success else 1)

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
            
    # Candidate pool for ingestion: List of {"fr": ..., "es": Optional[str], "source": ..., "metadata": ...}
    mined_candidates = []

    # 4. Mine Tatoeba
    if args.all or args.mine_tatoeba:
        print("\n" + "="*50)
        print("STEP: Mining Tatoeba Sentences")
        print("="*50)
        tatoeba_candidates = run_mine_tatoeba(
            mode=args.tatoeba_mode,
            limit=args.tatoeba_limit
        )
        mined_candidates.extend(tatoeba_candidates)

    # 5. Mine PDF
    if args.mine_pdf:
        print("\n" + "="*50)
        print(f"STEP: Mining PDF ({args.mine_pdf})")
        print("="*50)
        pdf_candidates = run_mine_pdf(args.mine_pdf, interactive=False)
        mined_candidates.extend(pdf_candidates)

    # 5b. Mine UniversalCEFR
    if args.all or args.mine_universal_cefr:
        print("\n" + "="*50)
        print("STEP: Mining UniversalCEFR Dataset (HuggingFace)")
        print("="*50)
        cefr_candidates = run_mine_universal_cefr(limit=args.universal_cefr_limit)
        mined_candidates.extend(cefr_candidates)

    # 5c. Mine Lingua.com Learner PDFs
    if args.mine_lingua:
        print("\n" + "="*50)
        print("STEP: Mining Lingua.com Learner PDFs")
        print("="*50)
        lingua_candidates = run_mine_lingua(limit=args.lingua_limit, interactive=False)
        mined_candidates.extend(lingua_candidates)

    # 5d. Mine vekkt/french_CEFR
    if args.mine_vekkt:
        print("\n" + "="*50)
        print("STEP: Mining vekkt/french_CEFR Dataset")
        print("="*50)
        vekkt_candidates = run_mine_vekkt_cefr(limit=args.vekkt_limit)
        mined_candidates.extend(vekkt_candidates)

    # 5e. Mine CATIE-AQ Everyday Conversations
    if args.mine_conversations:
        print("\n" + "="*50)
        print("STEP: Mining Spoken French (CATIE-AQ Conversations)")
        print("="*50)
        catie_candidates = run_mine_catie_conversations(limit=args.conversations_limit)
        mined_candidates.extend(catie_candidates)

    print(f"\nTotal raw candidate sentences collected across all sources: {len(mined_candidates)}")

    # 6. Post-Mining Processing & Ingestion
    if args.all or args.ingest:
        print("\n" + "="*50)
        print("STEP: Post-Mining Deduplication, Scoring, Translation & Ingestion")
        print("="*50)
        
        if not mined_candidates:
            print("Error: No candidate sentences were mined to ingest.")
            print("Please specify a mining source (e.g. --mine-tatoeba or --mine-pdf <path>) along with --ingest.")
            sys.exit(1)

        print("\n--- 1. Deduplicating All Candidates (MinHash LSH vs DB & Intra-Batch) ---")
        from backend.ingestion.pipeline import deduplicate_candidate_records, preprocess_and_score_candidates
        unique_candidates = deduplicate_candidate_records(mined_candidates)
        print(f"Candidates before deduplication: {len(mined_candidates)} | Unique remaining: {len(unique_candidates)}")

        if not unique_candidates:
            print("No new unique candidate sentences to process.")
            sys.exit(0)

        print("\n--- 2. Batch Scoring & Selective Translation ---")
        scored_records = preprocess_and_score_candidates(unique_candidates)
        print(f"Successfully preprocessed {len(scored_records)} records.")

        print("\n--- 3. Ingestion & Relational Linking ---")
        success = run_ingest_and_link(scored_records)
        if not success:
            print("Failed to ingest and link sentences.")
            sys.exit(1)
            
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)


if __name__ == "__main__":
    main()
