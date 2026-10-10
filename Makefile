# DeepTutor Makefile
# Standard development targets powered by uv with dynamic port configuration for parallel git worktrees.

BACKEND_PORT  ?= 8000
PORT          ?= $(BACKEND_PORT)
FRONTEND_PORT ?= 5173
HOST          ?= 0.0.0.0

.PHONY: help install install-gpu dev start build test ingest load-db dump-db clean

help:
	@echo "=================================================================="
	@echo "🇫🇷  DeepTutor Makefile - Standard Development Targets:"
	@echo "=================================================================="
	@echo "  make dev          Start full-stack dev servers (Vite HMR + FastAPI reload)"
	@echo "                    Custom ports example for parallel worktrees:"
	@echo "                    make dev BACKEND_PORT=8001 FRONTEND_PORT=5174"
	@echo "  make start        Start production FastAPI server (serves frontend/dist)"
	@echo "                    Custom port example: make start PORT=8080"
	@echo "  make build        Build production frontend bundle into frontend/dist"
	@echo "  make test         Run fast unit & API tests via uv pytest"
	@echo "  make ingest       Run database ingestion & mining pipeline (pass ARGS=\"...\")"
	@echo "  make load-db      Restore database snapshot into data/app.db (skip ingestion)"
	@echo "  make dump-db      Create clean snapshot from current database into data/snapshots/"
	@echo "  make install      Install Python dependencies (uv sync CPU) & frontend npm"
	@echo "  make install-gpu  Install Python dependencies with CUDA 12.4 GPU acceleration"
	@echo "  make clean        Remove build artifacts, caches, and __pycache__"
	@echo "=================================================================="

install:
	uv sync --extra cpu
	cd frontend && npm install

install-gpu:
	uv sync --extra cu124
	cd frontend && npm install

SHELL := /bin/bash

dev:
	@echo "=================================================================="
	@echo "🇫🇷  Starting DeepTutor Full-Stack Development Environment"
	@echo "• Backend API:    http://localhost:$(PORT)"
	@echo "• API Docs:       http://localhost:$(PORT)/docs"
	@echo "• Frontend (HMR): http://localhost:$(FRONTEND_PORT)"
	@echo "=================================================================="
	@set -m; \
	cleanup() { \
		trap - EXIT INT TERM; \
		kill -- -$${BACKEND_PID} 2>/dev/null || true; \
		kill -- -$${FRONTEND_PID} 2>/dev/null || true; \
		wait 2>/dev/null || true; \
	}; \
	trap cleanup EXIT INT TERM; \
	(PORT=$(PORT) BACKEND_PORT=$(PORT) HOST=$(HOST) uv run --no-sync python -m backend.server --port $(PORT) --host $(HOST) --reload --reload-dir backend) & \
	BACKEND_PID=$$!; \
	(cd frontend && BACKEND_PORT=$(PORT) PORT=$(PORT) npm run dev -- --host $(HOST) --port $(FRONTEND_PORT)) & \
	FRONTEND_PID=$$!; \
	wait $${BACKEND_PID} $${FRONTEND_PID} 2>/dev/null || true


start:
	@if [ ! -d "frontend/dist" ]; then \
		echo "frontend/dist not found. Building frontend first..."; \
		$(MAKE) build; \
	fi
	PORT=$(PORT) BACKEND_PORT=$(PORT) HOST=$(HOST) \
		uv run --no-sync python -m backend.server --port $(PORT) --host $(HOST)

build:
	cd frontend && npm run build

test:
	uv run --no-sync pytest

ingest:
	uv run --no-sync python -m backend.ingestion.orchestrator $(ARGS)

load-db:
	uv run --no-sync python -m backend.db.snapshot restore $(ARGS)

dump-db:
	uv run --no-sync python -m backend.db.snapshot dump $(ARGS)

clean:
	rm -rf frontend/dist .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
