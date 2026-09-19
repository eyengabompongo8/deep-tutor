# FrenchTutor Makefile
# Standard development targets powered by uv with dynamic port configuration for parallel git worktrees.

BACKEND_PORT  ?= 8000
PORT          ?= $(BACKEND_PORT)
FRONTEND_PORT ?= 5173
HOST          ?= 0.0.0.0

.PHONY: help install install-gpu dev start build test clean

help:
	@echo "=================================================================="
	@echo "🇫🇷  FrenchTutor Makefile - Standard Development Targets:"
	@echo "=================================================================="
	@echo "  make dev          Start full-stack dev servers (Vite HMR + FastAPI reload)"
	@echo "                    Custom ports example for parallel worktrees:"
	@echo "                    make dev BACKEND_PORT=8001 FRONTEND_PORT=5174"
	@echo "  make start        Start production FastAPI server (serves frontend/dist)"
	@echo "                    Custom port example: make start PORT=8080"
	@echo "  make build        Build production frontend bundle into frontend/dist"
	@echo "  make test         Run fast unit & API tests via uv pytest"
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

dev:
	@echo "=================================================================="
	@echo "🇫🇷  Starting FrenchTutor Full-Stack Development Environment"
	@echo "• Backend API:    http://localhost:$(PORT)"
	@echo "• API Docs:       http://localhost:$(PORT)/docs"
	@echo "• Frontend (HMR): http://localhost:$(FRONTEND_PORT)"
	@echo "=================================================================="
	@trap 'kill 0' EXIT INT TERM; \
		(PORT=$(PORT) BACKEND_PORT=$(PORT) HOST=$(HOST) uv run python -m backend.server --port $(PORT) --host $(HOST) --reload) & \
		(cd frontend && BACKEND_PORT=$(PORT) PORT=$(PORT) npm run dev -- --port $(FRONTEND_PORT)) & \
		wait

start:
	@if [ ! -d "frontend/dist" ]; then \
		echo "frontend/dist not found. Building frontend first..."; \
		$(MAKE) build; \
	fi
	PORT=$(PORT) BACKEND_PORT=$(PORT) HOST=$(HOST) \
		uv run python -m backend.server --port $(PORT) --host $(HOST)

build:
	cd frontend && npm run build

test:
	uv run pytest

clean:
	rm -rf frontend/dist .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
