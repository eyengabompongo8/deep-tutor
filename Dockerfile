# syntax=docker/dockerfile:1

# Stage 1: Build the frontend (Vite)
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build the backend and runtime environment
FROM python:3.11-slim
WORKDIR /app

# Install minimal system dependencies (including headless Java for LanguageTool)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install uv (astral-sh/uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure uv and Hugging Face cache behavior
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/root/.cache/huggingface

# Define the build argument for the PyTorch hardware target
# Available options in pyproject.toml: cpu, cu118, cu124
ARG UV_EXTRA=cpu
ENV UV_EXTRA=${UV_EXTRA}

# Copy package requirements
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (no dev dependencies, frozen lockfile)
# This step is cached by Docker unless pyproject.toml or uv.lock change!
RUN uv sync --extra ${UV_EXTRA} --no-dev --frozen

# Copy the backend code, data, and environment template
COPY backend/ ./backend/
COPY data/ ./data/
COPY .env.example .env

# Copy the frontend built assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose backend port
EXPOSE 8000

# Start the application using uv run
CMD ["uv", "run", "--no-sync", "python", "-m", "backend.server", "--host", "0.0.0.0", "--port", "8000"]
