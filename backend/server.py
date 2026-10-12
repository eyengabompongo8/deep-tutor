import os
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import deps
from backend.api.users import router as users_router
from backend.api.exercises import router as exercises_router
from backend.api.audio import router as audio_router
from backend.api.stats import router as stats_router
from backend.config import HOST, PORT, RELOAD

logger = logging.getLogger("backend.server")

# Lazy load MLEngine with thread-safe lock
ml_engine = None
ml_engine_lock = threading.Lock()


def get_ml_engine():
    global ml_engine
    if ml_engine is None:
        with ml_engine_lock:
            if ml_engine is None:
                logger.info("Loading MLEngine models...")
                from backend.ml.engine import MLEngine
                ml_engine = MLEngine()
                logger.info("MLEngine ready.")
    return ml_engine


# Component imports
from backend.db import DataManager
from backend.scoring.engine import ScoringEngine
from backend.nlp.tts_engine import get_tts_engine
from backend.ml.precompute import EagerPrecomputeManager

# Global singleton handles populated by lifespan (kept on server for backwards compatibility)
data_manager = None
scoring_engine = None
tts_engine = None
eager_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_manager, scoring_engine, tts_engine, eager_processor
    # Populate shared singletons in deps
    deps.data_manager = DataManager()
    deps.scoring_engine = ScoringEngine(data_manager=deps.data_manager)
    deps.tts_engine = get_tts_engine(data_manager=deps.data_manager)
    deps.eager_processor = EagerPrecomputeManager(
        get_engine_fn=get_ml_engine,
        data_manager=deps.data_manager,
    )
    data_manager = deps.data_manager
    scoring_engine = deps.scoring_engine
    tts_engine = deps.tts_engine
    eager_processor = deps.eager_processor

    logger.info("DeepTutor API backend started.")
    yield

    if deps.eager_processor is not None:
        try:
            deps.eager_processor.shutdown(wait=False)
        except Exception:
            pass
    if ml_engine is not None:
        try:
            ml_engine.close()
        except Exception:
            pass
    logger.info("DeepTutor API backend shutting down.")


def create_app() -> FastAPI:
    application = FastAPI(
        title="DeepTutor API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware for development
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API submodule routers
    application.include_router(users_router)
    application.include_router(exercises_router)
    application.include_router(audio_router)
    application.include_router(stats_router)

    # Mount static files if frontend build exists
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frontend_dist = os.path.join(root_dir, "frontend", "dist")
    if os.path.exists(frontend_dist):
        application.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="DeepTutor API Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind to")
    parser.add_argument("--host", type=str, default=HOST, help="Host interface to bind to")
    parser.add_argument("--reload", action="store_true", default=RELOAD, help="Enable auto-reload")
    parser.add_argument("--reload-dir", type=str, default="backend", help="Directory to monitor for auto-reload")
    args = parser.parse_args()
    reload_dirs = [args.reload_dir] if args.reload and args.reload_dir else None
    uvicorn.run("backend.server:app", host=args.host, port=args.port, reload=args.reload, reload_dirs=reload_dirs)
