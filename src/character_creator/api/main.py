"""FastAPI application for character creator service."""

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from character_creator import __version__
from character_creator.api.models import HealthResponse
from character_creator.api.routes import characters, evolution, interactions
from character_creator.utils.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:  # type: ignore[return]
    """Application lifespan management.

    Startup: Initialize character database with defaults.
    Shutdown: Cleanup and logging.

    Args:
        app: FastAPI application instance.

    Yields:
        Control to FastAPI.

    """
    logger.info("Character Creator API starting up...")
    # Initialize database and load/create default characters
    characters.initialize_default_characters()
    logger.info("Character database initialized")
    yield
    logger.info("Character Creator API shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.

    """
    app = FastAPI(
        title="Character Creator API",
        description="Interactive LLM-backed character creation and dialogue system",
        version=__version__,
        lifespan=lifespan,
    )

    # Configure CORS — restrict origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> dict[str, str]:
        """Health check endpoint.

        Returns:
            Health status and API version.

        """
        return {"status": "healthy", "version": __version__}

    # Include routers
    app.include_router(characters.router)
    app.include_router(interactions.router)
    app.include_router(evolution.router)

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        """Root endpoint with API information.

        Returns:
            API information and documentation links.

        """
        return {
            "message": "Welcome to Character Creator API",
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    return app


# Create application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "character_creator.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
