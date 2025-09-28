"""
FastAPI application for social infrastructure prediction system.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from utils.exceptions import SocialInfraError
from routes.prediction import prediction_router
from routes.data import data_router


# Create FastAPI application
app = FastAPI(
    title="Social Infrastructure Prediction API",
    description="REST API for predicting infrastructure maintenance needs and investment priorities",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this properly in production
)


# Exception handlers
@app.exception_handler(SocialInfraError)
async def social_infra_exception_handler(request, exc: SocialInfraError):
    """Handle custom application exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": str(exc), "type": type(exc).__name__}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "type": "HTTPException"}
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "Social Infrastructure Prediction API"
    }


# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Social Infrastructure Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(prediction_router, prefix="/api/predict", tags=["predictions"])
app.include_router(data_router, prefix="/api/data", tags=["data"])


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("Starting Social Infrastructure Prediction API...")
    # Initialize database connections, load models, etc.


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down Social Infrastructure Prediction API...")
    # Close database connections, cleanup resources, etc.


def main():
    """Run the API server."""
    uvicorn.run(
        "app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()