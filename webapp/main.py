from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os
from datetime import datetime
import logging

from config_fastapi import get_settings
from api.auth import auth_router
from api.organizations import organizations_router
from api.jobs import jobs_router
from api.equipment import equipment_router
from api.templates import templates_router

# Global variables
redis_client = None
async_session_maker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    global redis_client, async_session_maker
    settings = get_settings()
    
    # Initialize Redis
    redis_client = redis.from_url(settings.redis_url)
    
    # Initialize Database
    engine = create_async_engine(
        settings.database_url.replace('postgresql://', 'postgresql+asyncpg://'),
        echo=settings.debug
    )
    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
    
    # Store in app state
    app.state.redis = redis_client
    app.state.db = async_session_maker
    
    yield
    
    # Shutdown
    await redis_client.close()
    await engine.dispose()

# Create FastAPI app
app = FastAPI(
    title="ParaMaterial API",
    description="Materials testing data parameterization platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Settings
settings = get_settings()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative React port
        "https://paramaterial.vercel.app",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Dependency to get database session
async def get_db():
    """Get database session"""
    async with app.state.db() as session:
        try:
            yield session
        finally:
            await session.close()

# Dependency to get Redis client
async def get_redis():
    """Get Redis client"""
    return app.state.redis

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        async with app.state.db() as session:
            await session.execute("SELECT 1")
        
        # Test Redis connection
        await app.state.redis.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "ParaMaterial API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(organizations_router, prefix="/api/v1/organizations", tags=["organizations"])
app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(equipment_router, prefix="/api/v1/equipment", tags=["equipment"])
app.include_router(templates_router, prefix="/api/v1/templates", tags=["templates"])

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_level="info"
    )