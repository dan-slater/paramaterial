# ParaMaterial - FastAPI + React Stack

Materials testing data parameterization platform built with FastAPI backend and React frontend.

## Architecture

### Backend (FastAPI)
- **FastAPI** - Modern async Python web framework
- **PostgreSQL** - Primary database with async SQLAlchemy
- **Redis** - Caching and session storage
- **Pydantic** - Data validation and serialization
- **JWT** - Token-based authentication
- **Background Tasks** - Async materials data processing

### Frontend (React)
- **React + TypeScript** - Component-based UI
- **Tailwind CSS** - Utility-first styling with plasma colormap theme
- **Vite** - Fast build tool and dev server
- **Axios** - HTTP client for API communication

## Quick Start

### Development Setup

1. **Start the backend services:**
```bash
make dev
# Starts PostgreSQL, Redis, and FastAPI on port 8000
```

2. **Start the frontend (in another terminal):**
```bash
cd frontend
npm install
npm run dev
# Starts React dev server on port 3000
```

3. **Or use Docker for everything:**
```bash
make up
# Starts full stack with Docker Compose
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Key Features

- **User Authentication** - JWT-based registration/login
- **File Upload** - Drag & drop with validation for materials data
- **Background Processing** - Async analysis of CSV/Excel files
- **Real-time Updates** - Job status polling
- **Organization Management** - Multi-tenant with role-based access
- **Plasma Theme** - Python matplotlib plasma colormap inspired design

## Development Commands

```bash
make install       # Install Python dependencies
make dev          # Start FastAPI development server
make up           # Start full stack with Docker
make down         # Stop all services
make test         # Run tests
make clean        # Clean up containers
```

## File Structure

```
webapp/
├── main.py                 # FastAPI application
├── config_fastapi.py      # Pydantic settings
├── database.py            # Async SQLAlchemy setup
├── requirements.txt       # Python dependencies
├── api/                   # FastAPI route modules
├── models/                # SQLAlchemy models
├── schemas/               # Pydantic schemas
├── services/              # Business logic
├── archive/               # Deprecated Flask files
└── frontend/              # React application
```

## API Endpoints

- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/auth/me` - Current user info
- `POST /api/v1/jobs` - Create analysis job
- `GET /api/v1/jobs` - List user jobs
- `GET /api/v1/jobs/{id}` - Job details
- `GET /api/v1/jobs/{id}/status` - Job status (for polling)

## Materials Processing

The platform processes materials testing data through:

1. **File Upload** - CSV, Excel, JSON, TXT files
2. **Validation** - File format and data integrity checks
3. **Background Analysis** - Pandas/numpy data processing
4. **Results Storage** - Structured results in PostgreSQL
5. **Real-time Updates** - Status polling for progress tracking

## Environment Variables

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/paramaterial
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret
DEBUG=true
```

## Migration from Flask

This codebase was migrated from Flask to FastAPI. Legacy Flask files are archived in the `archive/` directory. Key changes:

- Flask-SQLAlchemy → Pure SQLAlchemy with async support
- Flask routes → FastAPI routers with automatic OpenAPI docs
- Session-based auth → JWT token authentication
- Synchronous processing → Async background tasks
- Jinja2 templates → React frontend

## Contributing

1. Follow the existing code style (Black formatting)
2. Add type hints for all functions
3. Write tests for new features
4. Update API documentation
5. Test with both development and Docker environments