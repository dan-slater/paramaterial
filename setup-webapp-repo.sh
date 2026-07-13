#!/bin/bash

# Script to set up the new paramaterial-webapp private repository
# Run this script from /Users/ds/CascadeProjects/paramaterial

echo "🚀 Setting up paramaterial-webapp private repository..."

# Navigate to parent directory
cd ..

# Create new repository directory
if [ ! -d "paramaterial-webapp" ]; then
    mkdir paramaterial-webapp
    echo "✅ Created paramaterial-webapp directory"
fi

cd paramaterial-webapp

# Initialize git repository
if [ ! -d ".git" ]; then
    git init
    echo "✅ Initialized git repository"
fi

# Copy all webapp files (if not already done)
if [ ! -f "main.py" ]; then
    cp -r ../paramaterial/webapp/* ./
    cp -r ../paramaterial/.github ./
    cp ../paramaterial/deploy.sh ./
    cp ../paramaterial/DEPLOYMENT.md ./
    echo "✅ Copied webapp files"
fi

# Create additional required files
cat > README.md << 'EOF'
# ParaMaterial WebApp

**Private repository for ParaMaterial web application deployment**

This repository contains the production-ready FastAPI backend for ParaMaterial, separated from the main public repository for security and deployment purposes.

## 🏗️ Architecture

- **Backend**: FastAPI + SQLModel + PostgreSQL + Redis
- **Frontend**: React + TypeScript + Zustand (deployed separately)
- **Deployment**: Docker Compose + Nginx reverse proxy
- **CI/CD**: GitHub Actions for automated deployment

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.development.example .env.development
export $(cat .env.development | xargs)

# Start services
docker-compose up -d postgres redis

# Run migrations
alembic upgrade head

# Start API
uvicorn main:app --reload
```

### Production Deployment
```bash
# Automatic: Push to main branch triggers deployment
git push origin main

# Manual: Run deployment script on droplet
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/paramaterial-webapp/main/deploy.sh | bash
```

## 🔧 Configuration

### GitHub Secrets (Required)
- `DROPLET_SSH_KEY`: SSH private key for droplet access
- `POSTGRES_PASSWORD`: Database password
- `REDIS_PASSWORD`: Redis password
- `SECRET_KEY`: Application secret
- `JWT_SECRET_KEY`: JWT secret

## 🌐 Deployment Target

- **Server**: 159.89.0.169
- **API**: http://159.89.0.169/api/v1/
- **Docs**: http://159.89.0.169/docs
- **Health**: http://159.89.0.169/health

## 🔒 Security

- Rate limiting and security headers via Nginx
- Password-protected Redis
- Environment-based secrets management
- Internal-only database access
- CORS configuration for allowed origins
EOF

cat > .env.development.example << 'EOF'
# Development Environment Variables
# Copy this to .env.development and fill in your values

DATABASE_URL=postgresql://paramaterial_user:paramaterial_password@localhost:5433/paramaterial
REDIS_URL=redis://localhost:6380/0
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=jwt-secret-key-change-in-production
DEBUG=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
EOF

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
pamwebenv/

# Environment files
.env.production
.env.development
.env.local

# Database
*.db
*.sqlite3

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Docker
.dockerignore

# Uploads
uploads/
static/uploads/

# Alembic
alembic.ini.bak

# Testing
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints/

# Node (if any frontend assets)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
EOF

# Update deployment script to use correct repo name and username
sed -i.bak 's/YOUR_USERNAME/dan-slater/g' deploy.sh
sed -i.bak 's/paramaterial\.git/paramaterial-webapp.git/g' deploy.sh
sed -i.bak 's/cd paramaterial$/cd paramaterial-webapp/g' deploy.sh
rm -f deploy.sh.bak

# Update GitHub Actions workflow
sed -i.bak 's|scp -r ./webapp|scp -r ./|g' .github/workflows/deploy.yml
rm -f .github/workflows/deploy.yml.bak

# Add all files to git
git add .

# Initial commit
git commit -m "Initial commit: ParaMaterial webapp for production deployment

- FastAPI backend with SQLModel and async PostgreSQL
- Redis integration for caching and sessions  
- Docker Compose production configuration with Nginx
- GitHub Actions for automated deployment to droplet
- Security: rate limiting, headers, password-protected services
- Environment-based configuration management
- Database migrations with Alembic
- Comprehensive health checks and monitoring

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo ""
echo "✅ Repository setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create a new PRIVATE repository on GitHub called 'paramaterial-webapp'"
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/dan-slater/paramaterial-webapp.git"
echo "3. Push to GitHub:"
echo "   git push -u origin main"
echo "4. Configure GitHub secrets in repository settings:"
echo "   - DROPLET_SSH_KEY"
echo "   - POSTGRES_PASSWORD"
echo "   - REDIS_PASSWORD"
echo "   - SECRET_KEY"
echo "   - JWT_SECRET_KEY"
echo "5. deploy.sh is already configured for dan-slater"
echo "6. Push to main branch to trigger deployment!"
echo ""
echo "🌐 After setup, your API will be available at: http://159.89.0.169/"