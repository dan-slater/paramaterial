#!/bin/bash

# Script to sync webapp files to the private paramaterial-webapp repository
# Run this from the main paramaterial directory

echo "🔄 Syncing webapp files to paramaterial-webapp repository..."

WEBAPP_REPO="../paramaterial-webapp"

# Check if the webapp repo exists
if [ ! -d "$WEBAPP_REPO" ]; then
    echo "❌ paramaterial-webapp repository not found!"
    echo "Please clone it first:"
    echo "cd /Users/ds/CascadeProjects"
    echo "git clone git@github.com:dan-slater/paramaterial-webapp.git"
    exit 1
fi

# Backup current directory
CURRENT_DIR=$(pwd)

echo "📁 Current directory: $CURRENT_DIR"
echo "📁 Webapp repo: $WEBAPP_REPO"

# Sync webapp files (excluding .git directory)
echo "📋 Copying webapp files..."
rsync -av --exclude='.git' webapp/ "$WEBAPP_REPO/"

# Copy deployment-related files from main repo
echo "📋 Copying deployment files..."
cp -f .github/workflows/deploy.yml "$WEBAPP_REPO/.github/workflows/" 2>/dev/null || {
    mkdir -p "$WEBAPP_REPO/.github/workflows"
    cp -f .github/workflows/deploy.yml "$WEBAPP_REPO/.github/workflows/"
}
cp -f deploy.sh "$WEBAPP_REPO/"
cp -f DEPLOYMENT.md "$WEBAPP_REPO/"

# Create/update webapp-specific files
echo "📝 Creating webapp-specific files..."

# Update README for webapp repo
cat > "$WEBAPP_REPO/README.md" << 'EOF'
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
curl -fsSL https://raw.githubusercontent.com/dan-slater/paramaterial-webapp/main/deploy.sh | bash
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

## 🔄 Development Workflow

1. Make changes in main paramaterial repo
2. Run sync script: `../paramaterial/sync-to-webapp-repo.sh`
3. Commit and push changes
4. GitHub Actions automatically deploys to production
5. Monitor via logs and health checks
EOF

# Create .env.development.example
cat > "$WEBAPP_REPO/.env.development.example" << 'EOF'
# Development Environment Variables
# Copy this to .env.development and fill in your values

DATABASE_URL=postgresql://paramaterial_user:paramaterial_password@localhost:5433/paramaterial
REDIS_URL=redis://localhost:6380/0
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=jwt-secret-key-change-in-production
DEBUG=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
EOF

# Create comprehensive .gitignore
cat > "$WEBAPP_REPO/.gitignore" << 'EOF'
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

# Fix URLs and usernames in copied files
echo "🔧 Updating URLs and usernames..."
cd "$WEBAPP_REPO"

# Update deploy.sh
sed -i.bak 's/YOUR_USERNAME/dan-slater/g' deploy.sh
sed -i.bak 's/paramaterial\.git/paramaterial-webapp.git/g' deploy.sh
sed -i.bak 's/cd paramaterial$/cd paramaterial-webapp/g' deploy.sh
rm -f deploy.sh.bak

# Update GitHub Actions workflow
sed -i.bak 's|scp -r ./webapp|scp -r ./|g' .github/workflows/deploy.yml
rm -f .github/workflows/deploy.yml.bak

# Update DEPLOYMENT.md
sed -i.bak 's/YOUR_USERNAME/dan-slater/g' DEPLOYMENT.md
sed -i.bak 's/paramaterial\.git/paramaterial-webapp.git/g' DEPLOYMENT.md
rm -f DEPLOYMENT.md.bak

cd "$CURRENT_DIR"

echo ""
echo "✅ Sync completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. cd ../paramaterial-webapp"
echo "2. git add ."
echo "3. git commit -m 'Sync latest webapp files and deployment configuration'"
echo "4. git push origin main"
echo ""
echo "🚀 This will trigger automatic deployment to 159.89.0.169!"
echo ""
echo "🔍 Don't forget to configure GitHub secrets if not already done:"
echo "   - DROPLET_SSH_KEY"
echo "   - POSTGRES_PASSWORD"
echo "   - REDIS_PASSWORD"
echo "   - SECRET_KEY"
echo "   - JWT_SECRET_KEY"