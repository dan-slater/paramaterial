#!/bin/bash

# ParaMaterial Production Deployment Script
# Run this on your droplet: 159.89.0.169

set -e

echo "🚀 Starting ParaMaterial deployment..."

# Configuration
DEPLOY_PATH="/opt/paramaterial"
REPO_URL="https://github.com/YOUR_USERNAME/paramaterial.git"  # Update this
BRANCH="main"

# Create deployment directory
sudo mkdir -p $DEPLOY_PATH
cd $DEPLOY_PATH

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Clone or update repository
if [ -d "paramaterial" ]; then
    echo "📥 Updating existing repository..."
    cd paramaterial
    git fetch origin
    git reset --hard origin/$BRANCH
else
    echo "📥 Cloning repository..."
    git clone -b $BRANCH $REPO_URL paramaterial
    cd paramaterial
fi

cd webapp

# Create production environment file
echo "⚙️ Setting up production environment..."
cat > .env.production << EOF
DATABASE_URL=postgresql://paramaterial_user:$(openssl rand -base64 32)@postgres:5432/paramaterial
REDIS_URL=redis://:$(openssl rand -base64 32)@redis:6379/0
SECRET_KEY=$(openssl rand -base64 64)
JWT_SECRET_KEY=$(openssl rand -base64 64)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
CORS_ORIGINS=http://159.89.0.169,https://paramaterial.vercel.app
DEBUG=false
EOF

echo "🛑 Stopping existing services..."
docker-compose -f docker-compose.prod.yml down || true

echo "🔨 Building production images..."
docker-compose -f docker-compose.prod.yml build --no-cache

echo "🚀 Starting production services..."
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d

echo "⏳ Waiting for services to start..."
sleep 15

echo "🗄️ Running database migrations..."
docker-compose -f docker-compose.prod.yml exec -T api alembic upgrade head || echo "⚠️ Migration failed, may be first run"

echo "🧹 Cleaning up..."
docker system prune -f

echo "✅ Deployment completed!"
echo "🌐 API available at: http://159.89.0.169/"
echo "📚 Documentation: http://159.89.0.169/docs"
echo "❤️ Health check: http://159.89.0.169/health"