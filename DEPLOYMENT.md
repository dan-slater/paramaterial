# ParaMaterial Deployment Guide

## 🚀 Automated Deployment to Droplet (159.89.0.169)

### Prerequisites

1. **GitHub Secrets Setup** (Required for GitHub Actions):
   ```
   DROPLET_SSH_KEY: Your private SSH key for droplet access
   POSTGRES_PASSWORD: Secure database password
   REDIS_PASSWORD: Secure Redis password  
   SECRET_KEY: Application secret key
   JWT_SECRET_KEY: JWT signing key
   ```

2. **Droplet Requirements**:
   - Ubuntu 20.04+ LTS
   - Docker and Docker Compose installed
   - SSH access configured

### 🤖 Automatic Deployment

**Every push to `main` branch triggers automatic deployment:**

1. Code is pushed to `main` branch
2. GitHub Actions workflow starts
3. Files are copied to droplet
4. Production environment is configured
5. Docker containers are built and deployed
6. Database migrations run automatically
7. Health checks verify deployment success

### 🛠️ Manual Deployment

#### Option 1: Using the deployment script

```bash
# On your droplet (159.89.0.169)
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/paramaterial/main/deploy.sh | bash
```

#### Option 2: Manual steps

```bash
# 1. SSH into droplet
ssh root@159.89.0.169

# 2. Create deployment directory
sudo mkdir -p /opt/paramaterial
cd /opt/paramaterial

# 3. Clone repository
git clone https://github.com/YOUR_USERNAME/paramaterial.git
cd paramaterial/webapp

# 4. Create production environment
cp .env.production.example .env.production
# Edit .env.production with your secure values

# 5. Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d --build

# 6. Run migrations
docker-compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### 🔧 Production Configuration

#### Environment Variables
```bash
DATABASE_URL=postgresql://paramaterial_user:SECURE_PASSWORD@postgres:5432/paramaterial
REDIS_URL=redis://:SECURE_REDIS_PASSWORD@redis:6379/0
SECRET_KEY=your-super-secure-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
CORS_ORIGINS=http://159.89.0.169,https://paramaterial.vercel.app
DEBUG=false
```

#### Services
- **FastAPI Backend**: Port 8000 (internal)
- **PostgreSQL**: Internal network only
- **Redis**: Internal network only  
- **Nginx**: Ports 80/443 (public)

### 🌐 Access Points

- **API Root**: `http://159.89.0.169/`
- **API Documentation**: `http://159.89.0.169/docs`
- **Health Check**: `http://159.89.0.169/health`
- **API Endpoints**: `http://159.89.0.169/api/v1/*`

### 📊 Monitoring & Logs

```bash
# View all service logs
docker-compose -f docker-compose.prod.yml logs -f

# View specific service logs
docker-compose -f docker-compose.prod.yml logs -f api
docker-compose -f docker-compose.prod.yml logs -f postgres
docker-compose -f docker-compose.prod.yml logs -f nginx

# Check service status
docker-compose -f docker-compose.prod.yml ps

# Health check
curl http://159.89.0.169/health
```

### 🔄 Updates & Rollbacks

#### Update to latest version:
```bash
cd /opt/paramaterial/paramaterial
git pull origin main
cd webapp
docker-compose -f docker-compose.prod.yml up -d --build
```

#### Rollback to previous version:
```bash
cd /opt/paramaterial/paramaterial
git log --oneline -10  # Find commit hash
git checkout COMMIT_HASH
cd webapp
docker-compose -f docker-compose.prod.yml up -d --build
```

### 🛡️ Security Features

- Rate limiting (10 requests/second)
- Security headers (X-Frame-Options, X-Content-Type-Options, etc.)
- Password-protected Redis
- Database connection over internal network only
- Environment-based secrets management

### 🆘 Troubleshooting

#### Common Issues:

1. **Services won't start**: Check logs and ensure all environment variables are set
2. **Database connection errors**: Verify PostgreSQL container is healthy
3. **Permission errors**: Ensure proper file ownership and Docker permissions
4. **Port conflicts**: Check that ports 80/443 are available

#### Debug Commands:
```bash
# Check container health
docker-compose -f docker-compose.prod.yml ps

# Inspect specific container
docker inspect paramaterial_api_prod

# Access database directly
docker-compose -f docker-compose.prod.yml exec postgres psql -U paramaterial_user -d paramaterial

# Test API connectivity
curl -v http://159.89.0.169/health
```