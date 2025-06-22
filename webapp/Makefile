.PHONY: help install dev build up down clean test migrate shell logs

# Default target
help:
	@echo "ParaMaterial Development Commands"
	@echo "================================"
	@echo "install     - Install Python dependencies"
	@echo "dev         - Start development environment"
	@echo "build       - Build Docker containers"
	@echo "up          - Start all services with Docker Compose"
	@echo "down        - Stop all services"
	@echo "clean       - Clean up containers and volumes"
	@echo "test        - Run tests"
	@echo "migrate     - Run database migrations"
	@echo "shell       - Open Python shell with app context"
	@echo "db-shell    - Open PostgreSQL shell"
	@echo "logs        - View application logs"
	@echo "reset-db    - Reset database (WARNING: deletes all data)"

# Install dependencies
install:
	pip install -r requirements.txt


# Start development environment
dev:
	@echo "Starting FastAPI development environment..."
	@echo "1. Starting database and Redis..."
	docker-compose up -d postgres redis
	@echo "2. Waiting for database to be ready..."
	@sleep 5
	@echo "3. Starting FastAPI app on port 8000..."
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload


# Build Docker containers
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d
	@echo "Services started. FastAPI available at http://localhost:8000"
	@echo "API documentation available at http://localhost:8000/docs"

# Stop all services
down:
	docker-compose down

# Clean up containers and volumes
clean:
	docker-compose down -v
	docker-compose rm -f
	docker system prune -f

# Run tests
test:
	pytest tests/ -v

# Run database migrations (use Alembic directly)
migrate:
	alembic upgrade head

# Create new migration
migration:
	@read -p "Migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

# Open Python shell
shell:
	python

# Open PostgreSQL shell
db-shell:
	docker-compose exec postgres psql -U paramaterial_user -d paramaterial

# View logs
logs:
	docker-compose logs -f webapp

# View database logs
db-logs:
	docker-compose logs -f postgres

# Reset database (dangerous!)
reset-db:
	@echo "WARNING: This will delete ALL data!"
	@read -p "Are you sure? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		docker-compose down; \
		docker volume rm webapp_postgres_data; \
		docker-compose up -d postgres redis; \
		sleep 5; \
		alembic upgrade head; \
		echo "Database reset complete."; \
	else \
		echo "Reset cancelled."; \
	fi

# Initialize Alembic
init-db:
	alembic init alembic
	alembic revision --autogenerate -m "Initial migration"
	alembic upgrade head

# Check services status
status:
	docker-compose ps

# Create admin user (implement as needed)
create-admin:
	@echo "Admin user creation needs to be implemented via FastAPI script"

# Backup database
backup:
	@mkdir -p backups
	@echo "Creating database backup..."
	docker-compose exec postgres pg_dump -U paramaterial_user paramaterial > backups/paramaterial_backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/ directory"

# Restore database from backup
restore:
	@read -p "Backup file path: " backup_file; \
	if [ -f "$$backup_file" ]; then \
		echo "Restoring database from $$backup_file..."; \
		docker-compose exec -T postgres psql -U paramaterial_user -d paramaterial < "$$backup_file"; \
		echo "Database restored successfully!"; \
	else \
		echo "Backup file not found: $$backup_file"; \
	fi