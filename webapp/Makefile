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
	@echo "Starting development environment..."
	@echo "1. Starting database and Redis..."
	docker-compose up -d postgres redis
	@echo "2. Waiting for database to be ready..."
	@sleep 5
	@echo "3. Running migrations..."
	FLASK_APP=app_postgres.py flask db upgrade
	@echo "4. Starting Flask app on port 5555..."
	FLASK_APP=app_postgres.py FLASK_ENV=development python app_postgres.py

# Build Docker containers
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d
	@echo "Services started. Application available at http://localhost:5555"

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
	FLASK_ENV=testing python -m pytest tests/ -v

# Run database migrations
migrate:
	FLASK_APP=app_postgres.py flask db upgrade

# Create new migration
migration:
	@read -p "Migration message: " msg; \
	FLASK_APP=app_postgres.py flask db migrate -m "$$msg"

# Open Python shell with app context
shell:
	FLASK_APP=app_postgres.py flask shell

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
		FLASK_APP=app_postgres.py flask db upgrade; \
		echo "Database reset complete."; \
	else \
		echo "Reset cancelled."; \
	fi

# Initialize Flask-Migrate
init-db:
	FLASK_APP=app_postgres.py flask db init
	FLASK_APP=app_postgres.py flask db migrate -m "Initial migration"
	FLASK_APP=app_postgres.py flask db upgrade

# Check services status
status:
	docker-compose ps

# Create admin user
create-admin:
	@read -p "Admin email: " email; \
	read -s -p "Admin password: " password; \
	echo ""; \
	FLASK_APP=app_postgres.py python -c "\
		from app_postgres import app; \
		from models import db, User; \
		with app.app_context(): \
			user = User(email='$$email', first_name='Admin', last_name='User', is_verified=True); \
			user.set_password('$$password'); \
			db.session.add(user); \
			db.session.commit(); \
			print('Admin user created successfully!')"

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