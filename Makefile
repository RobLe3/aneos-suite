.PHONY: bootstrap up down test spec lint

bootstrap:
	@# Create init.sql if missing
	@test -f init.sql || echo "-- aNEOS bootstrap\nCREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" > init.sql
	@# Generate self-signed SSL certs if missing
	@mkdir -p ssl
	@if [ ! -f ssl/cert.pem ]; then \
		openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
			-keyout ssl/key.pem -out ssl/cert.pem \
			-subj "/CN=localhost" 2>/dev/null; \
		echo "Generated self-signed SSL certificates in ssl/"; \
	fi
	@# Copy .env.example to .env if missing
	@test -f .env || (test -f .env.example && cp .env.example .env && echo "Copied .env.example -> .env") || true

up:
	docker-compose up -d

down:
	docker-compose down

test:
	python -m pytest tests/ -v --tb=short

spec:
	@mkdir -p docs/api
	python -c "\
import sys, json; sys.path.insert(0, '.'); \
from aneos_api.app import create_app; \
app = create_app(); \
print(json.dumps(app.openapi(), indent=2, sort_keys=True))" \
	> docs/api/openapi.json
	@echo "OpenAPI spec written to docs/api/openapi.json"

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
	elif command -v flake8 >/dev/null 2>&1; then \
		flake8 .; \
	else \
		echo "Neither ruff nor flake8 found. Install one with: pip install ruff"; \
		exit 1; \
	fi
