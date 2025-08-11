# aNEOS Deployment Guide

Production Deployment Guide for aNEOS - Docker, Kubernetes, and Cloud Platforms

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Local Development Deployment](#local-development-deployment)
4. [Docker Containerization](#docker-containerization)
5. [Docker Compose Deployment](#docker-compose-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Cloud Platform Deployment](#cloud-platform-deployment)
8. [Production Configuration](#production-configuration)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Security Considerations](#security-considerations)
11. [Scaling and Performance](#scaling-and-performance)
12. [Backup and Disaster Recovery](#backup-and-disaster-recovery)

---

## Introduction

### Deployment Overview

aNEOS supports multiple deployment strategies to meet different operational requirements:

- **Development**: Local development with Docker
- **Testing**: Docker Compose multi-service setup
- **Staging**: Kubernetes cluster deployment
- **Production**: Full enterprise deployment with monitoring
- **Cloud**: AWS, GCP, Azure deployment options

### Deployment Architectures

```
┌─────────────────────────────────────────────────────────┐
│                    Production Architecture               │
├─────────────────────────────────────────────────────────┤
│  Load Balancer                                          │
│  ├── API Gateway (nginx/traefik)                        │
│  └── SSL Termination                                    │
├─────────────────────────────────────────────────────────┤
│  Application Layer                                      │
│  ├── aNEOS API (3+ replicas)                           │
│  ├── Dashboard Service                                  │
│  └── ML Inference Service                              │
├─────────────────────────────────────────────────────────┤
│  Data Processing Layer                                  │
│  ├── Analysis Workers                                   │
│  ├── ML Training Service                               │
│  └── Batch Processing                                  │
├─────────────────────────────────────────────────────────┤
│  Data Layer                                             │
│  ├── PostgreSQL (Primary/Replica)                      │
│  ├── Redis Cache                                       │
│  └── Object Storage                                    │
├─────────────────────────────────────────────────────────┤
│  Monitoring Layer                                       │
│  ├── Prometheus + Grafana                              │
│  ├── ELK Stack (Logging)                              │
│  └── Alert Manager                                     │
└─────────────────────────────────────────────────────────┘
```

### Prerequisites

#### System Requirements
- **CPU**: 4+ cores (8+ for production)
- **RAM**: 8GB minimum (32GB+ for production)
- **Storage**: 100GB+ SSD storage
- **Network**: High-speed internet for NEO data APIs

#### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.20+ (for K8s deployment)
- kubectl CLI tool
- Helm 3.0+ (recommended)

---

## System Architecture

### Component Overview

#### Core Services

1. **aNEOS API Service**
   - FastAPI-based REST API
   - Scientific analysis endpoints
   - ML prediction services
   - Real-time monitoring

2. **Dashboard Service**  
   - Web-based user interface
   - Real-time data visualization
   - System management tools
   - Analysis result browser

3. **Analysis Worker Service**
   - Background NEO analysis processing
   - Batch job execution
   - Scientific indicator computation
   - ML model inference

4. **Database Services**
   - PostgreSQL for structured data
   - Redis for caching and sessions
   - Object storage for large datasets

#### Supporting Services

1. **Message Queue**
   - Redis/RabbitMQ for task distribution
   - Async job processing
   - Event-driven communication

2. **Monitoring Stack**
   - Prometheus for metrics
   - Grafana for visualization
   - AlertManager for notifications

3. **Logging Stack**
   - Elasticsearch for log storage
   - Logstash for log processing
   - Kibana for log analysis

### Service Dependencies

```mermaid
graph TD
    A[Load Balancer] --> B[aNEOS API]
    A --> C[Dashboard]
    
    B --> D[PostgreSQL]
    B --> E[Redis]
    B --> F[Analysis Workers]
    
    C --> B
    
    F --> D
    F --> E
    F --> G[ML Models]
    
    H[Prometheus] --> B
    H --> C
    H --> F
    
    I[Grafana] --> H
    
    J[ELK Stack] --> B
    J --> C
    J --> F
```

---

## Local Development Deployment

### Quick Development Setup

#### 1. Development Environment with Docker

```bash
# Clone repository
git clone <repository-url>
cd aneos-project

# Build development image
docker build -f Dockerfile.dev -t aneos:dev .

# Run development container
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  -e ANEOS_ENV=development \
  aneos:dev python aneos_menu.py
```

#### 2. Development with Docker Compose

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  aneos-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    environment:
      - ANEOS_ENV=development
      - ANEOS_DATABASE_URL=postgresql://aneos:aneos123@db:5432/aneos_dev
      - ANEOS_REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
    depends_on:
      - db
      - redis
    command: python start_api.py --dev --reload
  
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=aneos_dev
      - POSTGRES_USER=aneos
      - POSTGRES_PASSWORD=aneos123
    volumes:
      - dev_postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - dev_redis_data:/data

volumes:
  dev_postgres_data:
  dev_redis_data:
```

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f aneos-api

# Stop environment
docker-compose -f docker-compose.dev.yml down
```

### Development Tools Integration

#### Visual Studio Code Integration

Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "aNEOS Development",
  "dockerComposeFile": ["../docker-compose.dev.yml"],
  "service": "aneos-api",
  "workspaceFolder": "/app",
  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.11"
    }
  },
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.flake8"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.formatting.provider": "black",
        "python.linting.enabled": true,
        "python.linting.flake8Enabled": true
      }
    }
  },
  "postCreateCommand": "pip install -e .",
  "forwardPorts": [8000, 5432, 6379]
}
```

---

## Docker Containerization

### Dockerfile Configuration

#### Production Dockerfile

```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r aneos && useradd -r -g aneos aneos

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/aneos/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aneos:aneos . .

# Create required directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache && \
    chown -R aneos:aneos /app

# Switch to non-root user
USER aneos

# Set Python path
ENV PATH="/home/aneos/.local/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "start_api.py"]
```

#### Development Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Create directories
RUN mkdir -p /app/logs /app/data /app/models /app/cache

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV ANEOS_ENV=development

# Default command for development
CMD ["python", "start_api.py", "--dev"]
```

### Container Build Scripts

#### Build Script (`build.sh`)

```bash
#!/bin/bash

set -e

# Configuration
IMAGE_NAME="aneos"
VERSION=${1:-"latest"}
BUILD_TYPE=${2:-"production"}

# Build arguments
if [ "$BUILD_TYPE" = "development" ]; then
    DOCKERFILE="Dockerfile.dev"
    TAG="$IMAGE_NAME:dev"
else
    DOCKERFILE="Dockerfile"
    TAG="$IMAGE_NAME:$VERSION"
fi

echo "Building $TAG using $DOCKERFILE..."

# Build image
docker build \
    -f $DOCKERFILE \
    -t $TAG \
    --build-arg VERSION=$VERSION \
    --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
    --build-arg VCS_REF=$(git rev-parse --short HEAD) \
    .

echo "Build completed: $TAG"

# Tag latest if building production
if [ "$BUILD_TYPE" = "production" ] && [ "$VERSION" != "latest" ]; then
    docker tag $TAG $IMAGE_NAME:latest
    echo "Tagged as: $IMAGE_NAME:latest"
fi
```

#### Push Script (`push.sh`)

```bash
#!/bin/bash

set -e

# Configuration
REGISTRY=${REGISTRY:-"your-registry.com"}
IMAGE_NAME="aneos"
VERSION=${1:-"latest"}

# Tag for registry
docker tag $IMAGE_NAME:$VERSION $REGISTRY/$IMAGE_NAME:$VERSION

# Push to registry
docker push $REGISTRY/$IMAGE_NAME:$VERSION

echo "Pushed: $REGISTRY/$IMAGE_NAME:$VERSION"
```

---

## Docker Compose Deployment

### Production Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Reverse proxy and load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - nginx_cache:/var/cache/nginx
    depends_on:
      - aneos-api
    restart: unless-stopped

  # aNEOS API service
  aneos-api:
    image: aneos:latest
    environment:
      - ANEOS_ENV=production
      - ANEOS_DATABASE_URL=postgresql://aneos:${POSTGRES_PASSWORD}@postgres:5432/aneos
      - ANEOS_REDIS_URL=redis://redis:6379/0
      - ANEOS_SECRET_KEY=${SECRET_KEY}
      - ANEOS_WORKERS=4
    volumes:
      - aneos_data:/app/data
      - aneos_models:/app/models
      - aneos_logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G

  # Analysis workers
  aneos-worker:
    image: aneos:latest
    command: python -m aneos_core.workers.analysis_worker
    environment:
      - ANEOS_ENV=production
      - ANEOS_DATABASE_URL=postgresql://aneos:${POSTGRES_PASSWORD}@postgres:5432/aneos
      - ANEOS_REDIS_URL=redis://redis:6379/0
      - ANEOS_WORKER_CONCURRENCY=4
    volumes:
      - aneos_data:/app/data
      - aneos_models:/app/models
      - aneos_logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: 2

  # PostgreSQL database
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=aneos
      - POSTGRES_USER=aneos
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G

  # Redis cache and message broker
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Prometheus monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  # Grafana dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    restart: unless-stopped

  # Elasticsearch for logging
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    restart: unless-stopped

  # Kibana log viewer
  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  aneos_data:
  aneos_models:
  aneos_logs:
  nginx_cache:
  prometheus_data:
  grafana_data:
  elasticsearch_data:

networks:
  default:
    driver: bridge
```

### Environment Configuration

Create `.env`:

```bash
# Database
POSTGRES_PASSWORD=your_secure_postgres_password

# Redis
REDIS_PASSWORD=your_secure_redis_password

# aNEOS
SECRET_KEY=your_very_secure_secret_key_here
ANEOS_ENV=production

# Monitoring
GRAFANA_PASSWORD=your_grafana_admin_password

# External APIs
ANEOS_REQUEST_TIMEOUT=30
ANEOS_MAX_RETRIES=3

# Performance
ANEOS_MAX_WORKERS=20
ANEOS_BATCH_SIZE=100
```

### Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream aneos_api {
        least_conn;
        server aneos-api:8000 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=dashboard:10m rate=5r/s;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Main server
    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://aneos_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Dashboard
        location /dashboard/ {
            limit_req zone=dashboard burst=10 nodelay;
            proxy_pass http://aneos_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://aneos_api;
            access_log off;
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### Deployment Commands

```bash
# Start full production stack
docker-compose up -d

# Scale API service
docker-compose up -d --scale aneos-api=5

# View logs
docker-compose logs -f aneos-api

# Update services
docker-compose pull
docker-compose up -d

# Stop services
docker-compose down

# Backup data
docker-compose exec postgres pg_dump -U aneos aneos > backup.sql
```

---

## Kubernetes Deployment

### Namespace and Configuration

#### 1. Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: aneos
  labels:
    name: aneos
---
# ConfigMap for application configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: aneos-config
  namespace: aneos
data:
  ANEOS_ENV: "production"
  ANEOS_LOG_LEVEL: "INFO"
  ANEOS_MAX_WORKERS: "20"
  ANEOS_BATCH_SIZE: "100"
  ANEOS_REQUEST_TIMEOUT: "30"
  ANEOS_MAX_RETRIES: "3"
---
# Secret for sensitive configuration
apiVersion: v1
kind: Secret
metadata:
  name: aneos-secrets
  namespace: aneos
type: Opaque
data:
  # Base64 encoded values
  SECRET_KEY: eW91cl9zZWNyZXRfa2V5X2hlcmU=
  POSTGRES_PASSWORD: cG9zdGdyZXNfcGFzc3dvcmQ=
  REDIS_PASSWORD: cmVkaXNfcGFzc3dvcmQ=
```

#### 2. Persistent Volumes

```yaml
# storage.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: aneos-data-pv
  namespace: aneos
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fast-ssd
  hostPath:
    path: /data/aneos
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: aneos-data-pvc
  namespace: aneos
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-data-pvc
  namespace: aneos
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

#### 3. Database Deployment

```yaml
# postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: aneos
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "aneos"
        - name: POSTGRES_USER
          value: "aneos"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: aneos-secrets
              key: POSTGRES_PASSWORD
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: aneos
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

#### 4. Redis Deployment

```yaml
# redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: aneos
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command:
        - redis-server
        - --requirepass
        - "$(REDIS_PASSWORD)"
        - --appendonly
        - "yes"
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: aneos-secrets
              key: REDIS_PASSWORD
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: aneos
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
  type: ClusterIP
```

#### 5. aNEOS API Deployment

```yaml
# aneos-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aneos-api
  namespace: aneos
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: aneos-api
  template:
    metadata:
      labels:
        app: aneos-api
    spec:
      containers:
      - name: aneos-api
        image: aneos:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANEOS_DATABASE_URL
          value: "postgresql://aneos:$(POSTGRES_PASSWORD)@postgres-service:5432/aneos"
        - name: ANEOS_REDIS_URL
          value: "redis://:$(REDIS_PASSWORD)@redis-service:6379/0"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: aneos-secrets
              key: SECRET_KEY
        envFrom:
        - configMapRef:
            name: aneos-config
        volumeMounts:
        - name: aneos-data
          mountPath: /app/data
        - name: aneos-models
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: aneos-data
        persistentVolumeClaim:
          claimName: aneos-data-pvc
      - name: aneos-models
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: aneos-api-service
  namespace: aneos
spec:
  selector:
    app: aneos-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aneos-api-hpa
  namespace: aneos
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aneos-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 6. Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aneos-ingress
  namespace: aneos
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - aneos.yourdomain.com
    secretName: aneos-tls
  rules:
  - host: aneos.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aneos-api-service
            port:
              number: 8000
```

### Kubernetes Deployment Commands

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f storage.yaml
kubectl apply -f postgres.yaml
kubectl apply -f redis.yaml
kubectl apply -f aneos-api.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n aneos
kubectl get services -n aneos
kubectl get ingress -n aneos

# Scale deployment
kubectl scale deployment aneos-api --replicas=5 -n aneos

# Rolling update
kubectl set image deployment/aneos-api aneos-api=aneos:v2.0 -n aneos

# View logs
kubectl logs -f deployment/aneos-api -n aneos

# Port forward for local access
kubectl port-forward service/aneos-api-service 8000:8000 -n aneos
```

### Helm Chart Deployment

Create `Chart.yaml`:

```yaml
apiVersion: v2
name: aneos
description: A Helm chart for aNEOS deployment
type: application
version: 1.0.0
appVersion: "1.0.0"
```

Create `values.yaml`:

```yaml
# Default values for aneos
replicaCount: 3

image:
  repository: aneos
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: aneos.example.com
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls:
    - secretName: aneos-tls
      hosts:
        - aneos.example.com

resources:
  limits:
    cpu: 2
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

postgresql:
  enabled: true
  auth:
    database: aneos
    username: aneos
  primary:
    persistence:
      size: 50Gi

redis:
  enabled: true
  auth:
    enabled: true
  master:
    persistence:
      size: 10Gi
```

Deploy with Helm:

```bash
# Install Helm chart
helm install aneos ./helm/aneos -n aneos --create-namespace

# Upgrade deployment
helm upgrade aneos ./helm/aneos -n aneos

# Uninstall
helm uninstall aneos -n aneos
```

---

## Cloud Platform Deployment

### AWS Deployment

#### 1. EKS Cluster Setup

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name aneos-cluster \
  --version 1.24 \
  --region us-west-2 \
  --nodegroup-name aneos-nodes \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 6 \
  --managed
```

#### 2. AWS Load Balancer Controller

```bash
# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=aneos-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

#### 3. RDS Database

```yaml
# rds-postgres.yaml
apiVersion: rds.aws.crossplane.io/v1alpha1
kind: RDSInstance
metadata:
  name: aneos-postgres
spec:
  forProvider:
    region: us-west-2
    dbInstanceClass: db.t3.medium
    allocatedStorage: 100
    engine: postgres
    engineVersion: "15.3"
    dbName: aneos
    masterUsername: aneos
    autoMinorVersionUpgrade: true
    backupRetentionPeriod: 7
    multiAZ: true
    storageEncrypted: true
    vpcSecurityGroupIds:
      - sg-xxxxxxxxx
  writeConnectionSecretsToRef:
    name: aneos-postgres-conn
    namespace: aneos
```

#### 4. ElastiCache Redis

```yaml
# elasticache-redis.yaml
apiVersion: cache.aws.crossplane.io/v1alpha1
kind: ReplicationGroup
metadata:
  name: aneos-redis
spec:
  forProvider:
    region: us-west-2
    cacheNodeType: cache.t3.medium
    numCacheClusters: 2
    engine: redis
    engineVersion: "7.0"
    parameterGroupName: default.redis7
    port: 6379
    atRestEncryptionEnabled: true
    transitEncryptionEnabled: true
  writeConnectionSecretsToRef:
    name: aneos-redis-conn
    namespace: aneos
```

### Google Cloud Platform (GCP)

#### 1. GKE Cluster

```bash
# Create GKE cluster
gcloud container clusters create aneos-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=6 \
  --enable-autorepair \
  --enable-autoupgrade
```

#### 2. Cloud SQL PostgreSQL

```bash
# Create Cloud SQL instance
gcloud sql instances create aneos-postgres \
  --database-version=POSTGRES_15 \
  --tier=db-g1-small \
  --region=us-central1 \
  --backup-start-time=02:00 \
  --enable-bin-log \
  --storage-auto-increase
```

### Azure Deployment

#### 1. AKS Cluster

```bash
# Create resource group
az group create --name aneos-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group aneos-rg \
  --name aneos-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 6 \
  --generate-ssh-keys
```

#### 2. Azure Database for PostgreSQL

```bash
# Create PostgreSQL server
az postgres server create \
  --resource-group aneos-rg \
  --name aneos-postgres \
  --location eastus \
  --admin-user aneos \
  --admin-password SecurePassword123! \
  --sku-name GP_Gen5_2 \
  --version 15
```

---

## Production Configuration

### Environment Configuration

#### Production Settings

```python
# config/production.py
import os
from pathlib import Path

class ProductionConfig:
    """Production configuration settings."""
    
    # Application
    ENV = 'production'
    DEBUG = False
    TESTING = False
    
    # Security
    SECRET_KEY = os.environ['SECRET_KEY']
    WTF_CSRF_ENABLED = True
    
    # Database
    DATABASE_URL = os.environ['DATABASE_URL']
    DATABASE_POOL_SIZE = 20
    DATABASE_POOL_TIMEOUT = 30
    DATABASE_POOL_RECYCLE = 3600
    
    # Redis
    REDIS_URL = os.environ['REDIS_URL']
    CACHE_TYPE = 'redis'
    CACHE_DEFAULT_TIMEOUT = 3600
    
    # API
    API_RATE_LIMIT = '1000 per hour'
    API_TIMEOUT = 30
    API_MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    
    # Workers
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_WORKER_CONCURRENCY = 8
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = '/app/logs/aneos.log'
    LOG_MAX_BYTES = 100 * 1024 * 1024  # 100MB
    LOG_BACKUP_COUNT = 10
    
    # Monitoring
    METRICS_ENABLED = True
    METRICS_PORT = 9090
    HEALTH_CHECK_ENABLED = True
    
    # Performance
    MAX_WORKERS = 20
    BATCH_SIZE = 100
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
```

### Security Configuration

#### SSL/TLS Configuration

```yaml
# ssl-cert.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: aneos-tls
  namespace: aneos
spec:
  secretName: aneos-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  commonName: aneos.yourdomain.com
  dnsNames:
  - aneos.yourdomain.com
  - api.aneos.yourdomain.com
```

#### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: aneos-network-policy
  namespace: aneos
spec:
  podSelector:
    matchLabels:
      app: aneos-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: aneos-worker
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Performance Optimization

#### Resource Management

```yaml
# resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: aneos-quota
  namespace: aneos
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "5"
    services: "5"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: aneos-limits
  namespace: aneos
spec:
  limits:
  - default:
      cpu: "2"
      memory: "4Gi"
    defaultRequest:
      cpu: "500m"
      memory: "1Gi"
    type: Container
```

#### Pod Disruption Budget

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: aneos-api-pdb
  namespace: aneos
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: aneos-api
```

---

This completes the comprehensive Deployment Guide for aNEOS. The guide covers all major deployment scenarios from local development to enterprise production deployments with proper security, monitoring, and scaling configurations.