# aNEOS Configuration Reference

Complete reference for all aNEOS configuration options and parameters

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Environment Variables](#environment-variables)
3. [Configuration Files](#configuration-files)
4. [Database Configuration](#database-configuration)
5. [API Configuration](#api-configuration)
6. [Analysis Configuration](#analysis-configuration)
7. [Machine Learning Configuration](#machine-learning-configuration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Performance Tuning](#performance-tuning)
10. [Security Configuration](#security-configuration)
11. [Deployment Configuration](#deployment-configuration)
12. [Advanced Configuration](#advanced-configuration)

---

## Configuration Overview

### Configuration Hierarchy

aNEOS uses a hierarchical configuration system with the following precedence (highest to lowest):

```
1. Command Line Arguments (--parameter value)
2. Environment Variables (ANEOS_PARAMETER)
3. Configuration Files (.env, config.yaml, config.json)
4. Default Values (built-in defaults)
```

### Configuration Sources

```
┌─────────────────────────────────────────────────────────┐
│                Configuration Sources                    │
├─────────────────────────────────────────────────────────┤
│  Command Line    Environment     Config Files   Defaults│
│  Arguments  ──→  Variables  ──→  (.env, yaml) ──→ Values│
│                                                         │
│  Highest Priority                    Lowest Priority    │
└─────────────────────────────────────────────────────────┘
```

### Configuration Validation

All configuration parameters are validated at startup:

- **Type checking**: Ensures correct data types
- **Range validation**: Checks numeric ranges
- **Dependency validation**: Verifies related settings
- **Security validation**: Checks for secure defaults

---

## Environment Variables

### Core System Variables

#### Application Environment

```bash
# Application Environment
ANEOS_ENV=development                    # development, testing, staging, production
ANEOS_DEBUG=false                       # Enable debug mode (boolean)
ANEOS_LOG_LEVEL=INFO                    # CRITICAL, ERROR, WARNING, INFO, DEBUG
ANEOS_SECRET_KEY=your_secret_key_here   # Application secret key (required)

# Server Configuration
ANEOS_HOST=0.0.0.0                      # Server bind address
ANEOS_PORT=8000                         # Server port number
ANEOS_WORKERS=1                         # Number of worker processes (auto = CPU count)
ANEOS_WORKER_CLASS=uvicorn.workers.UvicornWorker  # Worker class
ANEOS_RELOAD=false                      # Enable auto-reload (development only)

# Application Paths
ANEOS_DATA_DIR=dataneos                 # Data directory path
ANEOS_LOG_DIR=logs                      # Log directory path  
ANEOS_MODEL_DIR=models                  # ML models directory
ANEOS_CACHE_DIR=cache                   # Cache directory
```

#### Database Configuration

```bash
# Database URL (SQLite)
ANEOS_DATABASE_URL=sqlite:///./aneos.db

# Database URL (PostgreSQL - Production)
ANEOS_DATABASE_URL=postgresql://user:password@host:port/database

# Connection Pool Settings
ANEOS_DATABASE_POOL_SIZE=20             # Maximum connections in pool
ANEOS_DATABASE_POOL_OVERFLOW=0          # Additional overflow connections
ANEOS_DATABASE_POOL_TIMEOUT=30          # Connection timeout (seconds)
ANEOS_DATABASE_POOL_RECYCLE=3600        # Connection recycle time (seconds)

# Query Settings  
ANEOS_DATABASE_ECHO=false               # Log SQL queries (boolean)
ANEOS_DATABASE_ECHO_POOL=false          # Log connection pool activity
```

#### Cache and Session Configuration

```bash
# Redis Configuration
ANEOS_REDIS_URL=redis://localhost:6379/0  # Redis connection URL
ANEOS_REDIS_PASSWORD=                      # Redis password (if required)
ANEOS_REDIS_SSL=false                      # Use SSL/TLS for Redis connection

# Cache Settings
ANEOS_CACHE_TTL=3600                    # Default cache TTL (seconds)
ANEOS_CACHE_SIZE=1000                   # Maximum cache entries
ANEOS_CACHE_ENABLED=true                # Enable caching (boolean)

# Session Configuration  
ANEOS_SESSION_LIFETIME=86400            # Session lifetime (seconds)
ANEOS_SESSION_COOKIE_SECURE=false      # Secure session cookies (HTTPS only)
ANEOS_SESSION_COOKIE_HTTPONLY=true     # HTTP-only session cookies
```

### External API Configuration

#### NASA/JPL APIs

```bash
# API URLs
ANEOS_SBDB_URL=https://ssd-api.jpl.nasa.gov/sbdb.api          # Small Body Database
ANEOS_CAD_URL=https://ssd-api.jpl.nasa.gov/cad.api            # Close Approach Database  
ANEOS_HORIZONS_URL=https://ssd.jpl.nasa.gov/api/horizons.api  # JPL Horizons
ANEOS_MPC_URL=https://www.minorplanetcenter.net/              # Minor Planet Center
ANEOS_NEODYS_URL=https://newton.spacedys.com/neodys/api/      # NEODyS

# Request Configuration
ANEOS_REQUEST_TIMEOUT=10                # HTTP request timeout (seconds)
ANEOS_MAX_RETRIES=3                     # Maximum retry attempts
ANEOS_INITIAL_RETRY_DELAY=3             # Initial retry delay (seconds)
ANEOS_BACKOFF_FACTOR=2.0               # Exponential backoff factor
ANEOS_MAX_RETRY_DELAY=60               # Maximum retry delay (seconds)

# Rate Limiting
ANEOS_API_RATE_LIMIT=100               # Requests per minute per API
ANEOS_API_CONCURRENT_REQUESTS=10       # Concurrent requests per API

# User Agent
ANEOS_USER_AGENT=aNEOS/1.0 (Near Earth Object Analysis System)
```

### Analysis Configuration

#### Scientific Analysis Parameters

```bash
# Analysis Processing
ANEOS_MAX_WORKERS=10                    # Maximum worker threads for analysis
ANEOS_MAX_SUBPOINT_WORKERS=20          # Workers for subpoint calculations
ANEOS_BATCH_SIZE=100                   # Batch processing size
ANEOS_ANALYSIS_TIMEOUT=300             # Analysis timeout per object (seconds)

# Data Quality
ANEOS_MIN_OBSERVATIONS=5               # Minimum observations required
ANEOS_MAX_DATA_AGE_DAYS=30            # Maximum age of cached data (days)
ANEOS_REQUIRE_ORBITAL_ELEMENTS=true   # Require orbital elements (boolean)
ANEOS_REQUIRE_PHYSICAL_DATA=false     # Require physical properties (boolean)

# Data Source Priority
ANEOS_DATA_SOURCES_PRIORITY=SBDB,NEODyS,MPC,Horizons  # Comma-separated list
```

#### Anomaly Detection Thresholds

```bash
# Orbital Element Thresholds
ANEOS_THRESHOLD_ECCENTRICITY=0.8       # High eccentricity threshold
ANEOS_THRESHOLD_INCLINATION=45.0       # High inclination threshold (degrees)
ANEOS_THRESHOLD_SEMI_MAJOR_AXIS_MIN=0.8  # Minimum semi-major axis (AU)
ANEOS_THRESHOLD_SEMI_MAJOR_AXIS_MAX=4.0  # Maximum semi-major axis (AU)

# Velocity and Dynamics
ANEOS_THRESHOLD_VELOCITY_SHIFT=5.0     # Velocity anomaly threshold (km/s)
ANEOS_THRESHOLD_ACCELERATION=0.0005    # Non-gravitational acceleration (km/s²)
ANEOS_THRESHOLD_TEMPORAL_INERTIA=100.0 # Temporal pattern threshold (days)

# Physical Properties  
ANEOS_THRESHOLD_DIAMETER_MIN=0.1       # Minimum diameter (km)
ANEOS_THRESHOLD_DIAMETER_MAX=10.0      # Maximum diameter (km)
ANEOS_THRESHOLD_ALBEDO_MIN=0.05        # Minimum albedo
ANEOS_THRESHOLD_ALBEDO_MAX=0.5         # Maximum normal albedo
ANEOS_THRESHOLD_ALBEDO_ARTIFICIAL=0.6  # High albedo threshold (artificial)

# Geographic Clustering
ANEOS_THRESHOLD_GEO_EPS=5              # Geographic clustering epsilon (km)
ANEOS_THRESHOLD_GEO_MIN_SAMPLES=2      # Minimum samples for cluster
ANEOS_THRESHOLD_GEO_MIN_CLUSTERS=2     # Minimum clusters for anomaly
ANEOS_THRESHOLD_MIN_SUBPOINTS=2        # Minimum subpoints required

# Temporal Analysis
ANEOS_THRESHOLD_OBSERVATION_GAP_MULTIPLIER=3  # Expected vs actual observation gap
```

#### Indicator Weights

```bash
# Scientific Indicator Weights (relative importance)
ANEOS_WEIGHT_ORBITAL_MECHANICS=1.5     # Orbital anomaly weight
ANEOS_WEIGHT_VELOCITY_SHIFTS=2.0       # Velocity pattern weight  
ANEOS_WEIGHT_CLOSE_APPROACH_REGULARITY=2.0  # Temporal pattern weight
ANEOS_WEIGHT_PURPOSE_DRIVEN=2.0        # Purpose-driven behavior weight
ANEOS_WEIGHT_PHYSICAL_ANOMALIES=1.0    # Physical characteristic weight
ANEOS_WEIGHT_TEMPORAL_ANOMALIES=1.0    # Temporal anomaly weight
ANEOS_WEIGHT_GEOGRAPHIC_CLUSTERING=1.0 # Geographic pattern weight
ANEOS_WEIGHT_ACCELERATION_ANOMALIES=2.0 # Acceleration anomaly weight
ANEOS_WEIGHT_SPECTRAL_ANOMALIES=1.5    # Spectral signature weight
ANEOS_WEIGHT_OBSERVATION_HISTORY=1.0   # Observation pattern weight
ANEOS_WEIGHT_DETECTION_HISTORY=1.0     # Detection circumstances weight
```

### Machine Learning Configuration

#### Model Training Parameters

```bash
# General ML Settings
ANEOS_ML_ENABLED=true                  # Enable ML features (boolean)
ANEOS_ML_MODEL_PATH=models/            # ML model storage path
ANEOS_ML_FEATURE_CACHE_SIZE=10000     # Feature cache size
ANEOS_ML_PREDICTION_CACHE_TTL=1800    # Prediction cache TTL (seconds)

# Training Configuration
ANEOS_ML_TRAINING_BATCH_SIZE=32       # Training batch size
ANEOS_ML_VALIDATION_SPLIT=0.2         # Validation data ratio
ANEOS_ML_RANDOM_SEED=42               # Random seed for reproducibility
ANEOS_ML_MAX_TRAINING_TIME=7200       # Maximum training time (seconds)

# Model Selection
ANEOS_ML_DEFAULT_MODEL=ensemble       # Default model type
ANEOS_ML_ENSEMBLE_MODELS=isolation_forest,one_class_svm,autoencoder  # Ensemble components
ANEOS_ML_MODEL_SELECTION_METRIC=f1_score  # Model selection metric

# Feature Engineering
ANEOS_ML_FEATURE_SELECTION=true       # Enable feature selection
ANEOS_ML_FEATURE_SCALING=standard     # Feature scaling method (standard, minmax, robust)
ANEOS_ML_HANDLE_MISSING=impute        # Missing value handling (impute, drop, flag)
ANEOS_ML_OUTLIER_TREATMENT=clip       # Outlier treatment (clip, transform, flag)
```

#### Individual Model Parameters

```bash
# Isolation Forest
ANEOS_ML_IF_N_ESTIMATORS=200          # Number of trees
ANEOS_ML_IF_CONTAMINATION=0.1         # Expected contamination ratio
ANEOS_ML_IF_MAX_SAMPLES=auto          # Samples per tree
ANEOS_ML_IF_MAX_FEATURES=1.0          # Features per tree ratio

# One-Class SVM
ANEOS_ML_SVM_KERNEL=rbf               # Kernel type (rbf, linear, poly, sigmoid)
ANEOS_ML_SVM_NU=0.05                  # Nu parameter (expected anomaly fraction)
ANEOS_ML_SVM_GAMMA=scale              # Gamma parameter (scale, auto, float)
ANEOS_ML_SVM_DEGREE=3                 # Polynomial degree (for poly kernel)

# Autoencoder  
ANEOS_ML_AE_ENCODER_LAYERS=128,64,32  # Encoder architecture (comma-separated)
ANEOS_ML_AE_DECODER_LAYERS=32,64,128  # Decoder architecture
ANEOS_ML_AE_LATENT_DIM=16             # Latent space dimension
ANEOS_ML_AE_LEARNING_RATE=0.001       # Learning rate
ANEOS_ML_AE_EPOCHS=100                # Maximum training epochs
ANEOS_ML_AE_BATCH_SIZE=32             # Training batch size
ANEOS_ML_AE_DROPOUT_RATE=0.1          # Dropout rate
ANEOS_ML_AE_PATIENCE=10               # Early stopping patience
```

### Performance and Resource Configuration

#### Resource Limits

```bash
# Memory Management
ANEOS_MAX_MEMORY_MB=4096              # Maximum memory usage (MB, 0 = unlimited)
ANEOS_MEMORY_WARNING_THRESHOLD=0.8    # Memory usage warning threshold (0-1)
ANEOS_GARBAGE_COLLECTION_FREQUENCY=100 # GC frequency (requests)

# CPU and Threading
ANEOS_MAX_CPU_PERCENT=80.0            # Maximum CPU usage percentage
ANEOS_THREAD_POOL_SIZE=20             # Thread pool size
ANEOS_ASYNC_CONCURRENCY=50            # Async operation concurrency

# Network and I/O
ANEOS_MAX_CONNECTIONS=1000            # Maximum concurrent connections
ANEOS_CONNECTION_TIMEOUT=30           # Connection timeout (seconds)
ANEOS_KEEP_ALIVE_TIMEOUT=5            # Keep-alive timeout (seconds)
ANEOS_MAX_REQUEST_SIZE=100            # Maximum request size (MB)

# File System
ANEOS_MAX_FILE_SIZE_MB=100            # Maximum file size for uploads
ANEOS_TEMP_FILE_TTL=3600             # Temporary file TTL (seconds)
ANEOS_LOG_FILE_SIZE_MB=100           # Log file rotation size
ANEOS_LOG_BACKUP_COUNT=10            # Number of log backup files
```

#### Performance Optimization

```bash
# Caching Strategy
ANEOS_CACHE_STRATEGY=lru             # Cache eviction strategy (lru, lfu, fifo)
ANEOS_CACHE_COMPRESSION=true         # Enable cache compression
ANEOS_CACHE_ENCRYPTION=false         # Enable cache encryption

# Database Optimization
ANEOS_DB_QUERY_TIMEOUT=30            # Database query timeout (seconds)
ANEOS_DB_SLOW_QUERY_THRESHOLD=1.0    # Slow query logging threshold (seconds)
ANEOS_DB_CONNECTION_RETRY_ATTEMPTS=3 # Connection retry attempts
ANEOS_DB_ENABLE_QUERY_CACHE=true     # Enable query result caching

# HTTP Optimization
ANEOS_HTTP_KEEPALIVE=true            # Enable HTTP keep-alive
ANEOS_HTTP_COMPRESSION=true          # Enable response compression
ANEOS_HTTP_CACHE_CONTROL=public,max-age=3600  # Cache control header
```

---

## Configuration Files

### Environment File (.env)

Create a `.env` file in the project root:

```bash
# .env file example

# Application Configuration
ANEOS_ENV=production
ANEOS_DEBUG=false
ANEOS_SECRET_KEY=your-super-secret-key-change-this-in-production
ANEOS_HOST=0.0.0.0
ANEOS_PORT=8000
ANEOS_WORKERS=4

# Database Configuration
ANEOS_DATABASE_URL=postgresql://aneos:secure_password@localhost:5432/aneos
ANEOS_DATABASE_POOL_SIZE=20

# Redis Configuration
ANEOS_REDIS_URL=redis://:redis_password@localhost:6379/0

# External APIs
ANEOS_REQUEST_TIMEOUT=30
ANEOS_MAX_RETRIES=5

# Analysis Configuration
ANEOS_MAX_WORKERS=20
ANEOS_BATCH_SIZE=100

# Machine Learning
ANEOS_ML_ENABLED=true
ANEOS_ML_DEFAULT_MODEL=ensemble

# Logging
ANEOS_LOG_LEVEL=INFO
ANEOS_LOG_FILE=logs/aneos.log
```

### YAML Configuration (config.yaml)

```yaml
# config.yaml - Structured configuration file

aneos:
  # Application settings
  app:
    env: production
    debug: false
    secret_key: ${ANEOS_SECRET_KEY}
    host: 0.0.0.0
    port: 8000
    workers: 4

  # Database configuration
  database:
    url: ${ANEOS_DATABASE_URL}
    pool:
      size: 20
      overflow: 10
      timeout: 30
      recycle: 3600
    options:
      echo: false
      echo_pool: false

  # External API configuration
  apis:
    sbdb:
      url: https://ssd-api.jpl.nasa.gov/sbdb.api
      timeout: 10
      retries: 3
    
    cad:
      url: https://ssd-api.jpl.nasa.gov/cad.api  
      timeout: 10
      retries: 3
    
    horizons:
      url: https://ssd.jpl.nasa.gov/api/horizons.api
      timeout: 15
      retries: 3

  # Analysis configuration
  analysis:
    processing:
      max_workers: 20
      batch_size: 100
      timeout: 300
    
    thresholds:
      eccentricity: 0.8
      inclination: 45.0
      velocity_shift: 5.0
      temporal_inertia: 100.0
    
    weights:
      orbital_mechanics: 1.5
      velocity_shifts: 2.0
      close_approach_regularity: 2.0
      geographic_clustering: 1.0

  # Machine Learning configuration  
  ml:
    enabled: true
    model_path: models/
    default_model: ensemble
    
    training:
      batch_size: 32
      validation_split: 0.2
      random_seed: 42
    
    models:
      isolation_forest:
        n_estimators: 200
        contamination: 0.1
      
      one_class_svm:
        kernel: rbf
        nu: 0.05
        gamma: scale
      
      autoencoder:
        encoder_layers: [128, 64, 32]
        decoder_layers: [32, 64, 128] 
        latent_dim: 16
        learning_rate: 0.001
        epochs: 100

  # Logging configuration
  logging:
    level: INFO
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file: logs/aneos.log
    max_size_mb: 100
    backup_count: 10
    
  # Monitoring configuration
  monitoring:
    enabled: true
    metrics_port: 9090
    health_check_interval: 30
    
  # Security configuration
  security:
    cors_enabled: false
    cors_origins: []
    rate_limiting:
      enabled: true
      requests_per_minute: 100
      burst_size: 200
```

### JSON Configuration (config.json)

```json
{
  "aneos": {
    "app": {
      "env": "production",
      "debug": false,
      "host": "0.0.0.0",
      "port": 8000,
      "workers": 4
    },
    "database": {
      "url": "${ANEOS_DATABASE_URL}",
      "pool": {
        "size": 20,
        "overflow": 10,
        "timeout": 30
      }
    },
    "analysis": {
      "processing": {
        "max_workers": 20,
        "batch_size": 100
      },
      "thresholds": {
        "eccentricity": 0.8,
        "inclination": 45.0,
        "velocity_shift": 5.0
      },
      "weights": {
        "orbital_mechanics": 1.5,
        "velocity_shifts": 2.0,
        "close_approach_regularity": 2.0
      }
    },
    "ml": {
      "enabled": true,
      "default_model": "ensemble",
      "models": {
        "isolation_forest": {
          "n_estimators": 200,
          "contamination": 0.1
        }
      }
    }
  }
}
```

---

## Database Configuration

### SQLite Configuration (Development)

```bash
# Basic SQLite setup
ANEOS_DATABASE_URL=sqlite:///./aneos.db

# SQLite-specific optimizations
ANEOS_SQLITE_PRAGMA_JOURNAL_MODE=WAL      # Write-Ahead Logging
ANEOS_SQLITE_PRAGMA_SYNCHRONOUS=NORMAL    # Sync mode
ANEOS_SQLITE_PRAGMA_CACHE_SIZE=10000      # Cache size (pages)
ANEOS_SQLITE_PRAGMA_TEMP_STORE=memory     # Temp storage location
ANEOS_SQLITE_PRAGMA_FOREIGN_KEYS=ON       # Enable foreign keys
```

### PostgreSQL Configuration (Production)

```bash
# Connection Configuration
ANEOS_DATABASE_URL=postgresql://username:password@hostname:port/database
ANEOS_DATABASE_POOL_SIZE=20               # Connection pool size
ANEOS_DATABASE_POOL_OVERFLOW=10           # Additional connections
ANEOS_DATABASE_POOL_TIMEOUT=30            # Connection timeout
ANEOS_DATABASE_POOL_RECYCLE=3600          # Connection recycle time

# Performance Settings
ANEOS_DATABASE_STATEMENT_TIMEOUT=30000    # Statement timeout (ms)
ANEOS_DATABASE_LOCK_TIMEOUT=10000         # Lock timeout (ms)
ANEOS_DATABASE_IDLE_IN_TRANSACTION_SESSION_TIMEOUT=300000  # Idle timeout (ms)

# SSL Configuration
ANEOS_DATABASE_SSL_MODE=require           # SSL mode (disable, allow, prefer, require, verify-ca, verify-full)
ANEOS_DATABASE_SSL_CERT_FILE=client.crt  # Client certificate file
ANEOS_DATABASE_SSL_KEY_FILE=client.key   # Client key file
ANEOS_DATABASE_SSL_ROOT_CERT_FILE=ca.crt # CA certificate file

# PostgreSQL-specific Settings
ANEOS_PG_APPLICATION_NAME=aNEOS           # Application name for pg_stat_activity
ANEOS_PG_SEARCH_PATH=public,aneos         # Schema search path
ANEOS_PG_TIME_ZONE=UTC                    # Time zone setting
```

### Database Migration Configuration

```bash
# Alembic Migration Settings
ANEOS_ALEMBIC_CONFIG=alembic.ini          # Alembic config file
ANEOS_ALEMBIC_SCRIPT_LOCATION=migrations/  # Migration scripts directory
ANEOS_ALEMBIC_AUTO_GENERATE=false         # Auto-generate migrations
ANEOS_ALEMBIC_COMPARE_TYPE=true           # Compare column types
ANEOS_ALEMBIC_COMPARE_SERVER_DEFAULT=true # Compare server defaults

# Backup Configuration
ANEOS_DB_BACKUP_ENABLED=true              # Enable automatic backups
ANEOS_DB_BACKUP_SCHEDULE=0 2 * * *        # Backup schedule (cron format)
ANEOS_DB_BACKUP_RETENTION_DAYS=30         # Backup retention period
ANEOS_DB_BACKUP_LOCATION=backups/         # Backup storage location
ANEOS_DB_BACKUP_COMPRESSION=gzip          # Backup compression (none, gzip, bzip2)
```

---

## API Configuration

### FastAPI Configuration

```bash
# Server Configuration
ANEOS_TITLE=aNEOS API                     # API title
ANEOS_DESCRIPTION=Advanced Near Earth Object detection System API
ANEOS_VERSION=1.0.0                       # API version
ANEOS_OPENAPI_URL=/openapi.json          # OpenAPI schema URL
ANEOS_DOCS_URL=/docs                      # Swagger UI URL
ANEOS_REDOC_URL=/redoc                    # ReDoc URL

# Request/Response Configuration
ANEOS_MAX_REQUEST_SIZE=100                # Maximum request size (MB)
ANEOS_MAX_RESPONSE_SIZE=50                # Maximum response size (MB)
ANEOS_REQUEST_TIMEOUT=30                  # Request timeout (seconds)
ANEOS_RESPONSE_TIMEOUT=60                 # Response timeout (seconds)

# CORS Configuration
ANEOS_CORS_ENABLED=false                  # Enable CORS (boolean)
ANEOS_CORS_ORIGINS=*                      # Allowed origins (comma-separated or *)
ANEOS_CORS_METHODS=GET,POST,PUT,DELETE    # Allowed methods
ANEOS_CORS_HEADERS=*                      # Allowed headers
ANEOS_CORS_CREDENTIALS=false              # Allow credentials

# Rate Limiting
ANEOS_RATE_LIMIT_ENABLED=true             # Enable rate limiting
ANEOS_RATE_LIMIT_REQUESTS_PER_MINUTE=100  # Requests per minute per IP
ANEOS_RATE_LIMIT_BURST_SIZE=200           # Burst allowance
ANEOS_RATE_LIMIT_STORAGE=redis            # Rate limit storage (memory, redis)
```

### API Authentication

```bash
# Authentication Configuration  
ANEOS_AUTH_ENABLED=true                   # Enable authentication
ANEOS_AUTH_METHOD=api_key                 # Authentication method (api_key, jwt, oauth2)
ANEOS_AUTH_HEADER_NAME=Authorization      # Auth header name
ANEOS_AUTH_SCHEME=Bearer                  # Auth scheme

# API Key Configuration
ANEOS_API_KEY_LENGTH=32                   # API key length
ANEOS_API_KEY_EXPIRY_DAYS=90             # API key expiry (0 = no expiry)
ANEOS_API_KEY_PREFIX=aneos_              # API key prefix

# JWT Configuration  
ANEOS_JWT_ALGORITHM=HS256                 # JWT algorithm
ANEOS_JWT_EXPIRY_MINUTES=60              # JWT token expiry
ANEOS_JWT_REFRESH_EXPIRY_DAYS=7          # Refresh token expiry
ANEOS_JWT_ISSUER=aneos                   # JWT issuer

# OAuth2 Configuration
ANEOS_OAUTH2_PROVIDER=                   # OAuth2 provider URL
ANEOS_OAUTH2_CLIENT_ID=                  # OAuth2 client ID
ANEOS_OAUTH2_CLIENT_SECRET=              # OAuth2 client secret
ANEOS_OAUTH2_SCOPE=openid,profile,email # OAuth2 scopes
```

### API Versioning

```bash
# Versioning Configuration
ANEOS_API_VERSION_SCHEME=path            # Versioning scheme (path, header, query)
ANEOS_API_DEFAULT_VERSION=v1             # Default API version
ANEOS_API_SUPPORTED_VERSIONS=v1,v2       # Supported versions (comma-separated)
ANEOS_API_VERSION_DEPRECATION_NOTICE=true # Show deprecation notices

# Backwards Compatibility
ANEOS_API_STRICT_VERSION_CHECKING=false  # Strict version checking
ANEOS_API_VERSION_FALLBACK=true          # Fallback to default version
```

---

## Analysis Configuration

### Processing Configuration

```bash
# Parallel Processing
ANEOS_ANALYSIS_PARALLEL=true             # Enable parallel processing
ANEOS_ANALYSIS_MAX_WORKERS=10            # Maximum worker processes
ANEOS_ANALYSIS_QUEUE_SIZE=1000           # Analysis queue size
ANEOS_ANALYSIS_TIMEOUT=300               # Analysis timeout per object (seconds)

# Batch Processing
ANEOS_BATCH_PROCESSING_ENABLED=true      # Enable batch processing
ANEOS_BATCH_SIZE=100                     # Objects per batch
ANEOS_BATCH_MAX_SIZE=1000                # Maximum batch size
ANEOS_BATCH_TIMEOUT=3600                 # Batch processing timeout (seconds)

# Memory Management
ANEOS_ANALYSIS_MEMORY_LIMIT=4096         # Memory limit per analysis (MB)
ANEOS_ANALYSIS_TEMP_DIR=/tmp/aneos       # Temporary directory for analysis
ANEOS_ANALYSIS_CLEANUP_TEMP_FILES=true   # Clean up temporary files
```

### Data Source Configuration

```bash
# Data Source Priorities
ANEOS_DATA_SOURCES_PRIMARY=SBDB          # Primary data source
ANEOS_DATA_SOURCES_FALLBACK=NEODyS,MPC   # Fallback sources (comma-separated)
ANEOS_DATA_SOURCES_TIMEOUT=10            # Data source timeout (seconds)
ANEOS_DATA_SOURCES_RETRY_ATTEMPTS=3      # Retry attempts per source

# Data Validation
ANEOS_VALIDATE_INPUT_DATA=true           # Validate input data
ANEOS_REQUIRE_MINIMUM_OBSERVATIONS=5     # Minimum observations required
ANEOS_REQUIRE_ORBITAL_ELEMENTS=true      # Require orbital elements
ANEOS_REQUIRE_PHYSICAL_PROPERTIES=false  # Require physical properties
ANEOS_MAX_DATA_AGE_HOURS=24             # Maximum data age (hours)

# Data Quality Filters
ANEOS_FILTER_INCOMPLETE_DATA=true        # Filter incomplete data
ANEOS_FILTER_UNRELIABLE_DATA=true       # Filter unreliable data
ANEOS_DATA_QUALITY_THRESHOLD=0.7         # Data quality threshold (0-1)
```

### Scientific Indicators Configuration

```bash
# Indicator Processing
ANEOS_INDICATORS_PARALLEL=true           # Process indicators in parallel
ANEOS_INDICATORS_TIMEOUT=60              # Indicator processing timeout (seconds)
ANEOS_INDICATORS_CACHE_RESULTS=true      # Cache indicator results
ANEOS_INDICATORS_CACHE_TTL=3600          # Indicator cache TTL (seconds)

# Indicator Selection
ANEOS_INDICATORS_ENABLED=all             # Enabled indicators (all, orbital, velocity, temporal, geographic, physical)
ANEOS_INDICATORS_DISABLED=               # Disabled indicators (comma-separated)
ANEOS_INDICATORS_CUSTOM_WEIGHTS=false    # Use custom indicator weights

# Statistical Analysis
ANEOS_STATISTICAL_SIGNIFICANCE_LEVEL=0.05 # Statistical significance level
ANEOS_STATISTICAL_CONFIDENCE_INTERVAL=0.95 # Confidence interval
ANEOS_STATISTICAL_BOOTSTRAP_SAMPLES=1000  # Bootstrap sample size
ANEOS_STATISTICAL_MONTE_CARLO_ITERATIONS=10000 # Monte Carlo iterations
```

---

## Machine Learning Configuration

### Model Management

```bash
# Model Storage
ANEOS_ML_MODEL_STORAGE=filesystem        # Model storage (filesystem, s3, database)
ANEOS_ML_MODEL_VERSIONING=true          # Enable model versioning
ANEOS_ML_MODEL_AUTO_BACKUP=true         # Automatic model backup
ANEOS_ML_MODEL_BACKUP_COUNT=5           # Number of model backups to keep

# Model Loading
ANEOS_ML_MODEL_LAZY_LOADING=true        # Lazy load models
ANEOS_ML_MODEL_PRELOAD_DEFAULT=true     # Preload default model
ANEOS_ML_MODEL_CACHE_SIZE=3             # Number of models to keep in memory
ANEOS_ML_MODEL_LOAD_TIMEOUT=60          # Model loading timeout (seconds)

# Model Validation
ANEOS_ML_VALIDATE_MODELS_ON_LOAD=true   # Validate models when loading
ANEOS_ML_MODEL_CHECKSUM_VERIFICATION=true # Verify model checksums
ANEOS_ML_MODEL_COMPATIBILITY_CHECK=true  # Check model compatibility
```

### Training Configuration

```bash
# Training Process
ANEOS_ML_TRAINING_ENABLED=true          # Enable training functionality
ANEOS_ML_TRAINING_PARALLEL=false        # Parallel training (experimental)
ANEOS_ML_TRAINING_GPU_ENABLED=auto      # GPU training (auto, true, false)
ANEOS_ML_TRAINING_MEMORY_LIMIT=8192     # Training memory limit (MB)

# Data Preparation
ANEOS_ML_TRAINING_DATA_SPLIT=0.8        # Training data split ratio
ANEOS_ML_VALIDATION_DATA_SPLIT=0.2      # Validation data split ratio
ANEOS_ML_TEST_DATA_SPLIT=0.0            # Test data split ratio (if > 0)
ANEOS_ML_DATA_AUGMENTATION=false        # Enable data augmentation
ANEOS_ML_DATA_NORMALIZATION=standard    # Data normalization (standard, minmax, robust, none)

# Training Parameters
ANEOS_ML_MAX_TRAINING_EPOCHS=1000       # Maximum training epochs
ANEOS_ML_EARLY_STOPPING_PATIENCE=50     # Early stopping patience
ANEOS_ML_LEARNING_RATE_SCHEDULE=plateau # Learning rate schedule (plateau, exponential, cosine)
ANEOS_ML_CHECKPOINT_FREQUENCY=10        # Model checkpoint frequency (epochs)

# Hyperparameter Optimization
ANEOS_ML_HYPEROPT_ENABLED=false         # Enable hyperparameter optimization
ANEOS_ML_HYPEROPT_TRIALS=100           # Number of optimization trials
ANEOS_ML_HYPEROPT_ALGORITHM=tpe        # Optimization algorithm (tpe, random, adaptive)
```

### Inference Configuration

```bash
# Inference Processing
ANEOS_ML_INFERENCE_BATCH_SIZE=32        # Inference batch size
ANEOS_ML_INFERENCE_TIMEOUT=10           # Inference timeout (seconds)
ANEOS_ML_INFERENCE_CACHE_RESULTS=true   # Cache inference results
ANEOS_ML_INFERENCE_CACHE_TTL=1800       # Inference cache TTL (seconds)

# Prediction Thresholds
ANEOS_ML_ANOMALY_THRESHOLD=0.5          # Binary classification threshold
ANEOS_ML_CONFIDENCE_THRESHOLD=0.7       # Minimum confidence threshold
ANEOS_ML_UNCERTAINTY_QUANTIFICATION=true # Enable uncertainty quantification

# Model Ensemble
ANEOS_ML_ENSEMBLE_VOTING_STRATEGY=weighted # Ensemble voting (weighted, majority, soft)
ANEOS_ML_ENSEMBLE_WEIGHT_OPTIMIZATION=true # Optimize ensemble weights
ANEOS_ML_ENSEMBLE_DIVERSITY_PENALTY=0.1    # Diversity penalty weight
```

---

## Monitoring and Logging

### Logging Configuration

```bash
# General Logging
ANEOS_LOG_LEVEL=INFO                    # Logging level (CRITICAL, ERROR, WARNING, INFO, DEBUG)
ANEOS_LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
ANEOS_LOG_DATE_FORMAT=%Y-%m-%d %H:%M:%S # Date format

# File Logging
ANEOS_LOG_TO_FILE=true                  # Enable file logging
ANEOS_LOG_FILE=logs/aneos.log           # Log file path
ANEOS_LOG_FILE_MAX_SIZE=100             # Max log file size (MB)
ANEOS_LOG_FILE_BACKUP_COUNT=10          # Number of backup files
ANEOS_LOG_FILE_ROTATION=time            # Rotation method (size, time)

# Console Logging
ANEOS_LOG_TO_CONSOLE=true               # Enable console logging
ANEOS_LOG_CONSOLE_LEVEL=INFO            # Console logging level
ANEOS_LOG_CONSOLE_FORMAT=%(levelname)s: %(message)s

# Structured Logging
ANEOS_LOG_STRUCTURED=false              # Enable structured (JSON) logging
ANEOS_LOG_INCLUDE_EXTRA_FIELDS=true     # Include extra fields in logs
ANEOS_LOG_CORRELATION_ID=true           # Include correlation IDs

# Component-specific Logging
ANEOS_LOG_DATABASE_QUERIES=false        # Log database queries
ANEOS_LOG_API_REQUESTS=true             # Log API requests
ANEOS_LOG_ANALYSIS_DETAILS=false        # Log detailed analysis information
ANEOS_LOG_ML_OPERATIONS=false           # Log ML operations

# External Logging
ANEOS_LOG_SYSLOG_ENABLED=false          # Enable syslog
ANEOS_LOG_SYSLOG_FACILITY=local0        # Syslog facility
ANEOS_LOG_ELASTICSEARCH_ENABLED=false   # Enable Elasticsearch logging
ANEOS_LOG_ELASTICSEARCH_URL=http://localhost:9200
```

### Metrics and Monitoring

```bash
# Metrics Collection
ANEOS_METRICS_ENABLED=true              # Enable metrics collection
ANEOS_METRICS_PORT=9090                 # Metrics server port
ANEOS_METRICS_PATH=/metrics             # Metrics endpoint path
ANEOS_METRICS_UPDATE_INTERVAL=30        # Metrics update interval (seconds)

# Prometheus Configuration
ANEOS_PROMETHEUS_METRICS=true           # Enable Prometheus metrics
ANEOS_PROMETHEUS_NAMESPACE=aneos        # Metrics namespace
ANEOS_PROMETHEUS_SUBSYSTEM=             # Metrics subsystem
ANEOS_PROMETHEUS_LABELS=environment,version # Default labels

# Health Checks
ANEOS_HEALTH_CHECK_ENABLED=true         # Enable health checks
ANEOS_HEALTH_CHECK_PATH=/health         # Health check endpoint
ANEOS_HEALTH_CHECK_INTERVAL=30          # Health check interval (seconds)
ANEOS_HEALTH_CHECK_TIMEOUT=10           # Health check timeout (seconds)

# Performance Monitoring
ANEOS_PERFORMANCE_MONITORING=true       # Enable performance monitoring
ANEOS_PERFORMANCE_SAMPLE_RATE=0.1       # Performance sampling rate
ANEOS_PERFORMANCE_SLOW_THRESHOLD=1.0    # Slow operation threshold (seconds)

# Alerting
ANEOS_ALERTING_ENABLED=false            # Enable alerting
ANEOS_ALERTING_WEBHOOK_URL=             # Webhook URL for alerts
ANEOS_ALERTING_EMAIL_ENABLED=false      # Enable email alerts
ANEOS_ALERTING_SLACK_ENABLED=false      # Enable Slack alerts
```

---

## Security Configuration

### Application Security

```bash
# Security Headers
ANEOS_SECURITY_HSTS_ENABLED=true        # Enable HTTP Strict Transport Security
ANEOS_SECURITY_HSTS_MAX_AGE=31536000    # HSTS max age (seconds)
ANEOS_SECURITY_CSP_ENABLED=true         # Enable Content Security Policy
ANEOS_SECURITY_CSP_POLICY=default-src 'self' # CSP policy
ANEOS_SECURITY_X_FRAME_OPTIONS=DENY     # X-Frame-Options header
ANEOS_SECURITY_X_CONTENT_TYPE_OPTIONS=nosniff # X-Content-Type-Options

# Session Security
ANEOS_SECURITY_SESSION_SECURE=true      # Secure session cookies (HTTPS only)
ANEOS_SECURITY_SESSION_HTTPONLY=true    # HTTP-only session cookies
ANEOS_SECURITY_SESSION_SAMESITE=strict  # SameSite cookie attribute

# Input Validation
ANEOS_SECURITY_INPUT_VALIDATION=strict  # Input validation level (strict, normal, loose)
ANEOS_SECURITY_SANITIZE_INPUT=true      # Sanitize user input
ANEOS_SECURITY_MAX_INPUT_LENGTH=10000   # Maximum input length

# API Security
ANEOS_SECURITY_API_RATE_LIMITING=true   # Enable API rate limiting
ANEOS_SECURITY_API_IP_WHITELIST=        # IP whitelist (comma-separated)
ANEOS_SECURITY_API_IP_BLACKLIST=        # IP blacklist (comma-separated)
ANEOS_SECURITY_API_REQUIRE_HTTPS=false  # Require HTTPS for API access
```

### Encryption Configuration

```bash
# Data Encryption
ANEOS_ENCRYPTION_ENABLED=false          # Enable data encryption at rest
ANEOS_ENCRYPTION_KEY_FILE=              # Encryption key file path
ANEOS_ENCRYPTION_ALGORITHM=AES-256-GCM  # Encryption algorithm

# SSL/TLS Configuration
ANEOS_SSL_ENABLED=false                 # Enable SSL/TLS
ANEOS_SSL_CERT_FILE=                    # SSL certificate file
ANEOS_SSL_KEY_FILE=                     # SSL private key file
ANEOS_SSL_CA_FILE=                      # SSL CA certificate file
ANEOS_SSL_VERIFY_MODE=CERT_REQUIRED     # SSL verification mode

# Password Security
ANEOS_PASSWORD_HASH_ALGORITHM=argon2    # Password hashing algorithm
ANEOS_PASSWORD_MIN_LENGTH=8             # Minimum password length
ANEOS_PASSWORD_REQUIRE_COMPLEXITY=true  # Require complex passwords
```

---

## Deployment Configuration

### Docker Configuration

```bash
# Container Configuration
ANEOS_DOCKER_IMAGE=aneos:latest         # Docker image name
ANEOS_DOCKER_REGISTRY=                  # Docker registry URL
ANEOS_DOCKER_USER=aneos                 # Container user
ANEOS_DOCKER_GROUP=aneos                # Container group
ANEOS_DOCKER_UID=1000                   # Container user ID
ANEOS_DOCKER_GID=1000                   # Container group ID

# Resource Limits
ANEOS_DOCKER_MEMORY_LIMIT=4g            # Memory limit
ANEOS_DOCKER_CPU_LIMIT=2                # CPU limit (cores)
ANEOS_DOCKER_CPU_REQUEST=1              # CPU request (cores)
ANEOS_DOCKER_MEMORY_REQUEST=2g          # Memory request

# Networking
ANEOS_DOCKER_NETWORK=aneos-network      # Docker network name
ANEOS_DOCKER_EXPOSE_PORTS=8000          # Exposed ports (comma-separated)
```

### Kubernetes Configuration

```bash
# Kubernetes Deployment
ANEOS_K8S_NAMESPACE=aneos               # Kubernetes namespace
ANEOS_K8S_DEPLOYMENT_NAME=aneos-api     # Deployment name
ANEOS_K8S_SERVICE_NAME=aneos-service    # Service name
ANEOS_K8S_INGRESS_NAME=aneos-ingress    # Ingress name

# Pod Configuration
ANEOS_K8S_REPLICAS=3                    # Number of pod replicas
ANEOS_K8S_STRATEGY=RollingUpdate        # Deployment strategy
ANEOS_K8S_MAX_UNAVAILABLE=1             # Max unavailable pods during update
ANEOS_K8S_MAX_SURGE=1                   # Max surge pods during update

# Resource Management
ANEOS_K8S_MEMORY_REQUEST=1Gi            # Memory request
ANEOS_K8S_MEMORY_LIMIT=4Gi             # Memory limit
ANEOS_K8S_CPU_REQUEST=500m              # CPU request (millicores)
ANEOS_K8S_CPU_LIMIT=2                   # CPU limit (cores)

# Health Checks
ANEOS_K8S_LIVENESS_PROBE_PATH=/health   # Liveness probe path
ANEOS_K8S_READINESS_PROBE_PATH=/ready   # Readiness probe path
ANEOS_K8S_PROBE_INITIAL_DELAY=30        # Initial delay for probes (seconds)
ANEOS_K8S_PROBE_PERIOD=10               # Probe period (seconds)

# Autoscaling
ANEOS_K8S_HPA_ENABLED=true              # Enable Horizontal Pod Autoscaler
ANEOS_K8S_HPA_MIN_REPLICAS=3            # Minimum replicas
ANEOS_K8S_HPA_MAX_REPLICAS=10           # Maximum replicas
ANEOS_K8S_HPA_CPU_TARGET=70             # Target CPU utilization (%)
ANEOS_K8S_HPA_MEMORY_TARGET=80          # Target memory utilization (%)
```

---

## Advanced Configuration

### Experimental Features

```bash
# Feature Flags
ANEOS_FEATURE_ADVANCED_ML=false         # Enable advanced ML features
ANEOS_FEATURE_REAL_TIME_ANALYSIS=false  # Enable real-time analysis
ANEOS_FEATURE_DISTRIBUTED_PROCESSING=false # Enable distributed processing
ANEOS_FEATURE_QUANTUM_COMPUTING=false   # Enable quantum computing features (future)

# Advanced Analytics
ANEOS_ANALYTICS_ENABLED=false           # Enable advanced analytics
ANEOS_ANALYTICS_PROVIDER=               # Analytics provider (google, mixpanel, custom)
ANEOS_ANALYTICS_TRACKING_ID=            # Analytics tracking ID
ANEOS_ANALYTICS_SAMPLE_RATE=1.0         # Analytics sampling rate

# Integration Features
ANEOS_INTEGRATION_WEBHOOK_ENABLED=false # Enable webhook integrations
ANEOS_INTEGRATION_SLACK_ENABLED=false   # Enable Slack integration
ANEOS_INTEGRATION_TEAMS_ENABLED=false   # Enable Microsoft Teams integration
ANEOS_INTEGRATION_DISCORD_ENABLED=false # Enable Discord integration
```

### Development and Debug Configuration

```bash
# Development Mode
ANEOS_DEV_MODE=false                    # Enable development mode
ANEOS_DEV_AUTO_RELOAD=false             # Enable auto-reload
ANEOS_DEV_HOT_RELOAD=false              # Enable hot reload (experimental)
ANEOS_DEV_PROFILING=false               # Enable profiling
ANEOS_DEV_MEMORY_PROFILING=false        # Enable memory profiling

# Debug Configuration
ANEOS_DEBUG_SQL_QUERIES=false           # Debug SQL queries
ANEOS_DEBUG_API_CALLS=false             # Debug external API calls
ANEOS_DEBUG_CACHE_OPERATIONS=false      # Debug cache operations
ANEOS_DEBUG_ML_TRAINING=false           # Debug ML training
ANEOS_DEBUG_ANALYSIS_PIPELINE=false     # Debug analysis pipeline

# Testing Configuration
ANEOS_TESTING_MODE=false                # Enable testing mode
ANEOS_TESTING_MOCK_EXTERNAL_APIS=false  # Mock external APIs
ANEOS_TESTING_SEED_DATABASE=false       # Seed database with test data
ANEOS_TESTING_FAST_ANALYSIS=false       # Enable fast analysis mode
```

### Configuration Validation

All configuration parameters are automatically validated when aNEOS starts. The validation includes:

- **Type checking**: Ensures values match expected types
- **Range validation**: Numeric values within acceptable ranges
- **Dependency validation**: Related settings are compatible
- **Security validation**: Secure defaults and no obvious vulnerabilities
- **Performance validation**: Settings won't cause performance issues

If validation fails, aNEOS will log detailed error messages and refuse to start with invalid configuration.

---

This completes the comprehensive Configuration Reference for aNEOS. All parameters are documented with their purpose, acceptable values, and dependencies to ensure proper system configuration across all deployment scenarios.