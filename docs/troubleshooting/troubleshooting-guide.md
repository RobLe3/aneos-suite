# aNEOS Troubleshooting Guide

Comprehensive troubleshooting and problem resolution guide for aNEOS

## Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Installation Issues](#installation-issues)
3. [Runtime Errors](#runtime-errors)
4. [Database Problems](#database-problems)
5. [API Service Issues](#api-service-issues)
6. [Analysis Pipeline Errors](#analysis-pipeline-errors)
7. [ML Model Problems](#ml-model-problems)
8. [Performance Issues](#performance-issues)
9. [Docker and Deployment Issues](#docker-and-deployment-issues)
10. [Network and Connectivity Problems](#network-and-connectivity-problems)
11. [Security and Authentication Issues](#security-and-authentication-issues)
12. [Data Quality and Validation Errors](#data-quality-and-validation-errors)
13. [Log Analysis Guide](#log-analysis-guide)
14. [Emergency Procedures](#emergency-procedures)

---

## Quick Diagnostic Tools

### System Health Check

First step for any issue - run the comprehensive health check:

```bash
# Interactive menu health check
python aneos_menu.py
# → 4 (Health & Diagnostics)
# → 1 (System Health Check)

# Command line health check
python -c "from aneos_api.database import get_database_status; print(get_database_status())"
```

### Diagnostic Commands

```bash
# System status overview
python aneos_menu.py
# → 4 → 2 (Basic System Status)

# Quick installation check
python install.py --check

# Dependency verification
python aneos_menu.py
# → 3 → 7 (Dependency Check)

# Basic system tests
python aneos_menu.py  
# → 4 → 3 (Run Basic Tests)
```

### Log Locations

Important log files to check:

```bash
# Main application log
tail -f aneos.log

# API server logs (if running)
tail -f logs/api.log

# Analysis logs
tail -f dataneos/logs/enhanced_neo_poller.log

# Database logs (if available)
tail -f logs/database.log

# System logs
tail -f /var/log/syslog  # Linux
tail -f /var/log/system.log  # macOS
```

---

## Installation Issues

### Common Installation Problems

#### Problem: `ModuleNotFoundError` during startup

**Symptoms:**
```
ModuleNotFoundError: No module named 'fastapi'
ModuleNotFoundError: No module named 'sqlalchemy'
ModuleNotFoundError: No module named 'rich'
```

**Diagnosis:**
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep fastapi
pip list | grep sqlalchemy
pip list | grep rich

# Check installation status
python install.py --check
```

**Solutions:**

1. **Fix dependencies:**
   ```bash
   python install.py --fix-deps
   ```

2. **Manual installation:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Clean reinstall:**
   ```bash
   pip uninstall -y $(pip freeze)
   python install.py --full
   ```

4. **Virtual environment issues:**
   ```bash
   # Create new virtual environment
   python -m venv aneos-env
   source aneos-env/bin/activate  # Linux/Mac
   # or
   aneos-env\Scripts\activate  # Windows
   
   python install.py --full
   ```

#### Problem: Permission denied errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/usr/local/lib/python3.11/site-packages'
```

**Solutions:**

1. **Use virtual environment (recommended):**
   ```bash
   python -m venv aneos-env
   source aneos-env/bin/activate
   python install.py --full
   ```

2. **User installation:**
   ```bash
   pip install --user -r requirements.txt
   ```

3. **Fix permissions (Linux/Mac):**
   ```bash
   sudo chown -R $USER:$USER ~/.local/lib/python*/site-packages
   ```

#### Problem: SSL certificate errors during installation

**Symptoms:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**

1. **Upgrade certificates:**
   ```bash
   # macOS
   /Applications/Python\ 3.11/Install\ Certificates.command
   
   # Linux
   sudo apt-get update && sudo apt-get install ca-certificates
   
   # Alternative: Use --trusted-host
   pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```

#### Problem: Compilation errors for C extensions

**Symptoms:**
```
error: Microsoft Visual C++ 14.0 is required  # Windows
error: gcc: command not found  # Linux
```

**Solutions:**

1. **Windows:**
   ```bash
   # Install Visual Studio Build Tools
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
   # Or use conda
   conda install -c conda-forge package-name
   ```

2. **Linux:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential python3-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel
   ```

3. **macOS:**
   ```bash
   xcode-select --install
   ```

---

## Runtime Errors

### Application Startup Issues

#### Problem: Database connection failed

**Symptoms:**
```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file
sqlalchemy.exc.OperationalError: could not connect to server
```

**Diagnosis:**
```bash
# Check database file exists and permissions
ls -la aneos.db

# Test database connection
python -c "
from aneos_api.database import get_database_status
print(get_database_status())
"
```

**Solutions:**

1. **Initialize database:**
   ```bash
   python aneos_menu.py
   # → 3 (System Management)
   # → 2 (Database Management)
   ```

2. **Fix database file permissions:**
   ```bash
   chmod 664 aneos.db
   chown $USER:$USER aneos.db
   ```

3. **Reset database:**
   ```bash
   # Backup existing database
   cp aneos.db aneos.db.backup
   
   # Remove and recreate
   rm aneos.db
   python -c "from aneos_api.database import init_database; init_database()"
   ```

4. **Check disk space:**
   ```bash
   df -h .
   # Ensure sufficient disk space available
   ```

#### Problem: Port already in use

**Symptoms:**
```
OSError: [Errno 48] Address already in use
uvicorn.main:ERROR - Error binding to address
```

**Diagnosis:**
```bash
# Check what's using the port
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Check for running aNEOS processes
ps aux | grep aneos  # Linux/Mac
tasklist | findstr python  # Windows
```

**Solutions:**

1. **Kill existing process:**
   ```bash
   # Linux/Mac
   kill $(lsof -ti:8000)
   
   # Windows
   taskkill /F /PID <PID>
   ```

2. **Use different port:**
   ```bash
   python start_api.py --port 8001
   ```

3. **Find and terminate aNEOS processes:**
   ```bash
   pkill -f aneos  # Linux/Mac
   ```

#### Problem: Import errors for optional dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'sklearn'
```

**Solutions:**

1. **Install ML dependencies:**
   ```bash
   pip install torch scikit-learn
   ```

2. **Full installation:**
   ```bash
   python install.py --full
   ```

3. **Check feature availability:**
   ```bash
   python -c "
   from aneos_menu import HAS_TORCH, HAS_SKLEARN
   print(f'PyTorch: {HAS_TORCH}')
   print(f'Scikit-learn: {HAS_SKLEARN}')
   "
   ```

### Memory and Resource Issues

#### Problem: Out of memory errors

**Symptoms:**
```
MemoryError
OSError: [Errno 12] Cannot allocate memory
```

**Diagnosis:**
```bash
# Check memory usage
free -h  # Linux
vm_stat  # macOS
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory  # Windows

# Check aNEOS memory usage
ps aux | grep aneos  # Linux/Mac
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   export ANEOS_BATCH_SIZE=50
   ```

2. **Limit workers:**
   ```bash
   export ANEOS_MAX_WORKERS=2
   ```

3. **Clear cache:**
   ```bash
   python aneos_menu.py
   # → 3 → 3 (System Cleanup)
   ```

4. **Increase system memory or use swap:**
   ```bash
   # Linux - create swap file
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Database Problems

### SQLite Issues

#### Problem: Database is locked

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**

1. **Check for open connections:**
   ```bash
   lsof aneos.db  # Linux/Mac
   ```

2. **Force unlock:**
   ```bash
   # Stop all aNEOS processes
   pkill -f aneos
   
   # Remove lock files
   rm -f aneos.db-wal aneos.db-shm
   ```

3. **Database recovery:**
   ```bash
   # Backup current database
   cp aneos.db aneos.db.corrupt
   
   # Attempt recovery
   sqlite3 aneos.db ".backup aneos.db.recovered"
   mv aneos.db.recovered aneos.db
   ```

#### Problem: Database corruption

**Symptoms:**
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions:**

1. **Check database integrity:**
   ```bash
   sqlite3 aneos.db "PRAGMA integrity_check;"
   ```

2. **Repair database:**
   ```bash
   # Dump and restore
   sqlite3 aneos.db ".dump" | sqlite3 aneos_repaired.db
   mv aneos.db aneos.db.corrupt
   mv aneos_repaired.db aneos.db
   ```

3. **Restore from backup:**
   ```bash
   # If you have a backup
   cp aneos.db.backup aneos.db
   ```

### PostgreSQL Issues (Production)

#### Problem: Connection pool exhausted

**Symptoms:**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 20 overflow 0 reached
```

**Solutions:**

1. **Increase pool size:**
   ```bash
   export ANEOS_DATABASE_POOL_SIZE=50
   export ANEOS_DATABASE_POOL_OVERFLOW=20
   ```

2. **Check for connection leaks:**
   ```sql
   -- Check active connections
   SELECT pid, usename, application_name, state 
   FROM pg_stat_activity 
   WHERE datname = 'aneos';
   ```

3. **Restart application:**
   ```bash
   # Docker
   docker-compose restart aneos-api
   
   # Kubernetes
   kubectl rollout restart deployment/aneos-api -n aneos
   ```

#### Problem: Database migration failures

**Symptoms:**
```
alembic.util.exc.CommandError: Can't locate revision identified by
```

**Solutions:**

1. **Reset migration history:**
   ```bash
   # Backup database first
   pg_dump aneos > backup.sql
   
   # Reset alembic
   alembic stamp head
   ```

2. **Manual migration:**
   ```bash
   # Check current version
   alembic current
   
   # Apply specific migration
   alembic upgrade <revision_id>
   ```

---

## API Service Issues

### FastAPI Startup Problems

#### Problem: OpenAPI schema generation errors

**Symptoms:**
```
pydantic.error_wrappers.ValidationError: field required
fastapi.exceptions.RequestValidationError
```

**Solutions:**

1. **Check model definitions:**
   ```python
   # Verify Pydantic models
   python -c "
   from aneos_api.models import NEOData
   print(NEOData.schema())
   "
   ```

2. **Update dependencies:**
   ```bash
   pip install --upgrade fastapi pydantic uvicorn
   ```

#### Problem: CORS errors in browser

**Symptoms:**
```
Access to fetch at 'http://localhost:8000/api/v1/analysis' from origin 'http://localhost:3000' 
has been blocked by CORS policy
```

**Solutions:**

1. **Check CORS configuration:**
   ```python
   # In aneos_api/app.py
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Configure appropriately
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Development mode CORS:**
   ```bash
   # Start with CORS enabled
   ANEOS_CORS_ENABLED=true python start_api.py --dev
   ```

### Authentication Issues

#### Problem: API key authentication fails

**Symptoms:**
```
401 Unauthorized
detail: Invalid API key
```

**Solutions:**

1. **Generate new API key:**
   ```bash
   python aneos_menu.py
   # → 2 → 6 (Manage API Keys)
   ```

2. **Check API key format:**
   ```bash
   # API keys should be in header
   curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/api/v1/health
   ```

3. **Verify API key configuration:**
   ```python
   python -c "
   import os
   print('SECRET_KEY configured:', bool(os.environ.get('ANEOS_SECRET_KEY')))
   "
   ```

---

## Analysis Pipeline Errors

### Scientific Analysis Issues

#### Problem: NEO data retrieval fails

**Symptoms:**
```
Exception: Failed to fetch data from NASA SBDB API
requests.exceptions.ConnectTimeout: HTTPSConnectionPool
```

**Diagnosis:**
```bash
# Test external API connectivity
curl -I https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=2024AB123

# Check DNS resolution
nslookup ssd-api.jpl.nasa.gov

# Test with different timeout
export ANEOS_REQUEST_TIMEOUT=60
```

**Solutions:**

1. **Network connectivity:**
   ```bash
   # Check internet connection
   ping google.com
   
   # Check specific APIs
   curl -v https://ssd-api.jpl.nasa.gov/cad.api
   ```

2. **Proxy configuration:**
   ```bash
   # If behind corporate proxy
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **API timeout adjustment:**
   ```bash
   export ANEOS_REQUEST_TIMEOUT=30
   export ANEOS_MAX_RETRIES=5
   ```

#### Problem: Analysis indicators fail

**Symptoms:**
```
Error in indicator orbital_mechanics for 2024 AB123: division by zero
Indicator velocity_analysis failed: list index out of range
```

**Diagnosis:**
```bash
# Test specific indicator
python -c "
from aneos_core.analysis.indicators.orbital import EccentricityAnalysis
from aneos_core.config.settings import ThresholdConfig, WeightConfig
from aneos_core.analysis.indicators.base import IndicatorConfig

config = IndicatorConfig(weight=1.0, enabled=True)
indicator = EccentricityAnalysis('test', config)
print('Indicator created successfully')
"
```

**Solutions:**

1. **Check data quality:**
   ```python
   # Validate NEO data before analysis
   python -c "
   from aneos_core.data.models import NEOData
   # Check if orbital elements are complete
   print('Data validation check...')
   "
   ```

2. **Adjust thresholds:**
   ```bash
   # Modify configuration
   export ANEOS_THRESHOLD_ECCENTRICITY=0.9
   export ANEOS_THRESHOLD_INCLINATION=50.0
   ```

3. **Enable debug logging:**
   ```bash
   export ANEOS_LOG_LEVEL=DEBUG
   python aneos_menu.py
   ```

### Data Processing Errors

#### Problem: Feature extraction fails

**Symptoms:**
```
ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
KeyError: 'eccentricity'
```

**Solutions:**

1. **Data validation:**
   ```python
   # Check for missing/invalid data
   python -c "
   import numpy as np
   import pandas as pd
   
   # Your data validation code here
   data = np.array([1, 2, np.nan, 4])
   print('NaN values:', np.isnan(data).sum())
   print('Inf values:', np.isinf(data).sum())
   "
   ```

2. **Handle missing values:**
   ```bash
   # Configure feature handling
   export ANEOS_HANDLE_MISSING_VALUES=impute
   export ANEOS_OUTLIER_TREATMENT=clip
   ```

---

## ML Model Problems

### Model Training Issues

#### Problem: PyTorch CUDA errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
RuntimeError: No CUDA devices are available
```

**Solutions:**

1. **Force CPU usage:**
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Reduce batch size:**
   ```python
   # In model configuration
   config.parameters['batch_size'] = 16  # Reduce from 32
   ```

3. **Clear GPU memory:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

#### Problem: Scikit-learn version compatibility

**Symptoms:**
```
AttributeError: 'IsolationForest' object has no attribute 'decision_function'
ValueError: Unknown metric: roc_auc
```

**Solutions:**

1. **Check versions:**
   ```bash
   pip show scikit-learn
   python -c "import sklearn; print(sklearn.__version__)"
   ```

2. **Update scikit-learn:**
   ```bash
   pip install --upgrade scikit-learn
   ```

3. **Version-specific fixes:**
   ```python
   # Handle version differences
   from sklearn import __version__ as sklearn_version
   if sklearn_version < '0.22':
       # Use older API
       pass
   ```

### Model Inference Issues

#### Problem: Model loading fails

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/model.pkl'
pickle.UnpicklingError: invalid load key
```

**Solutions:**

1. **Check model files:**
   ```bash
   ls -la models/
   find . -name "*.pkl" -type f
   ```

2. **Regenerate models:**
   ```bash
   python aneos_menu.py
   # → 9 → 1 → 1 (Model Training)
   ```

3. **Model path configuration:**
   ```bash
   export ANEOS_MODEL_PATH=/path/to/models
   ```

---

## Performance Issues

### Slow Analysis Performance

#### Problem: Analysis takes too long

**Diagnosis:**
```bash
# Profile analysis performance
python -m cProfile -o analysis_profile.prof aneos.py analyze "2024 AB123"

# View profile
python -c "
import pstats
p = pstats.Stats('analysis_profile.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

**Solutions:**

1. **Increase parallelism:**
   ```bash
   export ANEOS_MAX_WORKERS=20
   export ANEOS_MAX_SUBPOINT_WORKERS=40
   ```

2. **Optimize caching:**
   ```bash
   export ANEOS_CACHE_TTL=7200  # 2 hours
   ```

3. **Database optimization:**
   ```bash
   # SQLite optimization
   sqlite3 aneos.db "PRAGMA optimize;"
   sqlite3 aneos.db "PRAGMA analysis_limit=1000;"
   ```

### High Memory Usage

#### Problem: Memory usage grows over time

**Diagnosis:**
```bash
# Monitor memory usage
python -c "
import psutil
import os

process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'CPU: {process.cpu_percent()}%')
"
```

**Solutions:**

1. **Memory profiling:**
   ```bash
   pip install memory-profiler
   python -m memory_profiler aneos.py
   ```

2. **Garbage collection:**
   ```python
   import gc
   gc.collect()
   ```

3. **Reduce cache size:**
   ```bash
   export ANEOS_CACHE_SIZE=100  # Reduce cache entries
   ```

---

## Docker and Deployment Issues

### Container Build Problems

#### Problem: Docker build fails

**Symptoms:**
```
failed to solve: executor failed running [/bin/sh -c pip install -r requirements.txt]
```

**Solutions:**

1. **Check Dockerfile syntax:**
   ```bash
   docker build --no-cache -t aneos:test .
   ```

2. **Build with verbose output:**
   ```bash
   docker build --progress=plain -t aneos:test .
   ```

3. **Multi-platform build issues:**
   ```bash
   docker buildx build --platform linux/amd64 -t aneos:test .
   ```

### Container Runtime Issues

#### Problem: Container exits immediately

**Diagnosis:**
```bash
# Check container logs
docker logs aneos-container

# Run interactively
docker run -it aneos:latest /bin/bash
```

**Solutions:**

1. **Fix entrypoint:**
   ```dockerfile
   # Make sure script is executable
   RUN chmod +x /app/entrypoint.sh
   ```

2. **Check user permissions:**
   ```dockerfile
   # Ensure proper ownership
   RUN chown -R aneos:aneos /app
   USER aneos
   ```

### Docker Compose Issues

#### Problem: Services won't start

**Symptoms:**
```
ERROR: Service 'aneos-api' failed to build
ERROR: for postgres  Cannot start service postgres: driver failed programming external connectivity
```

**Solutions:**

1. **Check port conflicts:**
   ```bash
   docker-compose ps
   netstat -tulpn | grep :5432
   ```

2. **Check Docker daemon:**
   ```bash
   docker info
   systemctl status docker  # Linux
   ```

3. **Rebuild services:**
   ```bash
   docker-compose build --no-cache
   docker-compose up --force-recreate
   ```

---

## Network and Connectivity Problems

### External API Issues

#### Problem: NASA API timeouts

**Symptoms:**
```
requests.exceptions.ReadTimeout: HTTPSConnectionPool('ssd-api.jpl.nasa.gov', port=443): Read timed out.
```

**Solutions:**

1. **Increase timeouts:**
   ```bash
   export ANEOS_REQUEST_TIMEOUT=60
   export ANEOS_MAX_RETRIES=5
   ```

2. **Check API status:**
   ```bash
   # Test NASA APIs
   curl -v "https://ssd-api.jpl.nasa.gov/cad.api?date-min=2024-01-01&date-max=2024-12-31"
   ```

3. **Implement retry logic:**
   ```python
   # Exponential backoff already implemented
   # Adjust retry parameters if needed
   ```

### Firewall and Security Issues

#### Problem: Connection blocked by firewall

**Solutions:**

1. **Check firewall rules:**
   ```bash
   # Linux
   sudo iptables -L
   sudo ufw status
   
   # Windows
   netsh advfirewall show allprofiles
   ```

2. **Open required ports:**
   ```bash
   # UFW (Ubuntu)
   sudo ufw allow 8000
   
   # iptables
   sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
   ```

---

## Security and Authentication Issues

### SSL/TLS Problems

#### Problem: Certificate verification fails

**Symptoms:**
```
requests.exceptions.SSLError: HTTPSConnectionPool
certificate verify failed: unable to get local issuer certificate
```

**Solutions:**

1. **Update certificates:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install ca-certificates
   
   # macOS
   /Applications/Python\ 3.11/Install\ Certificates.command
   ```

2. **Bypass SSL verification (development only):**
   ```bash
   export PYTHONHTTPSVERIFY=0
   # Or in code: requests.get(..., verify=False)
   ```

### Authentication Errors

#### Problem: JWT token issues

**Solutions:**

1. **Check token expiry:**
   ```python
   import jwt
   token = "your_token_here"
   decoded = jwt.decode(token, options={"verify_signature": False})
   print(decoded)
   ```

2. **Regenerate tokens:**
   ```bash
   python aneos_menu.py
   # → 2 → 6 (Manage API Keys)
   ```

---

## Data Quality and Validation Errors

### Input Data Issues

#### Problem: Invalid NEO designation format

**Symptoms:**
```
ValueError: Invalid NEO designation format: 'invalid-name'
```

**Solutions:**

1. **Check designation format:**
   ```python
   # Valid formats:
   # 2024 AB123
   # (99942) Apophis
   # 2004 MN4
   ```

2. **Data validation:**
   ```bash
   # Test with known good designation
   python aneos.py analyze "2024 AB1"
   ```

### Missing Data Handling

#### Problem: Insufficient data for analysis

**Symptoms:**
```
Warning: Insufficient orbital elements for analysis
Error: No close approach data available
```

**Solutions:**

1. **Check data sources:**
   ```bash
   # Test data source connectivity
   python -c "
   from aneos_core.data.sources.sbdb import SBDBDataSource
   source = SBDBDataSource()
   result = source.fetch_neo_data('2024 AB1')
   print(f'Data retrieved: {result is not None}')
   "
   ```

2. **Adjust data requirements:**
   ```bash
   export ANEOS_MIN_OBSERVATIONS=5
   export ANEOS_REQUIRE_PHYSICAL_DATA=false
   ```

---

## Log Analysis Guide

### Log Levels and Interpretation

#### Setting Log Levels

```bash
# Environment variable
export ANEOS_LOG_LEVEL=DEBUG

# In configuration
ANEOS_LOG_LEVEL=INFO  # ERROR, WARNING, INFO, DEBUG
```

#### Common Log Messages

**INFO Level:**
```
2024-08-06 10:15:30 - aneos.analysis - INFO - Starting analysis for 2024 AB123
2024-08-06 10:15:31 - aneos.analysis - INFO - Analysis completed: score=0.234
```

**WARNING Level:**
```
2024-08-06 10:15:30 - aneos.data - WARNING - API request failed, retrying (attempt 2/3)
2024-08-06 10:15:30 - aneos.cache - WARNING - Cache size approaching limit (90%)
```

**ERROR Level:**
```
2024-08-06 10:15:30 - aneos.database - ERROR - Database connection failed: timeout
2024-08-06 10:15:30 - aneos.analysis - ERROR - Analysis failed for 2024 AB123: invalid data
```

**DEBUG Level:**
```
2024-08-06 10:15:30 - aneos.indicators - DEBUG - Eccentricity indicator: raw_score=0.123
2024-08-06 10:15:30 - aneos.ml - DEBUG - Feature vector created: 45 features
```

### Log Analysis Commands

```bash
# Search for errors
grep -i error aneos.log

# Count error types
grep -i error aneos.log | cut -d'-' -f4 | sort | uniq -c

# Analysis performance tracking
grep "Analysis completed" aneos.log | tail -20

# API request monitoring
grep "API request" aneos.log | grep -v "200"

# Memory usage tracking
grep -i "memory" aneos.log

# Database operation monitoring
grep -i "database" aneos.log
```

### Log Rotation and Management

```bash
# Configure log rotation
cat > /etc/logrotate.d/aneos << EOF
/path/to/aneos.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 aneos aneos
}
EOF

# Manual log rotation
mv aneos.log aneos.log.1
touch aneos.log
chmod 644 aneos.log
```

---

## Emergency Procedures

### System Recovery

#### Complete System Reset

```bash
# 1. Stop all processes
pkill -f aneos
docker-compose down  # If using Docker

# 2. Backup important data
cp aneos.db aneos.db.emergency.backup
tar -czf aneos_data_backup.tar.gz dataneos/

# 3. Clean system
rm -rf __pycache__/ .pytest_cache/ cache/
rm -f *.pyc *.pyo

# 4. Reinstall
python install.py --full

# 5. Restore data
cp aneos.db.emergency.backup aneos.db

# 6. Test system
python aneos_menu.py
# → 4 → 1 (System Health Check)
```

#### Database Recovery

```bash
# 1. Stop all database connections
pkill -f aneos

# 2. Check database integrity
sqlite3 aneos.db "PRAGMA integrity_check;"

# 3. If corrupted, attempt repair
sqlite3 aneos.db ".backup aneos_temp.db"
mv aneos.db aneos_corrupted.db
mv aneos_temp.db aneos.db

# 4. If repair fails, restore from backup
cp aneos.db.backup aneos.db  # If backup exists

# 5. Last resort - reinitialize
rm aneos.db
python -c "from aneos_api.database import init_database; init_database()"
```

### Emergency Contacts and Resources

#### Getting Help

1. **System Diagnostics:**
   ```bash
   python aneos_menu.py
   # → 4 → 4 (System Diagnostics)
   ```

2. **Generate Support Bundle:**
   ```bash
   # Create comprehensive support information
   cat > support_info.txt << EOF
   System Information:
   - OS: $(uname -a)
   - Python: $(python --version)
   - aNEOS Status: $(python aneos_menu.py --status 2>&1 | head -20)
   - Disk Space: $(df -h .)
   - Memory: $(free -h 2>/dev/null || vm_stat)
   - Last 50 log lines: 
   $(tail -50 aneos.log)
   EOF
   ```

3. **Documentation References:**
   - User Guide: `docs/user-guide/user-guide.md`
   - Scientific Documentation: `docs/scientific/scientific-documentation.md`
   - Installation Guide: `docs/user-guide/installation.md`

### Prevention and Monitoring

#### Automated Health Checks

Create `health_check.sh`:

```bash
#!/bin/bash

# Health check script
LOG_FILE="health_check.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting health check..." >> $LOG_FILE

# Check system status
python -c "
from aneos_api.database import get_database_status
status = get_database_status()
print(f'Database: {\"OK\" if status.get(\"available\") else \"ERROR\"}')
" >> $LOG_FILE 2>&1

# Check API health
curl -s http://localhost:8000/health >> $LOG_FILE 2>&1

# Check disk space
df -h . | tail -1 >> $LOG_FILE

echo "[$DATE] Health check completed" >> $LOG_FILE
```

```bash
# Add to crontab for regular checks
crontab -e
# Add line: */15 * * * * /path/to/health_check.sh
```

#### Backup Automation

Create `backup.sh`:

```bash
#!/bin/bash

BACKUP_DIR="/backups/aneos"
DATE=$(date '+%Y%m%d_%H%M%S')

mkdir -p $BACKUP_DIR

# Database backup
cp aneos.db "$BACKUP_DIR/aneos_${DATE}.db"

# Data backup
tar -czf "$BACKUP_DIR/aneos_data_${DATE}.tar.gz" dataneos/ models/ logs/

# Keep last 30 days of backups
find $BACKUP_DIR -name "aneos_*" -mtime +30 -delete

echo "Backup completed: $DATE"
```

---

This completes the comprehensive Troubleshooting Guide for aNEOS. The guide provides systematic approaches to diagnosing and resolving common issues, emergency procedures, and preventive measures to maintain system health.