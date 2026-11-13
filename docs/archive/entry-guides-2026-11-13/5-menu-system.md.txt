# aNEOS Menu System Guide

## Quick Start

The aNEOS system now provides a comprehensive menu system that combines all functionality into a single, user-friendly interface.

### 🚀 Launch Options

#### 1. Interactive Menu (Recommended)
```bash
# Start the full interactive menu system
python aneos.py

# Or explicitly start the menu
python aneos.py menu
```

#### 2. Direct Commands
```bash
# Quick NEO analysis
python aneos.py analyze "2024 AB123"

# Start API server
python aneos.py api

# Start API in development mode
python aneos.py api --dev

# Start with Docker Compose
python aneos.py docker

# Check system status
python aneos.py status
```

#### 3. Legacy Scripts (Still Available)
```bash
# Original analysis script
python neos_o3high_v6.19.1.py

# API startup script
python start_api.py --dev

# Testing script
python test_phase2.py
```

---

## 📋 Menu System Overview

### Main Menu Categories

1. **🔬 Scientific Analysis**
   - Single NEO Analysis
   - Batch Processing
   - Interactive Analysis Mode
   - Results Viewer
   - Configuration Management

2. **🤖 Machine Learning**
   - Model Training
   - Real-time Predictions
   - Model Management
   - Feature Analysis
   - Performance Monitoring

3. **🌐 API Services**
   - REST API Server
   - Web Dashboard
   - Streaming Services
   - Development Mode
   - API Documentation

4. **📊 Monitoring & Diagnostics**
   - Live System Monitor
   - Alert Management
   - Performance Metrics
   - Health Checks
   - System Diagnostics

5. **⚙️ System Management**
   - Database Management
   - System Cleanup
   - Configuration
   - User Management
   - Maintenance Tasks

6. **🛠️ Development Tools**
   - Testing Suites
   - Debug Mode
   - Code Analysis
   - Performance Profiling
   - Documentation Generation

7. **🐳 Docker & Deployment**
   - Container Management
   - Kubernetes Deployment
   - Service Scaling
   - Log Management
   - Production Deployment

8. **📚 Help & Documentation**
   - User Guides
   - API Documentation
   - Troubleshooting
   - System Requirements

---

## 🎯 Common Use Cases

### For Scientists and Researchers

#### Quick NEO Analysis
```bash
# Command line (fastest)
python aneos.py analyze "2024 AB123"

# Or through menu
python aneos.py
# → Select 1 (Scientific Analysis)
# → Select 1 (Single NEO Analysis)
# → Enter designation
```

#### Batch Analysis
```bash
# Through menu system
python aneos.py
# → Select 1 (Scientific Analysis)
# → Select 2 (Batch Analysis)
# → Provide file with NEO designations
```

#### Interactive Analysis
```bash
python aneos.py
# → Select 1 (Scientific Analysis)
# → Select 3 (Interactive Analysis)
# → Follow guided analysis process
```

### For Developers

#### Development API Server
```bash
# Quick start
python aneos.py api --dev

# Or through menu
python aneos.py
# → Select 3 (API Services)
# → Select 4 (Development Mode)
```

#### System Testing
```bash
python aneos.py
# → Select 6 (Development Tools)
# → Select 1 (Run Tests)
```

#### Code Analysis
```bash
python aneos.py
# → Select 6 (Development Tools)
# → Select 3 (Code Analysis)
```

### For System Administrators

#### Production Deployment
```bash
# Docker Compose
python aneos.py docker

# Or through menu
python aneos.py
# → Select 7 (Docker & Deployment)
# → Select 2 (Docker Compose Up)
```

#### System Monitoring
```bash
python aneos.py
# → Select 4 (Monitoring & Diagnostics)
# → Select 1 (Live System Monitor)
```

#### Health Checks
```bash
# Quick check
python aneos.py status

# Detailed check through menu
python aneos.py
# → Select 4 (Monitoring & Diagnostics)
# → Select 4 (Health Check)
```

### For ML Engineers

#### Model Training
```bash
python aneos.py
# → Select 2 (Machine Learning)
# → Select 1 (Model Training)
# → Configure training parameters
```

#### Real-time Predictions
```bash
python aneos.py
# → Select 2 (Machine Learning)
# → Select 2 (Real-time Predictions)
# → Enter NEO designation
```

#### Model Management
```bash
python aneos.py
# → Select 2 (Machine Learning)
# → Select 3 (Model Management)
# → View/activate models
```

---

## 🔧 Configuration and Setup

### First-Time Setup

1. **Check System Status**
   ```bash
   python aneos.py status
   ```

2. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database**
   ```bash
   python aneos.py
   # → Select 5 (System Management)
   # → Select 1 (Database Management)
   ```

4. **Start Services**
   ```bash
   # Development
   python aneos.py api --dev
   
   # Production
   python aneos.py docker
   ```

### Environment Variables

The system supports various environment variables:

```bash
# Database
export ANEOS_DATABASE_URL="postgresql://user:pass@localhost/aneos"

# API Configuration
export ANEOS_LOG_LEVEL="INFO"
export ANEOS_ENV="production"

# Redis (for caching)
export ANEOS_REDIS_URL="redis://localhost:6379/0"
```

---

## 🌐 Web Interfaces

When the API server is running, you have access to:

- **Interactive API Documentation**: http://localhost:8000/docs
- **Web Dashboard**: http://localhost:8000/dashboard
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/api/v1/monitoring/metrics

With Docker Compose, additional services are available:
- **Grafana Dashboard**: http://localhost:3000 (admin/aneos)
- **Prometheus Metrics**: http://localhost:9090

---

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Check system status
   python aneos.py status
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **Database Connection Issues**
   ```bash
   python aneos.py
   # → Select 5 (System Management)
   # → Select 1 (Database Management)
   # → Initialize database
   ```

3. **Port Already in Use**
   ```bash
   # Use different port
   python aneos.py api --port 8001
   
   # Or check what's using the port
   lsof -i :8000
   ```

4. **Docker Issues**
   ```bash
   # Check Docker status
   docker --version
   docker-compose --version
   
   # Clean up containers
   python aneos.py
   # → Select 7 (Docker & Deployment)
   # → Select 8 (Cleanup Containers)
   ```

### Getting Help

1. **In-Menu Help**
   ```bash
   python aneos.py
   # → Select 8 (Help & Documentation)
   ```

2. **Command Line Help**
   ```bash
   python aneos.py --help
   ```

3. **System Diagnostics**
   ```bash
   python aneos.py
   # → Select 4 (Monitoring & Diagnostics)
   # → Select 5 (System Diagnostics)
   ```

---

## 🔄 Migration from Legacy Scripts

### Old Way → New Way

| Old Command | New Command |
|-------------|-------------|
| `python neos_o3high_v6.19.1.py` | `python aneos.py` → Scientific Analysis |
| `python start_v1.01.py` | `python aneos.py api` |
| `python test_phase2.py` | `python aneos.py` → Development Tools → Run Tests |
| Manual Docker commands | `python aneos.py docker` |

### Legacy Scripts Still Work

All original scripts remain functional:
- `neos_o3high_v6.19.1.py` - Original analysis script
- `start_v1.01.py` - Original startup script
- `test_phase2.py` - Phase 2 testing
- `start_api.py` - API server startup

---

## 🎉 Quick Examples

### 1. Analyze a NEO (Command Line)
```bash
python aneos.py analyze "2024 BX1"
```

### 2. Start Development Server
```bash
python aneos.py api --dev
# Opens: http://localhost:8000/docs
```

### 3. Production Deployment
```bash
python aneos.py docker
# Services available at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8000/dashboard  
# - Grafana: http://localhost:3000
```

### 4. System Health Check
```bash
python aneos.py status
```

### 5. Interactive Analysis Session
```bash
python aneos.py
# → 1 (Scientific Analysis)
# → 3 (Interactive Analysis)
# Follow prompts for guided analysis
```

---

**The aNEOS menu system provides a unified interface for all functionality while maintaining backward compatibility with existing scripts. Choose the method that best fits your workflow!**