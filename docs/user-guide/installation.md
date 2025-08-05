# aNEOS Installation Guide

Complete installation and setup guide for the aNEOS (Advanced Near Earth Object detection System) platform.

## 🚀 Quick Installation

### Automated Installation (Recommended)

```bash
# Clone or download the aNEOS project
cd aneos-project

# Run the automated installer
python install.py

# Verify installation
python aneos.py status
```

The automated installer will:
- Check system requirements
- Install all dependencies
- Set up directories and configuration
- Initialize the database
- Run verification tests
- Create an installation report

### Manual Installation

If you prefer manual control over the installation process:

```bash
# 1. Check Python version (3.8+ required)
python --version

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Set up directories
mkdir -p data logs models cache exports backups temp

# 4. Create configuration files
cp .env.example .env  # Edit as needed

# 5. Initialize database
python -c "from aneos_api.database import init_database; init_database()"

# 6. Verify installation
python aneos.py status
```

## 📋 System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: 4 GB RAM (8 GB recommended)
- **Storage**: 5 GB free disk space (10 GB recommended)
- **Network**: Internet connection for downloading data

### Recommended Requirements

- **Operating System**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.10 or higher
- **Memory**: 16 GB RAM
- **Storage**: 50 GB free disk space (SSD recommended)
- **CPU**: Multi-core processor (4+ cores recommended)

### Optional Components

- **Docker**: For containerized deployment
- **Docker Compose**: For multi-service orchestration
- **Kubernetes**: For production cluster deployment
- **Git**: For version control and updates
- **Redis**: For enhanced caching performance

## 🔧 Installation Options

### Option 1: Full Installation (Recommended)

Installs all components including optional ML libraries:

```bash
python install.py --full
```

**Includes:**
- Core scientific libraries (astropy, numpy, pandas, scipy)
- Machine learning libraries (scikit-learn, torch)
- API framework (FastAPI, uvicorn)
- Database support (SQLAlchemy, PostgreSQL drivers)
- Monitoring tools (Prometheus, Grafana integration)
- Web interface components
- Development tools

### Option 2: Minimal Installation

Installs only core dependencies for basic functionality:

```bash
python install.py --minimal
```

**Includes:**
- Core scientific libraries
- Basic analysis capabilities
- SQLite database support
- Command-line interface

### Option 3: Custom Installation

Interactive installation with component selection:

```bash
python install.py
```

Choose specific components based on your needs:
- Scientific analysis only
- API services
- Machine learning capabilities
- Development tools
- Monitoring and alerting

## 🐳 Docker Installation

### Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Quick Docker Setup

```bash
# Start all services
python aneos.py docker

# Or manually
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs aneos-api
```

**Services included:**
- aNEOS API server
- PostgreSQL database
- Redis cache
- Nginx reverse proxy
- Prometheus monitoring
- Grafana dashboards

### Docker Development

```bash
# Build development image
docker build -t aneos:dev .

# Run development container
docker run -it --rm -p 8000:8000 -v $(pwd):/app aneos:dev python aneos.py api --dev
```

## ☸️ Kubernetes Installation

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm (optional, for easier management)

### Deploy to Kubernetes

```bash
# Deploy all components
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=aneos-api

# Access the service
kubectl port-forward service/aneos-api-service 8000:80
```

### Scale the deployment

```bash
# Scale API pods
kubectl scale deployment aneos-api --replicas=3

# Check scaling
kubectl get pods -l app=aneos-api
```

## 🔍 Installation Verification

### Automated Verification

```bash
# Run comprehensive system check
python aneos.py status

# Run installation verification
python install.py --check

# Test core functionality
python aneos.py analyze "2024 AB123"
```

### Manual Verification Steps

1. **Check Python imports**:
   ```bash
   python -c "import aneos_core; print('Core modules OK')"
   python -c "import aneos_api; print('API modules OK')"
   ```

2. **Test database connection**:
   ```bash
   python -c "from aneos_api.database import get_database_status; print(get_database_status())"
   ```

3. **Start API server**:
   ```bash
   python aneos.py api --dev
   # Should start server on http://localhost:8000
   ```

4. **Check web interface**:
   - API documentation: http://localhost:8000/docs
   - Dashboard: http://localhost:8000/dashboard
   - Health check: http://localhost:8000/health

### Expected Output

After successful installation, you should see:

```
✅ aNEOS Installation Complete!

System Status:
✅ Core Components: Available
✅ Database: Connected (sqlite:///./aneos.db)
✅ API Services: Available
✅ File System: All directories exist

Next steps:
1. Run: python aneos.py status
2. Run: python aneos.py
3. Run: python aneos.py api --dev
```

## 🛠️ Troubleshooting Installation

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version
python3 --version

# Use specific Python version
python3.10 install.py
```

#### Permission Issues
```bash
# Linux/macOS: Use user installation
pip install --user -r requirements.txt

# Or use virtual environment
python -m venv aneos-env
source aneos-env/bin/activate  # Linux/macOS
# aneos-env\Scripts\activate     # Windows
pip install -r requirements.txt
```

#### Network/Firewall Issues
```bash
# Use alternative PyPI mirror
pip install -r requirements.txt -i https://pypi.org/simple/

# Install offline (if packages downloaded)
pip install --no-index --find-links ./offline-packages -r requirements.txt
```

#### Dependency Conflicts
```bash
# Clear pip cache
pip cache purge

# Reinstall in clean environment
python -m venv fresh-env
source fresh-env/bin/activate
pip install -r requirements.txt
```

#### Database Issues
```bash
# Reset database
rm aneos.db
python -c "from aneos_api.database import init_database; init_database()"

# Check database status
python -c "from aneos_api.database import get_database_status; print(get_database_status())"
```

### Installation Logs

Check installation logs for detailed error information:

```bash
# View installation report
cat installation_report.json

# Check system logs
tail -f logs/aneos.log
```

### Getting Help

1. **Run diagnostics**: `python install.py --check`
2. **Check system status**: `python aneos.py status`
3. **Review installation report**: `installation_report.json`
4. **Check logs**: `logs/` directory
5. **Dependency issues**: `python install.py --fix-deps`

## 🔄 Post-Installation Setup

### 1. Configuration

Edit `.env` file for your environment:

```bash
# Database (production)
ANEOS_DATABASE_URL=postgresql://user:pass@localhost/aneos

# API settings
ANEOS_ENV=production
ANEOS_HOST=0.0.0.0
ANEOS_PORT=8000

# Security (change these!)
ANEOS_SECRET_KEY=your-secure-secret-key
ANEOS_API_KEY_SALT=your-secure-salt
```

### 2. Create Admin User

```bash
python aneos.py
# → 5 (System Management)
# → 4 (User Management)
# → Create admin user
```

### 3. Initialize Data

```bash
# Download initial NEO data (optional)
python aneos.py
# → 1 (Scientific Analysis)
# → 6 (Data Management)
```

### 4. Set Up Monitoring (Production)

```bash
# Start monitoring stack
docker-compose up -d prometheus grafana

# Access Grafana: http://localhost:3000 (admin/aneos)
# Import aNEOS dashboards from grafana/dashboards/
```

## 🚀 Next Steps

After successful installation:

1. **Explore the system**: `python aneos.py` (interactive menu)
2. **Analyze your first NEO**: `python aneos.py analyze "2024 AB123"`
3. **Start API server**: `python aneos.py api --dev`
4. **Read the documentation**: Check `docs/` directory
5. **Run tests**: `python aneos.py` → Development Tools → Run Tests

## 📚 Related Documentation

- [Quick Start Guide](quick-start.md) - Get up and running quickly
- [Menu System Guide](menu-system.md) - Interactive menu usage
- [Troubleshooting](../troubleshooting/installation.md) - Installation-specific issues
- [Development Setup](../development/setup.md) - Development environment

---

**Installation complete!** You're ready to start detecting artificial NEOs with aNEOS. 🚀