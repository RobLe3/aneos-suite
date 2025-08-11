# aNEOS Quick Start Guide

Get up and running with aNEOS in under 5 minutes!

## ⚡ Super Quick Start

```bash
# 1. Install aNEOS
python install.py

# 2. Analyze your first NEO
python aneos.py analyze "2024 AB123"

# 3. Start the interactive menu
python aneos.py
```

That's it! You're ready to detect artificial NEOs. 🚀

## 🎯 Common Use Cases

### 1. Analyze a Single NEO

**Command Line (Fastest):**
```bash
python aneos.py analyze "2024 BX1"
```

**Interactive Menu:**
```bash
python aneos.py
# → 1 (Scientific Analysis)
# → 1 (Single NEO Analysis)
# → Enter: 2024 BX1
```

**Expected Output:**
```
🔬 Analyzing NEO: 2024 BX1

📊 Analysis Results for 2024 BX1:
Overall Score: 0.234
Classification: natural
Confidence: 0.891
Processing Time: 1.45s
```

### 2. Start the Web Interface

```bash
# Development mode (auto-reload)
python aneos.py api --dev

# Production mode
python aneos.py api
```

**Access:**
- **API Documentation**: http://localhost:8000/docs
- **Web Dashboard**: http://localhost:8000/dashboard
- **Health Check**: http://localhost:8000/health

### 3. Run with Docker

```bash
# Start all services
python aneos.py docker

# Check status
docker-compose ps
```

**Access:**
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8000/dashboard
- **Grafana**: http://localhost:3000 (admin/aneos)

### 4. Batch Analysis

Create a file `neos.txt` with NEO designations:
```
2024 AB123
2024 BX1
2024 CY2
```

Then run:
```bash
python aneos.py
# → 1 (Scientific Analysis)
# → 2 (Batch Analysis)
# → Enter file path: neos.txt
```

### 5. System Health Check

```bash
# Quick check
python aneos.py status

# Detailed check
python aneos.py
# → 4 (Monitoring & Diagnostics)
# → 4 (Health Check)
```

## 🧭 Navigation Guide

### Main Menu Structure

```
aNEOS Main Menu
├── 1. 🔬 Scientific Analysis
│   ├── Single NEO Analysis
│   ├── Batch Analysis
│   ├── Interactive Analysis
│   └── Results Viewer
├── 2. 🤖 Machine Learning
│   ├── Model Training
│   ├── Real-time Predictions
│   └── Model Management
├── 3. 🌐 API Services
│   ├── Start API Server
│   ├── Web Dashboard
│   └── Development Mode
├── 4. 📊 Monitoring & Diagnostics
│   ├── Live System Monitor
│   ├── Health Check
│   └── Performance Metrics
├── 5. ⚙️ System Management
│   ├── Database Management
│   ├── System Cleanup
│   └── Configuration
├── 6. 🛠️ Development Tools
│   ├── Run Tests
│   ├── Debug Mode
│   └── Code Analysis
├── 7. 🐳 Docker & Deployment
│   ├── Docker Compose
│   ├── Container Status
│   └── Kubernetes Deploy
└── 8. 📚 Help & Documentation
```

### Command Line Shortcuts

| Action | Command |
|--------|---------|
| Interactive Menu | `python aneos.py` |
| Analyze NEO | `python aneos.py analyze "2024 AB123"` |
| Start API | `python aneos.py api` |
| Development API | `python aneos.py api --dev` |
| Docker Services | `python aneos.py docker` |
| System Status | `python aneos.py status` |
| Help | `python aneos.py --help` |

## 🔬 Understanding NEO Analysis Results

### Anomaly Score Interpretation

| Score Range | Classification | Meaning |
|-------------|----------------|---------|
| 0.0 - 0.3 | **Natural** | Typical natural NEO behavior |
| 0.3 - 0.6 | **Suspicious** | Some unusual characteristics |
| 0.6 - 0.8 | **Highly Suspicious** | Multiple anomalous indicators |
| 0.8 - 1.0 | **Artificial** | Strong evidence of artificial origin |

### Key Indicators

aNEOS analyzes multiple indicators:
- **Orbital Mechanics**: Unusual orbital elements
- **Velocity Patterns**: Atypical velocity changes
- **Temporal Behavior**: Irregular timing patterns
- **Geographic Distribution**: Unusual approach patterns

### Example Analysis Output

```
📊 Analysis Results for 2024 AB123:
Overall Score: 0.756
Classification: highly_suspicious
Confidence: 0.923
Processing Time: 2.31s

🚨 Risk Factors:
  • Unusual orbital eccentricity (3.2σ deviation)
  • Velocity pattern anomaly detected
  • Atypical approach geometry
  • High artificial probability (ML): 0.834
```

## 🌐 Web Interface Overview

### API Documentation (http://localhost:8000/docs)

Interactive API documentation with:
- All endpoint descriptions
- Request/response examples
- Try-it-yourself interface
- Authentication examples

### Web Dashboard (http://localhost:8000/dashboard)

Real-time dashboard featuring:
- System status overview
- Recent analysis results
- Performance metrics
- Alert notifications
- Quick analysis tools

### Key Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | System health check |
| `POST /api/v1/analysis/analyze` | Analyze single NEO |
| `GET /api/v1/monitoring/metrics` | System metrics |
| `GET /api/v1/prediction/predict` | ML predictions |
| `WebSocket /api/v1/stream/ws/{session}` | Real-time data |

## 🛠️ Development Quick Start

### Setting Up Development Environment

```bash
# 1. Clone the repository
git clone <repository-url>
cd aneos-project

# 2. Install in development mode
python install.py --full

# 3. Start development server
python aneos.py api --dev

# 4. Run tests
python aneos.py
# → 6 (Development Tools)
# → 1 (Run Tests)
```

### Making Your First Analysis

```python
# Quick Python script example
import sys
sys.path.append('.')

from aneos_core.analysis.pipeline import create_analysis_pipeline
import asyncio

async def analyze_neo(designation):
    pipeline = create_analysis_pipeline()
    result = await pipeline.analyze_neo(designation)
    
    if result:
        score = result.anomaly_score.overall_score
        classification = result.anomaly_score.classification
        print(f"{designation}: {score:.3f} ({classification})")
    else:
        print(f"Analysis failed for {designation}")

# Run analysis
asyncio.run(analyze_neo("2024 AB123"))
```

## 🐳 Docker Quick Start

### Development with Docker

```bash
# Build development image
docker build -t aneos:dev .

# Run development container
docker run -it --rm -p 8000:8000 \
  -v $(pwd):/app aneos:dev \
  python aneos.py api --dev
```

### Production with Docker Compose

```bash
# Start all services
docker-compose up -d

# Scale API instances
docker-compose up -d --scale aneos-api=3

# View logs
docker-compose logs -f aneos-api

# Stop services
docker-compose down
```

## 🔧 Configuration Quick Setup

### Environment Variables

Create `.env` file:
```bash
# Basic configuration
ANEOS_ENV=development
ANEOS_LOG_LEVEL=INFO

# Database
ANEOS_DATABASE_URL=sqlite:///./aneos.db

# API
ANEOS_HOST=0.0.0.0
ANEOS_PORT=8000

# Security (change in production!)
ANEOS_SECRET_KEY=dev-secret-key
```

### Quick Configuration Changes

```bash
# Through menu system
python aneos.py
# → 5 (System Management)
# → 3 (Configuration Management)

# Or edit directly
nano .env
```

## 🚨 Quick Troubleshooting

### Installation Issues

```bash
# Check system requirements
python install.py --check

# Fix dependency issues
python install.py --fix-deps

# Clean installation
rm -rf __pycache__ *.pyc
python install.py --full
```

### Runtime Issues

```bash
# Check system status
python aneos.py status

# View logs
tail -f logs/aneos.log

# Reset database
rm aneos.db
python -c "from aneos_api.database import init_database; init_database()"
```

### Common Error Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError` | `python install.py --fix-deps` |
| `Database connection failed` | `python aneos.py` → System Management → Database |
| `Port 8000 in use` | `python aneos.py api --port 8001` |
| `Permission denied` | Use virtual environment or `--user` flag |

## 📚 Next Steps

Now that you're up and running:

1. **Explore more features**: Try batch analysis and ML predictions
2. **Read the documentation**: Check out [Menu System Guide](menu-system.md)
3. **Set up monitoring**: Use Docker Compose for full monitoring stack
4. **Customize configuration**: Modify `.env` for your needs
5. **Contribute**: See [Development Guide](../development/setup.md)

## 🆘 Getting Help

- **Interactive help**: `python aneos.py` → Help & Documentation
- **Command help**: `python aneos.py --help`
- **System diagnostics**: `python aneos.py status`
- **Installation check**: `python install.py --check`
- **Documentation**: Browse `docs/` directory

---

**You're all set!** Start exploring the universe of Near Earth Objects and detect potential artificial ones with aNEOS. 🌌✨