# aNEOS User Guide

Complete user documentation for the aNEOS (artificial Near Earth Object detection System).

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Features](#core-features)
4. [Menu System Guide](#menu-system-guide)
5. [Scientific Analysis](#scientific-analysis)
6. [API Services](#api-services)
7. [System Management](#system-management)
8. [Health & Diagnostics](#health--diagnostics)
9. [Configuration](#configuration)
10. [Command Line Interface](#command-line-interface)
11. [Web Interface](#web-interface)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

---

## Introduction

aNEOS is a Production-Ready Scientific Analysis Platform designed for detecting artificial Near Earth Objects (NEOs) using advanced analysis techniques and machine learning algorithms.

### Key Capabilities

- **Scientific NEO Analysis**: A-grade (95/100) validation with 5-stage pipeline and enhanced modules
- **Basic API Services**: REST API with health endpoints and interactive documentation
- **Enhanced Validation**: Phase 1-3 modules including ΔBIC, spectral, radar, thermal-IR, and Gaia analysis
- **Interactive Menu System**: User-friendly command-line interface with rich console support
- **Web Dashboard**: Real-time monitoring interface with system metrics
- **Multi-source Data**: Integration with NASA, ESA, MPC and other NEO databases with fallback handling
- **Production Ready**: Robust error handling, performance optimization, and scientific rigor

### Advanced Features (Coming Soon)
- **Machine Learning Training**: Advanced ML model development and training
- **Docker Deployment**: Production containerization and orchestration  
- **Streaming Services**: Real-time data streaming and WebSocket support

### System Architecture

```
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│   Data Sources  │  │ Enhanced Pipeline│  │  Interface Layer│
│  • NASA SBDB    │──│  • 5-Stage Valid.│──│  • Menu System  │
│  • ESA/MPC      │  │  • Phase 1-3     │  │  • Basic API    │
│  • Fallback     │  │  • 12 Swarms     │  │  • Dashboard    │
└─────────────────┘  └──────────────────┘  └─────────────────┘
         │                       │                       │
    ┌─────────┐        ┌─────────────────┐        ┌─────────────┐
    │ Polling │        │ Scientific Rigor │        │ Monitoring  │
    │ System  │        │   A-grade 95%    │        │ & Health    │
    └─────────┘        └─────────────────┘        └─────────────┘
                                 │
                  ┌──────────────────┐
                  │  Storage Layer   │
                  │  • SQLite DB     │
                  │  • Smart Cache   │
                  │  • File System   │
                  └──────────────────┘
```

---

## Getting Started

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd aneos-project
   ```

2. **Install aNEOS**:
   ```bash
   python install.py
   ```

3. **Verify installation**:
   ```bash
   python aneos_menu.py
   ```

### First Analysis

Analyze your first NEO:

```bash
# Start interactive menu
python aneos_menu.py
# → 1 (Scientific Analysis)
# → 1 (Single NEO Analysis)
# → Enter: 2024 AB123
```

### Quick API Start

Start the web interface:

```bash
python aneos_menu.py
# → 2 (Basic API Services)
# → 1 (Start API Server)
```

Then visit:
- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8000/dashboard

---

## Core Features

### Scientific Analysis Engine

aNEOS analyzes NEOs using multiple scientific indicators:

1. **Orbital Mechanics Indicators**:
   - Eccentricity anomalies
   - Inclination patterns
   - Semi-major axis analysis

2. **Velocity Pattern Analysis**:
   - Velocity shift detection
   - Acceleration anomalies
   - Trajectory consistency

3. **Temporal Behavior Analysis**:
   - Close approach regularity
   - Observation history patterns
   - Temporal clustering

4. **Geographic Distribution**:
   - Subpoint clustering analysis
   - Regional approach patterns
   - Strategic location passes

5. **Physical Characteristics**:
   - Size and diameter analysis
   - Albedo anomalies
   - Spectral characteristics

### Anomaly Scoring System

Each NEO receives a comprehensive anomaly score:

| Score Range | Classification | Description |
|-------------|----------------|-------------|
| 0.0 - 0.3   | **Natural** | Typical natural NEO behavior |
| 0.3 - 0.6   | **Suspicious** | Some unusual characteristics |
| 0.6 - 0.8   | **Highly Suspicious** | Multiple anomalous indicators |
| 0.8 - 1.0   | **Artificial** | Strong evidence of artificial origin |

---

## Menu System Guide

### Main Menu Structure

```
aNEOS Main Menu
├── 1. 🔬 Scientific Analysis
├── 2. 🌐 Basic API Services  
├── 3. ⚙️ System Management
├── 4. 🔍 Health & Diagnostics
├── 5. 📚 Help & Documentation
└── 9. 🚀 Advanced Features (Postponed)
```

### Navigation

- **Numbers**: Select menu options (0-9)
- **Enter**: Confirm selection
- **Ctrl+C**: Exit current operation
- **0**: Return to previous menu or exit

### Menu Options Explained

#### 1. Scientific Analysis
Core NEO analysis functionality:
- Single NEO analysis by designation
- Batch processing from file lists
- NEO API polling across time periods
- Interactive guided analysis
- Results viewing and export

#### 2. Basic API Services
Development and basic production APIs:
- Start REST API server
- Development mode with auto-reload
- API health checks
- Interactive documentation access

#### 3. System Management
System administration tools:
- Installation and dependency management
- Database initialization and management
- System cleanup and maintenance
- Configuration management

#### 4. Health & Diagnostics
System monitoring and diagnostics:
- Comprehensive health checks
- System status reporting
- Basic system tests
- Performance diagnostics

#### 5. Help & Documentation
Documentation and help resources:
- User guides and tutorials
- Scientific methodology documentation
- API reference documentation
- Troubleshooting guides

---

## Scientific Analysis

### Single NEO Analysis

Analyze individual NEOs by designation:

1. **Access**: Main Menu → 1 → 1
2. **Input**: Enter NEO designation (e.g., "2024 AB123")
3. **Processing**: Automatic data retrieval and analysis
4. **Results**: Comprehensive anomaly assessment

#### Example Output
```
🔬 Analyzing NEO: 2024 AB123

📊 Analysis Results:
Overall Score: 0.756
Classification: highly_suspicious
Confidence: 0.923
Processing Time: 2.31s

🔍 Anomaly Indicators:
  • Orbital Mechanics: 0.82 (High eccentricity)
  • Velocity Patterns: 0.71 (Irregular shifts)
  • Temporal Behavior: 0.45 (Moderate clustering)
  • Geographic Distribution: 0.89 (Strategic clustering)
  • Physical Characteristics: 0.34 (Normal range)

🚨 Risk Factors:
  • Unusual orbital eccentricity (3.2σ deviation)
  • Velocity pattern anomaly detected
  • Atypical approach geometry
  • Geographic clustering near strategic locations
```

### Batch Analysis

Process multiple NEOs from a file:

1. **Prepare file**: Create text file with NEO designations (one per line)
   ```
   2024 AB123
   2024 BX1
   2024 CY2
   2019 JM
   ```

2. **Execute**: Main Menu → 1 → 2
3. **Input**: Provide file path
4. **Monitor**: Progress tracking with real-time updates
5. **Results**: Batch summary with individual results

### NEO API Polling

Systematic polling of NEO APIs across time periods:

1. **Access**: Main Menu → 1 → 3
2. **Selection**: Choose time period (1m, 1w, 1y, max)
3. **Processing**: Automatic API queries and analysis
4. **Export**: Results saved to JSON files

#### Supported Time Periods
- **1m**: Last month
- **1w**: Last week  
- **1d**: Last day
- **6m**: Last 6 months
- **1y**: Last year
- **25y**: Last 25 years
- **max**: All available data

### Analysis Results

#### Understanding Scores

**Raw Scores**: Individual indicator values (0-1)
**Weighted Scores**: Indicators multiplied by importance weights
**Overall Score**: Composite score from all indicators
**Confidence**: Statistical confidence in the result

#### Metadata Fields

Each analysis includes comprehensive metadata:
- **Processing time**: Analysis duration
- **Data sources**: APIs and databases queried
- **Feature counts**: Number of data points analyzed
- **Error flags**: Any issues encountered
- **Version info**: Analysis algorithm version

---

## API Services

### REST API

#### Starting the API Server

**Development Mode**:
```bash
# Through menu
Main Menu → 2 → 2

# Direct command
python start_api.py --dev
```

**Production Mode**:
```bash
# Through menu
Main Menu → 2 → 1

# With custom settings
python start_api.py --host 0.0.0.0 --port 8000 --workers 4
```

#### Key Endpoints

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/health` | GET | System health check |
| `/docs` | GET | Interactive API documentation |
| `/dashboard` | GET | Web dashboard |
| `/api/v1/analysis/analyze` | POST | Single NEO analysis |
| `/api/v1/analysis/batch` | POST | Batch analysis |
| `/api/v1/monitoring/metrics` | GET | System metrics |

#### Authentication

API keys can be managed through:
```bash
Main Menu → 2 → 6 (Manage API Keys)
```

### Web Dashboard

The dashboard provides:
- **Real-time system status**
- **Recent analysis results**
- **Performance metrics**
- **Quick analysis tools**
- **System alerts**

#### Dashboard Sections

1. **Overview**: System status and key metrics
2. **Analysis**: Quick NEO analysis interface
3. **Results**: Recent analysis results browser
4. **Monitoring**: System performance graphs
5. **Alerts**: Important notifications

---

## System Management

### Installation Management

Comprehensive installation and dependency management:

#### Full Installation
```bash
Main Menu → 3 → 1 → 1
```
Installs all components including ML dependencies.

#### Minimal Installation
```bash
Main Menu → 3 → 1 → 2
```
Core components only for basic functionality.

#### System Check
```bash
Main Menu → 3 → 1 → 3
```
Verifies system requirements and dependencies.

#### Fix Dependencies
```bash
Main Menu → 3 → 1 → 4
```
Automatically resolves dependency issues.

### Database Management

#### Initialize Database
```bash
Main Menu → 3 → 2
```

Supports multiple database backends:
- **SQLite**: Default for development
- **PostgreSQL**: Recommended for production
- **MySQL**: Alternative production option

#### Database Operations
- **Backup**: Automatic daily backups
- **Restore**: Point-in-time recovery
- **Migration**: Schema updates
- **Cleanup**: Remove old data

### System Cleanup

Regular maintenance operations:

```bash
Main Menu → 3 → 3
```

Cleans:
- **Cache files**: Temporary analysis cache
- **Log files**: Rotates old log files  
- **Database**: Removes expired data
- **Downloads**: Clears temporary downloads

---

## Health & Diagnostics

### System Health Check

Comprehensive system assessment:

```bash
Main Menu → 4 → 1
```

#### Health Check Components

1. **Core Components**: Analysis pipeline availability
2. **Database**: Connection and performance
3. **File System**: Directory structure and permissions
4. **Dependencies**: Required packages and versions
5. **External APIs**: Connectivity to data sources
6. **Memory Usage**: Current memory consumption
7. **Disk Space**: Available storage capacity

#### Health Check Output

```
🔍 System Health Check

Component                Status        Details
Core Components         ✅ Available   All modules loaded
Database               ✅ Connected   SQLite engine ready
File System            ✅ OK          All directories exist  
Dependencies           ⚠️ Missing     ML dependencies incomplete
External APIs          ✅ Connected   All sources responding
Memory Usage           ✅ Normal      245MB / 8GB used
Disk Space            ✅ Available   15GB free
```

### System Status

Quick system overview:

```bash
Main Menu → 4 → 2
```

Shows:
- **Component availability**
- **Database status**
- **API services status**
- **Current operations**

### Basic Tests

Validation test suite:

```bash
Main Menu → 4 → 3
```

Tests:
1. **Core functionality**: Basic analysis pipeline
2. **Database operations**: CRUD operations
3. **File system**: Read/write permissions
4. **Network connectivity**: External API access

---

## Configuration

### Configuration Files

aNEOS uses hierarchical configuration:

1. **Default settings**: Built-in configuration
2. **Environment variables**: Runtime overrides
3. **Config files**: YAML or JSON configuration
4. **Command line**: Runtime parameters

### Environment Variables

Key configuration variables:

```bash
# Database
ANEOS_DATABASE_URL=sqlite:///./aneos.db

# API Settings  
ANEOS_HOST=0.0.0.0
ANEOS_PORT=8000
ANEOS_WORKERS=1

# External APIs
ANEOS_REQUEST_TIMEOUT=10
ANEOS_MAX_RETRIES=3

# Processing
ANEOS_MAX_WORKERS=10
ANEOS_BATCH_SIZE=100
ANEOS_CACHE_TTL=3600

# Logging
ANEOS_LOG_LEVEL=INFO
ANEOS_LOG_FILE=aneos.log
```

### Configuration Management

Access configuration tools:

```bash
Main Menu → 3 → 4
```

#### Configuration Operations
- **View current**: Display active configuration
- **Edit settings**: Modify configuration values
- **Reset to defaults**: Restore factory settings
- **Export config**: Save configuration to file
- **Import config**: Load configuration from file

### Threshold Configuration

Anomaly detection thresholds:

```python
# Orbital thresholds
eccentricity: 0.8           # High eccentricity threshold
inclination: 45.0           # Unusual inclination (degrees)
velocity_shift: 5.0         # Velocity anomaly threshold

# Temporal thresholds
temporal_inertia: 100.0     # Days between regular approaches
observation_gap_multiplier: 3  # Expected vs actual observation gaps

# Physical thresholds
diameter_min: 0.1           # Minimum diameter (km)
diameter_max: 10.0          # Maximum diameter (km)  
albedo_artificial: 0.6      # High albedo threshold
```

### Weight Configuration

Indicator importance weights:

```python
# Analysis weights
orbital_mechanics: 1.5       # Orbital anomaly importance
velocity_shifts: 2.0         # Velocity pattern importance
close_approach_regularity: 2.0  # Temporal pattern importance
geographic_clustering: 1.0   # Geographic importance
physical_anomalies: 1.0      # Physical characteristic importance
```

---

## Command Line Interface

### Direct Commands

aNEOS supports direct command-line operations:

```bash
# Analysis commands
python aneos.py analyze "2024 AB123"
python aneos.py batch analyze neos.txt
python aneos.py poll --period 1w

# API commands  
python aneos.py api
python aneos.py api --dev --port 8001

# System commands
python aneos.py status
python aneos.py health-check
python aneos.py install --full

# Note: Docker commands are not yet available in current version
# Use the menu system for full functionality
```

### Command Options

#### Global Options
- `--verbose, -v`: Enable verbose logging
- `--quiet, -q`: Suppress non-essential output
- `--config`: Specify configuration file
- `--log-level`: Set logging level

#### Analysis Options
- `--format`: Output format (json, csv, txt)
- `--output`: Output file path
- `--no-cache`: Disable caching
- `--sources`: Specify data sources

#### API Options
- `--host`: Bind address (default: 0.0.0.0)
- `--port`: Port number (default: 8000)
- `--workers`: Number of workers
- `--dev`: Development mode

---

## Web Interface

### Dashboard Overview

The web dashboard provides comprehensive system management through a browser interface.

#### Access
- **URL**: http://localhost:8000/dashboard
- **Authentication**: API key or session-based
- **Browser Support**: Chrome, Firefox, Safari, Edge

#### Main Sections

**System Status**:
- Real-time component health
- Performance metrics
- Active operations
- Resource utilization

**Analysis Tools**:
- Quick NEO analysis
- Batch job submission
- Result browsing
- Export functionality

**Monitoring**:
- System metrics graphs
- Alert management
- Log viewing
- Performance trends

**Configuration**:
- System settings
- User preferences
- API key management
- Backup/restore

### API Documentation Interface

Interactive API documentation at `/docs`:

- **Endpoint explorer**: All available endpoints
- **Try it out**: Execute API calls directly
- **Schema browser**: Request/response models
- **Authentication**: API key testing
- **Examples**: Sample requests and responses

---

## Best Practices

### Analysis Best Practices

1. **Data Quality**:
   - Always verify NEO designation format
   - Check data source availability
   - Validate analysis results
   - Review confidence scores

2. **Batch Processing**:
   - Use appropriate batch sizes (100-1000 items)
   - Monitor processing progress
   - Handle failures gracefully
   - Save intermediate results

3. **Result Interpretation**:
   - Consider confidence levels
   - Review individual indicators
   - Cross-validate with multiple sources
   - Document unusual findings

### System Administration

1. **Regular Maintenance**:
   - Perform health checks weekly
   - Clean system logs monthly
   - Update dependencies quarterly
   - Backup database daily

2. **Performance Optimization**:
   - Monitor memory usage
   - Optimize database queries
   - Use appropriate caching
   - Scale workers based on load

3. **Security**:
   - Rotate API keys regularly
   - Monitor access logs
   - Use HTTPS in production
   - Implement rate limiting

### Development Workflow

1. **Testing**:
   ```bash
   # Run system tests
   Main Menu → 4 → 3
   
   # Validate installation
   python install.py --check
   ```

2. **Debugging**:
   ```bash
   # Enable verbose logging
   python aneos.py --verbose
   
   # Check system diagnostics
   Main Menu → 4 → 4
   ```

3. **Code Quality**:
   - Follow PEP 8 style guidelines
   - Add comprehensive docstrings
   - Include unit tests
   - Use type hints

---

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: `ModuleNotFoundError` during startup
```bash
# Solution
python install.py --fix-deps
```

**Issue**: Database connection failed  
```bash
# Solution
Main Menu → 3 → 2 (Database Management)
# or
rm aneos.db && python -c "from aneos_api.database import init_database; init_database()"
```

**Issue**: Port already in use
```bash
# Solution
python start_api.py --port 8001
```

#### Runtime Issues

**Issue**: Analysis fails with API timeout
```bash
# Check external connectivity
curl -I https://ssd-api.jpl.nasa.gov/cad.api

# Increase timeout
export ANEOS_REQUEST_TIMEOUT=30
```

**Issue**: High memory usage
```bash
# Monitor memory
Main Menu → 4 → 1

# Reduce batch size
export ANEOS_BATCH_SIZE=50
```

**Issue**: Slow analysis performance
```bash
# Check system resources
Main Menu → 4 → 4

# Increase workers
export ANEOS_MAX_WORKERS=20
```

### Diagnostic Commands

```bash
# System health
python aneos_menu.py
# → 4 → 1

# Component status
python aneos_menu.py  
# → 4 → 2

# Basic tests
python aneos_menu.py
# → 4 → 3

# Installation check
python install.py --check
```

### Log Files

Important log locations:
- **Main log**: `aneos.log`
- **API log**: `logs/api.log`
- **Database log**: `logs/database.log`
- **Analysis log**: `dataneos/logs/enhanced_neo_poller.log`

### Getting Help

1. **Built-in help**: `python aneos.py --help`
2. **System diagnostics**: Main Menu → 4 → 4
3. **Documentation**: `docs/` directory
4. **Health check**: Main Menu → 4 → 1
5. **Installation check**: `python install.py --check`

---

## Advanced Topics

### Custom Indicator Development

Create custom anomaly indicators by extending the base classes:

```python
from aneos_core.analysis.indicators.base import AnomalyIndicator

class CustomIndicator(AnomalyIndicator):
    def evaluate(self, neo_data):
        # Custom analysis logic
        score = custom_analysis(neo_data)
        
        return IndicatorResult(
            indicator_name=self.name,
            raw_score=score,
            weighted_score=self.calculate_weighted_score(score),
            metadata={'custom_field': 'value'}
        )
```

### API Integration

Integrate with external systems using the REST API:

```python
import requests

# Analyze NEO via API
response = requests.post(
    'http://localhost:8000/api/v1/analysis/analyze',
    json={'designation': '2024 AB123'},
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)

result = response.json()
print(f"Score: {result['overall_score']}")
```

### Production Deployment

For production deployment, use the direct Python approach:

```bash
# Start production API server
python aneos.py api --host 0.0.0.0 --port 8000

# Or use the menu system
python aneos_menu.py
# → 2 (Basic API Services) → 1 (Start API Server)

# Monitor system health
python aneos.py status
```

**Note**: Docker deployment and advanced containerization features are planned for future releases.

---

This completes the comprehensive aNEOS User Guide. For additional help, consult the other documentation sections or use the built-in help system.