# aNEOS - Advanced Near Earth Object detection System

**A comprehensive, multi-SWARM platform for artificial NEO detection and analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/README.md)

---

## 🚀 Overview

aNEOS has evolved from a simple 2-script system into a sophisticated **multi-SWARM platform** capable of processing massive astronomical datasets with professional-grade analysis workflows. The system implements the **Artificial NEOs Theory** - the hypothesis that some Near-Earth Objects may be artificially influenced or controlled for surveillance purposes.

**What makes aNEOS unique:**
- **Multi-stage refinement pipeline**: Processes 50K+ → 5K → 500 → 50 candidates
- **20+ specialized SWARM systems** for different analysis types
- **200-year historical polling capability** with NASA API integration
- **Professional menu system** with intuitive navigation
- **Real-time validation** through multiple analysis methods
- **Publication-ready scientific rigor** with comprehensive statistical testing

---

## 🎯 Quick Start

### Interactive Menu System
```bash
# Launch the main menu system
python aneos_menu.py

# Available main options:
# 1. NEO Detection & Analysis
# 2. Mission Status & Intelligence  
# 3. Scientific Tools
# 9. Mission Control (Advanced)
```

### One-Command Analysis
```bash
# Quick scan for recent NEOs
python aneos_menu.py → 1 → 1

# 200-year historical poll
python aneos_menu.py → 1 → 2 → 1

# Scientific validation pipeline
python aneos_menu.py → 3 → 1
```

---

## 🏗️ System Architecture

### Multi-Stage Analysis Pipeline

**Stage 1: Historical Polling**
- **200-year chunked polling** from NASA APIs
- **50,000+ raw objects** processed
- **Intelligent caching** and progress tracking

**Stage 2: XVIII SWARM First-Stage Review**
- **Automatic anomaly scoring** with 6 clue categories
- **5,000 candidates** selected (90% reduction)
- **Continuous scoring** (0-1 scale) instead of binary classification

**Stage 3: Multi-Stage Validation** 
- **500 candidates** through enhanced validation
- **12+ validation systems** (radar, thermal, spectral, etc.)
- **Cross-reference** with space debris catalogs

**Stage 4: Expert Review Queue**
- **50 final candidates** for human analysis
- **Comprehensive reports** with confidence metrics
- **Publication-ready** documentation

### Core Data Sources

**NASA Integration:**
- Small-Body Database (SBDB)
- JPL Horizons System
- NEODyS (Near Earth Objects Dynamic Site)
- Minor Planet Center (MPC)

**Validation Databases:**
- DISCOS (ESA Space Debris Database)
- SATCAT (Satellite Catalog)
- SPACE-TRACK.ORG
- CSpOC (Combined Space Operations Center)

---

## 🔬 SWARM Systems

aNEOS employs specialized **SWARM systems** for different analysis types:

### Scientific Analysis SWARMs
- **CLAUDETTE SWARM**: Statistical testing and false positive prevention
- **KAPPA SWARM**: Radar polarization analysis
- **LAMBDA SWARM**: Thermal-infrared beaming analysis
- **MU SWARM**: Gaia astrometric precision calibration
- **THETA SWARM**: Human hardware analysis
- **XVIII SWARM**: Advanced anomaly scoring

### Validation SWARMs
- **Multi-stage validator** with 5-stage systematic validation
- **Uncertainty quantification** with Monte Carlo analysis
- **Spectral outlier analysis** for material identification
- **Delta-BIC analysis** for model comparison
- **False positive prevention** system

---

## 📊 Key Features

### Professional Menu System
- **Mission Control interface** with intelligence dashboard
- **Learning mode** with guided tutorials
- **Scientific tools** for advanced analysis
- **System diagnostics** and health monitoring
- **Development tools** for customization

### Analysis Capabilities
- **Orbital dynamics modeling** with perturbation analysis
- **Physical property estimation** (size, mass, composition)
- **Trajectory prediction** with uncertainty bounds
- **Cross-reference validation** against known catalogs
- **Statistical significance testing** with multiple corrections

### Data Processing
- **Intelligent caching** for fast repeated analysis
- **Parallel processing** for large datasets
- **Progress tracking** with rich UI elements
- **Export functionality** in multiple formats
- **Real-time monitoring** of analysis pipelines

---

## 🛠️ Installation

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/your-org/aneos-project.git
cd aneos-project

# Run the installation script
python install.py

# Launch the system
python aneos_menu.py
```

### System Requirements
- **Python 3.8+**
- **8GB+ RAM** (16GB recommended for full historical polls)
- **10GB+ storage** for cache and results
- **Internet connection** for NASA API access

### Dependencies
Core dependencies are automatically installed:
- `requests`, `numpy`, `scipy` (data processing)
- `rich` (UI), `asyncio` (async processing)
- `sqlite3` (local database)
- `astropy`, `astroquery` (astronomical calculations)

---

## 📖 Documentation

**Complete documentation available in [`docs/`](docs/README.md):**

- **[Installation Guide](docs/user-guide/installation.md)** - Complete setup instructions
- **[Menu System Guide](docs/user-guide/menu-system.md)** - Navigation and usage
- **[Scientific Documentation](docs/scientific/scientific-documentation.md)** - Analysis methods
- **[API Reference](docs/api/rest-api.md)** - REST API endpoints
- **[Troubleshooting Guide](docs/troubleshooting/troubleshooting-guide.md)** - Common issues

---

## 🔍 Usage Examples

### Basic NEO Analysis
```python
from aneos_core.analysis.pipeline import create_analysis_pipeline

# Create analysis pipeline
pipeline = create_analysis_pipeline()

# Analyze a specific NEO
result = await pipeline.analyze_neo("2025 OA")
print(f"Anomaly Score: {result.anomaly_score}")
print(f"Classification: {result.classification}")
```

### Historical Data Poll
```python
from aneos_core.polling.historical_chunked_poller import HistoricalChunkedPoller

# 5-year historical poll
poller = HistoricalChunkedPoller()
results = await poller.poll_time_range(years=5)
print(f"Found {results.total_objects_found} NEOs")
```

### Multi-Stage Validation
```python
from aneos_core.validation.multi_stage_validator import MultiStageValidator

# Validate a candidate object
validator = MultiStageValidator()
validation_result = await validator.validate_object(neo_data)
print(f"Validation Score: {validation_result.overall_score}")
print(f"Passed Stages: {validation_result.stages_passed}/5")
```

---

## 🎯 The Science

### Artificial NEOs Theory
The **Artificial NEOs Theory** proposes that some Near-Earth Objects exhibit:
- **Orbital behaviors** unexplainable by natural dynamics alone
- **Suspiciously stable orbits** maintained over long periods
- **Precise timing** of close approaches to Earth
- **Unusual physical properties** (spectral, thermal, radar signatures)

### Detection Methodology
1. **Quantitative anomaly scoring** using 6 indicator categories
2. **Statistical significance testing** with proper corrections  
3. **Cross-validation** against known space debris
4. **Physical plausibility** assessment
5. **Human expert review** for final classification

### Scientific Rigor
- **Publication-ready** statistical methods
- **Reproducible results** with version control
- **Comprehensive uncertainty** quantification
- **Peer-review quality** documentation

---

## 🌐 Web Interface & API

### Launch Web Dashboard
```bash
python aneos_menu.py → 2 → 2
# Access at: http://localhost:8000
```

### API Services
```bash
python aneos_menu.py → 2 → 1
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

---

## 🔧 Configuration

### Main Configuration
Edit `aneos_core/config/settings.py` for:
- API endpoints and keys
- Analysis parameters
- Cache settings
- Output formats

### SWARM System Configuration
Individual SWARM systems have configuration files in:
- `aneos_core/config/advanced_scoring_weights.json` (XVIII SWARM)
- Individual validation system configs in `aneos_core/validation/`

---

## 📈 Performance

### Typical Processing Rates
- **Quick scan (1 day)**: ~1-2 minutes
- **Survey mission (1 week)**: ~5-10 minutes  
- **Historical poll (1 year)**: ~30-60 minutes
- **Full historical (200 years)**: ~4-8 hours

### Resource Usage
- **Memory**: 2-8GB depending on dataset size
- **Storage**: ~1GB per year of historical data
- **CPU**: Scales with available cores (parallel processing)

---

## 🤝 Contributing

We welcome contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

**Key areas for contribution:**
- Additional validation methods
- New data source integrations
- Performance optimizations
- Documentation improvements
- Test coverage expansion

---

## 📄 License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

---

## 🆘 Support

- **Documentation**: [`docs/README.md`](docs/README.md)
- **Issues**: Use GitHub Issues for bug reports
- **Troubleshooting**: [`docs/troubleshooting/troubleshooting-guide.md`](docs/troubleshooting/troubleshooting-guide.md)
- **System Diagnostics**: Run `python aneos_menu.py → 4 → 1` for health check

---

## 🏆 Acknowledgments

**Special thanks to:**
- **NASA JPL** for SBDB and Horizons APIs
- **ESA** for DISCOS space debris data
- **Minor Planet Center** for orbital elements
- **Gaia mission** for precision astrometry
- **The SWARM development teams** for specialized analysis systems

---

*"If we find artificial NEOs, we will have answered one of humanity's greatest questions: Are we alone?"*
