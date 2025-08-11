# **aNEOS User Workflow Guide**

## **Getting Started**

### **System Launch**
```bash
cd aneos-project
python3 aneos_menu.py
```

The system will initialize and display the main mission control interface.

## **Primary Workflows**

### **🎯 Quick Analysis: Single NEO Investigation**

**Navigation**: Main Menu → 1 (NEO Detection) → 1 (Individual Object Analysis)

**Purpose**: Deep analysis of a specific NEO with comprehensive validation

**Steps**:
1. Enter NEO designation (e.g., "2024 AB1", "Apophis")
2. System fetches orbital data from multiple sources
3. XVIII SWARM performs automated anomaly scoring
4. Multi-stage validation pipeline processes the object
5. Results displayed with confidence scores and detailed analysis

**Expected Output**:
- Overall anomaly score (0.0-1.0)
- Detailed breakdown across 6 clue categories
- Validation results from all analysis systems
- Professional report with visualizations

---

### **🔄 Continuous Monitoring: Complete Pipeline Analysis**

**Navigation**: Main Menu → 1 (NEO Detection) → 3 (Continuous Monitoring)

**Purpose**: Large-scale analysis with 200-year historical polling and automatic review

**Process Overview**:
```
📊 Historical Data Polling     ████████████ 100% ✅ Processing chunks...
🧠 XVIII SWARM First-Stage    ████████████ 100% ✅ Analyzing candidates...  
🔬 Multi-Stage Validation     ████████████ 100% ✅ Validating objects...
👨‍🔬 Expert Review Queue       ████████████ 100% ✅ Preparing final list...
```

**What Happens**:
1. **Historical Polling**: System retrieves 200 years of NEO data in intelligent chunks
2. **XVIII SWARM Review**: Automatic scoring identifies ~5,000 candidates from ~50,000 objects
3. **Multi-Stage Validation**: Comprehensive analysis reduces to ~500 validated candidates
4. **Expert Queue**: Final refinement produces ~50 high-priority investigation targets

**Results Location**: `neo_data/pipeline_results/pipeline_result_[timestamp].json`

---

### **📈 System Status & Health Monitoring**

**Navigation**: Main Menu → 7 (System Diagnostics) → Various options

**Available Diagnostics**:
- **Component Status**: Check all SWARM systems availability
- **API Health**: Verify NASA/ESA data source connectivity
- **Performance Metrics**: Review processing speed and efficiency
- **Cache Status**: Monitor data storage and cleanup needs

---

### **🎓 Learning Mode: Educational Workflows**

**Navigation**: Main Menu → 8 (Learning Center)

**Features**:
- **System Overview**: Interactive introduction to aNEOS capabilities
- **Anomaly Glossary**: Definitions of orbital characteristics and analysis terms
- **Workflow Tutorials**: Step-by-step guides for different user types
- **Sample Analysis**: Pre-configured examples with known anomalous objects

---

## **User Types & Recommended Workflows**

### **🔬 Research Scientists**

**Primary Workflow**: Continuous Monitoring for population studies
**Navigation**: Menu → 1 → 3 (200-year analysis)
**Benefits**: 
- Publication-ready statistical validation
- Comprehensive false positive prevention
- Reproducible results with audit trails
- Professional visualizations and reports

**Advanced Options**:
- Custom time period analysis: Menu → 1 → 4
- Individual object deep-dive: Menu → 1 → 1
- Database export: Menu → 5 → Various export options

### **🎓 Graduate Students & Amateur Astronomers**

**Primary Workflow**: Individual Object Analysis for learning
**Navigation**: Menu → 1 → 1 (Single object analysis)
**Benefits**:
- Educational explanations of analysis methods
- Detailed scoring breakdowns
- Clear visualization of anomaly indicators
- Safe learning environment with guided tutorials

**Learning Path**:
1. Start with Learning Center (Menu → 8)
2. Analyze famous NEOs (Apophis, Bennu, Ryugu)
3. Progress to small population studies
4. Advanced: Custom analysis workflows

### **🛡️ Planetary Defense Researchers**

**Primary Workflow**: Targeted object analysis with threat assessment
**Navigation**: Menu → 2 (Mission Intelligence) → Various threat analysis tools
**Benefits**:
- Rapid assessment of newly discovered objects
- Automated threat scoring and prioritization
- Integration with existing planetary defense databases
- Real-time monitoring capabilities

---

## **Data Management Workflows**

### **📁 Results & Data Export**

**Navigation**: Main Menu → 5 (Data Management)

**Export Options**:
- **CSV Export**: Spreadsheet-compatible candidate lists
- **JSON Export**: Machine-readable analysis results
- **Scientific Format**: Publication-ready data with metadata
- **Visualization Export**: High-quality plots and charts

**Data Locations**:
```
neo_data/
├── pipeline_results/         # Complete analysis results
├── historical_cache/         # Chunked polling cache (for performance)
├── historical_results/       # Long-term analysis storage
└── exports/                  # User-requested data exports
```

### **🧹 Cache & Storage Management**

**Automatic Management**:
- System automatically manages cache for optimal performance
- Old results archived with timestamps
- Intelligent cleanup prevents storage overflow

**Manual Management**:
- Menu → 7 (System Diagnostics) → Cache Management
- Clear specific date ranges or analysis types
- Export before cleanup for permanent storage

---

## **Advanced Workflows**

### **🔧 Custom Analysis Configuration**

**Navigation**: Main Menu → 6 (Advanced Tools) → Pipeline Configuration

**Configurable Parameters**:
- **Scoring Thresholds**: Adjust XVIII SWARM sensitivity
- **Time Periods**: Custom historical polling ranges
- **Analysis Depth**: Enable/disable specific validation systems
- **Output Formats**: Customize result reporting

### **🌐 API & Integration Workflows**

**RESTful API** (if enabled):
```bash
# Start API server
python3 -m aneos_core.api.server

# Example API calls
curl http://localhost:8000/api/analyze/2024AB1
curl http://localhost:8000/api/pipeline/status
```

**Jupyter Integration**:
```python
from aneos_core.integration.pipeline_integration import PipelineIntegration
integration = PipelineIntegration()
result = await integration.run_historical_polling_workflow(years_back=10)
```

---

## **Understanding Results**

### **Anomaly Scores**
- **0.0-0.3**: Natural orbital characteristics
- **0.3-0.6**: Potentially interesting anomalies
- **0.6-0.8**: Significant anomalies requiring investigation
- **0.8-1.0**: Highly anomalous objects warranting immediate analysis

### **XVIII SWARM Categories**
1. **Orbital Eccentricity**: Unusual elliptical patterns
2. **Inclination Anomalies**: Non-ecliptic orbital planes
3. **Velocity Patterns**: Acceleration/deceleration inconsistencies
4. **Close Approach Regularity**: Suspiciously precise return cycles
5. **Thermal Signatures**: Unusual heat emission patterns
6. **Radar Characteristics**: Anomalous material composition indicators

### **Validation Pipeline Results**
- **KAPPA SWARM**: Radar polarization analysis results
- **LAMBDA SWARM**: Thermal-IR signature validation
- **CLAUDETTE SWARM**: Statistical false positive assessment
- **MU SWARM**: Astrometric precision verification
- **Spectral Analysis**: Multi-wavelength signature validation

---

## **Troubleshooting**

### **Common Issues**

**"No objects found"**:
- Check internet connectivity for NASA API access
- Verify date ranges are valid
- Try smaller time periods first

**"Pipeline components not available"**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: requires 3.8+
- Verify file permissions for neo_data/ directory

**"Slow performance"**:
- System processes large datasets - 200-year polls take 10-30 minutes
- Use smaller time periods for testing: Menu → 1 → 4 (Custom Analysis)
- Monitor system resources during large analysis sessions

### **Performance Optimization**
- **First Run**: May take longer due to cache building
- **Subsequent Runs**: Cache provides significant speedup
- **Memory**: 8GB+ recommended for 200-year analysis
- **Storage**: 10GB+ recommended for historical caching

---

## **Best Practices**

### **For New Users**
1. **Start Small**: Begin with Learning Center and individual object analysis
2. **Understand Scoring**: Review anomaly categories before large-scale analysis
3. **Verify Results**: Cross-reference findings with known astronomical databases
4. **Build Experience**: Progress from single objects to population studies

### **For Researchers**
1. **Document Parameters**: Record analysis configurations for reproducibility
2. **Validate Methods**: Understand statistical methods and limitations
3. **Peer Review**: Share methodology and results for external validation
4. **Contribute**: Report bugs and suggest improvements to the research community

---

*This workflow guide provides comprehensive coverage of the aNEOS platform capabilities, enabling users to effectively leverage the sophisticated analysis systems for scientific research and discovery.*