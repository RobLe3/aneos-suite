# aNEOS Development Roadmap

## Active plan — 2026-11-13

Programme [#2](https://github.com/RobLe3/aneos-suite/issues/2). The
[evidence-backed discovery/integrity review](docs/engineering/discovery-integrity-roadmap-review-2026-11-13.md)
is the current dependency map, proposed-contract specification and verification plan.
The older 2025 plan below is historical design, **not current readiness evidence**.

| Horizon | Work / dependencies | Exit gate |
|---|---|---|
| Short term | #83 receipts → #84 bounded root/menu/CLI polling (#73); #16/#40 source retention/eligibility; #86 exploratory inventory/contract; #87 lineage/threat model; #17/#41/#44/#46 docs and QA | Existing CI gates plus native positive, empty/missing/invalid/outage/replay controls; preserved histories/rankings. No science/accuracy claim |
| Medium term | Qualified #86 groups → #10 conditional/dependency-aware follow-up; #40/#13 → #87 compatible comparisons → #7 KY26 history; #5/#7 → #8 valid fits → #9 hypothesis tests; #11 independent controls; #12/#42 presentation | Measured useful coverage, reproducibility and false-positive control, with operation-specific scientific prerequisites and unresolved outcomes |
| Long term | #10/#11 prospective targets, object/time holdouts, Brier/reliability/PR; calibrated search-wide significance only if supportable; #87 independent checkpoints/security/recovery; #13/#14 cases/predictions | Independent generalization/review, provenance and recoverability within explicit threat-model limits |

Horizons express prerequisites, not dates. Beta **2.2.0b2** is published; main has
later scoped improvements. PR85's foundation is separate from #84 integration and
scientific qualification. The next integrated polling version needs a scope/CI
check after #84; no calendar ETA is substantiated yet. Do not delay qualified
engineering increments for all future science, or label them scientifically validated.

### Rules for every horizon

- Preserve ranking policies and the 5.0 ranking-unit threshold. No tuning to KY26;
  never call an uncalibrated rank five sigma.
- Separate exploratory anomalies, quality/completeness, calibrated residual
  significance, follow-up priority, physical origin and historical integrity.
- Restore diagnostics, not unsafe legacy origin classification. No parallel engine,
  persistence system or new bounded context without demonstrated necessity.
- Preserve revisions append-only. Hashes prove consistency with saved bytes, not
  initial authenticity. A revised orbit/discrepancy/outage alone is not tampering.
- 100% query coverage needs an explicit eligible denominator. Scientific completeness
  cannot be manufactured: unavailable measurements stay missing.
- Brier scoring needs a defined independently verifiable outcome/window. Unknown
  origin is not natural truth; synthetic tests are not independent validation.
- No hidden polling during scoring/visualization. Provider budgets/licensing, actual
  menu tests, uncertainty/provenance and existing quality gates remain required.

## Historical plan — August 2025 (superseded as current status)

<details>
<summary>Archived aspirations and original phase claims; not acceptance evidence</summary>

# aNEOS Development Roadmap
**Advanced Near Earth Object detection System**

This document outlines the completed development phases and future enhancements for the aNEOS platform, building upon the foundational work from the [neo-analyzer-repo](https://github.com/RobLe3/neo-analyzer-repo).

## ✅ Stabilized Phases (August 2025)

### Phase 1: Core System Foundation ✅
- **✅ Installation System**: Comprehensive installer with dependency management (`install.py`)
- **✅ Project Architecture**: Modular structure with `aneos_core/`, `aneos_api/`, and organized directories
- **✅ Error Handling**: Robust `@safe_execute` decorator and graceful error recovery
- **✅ Security**: Custom-generated API keys for each installation, no hardcoded secrets

### Phase 2: NEO Analysis Engine ✅
- **✅ Simple NEO Analyzer**: Basic artificial signature detection (`simple_neo_analyzer.py`)
- **✅ Enhanced NEO Poller**: Multi-source data enrichment with TAS scoring (`enhanced_neo_poller.py`)
- **✅ Data Quality System**: 100% completeness validation before analysis
- **✅ Professional Reporting**: Academic-quality output with AI validation

### Phase 3: System Integration ✅  
- **✅ Interactive Menu System**: 25+ features across 6 categories (`aneos_menu.py`)
- **✅ Command-Line Interface**: Streamlined CLI with help system (`aneos.py`)
- **✅ API Services**: 52 functional endpoints with authentication (`aneos_api/`)
- **✅ Database Integration**: SQLite with 7 tables, SQLAlchemy 2.0+ compatibility

### Phase 4: Testing & Validation ✅
- **✅ Comprehensive Testing**: 100% basic menu feature validation
- **✅ System Health Monitoring**: Complete status checks and diagnostics
- **✅ Import Resolution**: Fixed NumPy 2.x compatibility and authentication issues
- **✅ End-to-End Validation**: All core functionality operational

## 🎯 Current Status: Stabilization In Progress

**System State**: Core flows operate in development, but real-data polling still
leans on mock fallbacks when optional integrations are absent.
**Test Coverage**: 61 automated tests cover cache, configuration, detection,
and integration harnesses; further operational drills are planned.
**Documentation**: Installation and quick-start guides exist, yet several
sections still overstate production readiness and need revision.
**Stability**: Critical regressions from the 0.7 upgrade have been mitigated,
while external dependency monitoring and sigma-5 validation remain on the
roadmap.

### Available Capabilities
```bash
# System health and basic analysis
python aneos.py status                    # System diagnostics
python aneos.py simple "test"             # Basic artificial detection
python aneos.py help                      # Command reference

# Enhanced analysis
python enhanced_neo_poller.py --period 1w # Multi-source data enrichment
python aneos.py api --dev                 # Web API and dashboard

# Interactive analysis
python aneos.py                           # Full menu system
```

## 🚧 Future Development Phases

### Phase 5: Academic Enhancement (Medium Priority)
- **Statistical Framework**: Formal hypothesis testing with p-values and confidence intervals
- **Physical Modeling**: Advanced orbital dynamics including Yarkovsky effects
- **Hardware Cross-Matching**: TLE database integration to exclude known satellites  
- **Synthetic Population Validation**: False positive rate calibration with simulated data

### Phase 6: Advanced Analytics (Low Priority)
- **Machine Learning Pipeline**: Deep learning models for pattern recognition
- **Real-Time Processing**: Stream processing for continuous monitoring
- **Distributed Computing**: Kubernetes deployment for large-scale analysis
- **External Integrations**: Third-party astronomy service connections

### Phase 7: Research Publication (Future)
- **Peer Review Preparation**: Academic paper-ready documentation and methodology
- **Collaborative Framework**: Multi-institution research coordination
- **Open Science Integration**: Data sharing with astronomical community
- **SETI Collaboration**: Integration with existing search for intelligence programs

## 📊 Technical Roadmap

### Infrastructure Improvements
- **Performance Optimization**: Caching and parallel processing enhancements
- **Monitoring & Observability**: Comprehensive metrics and alerting
- **Security Hardening**: Production-grade authentication and authorization
- **Backup & Recovery**: Data persistence and disaster recovery systems

### Analysis Capabilities
- **Multi-Modal Detection**: Enhanced evidence fusion from independent sources
- **Historical Analysis**: 200-year comprehensive orbital pattern analysis
- **Anomaly Classification**: Refined artificial vs natural object categorization
- **Confidence Scoring**: Statistical uncertainty quantification improvements

## 🔬 Scientific Development

### Methodology Enhancement
- **Sigma 5 Statistical Validation**: Meeting astronomical discovery publication standards
- **Ground Truth Datasets**: Verified artificial vs natural object libraries
- **Cross-Validation**: Independent verification systems preventing false positives
- **Reproducibility Framework**: Complete methodology documentation for peer review

### Research Applications
- **Survey Completeness**: Comprehensive NEO population characterization
- **Discovery Pipeline**: Automated flagging of potentially artificial objects
- **Evidence Analysis**: Multi-parameter correlation for detection confidence
- **Publication Pipeline**: Academic paper generation from analysis results

## 🤝 Community & Collaboration

### Open Science
- **Code Availability**: Open source development with community contributions
- **Data Sharing**: Standardized formats for research collaboration
- **Documentation**: Comprehensive guides for researchers and developers
- **Validation**: Independent verification of detection methodologies

### Academic Integration
- **University Partnerships**: Research collaboration opportunities
- **Student Projects**: Educational applications for astronomy coursework
- **Conference Presentations**: Scientific meeting participation
- **Journal Publications**: Peer-reviewed research dissemination

## 📈 Success Metrics

### Technical Success
- **System Reliability**: 99.9% uptime for analysis operations
- **Processing Speed**: <1 second single object analysis, <30 seconds batch processing
- **Accuracy**: <5.7×10⁻⁷ false positive rate maintained
- **Scalability**: Handle 10,000+ objects per analysis session

### Scientific Success
- **Detection Validation**: Successful identification of known artificial objects
- **Academic Recognition**: Peer-reviewed publications and citations
- **Community Adoption**: Usage by independent research groups
- **Discovery Potential**: Identification of previously unknown artificial objects

## 🛠️ Development Guidelines

### Contributing
- All development follows the C&C + Implementation + Q&A framework
- External validation required for detection algorithm changes
- Comprehensive testing mandatory for all feature additions
- Documentation updates required for all user-facing changes

### Quality Standards  
- **Code Quality**: Comprehensive error handling and logging
- **Testing Coverage**: 100% validation for core functionality
- **Documentation**: Complete user guides and technical references
- **Performance**: Sub-second response times for interactive operations

---

**Evolution from neo-analyzer-repo**: This roadmap traces the path from the
foundational theoretical work toward an operationally verified sigma-5
detection platform. Additional hardening is required before claiming full
production readiness.

**Next Review**: After integration health checks and end-to-end sigma-5
validations are documented.

---

*This roadmap reflects the stabilization goals for aNEOS and outlines the work
still needed to reach publication-grade artificial NEO detection.*
</details>
