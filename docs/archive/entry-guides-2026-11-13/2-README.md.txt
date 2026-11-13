# aNEOS Documentation

Welcome to the comprehensive documentation for the aNEOS (Advanced Near Earth Object detection System) platform.

## Documentation Structure

### Getting Started
- **[Installation Guide](user-guide/installation.md)** — Complete installation and setup
- **[Quick Start Guide](user-guide/quick-start.md)** — Get up and running quickly
- **[Menu System Guide](user-guide/menu-system.md)** — 15-option interactive menu usage
- **[User Guide](user-guide/user-guide.md)** — Full feature reference
- **[Profession Guide](user-guide/profession-guide.md)** — Use-case guides by role

### Scientific Foundation
- **[Scientific Documentation](scientific/scientific-documentation.md)** — Full methodology: Bayesian framework, Fisher's method, sigma-5 threshold, indicator categories
- **[Artificial NEOs Theory](scientific/theory.md)** — The core hypothesis: statistical signatures of artificial vs natural heliocentric objects
- **[Validation Integrity Audit](scientific/VALIDATION_INTEGRITY.md)** — Honest uncertainty assessment: what the F1=1.00 on N=4 actually means

### Architecture
- **[Architecture Decision Records](architecture/ADR.md)** — 60 ADRs documenting every significant design choice
- **[Domain-Driven Design](architecture/DDD.md)** — 11 Bounded Contexts: entities, value objects, domain events, context boundaries

### API Reference
- **[REST API Reference](api/rest-api.md)** — Key endpoint documentation
- **[OpenAPI Specification](api/openapi.json)** — Machine-readable, auto-generated (regenerate: `make spec`)

### Engineering
- **[Maturity Assessment](engineering/maturity_assessment.md)** — Stabilization findings and outstanding risks
- **[Sigma-5 Success Criteria](engineering/sigma5_success_criteria.md)** — Acceptance criteria for detection quality

### Deployment
- **[Deployment Guide](deployment/deployment-guide.md)** — Docker and production setup

### Troubleshooting
- **[Installation Issues](troubleshooting/installation.md)** — Installation-specific troubleshooting
- **[Troubleshooting Guide](troubleshooting/troubleshooting-guide.md)** — Common issues and fixes

### Release Notes
- **[Current State Summary](current-state-summary.md)** — v1.2.2 capabilities, limitations, and architecture overview
- **[v0.7.0 Release Notes](releases/v0.7.0.md)** — Legacy branch information

### Archive
- **[docs/archive/](archive/)** — Superseded phase plans and gap analyses (kept for historical reference)

## Quick Navigation

### For Scientists and Researchers
- Start with: [Artificial NEOs Theory](scientific/theory.md)
- Methodology: [Scientific Documentation](scientific/scientific-documentation.md)
- Honest caveats: [Validation Integrity Audit](scientific/VALIDATION_INTEGRITY.md)
- Menu guide: [Menu System Guide](user-guide/menu-system.md)

### For Developers / Contributors
- Start with: [CONTRIBUTING.md](../CONTRIBUTING.md) in the project root
- Architecture: [ADR.md](architecture/ADR.md) + [DDD.md](architecture/DDD.md)
- API: [REST API Reference](api/rest-api.md)

### For New Users
- [Quick Start Guide](user-guide/quick-start.md)
- [Installation Guide](user-guide/installation.md)
- [Troubleshooting Guide](troubleshooting/troubleshooting-guide.md)

## Getting Help

1. **Run the menu** — `python aneos.py` then choose Option 14 (Scientific Help) or Option 11 (System Health)
2. **Check the documentation** — most questions are answered in [Scientific Documentation](scientific/scientific-documentation.md)
3. **Community forum** — https://community.openastronomy.org/t/open-source-python-tool-for-checking-neo-anomalies-aneos/1374
4. **GitHub issues** — https://github.com/RobLe3/aneos-suite/issues

## 📖 Documentation Conventions

- **Code blocks** - Executable commands and code examples
- **File paths** - Relative to project root unless specified
- **Environment variables** - Configuration options
- **Menu paths** - Interactive menu navigation (e.g., Menu → Scientific Analysis → Single NEO Analysis)

## 🔄 Keeping Documentation Updated

The documentation is version-controlled with the code. When contributing:

1. Update relevant documentation with code changes
2. Follow the existing documentation structure
3. Include examples and use cases
4. Test all command examples
5. Update this index when adding new documentation

---

**Project Version**: 1.2.2 (Phase 24)
**Documentation Last Updated**: 2026-03-11
**Maintainer**: aNEOS Development Team
**Community**: https://community.openastronomy.org/t/open-source-python-tool-for-checking-neo-anomalies-aneos/1374