# aNEOS System Requirements

Comprehensive system requirements and specifications for aNEOS deployment

## Table of Contents

1. [Overview](#overview)
2. [Minimum System Requirements](#minimum-system-requirements)
3. [Recommended System Requirements](#recommended-system-requirements)
4. [Production System Requirements](#production-system-requirements)
5. [Software Dependencies](#software-dependencies)
6. [Network Requirements](#network-requirements)
7. [Storage Requirements](#storage-requirements)
8. [Performance Specifications](#performance-specifications)
9. [Security Requirements](#security-requirements)
10. [Scalability Considerations](#scalability-considerations)
11. [Platform-Specific Requirements](#platform-specific-requirements)
12. [Cloud Platform Requirements](#cloud-platform-requirements)

---

## Overview

aNEOS is designed to operate across various computing environments, from development laptops to enterprise production clusters. This document outlines the hardware, software, and infrastructure requirements for different deployment scenarios.

### System Tiers

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Development   │  │     Testing     │  │     Staging     │  │   Production    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Laptop/Desktop  │  │ Single Server   │  │ Small Cluster   │  │ Full Cluster    │
│ Local Dev Only  │  │ Integration     │  │ Pre-production  │  │ High Availability│
│ 8GB RAM         │  │ 16GB RAM        │  │ 64GB+ RAM       │  │ 256GB+ RAM      │
│ 2-4 CPU cores   │  │ 4-8 CPU cores   │  │ 16+ CPU cores   │  │ 64+ CPU cores   │
│ 50GB Storage    │  │ 200GB Storage   │  │ 1TB+ Storage    │  │ 10TB+ Storage   │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Minimum System Requirements

### Hardware Requirements

#### CPU
- **Architecture**: x86_64 (AMD64)
- **Cores**: 2 physical cores (4 threads)
- **Clock Speed**: 2.0 GHz minimum
- **Features**: SSE4.2, AVX support recommended

#### Memory
- **RAM**: 4GB minimum
- **Available**: At least 2GB free for aNEOS processes
- **Swap**: 2GB recommended (if RAM < 8GB)

#### Storage
- **Free Space**: 20GB minimum
- **Type**: Any (HDD acceptable for development)
- **IOPS**: No specific requirement
- **File System**: ext4, NTFS, APFS, or any POSIX-compliant FS

#### Network
- **Internet Connection**: Required for NEO data APIs
- **Bandwidth**: 1 Mbps minimum
- **Latency**: <1000ms to NASA/ESA servers

### Supported Platforms

#### Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+, RHEL 7+, Debian 9+
- **macOS**: 10.14 (Mojave) or later
- **Windows**: Windows 10 or Windows Server 2016+

#### Container Platforms
- **Docker**: 19.03+
- **Podman**: 3.0+
- **containerd**: 1.4+

#### Python Versions
- **Python**: 3.8.0 - 3.11.x
- **Package Manager**: pip 20.0+
- **Virtual Environment**: venv, conda, or virtualenv

---

## Recommended System Requirements

### Hardware Specifications

#### CPU
- **Architecture**: x86_64 with AVX2 support
- **Cores**: 4-8 physical cores (8-16 threads)
- **Clock Speed**: 3.0 GHz base frequency
- **Features**: AVX2, FMA3 for ML acceleration
- **Examples**:
  - Intel: Core i5-8400, Core i7-8700, Xeon E-2288G
  - AMD: Ryzen 5 3600, Ryzen 7 3700X, EPYC 7282

#### Memory
- **RAM**: 16GB total
- **Available**: 8GB dedicated to aNEOS
- **Type**: DDR4-2400 or faster
- **ECC**: Recommended for production

#### Storage
- **Primary**: 500GB+ SSD
- **IOPS**: 5000+ read, 3000+ write
- **Sequential**: 500MB/s read, 300MB/s write
- **File System**: ext4 (Linux), APFS (macOS), NTFS (Windows)
- **Backup**: Additional storage for data backup

#### Network
- **Internet**: 10+ Mbps broadband
- **Latency**: <200ms to major NEO APIs
- **Reliability**: 99.9%+ uptime

### Software Environment

#### Operating Systems
- **Linux**: Ubuntu 20.04 LTS, CentOS 8, RHEL 8
- **macOS**: macOS 11 (Big Sur) or later
- **Windows**: Windows 10 Pro, Windows Server 2019

#### Python Environment
- **Python**: 3.9.x or 3.10.x
- **Virtual Environment**: conda recommended
- **Package Manager**: pip 21.0+

---

## Production System Requirements

### High-Availability Architecture

```
Load Balancer Tier:
├── 2x Load Balancer nodes (4 CPU, 8GB RAM)
│
Application Tier:
├── 3-5x API nodes (8 CPU, 16GB RAM each)
├── 2-4x Worker nodes (16 CPU, 32GB RAM each)
├── 1x ML training node (32 CPU, 64GB RAM, GPU optional)
│
Data Tier:
├── PostgreSQL cluster (16 CPU, 32GB RAM per node)
├── Redis cluster (8 CPU, 16GB RAM per node)
├── Object storage (distributed/cloud)
│
Monitoring Tier:
├── Prometheus + Grafana (4 CPU, 8GB RAM)
├── ELK Stack (8 CPU, 16GB RAM)
```

### Hardware Specifications

#### Compute Nodes

**API Server Nodes (3-5 instances)**
- **CPU**: 8 cores (16 threads), 3.2GHz+
- **RAM**: 16GB DDR4 ECC
- **Storage**: 200GB NVMe SSD (OS + logs)
- **Network**: 1Gbps+ Ethernet

**Analysis Worker Nodes (2-4 instances)**
- **CPU**: 16 cores (32 threads), 3.0GHz+
- **RAM**: 32GB DDR4 ECC
- **Storage**: 500GB NVMe SSD
- **Network**: 1Gbps+ Ethernet
- **GPU**: Optional - CUDA 11.0+ compatible

**ML Training Node (1 instance)**
- **CPU**: 32 cores (64 threads), 2.8GHz+
- **RAM**: 64GB+ DDR4 ECC
- **Storage**: 1TB NVMe SSD
- **GPU**: NVIDIA RTX 4000/A100 or equivalent
- **Network**: 10Gbps+ recommended

**Database Nodes (2-3 instances)**
- **CPU**: 16 cores (32 threads), 3.0GHz+
- **RAM**: 32GB+ DDR4 ECC
- **Storage**: 1TB+ NVMe SSD (high IOPS)
- **Network**: 10Gbps+ for replication
- **RAID**: RAID 10 for data protection

#### Storage Requirements

**Database Storage**
- **Capacity**: 2TB+ usable (with growth)
- **Performance**: 50,000+ IOPS
- **Redundancy**: RAID 10 or distributed storage
- **Backup**: 3x database size for backups

**Object Storage**
- **Capacity**: 10TB+ (for models, analysis results)
- **Durability**: 99.999999999% (11 9's)
- **Availability**: 99.99%+
- **Access**: S3-compatible API

#### Network Infrastructure

**Internal Network**
- **Bandwidth**: 10Gbps+ between tiers
- **Latency**: <1ms between nodes
- **Redundancy**: Dual network paths

**External Network**
- **Bandwidth**: 1Gbps+ internet connection
- **CDN**: Content delivery network for static assets
- **DDoS Protection**: Layer 3/4 and Layer 7 protection

### Performance Requirements

#### Response Time Targets
- **Health Check**: <100ms
- **Single NEO Analysis**: <30 seconds
- **Batch Analysis (100 NEOs)**: <10 minutes
- **API Endpoints**: <2 seconds (95th percentile)
- **Dashboard Load**: <3 seconds

#### Throughput Targets
- **Concurrent Users**: 100+
- **Analysis Requests**: 1000+ per hour
- **API Requests**: 10,000+ per hour
- **Batch Jobs**: 10+ concurrent

#### Availability Targets
- **System Uptime**: 99.9%+ (8.77 hours downtime/year)
- **Planned Maintenance**: <4 hours/month
- **Recovery Time**: <30 minutes (RTO)
- **Data Loss**: <5 minutes (RPO)

---

## Software Dependencies

### Core Dependencies

#### Python Packages

**Required (Core Functionality)**
```
Python >= 3.8.0
fastapi >= 0.95.0
uvicorn >= 0.20.0
sqlalchemy >= 2.0.0
psycopg2-binary >= 2.9.0  # PostgreSQL
redis >= 4.5.0
requests >= 2.28.0
numpy >= 1.21.0
pandas >= 1.5.0
pydantic >= 1.10.0
rich >= 13.0.0
click >= 8.0.0
```

**Optional (Enhanced Features)**
```
# Machine Learning
torch >= 1.13.0
scikit-learn >= 1.2.0
joblib >= 1.2.0

# Scientific Computing
scipy >= 1.9.0
matplotlib >= 3.6.0
astropy >= 5.2.0

# Development Tools
pytest >= 7.2.0
black >= 22.10.0
flake8 >= 5.0.0
mypy >= 0.991
```

#### System Libraries

**Linux (Ubuntu/Debian)**
```bash
# Build tools
build-essential
python3-dev
libpq-dev
libssl-dev
libffi-dev

# Optional: CUDA support
nvidia-cuda-toolkit
libcudnn8-dev

# Monitoring tools
htop
iotop
nethogs
```

**macOS**
```bash
# Xcode command line tools
xcode-select --install

# Homebrew packages
brew install postgresql
brew install redis
brew install curl
```

**Windows**
```powershell
# Visual C++ Build Tools
# Download from Microsoft

# PostgreSQL client
# Download from postgresql.org

# Redis (via WSL or Windows port)
```

### Database Requirements

#### PostgreSQL (Recommended for Production)
- **Version**: 12.0+, 15.0+ recommended
- **Extensions**: None required (using standard SQL)
- **Configuration**: Tuned for OLTP workloads
- **Memory**: 25% of system RAM for shared_buffers
- **Connections**: 100+ concurrent connections

#### SQLite (Development/Testing)
- **Version**: 3.35.0+
- **Features**: WAL mode enabled
- **File**: Single database file
- **Limitations**: Single writer, limited concurrency

#### Redis (Caching/Sessions)
- **Version**: 6.0+, 7.0+ recommended
- **Memory**: 4GB+ for production
- **Persistence**: AOF + RDB snapshots
- **Clustering**: Recommended for high availability

### Container Requirements

#### Docker Environment
- **Docker Engine**: 20.10+
- **Docker Compose**: 2.0+
- **Base Images**: python:3.10-slim, postgres:15, redis:7-alpine
- **Registry**: Docker Hub or private registry

#### Kubernetes Environment
- **Version**: 1.20+, 1.25+ recommended
- **Container Runtime**: containerd 1.5+
- **Storage Class**: Fast SSD storage class
- **Networking**: CNI-compatible network plugin
- **Ingress**: nginx-ingress or similar

---

## Network Requirements

### Connectivity Requirements

#### External APIs
- **NASA SBDB API**: https://ssd-api.jpl.nasa.gov
- **NASA CAD API**: https://ssd-api.jpl.nasa.gov/cad.api  
- **JPL Horizons**: https://ssd.jpl.nasa.gov/api/horizons.api
- **NEODyS**: https://newton.spacedys.com/neodys/
- **MPC**: https://www.minorplanetcenter.net/

#### Firewall Rules

**Outbound Rules (Required)**
```
Protocol: HTTPS (443/tcp)
Destinations: 
  - ssd-api.jpl.nasa.gov
  - ssd.jpl.nasa.gov  
  - newton.spacedys.com
  - minorplanetcenter.net

Protocol: HTTP (80/tcp)
Purpose: Certificate validation, package downloads

Protocol: DNS (53/udp)
Purpose: Domain name resolution
```

**Inbound Rules (Application)**
```
Port 8000/tcp: aNEOS API service
Port 5432/tcp: PostgreSQL (internal only)
Port 6379/tcp: Redis (internal only)
Port 3000/tcp: Grafana dashboard
Port 9090/tcp: Prometheus metrics
```

#### Load Balancer Requirements
- **SSL Termination**: TLS 1.2+ support
- **Health Checks**: HTTP health endpoints
- **Session Persistence**: Cookie-based (optional)
- **Rate Limiting**: Per-IP and per-user limits
- **DDoS Protection**: Basic flood protection

### Bandwidth and Latency

#### Bandwidth Requirements
- **Development**: 5+ Mbps
- **Testing**: 10+ Mbps  
- **Production**: 100+ Mbps
- **Peak Usage**: 500+ Mbps (burst)

#### Latency Requirements
- **NASA APIs**: <500ms preferred, <1000ms acceptable
- **Database**: <10ms (local network)
- **Cache**: <1ms (local network)
- **User Requests**: <100ms (local processing)

---

## Storage Requirements

### Storage Tiers

#### Tier 1: Hot Storage (High Performance)
**Purpose**: Active databases, caches, logs
```
Type: NVMe SSD
IOPS: 50,000+ read, 30,000+ write
Latency: <1ms
Capacity: 2TB+
Redundancy: RAID 10 or distributed
```

#### Tier 2: Warm Storage (Balanced)
**Purpose**: Analysis results, models, backups
```  
Type: SATA SSD or fast HDD
IOPS: 5,000+ read, 3,000+ write
Latency: <5ms
Capacity: 10TB+
Redundancy: RAID 5/6 or erasure coding
```

#### Tier 3: Cold Storage (Archive)
**Purpose**: Historical data, long-term backups
```
Type: Cloud object storage or tape
Access Time: Minutes to hours
Capacity: 100TB+
Durability: 99.999999999%
Cost: Optimized for storage cost
```

### Storage Components

#### Database Storage
- **PostgreSQL Data**: 500GB+ (growth: 10GB/month)
- **Indexes**: 20% of data size
- **WAL/Logs**: 100GB+ (with archiving)
- **Backups**: 3x database size

#### Application Storage
- **ML Models**: 50GB+ (multiple model versions)
- **Analysis Cache**: 100GB+ (configurable TTL)
- **Log Files**: 50GB+ (with rotation)
- **Temporary Files**: 20GB+ (processing workspace)

#### Media Storage
- **Documentation**: 1GB
- **Static Assets**: 500MB
- **User Uploads**: Variable (if implemented)

### Backup Strategy

#### Recovery Point Objective (RPO)
- **Critical Data**: 15 minutes
- **Analysis Results**: 1 hour
- **Configuration**: 24 hours

#### Recovery Time Objective (RTO)
- **System Restore**: 30 minutes
- **Data Restore**: 2 hours
- **Full Rebuild**: 4 hours

#### Backup Types
```
Type 1: Continuous (WAL shipping)
Frequency: Real-time
Retention: 7 days

Type 2: Incremental (daily)
Frequency: 24 hours
Retention: 30 days

Type 3: Full (weekly)
Frequency: 7 days  
Retention: 12 weeks

Type 4: Archive (monthly)
Frequency: 30 days
Retention: 7 years
```

---

## Performance Specifications

### Response Time Requirements

#### API Endpoints
```
Endpoint                    Target    Limit
/health                     <50ms     100ms
/api/v1/analysis/single     <10s      30s
/api/v1/analysis/batch      <5min     15min
/api/v1/monitoring/metrics  <200ms    500ms
/dashboard/*                <1s       3s
```

#### Analysis Performance
```
Operation               Target      Scale
Single NEO Analysis     <15s        Light workload
Batch Analysis (100)    <8min       Medium workload  
ML Training             <2hr        Heavy workload
Model Inference         <100ms      Per prediction
```

### Throughput Requirements

#### Concurrent Operations
```
Metric                  Min    Recommended    Peak
Concurrent Users        10     50             200
API Requests/sec        5      25             100
Analysis Jobs/hr        10     60             300
ML Predictions/sec      1      10             50
Database Connections    20     100            200
```

#### Data Processing Rates
```
Data Type               Rate           Scale
NEO Records/sec         10+            Ingestion
Feature Extraction      100+/sec       ML Pipeline  
Analysis Results        50+/sec        Output
Batch Processing        1000+/hr       Large datasets
```

### Resource Utilization Targets

#### CPU Utilization
- **Development**: <50% average, <80% peak
- **Production**: <70% average, <90% peak
- **Analysis Workers**: <80% average, 100% peak acceptable

#### Memory Utilization
- **System Reserved**: 25% of total RAM
- **Application**: <70% of remaining RAM
- **Cache Hit Rate**: >90% for frequently accessed data

#### Storage Utilization  
- **Database**: <80% of allocated space
- **File System**: <85% of total capacity
- **I/O Wait**: <5% average, <15% peak

#### Network Utilization
- **Bandwidth**: <50% of available capacity
- **Packet Loss**: <0.1%
- **Connection Pool**: <80% of maximum connections

---

## Security Requirements

### System Hardening

#### Operating Systems
- **Updates**: Security patches within 30 days
- **Services**: Minimal required services only
- **Firewall**: Host-based firewall configured
- **Users**: No shared accounts, sudo access controlled
- **SSH**: Key-based authentication, no root login

#### Application Security
- **Authentication**: Multi-factor authentication supported
- **Authorization**: Role-based access control (RBAC)
- **Session Management**: Secure session handling
- **Input Validation**: All inputs validated and sanitized
- **Error Handling**: No sensitive information in error messages

### Network Security

#### Encryption
- **In Transit**: TLS 1.2+ for all communications
- **At Rest**: Database encryption (if supported)
- **Keys**: Secure key management system
- **Certificates**: Automatic renewal (Let's Encrypt or CA)

#### Access Control
- **VPN**: Required for administrative access (production)
- **Bastion Hosts**: Jump servers for production access
- **Network Segmentation**: DMZ for public services
- **Intrusion Detection**: Network-based IDS/IPS

### Compliance Requirements

#### Data Protection
- **Personal Data**: GDPR compliance (if applicable)
- **Research Data**: Institutional review board approval
- **Export Control**: ITAR/EAR compliance check
- **Data Residency**: Regional data storage requirements

#### Audit Requirements
- **Access Logs**: All administrative actions logged
- **Change Management**: All configuration changes tracked
- **Security Events**: Security incidents logged and alerted
- **Compliance Reporting**: Regular compliance reports generated

---

## Scalability Considerations

### Horizontal Scaling

#### Stateless Components
```
Component           Scaling Method      Max Instances
API Servers         Load balancer       20+
Analysis Workers    Queue-based         50+
ML Inference        Auto-scaling        10+
Dashboard           Load balancer       5+
```

#### Stateful Components
```
Component           Scaling Method      Considerations
PostgreSQL         Master-replica       Read replicas
Redis              Cluster mode         Sharding
Object Storage     Distributed          Elastic scaling
```

### Vertical Scaling

#### Resource Limits
```
Component          CPU Limit    Memory Limit    Storage Limit
API Server         16 cores     32GB           500GB
Worker Node        32 cores     64GB           1TB  
Database           64 cores     256GB          10TB
ML Training        128 cores    512GB          5TB
```

### Performance Scaling

#### Load Testing Targets
```
Test Type          Duration    Load Level      Success Criteria
Smoke Test         5 min       10% normal      Zero errors
Load Test          30 min      100% normal     <2% errors, RTO met
Stress Test        60 min      150% normal     Graceful degradation
Spike Test         10 min      500% peak       System recovers
Volume Test        8 hours     Normal load     No memory leaks
```

#### Auto-scaling Configuration
```
Metric             Scale Up     Scale Down    Cooldown
CPU Utilization    >70%         <30%         5 min
Memory Usage       >80%         <40%         5 min
Request Rate       >100/sec     <20/sec      10 min
Queue Length       >50          <10          5 min
```

---

## Platform-Specific Requirements

### Linux Distributions

#### Ubuntu 20.04/22.04 LTS
```bash
# System packages
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
sudo apt install postgresql-client redis-tools curl wget
sudo apt install build-essential libssl-dev libffi-dev libpq-dev

# Optional: NVIDIA drivers for ML
sudo apt install nvidia-driver-515 nvidia-cuda-toolkit

# System configuration
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'net.core.somaxconn=65535' | sudo tee -a /etc/sysctl.conf
```

#### CentOS 8/RHEL 8
```bash
# Enable repositories  
sudo dnf install epel-release
sudo dnf config-manager --set-enabled powertools

# System packages
sudo dnf install python39 python39-devel
sudo dnf install postgresql redis curl wget
sudo dnf install gcc gcc-c++ openssl-devel libffi-devel postgresql-devel

# SELinux configuration (if enabled)
sudo setsebool -P httpd_can_network_connect 1
```

### macOS (Development)

#### macOS 11+ (Big Sur/Monterey/Ventura)
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 postgresql redis
brew install --cask docker

# Development tools
xcode-select --install

# Optional: GPU support (Apple Silicon)
# MPS (Metal Performance Shaders) support in PyTorch 1.12+
```

#### Resource Limits
```bash
# Increase file descriptor limits
echo 'ulimit -n 65536' >> ~/.zshrc  # or ~/.bash_profile
echo 'kern.maxfiles=65536' | sudo tee -a /etc/sysctl.conf
echo 'kern.maxfilesperproc=32768' | sudo tee -a /etc/sysctl.conf
```

### Windows (Development/Testing)

#### Windows 10/11 Pro
```powershell
# Install Python 3.10
# Download from python.org

# Install Git for Windows
winget install Git.Git

# Install Docker Desktop
winget install Docker.DockerDesktop

# Windows Subsystem for Linux (WSL2)
wsl --install -d Ubuntu-20.04
```

#### Performance Tuning
```powershell
# Increase virtual memory
# Control Panel > System > Advanced > Performance Settings > Advanced > Virtual Memory

# Windows Defender exclusions
Add-MpPreference -ExclusionPath "C:\path\to\aneos-project"

# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Cloud Platform Requirements

### Amazon Web Services (AWS)

#### Compute Services
```
Service             Instance Type    vCPU    Memory    Storage    Cost/Month
API Servers         t3.large         2       8GB       EBS        ~$60
Worker Nodes        m5.xlarge        4       16GB      EBS        ~$140  
ML Training         p3.2xlarge       8       61GB      EBS+GPU    ~$2,000
Database            db.r5.xlarge     4       32GB      EBS        ~$350
Load Balancer       ALB              -       -         -          ~$25
```

#### Storage Services
```
Service             Type             Capacity    IOPS       Cost/Month
Database Storage    EBS gp3          1TB         3000       ~$80
Object Storage      S3 Standard      10TB        -          ~$230
Backup Storage      S3 IA            5TB         -          ~$65
Archive Storage     S3 Glacier       100TB       -          ~$400
```

#### Network Services
```
Service             Configuration              Cost/Month
VPC                 Multi-AZ, 3 subnets       Free
CloudFront          Global CDN                 ~$20
Route 53            DNS hosting                ~$0.50
NAT Gateway         HA configuration           ~$45
```

### Google Cloud Platform (GCP)

#### Compute Services
```
Service             Machine Type     vCPU    Memory    Storage    Cost/Month
API Servers         n1-standard-2    2       7.5GB     SSD        ~$50
Worker Nodes        n1-standard-4    4       15GB      SSD        ~$120
ML Training         n1-highmem-8     8       52GB      SSD+GPU    ~$1,800
Database            db-n1-standard-4 4       15GB      SSD        ~$320
Load Balancer       Global LB        -       -         -          ~$20
```

#### Storage Services
```
Service             Type             Capacity    IOPS       Cost/Month
Database Storage    SSD Persistent   1TB         3000       ~$170
Object Storage      Standard         10TB        -          ~$200
Backup Storage      Nearline         5TB         -          ~$50
Archive Storage     Coldline         100TB       -          ~$400
```

### Microsoft Azure

#### Compute Services
```
Service             VM Size          vCPU    Memory    Storage    Cost/Month
API Servers         Standard_B2ms    2       8GB       SSD        ~$60
Worker Nodes        Standard_D4s_v3  4       16GB      SSD        ~$140
ML Training         Standard_NC6     6       56GB      SSD+GPU    ~$1,900
Database            GP_Gen5_4        4       20GB      SSD        ~$350
Load Balancer       Standard         -       -         -          ~$25
```

#### Storage Services
```
Service             Type             Capacity    IOPS       Cost/Month
Database Storage    Premium SSD      1TB         5000       ~$150
Object Storage      Blob Standard    10TB        -          ~$180
Backup Storage      Cool Tier        5TB         -          ~$50
Archive Storage     Archive Tier     100TB       -          ~$200
```

### Kubernetes Requirements

#### Cluster Specifications
```
Component           Minimum         Recommended     Enterprise
Master Nodes        1               3               5
Worker Nodes        2               5               20+
Node CPU            4 cores         8 cores         16+ cores
Node Memory         8GB             16GB            32GB+
Node Storage        100GB           500GB           1TB+
```

#### Required Add-ons
```
Add-on              Purpose                     Version
CNI Plugin          Pod networking              Latest stable
DNS                 Service discovery           CoreDNS 1.8+
Metrics Server      Resource monitoring         0.6+
Ingress Controller  External access             nginx 1.5+
Cert Manager        SSL certificate mgmt        1.10+
```

#### Storage Classes
```
Storage Class       Type            Use Case                Performance
fast-ssd           NVMe SSD        Database, cache         High IOPS
standard-ssd       SATA SSD        Application data        Balanced  
standard-hdd       Spinning disk   Backup, archive         High capacity
```

---

This completes the comprehensive System Requirements specification for aNEOS. The document covers all deployment scenarios from development environments to enterprise production clusters, ensuring proper resource allocation and performance optimization across different platforms and use cases.