# Installation Troubleshooting Guide

Complete troubleshooting guide for aNEOS installation issues.

## 🔍 Quick Diagnosis

### Run Automated Diagnostics

```bash
# Check system requirements
python install.py --check

# Check current installation status
python aneos.py status

# Fix dependency issues
python install.py --fix-deps

# Run installation verification
python -c "
import sys
sys.path.append('.')
try:
    from aneos_core.analysis.pipeline import create_analysis_pipeline
    from aneos_api.app import create_app
    from aneos_api.database import get_database_status
    print('✅ All core modules available')
    print('✅ Installation appears successful')
except ImportError as e:
    print(f'❌ Installation issue: {e}')
"
```

## 🛠️ Common Installation Issues

### 1. Python Version Issues

**Problem**: Wrong Python version or multiple Python installations

**Symptoms**:
```
ERROR: Python 3.7 is not supported
ModuleNotFoundError: No module named 'dataclasses'
```

**Solutions**:

```bash
# Check Python versions
python --version
python3 --version
python3.8 --version
python3.9 --version
python3.10 --version

# Use specific Python version
python3.10 install.py

# Create virtual environment with specific version
python3.10 -m venv aneos-env
source aneos-env/bin/activate
python install.py
```

**Windows specific**:
```cmd
# Check Python versions
py -0  # List all Python versions
py -3.10 install.py  # Use specific version
```

### 2. Permission Issues

**Problem**: Insufficient permissions to install packages

**Symptoms**:
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied
```

**Solutions**:

```bash
# Option 1: Use virtual environment (recommended)
python -m venv aneos-env
source aneos-env/bin/activate  # Linux/macOS
# aneos-env\Scripts\activate   # Windows
python install.py

# Option 2: User installation
pip install --user -r requirements.txt

# Option 3: Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER ~/.local/lib/python*
```

### 3. Network and Firewall Issues

**Problem**: Cannot download packages due to network restrictions

**Symptoms**:
```
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None))
ERROR: Could not find a version that satisfies the requirement
```

**Solutions**:

```bash
# Option 1: Use different PyPI mirror
pip install -r requirements.txt -i https://pypi.org/simple/
pip install -r requirements.txt -i https://pypi.python.org/simple/

# Option 2: Increase timeout
pip install -r requirements.txt --timeout 300

# Option 3: Use proxy (if applicable)
pip install -r requirements.txt --proxy http://proxy.company.com:8080

# Option 4: Corporate firewall - use trusted hosts
pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

### 4. Dependency Conflicts

**Problem**: Conflicting package versions

**Symptoms**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages
ERROR: Cannot install package1 and package2 because they have conflicting dependencies
```

**Solutions**:

```bash
# Option 1: Clean installation in new environment
python -m venv fresh-aneos-env
source fresh-aneos-env/bin/activate
python install.py

# Option 2: Clear pip cache
pip cache purge
pip install -r requirements.txt

# Option 3: Install with no dependencies first, then fix
pip install --no-deps -r requirements.txt
python install.py --fix-deps

# Option 4: Use conda instead of pip
conda create -n aneos python=3.10
conda activate aneos
conda install --file requirements.txt
```

### 5. Compilation Errors

**Problem**: C/C++ compilation errors for packages with native extensions

**Symptoms**:
```
error: Microsoft Visual C++ 14.0 is required  # Windows
error: command 'gcc' failed  # Linux
clang: error: linker command failed  # macOS
```

**Solutions**:

**Windows**:
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

**Linux (Ubuntu/Debian)**:
```bash
# Install build tools
sudo apt update
sudo apt install build-essential python3-dev

# Install package-specific dependencies
sudo apt install libffi-dev libssl-dev  # For cryptography
sudo apt install python3-tk  # For matplotlib
```

**macOS**:
```bash
# Install Xcode command line tools
xcode-select --install

# Install homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python through homebrew
brew install python@3.10
```

### 6. Disk Space Issues

**Problem**: Insufficient disk space

**Symptoms**:
```
OSError: [Errno 28] No space left on device
```

**Solutions**:

```bash
# Check disk space
df -h  # Linux/macOS
dir   # Windows

# Clean pip cache
pip cache purge

# Clean system caches
sudo apt autoclean  # Ubuntu/Debian
brew cleanup        # macOS

# Use different temporary directory
export TMPDIR=/path/to/larger/disk/tmp
pip install -r requirements.txt
```

### 7. SSL Certificate Issues

**Problem**: SSL certificate verification failures

**Symptoms**:
```
SSL: CERTIFICATE_VERIFY_FAILED
WARNING: Retrying with --trusted-host
```

**Solutions**:

```bash
# Option 1: Update certificates
pip install --upgrade certifi

# Option 2: Use trusted hosts (temporary)
pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

# Option 3: Corporate network - get certificates
curl -k https://pypi.org/simple/  # Test connection
```

## 🔧 Platform-Specific Issues

### Windows Issues

**1. Long Path Names**
```bash
# Enable long paths in Windows
# Run as Administrator in PowerShell:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**2. PowerShell Execution Policy**
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**3. Missing Visual C++ Redistributables**
```bash
# Download and install from Microsoft:
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### macOS Issues

**1. Command Line Tools**
```bash
# Install Xcode command line tools
xcode-select --install

# If still issues, install full Xcode from App Store
```

**2. Homebrew Python vs System Python**
```bash
# Use Homebrew Python
brew install python@3.10
export PATH="/opt/homebrew/bin:$PATH"  # M1 Macs
export PATH="/usr/local/bin:$PATH"     # Intel Macs
```

### Linux Issues

**1. Missing System Libraries**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev python3-pip build-essential
sudo apt install libffi-dev libssl-dev libsqlite3-dev

# CentOS/RHEL/Fedora
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel python3-pip
sudo yum install libffi-devel openssl-devel sqlite-devel
```

**2. SELinux Issues**
```bash
# Check SELinux status
sestatus

# Temporarily disable (if needed)
sudo setenforce 0

# Or create proper policies (recommended)
```

## 🗄️ Database-Specific Issues

### SQLite Issues

**Problem**: SQLite database creation/access issues

**Solutions**:
```bash
# Check SQLite version
python -c "import sqlite3; print(sqlite3.sqlite_version)"

# Reset database
rm aneos.db
python -c "from aneos_api.database import init_database; init_database()"

# Check permissions
ls -la aneos.db
chmod 664 aneos.db  # If needed
```

### PostgreSQL Issues

**Problem**: PostgreSQL connection issues

**Solutions**:
```bash
# Install PostgreSQL client libraries
# Ubuntu/Debian
sudo apt install libpq-dev python3-dev

# CentOS/RHEL
sudo yum install postgresql-devel python3-devel

# Check connection
python -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://user:pass@localhost/aneos')
    print('✅ PostgreSQL connection OK')
    conn.close()
except Exception as e:
    print(f'❌ PostgreSQL error: {e}')
"
```

## 🧪 Testing Installation

### Automated Testing

```bash
# Run comprehensive installation test
python -c "
import sys
import traceback

tests = []

# Test 1: Core imports
try:
    from aneos_core.analysis.pipeline import create_analysis_pipeline
    from aneos_core.config.settings import get_config
    tests.append(('Core imports', True, None))
except Exception as e:
    tests.append(('Core imports', False, str(e)))

# Test 2: API imports
try:
    from aneos_api.app import create_app
    from aneos_api.database import get_database_status
    tests.append(('API imports', True, None))
except Exception as e:
    tests.append(('API imports', False, str(e)))

# Test 3: Database
try:
    from aneos_api.database import get_database_status
    status = get_database_status()
    tests.append(('Database', status.get('available', False), status.get('error')))
except Exception as e:
    tests.append(('Database', False, str(e)))

# Test 4: ML imports (optional)
try:
    import sklearn
    import numpy
    import pandas
    tests.append(('ML libraries', True, None))
except Exception as e:
    tests.append(('ML libraries', False, str(e)))

# Print results
print('\\n📊 Installation Test Results:')
print('=' * 40)
for test_name, passed, error in tests:
    status = '✅ PASS' if passed else '❌ FAIL'
    print(f'{test_name:20} {status}')
    if error and not passed:
        print(f'    Error: {error}')

passed_tests = sum(1 for _, passed, _ in tests if passed)
total_tests = len(tests)
print(f'\\n📈 Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.0f}%)')
"
```

### Manual Testing Steps

1. **Test Core Functionality**:
   ```bash
   python aneos.py analyze "2024 AB123"
   ```

2. **Test API Server**:
   ```bash
   python aneos.py api --dev &
   curl http://localhost:8000/health
   pkill -f "aneos.py api"
   ```

3. **Test Database**:
   ```bash
   python -c "
   from aneos_api.database import init_database, get_database_status
   print('Database status:', get_database_status())
   "
   ```

4. **Test Menu System**:
   ```bash
   # Should start without errors
   timeout 5 python aneos.py || echo "Menu system OK"
   ```

## 🔄 Recovery Procedures

### Complete Reset

If all else fails, perform a complete reset:

```bash
# 1. Remove all caches and temporary files
rm -rf __pycache__ .pytest_cache *.pyc
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# 2. Remove database
rm -f aneos.db

# 3. Remove virtual environment (if using)
deactivate  # If in virtual environment
rm -rf aneos-env

# 4. Clear pip cache
pip cache purge

# 5. Fresh installation
python -m venv aneos-env
source aneos-env/bin/activate
python install.py --full
```

### Partial Reset

For specific component issues:

```bash
# Reset database only
rm aneos.db
python -c "from aneos_api.database import init_database; init_database()"

# Reinstall Python packages only
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Reset configuration
rm .env
python install.py --minimal  # Will recreate config
```

## 📋 Getting Help

### Information to Collect

When seeking help, collect this information:

```bash
# System information
python install.py --check > installation_check.txt

# Python environment
python -c "
import sys
import platform
print('Python version:', sys.version)
print('Platform:', platform.platform())
print('Architecture:', platform.architecture())
print('Python path:', sys.executable)
" > python_info.txt

# Package versions
pip list > package_versions.txt

# Recent logs
tail -50 logs/aneos.log > recent_logs.txt  # If logs exist
```

### Support Channels

1. **Self-Diagnosis**: `python install.py --check`
2. **Documentation**: Check `docs/troubleshooting/`
3. **System Status**: `python aneos.py status`
4. **Installation Report**: Review `installation_report.json`
5. **Community Support**: Include system info and error messages

---

**Most installation issues can be resolved by using a clean virtual environment and ensuring system requirements are met. When in doubt, start fresh!** 🚀