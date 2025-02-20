#!/usr/bin/env python3
import os
import sys
import subprocess

# Ensure Python version compatibility
REQUIRED_PYTHON = (3, 12, 1)
if sys.version_info < REQUIRED_PYTHON:
    print(f"\n❌ Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.{REQUIRED_PYTHON[2]} or later is required.")
    print(f"   You are using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}.")
    sys.exit(1)

# Bootstrap pkg_resources if not available
try:
    import pkg_resources
except ImportError:
    print("pkg_resources not found. Installing setuptools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    import pkg_resources

# Required directories from both scripts
def get_required_directories():
    return {
        "dataneos", "dataneos/daily_outputs", "dataneos/data",
        "dataneos/reporting", "dataneos/orbital_elements",
        "dataneos/orbital_elements_cache"
    }

# Ensure all required directories exist
def check_directories():
    dirs = get_required_directories()
    created_dirs = []
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
            created_dirs.append(d)
    
    print("\n==== Directory Check Report ====")
    if created_dirs:
        print("✅ The following directories were created:")
        for d in created_dirs:
            print(f"  - {d}")
    else:
        print("✅ All required directories already exist.")
    print("================================\n")

# Ensure dependencies from requirements.txt are installed
def check_dependencies(install_missing=False):
    try:
        with open("requirements.txt") as f:
            required = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("\n❌ requirements.txt not found! Script execution aborted.")
        sys.exit(1)
    
    missing_packages = []
    print("\n==== Dependency Check Report ====")
    for req in required:
        package = req.split("==")[0]
        try:
            pkg_resources.get_distribution(package)
            print(f"✅ Dependency '{package}' is installed.")
        except pkg_resources.DistributionNotFound:
            print(f"❌ Dependency '{package}' is MISSING.")
            missing_packages.append(req)
    
    if missing_packages:
        if install_missing:
            print("\n🔄 Installing missing dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All missing dependencies installed successfully.")
        else:
            print("\n❌ Missing dependencies detected. Script execution aborted.")
            sys.exit(1)
    print("================================\n")

if __name__ == "__main__":
    check_directories()
    check_dependencies(install_missing=True)  # Set to False if manual install is preferred
    
    print("\n🎉 Setup complete. You can now manually start the analysis scripts.")

