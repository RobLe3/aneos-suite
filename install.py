#!/usr/bin/env python3
"""
aNEOS Installation and Dependency Management System

Comprehensive installer that handles all dependencies, system requirements,
and initial setup for the aNEOS platform across different environments.
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Try to import rich for better UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

class ANEOSInstaller:
    """Comprehensive aNEOS installation and dependency management."""
    
    def __init__(self):
        self.console = console if HAS_RICH else None
        self.project_root = Path.cwd()
        self.python_executable = sys.executable
        self.system_info = self.get_system_info()
        self.install_log = []
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'python_compiler': platform.python_compiler(),
            'python_executable': sys.executable,
            'working_directory': str(Path.cwd()),
            'user': os.getenv('USER', os.getenv('USERNAME', 'unknown')),
            'home': str(Path.home()),
            'path': os.getenv('PATH', ''),
        }
    
    def display_header(self):
        """Display installer header."""
        if self.console:
            header = Panel.fit(
                "[bold blue]aNEOS Installation System[/bold blue]\n"
                "[bold]Advanced Near Earth Object detection System[/bold]\n"
                "[dim]Dependency Management & System Setup[/dim]",
                border_style="blue"
            )
            self.console.print(header)
        else:
            print("=" * 60)
            print("aNEOS Installation System")
            print("Advanced Near Earth Object detection System")
            print("=" * 60)
    
    def display_system_info(self):
        """Display system information."""
        if self.console:
            table = Table(title="System Information", show_header=True, header_style="bold cyan")
            table.add_column("Component", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Operating System", f"{self.system_info['platform']} {self.system_info['platform_release']}")
            table.add_row("Architecture", self.system_info['architecture'])
            table.add_row("Python Version", self.system_info['python_version'])
            table.add_row("Python Implementation", self.system_info['python_implementation'])
            table.add_row("Python Executable", self.system_info['python_executable'])
            table.add_row("Working Directory", self.system_info['working_directory'])
            
            self.console.print(table)
        else:
            print("\nSystem Information:")
            print(f"OS: {self.system_info['platform']} {self.system_info['platform_release']}")
            print(f"Architecture: {self.system_info['architecture']}")
            print(f"Python: {self.system_info['python_version']}")
            print(f"Python Path: {self.system_info['python_executable']}")
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            self.log_success(f"Python version {version.major}.{version.minor}.{version.micro} meets requirements (>= 3.8)")
            return True
        else:
            self.log_error(f"Python version {version.major}.{version.minor}.{version.micro} is too old. Minimum required: 3.8")
            return False
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements."""
        checks = {}
        
        # Python version
        checks['python_version'] = self.check_python_version()
        
        # pip availability
        try:
            subprocess.run([self.python_executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
            checks['pip'] = True
            self.log_success("pip is available")
        except subprocess.CalledProcessError:
            checks['pip'] = False
            self.log_error("pip is not available")
        
        # Git availability (optional)
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            checks['git'] = True
            self.log_success("Git is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            checks['git'] = False
            self.log_warning("Git is not available (optional)")
        
        # Docker availability (optional)
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            checks['docker'] = True
            self.log_success("Docker is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            checks['docker'] = False
            self.log_warning("Docker is not available (optional)")
        
        # Docker Compose availability (optional)
        try:
            subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
            checks['docker_compose'] = True
            self.log_success("Docker Compose is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            checks['docker_compose'] = False
            self.log_warning("Docker Compose is not available (optional)")
        
        # Check available disk space
        disk_usage = shutil.disk_usage(self.project_root)
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb >= 5.0:
            checks['disk_space'] = True
            self.log_success(f"Sufficient disk space: {free_gb:.1f} GB available")
        else:
            checks['disk_space'] = False
            self.log_error(f"Insufficient disk space: {free_gb:.1f} GB available (minimum 5 GB required)")
        
        # Check memory (rough estimate)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb >= 4.0:
                checks['memory'] = True
                self.log_success(f"Sufficient memory: {memory_gb:.1f} GB available")
            else:
                checks['memory'] = False
                self.log_warning(f"Limited memory: {memory_gb:.1f} GB available (recommended: 8+ GB)")
        except ImportError:
            checks['memory'] = True  # Can't check, assume OK
            self.log_warning("Cannot check memory (psutil not available)")
        
        return checks
    
    def install_core_dependencies(self, mode: str = "core") -> bool:
        """Install core Python dependencies with different installation modes."""
        self.log_info(f"Installing {mode} dependencies...")
        
        # Select requirements file based on mode
        requirements_files = {
            "minimal": "requirements-minimal.txt",
            "core": "requirements-core.txt", 
            "full": "requirements.txt"
        }
        
        requirements_file = self.project_root / requirements_files.get(mode, "requirements-core.txt")
        
        if not requirements_file.exists():
            # Fallback to main requirements.txt if specific file doesn't exist
            requirements_file = self.project_root / "requirements.txt"
            if not requirements_file.exists():
                self.log_error("No requirements file found")
                return False
        
        # Check for externally managed environment and prepare pip command
        pip_install_cmd = [self.python_executable, "-m", "pip", "install"]
        user_install = False
        
        # Test if we're in an externally managed environment
        try:
            test_result = subprocess.run(
                pip_install_cmd + ["--dry-run", "--quiet", "pip"],
                capture_output=True, text=True
            )
            if "externally-managed-environment" in test_result.stderr.lower():
                self.log_warning("Detected externally managed Python environment")
                if self.console:
                    from rich.prompt import Confirm
                    user_install = Confirm.ask(
                        "Install packages in user directory (recommended)?", 
                        default=True
                    )
                    if user_install:
                        pip_install_cmd.append("--user")
                        self.log_info("Using --user flag for installation")
                else:
                    print("Using --user flag for externally managed environment")
                    pip_install_cmd.append("--user")
                    user_install = True
        except Exception:
            # If test fails, continue with normal installation
            pass
        
        # Attempt to upgrade pip (optional, don't fail if it doesn't work)
        try:
            if self.console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Checking pip...", total=None)
                    try:
                        subprocess.run(
                            pip_install_cmd + ["--upgrade", "pip"],
                            capture_output=True, check=True, timeout=30
                        )
                        self.log_success("pip upgraded")
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        self.log_warning("pip upgrade skipped (continuing with existing version)")
                    progress.update(task, completed=True)
            else:
                print("Checking pip...")
                try:
                    subprocess.run(pip_install_cmd + ["--upgrade", "pip"], 
                                 check=True, timeout=30, capture_output=True)
                    self.log_success("pip upgraded")
                except:
                    self.log_warning("pip upgrade skipped")
        except Exception:
            self.log_warning("pip check skipped")
        
        # Install requirements
        if not requirements_file.exists():
            self.log_error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            if self.console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Installing dependencies...", total=100)
                    
                    # Install with progress updates using prepared command
                    install_cmd = pip_install_cmd + ["-r", str(requirements_file)]
                    process = subprocess.Popen(
                        install_cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.STDOUT, 
                        text=True
                    )
                    
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            # Simulate progress based on output
                            if "Collecting" in output:
                                progress.advance(task, 5)
                            elif "Installing" in output:
                                progress.advance(task, 10)
                    
                    progress.update(task, completed=100)
                    
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, "pip install")
            else:
                print(f"Installing dependencies from {requirements_file.name}...")
                install_cmd = pip_install_cmd + ["-r", str(requirements_file)]
                subprocess.run(install_cmd, check=True)
            
            self.log_success("Core dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_error(f"Failed to install dependencies: {e}")
            return False
    
    def install_optional_dependencies(self) -> Dict[str, bool]:
        """Install optional dependencies for enhanced functionality."""
        optional_packages = {
            'torch': {
                'package': 'torch>=2.1.0',
                'description': 'PyTorch for neural network models',
                'category': 'ml'
            },
            'tensorflow': {
                'package': 'tensorflow>=2.15.0',
                'description': 'TensorFlow for ML models (alternative to PyTorch)',
                'category': 'ml'
            },
            'jupyter': {
                'package': 'jupyter notebook jupyterlab',
                'description': 'Jupyter notebooks for interactive analysis',
                'category': 'analysis'
            },
            'plotly-dash': {
                'package': 'dash plotly-dash',
                'description': 'Interactive web dashboards',
                'category': 'visualization'
            },
            'prometheus': {
                'package': 'prometheus-client',
                'description': 'Prometheus metrics integration',
                'category': 'monitoring'
            },
            'redis': {
                'package': 'redis hiredis',
                'description': 'Redis caching support',
                'category': 'performance'
            }
        }
        
        results = {}
        
        if self.console:
            self.console.print("\n[bold]Optional Dependencies[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Package", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Category", style="dim")
            
            for name, info in optional_packages.items():
                table.add_row(name, info['description'], info['category'])
            
            self.console.print(table)
            
            install_optional = Confirm.ask("Install optional dependencies?", default=True)
        else:
            print("\nOptional Dependencies Available:")
            for name, info in optional_packages.items():
                print(f"  {name}: {info['description']}")
            
            install_optional = input("Install optional dependencies? (y/n): ").lower().startswith('y')
        
        if not install_optional:
            return {name: False for name in optional_packages}
        
        for name, info in optional_packages.items():
            try:
                if self.console:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn(f"Installing {name}..."),
                        console=self.console
                    ) as progress:
                        task = progress.add_task(f"Installing {name}...", total=None)
                        subprocess.run([
                            self.python_executable, "-m", "pip", "install"
                        ] + info['package'].split(), capture_output=True, check=True)
                        progress.update(task, completed=True)
                else:
                    print(f"Installing {name}...")
                    subprocess.run([
                        self.python_executable, "-m", "pip", "install"
                    ] + info['package'].split(), check=True)
                
                results[name] = True
                self.log_success(f"Installed {name}")
                
            except subprocess.CalledProcessError:
                results[name] = False
                self.log_warning(f"Failed to install {name} (optional)")
        
        return results
    
    def setup_directories(self) -> bool:
        """Create necessary directories."""
        directories = [
            'data',
            'logs', 
            'models',
            'cache',
            'exports',
            'backups',
            'temp'
        ]
        
        self.log_info("Creating directory structure...")
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            try:
                dir_path.mkdir(exist_ok=True)
                self.log_success(f"Created directory: {dir_name}")
            except Exception as e:
                self.log_error(f"Failed to create directory {dir_name}: {e}")
                return False
        
        # Create .gitkeep files to preserve empty directories
        for dir_name in directories:
            gitkeep_path = self.project_root / dir_name / '.gitkeep'
            try:
                gitkeep_path.touch(exist_ok=True)
            except Exception:
                pass  # Non-critical
        
        return True
    
    def setup_database(self) -> bool:
        """Initialize database if needed."""
        self.log_info("Setting up database...")
        
        try:
            # Import and initialize database
            sys.path.insert(0, str(self.project_root))
            from aneos_api.database import init_database, get_database_status
            
            # Check current status
            db_status = get_database_status()
            if db_status.get('available'):
                self.log_success("Database already initialized")
                return True
            
            # Initialize database
            if self.console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("Initializing database..."),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Initializing database...", total=None)
                    success = init_database()
                    progress.update(task, completed=True)
            else:
                print("Initializing database...")
                success = init_database()
            
            if success:
                self.log_success("Database initialized successfully")
                return True
            else:
                self.log_error("Database initialization failed")
                return False
                
        except ImportError as e:
            self.log_warning(f"Cannot initialize database: {e}")
            return True  # Non-critical for basic functionality
        except Exception as e:
            self.log_error(f"Database setup failed: {e}")
            return False
    
    def create_config_files(self) -> bool:
        """Create default configuration files."""
        self.log_info("Creating configuration files...")
        
        # Create .env file
        env_file = self.project_root / '.env'
        if not env_file.exists():
            env_content = f"""# aNEOS Environment Configuration
# Database
ANEOS_DATABASE_URL=sqlite:///./aneos.db

# API Configuration  
ANEOS_ENV=development
ANEOS_LOG_LEVEL=INFO
ANEOS_HOST=0.0.0.0
ANEOS_PORT=8000

# Redis (optional)
# ANEOS_REDIS_URL=redis://localhost:6379/0

# ML Configuration
ANEOS_ML_CACHE_SIZE=1000
ANEOS_FEATURE_CACHE_TTL=1800

# Monitoring
ANEOS_METRICS_INTERVAL=60
ANEOS_ALERT_COOLDOWN=300

# Security
ANEOS_SECRET_KEY=your-secret-key-change-in-production
ANEOS_API_KEY_SALT=your-api-key-salt-change-in-production
"""
            try:
                env_file.write_text(env_content)
                self.log_success("Created .env configuration file")
            except Exception as e:
                self.log_error(f"Failed to create .env file: {e}")
                return False
        
        # Create logging configuration
        logging_config = self.project_root / 'logging.conf'
        if not logging_config.exists():
            logging_content = """[loggers]
keys=root,aneos

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_aneos]
level=INFO
handlers=consoleHandler,fileHandler
qualname=aneos
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=INFO
formatter=detailedFormatter
args=('logs/aneos.log', 'a')

[formatter_simpleFormatter]
format=%(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
"""
            try:
                logging_config.write_text(logging_content)
                self.log_success("Created logging configuration")
            except Exception as e:
                self.log_warning(f"Failed to create logging config: {e}")
        
        return True
    
    def run_initial_tests(self) -> bool:
        """Run basic tests to verify installation."""
        self.log_info("Running installation verification tests...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Python environment and basic imports
        total_tests += 1
        try:
            import json, os, sys, pathlib, datetime
            self.log_success("✓ Python environment and basic imports working")
            tests_passed += 1
        except ImportError as e:
            self.log_error(f"✗ Basic Python imports failed: {e}")
        
        # Test 2: Enhanced NEO poller availability 
        total_tests += 1
        try:
            enhanced_poller_path = self.project_root / "enhanced_neo_poller.py"
            if enhanced_poller_path.exists():
                self.log_success("✓ Enhanced NEO poller available")
                tests_passed += 1
            else:
                self.log_warning("✗ Enhanced NEO poller not found")
        except Exception as e:
            self.log_warning(f"✗ Enhanced NEO poller test failed: {e}")
        
        # Test 3: Core directory structure
        total_tests += 1
        try:
            core_dirs = ['aneos_core', 'aneos_api', 'data', 'logs', 'cache']
            missing_dirs = []
            for dir_name in core_dirs:
                if not (self.project_root / dir_name).exists():
                    missing_dirs.append(dir_name)
            
            if not missing_dirs:
                self.log_success("✓ Core directory structure complete")
                tests_passed += 1
            else:
                self.log_warning(f"✗ Missing directories: {', '.join(missing_dirs)}")
        except Exception as e:
            self.log_error(f"✗ Directory structure test failed: {e}")
        
        # Test 4: Core dependencies availability (non-blocking)
        total_tests += 1
        try:
            missing_deps = []
            # Only check external packages that need to be installed
            core_packages = ['requests']
            optional_packages = ['rich', 'numpy', 'pandas', 'astropy', 'tqdm', 'humanize']
            
            # Check core packages
            for pkg in core_packages:
                try:
                    __import__(pkg)
                except ImportError:
                    missing_deps.append(pkg)
            
            # Check optional packages (don't fail on these)
            optional_missing = []
            for pkg in optional_packages:
                try:
                    __import__(pkg)
                except ImportError:
                    optional_missing.append(pkg)
            
            if not missing_deps:
                self.log_success("✓ Core dependencies available")
                if optional_missing:
                    self.log_info(f"Optional packages missing: {', '.join(optional_missing)} (install as needed)")
                tests_passed += 1
            else:
                self.log_warning(f"✗ Missing core dependencies: {', '.join(missing_deps)}")
                
        except Exception as e:
            self.log_warning(f"✗ Dependency test failed: {e}")
        
        success_rate = (tests_passed / total_tests) * 100
        
        if self.console:
            if success_rate >= 75:
                self.console.print(f"[green]✓ Installation verification: {success_rate:.0f}% tests passed[/green]")
            else:
                self.console.print(f"[yellow]⚠ Installation verification: {success_rate:.0f}% tests passed[/yellow]")
        else:
            print(f"Installation verification: {success_rate:.0f}% tests passed ({tests_passed}/{total_tests})")
        
        return success_rate >= 75
    
    def create_installation_report(self) -> bool:
        """Create installation report."""
        report_path = self.project_root / 'installation_report.json'
        
        report = {
            'installation_date': str(datetime.now()),
            'system_info': self.system_info,
            'installation_log': self.install_log,
            'python_version': self.system_info['python_version'],
            'python_executable': self.system_info['python_executable'],
            'project_root': str(self.project_root),
            'success': len([log for log in self.install_log if log['level'] == 'SUCCESS']) > 0
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.log_success(f"Installation report created: {report_path}")
            return True
        except Exception as e:
            self.log_error(f"Failed to create installation report: {e}")
            return False
    
    def log_info(self, message: str):
        """Log info message."""
        self.install_log.append({'level': 'INFO', 'message': message, 'timestamp': str(datetime.now())})
        if self.console:
            self.console.print(f"[blue]ℹ️  {message}[/blue]")
        else:
            print(f"ℹ️  {message}")
    
    def log_success(self, message: str):
        """Log success message."""
        self.install_log.append({'level': 'SUCCESS', 'message': message, 'timestamp': str(datetime.now())})
        if self.console:
            self.console.print(f"[green]✅ {message}[/green]")
        else:
            print(f"✅ {message}")
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.install_log.append({'level': 'WARNING', 'message': message, 'timestamp': str(datetime.now())})
        if self.console:
            self.console.print(f"[yellow]⚠️  {message}[/yellow]")
        else:
            print(f"⚠️  {message}")
    
    def log_error(self, message: str):
        """Log error message."""
        self.install_log.append({'level': 'ERROR', 'message': message, 'timestamp': str(datetime.now())})
        if self.console:
            self.console.print(f"[red]❌ {message}[/red]")
        else:
            print(f"❌ {message}")
    
    def full_installation(self) -> bool:
        """Run complete installation process."""
        self.display_header()
        self.display_system_info()
        
        # Check system requirements
        self.log_info("Checking system requirements...")
        requirements = self.check_system_requirements()
        
        # Critical requirements check
        critical_failed = []
        if not requirements.get('python_version'):
            critical_failed.append('Python version')
        if not requirements.get('pip'):
            critical_failed.append('pip')
        if not requirements.get('disk_space'):
            critical_failed.append('Disk space')
        
        if critical_failed:
            self.log_error(f"Critical requirements failed: {', '.join(critical_failed)}")
            return False
        
        # Install dependencies (full mode)
        if not self.install_core_dependencies("full"):
            self.log_error("Core dependency installation failed")
            return False
        
        # Install optional dependencies
        optional_results = self.install_optional_dependencies()
        
        # Setup directories
        if not self.setup_directories():
            self.log_error("Directory setup failed")
            return False
        
        # Setup database
        self.setup_database()
        
        # Create config files
        if not self.create_config_files():
            self.log_error("Configuration file creation failed")
            return False
        
        # Run verification tests
        if not self.run_initial_tests():
            self.log_warning("Some verification tests failed")
        
        # Create installation report
        self.create_installation_report()
        
        # Final success message
        if self.console:
            success_panel = Panel.fit(
                "[bold green]✅ aNEOS Installation Complete![/bold green]\n\n"
                "Next steps:\n"
                "1. Run: [bold cyan]python aneos.py status[/bold cyan] to check system\n"
                "2. Run: [bold cyan]python aneos.py[/bold cyan] for interactive menu\n"
                "3. Run: [bold cyan]python aneos.py api --dev[/bold cyan] for development server\n\n"
                "Documentation: [bold]docs/[/bold] directory\n"
                "Issues: Check [bold]installation_report.json[/bold]",
                border_style="green"
            )
            self.console.print(success_panel)
        else:
            print("\n" + "="*60)
            print("✅ aNEOS Installation Complete!")
            print("="*60)
            print("Next steps:")
            print("1. Run: python aneos.py status")
            print("2. Run: python aneos.py")
            print("3. Run: python aneos.py api --dev")
            print("\nDocumentation: docs/ directory")
            print("Issues: Check installation_report.json")
        
        return True
    
    def minimal_installation(self) -> bool:
        """Run minimal installation process - just essential dependencies."""
        self.log_info("Starting minimal installation...")
        self.display_header()
        
        # Basic system checks
        if not self.check_python_version():
            return False
        
        # Install minimal dependencies
        if not self.install_core_dependencies("minimal"):
            self.log_error("Minimal dependency installation failed")
            return False
        
        # Setup basic directories
        if not self.setup_directories():
            self.log_error("Directory setup failed")
            return False
        
        self.log_success("✅ Minimal installation completed successfully!")
        self.log_info("Core NEO analysis functionality is now available")
        return True
    
    def core_installation(self) -> bool:
        """Run core installation process - essential plus recommended dependencies."""
        self.log_info("Starting core installation...")
        self.display_header()
        
        # Check system requirements
        requirements = self.check_system_requirements()
        
        # Install core dependencies
        if not self.install_core_dependencies("core"):
            self.log_error("Core dependency installation failed")
            return False
        
        # Setup directories
        if not self.setup_directories():
            self.log_error("Directory setup failed")
            return False
        
        # Basic config
        self.create_config_files()
        
        # Quick verification
        self.run_initial_tests()
        
        self.log_success("✅ Core installation completed successfully!")
        self.log_info("Enhanced NEO analysis with academic rigor is now available")
        return True
    
    def dependency_check(self) -> bool:
        """Check and fix dependency issues."""
        self.log_info("Checking current dependencies...")
        
        # Run the improved verification tests
        return self.run_initial_tests()

def main():
    """Main installer entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="aNEOS Installation and Dependency Management")
    parser.add_argument('--check', action='store_true', help='Check dependencies only')
    parser.add_argument('--minimal', action='store_true', help='Minimal installation (essential only)')
    parser.add_argument('--core', action='store_true', help='Core installation (recommended)')
    parser.add_argument('--full', action='store_true', help='Full installation with optional packages')
    parser.add_argument('--fix-deps', action='store_true', help='Fix dependency issues')
    parser.add_argument('--no-interaction', action='store_true', help='Non-interactive mode')
    
    args = parser.parse_args()
    
    installer = ANEOSInstaller()
    
    # Check if we're in the right directory
    if not Path('aneos_core').exists():
        print("❌ Error: Please run this script from the aNEOS project root directory")
        print("Current directory should contain 'aneos_core' folder")
        sys.exit(1)
    
    try:
        if args.check:
            installer.display_header()
            installer.display_system_info()
            requirements = installer.check_system_requirements()
            
            if all(requirements.values()):
                print("✅ All system requirements met")
                sys.exit(0)
            else:
                print("❌ Some system requirements not met")
                sys.exit(1)
                
        elif args.fix_deps:
            success = installer.dependency_check()
            sys.exit(0 if success else 1)
            
        elif args.minimal:
            success = installer.minimal_installation()
            sys.exit(0 if success else 1)
            
        elif args.core:
            success = installer.core_installation()
            sys.exit(0 if success else 1)
            
        elif args.full:
            success = installer.full_installation()
            sys.exit(0 if success else 1)
        else:
            # Default to core installation (balanced approach)
            success = installer.core_installation()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n🛑 Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Installation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()