#!/usr/bin/env python3
"""
aNEOS Main Menu System

Comprehensive menu system for all aNEOS operations including scientific analysis,
ML training, API services, monitoring, and system management.
"""

import os
import sys
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import core aNEOS components (essential for menu functionality)
HAS_ANEOS_CORE = False
HAS_ANALYSIS_PIPELINE = False
HAS_DATABASE = False
HAS_API = False

# Import core analysis components
try:
    from aneos_core.analysis.pipeline import create_analysis_pipeline
    HAS_ANALYSIS_PIPELINE = True
except ImportError:
    pass

# Import database components  
try:
    import sys
    import os
    sys.path.insert(0, str(PROJECT_ROOT / "aneos_api"))
    from database import init_database, get_database_status
    HAS_DATABASE = True
except ImportError:
    try:
        # Fallback: try direct import
        import aneos_api.database as db_module
        init_database = db_module.init_database
        get_database_status = db_module.get_database_status
        HAS_DATABASE = True
    except ImportError:
        HAS_DATABASE = False

# Import API components
try:
    from app import create_app, get_aneos_app
    HAS_API = True
except ImportError:
    try:
        # Fallback: try direct import
        import aneos_api.app as app_module
        create_app = app_module.create_app
        get_aneos_app = app_module.get_aneos_app
        HAS_API = True
    except ImportError:
        HAS_API = False

# Set overall core availability
HAS_ANEOS_CORE = HAS_ANALYSIS_PIPELINE or HAS_DATABASE or HAS_API

# Advanced components (ML, monitoring) are imported on-demand only when needed
# This prevents NumPy/PyTorch conflicts for core menu functionality

try:
    import rich
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import uvicorn
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False

# Setup console
console = Console() if HAS_RICH else None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ANEOSMenu:
    """Main aNEOS menu system with all operational modes."""
    
    def __init__(self):
        self.console = console if HAS_RICH else None
        self.running = True
        
    def display_header(self):
        """Display the main header."""
        if not self.console:
            print("=" * 80)
            print("aNEOS - Advanced Near Earth Object detection System")
            print("=" * 80)
            return
            
        header = Panel.fit(
            "[bold blue]aNEOS[/bold blue]\n"
            "[bold]Advanced Near Earth Object detection System[/bold]\n"
            "[dim]Production-Ready Scientific Analysis Platform[/dim]",
            border_style="blue"
        )
        self.console.print(header)
        
    def display_system_status(self):
        """Display current system status."""
        if not self.console:
            print("\n--- System Status ---")
            print(f"Core Components: {'Available' if HAS_ANEOS_CORE else 'Limited'}")
            print(f"Database: {'Checking...' if HAS_ANEOS_CORE else 'Not Available'}")
            return
            
        # Create status table
        table = Table(title="System Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # Core components
        table.add_row(
            "Core Analysis",
            "✅ Available" if HAS_ANEOS_CORE else "❌ Limited",
            "Scientific analysis pipeline ready" if HAS_ANEOS_CORE else "Install dependencies"
        )
        
        # Database status
        if HAS_DATABASE:
            try:
                db_status = get_database_status()
                db_available = db_status.get('available', False)
                table.add_row(
                    "Database",
                    "✅ Connected" if db_available else "⚠️ Offline",
                    f"Engine: {db_status.get('engine', 'Unknown')}" if db_available else db_status.get('error', 'Not connected')
                )
            except Exception as e:
                table.add_row("Database", "❌ Error", str(e))
        else:
            table.add_row("Database", "❌ Not Available", "Database components not loaded")
            
        # API status
        table.add_row(
            "API Services",
            "✅ Ready" if HAS_API else "⚠️ Limited",
            "REST API and dashboard available" if HAS_API else "API components not loaded"
        )
        
        self.console.print(table)
        
    def display_main_menu(self):
        """Display the main menu options."""
        if not self.console:
            print("\n--- Main Menu ---")
            print("1. Scientific Analysis")
            print("2. Basic API Services")
            print("3. System Management")
            print("4. Health & Diagnostics")
            print("5. Help & Documentation")
            print("")
            print("9. Advanced Features")
            print("")
            print("0. Exit")
            return
            
        menu_table = Table(show_header=False, box=None, padding=(0, 2))
        menu_table.add_column("Option", style="bold cyan")
        menu_table.add_column("Description", style="white")
        
        menu_items = [
            ("1", "🔬 Scientific Analysis", "NEO analysis, batch processing, data exploration"),
            ("2", "🌐 Basic API Services", "Development API server, basic dashboard"),
            ("3", "⚙️  System Management", "Installation, database, configuration, maintenance"),
            ("4", "🔍 Health & Diagnostics", "System health checks, basic monitoring"),
            ("5", "📚 Help & Documentation", "User guides, troubleshooting, academic methodology"),
            ("", "", ""),
            ("9", "🚀 Advanced Features", "ML, Docker, streaming, production deployment"),
            ("", "", ""),
            ("0", "🚪 Exit", "Close aNEOS menu system")
        ]
        
        for option, title, desc in menu_items:
            if option:
                menu_table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
            else:
                menu_table.add_row("", "", "")
                
        panel = Panel(menu_table, title="[bold]Main Menu[/bold]", border_style="green")
        self.console.print(panel)
        
    def scientific_analysis_menu(self):
        """Scientific analysis submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🎯 Single NEO Analysis", "Analyze individual NEO by designation"),
                    ("2", "📦 Batch Analysis", "Process multiple NEOs from file or list"),
                    ("3", "🌍 NEO API Polling", "Poll NEO APIs by time period (1m-200y)"),
                    ("4", "🔍 Interactive Analysis", "Step-by-step guided analysis"),
                    ("5", "📊 Analysis Results Viewer", "View and export previous results"),
                    ("6", "🔧 Analysis Configuration", "Configure indicators and parameters"),
                    ("7", "📈 Statistical Reports", "Generate statistical analysis reports"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🔬 Scientific Analysis[/bold]", border_style="blue")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            else:
                print("\n--- Scientific Analysis ---")
                print("1. Single NEO Analysis")
                print("2. Batch Analysis")
                print("3. NEO API Polling")
                print("4. Interactive Analysis")
                print("5. Analysis Results Viewer")
                print("6. Analysis Configuration")
                print("7. Statistical Reports")
                print("0. Back to Main Menu")
                choice = input("Select option (0-7): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.single_neo_analysis()
            elif choice == "2":
                self.batch_analysis()
            elif choice == "3":
                self.neo_api_polling()
            elif choice == "4":
                self.interactive_analysis()
            elif choice == "5":
                self.view_analysis_results()
            elif choice == "6":
                self.configure_analysis()
            elif choice == "7":
                self.generate_reports()
                
    def machine_learning_menu(self):
        """Machine learning submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🏋️  Model Training", "Train new ML models with data"),
                    ("2", "🎯 Real-time Predictions", "Make ML predictions for NEOs"),
                    ("3", "📊 Model Management", "View, activate, and manage models"),
                    ("4", "🔍 Feature Analysis", "Analyze feature importance and quality"),
                    ("5", "📈 Model Performance", "Evaluate model performance and metrics"),
                    ("6", "🔧 Training Configuration", "Configure training parameters"),
                    ("7", "💾 Model Export/Import", "Export and import trained models"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🤖 Machine Learning[/bold]", border_style="purple")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            else:
                print("\n--- Machine Learning ---")
                print("1. Model Training")
                print("2. Real-time Predictions")
                print("3. Model Management")
                print("4. Feature Analysis")
                print("5. Model Performance")
                print("6. Training Configuration")
                print("7. Model Export/Import")
                print("0. Back to Main Menu")
                choice = input("Select option (0-7): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.ml_training()
            elif choice == "2":
                self.ml_predictions()
            elif choice == "3":
                self.model_management()
            elif choice == "4":
                self.feature_analysis()
            elif choice == "5":
                self.model_performance()
            elif choice == "6":
                self.training_configuration()
            elif choice == "7":
                self.model_export_import()
                
    def basic_api_services_menu(self):
        """Basic API services submenu - core functionality only."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🚀 Start API Server", "Launch basic REST API server for development"),
                    ("2", "🔧 Development Mode", "Start in development mode with auto-reload"),
                    ("3", "📚 View API Documentation", "Open interactive API documentation"),
                    ("4", "🔍 API Health Check", "Test basic API functionality"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🌐 Basic API Services[/bold]", border_style="green")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
            else:
                print("\n--- Basic API Services ---")
                print("1. Start API Server")
                print("2. Development Mode")
                print("3. View API Documentation")
                print("4. API Health Check")
                print("0. Back to Main Menu")
                choice = input("Select option (0-4): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.start_api_server()
            elif choice == "2":
                self.development_mode()
            elif choice == "3":
                self.view_api_docs()
            elif choice == "4":
                self.api_health_check()

    def api_services_menu(self):
        """API services submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🚀 Start API Server", "Launch REST API server"),
                    ("2", "🌐 Start Web Dashboard", "Launch web-based dashboard"),
                    ("3", "📡 Start Streaming Services", "Launch WebSocket/SSE streaming"),
                    ("4", "🔧 Development Mode", "Start in development mode with auto-reload"),
                    ("5", "📊 API Performance Test", "Test API performance and load"),
                    ("6", "🔑 Manage API Keys", "Create and manage API authentication"),
                    ("7", "📚 View API Documentation", "Open interactive API documentation"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🌐 API Services[/bold]", border_style="green")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            else:
                print("\n--- API Services ---")
                print("1. Start API Server")
                print("2. Start Web Dashboard")
                print("3. Start Streaming Services")
                print("4. Development Mode")
                print("5. API Performance Test")
                print("6. Manage API Keys")
                print("7. View API Documentation")
                print("0. Back to Main Menu")
                choice = input("Select option (0-7): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.start_api_server()
            elif choice == "2":
                self.start_web_dashboard()
            elif choice == "3":
                self.start_streaming_services()
            elif choice == "4":
                self.development_mode()
            elif choice == "5":
                self.api_performance_test()
            elif choice == "6":
                self.manage_api_keys()
            elif choice == "7":
                self.view_api_docs()
                
    def health_diagnostics_menu(self):
        """Health and diagnostics submenu - basic monitoring only."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🔍 System Health Check", "Comprehensive system health assessment"),
                    ("2", "📊 Basic System Status", "View current system status and components"),
                    ("3", "🧪 Run System Tests", "Execute basic validation tests"),
                    ("4", "📋 System Diagnostics", "Basic system diagnostics and info"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🔍 Health & Diagnostics[/bold]", border_style="yellow")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
            else:
                print("\n--- Health & Diagnostics ---")
                print("1. System Health Check")
                print("2. Basic System Status")
                print("3. Run System Tests")
                print("4. System Diagnostics")
                print("0. Back to Main Menu")
                choice = input("Select option (0-4): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.health_check()
            elif choice == "2":
                self.basic_system_status()
            elif choice == "3":
                self.run_basic_tests()
            elif choice == "4":
                self.basic_system_diagnostics()

    def monitoring_menu(self):
        """Monitoring and diagnostics submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "📊 Live System Monitor", "Real-time system monitoring dashboard"),
                    ("2", "🚨 Alert Management", "View and manage system alerts"),
                    ("3", "📈 Performance Metrics", "System performance analysis"),
                    ("4", "🔍 Health Check", "Comprehensive system health check"),
                    ("5", "📋 System Diagnostics", "Detailed system diagnostics"),
                    ("6", "📊 Metrics Export", "Export metrics to various formats"),
                    ("7", "🔧 Configure Monitoring", "Configure monitoring parameters"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]📊 Monitoring & Diagnostics[/bold]", border_style="yellow")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            else:
                print("\n--- Monitoring & Diagnostics ---")
                print("1. Live System Monitor")
                print("2. Alert Management")
                print("3. Performance Metrics")
                print("4. Health Check")
                print("5. System Diagnostics")
                print("6. Metrics Export")
                print("7. Configure Monitoring")
                print("0. Back to Main Menu")
                choice = input("Select option (0-7): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.live_system_monitor()
            elif choice == "2":
                self.alert_management()
            elif choice == "3":
                self.performance_metrics()
            elif choice == "4":
                self.health_check()
            elif choice == "5":
                self.system_diagnostics()
            elif choice == "6":
                self.metrics_export()
            elif choice == "7":
                self.configure_monitoring()
                
    def system_management_menu(self):
        """System management submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "📦 Installation & Dependencies", "Install/fix dependencies and setup system"),
                    ("2", "🗄️  Database Management", "Initialize, backup, restore database"),
                    ("3", "🧹 System Cleanup", "Clean caches, logs, temporary files"),
                    ("4", "⚙️  Configuration Management", "View and modify system configuration"),
                    ("5", "👥 User Management", "Manage user accounts and permissions"),
                    ("6", "🔧 System Maintenance", "Run system maintenance tasks"),
                    ("7", "📦 Dependency Check", "Quick dependency verification"),
                    ("8", "🔄 System Reset", "Reset system to default state"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]⚙️ System Management[/bold]", border_style="red")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
            else:
                print("\n--- System Management ---")
                print("1. Installation & Dependencies")
                print("2. Database Management")
                print("3. System Cleanup")
                print("4. Configuration Management")
                print("5. User Management")
                print("6. System Maintenance")
                print("7. Dependency Check")
                print("8. System Reset")
                print("0. Back to Main Menu")
                choice = input("Select option (0-8): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.installation_management()
            elif choice == "2":
                self.database_management()
            elif choice == "3":
                self.system_cleanup()
            elif choice == "4":
                self.configuration_management()
            elif choice == "5":
                self.user_management()
            elif choice == "6":
                self.system_maintenance()
            elif choice == "7":
                self.dependency_check()
            elif choice == "8":
                self.system_reset()
                
    def development_tools_menu(self):
        """Development tools submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🧪 Run Tests", "Execute test suites and validation"),
                    ("2", "🐛 Debug Mode", "Start system in debug mode"),
                    ("3", "📊 Code Analysis", "Run code quality and analysis tools"),
                    ("4", "⚡ Performance Profiling", "Profile system performance"),
                    ("5", "🔍 Memory Analysis", "Analyze memory usage and leaks"),
                    ("6", "📝 Generate Documentation", "Generate API and code documentation"),
                    ("7", "🔧 Development Server", "Start development server with hot reload"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🛠️ Development Tools[/bold]", border_style="cyan")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            else:
                print("\n--- Development Tools ---")
                print("1. Run Tests")
                print("2. Debug Mode")
                print("3. Code Analysis")
                print("4. Performance Profiling")
                print("5. Memory Analysis")
                print("6. Generate Documentation")
                print("7. Development Server")
                print("0. Back to Main Menu")
                choice = input("Select option (0-7): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.run_tests()
            elif choice == "2":
                self.debug_mode()
            elif choice == "3":
                self.code_analysis()
            elif choice == "4":
                self.performance_profiling()
            elif choice == "5":
                self.memory_analysis()
            elif choice == "6":
                self.generate_documentation()
            elif choice == "7":
                self.development_server()
                
    def docker_deployment_menu(self):
        """Docker and deployment submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🐳 Build Docker Images", "Build production Docker images"),
                    ("2", "🚀 Docker Compose Up", "Start full stack with Docker Compose"),
                    ("3", "☸️  Kubernetes Deploy", "Deploy to Kubernetes cluster"),
                    ("4", "📊 Container Status", "Check container and pod status"),
                    ("5", "📋 View Logs", "View container and service logs"),
                    ("6", "🔧 Scale Services", "Scale up/down services"),
                    ("7", "🛑 Stop Services", "Stop running containers/services"),
                    ("8", "🧹 Cleanup Containers", "Remove stopped containers and images"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🐳 Docker & Deployment[/bold]", border_style="blue")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
            else:
                print("\n--- Docker & Deployment ---")
                print("1. Build Docker Images")
                print("2. Docker Compose Up")
                print("3. Kubernetes Deploy")
                print("4. Container Status")
                print("5. View Logs")
                print("6. Scale Services")
                print("7. Stop Services")
                print("8. Cleanup Containers")
                print("0. Back to Main Menu")
                choice = input("Select option (0-8): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.build_docker_images()
            elif choice == "2":
                self.docker_compose_up()
            elif choice == "3":
                self.kubernetes_deploy()
            elif choice == "4":
                self.container_status()
            elif choice == "5":
                self.view_logs()
            elif choice == "6":
                self.scale_services()
            elif choice == "7":
                self.stop_services()
            elif choice == "8":
                self.cleanup_containers()
                
    def help_documentation_menu(self):
        """Help and documentation submenu."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "📚 User Guide", "View comprehensive user guide"),
                    ("2", "🔬 Scientific Documentation", "NEO analysis methodology and indicators"),
                    ("3", "🤖 ML Documentation", "Machine learning models and features"),
                    ("4", "🌐 API Documentation", "REST API reference and examples"),
                    ("5", "🐳 Deployment Guide", "Docker and Kubernetes deployment"),
                    ("6", "🛠️  Troubleshooting", "Common issues and solutions"),
                    ("7", "📊 System Requirements", "Hardware and software requirements"),
                    ("8", "🔧 Configuration Reference", "Configuration options and parameters"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]📚 Help & Documentation[/bold]", border_style="magenta")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
            else:
                print("\n--- Help & Documentation ---")
                print("1. User Guide")
                print("2. Scientific Documentation")
                print("3. ML Documentation")
                print("4. API Documentation")
                print("5. Deployment Guide")
                print("6. Troubleshooting")
                print("7. System Requirements")
                print("8. Configuration Reference")
                print("0. Back to Main Menu")
                choice = input("Select option (0-8): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.show_user_guide()
            elif choice == "2":
                self.show_scientific_docs()
            elif choice == "3":
                self.show_ml_docs()
            elif choice == "4":
                self.show_api_docs()
            elif choice == "5":
                self.show_deployment_guide()
            elif choice == "6":
                self.show_troubleshooting()
            elif choice == "7":
                self.show_system_requirements()
            elif choice == "8":
                self.show_config_reference()
                
    # Implementation methods for each menu option
    def single_neo_analysis(self):
        """Perform single NEO analysis."""
        if not HAS_ANALYSIS_PIPELINE:
            self.show_error("Analysis pipeline not available. Please install core dependencies.")
            return
            
        designation = self.get_input("Enter NEO designation (e.g., '2024 AB123'): ")
        if not designation:
            return
            
        try:
            if self.console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task("Analyzing NEO...", total=None)
                    
                    # Create analysis pipeline
                    pipeline = create_analysis_pipeline()
                    
                    # Perform analysis
                    result = asyncio.run(pipeline.analyze_neo_async(designation))
                    
                    progress.update(task, completed=True)
                    
                if result:
                    self.display_analysis_result(result)
                else:
                    self.show_error(f"Analysis failed for {designation}")
            else:
                print(f"Analyzing {designation}...")
                pipeline = create_analysis_pipeline()
                result = asyncio.run(pipeline.analyze_neo_async(designation))
                
                if result:
                    print(f"Analysis complete for {designation}")
                    print(f"Overall Score: {result.anomaly_score.overall_score:.3f}")
                    print(f"Classification: {result.anomaly_score.classification}")
                else:
                    print(f"Analysis failed for {designation}")
                    
        except Exception as e:
            self.show_error(f"Error during analysis: {e}")
            
        self.wait_for_input()
        
    def batch_analysis(self):
        """Perform batch NEO analysis."""
        if not HAS_ANALYSIS_PIPELINE:
            self.show_error("Analysis pipeline not available. Please install core dependencies.")
            return
            
        file_path = self.get_input("Enter file path with NEO designations (one per line): ")
        if not file_path or not Path(file_path).exists():
            self.show_error("File not found.")
            return
            
        try:
            with open(file_path, 'r') as f:
                designations = [line.strip() for line in f if line.strip()]
                
            if not designations:
                self.show_error("No designations found in file.")
                return
                
            if self.console:
                self.console.print(f"Found {len(designations)} NEOs to analyze")
                
                with Progress(console=self.console) as progress:
                    task = progress.add_task("Batch analysis...", total=len(designations))
                    
                    pipeline = create_analysis_pipeline()
                    results = []
                    
                    for designation in designations:
                        try:
                            result = asyncio.run(pipeline.analyze_neo_async(designation))
                            if result:
                                results.append(result)
                            progress.advance(task)
                        except Exception as e:
                            self.console.print(f"Error analyzing {designation}: {e}")
                            progress.advance(task)
                            
                self.console.print(f"Batch analysis complete: {len(results)}/{len(designations)} successful")
            else:
                print(f"Analyzing {len(designations)} NEOs...")
                pipeline = create_analysis_pipeline()
                results = []
                
                for i, designation in enumerate(designations, 1):
                    print(f"Processing {i}/{len(designations)}: {designation}")
                    try:
                        result = asyncio.run(pipeline.analyze_neo_async(designation))
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error: {e}")
                        
                print(f"Batch analysis complete: {len(results)} successful")
                
        except Exception as e:
            self.show_error(f"Error during batch analysis: {e}")
            
        self.wait_for_input()
    
    def neo_api_polling(self):
        """Launch NEO API polling system."""
        try:
            if self.console:
                self.console.print("\n🌍 [bold blue]NEO API Polling System[/bold blue]")
                self.console.print("Poll multiple NEO APIs across different time periods")
                self.console.print("Systematically search for artificial NEO signatures\n")
            else:
                print("\n🌍 NEO API Polling System")
                print("Poll multiple NEO APIs across different time periods")
                print("Systematically search for artificial NEO signatures\n")
            
            # Launch the NEO poller
            import subprocess
            import sys
            
            if self.console:
                from rich.prompt import Confirm
                if Confirm.ask("Launch interactive NEO poller?"):
                    subprocess.run([sys.executable, "neo_poller.py"])
            else:
                choice = input("Launch interactive NEO poller? (y/n) [y]: ").lower()
                if not choice or choice[0] == 'y':
                    subprocess.run([sys.executable, "neo_poller.py"])
                    
        except FileNotFoundError:
            self.show_error("neo_poller.py not found. Please ensure it's in the same directory.")
        except Exception as e:
            self.show_error(f"Error launching NEO poller: {e}")
            
        self.wait_for_input()
        
    def start_api_server(self):
        """Start the API server."""
        if not HAS_API:
            self.show_error("API components not available. Please install API dependencies.")
            return
            
        host = self.get_input("Host (default: 0.0.0.0): ") or "0.0.0.0"
        port = self.get_input("Port (default: 8000): ") or "8000"
        workers = self.get_input("Workers (default: 1): ") or "1"
        
        try:
            port = int(port)
            workers = int(workers)
        except ValueError:
            self.show_error("Invalid port or workers value.")
            return
            
        if self.console:
            self.console.print(f"🚀 Starting aNEOS API server on {host}:{port}")
            self.console.print(f"📖 Documentation: http://{host}:{port}/docs")
            self.console.print(f"📊 Dashboard: http://{host}:{port}/dashboard")
            self.console.print(f"Press Ctrl+C to stop")
            
        try:
            # Use the startup script
            subprocess.run([
                sys.executable, "start_api.py",
                "--host", host,
                "--port", str(port),
                "--workers", str(workers)
            ])
        except KeyboardInterrupt:
            if self.console:
                self.console.print("🛑 Server stopped")
            else:
                print("Server stopped")
        except Exception as e:
            self.show_error(f"Error starting server: {e}")
            
    def development_mode(self):
        """Start in development mode."""
        if not HAS_API:
            self.show_error("API components not available. Please install API dependencies.")
            return
            
        if self.console:
            self.console.print("🔧 Starting development server with auto-reload...")
            self.console.print("📖 Documentation: http://localhost:8000/docs")
            self.console.print("Press Ctrl+C to stop")
            
        try:
            subprocess.run([sys.executable, "start_api.py", "--dev"])
        except KeyboardInterrupt:
            if self.console:
                self.console.print("🛑 Development server stopped")
            else:
                print("Development server stopped")
        except Exception as e:
            self.show_error(f"Error starting development server: {e}")
            
    def live_system_monitor(self):
        """Start live system monitoring."""
        if not HAS_ANEOS_CORE:
            self.show_error("aNEOS core components not available. Please install dependencies.")
            return
            
        try:
            # Lazy import of monitoring components (Advanced feature)
            try:
                from aneos_core.monitoring.dashboard import MonitoringDashboard
                from aneos_core.monitoring.metrics import MetricsCollector
                from aneos_core.monitoring.alerts import AlertManager
            except ImportError:
                self.show_error("Advanced monitoring components not available. This feature requires additional dependencies.")
                self.wait_for_input()
                return
                
            if self.console:
                self.console.print("🖥️  Starting live system monitor...")
                self.console.print("Press Ctrl+C to exit")
                
            # Create monitoring components
            metrics_collector = MetricsCollector(collection_interval=5)
            alert_manager = AlertManager()
            dashboard = MonitoringDashboard(metrics_collector, alert_manager)
            
            # Start monitoring
            metrics_collector.start_collection()
            dashboard.run_interactive()
            
        except KeyboardInterrupt:
            if self.console:
                self.console.print("🛑 Monitoring stopped")
            else:
                print("Monitoring stopped")
        except Exception as e:
            self.show_error(f"Error starting monitor: {e}")
        finally:
            try:
                metrics_collector.stop_collection()
            except:
                pass
                
    def health_check(self):
        """Perform comprehensive health check."""
        if self.console:
            self.console.print("🔍 Performing system health check...")
            
            # Create health check table
            table = Table(title="System Health Check", show_header=True, header_style="bold green")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Details", style="dim")
            
            # Check core components
            table.add_row(
                "Core Components",
                "✅ Available" if HAS_ANEOS_CORE else "❌ Missing",
                "All core modules loaded" if HAS_ANEOS_CORE else "Install dependencies"
            )
            
            # Check database
            if HAS_DATABASE:  
                try:
                    db_status = get_database_status()
                    table.add_row(
                        "Database",
                        "✅ Connected" if db_status.get('available') else "❌ Offline",
                        db_status.get('engine', 'Unknown') if db_status.get('available') else db_status.get('error')
                    )
                except Exception as e:
                    table.add_row("Database", "❌ Error", str(e))
            else:
                table.add_row("Database", "❌ Not Available", "Database components not loaded")
                
            # Check file system
            required_dirs = ['data', 'logs', 'models', 'cache']
            missing_dirs = []
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    missing_dirs.append(dir_name)
                    
            table.add_row(
                "File System",
                "✅ OK" if not missing_dirs else "⚠️ Issues",
                "All directories exist" if not missing_dirs else f"Missing: {', '.join(missing_dirs)}"
            )
            
            # Check dependencies
            missing_deps = []
            try:
                import uvicorn, fastapi, sqlalchemy
            except ImportError as e:
                missing_deps.append("API dependencies")
                
            try:
                import torch, sklearn
            except ImportError:
                missing_deps.append("ML dependencies")
                
            table.add_row(
                "Dependencies",
                "✅ Complete" if not missing_deps else "⚠️ Missing",
                "All dependencies installed" if not missing_deps else f"Missing: {', '.join(missing_deps)}"
            )
            
            self.console.print(table)
        else:
            print("Performing health check...")
            print(f"Core Components: {'✅ Available' if HAS_ANEOS_CORE else '❌ Missing'}")
            
            if HAS_DATABASE:
                try:
                    db_status = get_database_status()
                    print(f"Database: {'✅ Connected' if db_status.get('available') else '❌ Offline'}")
                except Exception as e:
                    print(f"Database: ❌ Error - {e}")
            else:
                print("Database: ❌ Not Available")
                
        self.wait_for_input()
        
    def docker_compose_up(self):
        """Start services with Docker Compose."""
        if not Path("docker-compose.yml").exists():
            self.show_error("docker-compose.yml not found in current directory.")
            return
            
        if self.console:
            self.console.print("🐳 Starting services with Docker Compose...")
            
        try:
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            
            if self.console:
                self.console.print("✅ Services started successfully")
                self.console.print("🌐 API: http://localhost:8000")
                self.console.print("📊 Dashboard: http://localhost:8000/dashboard")
                self.console.print("📈 Grafana: http://localhost:3000")
            else:
                print("Services started successfully")
                print("API: http://localhost:8000")
                
        except subprocess.CalledProcessError as e:
            self.show_error(f"Docker Compose failed: {e}")
        except FileNotFoundError:
            self.show_error("Docker Compose not found. Please install Docker.")
            
        self.wait_for_input()
        
    # Utility methods
    def get_input(self, prompt: str) -> str:
        """Get input from user."""
        if self.console:
            return Prompt.ask(prompt)
        else:
            return input(prompt)
            
    def show_error(self, message: str):
        """Show error message."""
        if self.console:
            self.console.print(f"[bold red]❌ Error:[/bold red] {message}")
        else:
            print(f"❌ Error: {message}")
            
    def show_info(self, message: str):
        """Show info message."""
        if self.console:
            self.console.print(f"[bold blue]ℹ️  Info:[/bold blue] {message}")
        else:
            print(f"ℹ️  Info: {message}")
            
    def wait_for_input(self):
        """Wait for user input to continue."""
        if self.console:
            Prompt.ask("Press Enter to continue", default="")
        else:
            input("Press Enter to continue...")
            
    def display_analysis_result(self, result):
        """Display analysis result in formatted way."""
        if self.console:
            table = Table(title=f"Analysis Result: {result.neo_data.designation}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Overall Score", f"{result.anomaly_score.overall_score:.3f}")
            table.add_row("Classification", result.anomaly_score.classification)
            table.add_row("Confidence", f"{result.anomaly_score.confidence:.3f}")
            table.add_row("Processing Time", f"{result.processing_time:.2f}s")
            
            self.console.print(table)
        else:
            print(f"\nAnalysis Result: {result.neo_data.designation}")
            print(f"Overall Score: {result.anomaly_score.overall_score:.3f}")
            print(f"Classification: {result.anomaly_score.classification}")
            print(f"Confidence: {result.anomaly_score.confidence:.3f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            
    # Placeholder methods for remaining functionality
    def interactive_analysis(self):
        self.show_info("Interactive analysis mode - Coming soon!")
        self.wait_for_input()
        
    def view_analysis_results(self):
        self.show_info("Analysis results viewer - Coming soon!")
        self.wait_for_input()
        
    def configure_analysis(self):
        self.show_info("Analysis configuration - Coming soon!")
        self.wait_for_input()
        
    def generate_reports(self):
        self.show_info("Statistical reports - Coming soon!")
        self.wait_for_input()
        
    def ml_training(self):
        self.show_info("ML model training - Coming soon!")
        self.wait_for_input()
        
    def ml_predictions(self):
        self.show_info("ML predictions - Coming soon!")
        self.wait_for_input()
        
    def model_management(self):
        self.show_info("Model management - Coming soon!")
        self.wait_for_input()
        
    def feature_analysis(self):
        self.show_info("Feature analysis - Coming soon!")
        self.wait_for_input()
        
    def model_performance(self):
        self.show_info("Model performance - Coming soon!")
        self.wait_for_input()
        
    def training_configuration(self):
        self.show_info("Training configuration - Coming soon!")
        self.wait_for_input()
        
    def model_export_import(self):
        self.show_info("Model export/import - Coming soon!")
        self.wait_for_input()
        
    def start_web_dashboard(self):
        self.show_info("Web dashboard will be available at /dashboard when API server is running")
        self.wait_for_input()
        
    def start_streaming_services(self):
        self.show_info("Streaming services will be available at /api/v1/stream when API server is running")
        self.wait_for_input()
        
    def api_performance_test(self):
        self.show_info("API performance testing - Coming soon!")
        self.wait_for_input()
        
    def manage_api_keys(self):
        self.show_info("API key management - Coming soon!")
        self.wait_for_input()
        
    def view_api_docs(self):
        self.show_info("API documentation will be available at /docs when API server is running")
        self.wait_for_input()
        
    def alert_management(self):
        self.show_info("Alert management - Coming soon!")
        self.wait_for_input()
        
    def performance_metrics(self):
        self.show_info("Performance metrics - Coming soon!")
        self.wait_for_input()
        
    def system_diagnostics(self):
        self.show_info("System diagnostics - Coming soon!")
        self.wait_for_input()
        
    def metrics_export(self):
        self.show_info("Metrics export - Coming soon!")
        self.wait_for_input()
        
    def configure_monitoring(self):
        self.show_info("Monitoring configuration - Coming soon!")
        self.wait_for_input()
        
    def database_management(self):
        self.show_info("Database management - Coming soon!")
        self.wait_for_input()
        
    def system_cleanup(self):
        self.show_info("System cleanup - Coming soon!")
        self.wait_for_input()
        
    def configuration_management(self):
        self.show_info("Configuration management - Coming soon!")
        self.wait_for_input()
        
    def user_management(self):
        self.show_info("User management - Coming soon!")
        self.wait_for_input()
        
    def system_maintenance(self):
        self.show_info("System maintenance - Coming soon!")
        self.wait_for_input()
        
    def installation_management(self):
        """Full installation and dependency management."""
        if self.console:
            self.console.print("📦 Installation & Dependency Management")
            
            options = [
                ("1", "🔧 Full Installation", "Complete aNEOS installation with all components"),
                ("2", "⚡ Minimal Installation", "Core components only"),
                ("3", "🔍 System Check", "Check system requirements and dependencies"),
                ("4", "🛠️  Fix Dependencies", "Fix missing or broken dependencies"),
                ("5", "📊 Installation Report", "View detailed installation status"),
                ("6", "🧹 Clean Install", "Clean installation (removes old data)"),
                ("", "", ""),
                ("0", "← Back", "Return to system management menu")
            ]
            
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Option", style="bold cyan")
            table.add_column("Description", style="white")
            
            for option, title, desc in options:
                if option:
                    table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                else:
                    table.add_row("", "", "")
            
            panel = Panel(table, title="[bold]📦 Installation Management[/bold]", border_style="green")
            self.console.print(panel)
            
            choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6"])
        else:
            print("\n--- Installation Management ---")
            print("1. Full Installation")
            print("2. Minimal Installation") 
            print("3. System Check")
            print("4. Fix Dependencies")
            print("5. Installation Report")
            print("6. Clean Install")
            print("0. Back")
            choice = input("Select option (0-6): ")
        
        if choice == "0":
            return
        elif choice == "1":
            self.run_installation("--full")
        elif choice == "2":
            self.run_installation("--minimal")
        elif choice == "3":
            self.run_installation("--check")
        elif choice == "4":
            self.run_installation("--fix-deps")
        elif choice == "5":
            self.show_installation_report()
        elif choice == "6":
            self.clean_install()
    
    def run_installation(self, args: str):
        """Run the installation script with specified arguments."""
        try:
            if self.console:
                self.console.print(f"🚀 Running installation with options: {args}")
                
            subprocess.run([sys.executable, "install.py"] + args.split())
            
            if self.console:
                self.console.print("✅ Installation command completed")
            else:
                print("✅ Installation command completed")
                
        except Exception as e:
            self.show_error(f"Installation failed: {e}")
        
        self.wait_for_input()
    
    def show_installation_report(self):
        """Show installation report if available."""
        report_path = Path("installation_report.json")
        if report_path.exists():
            try:
                import json
                with open(report_path) as f:
                    report = json.load(f)
                
                if self.console:
                    self.console.print("📊 Installation Report")
                    self.console.print(f"Installation Date: {report.get('installation_date', 'Unknown')}")
                    self.console.print(f"Python Version: {report.get('python_version', 'Unknown')}")
                    self.console.print(f"Success: {'✅ Yes' if report.get('success') else '❌ No'}")
                    
                    log_summary = {}
                    for log_entry in report.get('installation_log', []):
                        level = log_entry.get('level', 'UNKNOWN')
                        log_summary[level] = log_summary.get(level, 0) + 1
                    
                    self.console.print("\nLog Summary:")
                    for level, count in log_summary.items():
                        self.console.print(f"  {level}: {count}")
                else:
                    print(f"Installation Date: {report.get('installation_date', 'Unknown')}")
                    print(f"Python Version: {report.get('python_version', 'Unknown')}")
                    print(f"Success: {'Yes' if report.get('success') else 'No'}")
                    
            except Exception as e:
                self.show_error(f"Could not read installation report: {e}")
        else:
            self.show_info("No installation report found. Run installation first.")
        
        self.wait_for_input()
    
    def clean_install(self):
        """Perform clean installation after confirming with user."""
        if self.console:
            confirm = Confirm.ask("⚠️  This will remove all existing data and perform fresh installation. Continue?", default=False)
        else:
            confirm = input("⚠️  This will remove all existing data and perform fresh installation. Continue? (y/N): ").lower().startswith('y')
        
        if not confirm:
            self.show_info("Clean installation cancelled.")
            self.wait_for_input()
            return
        
        try:
            # Remove database
            db_files = ['aneos.db', 'aneos.db-wal', 'aneos.db-shm']
            for db_file in db_files:
                if Path(db_file).exists():
                    Path(db_file).unlink()
            
            # Remove cache directories
            cache_dirs = ['__pycache__', '.pytest_cache', 'cache']
            for cache_dir in cache_dirs:
                if Path(cache_dir).exists():
                    import shutil
                    shutil.rmtree(cache_dir)
            
            # Run fresh installation
            self.show_info("Running fresh installation...")
            subprocess.run([sys.executable, "install.py", "--full"])
            
            self.show_info("✅ Clean installation completed!")
            
        except Exception as e:
            self.show_error(f"Clean installation failed: {e}")
        
        self.wait_for_input()

    def dependency_check(self):
        """Quick dependency verification."""
        self.run_installation("--check")
        
    def system_reset(self):
        self.show_info("System reset - Coming soon!")
        self.wait_for_input()
        
    def run_tests(self):
        self.show_info("Running tests - Coming soon!")
        self.wait_for_input()
        
    def debug_mode(self):
        self.show_info("Debug mode - Coming soon!")
        self.wait_for_input()
        
    def code_analysis(self):
        self.show_info("Code analysis - Coming soon!")
        self.wait_for_input()
        
    def performance_profiling(self):
        self.show_info("Performance profiling - Coming soon!")
        self.wait_for_input()
        
    def memory_analysis(self):
        self.show_info("Memory analysis - Coming soon!")
        self.wait_for_input()
        
    def generate_documentation(self):
        self.show_info("Documentation generation - Coming soon!")
        self.wait_for_input()
        
    def development_server(self):
        self.development_mode()
        
    def build_docker_images(self):
        self.show_info("Building Docker images - Coming soon!")
        self.wait_for_input()
        
    def kubernetes_deploy(self):
        self.show_info("Kubernetes deployment - Coming soon!")
        self.wait_for_input()
        
    def container_status(self):
        self.show_info("Container status - Coming soon!")
        self.wait_for_input()
        
    def view_logs(self):
        self.show_info("Log viewer - Coming soon!")
        self.wait_for_input()
        
    def scale_services(self):
        self.show_info("Service scaling - Coming soon!")
        self.wait_for_input()
        
    def stop_services(self):
        self.show_info("Stop services - Coming soon!")
        self.wait_for_input()
        
    def cleanup_containers(self):
        self.show_info("Container cleanup - Coming soon!")
        self.wait_for_input()
        
    def show_user_guide(self):
        self.show_info("User guide - Coming soon!")
        self.wait_for_input()
        
    def show_scientific_docs(self):
        self.show_info("Scientific documentation - Coming soon!")
        self.wait_for_input()
        
    def show_ml_docs(self):
        self.show_info("ML documentation - Coming soon!")
        self.wait_for_input()
        
    def show_api_docs(self):
        self.view_api_docs()
        
    def show_deployment_guide(self):
        self.show_info("Deployment guide - Coming soon!")
        self.wait_for_input()
        
    def show_troubleshooting(self):
        self.show_info("Troubleshooting guide - Coming soon!")
        self.wait_for_input()
        
    def show_system_requirements(self):
        self.show_info("System requirements - Coming soon!")
        self.wait_for_input()
        
    def show_config_reference(self):
        self.show_info("Configuration reference - Coming soon!")
        self.wait_for_input()
        
    def advanced_features_menu(self):
        """Advanced features submenu - postponed until core is stable."""
        while True:
            if self.console:
                self.console.clear()
                self.display_header()
                
                # Warning panel
                warning = Panel(
                    "[bold yellow]⚠️  Advanced Features[/bold yellow]\n\n"
                    "These features are postponed until we have stable and reliable core functionality\n"
                    "and academic rigorous methods for NEO enumeration, assessment and classification.\n\n"
                    "Focus: Establish stable core before advanced orchestration.",
                    border_style="yellow"
                )
                self.console.print(warning)
                self.console.print()
                
                table = Table(show_header=False, box=None, padding=(0, 2))
                table.add_column("Option", style="bold cyan")
                table.add_column("Description", style="white")
                
                options = [
                    ("1", "🤖 Machine Learning", "ML training, predictions, model management [POSTPONED]"),
                    ("2", "🌐 Advanced API Services", "Streaming, performance testing, production APIs [POSTPONED]"),
                    ("3", "📊 Advanced Monitoring", "Real-time dashboards, metrics export, alerts [POSTPONED]"),
                    ("4", "🛠️  Development Tools", "Code analysis, profiling, advanced debugging [POSTPONED]"),
                    ("5", "🐳 Docker & Deployment", "Containerization, Kubernetes, production deployment [POSTPONED]"),
                    ("6", "📡 Stream Processing", "High-volume traffic analysis, distributed processing [POSTPONED]"),
                    ("", "", ""),
                    ("0", "← Back to Main Menu", "Return to main menu")
                ]
                
                for option, title, desc in options:
                    if option:
                        table.add_row(f"[bold]{option}[/bold]", f"{title}", f"[dim]{desc}[/dim]")
                    else:
                        table.add_row("", "", "")
                        
                panel = Panel(table, title="[bold]🚀 Advanced Features (Postponed)[/bold]", border_style="red")
                self.console.print(panel)
                
                choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "6"])
            else:
                print("\n--- Advanced Features (Postponed) ---")
                print("⚠️  These features are postponed until core functionality is stable")
                print("")
                print("1. Machine Learning [POSTPONED]")
                print("2. Advanced API Services [POSTPONED]")
                print("3. Advanced Monitoring [POSTPONED]")
                print("4. Development Tools [POSTPONED]")
                print("5. Docker & Deployment [POSTPONED]")
                print("6. Stream Processing [POSTPONED]")
                print("0. Back to Main Menu")
                choice = input("Select option (0-6): ")
                
            if choice == "0":
                break
            elif choice == "1":
                self.postponed_feature("Machine Learning", "ML training, predictions and model management will be available after core functionality is validated.")
            elif choice == "2":
                self.postponed_feature("Advanced API Services", "Streaming services, performance testing and production APIs will be available after core stability.")
            elif choice == "3":
                self.postponed_feature("Advanced Monitoring", "Real-time dashboards, metrics export and alert systems will be available after core validation.")
            elif choice == "4":
                self.postponed_feature("Development Tools", "Code analysis, profiling and advanced debugging tools will be available after core methodology is established.")
            elif choice == "5":
                self.postponed_feature("Docker & Deployment", "Containerization, Kubernetes and production deployment will be available after reliable core functionality.")
            elif choice == "6":
                self.postponed_feature("Stream Processing", "High-volume distributed processing will be available after academic rigorous methods are established.")
    
    def postponed_feature(self, feature_name: str, description: str):
        """Display postponed feature message."""
        if self.console:
            panel = Panel(
                f"[bold yellow]{feature_name}[/bold yellow]\n\n"
                f"{description}\n\n"
                "[bold]Current Priority:[/bold] Establish stable core functionality\n"
                "[bold]Focus Areas:[/bold]\n"
                "• Academic rigorous NEO classification methods\n"
                "• Reliable enumeration and assessment pipeline\n"
                "• 100% data quality assurance\n"
                "• Validated artificial NEO detection algorithms\n\n"
                "[dim]This feature will be enabled once core objectives are met.[/dim]",
                title=f"[bold red]⚠️  {feature_name} - Postponed[/bold red]",
                border_style="red"
            )
            self.console.print(panel)
        else:
            print(f"\n⚠️  {feature_name} - Postponed")
            print(f"{description}")
            print("\nCurrent Priority: Establish stable core functionality")
            print("This feature will be enabled once core objectives are met.")
        
        self.wait_for_input()
        
    def api_health_check(self):
        """Basic API health check."""
        self.show_info("Running basic API health check...")
        
        try:
            import requests
            # Try to check if API server is running
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                self.show_info("✅ API server is running and healthy")
            else:
                self.show_info(f"⚠️  API server responded with status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.show_info("❌ API server is not running. Start it first with 'Start API Server'.")
        except Exception as e:
            self.show_error(f"API health check failed: {e}")
            
        self.wait_for_input()
        
    def basic_system_status(self):
        """Display basic system status."""
        self.display_system_status()
        self.wait_for_input()
        
    def run_basic_tests(self):
        """Run basic system validation tests."""
        self.show_info("Running basic system validation tests...")
        
        tests_passed = 0
        total_tests = 4
        
        # Test 1: Core components
        if HAS_ANEOS_CORE:
            self.show_info("✅ Test 1/4: Core components available")
            tests_passed += 1
        else:
            self.show_info("❌ Test 1/4: Core components missing")
            
        # Test 2: Database
        if HAS_DATABASE:
            try:
                db_status = get_database_status()
                if db_status.get('available'):
                    self.show_info("✅ Test 2/4: Database connection working")
                    tests_passed += 1
                else:
                    self.show_info("❌ Test 2/4: Database connection failed")
            except Exception:
                self.show_info("❌ Test 2/4: Database test failed")
        else:
            self.show_info("❌ Test 2/4: Database test skipped (components not loaded)")
            
        # Test 3: File system
        required_dirs = ['data', 'logs', 'cache']
        missing_dirs = [d for d in required_dirs if not Path(d).exists()]
        if not missing_dirs:
            self.show_info("✅ Test 3/4: Required directories exist")
            tests_passed += 1
        else:
            self.show_info(f"❌ Test 3/4: Missing directories: {', '.join(missing_dirs)}")
            
        # Test 4: Enhanced NEO poller
        if Path('enhanced_neo_poller.py').exists():
            self.show_info("✅ Test 4/4: Enhanced NEO poller available")
            tests_passed += 1
        else:
            self.show_info("❌ Test 4/4: Enhanced NEO poller missing")
            
        success_rate = (tests_passed / total_tests) * 100
        
        if self.console:
            if success_rate >= 75:
                self.console.print(f"\n[green]✅ Basic tests: {success_rate:.0f}% passed ({tests_passed}/{total_tests})[/green]")
            else:
                self.console.print(f"\n[yellow]⚠️  Basic tests: {success_rate:.0f}% passed ({tests_passed}/{total_tests})[/yellow]")
        else:
            print(f"\nBasic tests: {success_rate:.0f}% passed ({tests_passed}/{total_tests})")
            
        self.wait_for_input()
        
    def basic_system_diagnostics(self):
        """Basic system diagnostics."""
        if self.console:
            self.console.print("🔍 Basic System Diagnostics")
            
            table = Table(title="System Information", show_header=True, header_style="bold cyan")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="white")
            
            table.add_row("Python Version", f"{sys.version.split()[0]}")
            table.add_row("Working Directory", str(Path.cwd()))
            table.add_row("Core Components", "Available" if HAS_ANEOS_CORE else "Limited")
            table.add_row("Rich UI", "Available" if HAS_RICH else "Not Available")
            
            # Check key files
            key_files = ['enhanced_neo_poller.py', 'install.py', 'aneos.py']
            for file_name in key_files:
                exists = Path(file_name).exists()
                table.add_row(f"File: {file_name}", "✅ Found" if exists else "❌ Missing")
                
            self.console.print(table)
        else:
            print("\n--- Basic System Diagnostics ---")
            print(f"Python Version: {sys.version.split()[0]}")
            print(f"Working Directory: {Path.cwd()}")
            print(f"Core Components: {'Available' if HAS_ANEOS_CORE else 'Limited'}")
            
        self.wait_for_input()
        
    def run(self):
        """Main menu loop."""
        # Show initial focus message
        if self.console:
            focus_panel = Panel(
                "[bold green]🎯 Current Focus: Stable Core Functionality[/bold green]\n\n"
                "Priority: Establish reliable NEO analysis with academic rigor\n"
                "• 100% data quality assurance\n"
                "• Enhanced TAS-based artificial detection\n"
                "• Validated enumeration and assessment pipeline\n\n"
                "[dim]Advanced orchestration features postponed until core is stable.[/dim]",
                title="[bold]aNEOS Strategic Focus[/bold]",
                border_style="green"
            )
            self.console.print(focus_panel)
            self.wait_for_input()
        
        while self.running:
            try:
                if self.console:
                    self.console.clear()
                else:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    
                self.display_header()
                print()
                self.display_system_status()
                print()
                self.display_main_menu()
                
                if self.console:
                    choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4", "5", "9"])
                else:
                    choice = input("\nSelect option (0-5, 9): ")
                    
                if choice == "0":
                    self.running = False
                elif choice == "1":
                    self.scientific_analysis_menu()
                elif choice == "2":
                    self.basic_api_services_menu()
                elif choice == "3":
                    self.system_management_menu()
                elif choice == "4":
                    self.health_diagnostics_menu()
                elif choice == "5":
                    self.help_documentation_menu()
                elif choice == "9":
                    self.advanced_features_menu()
                else:
                    self.show_error("Invalid option selected.")
                    self.wait_for_input()
                    
            except KeyboardInterrupt:
                if self.console:
                    self.console.print("\n👋 Goodbye!")
                else:
                    print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.show_error(f"Unexpected error: {e}")
                if self.console and Confirm.ask("Continue?"):
                    continue
                else:
                    break
                    
        if self.console:
            self.console.print("✅ aNEOS menu system closed")
        else:
            print("✅ aNEOS menu system closed")

def main():
    """Main entry point."""
    print("🚀 Starting aNEOS Menu System...")
    
    # Check if we're in the right directory
    if not Path("aneos_core").exists():
        print("❌ Error: Please run this script from the aNEOS project root directory")
        print("Current directory should contain 'aneos_core' folder")
        sys.exit(1)
        
    menu = ANEOSMenu()
    menu.run()

if __name__ == "__main__":
    main()