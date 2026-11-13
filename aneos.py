#!/usr/bin/env python3
"""
aNEOS Quick Launcher

Simple launcher script for aNEOS operations with command-line arguments
for both interactive menu and direct command execution.
"""

import sys
import subprocess
from pathlib import Path
from aneos_version import APPLICATION_VERSION, BUILD_CHANNEL

def main():
    """Main launcher with command-line argument support."""
    
    if len(sys.argv) == 2 and sys.argv[1] in ('--version', 'version'):
        print(f'aNEOS {APPLICATION_VERSION} ({BUILD_CHANNEL})')
        return

    # Check if we're in the right directory
    if not Path("aneos_core").exists():
        print("❌ Error: Please run this script from the aNEOS project root directory")
        print("Current directory should contain 'aneos_core' folder")
        sys.exit(1)
    
    # --legacy-menu selects the original grouped menu (full feature QA pending)
    if "--legacy-menu" in sys.argv:
        sys.argv.remove("--legacy-menu")
        print(f"Starting aNEOS {APPLICATION_VERSION} legacy grouped menu…")
        subprocess.run([sys.executable, "aneos_menu.py"])
        return

    # If no arguments, start the new v2 menu
    if len(sys.argv) == 1:
        print(f"Starting aNEOS {APPLICATION_VERSION} ({BUILD_CHANNEL}) menu…")
        subprocess.run([sys.executable, "aneos_menu_v2.py"])
        return

    # Parse command-line arguments for direct execution
    command = sys.argv[1].lower()

    if command in ['menu', 'interactive', 'm']:
        print(f"Starting aNEOS {APPLICATION_VERSION} ({BUILD_CHANNEL}) menu…")
        subprocess.run([sys.executable, "aneos_menu_v2.py"])
        
    elif command in ['api', 'server', 'start']:
        print("🚀 Starting aNEOS API Server...")
        args = ["python", "start_api.py"]
        
        # Parse additional arguments
        if len(sys.argv) > 2:
            if '--dev' in sys.argv or '-d' in sys.argv:
                args.append("--dev")
            if '--host' in sys.argv:
                idx = sys.argv.index('--host')
                if idx + 1 < len(sys.argv):
                    args.extend(["--host", sys.argv[idx + 1]])
            if '--port' in sys.argv:
                idx = sys.argv.index('--port')
                if idx + 1 < len(sys.argv):
                    args.extend(["--port", sys.argv[idx + 1]])
                    
        subprocess.run(args)
        
    elif command in ['analyze', 'analysis', 'a', 'simple', 'detect']:
        from aneos_core.analysis.cli import main as analysis_main
        return analysis_main(sys.argv[2:])

    elif command in ['docker', 'compose', 'up']:
        print("🐳 Starting Docker Compose services...")
        subprocess.run(["docker-compose", "up", "-d"])
        print("✅ Services started!")
        print("🌐 API: http://localhost:8000")
        print("📊 Dashboard: http://localhost:8000/dashboard")
        print("📈 Grafana: http://localhost:3000")
        
    elif command in ['status', 'health', 'check']:
        sys.path.insert(0, str(Path(__file__).parent))
        from aneos_core.utils.health import preflight_check, print_preflight_table
        results = preflight_check()
        print_preflight_table(results)
        sys.exit(0 if all(v["status"] == "ok" for v in results.values()) else 1)
        
    elif command in ['install', 'setup', 'i']:
        print("📦 Starting aNEOS Installation...")
        args = ["python", "install.py"]
        
        # Parse additional arguments
        if len(sys.argv) > 2:
            if '--full' in sys.argv:
                args.append("--full")
            elif '--minimal' in sys.argv:
                args.append("--minimal")
            elif '--check' in sys.argv:
                args.append("--check")
            elif '--fix-deps' in sys.argv:
                args.append("--fix-deps")
                
        subprocess.run(args)
        
    elif command in ['poll', 'poller']:
        print("🌍 Starting NEO API Poller...")
        args = [sys.executable, "neo_poller.py"]
        
        # Parse additional arguments for direct polling
        if len(sys.argv) > 2:
            # Check if it's a direct command
            if '--api' in sys.argv or '--period' in sys.argv:
                args.extend(sys.argv[2:])  # Pass all remaining arguments
            else:
                # Assume it's just a time period
                period = sys.argv[2]
                args.extend(["--api", "NASA_CAD", "--period", period])
                
        subprocess.run(args)
        
    elif command in ['help', '--help', '-h']:
        print_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        print_help()

def print_help():
    """Print help information."""
    help_text = """
🚀 aNEOS - Advanced Near Earth Object detection System

Usage: python aneos.py [command] [options]

Commands:
  (no command)    Start aNEOS menu (v2 — screening, evidence, results, system)
  menu, m         Start aNEOS menu (v2)
  --legacy-menu   Start legacy grouped menu (feature QA remains pending)
  
  install, setup  Install/setup aNEOS system
    --full        Full installation with all components
    --minimal     Minimal installation (core only)
    --check       Check system requirements
    --fix-deps    Fix dependency issues
  
  api, server     Start API server
    --dev, -d     Start in development mode
    --host HOST   Specify host (default: 0.0.0.0)
    --port PORT   Specify port (default: 8000)
  
  analyze DESIGNATION    Screen actual evidence (uncalibrated ranking)
    --json             Machine-readable report instead of Markdown
    --force-refresh    Bypass source cache
    --output PATH      Create report file without overwriting existing work
    Example: python aneos.py analyze "2024 AB123"
  
  simple, detect DESIGNATION  Aliases for the same actual screening pipeline
    Example: python aneos.py simple "1998 KY26"
  
  poll [PERIOD]          Poll NEO APIs for artificial signatures
    Example: python aneos.py poll 1m
    Example: python aneos.py poll --api NASA_CAD --period 6m
  
  docker, up      Start Docker Compose services
  status, health  Check system status
  help, -h        Show this help message

Examples:
  python aneos.py                           # Interactive menu
  python aneos.py simple "1998 KY26"             # Actual evidence screening; no origin confirmation
  python aneos.py poll 30                   # Poll last 30 days for artificial NEOs
  python aneos.py analyze "2024 XY123"      # Actual evidence screening
  python aneos.py api --dev                 # Development API server
  python aneos.py docker                    # Start with Docker
  python aneos.py status                    # System health check

For full functionality, use the interactive menu:
  python aneos.py menu

📚 Documentation: Available at /docs when API server is running
🌐 Web Dashboard: Available at /dashboard when API server is running
"""
    print(f'aNEOS {APPLICATION_VERSION} ({BUILD_CHANNEL})')
    print('  --version, version    Show application version without starting services')
    print(help_text)

if __name__ == "__main__":
    main()