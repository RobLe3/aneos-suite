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
        return subprocess.run([sys.executable, "aneos_menu.py"]).returncode

    # If no arguments, start the new v2 menu
    if len(sys.argv) == 1:
        print(f"Starting aNEOS {APPLICATION_VERSION} ({BUILD_CHANNEL}) menu…")
        return subprocess.run([sys.executable, "aneos_menu_v2.py"]).returncode

    # Parse command-line arguments for direct execution
    command = sys.argv[1].lower()

    if command in ['menu', 'interactive', 'm']:
        print(f"Starting aNEOS {APPLICATION_VERSION} ({BUILD_CHANNEL}) menu…")
        return subprocess.run([sys.executable, "aneos_menu_v2.py"]).returncode
        
    elif command in ['api', 'server', 'start']:
        print("🚀 Starting aNEOS API Server...")
        args = [sys.executable, "start_api.py"]
        
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
                    
        return subprocess.run(args).returncode
        
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
                
        return subprocess.run(args).returncode
        
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
                
        return subprocess.run(args).returncode
        
    elif command in ['help', '--help', '-h']:
        print_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        print_help()

def print_help():
    """Print help information."""
    help_text = """
aNEOS — Advanced Near Earth Object Suite

Usage: python aneos.py [command] [options]
Run from the checkout root with the installed environment activated.

Commands:
  (no command), menu, m  Start primary menu; full feature QA remains pending
  --legacy-menu         Start the original grouped menu
  analyze DESIGNATION   Screen actual evidence using an uncalibrated ranking
    --json              Machine-readable report instead of Markdown
    --force-refresh     Bypass application source cache, not upstream caches
    --output PATH       Create report without overwriting an existing file
  simple, detect        Aliases for the same named-object screening pipeline
  poll [PERIOD]         Bounded NASA_CAD encounter collection (development)
    Explicit source: python aneos.py poll --api NASA_CAD --period 1d
    Native row/enrichment limits: python neo_poller.py --help
    Other root bulk discovery routes remain unavailable.
  api, server           Start API; default host 0.0.0.0, port 8000
    Local testing: python aneos.py api --dev --host 127.0.0.1 --port 8000
  status, health        Component checks, not scientific certification
  install, setup        Legacy installer; prefer the isolated installation guide
  docker, up            Legacy Compose launcher; deployment is not qualified here
  help, -h              Show this help without polling providers

Examples:
  python aneos.py analyze "1998 KY26" --json --output beta-runtime/ky26.json
  python aneos.py analyze --help
  python aneos.py menu

Queries may be partial or unavailable. NOT_FLAGGED does not mean natural origin.
The 5.0 ranking threshold is not Gaussian sigma; origin probability is not calibrated.
See README.md and docs/README.md for current scope and release differences.
API schema: http://127.0.0.1:8000/docs when the local server is running.
"""
    print(f'aNEOS {APPLICATION_VERSION} ({BUILD_CHANNEL})')
    print('  --version, version    Show application version without starting services')
    print(help_text)

if __name__ == "__main__":
    sys.exit(main())