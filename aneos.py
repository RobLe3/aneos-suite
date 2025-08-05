#!/usr/bin/env python3
"""
aNEOS Quick Launcher

Simple launcher script for aNEOS operations with command-line arguments
for both interactive menu and direct command execution.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main launcher with command-line argument support."""
    
    # Check if we're in the right directory
    if not Path("aneos_core").exists():
        print("❌ Error: Please run this script from the aNEOS project root directory")
        print("Current directory should contain 'aneos_core' folder")
        sys.exit(1)
    
    # If no arguments, start interactive menu
    if len(sys.argv) == 1:
        print("🚀 Starting aNEOS Interactive Menu...")
        subprocess.run([sys.executable, "aneos_menu.py"])
        return
    
    # Parse command-line arguments for direct execution
    command = sys.argv[1].lower()
    
    if command in ['menu', 'interactive', 'm']:
        print("🚀 Starting aNEOS Interactive Menu...")
        subprocess.run([sys.executable, "aneos_menu.py"])
        
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
        
    elif command in ['analyze', 'analysis', 'a']:
        if len(sys.argv) < 3:
            print("❌ Usage: python aneos.py analyze <NEO_DESIGNATION>")
            print("Example: python aneos.py analyze '2024 AB123'")
            sys.exit(1)
            
        designation = sys.argv[2]
        print(f"🔬 Analyzing NEO: {designation}")
        
        # Quick analysis script
        analysis_script = f"""
import sys
sys.path.insert(0, '.')
try:
    from aneos_core.analysis.pipeline import create_analysis_pipeline
    import asyncio
    
    async def quick_analysis():
        pipeline = create_analysis_pipeline()
        result = await pipeline.analyze_neo('{designation}')
        
        if result:
            print(f"\\n📊 Analysis Results for {designation}:")
            print(f"Overall Score: {{result.anomaly_score.overall_score:.3f}}")
            print(f"Classification: {{result.anomaly_score.classification}}")
            print(f"Confidence: {{result.anomaly_score.confidence:.3f}}")
            print(f"Processing Time: {{result.processing_time:.2f}}s")
            
            if result.anomaly_score.risk_factors:
                print(f"\\n🚨 Risk Factors:")
                for factor in result.anomaly_score.risk_factors:
                    print(f"  • {{factor}}")
        else:
            print(f"❌ Analysis failed for {{designation}}")
            
    asyncio.run(quick_analysis())
    
except ImportError as e:
    print(f"❌ Error: Missing dependencies - {{e}}")
    print("Please run: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Analysis error: {{e}}")
"""
        
        # Execute the analysis
        result = subprocess.run([sys.executable, "-c", analysis_script])
        
    elif command in ['docker', 'compose', 'up']:
        print("🐳 Starting Docker Compose services...")
        subprocess.run(["docker-compose", "up", "-d"])
        print("✅ Services started!")
        print("🌐 API: http://localhost:8000")
        print("📊 Dashboard: http://localhost:8000/dashboard")
        print("📈 Grafana: http://localhost:3000")
        
    elif command in ['status', 'health', 'check']:
        print("🔍 Checking aNEOS system status...")
        
        status_script = """
import sys
sys.path.insert(0, '.')

try:
    from aneos_api.database import get_database_status
    
    print("📊 System Status Check:")
    print("=" * 40)
    
    # Check core components
    try:
        from aneos_core.analysis.pipeline import create_analysis_pipeline
        print("✅ Core Analysis: Available")
    except ImportError:
        print("❌ Core Analysis: Missing dependencies")
        
    # Check API components
    try:
        from aneos_api.app import create_app
        print("✅ API Services: Available")
    except ImportError:
        print("❌ API Services: Missing dependencies")
        
    # Check database
    try:
        db_status = get_database_status()
        if db_status.get('available'):
            print(f"✅ Database: Connected ({db_status.get('engine', 'Unknown')})")
        else:
            print(f"⚠️  Database: {db_status.get('error', 'Not connected')}")
    except Exception as e:
        print(f"❌ Database: Error - {e}")
        
    # Check required directories
    from pathlib import Path
    dirs = ['data', 'logs', 'models', 'cache']
    missing = [d for d in dirs if not Path(d).exists()]
    
    if not missing:
        print("✅ File System: All directories exist")
    else:
        print(f"⚠️  File System: Missing directories: {', '.join(missing)}")
        
    print("=" * 40)
    
except Exception as e:
    print(f"❌ Status check failed: {e}")
"""
        
        subprocess.run([sys.executable, "-c", status_script])
        
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
        
    elif command in ['simple', 'detect']:
        print("🔍 Starting Simple NEO Analyzer...")
        args = ["python3", "simple_neo_analyzer.py"]
        
        # Parse additional arguments
        if len(sys.argv) > 2:
            if command == 'simple':
                args.extend(["single", sys.argv[2]])
                
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
  (no command)    Start interactive menu system
  menu, m         Start interactive menu system
  
  install, setup  Install/setup aNEOS system
    --full        Full installation with all components
    --minimal     Minimal installation (core only)
    --check       Check system requirements
    --fix-deps    Fix dependency issues
  
  api, server     Start API server
    --dev, -d     Start in development mode
    --host HOST   Specify host (default: 0.0.0.0)
    --port PORT   Specify port (default: 8000)
  
  analyze DESIGNATION    Analyze a single NEO (complex analysis)
    Example: python aneos.py analyze "2024 AB123"
  
  simple DESIGNATION     Simple artificial NEO detection
    Example: python aneos.py simple "test"
  
  poll [PERIOD]          Poll NEO APIs for artificial signatures
    Example: python aneos.py poll 1m
    Example: python aneos.py poll --api NASA_CAD --period 6m
  
  docker, up      Start Docker Compose services
  status, health  Check system status
  help, -h        Show this help message

Examples:
  python aneos.py                           # Interactive menu
  python aneos.py simple "test"             # Simple artificial NEO detection (demo)
  python aneos.py poll 30                   # Poll last 30 days for artificial NEOs
  python aneos.py analyze "2024 XY123"      # Complex NEO analysis
  python aneos.py api --dev                 # Development API server
  python aneos.py docker                    # Start with Docker
  python aneos.py status                    # System health check

For full functionality, use the interactive menu:
  python aneos.py menu

📚 Documentation: Available at /docs when API server is running
🌐 Web Dashboard: Available at /dashboard when API server is running
"""
    print(help_text)

if __name__ == "__main__":
    main()