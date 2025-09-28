#!/usr/bin/env python3
"""
Artificial NEO Dashboard Launcher

Standalone launcher for the artificial NEO analysis dashboard that can be integrated
into any polling or analysis workflow. Provides comprehensive categorization and
visualization of artificial NEO detection results.
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

async def launch_dashboard_from_file(file_path: str):
    """Launch dashboard from saved polling results."""
    try:
        from aneos_core.analysis.artificial_neo_dashboard import create_dashboard_from_polling
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract polling results from file
        if 'results' in data:
            polling_results = data['results']
        elif isinstance(data, list):
            polling_results = data
        else:
            polling_results = [data]
        
        print(f"📂 Loading results from: {file_path}")
        print(f"📊 Found {len(polling_results)} objects to analyze")
        
        # Create and display dashboard
        dashboard = await create_dashboard_from_polling(
            polling_results=polling_results,
            display=True,
            save=True
        )
        
        return dashboard
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading dashboard: {e}")
        return None

async def launch_dashboard_from_polling(api: str = "NASA_CAD", period: str = "1d", max_results: int = 20):
    """Launch dashboard by running polling first."""
    try:
        from neo_poller import NEOPoller
        
        print(f"🔄 Running polling: {api} for {period}")
        
        # Run polling
        poller = NEOPoller()
        results = poller.poll_and_analyze(api, period, max_results)
        
        if not results:
            print("❌ No polling results available")
            return None
        
        print(f"📊 Polling complete: {len(results)} objects found")
        
        # Launch dashboard
        from aneos_core.analysis.artificial_neo_dashboard import create_dashboard_from_polling
        dashboard = await create_dashboard_from_polling(
            polling_results=results,
            display=True,
            save=True
        )
        
        return dashboard
        
    except Exception as e:
        print(f"❌ Polling and dashboard launch failed: {e}")
        return None

def find_recent_polling_results(results_dir: str = ".") -> List[str]:
    """Find recent polling result files."""
    results_path = Path(results_dir)
    
    # Look for NEO polling result files
    patterns = [
        "neo_poll_*.json",
        "enhanced_neo_poll_*.json",
        "*artificial_neo_analysis*.json"
    ]
    
    files = []
    for pattern in patterns:
        files.extend(results_path.glob(pattern))
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return [str(f) for f in files[:10]]  # Return top 10 recent files

async def interactive_dashboard_launcher():
    """Interactive launcher for the dashboard."""
    try:
        from rich.console import Console
        from rich.prompt import Prompt, Confirm
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        console.clear()
        
        # Title
        title = Panel(
            "🛸 ARTIFICIAL NEO ANALYSIS DASHBOARD LAUNCHER",
            style="bold cyan",
            padding=(1, 2)
        )
        console.print(title)
        
        while True:
            # Menu options
            console.print("\n📋 [bold]Available Options:[/bold]")
            
            options_table = Table(show_header=False, box=None, padding=(0, 2))
            options_table.add_column("Option", style="bold cyan")
            options_table.add_column("Description", style="white")
            
            options_table.add_row("1", "🔄 Run new polling and launch dashboard")
            options_table.add_row("2", "📂 Load dashboard from existing results file")
            options_table.add_row("3", "📁 Show recent result files")
            options_table.add_row("4", "❓ Help and information")
            options_table.add_row("0", "🚪 Exit")
            
            console.print(options_table)
            
            choice = Prompt.ask("Select option", choices=["0", "1", "2", "3", "4"])
            
            if choice == "0":
                console.print("👋 Goodbye!")
                break
            
            elif choice == "1":
                # Run new polling
                console.print("\n🔄 [bold]Configure New Polling[/bold]")
                
                apis = ["NASA_CAD", "NASA_SBDB", "MPC", "NEODyS"]
                periods = ["1d", "1w", "1m", "3m", "6m", "1y"]
                
                api = Prompt.ask("Select API", choices=apis, default="NASA_CAD")
                period = Prompt.ask("Select time period", choices=periods, default="1d")
                max_results = int(Prompt.ask("Maximum results", default="20"))
                
                dashboard = await launch_dashboard_from_polling(api, period, max_results)
                
                if dashboard and Confirm.ask("Continue with interactive analysis?"):
                    # Dashboard includes its own interactive explorer
                    pass
            
            elif choice == "2":
                # Load from file
                file_path = Prompt.ask("Enter path to results file")
                
                if Path(file_path).exists():
                    dashboard = await launch_dashboard_from_file(file_path)
                else:
                    console.print(f"❌ [red]File not found: {file_path}[/red]")
            
            elif choice == "3":
                # Show recent files
                console.print("\n📁 [bold]Recent Result Files:[/bold]")
                
                recent_files = find_recent_polling_results()
                
                if recent_files:
                    files_table = Table(show_header=True)
                    files_table.add_column("ID", style="bold")
                    files_table.add_column("File", style="cyan")
                    files_table.add_column("Modified", style="dim")
                    
                    for i, file_path in enumerate(recent_files, 1):
                        file_obj = Path(file_path)
                        mod_time = file_obj.stat().st_mtime
                        mod_str = str(file_obj.stat().st_mtime)[:10]  # Simplified timestamp
                        
                        files_table.add_row(str(i), file_obj.name, mod_str)
                    
                    console.print(files_table)
                    
                    if Confirm.ask("Load one of these files?"):
                        file_id = int(Prompt.ask("Enter file ID")) - 1
                        if 0 <= file_id < len(recent_files):
                            dashboard = await launch_dashboard_from_file(recent_files[file_id])
                        else:
                            console.print("❌ Invalid file ID")
                else:
                    console.print("📭 No recent result files found")
            
            elif choice == "4":
                # Help
                help_panel = Panel(
                    """
🛸 [bold cyan]Artificial NEO Analysis Dashboard[/bold cyan]

This dashboard analyzes NEO polling results and categorizes objects into:

• [red]ARTIFICIAL[/red] - High confidence artificial objects (≥80% probability)
• [yellow]SUSPICIOUS[/yellow] - Objects requiring investigation (50-80% probability)  
• [blue]EDGE_CASE[/blue] - Borderline/unusual objects (30-50% probability)
• [green]NATURAL[/green] - Confirmed natural objects (<30% probability)

[bold]Features:[/bold]
• Comprehensive risk factor analysis
• Orbital anomaly detection
• Multi-source data integration
• Interactive object exploration
• Professional reporting

[bold]Data Sources:[/bold]
NASA CAD, SBDB, MPC, NEODyS

[bold]Analysis Methods:[/bold]
Enhanced TAS scoring, Sigma-5 detection, Multi-modal analysis
                    """,
                    title="[bold]📚 Help & Information[/bold]",
                    border_style="green"
                )
                console.print(help_panel)
    
    except ImportError:
        # Fallback to text interface
        await text_interactive_launcher()

async def text_interactive_launcher():
    """Text-only interactive launcher."""
    print("\n🛸 ARTIFICIAL NEO ANALYSIS DASHBOARD LAUNCHER")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Run new polling and launch dashboard")
        print("2. Load dashboard from existing results file")
        print("3. Show recent result files")
        print("0. Exit")
        
        choice = input("\nSelect option (0-3): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        elif choice == "1":
            print("\nPolling Configuration:")
            api = input("API (NASA_CAD/NASA_SBDB/MPC/NEODyS) [NASA_CAD]: ").strip() or "NASA_CAD"
            period = input("Period (1d/1w/1m/3m/6m/1y) [1d]: ").strip() or "1d"
            max_results = int(input("Max results [20]: ").strip() or "20")
            
            dashboard = await launch_dashboard_from_polling(api, period, max_results)
        
        elif choice == "2":
            file_path = input("Enter path to results file: ").strip()
            if Path(file_path).exists():
                dashboard = await launch_dashboard_from_file(file_path)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == "3":
            recent_files = find_recent_polling_results()
            if recent_files:
                print("\nRecent result files:")
                for i, file_path in enumerate(recent_files, 1):
                    print(f"{i}. {Path(file_path).name}")
                
                try:
                    file_id = int(input("Enter file ID to load (0 to cancel): ")) - 1
                    if 0 <= file_id < len(recent_files):
                        dashboard = await launch_dashboard_from_file(recent_files[file_id])
                except ValueError:
                    print("❌ Invalid input")
            else:
                print("📭 No recent result files found")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Artificial NEO Analysis Dashboard Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python artificial_neo_dashboard_launcher.py                    # Interactive mode
  python artificial_neo_dashboard_launcher.py --file results.json # Load from file
  python artificial_neo_dashboard_launcher.py --poll NASA_CAD     # Run polling first
        """
    )
    
    parser.add_argument('--file', help='Load dashboard from results file')
    parser.add_argument('--poll', choices=['NASA_CAD', 'NASA_SBDB', 'MPC', 'NEODyS'],
                       help='Run polling first with specified API')
    parser.add_argument('--period', default='1d', help='Time period for polling')
    parser.add_argument('--max-results', type=int, default=20, help='Maximum results')
    
    args = parser.parse_args()
    
    if args.file:
        # Load from specific file
        await launch_dashboard_from_file(args.file)
    elif args.poll:
        # Run polling first
        await launch_dashboard_from_polling(args.poll, args.period, args.max_results)
    else:
        # Interactive mode
        await interactive_dashboard_launcher()

if __name__ == "__main__":
    asyncio.run(main())