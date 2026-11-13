#!/usr/bin/env python3
"""
NEO Poller - Comprehensive NEO API Polling and Analysis System

This module provides menu-driven access to multiple NEO data sources
with configurable time periods for systematic artificial NEO detection.

Based on the original neos_o3high_v6.19.1.py approach but simplified
and focused on the core mission of finding artificial NEOs.
"""

from aneos_core.data.provider_http import session_get as provider_session_get
from aneos_version import APPLICATION_VERSION
import os
import sys
import json
import requests
import datetime
import time
from typing import Dict, List, Any, Optional, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Try to import dateutil, fall back to simplified approach
try:
    from dateutil.relativedelta import relativedelta
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    print("⚠️  python-dateutil not available, using simplified time period handling")
    
    # Simple fallback for relativedelta functionality
    class relativedelta:
        def __init__(self, days=0, weeks=0, months=0, years=0):
            self.days = days + (weeks * 7)
            self.months = months
            self.years = years
        
        def __rsub__(self, other):
            # Simple approximation for date subtraction
            if isinstance(other, datetime.date):
                # Approximate months as 30 days and years as 365 days
                total_days = self.days + (self.months * 30) + (self.years * 365)
                return other - datetime.timedelta(days=total_days)
            return other

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None


class NEOPoller:
    """
    Comprehensive NEO polling system with multiple API sources and time periods.
    
    Supports:
    - Multiple NEO APIs (NASA CAD, SBDB, NEODyS, MPC)
    - Flexible time periods (1m to 200y)
    - Artificial NEO detection
    - Batch processing with progress tracking
    """
    
    def __init__(self):
        self.console = console if HAS_RICH else None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'aNEOS/{APPLICATION_VERSION} (NEO research screening)'
        })
        
        # API endpoints
        self.apis = {
            'NASA_CAD': {
                'name': 'NASA Close Approach Data',
                'url': 'https://ssd-api.jpl.nasa.gov/cad.api',
                'description': 'Close approach data for all NEOs',
                'time_support': True,
                'batch_capable': True
            },
            'NASA_SBDB': {
                'name': 'NASA Small Body Database',
                'url': 'https://ssd-api.jpl.nasa.gov/sbdb.api',
                'description': 'Detailed orbital and physical data',
                'time_support': False,
                'batch_capable': False
            },
            'MPC': {
                'name': 'Minor Planet Center',
                'url': 'https://www.minorplanetcenter.net/iau/MPCORB.html',
                'description': 'Comprehensive orbital elements',
                'time_support': False,
                'batch_capable': True
            },
            'NEODyS': {
                'name': 'NEODyS Database',
                'url': 'https://newton.spacedys.com/neodys/',
                'description': 'NEO orbital dynamics',
                'time_support': False,
                'batch_capable': True
            }
        }
        
        # Time period definitions
        self.time_periods = {
            '1d': {'name': '1 Day', 'delta': relativedelta(days=1)},
            '1w': {'name': '1 Week', 'delta': relativedelta(weeks=1)},
            '1m': {'name': '1 Month', 'delta': relativedelta(months=1)},
            '3m': {'name': '3 Months', 'delta': relativedelta(months=3)},
            '6m': {'name': '6 Months', 'delta': relativedelta(months=6)},
            '1y': {'name': '1 Year', 'delta': relativedelta(years=1)},
            '2y': {'name': '2 Years', 'delta': relativedelta(years=2)},
            '5y': {'name': '5 Years', 'delta': relativedelta(years=5)},
            '10y': {'name': '10 Years', 'delta': relativedelta(years=10)},
            '25y': {'name': '25 Years', 'delta': relativedelta(years=25)},
            '50y': {'name': '50 Years', 'delta': relativedelta(years=50)},
            '100y': {'name': '100 Years', 'delta': relativedelta(years=100)},
            '200y': {'name': '200 Years', 'delta': relativedelta(years=200)},
            'max': {'name': 'Maximum (200 Years)', 'delta': relativedelta(years=200)}
        }
        
        # Artificial NEO detection thresholds (from original script)
        self.detection_thresholds = {
            'orbital_eccentricity': 0.95,
            'retrograde_inclination': 150,
            'velocity_consistency': 0.1,
            'approach_regularity': 0.8,
            'geographic_clustering': 0.7
        }
        
        print(f"aNEOS {APPLICATION_VERSION} — NEO poller")
        print("Bulk polling temporarily unavailable; shared per-object screening is available.")
    
    def parse_custom_time_period(self, input_str: str) -> Optional[relativedelta]:
        """
        Parse custom time period input (from original script).
        
        Args:
            input_str: Time period string (e.g., '1d', '6m', '2y', 'max')
            
        Returns:
            relativedelta object or None if invalid
        """
        input_str = input_str.strip().lower()
        
        if input_str in self.time_periods:
            return self.time_periods[input_str]['delta']
        
        # Parse custom format: number + unit
        pattern = r'^(\d+)([dwmy])$'
        match = re.match(pattern, input_str)
        if not match:
            return None
            
        try:
            value = int(match.group(1))
            unit = match.group(2)
            
            if unit == 'd':
                return relativedelta(days=value)
            elif unit == 'w':
                return relativedelta(weeks=value)
            elif unit == 'm':
                return relativedelta(months=value)
            elif unit == 'y':
                return relativedelta(years=value)
        except ValueError:
            pass
            
        return None
    
    def fetch_cad_data(self, start_date: str, end_date: str, limit: int = 5000) -> Optional[Dict[str, Any]]:
        """
        Fetch Close Approach Data from NASA API.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of records
            
        Returns:
            CAD data dictionary or None if failed
        """
        try:
            params = {
                'date-min': start_date,
                'date-max': end_date,
                'sort': 'date',
                'limit': limit
            }
            
            response = provider_session_get(self.session, self.apis['NASA_CAD']['url'], params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                return data
            else:
                print(f"⚠️  No NEO data found for period {start_date} to {end_date}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Error fetching CAD data: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON response: {e}")
            return None
    
    def fetch_sbdb_data(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed data from NASA Small Body Database.
        
        Args:
            designation: NEO designation
            
        Returns:
            SBDB data dictionary or None if failed
        """
        try:
            # Clean and format designation for SBDB API
            clean_designation = designation.strip()
            
            # For space-separated designations, try URL encoding
            import urllib.parse
            encoded_designation = urllib.parse.quote(clean_designation)
            
            params = {
                'des': clean_designation,  # Try clean first
                'phys-par': 1,  # Physical parameters
                'full-prec': 1  # Request full precision
            }
            
            response = provider_session_get(self.session, self.apis['NASA_SBDB']['url'], params=params, timeout=15)
            
            # If first attempt fails with 400, try encoded version
            if response.status_code == 400:
                params['des'] = encoded_designation
                response = provider_session_get(self.session, self.apis['NASA_SBDB']['url'], params=params, timeout=15)
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check if response contains actual data
            if 'object' not in data or not data['object']:
                print(f"⚠️ No SBDB data found for {designation}")
                return None
                
            return data
            
        except requests.RequestException as e:
            print(f"❌ Error fetching SBDB data for {designation}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON response for {designation}: {e}")
            return None
    
    def analyze_sbdb_data_for_artificial_signatures(self, sbdb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Controlled deprecation: root legacy candidate inference is not scientific evidence.

        These source-schema/missing-value/synthetic-score defects are tracked in#73.
        Use the typed per-object DataFetcher and canonical screening path. Native
        retrieval helpers remain available; this does not declare discovery polling fixed.
        """
        raise NotImplementedError(
            'Legacy candidate polling/inference is temporarily unavailable (#73). '
            'Use python aneos.py analyze DESIGNATION for typed per-object acquisition; '
            'no synthetic scores or missing-as-zero measurements are returned.'
        )
    
    def fetch_mpc_recent_neos(self, max_results: int) -> List[Dict[str, Any]]:
        """Controlled deprecation: root legacy candidate inference is not scientific evidence.

        These source-schema/missing-value/synthetic-score defects are tracked in#73.
        Use the typed per-object DataFetcher and canonical screening path. Native
        retrieval helpers remain available; this does not declare discovery polling fixed.
        """
        raise NotImplementedError(
            'Legacy candidate polling/inference is temporarily unavailable (#73). '
            'Use python aneos.py analyze DESIGNATION for typed per-object acquisition; '
            'no synthetic scores or missing-as-zero measurements are returned.'
        )
    
    def fetch_neodys_recent_neos(self, max_results: int) -> List[Dict[str, Any]]:
        """Controlled deprecation: root legacy candidate inference is not scientific evidence.

        These source-schema/missing-value/synthetic-score defects are tracked in#73.
        Use the typed per-object DataFetcher and canonical screening path. Native
        retrieval helpers remain available; this does not declare discovery polling fixed.
        """
        raise NotImplementedError(
            'Legacy candidate polling/inference is temporarily unavailable (#73). '
            'Use python aneos.py analyze DESIGNATION for typed per-object acquisition; '
            'no synthetic scores or missing-as-zero measurements are returned.'
        )
    
    def analyze_cad_record_for_artificial_signatures(self, record: List[Any]) -> Dict[str, Any]:
        """Controlled deprecation: root legacy candidate inference is not scientific evidence.

        These source-schema/missing-value/synthetic-score defects are tracked in#73.
        Use the typed per-object DataFetcher and canonical screening path. Native
        retrieval helpers remain available; this does not declare discovery polling fixed.
        """
        raise NotImplementedError(
            'Legacy candidate polling/inference is temporarily unavailable (#73). '
            'Use python aneos.py analyze DESIGNATION for typed per-object acquisition; '
            'no synthetic scores or missing-as-zero measurements are returned.'
        )
    
    def display_api_selection_menu(self) -> str:
        """Display API selection menu and return selected API."""
        if self.console:
            # Rich interface
            table = Table(title="🌐 Available NEO APIs", show_header=True, header_style="bold blue")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("API Name", style="green")
            table.add_column("Description", style="white")
            table.add_column("Time Support", style="yellow")
            
            for api_id, api_info in self.apis.items():
                time_support = "✅" if api_info['time_support'] else "❌"
                table.add_row(api_id, api_info['name'], api_info['description'], time_support)
            
            console.print(table)
            console.print("\n💡 [bold cyan]Recommendation:[/bold cyan] Use NASA_CAD for time-based polling")
            
            choice = Prompt.ask("\n🔍 Select API", choices=list(self.apis.keys()), default="NASA_CAD")
            
        else:
            # Basic interface
            print("\n🌐 Available NEO APIs:")
            print("=" * 60)
            for i, (api_id, api_info) in enumerate(self.apis.items(), 1):
                time_support = "✅" if api_info['time_support'] else "❌"
                print(f"{i}. {api_id}")
                print(f"   Name: {api_info['name']}")
                print(f"   Description: {api_info['description']}")
                print(f"   Time Support: {time_support}")
                print()
            
            print("💡 Recommendation: Use NASA_CAD for time-based polling")
            
            while True:
                try:
                    choice_num = int(input("\n🔍 Select API (1-4) [1]: ") or "1")
                    if 1 <= choice_num <= len(self.apis):
                        choice = list(self.apis.keys())[choice_num - 1]
                        break
                    else:
                        print("❌ Invalid choice. Please select 1-4.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
        
        return choice
    
    def display_time_period_menu(self) -> str:
        """Display time period selection menu and return selected period."""
        if self.console:
            # Rich interface
            table = Table(title="📅 Time Period Selection", show_header=True, header_style="bold blue")
            table.add_column("Code", style="cyan", no_wrap=True)
            table.add_column("Period", style="green")
            table.add_column("Description", style="white")
            
            # Common periods
            common_periods = ['1d', '1w', '1m', '3m', '6m', '1y', '2y', '5y']
            for period_code in common_periods:
                period_info = self.time_periods[period_code]
                table.add_row(period_code, period_info['name'], "Common period")
            
            # Extended periods
            table.add_row("10y", "10 Years", "Extended period")
            table.add_row("25y", "25 Years", "Long-term period")
            table.add_row("50y", "50 Years", "Very long-term")
            table.add_row("100y", "100 Years", "Century scale")
            table.add_row("200y", "200 Years", "Maximum period")
            table.add_row("max", "Maximum", "Same as 200y")
            
            console.print(table)
            console.print("\n💡 [bold cyan]Custom format:[/bold cyan] Use format like '15d', '18m', '3y'")
            console.print("💡 [bold cyan]Recommendation:[/bold cyan] Start with '1m' for recent activity")
            
            period = Prompt.ask("\n⏱️  Select time period", default="1m")
            
        else:
            # Basic interface
            print("\n📅 Time Period Selection:")
            print("=" * 40)
            print("Common periods:")
            common_periods = ['1d', '1w', '1m', '3m', '6m', '1y', '2y', '5y']
            for period_code in common_periods:
                period_info = self.time_periods[period_code]
                print(f"  {period_code:4} - {period_info['name']}")
            
            print("\nExtended periods:")
            print("  10y  - 10 Years")
            print("  25y  - 25 Years") 
            print("  50y  - 50 Years")
            print("  100y - 100 Years")
            print("  200y - 200 Years")
            print("  max  - Maximum (200 Years)")
            
            print("\n💡 Custom format: Use format like '15d', '18m', '3y'")
            print("💡 Recommendation: Start with '1m' for recent activity")
            
            period = input("\n⏱️  Select time period [1m]: ").strip() or "1m"
        
        return period
    
    def poll_and_analyze(self, api_choice: str, time_period: str, max_results: int = 1000) -> List[Dict[str, Any]]:
        """Controlled deprecation: root legacy candidate inference is not scientific evidence.

        These source-schema/missing-value/synthetic-score defects are tracked in#73.
        Use the typed per-object DataFetcher and canonical screening path. Native
        retrieval helpers remain available; this does not declare discovery polling fixed.
        """
        raise NotImplementedError(
            'Legacy candidate polling/inference is temporarily unavailable (#73). '
            'Use python aneos.py analyze DESIGNATION for typed per-object acquisition; '
            'no synthetic scores or missing-as-zero measurements are returned.'
        )
    
    async def launch_analysis_dashboard(self, results: List[Dict[str, Any]]):
        """Display supplied records; opening a dashboard does not export or poll."""
        dashboard = self.display_results(results)
        if self.console:
            from rich.prompt import Confirm
            if Confirm.ask("Browse source-bound screening records?", console=self.console):
                self._interactive_object_explorer(dashboard)
        return dashboard

    def _interactive_object_explorer(self, dashboard):
        """Historical entrypoint delegating to the shared screening browser."""
        from aneos_core.analysis.poller_presentation import explore
        return explore(dashboard, self.console)

    def _show_detailed_analysis(self, classification):
        """Show the existing evidence review; unavailable origin stays unavailable."""
        from aneos_core.analysis.dashboard_view import render_detail
        return render_detail(classification, self.console)

    def display_results(self, results: List[Dict[str, Any]]):
        """Read-only screening summary; no independent origin thresholds."""
        from aneos_core.analysis.poller_presentation import display
        return display(results, self.console)

    def save_results(self, results: List[Dict[str, Any]], api_choice: str, time_period: str):
        """Explicit exclusive JSON/Markdown export, preserving requested context."""
        from aneos_core.analysis.poller_presentation import export
        return export(results, self.console, {
            'requested_source': api_choice, 'requested_period': time_period,
            'semantics': 'Requested invocation context, not verified source/date coverage',
        })

    def run_interactive_menu(self):
        """Run the interactive NEO polling menu."""
        if self.console:
            console.print(Panel.fit(
                "🚀 [bold blue]NEO Poller - Artificial NEO Detection System[/bold blue] 🚀\n"
                "Poll multiple NEO APIs across different time periods\n"
                "Systematically search for potentially artificial objects",
                border_style="blue"
            ))
        else:
            print("\n" + "=" * 60)
            print("🚀 NEO Poller - Artificial NEO Detection System 🚀")
            print("Poll multiple NEO APIs across different time periods")
            print("Systematically search for potentially artificial objects")
            print("=" * 60)
        
        try:
            while True:
                # API Selection
                api_choice = self.display_api_selection_menu()
                
                # Time Period Selection
                time_period = self.display_time_period_menu()
                
                # Validate time period
                if not self.parse_custom_time_period(time_period):
                    print(f"❌ Invalid time period: {time_period}")
                    continue
                
                # Max results
                if self.console:
                    max_results = IntPrompt.ask("🔢 Maximum results to analyze", default=1000)
                else:
                    try:
                        max_results = int(input("\n🔢 Maximum results to analyze [1000]: ") or "1000")
                    except ValueError:
                        max_results = 1000
                
                # Confirm and execute
                if self.console:
                    if not Confirm.ask(f"\n🚀 Start polling {self.apis[api_choice]['name']} for period {time_period}?"):
                        continue
                else:
                    confirm = input(f"\n🚀 Start polling {self.apis[api_choice]['name']} for period {time_period}? (y/n) [y]: ").lower()
                    if confirm and confirm[0] != 'y':
                        continue
                
                # Execute polling and analysis
                print(f"\n🔄 Starting NEO polling and analysis...")
                start_time = time.time()
                
                results = self.poll_and_analyze(api_choice, time_period, max_results)
                
                end_time = time.time()
                print(f"⏱️  Analysis completed in {end_time - start_time:.1f} seconds")
                
                # Display results
                self.display_results(results)
                
                # Save results
                self.save_results(results, api_choice, time_period)
                
                # Continue?
                if self.console:
                    if not Confirm.ask("\n🔄 Run another analysis?"):
                        break
                else:
                    continue_choice = input("\n🔄 Run another analysis? (y/n) [n]: ").lower()
                    if not continue_choice or continue_choice[0] != 'y':
                        break
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


def main():
    """Main entry point for NEO Poller."""
    parser = argparse.ArgumentParser(
        description="NEO Poller - Poll multiple NEO APIs for artificial object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python neo_poller.py                          # Interactive menu
  python neo_poller.py --api NASA_CAD --period 1m  # Direct polling
  python neo_poller.py --list-apis              # List available APIs
  python neo_poller.py --list-periods           # List time periods
        """
    )
    
    parser.add_argument('--api', choices=['NASA_CAD', 'NASA_SBDB', 'MPC', 'NEODyS'], 
                       help='API to use for polling')
    parser.add_argument('--period', default='1m',
                       help='Time period (e.g., 1d, 1m, 1y, max)')
    parser.add_argument('--max-results', type=int, default=1000,
                       help='Maximum results to analyze')
    parser.add_argument('--list-apis', action='store_true',
                       help='List available APIs and exit')
    parser.add_argument('--list-periods', action='store_true',
                       help='List available time periods and exit')
    
    args = parser.parse_args()
    
    poller = NEOPoller()
    
    if args.list_apis:
        print("\n🌐 Available NEO APIs:")
        for api_id, api_info in poller.apis.items():
            time_support = "✅" if api_info['time_support'] else "❌"
            print(f"  {api_id:12} - {api_info['name']}")
            print(f"               {api_info['description']}")
            print(f"               Time Support: {time_support}")
            print()
        return
    
    if args.list_periods:
        print("\n📅 Available Time Periods:")
        for period_code, period_info in poller.time_periods.items():
            print(f"  {period_code:4} - {period_info['name']}")
        print("\n💡 Custom format: Use format like '15d', '18m', '3y'")
        return
    
    if args.api:
        # Direct execution mode
        print(f"🔄 Direct polling mode: {args.api} for period {args.period}")
        results = poller.poll_and_analyze(args.api, args.period, args.max_results)
        poller.display_results(results)
        if poller.save_results(results, args.api, args.period) is None:
            return 1
        
        # Launch dashboard if results are available
        if results:
            try:
                import asyncio
                asyncio.run(poller.launch_analysis_dashboard(results))
            except Exception as e:
                print(f"⚠️ Dashboard launch failed: {e}")
    else:
        # Interactive menu mode
        poller.run_interactive_menu()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotImplementedError as exc:
        import sys
        print(f"Unavailable: {exc}", file=sys.stderr)
        sys.exit(2)