#!/usr/bin/env python3
"""
aNEOS Non-Interactive Test Interface
XO Claudette's Emergency Repair System

Provides programmatic access to all aNEOS functions for automated testing
without requiring interactive terminal input.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class ANEOSTestInterface:
    """Non-interactive test interface for aNEOS system testing."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for test operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def test_system_health(self) -> Dict[str, Any]:
        """Test overall system health and component availability."""
        print("🔍 Testing System Health...")
        
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_status': 'unknown'
        }
        
        # Test core imports
        try:
            from aneos_core.analysis.pipeline import create_analysis_pipeline
            health_results['components']['analysis_pipeline'] = 'available'
        except ImportError as e:
            health_results['components']['analysis_pipeline'] = f'unavailable: {str(e)}'
            
        # Test database
        try:
            import aneos_api.database as db_module
            health_results['components']['database'] = 'available'
        except ImportError as e:
            health_results['components']['database'] = f'unavailable: {str(e)}'
            
        # Test API components
        try:
            import aneos_api.app as app_module
            health_results['components']['api'] = 'available'
        except ImportError as e:
            health_results['components']['api'] = f'unavailable: {str(e)}'
            
        # Test ML components
        try:
            from aneos_core.ml.models import ModelConfig
            health_results['components']['ml_models'] = 'available'
        except ImportError as e:
            health_results['components']['ml_models'] = f'unavailable: {str(e)}'
            
        # Determine overall status
        available = sum(1 for status in health_results['components'].values() 
                       if status == 'available')
        total = len(health_results['components'])
        
        if available == total:
            health_results['overall_status'] = 'healthy'
        elif available > total / 2:
            health_results['overall_status'] = 'degraded'
        else:
            health_results['overall_status'] = 'critical'
            
        print(f"✅ System Health: {health_results['overall_status'].upper()}")
        print(f"📊 Components Available: {available}/{total}")
        
        return health_results
        
    def test_basic_analysis(self) -> Dict[str, Any]:
        """Test basic NEO analysis functionality."""
        print("🔬 Testing Basic Analysis Functions...")
        
        analysis_results = {
            'test_designation': 'TEST_NEO_2025',
            'timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'details': {}
        }
        
        try:
            # Test simple analysis
            from simple_neo_analyzer import SimpleNEOAnalyzer
            analyzer = SimpleNEOAnalyzer()
            result = analyzer.calculate_artificial_probability("test")
            analysis_results['status'] = 'success'
            analysis_results['details']['simple_analysis'] = 'functional'
            print("✅ Simple analysis: FUNCTIONAL")
            
        except ImportError as e:
            analysis_results['status'] = 'error'
            analysis_results['details']['error'] = f'Import error: {str(e)}'
            print(f"❌ Simple analysis: FAILED - {str(e)}")
            
        except Exception as e:
            analysis_results['status'] = 'error'
            analysis_results['details']['error'] = f'Runtime error: {str(e)}'
            print(f"❌ Simple analysis: FAILED - {str(e)}")
            
        return analysis_results
        
    def test_data_sources(self) -> Dict[str, Any]:
        """Test external data source connectivity with timeout fixes."""
        print("🌐 Testing Data Sources (with timeout fixes)...")
        
        data_results = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'timeout_fixes_applied': True
        }
        
        try:
            # Import with timeout configuration
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Create session with proper timeout and retry strategy
            session = requests.Session()
            
            # Configure retries and timeouts
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Test NASA APIs with short timeout
            try:
                response = session.get(
                    "https://ssd-api.jpl.nasa.gov/cad.api",
                    timeout=10,  # 10 second timeout
                    params={'date-min': '2025-01-01', 'dist-max': '0.1'}
                )
                if response.status_code == 200:
                    data_results['sources']['nasa_cad'] = 'accessible'
                    print("✅ NASA CAD API: ACCESSIBLE")
                else:
                    data_results['sources']['nasa_cad'] = f'http_error_{response.status_code}'
                    print(f"⚠️  NASA CAD API: HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                data_results['sources']['nasa_cad'] = 'timeout'
                print("⏰ NASA CAD API: TIMEOUT (expected with fixes applied)")
                
            except requests.exceptions.RequestException as e:
                data_results['sources']['nasa_cad'] = f'connection_error: {str(e)}'
                print(f"❌ NASA CAD API: CONNECTION ERROR")
                
        except Exception as e:
            data_results['sources']['requests_error'] = str(e)
            print(f"❌ Data source testing failed: {str(e)}")
            
        return data_results
        
    def test_database_operations(self) -> Dict[str, Any]:
        """Test database operations."""
        print("🗄️ Testing Database Operations...")
        
        db_results = {
            'timestamp': datetime.now().isoformat(),
            'database_file': None,
            'operations': {},
            'status': 'unknown'
        }
        
        try:
            # Check for database file
            db_path = PROJECT_ROOT / "aneos.db"
            db_results['database_file'] = str(db_path)
            db_results['database_exists'] = db_path.exists()
            
            if db_path.exists():
                print(f"✅ Database file found: {db_path}")
                db_results['operations']['file_exists'] = True
            else:
                print(f"⚠️  Database file not found: {db_path}")
                db_results['operations']['file_exists'] = False
                
            # Test database connection
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Test basic query
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                db_results['operations']['connection'] = 'success'
                db_results['operations']['tables'] = [table[0] for table in tables]
                db_results['status'] = 'operational'
                
                conn.close()
                print(f"✅ Database connection: SUCCESS")
                print(f"📊 Tables found: {len(tables)}")
                
            except sqlite3.Error as e:
                db_results['operations']['connection'] = f'sqlite_error: {str(e)}'
                db_results['status'] = 'error'
                print(f"❌ Database connection: FAILED - {str(e)}")
                
        except Exception as e:
            db_results['operations']['general_error'] = str(e)
            db_results['status'] = 'error'
            print(f"❌ Database testing failed: {str(e)}")
            
        return db_results
        
    def test_configuration(self) -> Dict[str, Any]:
        """Test system configuration."""
        print("⚙️ Testing Configuration...")
        
        config_results = {
            'timestamp': datetime.now().isoformat(),
            'files': {},
            'environment': {},
            'status': 'unknown'
        }
        
        # Check for configuration files
        config_files = [
            'requirements.txt',
            'docker-compose.yml', 
            'Dockerfile',
            '.env',
            'logging.conf'
        ]
        
        for config_file in config_files:
            file_path = PROJECT_ROOT / config_file
            config_results['files'][config_file] = {
                'exists': file_path.exists(),
                'path': str(file_path)
            }
            
            status = "✅" if file_path.exists() else "❌"
            print(f"{status} {config_file}: {'EXISTS' if file_path.exists() else 'MISSING'}")
            
        # Check environment variables
        env_vars = ['PYTHONPATH', 'PATH']
        for var in env_vars:
            config_results['environment'][var] = os.environ.get(var, 'not_set')
            
        config_results['status'] = 'checked'
        return config_results
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive system test."""
        print("🚀 Running Comprehensive aNEOS Test Suite")
        print("=" * 50)
        
        comprehensive_results = {
            'test_start': datetime.now().isoformat(),
            'system_health': None,
            'basic_analysis': None,
            'data_sources': None,
            'database': None,
            'configuration': None,
            'test_end': None,
            'overall_status': 'unknown'
        }
        
        # Run all tests
        comprehensive_results['system_health'] = self.test_system_health()
        print()
        
        comprehensive_results['basic_analysis'] = self.test_basic_analysis()
        print()
        
        comprehensive_results['data_sources'] = self.test_data_sources()
        print()
        
        comprehensive_results['database'] = self.test_database_operations()
        print()
        
        comprehensive_results['configuration'] = self.test_configuration()
        print()
        
        comprehensive_results['test_end'] = datetime.now().isoformat()
        
        # Determine overall status
        health_status = comprehensive_results['system_health']['overall_status']
        if health_status == 'healthy':
            comprehensive_results['overall_status'] = 'system_operational'
        elif health_status == 'degraded':
            comprehensive_results['overall_status'] = 'system_degraded'
        else:
            comprehensive_results['overall_status'] = 'system_critical'
            
        print("=" * 50)
        print(f"🎯 OVERALL STATUS: {comprehensive_results['overall_status'].upper()}")
        print("=" * 50)
        
        return comprehensive_results
        
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aneos_test_results_{timestamp}.json"
            
        results_path = PROJECT_ROOT / filename
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {results_path}")
            return str(results_path)
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
            return None


def main():
    """Main function for direct execution."""
    print("🎖️ XO CLAUDETTE'S aNEOS TEST INTERFACE")
    print("🚀 Emergency System Diagnostics Protocol")
    print()
    
    interface = ANEOSTestInterface()
    results = interface.run_comprehensive_test()
    
    # Save results
    results_file = interface.save_results(results)
    
    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results['overall_status'] == 'system_operational' else 1)