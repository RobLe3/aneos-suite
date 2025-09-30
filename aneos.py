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
        
        # Enhanced analysis script with comprehensive details
        analysis_script = f"""
import sys
sys.path.insert(0, '.')
try:
    from simple_neo_analyzer import SimpleNEOAnalyzer
    import asyncio
    from aneos_core.analysis.pipeline import create_analysis_pipeline
    
    async def comprehensive_analysis():
        print("=" * 80)
        
        # Try comprehensive analysis first
        analyzer = SimpleNEOAnalyzer()
        
        try:
            print("🔍 Attempting comprehensive analysis...")
            result = analyzer.analyze_neo_comprehensive('{designation}')
            
            if 'error' in result:
                print(f"⚠️  Comprehensive analysis failed: {{result['error']}}")
                print("🔄 Falling back to pipeline analysis...")
                
                # Fallback to pipeline analysis
                pipeline = create_analysis_pipeline()
                pipeline_result = await pipeline.analyze_neo('{designation}')
                
                if pipeline_result:
                    print(f"\\n📊 Pipeline Analysis Results for {designation}:")
                    print(f"Overall Score: {{pipeline_result.anomaly_score.overall_score:.3f}}")
                    print(f"Classification: {{pipeline_result.anomaly_score.classification}}")
                    print(f"Confidence: {{pipeline_result.anomaly_score.confidence:.3f}}")
                    print(f"Processing Time: {{pipeline_result.processing_time:.2f}}s")
                    
                    if pipeline_result.anomaly_score.risk_factors:
                        print(f"\\n🚨 Risk Factors:")
                        for factor in pipeline_result.anomaly_score.risk_factors:
                            print(f"  • {{factor}}")
                            
                    if hasattr(pipeline_result, 'errors') and pipeline_result.errors:
                        print(f"\\n⚠️  Errors encountered:")
                        for error in pipeline_result.errors:
                            print(f"  • {{error}}")
                else:
                    print(f"❌ Both comprehensive and pipeline analysis failed for {designation}")
                    print("\\nℹ️  The NEO may not be in any available databases.")
                    print("💡 Try polling for recent discoveries or check the designation format.")
            else:
                # Display comprehensive results
                print(f"\\n🎯 COMPREHENSIVE ANALYSIS RESULTS")
                print(f"  Designation: {{result.get('designation', '{designation}')}}")
                # Raw artificial probability hidden per interim assessment - show only calibrated posterior
                # Use calibrated classification if available, fallback to original
                calibrated = result.get('calibrated_assessment', {{}})
                classification = calibrated.get('calibrated_classification') or result.get('classification', 'UNKNOWN')
                print(f"  Classification: {{classification}}")
                print(f"  Is Artificial: {{'YES' if result.get('is_artificial', False) else 'NO'}}")
                print(f"  Anomaly Confidence: {{result.get('confidence_level', 'unknown')}}")
                
                if result.get('sigma_statistical_level'):
                    print(f"  Sigma Level: {{result.get('sigma_statistical_level'):.2f}}σ")
                
                print(f"\\n🚨 RISK ASSESSMENT")
                # Risk level removed - only show single risk label driven by P(impact) per interim assessment
                print(f"  Threat Level: {{result.get('threat_level', 'unknown')}}")
                
                print(f"\\n📊 ANALYSIS QUALITY")
                print(f"  Method: {{result.get('analysis_method', 'unknown')}}")
                print(f"  Detector: {{result.get('detector_used', 'unknown')}}")
                print(f"  Data Completeness: {{result.get('data_completeness', 'unknown')}}")
                print(f"  Analysis Quality: {{result.get('analysis_quality', 'unknown')}}")
                print(f"  Validation Status: {{result.get('validation_status', 'unknown')}}")
                
                if result.get('processing_time_ms'):
                    print(f"  Processing Time: {{result.get('processing_time_ms'):.2f}} ms")
                
                risk_factors = result.get('risk_factors', [])
                if risk_factors:
                    print(f"\\n⚠️  RISK FACTORS")
                    for factor in risk_factors:
                        print(f"  • {{factor}}")
                
                anomaly_indicators = result.get('anomaly_indicators', [])
                if anomaly_indicators:
                    print(f"\\n🔍 ANOMALY INDICATORS")
                    for indicator in anomaly_indicators:
                        print(f"  • {{indicator}}")
                
                metadata = result.get('metadata', {{}})
                if metadata:
                    # Hide debug info in production
                    display_metadata = {{k: v for k, v in metadata.items() if k not in ['capabilities_used']}}
                    # Also hide false_positive_rate from detection_metadata
                    if 'detection_metadata' in display_metadata and isinstance(display_metadata['detection_metadata'], dict):
                        display_metadata['detection_metadata'] = {{
                            k: v for k, v in display_metadata['detection_metadata'].items() 
                            if k != 'false_positive_rate'
                        }}
                    print(f"\\n📋 ADDITIONAL METADATA")
                    for key, value in display_metadata.items():
                        if isinstance(value, dict):
                            print(f"  {{key}}:")
                            for sub_key, sub_value in value.items():
                                print(f"    {{sub_key}}: {{sub_value}}")
                        else:
                            print(f"  {{key}}: {{value}}")
                
                # Detailed Explanations
                explanations = result.get('explanations', {{}})
                if explanations:
                    
                    # Classification reasoning
                    reasoning = explanations.get('classification_reasoning', [])
                    if reasoning:
                        print(f"\\n💭 CLASSIFICATION REASONING")
                        for reason in reasoning:
                            print(f"  • {{reason}}")
                    
                    # Suspicious indicators with explanations
                    suspicious = explanations.get('suspicious_indicators', [])
                    if suspicious:
                        print(f"\\n🔍 WHY THIS IS CONSIDERED SUSPICIOUS")
                        for indicator in suspicious:
                            print(f"  • {{indicator}}")
                    
                    # Data quality notes
                    data_quality = explanations.get('data_quality_notes', [])
                    if data_quality:
                        print(f"\\n📊 DATA QUALITY NOTES")
                        for note in data_quality:
                            print(f"  • {{note}}")
                    
                    # Confidence factors
                    confidence_factors = explanations.get('confidence_factors', [])
                    if confidence_factors:
                        print(f"\\n🎯 CONFIDENCE FACTORS")
                        for factor in confidence_factors:
                            print(f"  • {{factor}}")
                
                # Calibrated Assessment (new section)
                calibrated = result.get('calibrated_assessment', {{}})
                if calibrated and not calibrated.get('error'):
                    print(f"\\n🎯 CALIBRATED ASSESSMENT (CORRECTED)")
                    print(f"  Statistical Significance: {{calibrated.get('statistical_significance', 0.0):.1f}}%")
                    print(f"  Sigma Level: {{calibrated.get('sigma_level', 0.0):.2f}}σ")
                    print(f"  Significance Meaning: {{calibrated.get('significance_interpretation', 'Unknown')}}")
                    
                    calibrated_prob = calibrated.get('calibrated_artificial_probability', 0.0)
                    prior = calibrated.get('prior_artificial_rate', 1e-5)
                    
                    # Calculate likelihood ratio from Bayesian components
                    if prior > 0 and calibrated_prob > 0:
                        # LR ≈ posterior/(1-posterior) * (1-prior)/prior  
                        odds_posterior = calibrated_prob / (1 - calibrated_prob)
                        odds_prior = prior / (1 - prior)
                        likelihood_ratio = odds_posterior / odds_prior if odds_prior > 0 else 0
                    else:
                        likelihood_ratio = 0
                    
                    if calibrated_prob < 0.001:  # Less than 0.1%
                        print(f"  Calibrated Artificial Probability: {{calibrated_prob:.4%}} (prior: {{prior:.0e}}, LR: {{likelihood_ratio:.1f}})")
                    else:
                        print(f"  Calibrated Artificial Probability: {{calibrated_prob:.1%}} (prior: {{prior:.0e}}, LR: {{likelihood_ratio:.1f}})")
                    print(f"  Calibrated Classification: {{calibrated.get('calibrated_classification', 'unknown')}}")
                    
                    print(f"\\n📚 METHODOLOGY NOTES")
                    print(f"  • Statistical significance ≠ Artificial probability")
                    print(f"  • Uses Bayesian inference with base rates")
                    print(f"  • Includes multiple testing correction")
                    prior_rate = calibrated.get('prior_artificial_rate', 0.001)
                    if prior_rate <= 1e-4:  # Very small prior
                        print(f"  • Prior artificial rate: {{prior_rate:.2e}} ({{prior_rate*100:.4f}}%)")
                    else:
                        print(f"  • Prior artificial rate: {{prior_rate:.1%}}")
                    print(f"  • Testing {{calibrated.get('multiple_testing_factor', 1):,}} NEOs")
                
                # Impact Probability Assessment (NEW FEATURE)
                impact = result.get('impact_assessment', {{}})
                if impact and impact.get('status') == 'calculated':
                    print(f"\\n💥 EARTH IMPACT PROBABILITY ASSESSMENT")
                    print(f"  Collision Probability: {{impact.get('collision_probability', 0.0):.2e}}")
                    print(f"  Risk Level: {{impact.get('risk_level', 'unknown').upper()}}")
                    print(f"  Comparative Risk: {{impact.get('comparative_risk', 'Unknown')}}")
                    
                    # Only show impact time for meaningful probabilities  
                    impact_time = impact.get('time_to_impact_years')
                    collision_prob = impact.get('collision_probability', 0.0)
                    if impact_time and collision_prob > 1e-10:
                        print(f"  Most Probable Impact Time: {{impact_time:.0f}} years")
                    elif collision_prob <= 1e-10:
                        print(f"  Most Probable Impact Time: N/A (probability too low)")
                    
                    print(f"  Calculation Confidence: {{impact.get('calculation_confidence', 0.0):.1%}}")
                    print(f"  Methodology: {{impact.get('methodology', 'unknown')}}")
                    
                    # Physical impact effects
                    if impact.get('impact_energy_mt'):
                        print(f"\\n💥 POTENTIAL IMPACT EFFECTS")
                        # Add hypothetical note for negligible risk
                        risk_level = impact.get('risk_level', '').lower()
                        if risk_level == 'negligible':
                            print(f"  Note: Effects are hypothetical, not a forecast")
                        print(f"  Impact Energy: {{impact.get('impact_energy_mt'):.1f}} Megatons TNT")
                        print(f"  Impact Velocity: {{impact.get('impact_velocity_km_s', 0):.1f}} km/s")
                        
                        if impact.get('crater_diameter_km'):
                            print(f"  Crater Diameter: {{impact.get('crater_diameter_km'):.1f}} km")
                        if impact.get('damage_radius_km'):
                            print(f"  Damage Radius: {{impact.get('damage_radius_km'):.0f}} km")
                        
                        if impact.get('most_probable_impact_region'):
                            print(f"  Most Probable Region: {{impact.get('most_probable_impact_region')}}")
                    
                    # Risk factors and rationale
                    risk_factors = impact.get('primary_risk_factors', [])
                    if risk_factors:
                        print(f"\\n🎯 IMPACT RISK FACTORS")
                        for factor in risk_factors[:5]:  # Show top 5
                            print(f"  • {{factor}}")
                    
                    rationale = impact.get('rationale', [])
                    if rationale:
                        print(f"\\n🧠 SCIENTIFIC RATIONALE")
                        for reason in rationale[:5]:  # Show top 5 rationales
                            print(f"  • {{reason}}")
                    
                    # Special considerations - only show keyholes with concrete numbers
                    keyholes = impact.get('keyhole_passages', 0)
                    if keyholes > 0:
                        # Only show keyhole if we have concrete parameters
                        keyhole_data = impact.get('keyhole_details', {{}})
                        if keyhole_data.get('epoch') and keyhole_data.get('ca_distance_km') and keyhole_data.get('corridor_width_km'):
                            print(f"\\n🌀 GRAVITATIONAL KEYHOLES")
                            print(f"  Detected Passages: {{keyholes}}")
                            print(f"  Epoch: {{keyhole_data['epoch']}}")
                            print(f"  CA Distance: {{keyhole_data['ca_distance_km']:.0f}} km")
                            print(f"  Corridor Width: {{keyhole_data['corridor_width_km']:.1f}} km")
                        # Otherwise omit keyhole block entirely
                    
                    if impact.get('artificial_considerations'):
                        print(f"\\n🛰️  ARTIFICIAL OBJECT CONSIDERATIONS")
                        print(f"  Propulsive uncertainty affects trajectory prediction")
                        print(f"  Mission status and control capabilities unknown")
                    
                    # Moon Impact Assessment (NEW FEATURE)
                    # Per interim assessment: Only show if actual calculation ran with authority alignment
                    moon_status = impact.get('moon_status', 'calculated')
                    if moon_status == 'not_modeled':
                        print(f"\\n🌙 MOON IMPACT ASSESSMENT")  
                        print(f"  Moon probability: Not modeled (authority alignment required)")
                        print(f"  Will be populated when Moon module runs with authoritative ephemerides")
                    elif impact.get('moon_collision_probability') is not None:
                        print(f"\\n🌙 MOON IMPACT ASSESSMENT")
                        moon_prob = impact.get('moon_collision_probability', 0.0)
                        print(f"  Moon Collision Probability: {{moon_prob:.2e}}")
                        
                        earth_moon_ratio = impact.get('earth_vs_moon_impact_ratio', 1.0)
                        if earth_moon_ratio < 1.0:
                            print(f"  Moon impact {{1/earth_moon_ratio:.1f}}x MORE likely than Earth impact")
                        elif earth_moon_ratio > 1.0:
                            print(f"  Earth impact {{earth_moon_ratio:.1f}}x more likely than Moon impact")
                        else:
                            print(f"  Earth and Moon impact probabilities similar")
                        
                        if impact.get('moon_impact_energy_mt'):
                            print(f"  Moon Impact Energy: {{impact.get('moon_impact_energy_mt'):.1f}} Megatons TNT")
                        
                        # Moon impact effects
                        moon_effects = impact.get('moon_impact_effects', {{}})
                        if moon_effects:
                            if moon_effects.get('visible_from_earth'):
                                print(f"  Impact would be VISIBLE from Earth")
                            if moon_effects.get('crater_diameter_km'):
                                print(f"  Would create {{moon_effects.get('crater_diameter_km'):.1f}} km crater on Moon")
                            if moon_effects.get('lunar_mission_impact') != 'unknown':
                                print(f"  Lunar mission impact: {{moon_effects.get('lunar_mission_impact').upper()}}")
                    
                    # Calculation details
                    uncertainty = impact.get('probability_uncertainty_range', (0, 0))
                    if uncertainty and uncertainty != (0, 0):
                        print(f"\\n📊 UNCERTAINTY ANALYSIS")
                        print(f"  Probability Range: {{uncertainty[0]:.2e}} to {{uncertainty[1]:.2e}}")
                        print(f"  Confidence Interval: {{impact.get('calculation_confidence', 0):.0%}}")
                
                elif impact and impact.get('status') == 'calculator_unavailable':
                    print(f"\\n💥 EARTH IMPACT PROBABILITY ASSESSMENT")
                    print(f"  Status: Impact calculator not available")
                    print(f"  Note: Install impact probability module for full assessment")
                
                elif impact and impact.get('status') == 'no_orbital_data':
                    print(f"\\n💥 EARTH IMPACT PROBABILITY ASSESSMENT")
                    print(f"  Status: Insufficient orbital data")
                    print(f"  Note: Impact assessment requires orbital elements")
                    
                elif impact and impact.get('status') == 'calculation_failed':
                    print(f"\\n💥 EARTH IMPACT PROBABILITY ASSESSMENT")
                    print(f"  Status: Calculation failed")
                    if impact.get('error'):
                        print(f"  Error: {{impact.get('error')}}")
                
                else:
                    # No impact assessment available
                    print(f"\\n💥 EARTH IMPACT PROBABILITY ASSESSMENT")
                    print(f"  Status: Assessment not performed")
                    print(f"  Note: Impact assessment available for Earth-crossing asteroids")
                
                timestamp = result.get('analysis_timestamp')
                if timestamp:
                    print(f"\\n🕐 Analysis completed at: {{timestamp}}")
                    
        except Exception as e:
            print(f"❌ Analysis error: {{e}}")
            print(f"\\nℹ️  This could indicate:")
            print(f"  • NEO not found in available databases")
            print(f"  • Invalid designation format")
            print(f"  • Data source connectivity issues")
            print(f"\\n💡 Suggestions:")
            print(f"  • Verify the designation format (e.g., '2024 AB123')")
            print(f"  • Check if this is a recent discovery")
            print(f"  • Try using the interactive menu for more options")
        
        print("=" * 80)
            
    asyncio.run(comprehensive_analysis())
    
except ImportError as e:
    print(f"❌ Error: Missing dependencies - {{e}}")
    print("Please run: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Unexpected error: {{e}}")
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
import os
from pathlib import Path
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
    dirs = ['data', 'logs', 'models', 'cache', 'neo_data', 'exports']
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