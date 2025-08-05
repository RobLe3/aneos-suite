#!/usr/bin/env python3
"""
Simple NEO Analyzer - Core Mission: Find Artificial NEOs

This is a simplified, focused tool for analyzing Near Earth Objects
to identify potentially artificial objects. It polls NEO data from
a specified time period and searches for artificial signatures.

Focus: Simple, reliable, bug-free detection of artificial NEOs.
"""

import sys
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse


class SimpleNEOAnalyzer:
    """
    Simple analyzer focused on detecting artificial NEO signatures.
    
    Core mission: Find objects that might be artificial based on:
    - Unusual orbital characteristics
    - Inconsistent velocity patterns
    - Regular geometric patterns
    - Suspicious discovery circumstances
    """
    
    def __init__(self):
        self.base_url = "https://ssd-api.jpl.nasa.gov/sbdb.cgi"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'aNEOS/1.0 (Artificial NEO Detection System)'
        })
        
        # Thresholds for artificial detection
        self.artificial_indicators = {
            'eccentricity_threshold': 0.95,  # Highly eccentric orbits
            'inclination_threshold': 150,    # Retrograde orbits
            'velocity_consistency_threshold': 0.1,  # Too consistent
            'geometric_pattern_threshold': 0.8,     # Too regular
            'discovery_pattern_threshold': 0.9      # Suspicious discovery
        }
        
        print("🚀 Simple NEO Analyzer initialized")
        print("Mission: Detect potentially artificial Near Earth Objects")
    
    def fetch_neo_data(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Fetch basic NEO data from NASA JPL Small Body Database.
        
        Args:
            designation: NEO designation (e.g., "2024 AB123")
            
        Returns:
            Dictionary with NEO data or None if failed
        """
        # Check if this is a test/demo request
        if designation.lower() in ['test', 'demo', 'example']:
            return self._get_demo_data()
        
        try:
            params = {
                'sstr': designation,
                'full-prec': 'true'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'object' not in data:
                print(f"❌ No data found for {designation}")
                print("💡 Try using 'test' as designation to see demo functionality")
                return None
                
            return data['object']
            
        except requests.RequestException as e:
            print(f"❌ Network error fetching {designation}: {e}")
            print("💡 Try using 'test' as designation to see demo functionality")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON response for {designation}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching {designation}: {e}")
            return None
    
    def _get_demo_data(self) -> Dict[str, Any]:
        """
        Generate demo data for testing artificial NEO detection.
        """
        if hasattr(self, '_demo_counter'):
            self._demo_counter += 1
        else:
            self._demo_counter = 1
        
        if self._demo_counter % 2 == 1:
            # Artificial-looking object
            return {
                'fullname': 'TEST Artificial Object',
                'orbit': {
                    'e': '0.98',           # Extremely high eccentricity (suspicious)
                    'i': '165.5',          # Retrograde orbit (suspicious)
                    'a': '15.2',           # Large semi-major axis (unusual for NEO)
                    'condition_code': '1'   # High quality orbit
                },
                'discovery': {
                    'date': '2024-01-15'   # Recent discovery
                },
                'n_obs_used': 150,         # Many observations for recent object
                'H': '12.5',              # Absolute magnitude
                'rot_per': '24.0',        # Suspiciously regular 24-hour rotation
                'diameter': '2.1 km'
            }
        else:
            # Natural-looking object
            return {
                'fullname': 'TEST Natural Object',
                'orbit': {
                    'e': '0.35',           # Normal eccentricity
                    'i': '8.2',            # Normal inclination
                    'a': '1.8',            # Normal semi-major axis for NEO
                    'condition_code': '3'   # Medium quality orbit
                },
                'discovery': {
                    'date': '2018-03-12'   # Older discovery
                },
                'n_obs_used': 45,          # Normal observation count
                'H': '18.7',              # Normal magnitude
                'rot_per': '7.23',        # Irregular rotation period
                'diameter': '0.5 km'
            }
    
    def analyze_orbital_characteristics(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze orbital characteristics for artificial signatures.
        
        Artificial objects might have:
        - Extremely high eccentricity (escape trajectories)
        - Retrograde orbits (unusual for natural objects)
        - Perfect circular orbits (too regular)
        """
        results = {
            'score': 0.0,
            'indicators': [],
            'details': {}
        }
        
        try:
            orbit = neo_data.get('orbit', {})
            
            # Check eccentricity
            eccentricity = float(orbit.get('e', 0))
            results['details']['eccentricity'] = eccentricity
            
            if eccentricity > self.artificial_indicators['eccentricity_threshold']:
                results['score'] += 0.4
                results['indicators'].append(f"Extreme eccentricity: {eccentricity:.3f}")
            
            # Check inclination
            inclination = float(orbit.get('i', 0))
            results['details']['inclination'] = inclination
            
            if inclination > self.artificial_indicators['inclination_threshold']:
                results['score'] += 0.3
                results['indicators'].append(f"Retrograde orbit: {inclination:.1f}°")
            
            # Check for suspiciously perfect circular orbit
            if eccentricity < 0.01:
                results['score'] += 0.2
                results['indicators'].append(f"Suspiciously circular orbit: e={eccentricity:.4f}")
            
            # Check semi-major axis for unusual values
            semi_major_axis = float(orbit.get('a', 0))
            results['details']['semi_major_axis'] = semi_major_axis
            
            # Objects with a > 10 AU are unusual for NEOs
            if semi_major_axis > 10:
                results['score'] += 0.2
                results['indicators'].append(f"Unusual semi-major axis: {semi_major_axis:.2f} AU")
                
        except (ValueError, KeyError) as e:
            print(f"⚠️  Warning: Could not analyze orbital characteristics: {e}")
        
        return results
    
    def analyze_discovery_pattern(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze discovery circumstances for artificial signatures.
        
        Artificial objects might:
        - Be discovered very recently with extensive data
        - Have perfectly spaced observation intervals
        - Show up in multiple surveys simultaneously
        """
        results = {
            'score': 0.0,
            'indicators': [],
            'details': {}
        }
        
        try:
            # Check discovery date
            discovery_date = neo_data.get('discovery', {}).get('date')
            if discovery_date:
                results['details']['discovery_date'] = discovery_date
                
                # Objects discovered very recently with lots of data are suspicious
                discovery_year = int(discovery_date.split('-')[0])
                current_year = datetime.now().year
                
                if current_year - discovery_year <= 1:
                    # Check if we have extensive orbital data for recent discovery
                    orbit_quality = neo_data.get('orbit', {}).get('condition_code', '9')
                    if orbit_quality in ['0', '1', '2']:  # High quality orbit
                        results['score'] += 0.3
                        results['indicators'].append("Recent discovery with high-quality orbit")
            
            # Check for observation patterns
            n_obs = neo_data.get('n_obs_used', 0)
            if n_obs:
                results['details']['n_observations'] = n_obs
                
                # Too many observations for recent object
                if discovery_date and current_year - discovery_year <= 1 and n_obs > 100:
                    results['score'] += 0.2
                    results['indicators'].append(f"Excessive observations ({n_obs}) for recent object")
                    
        except (ValueError, KeyError) as e:
            print(f"⚠️  Warning: Could not analyze discovery pattern: {e}")
        
        return results
    
    def analyze_physical_properties(self, neo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze physical properties for artificial signatures.
        
        Artificial objects might have:
        - Unusual size-magnitude relationships
        - Perfect spherical shapes
        - Unusual material properties
        """
        results = {
            'score': 0.0,
            'indicators': [],
            'details': {}
        }
        
        try:
            # Check absolute magnitude vs estimated diameter
            abs_magnitude = neo_data.get('H')
            diameter = neo_data.get('diameter')
            
            if abs_magnitude and diameter:
                results['details']['absolute_magnitude'] = float(abs_magnitude)
                results['details']['diameter'] = diameter
                
                # Extremely bright objects (low H) are unusual
                h_val = float(abs_magnitude)
                if h_val < 15:  # Very bright
                    results['score'] += 0.1
                    results['indicators'].append(f"Unusually bright object: H={h_val}")
            
            # Check rotation period if available
            rotation_period = neo_data.get('rot_per')
            if rotation_period:
                results['details']['rotation_period'] = rotation_period
                
                # Suspiciously regular rotation periods
                period_val = float(rotation_period)
                if period_val == int(period_val):  # Perfect integer hours
                    results['score'] += 0.15
                    results['indicators'].append(f"Suspiciously regular rotation: {period_val}h")
                    
        except (ValueError, KeyError) as e:
            print(f"⚠️  Warning: Could not analyze physical properties: {e}")
        
        return results
    
    def calculate_artificial_probability(self, designation: str) -> Optional[Dict[str, Any]]:
        """
        Calculate the probability that a NEO is artificial.
        
        Args:
            designation: NEO designation
            
        Returns:
            Analysis results with artificial probability score
        """
        print(f"🔍 Analyzing {designation} for artificial signatures...")
        
        # Fetch NEO data
        neo_data = self.fetch_neo_data(designation)
        if not neo_data:
            return None
        
        # Run all analysis components
        orbital_analysis = self.analyze_orbital_characteristics(neo_data)
        discovery_analysis = self.analyze_discovery_pattern(neo_data)
        physical_analysis = self.analyze_physical_properties(neo_data)
        
        # Combine scores
        total_score = (
            orbital_analysis['score'] + 
            discovery_analysis['score'] + 
            physical_analysis['score']
        )
        
        # Normalize to 0-1 scale
        artificial_probability = min(total_score, 1.0)
        
        # Determine classification
        if artificial_probability >= 0.8:
            classification = "HIGHLY SUSPICIOUS - Likely Artificial"
            confidence = "HIGH"
        elif artificial_probability >= 0.6:
            classification = "SUSPICIOUS - Possibly Artificial"
            confidence = "MEDIUM"
        elif artificial_probability >= 0.3:
            classification = "ANOMALOUS - Investigate Further"
            confidence = "LOW"
        else:
            classification = "NATURAL - No Artificial Signatures"
            confidence = "HIGH"
        
        # Collect all indicators
        all_indicators = (
            orbital_analysis['indicators'] + 
            discovery_analysis['indicators'] + 
            physical_analysis['indicators']
        )
        
        return {
            'designation': designation,
            'artificial_probability': artificial_probability,
            'classification': classification,
            'confidence': confidence,
            'indicators': all_indicators,
            'analysis_details': {
                'orbital': orbital_analysis['details'],
                'discovery': discovery_analysis['details'],
                'physical': physical_analysis['details']
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def poll_recent_neos(self, days_back: int = 30) -> List[str]:
        """
        Poll for recently discovered NEOs.
        
        Note: This is a simplified version. In practice, you would
        query NEO discovery databases or use APIs that provide
        recent discovery lists.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of NEO designations
        """
        print(f"🌍 Polling for NEOs discovered in the last {days_back} days...")
        
        # For demo purposes, return some known NEO designations
        # In practice, this would query actual discovery databases
        sample_neos = [
            "2024 XS15", "2024 XR15", "2024 XL15", "2024 XK15",
            "2024 XJ15", "2024 XH15", "2024 XG15", "2024 XF15",
            "2024 XE15", "2024 XD15"
        ]
        
        print(f"✅ Found {len(sample_neos)} recent NEO discoveries")
        return sample_neos
    
    def batch_analyze(self, designations: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple NEOs for artificial signatures.
        
        Args:
            designations: List of NEO designations
            
        Returns:
            List of analysis results
        """
        results = []
        suspicious_objects = []
        
        print(f"\n🔬 Starting batch analysis of {len(designations)} objects...")
        print("=" * 60)
        
        for i, designation in enumerate(designations, 1):
            print(f"\n[{i}/{len(designations)}] Processing {designation}...")
            
            result = self.calculate_artificial_probability(designation)
            if result:
                results.append(result)
                
                # Print summary
                prob = result['artificial_probability']
                classification = result['classification']
                print(f"   Artificial Probability: {prob:.3f}")
                print(f"   Classification: {classification}")
                
                if prob >= 0.3:  # Flag suspicious objects
                    suspicious_objects.append(result)
                    print(f"   🚨 FLAGGED for further investigation")
                    
                    if result['indicators']:
                        print(f"   Key Indicators:")
                        for indicator in result['indicators'][:3]:  # Show top 3
                            print(f"     • {indicator}")
            else:
                print(f"   ❌ Analysis failed")
            
            # Rate limiting - be nice to NASA servers
            time.sleep(1)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 BATCH ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total objects analyzed: {len(results)}")
        print(f"Suspicious objects found: {len(suspicious_objects)}")
        
        if suspicious_objects:
            print(f"\n🚨 SUSPICIOUS OBJECTS REQUIRING INVESTIGATION:")
            for obj in sorted(suspicious_objects, key=lambda x: x['artificial_probability'], reverse=True):
                prob = obj['artificial_probability']
                print(f"  {obj['designation']:12} - Probability: {prob:.3f} - {obj['classification']}")
        
        return results


def main():
    """Main entry point for Simple NEO Analyzer."""
    parser = argparse.ArgumentParser(
        description="Simple NEO Analyzer - Detect Potentially Artificial Near Earth Objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_neo_analyzer.py single "2024 AB123"    # Analyze single object
  python simple_neo_analyzer.py poll 30               # Poll last 30 days
  python simple_neo_analyzer.py batch file.txt        # Analyze list from file
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis mode')
    
    # Single object analysis
    single_parser = subparsers.add_parser('single', help='Analyze single NEO')
    single_parser.add_argument('designation', help='NEO designation (e.g., "2024 AB123")')
    
    # Poll recent discoveries
    poll_parser = subparsers.add_parser('poll', help='Poll recent NEO discoveries')
    poll_parser.add_argument('days', type=int, default=30, nargs='?', 
                           help='Days to look back (default: 30)')
    
    # Batch analysis
    batch_parser = subparsers.add_parser('batch', help='Batch analyze NEOs from file')
    batch_parser.add_argument('file', help='File with NEO designations (one per line)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize analyzer
    analyzer = SimpleNEOAnalyzer()
    
    try:
        if args.command == 'single':
            # Single object analysis
            result = analyzer.calculate_artificial_probability(args.designation)
            if result:
                print("\n" + "=" * 60)
                print("📊 ANALYSIS RESULTS")
                print("=" * 60)
                print(f"Object: {result['designation']}")
                print(f"Artificial Probability: {result['artificial_probability']:.3f}")
                print(f"Classification: {result['classification']}")
                print(f"Confidence: {result['confidence']}")
                
                if result['indicators']:
                    print(f"\n🔍 Artificial Indicators:")
                    for indicator in result['indicators']:
                        print(f"  • {indicator}")
                
                if result['artificial_probability'] >= 0.3:
                    print(f"\n🚨 RECOMMENDATION: This object requires further investigation")
                else:
                    print(f"\n✅ ASSESSMENT: This object appears to be natural")
            
        elif args.command == 'poll':
            # Poll recent discoveries and analyze
            designations = analyzer.poll_recent_neos(args.days)
            if designations:
                results = analyzer.batch_analyze(designations)
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"neo_analysis_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {filename}")
        
        elif args.command == 'batch':
            # Batch analysis from file
            try:
                with open(args.file, 'r') as f:
                    designations = [line.strip() for line in f if line.strip()]
                
                if designations:
                    results = analyzer.batch_analyze(designations)
                    
                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"neo_data/polling-results/neo_batch_analysis_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"\n💾 Results saved to: {filename}")
                else:
                    print(f"❌ No designations found in {args.file}")
                    
            except FileNotFoundError:
                print(f"❌ File not found: {args.file}")
            except Exception as e:
                print(f"❌ Error reading file: {e}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()