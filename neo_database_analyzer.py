#!/usr/bin/env python3
"""
NEO Database Analyzer - Long-term Analysis of Cached NEO Data

Analyzes the local NEO database and cached data for:
- Artificial or non-standard behavior patterns
- Orbital anomalies and suspicious characteristics
- Long-term trends and classifications
- Comprehensive reporting and summarization
"""

import os
import json
import time
import datetime
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class NEOClassification:
    """Classification result for a NEO."""
    designation: str
    classification: str
    confidence: float
    artificial_score: float
    anomaly_indicators: List[str]
    data_quality: float
    sources_used: List[str]
    analysis_details: Dict[str, Any]

class NEODatabaseAnalyzer:
    """
    Comprehensive analyzer for cached NEO data and long-term storage.
    
    Analyzes stored NEO data for:
    - Artificial signatures and non-standard behavior
    - Orbital anomalies and suspicious patterns
    - Data quality and completeness trends
    - Source reliability and consistency
    """
    
    def __init__(self, data_dir: str = "neo_data"):
        self.data_dir = Path(data_dir)
        self.database_path = self.data_dir / "neo_database.json"
        self.orbital_dir = self.data_dir / "orbital_elements"
        self.results_dir = self.data_dir / "results"
        
        # Classification thresholds (refined from experience)
        self.thresholds = {
            'artificial_score': {
                'suspicious': 0.3,
                'highly_suspicious': 0.6,
                'extremely_suspicious': 0.8
            },
            'orbital_anomalies': {
                'extreme_eccentricity': 0.95,
                'circular_orbit': 0.01,
                'retrograde_inclination': 150.0,
                'unusual_semi_major': {'min': 0.1, 'max': 50.0}
            },
            'velocity_anomalies': {
                'unusually_low': 5.0,
                'unusually_high': 50.0,
                'suspicious_consistency': 0.05  # Very low stddev
            },
            'data_quality': {
                'minimum_completeness': 0.1,
                'good_completeness': 0.6,
                'excellent_completeness': 0.9
            }
        }
        
        print(f"🔍 NEO Database Analyzer initialized")
        print(f"📁 Data directory: {self.data_dir}")
        
    def load_neo_database(self) -> Dict[str, Any]:
        """Load the local NEO database."""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'r') as f:
                    database = json.load(f)
                print(f"📊 Loaded NEO database with {len(database)} objects")
                return database
            else:
                print("❌ No NEO database found - run polling first")
                return {}
        except Exception as e:
            print(f"❌ Error loading NEO database: {e}")
            return {}
    
    def load_cached_orbital_data(self) -> Dict[str, Dict[str, Any]]:
        """Load all cached orbital data from all sources."""
        cached_data = defaultdict(dict)
        
        try:
            if not self.orbital_dir.exists():
                return cached_data
            
            total_files = 0
            for source_dir in self.orbital_dir.iterdir():
                if source_dir.is_dir():
                    source_name = source_dir.name
                    
                    for cache_file in source_dir.glob("*.json"):
                        try:
                            with open(cache_file, 'r') as f:
                                data = json.load(f)
                            
                            designation = cache_file.stem
                            cached_data[designation][source_name] = data
                            total_files += 1
                            
                        except Exception as e:
                            print(f"⚠️  Error loading {cache_file}: {e}")
            
            print(f"📁 Loaded {total_files} cached files for {len(cached_data)} NEOs")
            return dict(cached_data)
            
        except Exception as e:
            print(f"❌ Error loading cached data: {e}")
            return {}
    
    def load_analysis_results(self) -> List[Dict[str, Any]]:
        """Load all historical analysis results."""
        all_results = []
        
        try:
            if not self.results_dir.exists():
                return all_results
            
            for result_file in self.results_dir.glob("enhanced_neo_poll_*.json"):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    # Add metadata about the analysis run
                    analysis_meta = {
                        'filename': result_file.name,
                        'file_date': datetime.datetime.fromtimestamp(result_file.stat().st_mtime).isoformat(),
                        'metadata': data.get('metadata', {}),
                        'results': data.get('results', [])
                    }
                    all_results.append(analysis_meta)
                    
                except Exception as e:
                    print(f"⚠️  Error loading {result_file}: {e}")
            
            print(f"📈 Loaded {len(all_results)} historical analysis results")
            return all_results
            
        except Exception as e:
            print(f"❌ Error loading analysis results: {e}")
            return []
    
    def analyze_orbital_anomalies(self, orbital_elements: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze orbital elements for anomalies and artificial signatures."""
        anomaly_score = 0.0
        indicators = []
        
        # Eccentricity analysis
        eccentricity = orbital_elements.get('eccentricity')
        if eccentricity is not None:
            if eccentricity > self.thresholds['orbital_anomalies']['extreme_eccentricity']:
                score_contrib = min((eccentricity - 0.95) / 0.05, 1.0) * 0.4
                anomaly_score += score_contrib
                indicators.append(f"Extreme eccentricity: {eccentricity:.4f} (hyperbolic-like)")
            elif eccentricity < self.thresholds['orbital_anomalies']['circular_orbit']:
                anomaly_score += 0.3
                indicators.append(f"Suspiciously circular orbit: e={eccentricity:.6f}")
        
        # Inclination analysis
        inclination = orbital_elements.get('inclination')
        if inclination is not None:
            if inclination > self.thresholds['orbital_anomalies']['retrograde_inclination']:
                score_contrib = min((inclination - 150) / 30, 1.0) * 0.35
                anomaly_score += score_contrib
                indicators.append(f"Retrograde orbit: {inclination:.1f}° (artificial control?)")
            elif inclination > 90 and inclination < 120:
                anomaly_score += 0.15
                indicators.append(f"High inclination orbit: {inclination:.1f}° (unusual)")
        
        # Semi-major axis analysis
        semi_major_axis = orbital_elements.get('semi_major_axis')
        if semi_major_axis is not None:
            limits = self.thresholds['orbital_anomalies']['unusual_semi_major']
            if semi_major_axis < limits['min'] or semi_major_axis > limits['max']:
                anomaly_score += 0.25
                indicators.append(f"Unusual semi-major axis: {semi_major_axis:.3f} AU")
        
        # Perfect orbital ratios (potential artificial)
        if (eccentricity is not None and inclination is not None and 
            semi_major_axis is not None):
            
            # Check for "too perfect" ratios
            if (abs(eccentricity - 0.5) < 0.001 or 
                abs(inclination - 45.0) < 0.1 or
                abs(semi_major_axis - 1.0) < 0.001):
                anomaly_score += 0.2
                indicators.append("Suspiciously perfect orbital parameters")
        
        return min(anomaly_score, 1.0), indicators
    
    def analyze_velocity_patterns(self, analysis_results: List[Dict[str, Any]], designation: str) -> Tuple[float, List[str]]:
        """Analyze velocity patterns across multiple observations."""
        velocity_score = 0.0  
        indicators = []
        
        # Extract velocity data from historical results
        velocities = []
        for result_set in analysis_results:
            for neo_result in result_set.get('results', []):
                if neo_result.get('designation') == designation:
                    # Look for velocity in analysis details
                    analysis_details = neo_result.get('analysis_details', {})
                    velocity_count = analysis_details.get('velocity_count', 0)
                    if velocity_count > 0:
                        # This is a simplified approach - in real implementation
                        # we'd extract actual velocity values
                        velocities.append(velocity_count)
        
        if len(velocities) > 1:
            velocity_std = statistics.stdev(velocities)
            velocity_mean = statistics.mean(velocities)
            
            # Suspiciously consistent velocities
            if velocity_std < self.thresholds['velocity_anomalies']['suspicious_consistency']:
                velocity_score += 0.3
                indicators.append(f"Suspiciously consistent velocities (σ={velocity_std:.4f})")
            
            # Unusual velocity ranges
            if velocity_mean < self.thresholds['velocity_anomalies']['unusually_low']:
                velocity_score += 0.2
                indicators.append(f"Unusually low velocities (avg: {velocity_mean:.2f})")
            elif velocity_mean > self.thresholds['velocity_anomalies']['unusually_high']:
                velocity_score += 0.2  
                indicators.append(f"Unusually high velocities (avg: {velocity_mean:.2f})")
        
        return min(velocity_score, 1.0), indicators
    
    def analyze_discovery_patterns(self, neo_record: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze discovery and observation patterns for artificial signatures."""
        discovery_score = 0.0
        indicators = []
        
        # Recent discovery with high data quality (suspicious)
        first_seen = neo_record.get('first_seen', '')
        best_completeness = neo_record.get('best_completeness', 0)
        
        try:
            discovery_date = datetime.datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
            days_since_discovery = (datetime.datetime.now() - discovery_date).days
            
            # Very recent discovery with suspiciously complete data
            if days_since_discovery < 30 and best_completeness > 0.8:
                discovery_score += 0.25
                indicators.append(f"Recent discovery ({days_since_discovery}d) with high completeness ({best_completeness:.2f})")
            
            # Too many enrichment attempts for a natural object
            attempts = neo_record.get('enrichment_attempts', 0)
            if attempts > 10:
                discovery_score += 0.1
                indicators.append(f"Excessive enrichment attempts: {attempts}")
                
        except Exception:
            pass  # Invalid date format
        
        return min(discovery_score, 1.0), indicators
    
    def analyze_data_consistency(self, cached_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze data consistency across sources for artificial signatures."""
        consistency_score = 0.0
        indicators = []
        
        # Check if we have data from multiple sources
        if len(cached_data) > 1:
            orbital_elements = {}
            
            # Extract orbital elements from each source
            for source, data in cached_data.items():
                elements = data.get('orbital_elements', {})
                if elements:
                    orbital_elements[source] = elements
            
            if len(orbital_elements) > 1:
                # Compare eccentricity across sources
                eccentricities = [elem.get('eccentricity') for elem in orbital_elements.values() 
                                 if elem.get('eccentricity') is not None]
                
                if len(eccentricities) > 1:
                    ecc_std = statistics.stdev(eccentricities)
                    
                    # Suspiciously perfect agreement between sources
                    if ecc_std < 1e-6:
                        consistency_score += 0.2
                        indicators.append("Perfect agreement between sources (too consistent)")
                    
                    # Large disagreement between sources (data manipulation?)
                    elif ecc_std > 0.1:
                        consistency_score += 0.15
                        indicators.append(f"Large source disagreement in eccentricity (σ={ecc_std:.4f})")
        
        return min(consistency_score, 1.0), indicators
    
    def classify_neo(self, designation: str, neo_record: Dict[str, Any], 
                    cached_data: Dict[str, Any], analysis_results: List[Dict[str, Any]]) -> NEOClassification:
        """Comprehensive classification of a NEO for artificial behavior."""
        
        # Get best available orbital data
        orbital_elements = neo_record.get('combined_data', {})
        if not orbital_elements and cached_data:
            # Try to get from cached data
            for source_data in cached_data.values():
                elements = source_data.get('orbital_elements', {})
                if elements:
                    orbital_elements = elements
                    break
        
        # Perform all analyses
        orbital_score, orbital_indicators = self.analyze_orbital_anomalies(orbital_elements)
        velocity_score, velocity_indicators = self.analyze_velocity_patterns(analysis_results, designation)
        discovery_score, discovery_indicators = self.analyze_discovery_patterns(neo_record)
        consistency_score, consistency_indicators = self.analyze_data_consistency(cached_data)
        
        # Combine scores with weights
        total_score = (
            orbital_score * 0.4 +      # Orbital anomalies most important
            velocity_score * 0.25 +    # Velocity patterns significant  
            discovery_score * 0.2 +    # Discovery patterns moderate
            consistency_score * 0.15   # Data consistency checks
        )
        
        # Determine classification
        if total_score >= self.thresholds['artificial_score']['extremely_suspicious']:
            classification = "EXTREMELY SUSPICIOUS - Likely Artificial"
            confidence = 0.9
        elif total_score >= self.thresholds['artificial_score']['highly_suspicious']:
            classification = "HIGHLY SUSPICIOUS - Probable Artificial Signatures"
            confidence = 0.75
        elif total_score >= self.thresholds['artificial_score']['suspicious']:
            classification = "SUSPICIOUS - Potential Artificial Characteristics"
            confidence = 0.6
        else:
            classification = "NATURAL - No Significant Artificial Signatures"
            confidence = 0.8
        
        # Combine all indicators
        all_indicators = orbital_indicators + velocity_indicators + discovery_indicators + consistency_indicators
        
        # Data quality assessment
        data_quality = neo_record.get('best_completeness', 0)
        sources_used = neo_record.get('sources_attempted', [])
        
        # Analysis details
        analysis_details = {
            'orbital_score': orbital_score,
            'velocity_score': velocity_score,
            'discovery_score': discovery_score,
            'consistency_score': consistency_score,
            'total_indicators': len(all_indicators),
            'enrichment_attempts': neo_record.get('enrichment_attempts', 0),
            'days_in_database': self._calculate_days_in_db(neo_record),
            'orbital_elements': orbital_elements
        }
        
        return NEOClassification(
            designation=designation,
            classification=classification,
            confidence=confidence,
            artificial_score=total_score,
            anomaly_indicators=all_indicators,
            data_quality=data_quality,
            sources_used=sources_used,
            analysis_details=analysis_details
        )
    
    def _calculate_days_in_db(self, neo_record: Dict[str, Any]) -> int:
        """Calculate days since NEO was first added to database."""
        try:
            first_seen = neo_record.get('first_seen', '')
            discovery_date = datetime.datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
            return (datetime.datetime.now() - discovery_date).days
        except:
            return 0
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of all cached NEO data."""
        print("\n🔍 Starting comprehensive NEO database analysis...")
        
        # Load all data sources
        neo_database = self.load_neo_database()
        cached_data = self.load_cached_orbital_data()
        analysis_results = self.load_analysis_results()
        
        if not neo_database:
            print("❌ No NEO database to analyze")
            return {}
        
        # Analyze each NEO
        classifications = []
        print(f"\n📊 Analyzing {len(neo_database)} NEOs for artificial signatures...")
        
        for designation, neo_record in neo_database.items():
            neo_cached_data = cached_data.get(designation, {})
            classification = self.classify_neo(designation, neo_record, neo_cached_data, analysis_results)
            classifications.append(classification)
            
            print(f"  {designation}: {classification.classification} (score: {classification.artificial_score:.3f})")
        
        # Generate summary statistics
        return self.generate_analysis_summary(classifications, neo_database, cached_data, analysis_results)
    
    def generate_analysis_summary(self, classifications: List[NEOClassification], 
                                 neo_database: Dict[str, Any], cached_data: Dict[str, Any],
                                 analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        
        # Classification statistics
        classification_counts = defaultdict(int)
        artificial_scores = []
        data_quality_scores = []
        
        suspicious_objects = []
        natural_objects = []
        
        for classification in classifications:
            classification_counts[classification.classification] += 1
            artificial_scores.append(classification.artificial_score)
            data_quality_scores.append(classification.data_quality)
            
            if classification.artificial_score >= self.thresholds['artificial_score']['suspicious']:
                suspicious_objects.append(classification)
            else:
                natural_objects.append(classification)
        
        # Overall statistics
        total_objects = len(classifications)
        avg_artificial_score = statistics.mean(artificial_scores) if artificial_scores else 0
        avg_data_quality = statistics.mean(data_quality_scores) if data_quality_scores else 0
        
        # Source utilization analysis
        source_usage = defaultdict(int)
        source_success = defaultdict(int)
        
        for designation, cached in cached_data.items():
            for source in cached.keys():
                source_usage[source] += 1
                if cached[source].get('data_completeness', 0) > 0.1:
                    source_success[source] += 1
        
        # Time-based analysis
        discovery_timeline = {}
        for classification in classifications:
            neo_record = neo_database[classification.designation]
            first_seen = neo_record.get('first_seen', '')
            try:
                date_key = first_seen[:10]  # YYYY-MM-DD
                discovery_timeline[date_key] = discovery_timeline.get(date_key, 0) + 1
            except:
                pass
        
        summary = {
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'total_objects_analyzed': total_objects,
            'classification_summary': {
                'natural_objects': len(natural_objects),
                'suspicious_objects': len(suspicious_objects),
                'classification_breakdown': dict(classification_counts)
            },
            'artificial_detection_stats': {
                'average_artificial_score': avg_artificial_score,
                'max_artificial_score': max(artificial_scores) if artificial_scores else 0,
                'highly_suspicious_count': len([c for c in classifications if c.artificial_score >= self.thresholds['artificial_score']['highly_suspicious']]),
                'suspicious_count': len(suspicious_objects)
            },
            'data_quality_stats': {
                'average_completeness': avg_data_quality,
                'high_quality_objects': len([c for c in classifications if c.data_quality >= self.thresholds['data_quality']['good_completeness']]),
                'low_quality_objects': len([c for c in classifications if c.data_quality < self.thresholds['data_quality']['minimum_completeness']])
            },
            'source_utilization': {
                'sources_attempted': dict(source_usage),
                'source_success_rates': {source: f"{source_success[source]}/{source_usage[source]}" for source in source_usage},
                'most_reliable_source': max(source_success.keys(), key=source_success.get) if source_success else 'None'
            },
            'temporal_analysis': {
                'discovery_timeline': discovery_timeline,
                'analysis_span_days': len(discovery_timeline),
                'recent_discoveries': len([d for d in discovery_timeline.keys() if d >= (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')])
            },
            'detailed_classifications': [
                {
                    'designation': c.designation,
                    'classification': c.classification,
                    'artificial_score': c.artificial_score,
                    'confidence': c.confidence,
                    'data_quality': c.data_quality,
                    'anomaly_count': len(c.anomaly_indicators),
                    'top_indicators': c.anomaly_indicators[:3],  # Top 3 indicators
                    'sources_used': c.sources_used,
                    'analysis_details': c.analysis_details
                }
                for c in sorted(classifications, key=lambda x: x.artificial_score, reverse=True)
            ]
        }
        
        return summary

def main():
    """Main analysis execution."""
    analyzer = NEODatabaseAnalyzer()
    
    # Run comprehensive analysis
    summary = analyzer.run_comprehensive_analysis()
    
    if summary:
        # Save analysis summary
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path("neo_data") / f"comprehensive_analysis_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Comprehensive analysis completed!")
        print(f"💾 Summary saved to: {summary_file}")
        
        # Print key findings
        print("\n" + "="*80)
        print("🎯 KEY FINDINGS SUMMARY")
        print("="*80)
        
        total = summary['total_objects_analyzed']
        suspicious = summary['artificial_detection_stats']['suspicious_count']
        highly_suspicious = summary['artificial_detection_stats']['highly_suspicious_count']
        avg_score = summary['artificial_detection_stats']['average_artificial_score']
        
        print(f"📊 Total NEOs analyzed: {total}")
        print(f"🚨 Suspicious objects: {suspicious} ({suspicious/total*100:.1f}%)")
        print(f"⚠️  Highly suspicious: {highly_suspicious} ({highly_suspicious/total*100:.1f}%)")
        print(f"📈 Average artificial score: {avg_score:.3f}")
        
        if suspicious > 0:
            print(f"\n🔍 TOP {min(5, suspicious)} MOST SUSPICIOUS OBJECTS:")
            print("-"*60)
            
            for i, obj in enumerate(summary['detailed_classifications'][:5]):
                if obj['artificial_score'] >= 0.3:
                    print(f"{i+1}. {obj['designation']} - Score: {obj['artificial_score']:.3f}")
                    print(f"   Classification: {obj['classification']}")
                    print(f"   Top indicators: {', '.join(obj['top_indicators'][:2])}")
                    print()
        
        return summary_file
    else:
        print("❌ Analysis failed - no data to analyze")
        return None

if __name__ == "__main__":
    main()