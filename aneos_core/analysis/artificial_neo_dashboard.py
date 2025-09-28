"""
Artificial NEO Analysis Dashboard

Comprehensive dashboard for displaying and categorizing artificial NEO detection results
after polling processes. Provides clear visualization of suspicious, edge case, and 
confirmed artificial NEOs with detailed analysis metrics.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.columns import Columns
    from rich.tree import Tree
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)

@dataclass
class NEOClassification:
    """Classification result for a NEO."""
    designation: str
    category: str  # 'artificial', 'suspicious', 'edge_case', 'natural'
    confidence: float
    artificial_probability: float
    risk_factors: List[str]
    analysis_details: Dict[str, Any]
    data_sources: List[str]
    analysis_timestamp: datetime
    orbital_elements: Dict[str, float]

@dataclass
class DashboardMetrics:
    """Dashboard summary metrics."""
    total_objects: int
    artificial_confirmed: int
    suspicious_objects: int
    edge_cases: int
    natural_objects: int
    high_confidence_detections: int
    data_quality_score: float
    analysis_timestamp: datetime

class ArtificialNEODashboard:
    """
    Comprehensive dashboard for artificial NEO analysis results.
    
    Categorizes NEOs into:
    - ARTIFICIAL: High confidence artificial objects (≥0.8 probability)
    - SUSPICIOUS: Objects requiring investigation (0.5-0.8 probability)
    - EDGE_CASE: Borderline objects with unusual characteristics (0.3-0.5 probability)
    - NATURAL: Confirmed natural objects (<0.3 probability)
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or (Console() if HAS_RICH else None)
        self.classifications: List[NEOClassification] = []
        self.metrics: Optional[DashboardMetrics] = None
        
        # Classification thresholds
        self.thresholds = {
            'artificial': 0.8,      # High confidence artificial
            'suspicious': 0.5,      # Requires investigation
            'edge_case': 0.3,       # Borderline/unusual
            'natural': 0.0          # Natural objects
        }
    
    async def analyze_polling_results(self, polling_results: List[Dict[str, Any]]) -> List[NEOClassification]:
        """
        Analyze polling results and classify NEOs for dashboard display.
        
        Args:
            polling_results: Raw results from polling process
            
        Returns:
            List of classified NEO objects
        """
        classifications = []
        
        if self.console:
            self.console.print("\n🔬 [bold cyan]Analyzing Polling Results for Artificial Signatures[/bold cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Classifying NEOs...", total=len(polling_results))
                
                for result in polling_results:
                    classification = await self._classify_neo(result)
                    classifications.append(classification)
                    progress.advance(task)
        else:
            print("\n🔬 Analyzing polling results for artificial signatures...")
            for i, result in enumerate(polling_results, 1):
                if i % 10 == 0:
                    print(f"   Processed {i}/{len(polling_results)} objects...")
                classification = await self._classify_neo(result)
                classifications.append(classification)
        
        self.classifications = classifications
        self.metrics = self._calculate_metrics(classifications)
        
        return classifications
    
    async def _classify_neo(self, result: Dict[str, Any]) -> NEOClassification:
        """Classify a single NEO based on analysis results."""
        
        # Extract key data
        designation = result.get('designation', 'Unknown')
        artificial_prob = result.get('artificial_probability', result.get('artificial_score', 0.0))
        risk_factors = result.get('risk_factors', result.get('indicators', []))
        data_sources = result.get('data_sources', [result.get('data_source', 'Unknown')])
        orbital_elements = result.get('orbital_elements', {})
        
        # Enhanced analysis using detection system
        try:
            from ..detection.multimodal_sigma5_artificial_neo_detector import MultiModalSigma5ArtificialNEODetector
            detector = MultiModalSigma5ArtificialNEODetector()
            
            # Run enhanced detection if orbital elements available
            if orbital_elements:
                enhanced_result = await detector.analyze_neo_data({
                    'designation': designation,
                    'orbital_elements': orbital_elements
                })
                
                if enhanced_result:
                    artificial_prob = max(artificial_prob, enhanced_result.get('artificial_probability', 0.0))
                    additional_factors = enhanced_result.get('risk_factors', [])
                    risk_factors.extend(additional_factors)
        except Exception as e:
            logger.debug(f"Enhanced detection failed for {designation}: {e}")
        
        # Determine category based on probability
        if artificial_prob >= self.thresholds['artificial']:
            category = 'artificial'
            confidence = 0.9 + (artificial_prob - 0.8) * 0.5  # Scale 0.8-1.0 to 0.9-0.95
        elif artificial_prob >= self.thresholds['suspicious']:
            category = 'suspicious'
            confidence = 0.6 + (artificial_prob - 0.5) * 0.6  # Scale 0.5-0.8 to 0.6-0.9
        elif artificial_prob >= self.thresholds['edge_case']:
            category = 'edge_case'
            confidence = 0.4 + (artificial_prob - 0.3) * 1.0  # Scale 0.3-0.5 to 0.4-0.6
        else:
            category = 'natural'
            confidence = 0.8 - artificial_prob * 2.0  # Higher confidence for lower prob
        
        # Additional analysis based on risk factors
        analysis_details = {
            'primary_indicators': risk_factors[:3],
            'data_completeness': result.get('data_completeness', 0.5),
            'analysis_method': result.get('analysis_method', 'standard'),
            'orbital_anomalies': self._detect_orbital_anomalies(orbital_elements),
            'temporal_patterns': self._analyze_temporal_patterns(result),
            'physical_characteristics': self._analyze_physical_characteristics(result)
        }
        
        return NEOClassification(
            designation=designation,
            category=category,
            confidence=min(0.95, max(0.1, confidence)),
            artificial_probability=artificial_prob,
            risk_factors=list(set(risk_factors)),  # Remove duplicates
            analysis_details=analysis_details,
            data_sources=data_sources if isinstance(data_sources, list) else [data_sources],
            analysis_timestamp=datetime.now(),
            orbital_elements=orbital_elements
        )
    
    def _detect_orbital_anomalies(self, orbital_elements: Dict[str, float]) -> List[str]:
        """Detect orbital anomalies that may indicate artificial origin."""
        anomalies = []
        
        if not orbital_elements:
            return anomalies
        
        # Eccentricity analysis
        eccentricity = orbital_elements.get('e', orbital_elements.get('eccentricity', 0.0))
        if eccentricity > 0.7:
            anomalies.append(f"Extremely high eccentricity: {eccentricity:.3f}")
        elif eccentricity > 0.4:
            anomalies.append(f"High eccentricity: {eccentricity:.3f}")
        
        # Inclination analysis
        inclination = orbital_elements.get('i', orbital_elements.get('inclination', 0.0))
        if inclination > 45:
            anomalies.append(f"High inclination: {inclination:.1f}°")
        elif inclination < 0.5:
            anomalies.append(f"Unusually low inclination: {inclination:.1f}°")
        
        # Semi-major axis analysis
        semi_major_axis = orbital_elements.get('a', orbital_elements.get('semi_major_axis', 1.0))
        if semi_major_axis < 0.8 or semi_major_axis > 3.0:
            anomalies.append(f"Unusual orbital size: {semi_major_axis:.3f} AU")
        
        # Check for suspiciously round numbers (potential artificial precision)
        for key, value in orbital_elements.items():
            if isinstance(value, (int, float)):
                # Check if value is suspiciously round
                if abs(value - round(value, 1)) < 0.001 and value != 0:
                    anomalies.append(f"Suspiciously precise {key}: {value}")
        
        return anomalies
    
    def _analyze_temporal_patterns(self, result: Dict[str, Any]) -> List[str]:
        """Analyze temporal patterns that may indicate artificial behavior."""
        patterns = []
        
        # Check for discovery patterns
        discovery_date = result.get('discovery_date')
        if discovery_date:
            # Recent discoveries are more likely to be artificial debris
            try:
                disc_date = datetime.fromisoformat(discovery_date.replace('Z', '+00:00'))
                if (datetime.now() - disc_date).days < 30:
                    patterns.append("Recent discovery (last 30 days)")
            except:
                pass
        
        # Check observation patterns
        observation_count = result.get('observation_count', 0)
        if observation_count < 10:
            patterns.append("Limited observational data")
        elif observation_count > 1000:
            patterns.append("Extensive observational history")
        
        return patterns
    
    def _analyze_physical_characteristics(self, result: Dict[str, Any]) -> List[str]:
        """Analyze physical characteristics for artificial signatures."""
        characteristics = []
        
        # Size analysis
        diameter = result.get('diameter', result.get('physical_properties', {}).get('diameter'))
        if diameter:
            if diameter < 1.0:  # Very small objects are often artificial
                characteristics.append(f"Very small size: {diameter:.2f}m")
            elif diameter > 1000:  # Very large objects are usually natural
                characteristics.append(f"Large natural-sized object: {diameter:.0f}m")
        
        # Albedo analysis
        albedo = result.get('albedo', result.get('physical_properties', {}).get('albedo'))
        if albedo:
            if albedo > 0.5:  # High albedo may indicate metallic/artificial surfaces
                characteristics.append(f"High albedo (metallic?): {albedo:.3f}")
            elif albedo < 0.03:  # Very dark objects
                characteristics.append(f"Very low albedo: {albedo:.3f}")
        
        return characteristics
    
    def _calculate_metrics(self, classifications: List[NEOClassification]) -> DashboardMetrics:
        """Calculate dashboard summary metrics."""
        if not classifications:
            return DashboardMetrics(0, 0, 0, 0, 0, 0, 0.0, datetime.now())
        
        total = len(classifications)
        artificial = len([c for c in classifications if c.category == 'artificial'])
        suspicious = len([c for c in classifications if c.category == 'suspicious'])
        edge_cases = len([c for c in classifications if c.category == 'edge_case'])
        natural = len([c for c in classifications if c.category == 'natural'])
        high_confidence = len([c for c in classifications if c.confidence >= 0.8])
        
        # Calculate data quality score
        avg_confidence = sum(c.confidence for c in classifications) / total
        data_quality = avg_confidence * 0.7 + (high_confidence / total) * 0.3
        
        return DashboardMetrics(
            total_objects=total,
            artificial_confirmed=artificial,
            suspicious_objects=suspicious,
            edge_cases=edge_cases,
            natural_objects=natural,
            high_confidence_detections=high_confidence,
            data_quality_score=data_quality,
            analysis_timestamp=datetime.now()
        )
    
    def display_dashboard(self, save_results: bool = True, output_dir: str = "dashboard_results"):
        """Display the comprehensive artificial NEO analysis dashboard."""
        
        if not self.classifications or not self.metrics:
            if self.console:
                self.console.print("❌ [red]No analysis results available. Run analysis first.[/red]")
            else:
                print("❌ No analysis results available. Run analysis first.")
            return
        
        if self.console:
            self._display_rich_dashboard()
        else:
            self._display_text_dashboard()
        
        if save_results:
            self._save_dashboard_results(output_dir)
    
    def _display_rich_dashboard(self):
        """Display dashboard using Rich interface."""
        # Clear screen and show header
        self.console.clear()
        
        # Title Panel
        title = Panel(
            Text("🛸 ARTIFICIAL NEO DETECTION DASHBOARD", style="bold cyan", justify="center"),
            title="aNEOS Analysis Results",
            border_style="cyan"
        )
        self.console.print(title)
        
        # Metrics Overview
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("", style="dim")
        
        metrics_table.add_row("🎯 Total Objects Analyzed", str(self.metrics.total_objects), "")
        metrics_table.add_row("🛸 Confirmed Artificial", str(self.metrics.artificial_confirmed), 
                              f"({self.metrics.artificial_confirmed/self.metrics.total_objects*100:.1f}%)")
        metrics_table.add_row("⚠️  Suspicious Objects", str(self.metrics.suspicious_objects),
                              f"({self.metrics.suspicious_objects/self.metrics.total_objects*100:.1f}%)")
        metrics_table.add_row("❓ Edge Cases", str(self.metrics.edge_cases),
                              f"({self.metrics.edge_cases/self.metrics.total_objects*100:.1f}%)")
        metrics_table.add_row("🌍 Natural Objects", str(self.metrics.natural_objects),
                              f"({self.metrics.natural_objects/self.metrics.total_objects*100:.1f}%)")
        metrics_table.add_row("📊 Data Quality Score", f"{self.metrics.data_quality_score:.3f}", "")
        
        metrics_panel = Panel(metrics_table, title="[bold]📊 Analysis Summary[/bold]", border_style="green")
        self.console.print(metrics_panel)
        
        # Category Details
        self._display_category_details()
        
        # Risk Assessment
        self._display_risk_assessment()
    
    def _display_category_details(self):
        """Display detailed breakdown by category."""
        categories = ['artificial', 'suspicious', 'edge_case', 'natural']
        category_icons = {'artificial': '🛸', 'suspicious': '⚠️', 'edge_case': '❓', 'natural': '🌍'}
        category_colors = {'artificial': 'red', 'suspicious': 'yellow', 'edge_case': 'blue', 'natural': 'green'}
        
        for category in categories:
            objects = [c for c in self.classifications if c.category == category]
            if not objects:
                continue
            
            # Create table for this category
            table = Table(show_header=True)
            table.add_column("Designation", style="bold")
            table.add_column("Probability", justify="center")
            table.add_column("Confidence", justify="center")
            table.add_column("Top Risk Factors", style="dim")
            
            # Sort by probability (highest first)
            sorted_objects = sorted(objects, key=lambda x: x.artificial_probability, reverse=True)
            
            for obj in sorted_objects[:10]:  # Show top 10 per category
                factors_str = ", ".join(obj.risk_factors[:2]) if obj.risk_factors else "None"
                if len(factors_str) > 50:
                    factors_str = factors_str[:47] + "..."
                
                table.add_row(
                    obj.designation,
                    f"{obj.artificial_probability:.3f}",
                    f"{obj.confidence:.3f}",
                    factors_str
                )
            
            title = f"{category_icons[category]} {category.upper()} OBJECTS ({len(objects)})"
            panel = Panel(table, title=f"[bold {category_colors[category]}]{title}[/bold {category_colors[category]}]", 
                         border_style=category_colors[category])
            self.console.print(panel)
            
            if len(objects) > 10:
                self.console.print(f"   [dim]... and {len(objects) - 10} more {category} objects[/dim]\n")
    
    def _display_risk_assessment(self):
        """Display overall risk assessment."""
        # Calculate risk levels
        high_risk = self.metrics.artificial_confirmed + self.metrics.suspicious_objects
        medium_risk = self.metrics.edge_cases
        total_concerning = high_risk + medium_risk
        
        # Risk assessment
        if total_concerning == 0:
            risk_level = "LOW"
            risk_color = "green"
            risk_msg = "No artificial or suspicious objects detected."
        elif total_concerning < self.metrics.total_objects * 0.1:
            risk_level = "MODERATE"
            risk_color = "yellow"
            risk_msg = f"{total_concerning} objects require further investigation."
        else:
            risk_level = "HIGH"
            risk_color = "red"
            risk_msg = f"{total_concerning} objects show artificial signatures - immediate review recommended."
        
        risk_table = Table(show_header=False, box=None)
        risk_table.add_column("", style="bold")
        risk_table.add_column("", style="white")
        
        risk_table.add_row("🎯 Risk Level:", f"[{risk_color}]{risk_level}[/{risk_color}]")
        risk_table.add_row("📝 Assessment:", risk_msg)
        risk_table.add_row("⏰ Analysis Time:", self.metrics.analysis_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
        
        risk_panel = Panel(risk_table, title="[bold]🚨 Risk Assessment[/bold]", border_style=risk_color)
        self.console.print(risk_panel)
    
    def _display_text_dashboard(self):
        """Display dashboard using text-only interface."""
        print("\n" + "="*80)
        print("🛸 ARTIFICIAL NEO DETECTION DASHBOARD")
        print("="*80)
        
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Total Objects: {self.metrics.total_objects}")
        print(f"   🛸 Artificial: {self.metrics.artificial_confirmed} ({self.metrics.artificial_confirmed/self.metrics.total_objects*100:.1f}%)")
        print(f"   ⚠️  Suspicious: {self.metrics.suspicious_objects} ({self.metrics.suspicious_objects/self.metrics.total_objects*100:.1f}%)")
        print(f"   ❓ Edge Cases: {self.metrics.edge_cases} ({self.metrics.edge_cases/self.metrics.total_objects*100:.1f}%)")
        print(f"   🌍 Natural: {self.metrics.natural_objects} ({self.metrics.natural_objects/self.metrics.total_objects*100:.1f}%)")
        print(f"   Data Quality: {self.metrics.data_quality_score:.3f}")
        
        # Show top objects in each category
        for category in ['artificial', 'suspicious', 'edge_case']:
            objects = [c for c in self.classifications if c.category == category]
            if objects:
                print(f"\n{category.upper()} OBJECTS ({len(objects)}):")
                sorted_objects = sorted(objects, key=lambda x: x.artificial_probability, reverse=True)
                for obj in sorted_objects[:5]:
                    print(f"  {obj.designation:15} - Prob: {obj.artificial_probability:.3f} - Conf: {obj.confidence:.3f}")
                    for factor in obj.risk_factors[:2]:
                        print(f"    • {factor}")
    
    def _save_dashboard_results(self, output_dir: str):
        """Save dashboard results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"artificial_neo_analysis_{timestamp}.json"
        
        dashboard_data = {
            'metadata': {
                'analysis_timestamp': self.metrics.analysis_timestamp.isoformat(),
                'total_objects': self.metrics.total_objects,
                'data_quality_score': self.metrics.data_quality_score
            },
            'summary': asdict(self.metrics),
            'classifications': [asdict(c) for c in self.classifications],
            'thresholds': self.thresholds
        }
        
        # Convert datetime objects to strings for JSON serialization
        for classification in dashboard_data['classifications']:
            classification['analysis_timestamp'] = classification['analysis_timestamp'].isoformat()
        
        with open(results_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        if self.console:
            self.console.print(f"\n💾 [green]Dashboard results saved to: {results_file}[/green]")
        else:
            print(f"\n💾 Dashboard results saved to: {results_file}")

# Integration function for polling workflow
async def create_dashboard_from_polling(polling_results: List[Dict[str, Any]], 
                                      display: bool = True, 
                                      save: bool = True) -> ArtificialNEODashboard:
    """
    Create and display dashboard from polling results.
    
    Args:
        polling_results: Results from NEO polling process
        display: Whether to display the dashboard
        save: Whether to save results to disk
        
    Returns:
        Configured dashboard instance
    """
    dashboard = ArtificialNEODashboard()
    await dashboard.analyze_polling_results(polling_results)
    
    if display:
        dashboard.display_dashboard(save_results=save)
    
    return dashboard