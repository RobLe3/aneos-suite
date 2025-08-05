#!/usr/bin/env python3
"""
Professional Visualization Components for aNEOS Core

Provides comprehensive visualization capabilities including orbital maps,
statistical charts, and interactive visualizations for NEO analysis results.
"""

import os
import json
from datetime import datetime

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class Visualizer:
    """
    Professional visualization generator for NEO analysis results.
    
    Provides 2D/3D orbital maps, statistical charts, and interactive
    visualizations with support for multiple output formats.
    """
    
    def __init__(self, output_dir: str = "visualizations", logger: Optional[logging.Logger] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for visualization output
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Check available libraries
        self.has_matplotlib = HAS_MATPLOTLIB
        self.has_plotly = HAS_PLOTLY
        self.has_seaborn = HAS_SEABORN
        self.has_pandas = HAS_PANDAS
        
        if not (self.has_matplotlib or self.has_plotly):
            self.logger.warning("No visualization libraries available. Install matplotlib or plotly for visualizations.")
        
        # Configuration for visualizations
        self.config = {
            "figure_size": (12, 8),
            "dpi": 300,
            "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "anomaly_color": "#d62728",
            "normal_color": "#1f77b4",
            "verified_color": "#2ca02c",
            "unverified_color": "#ff7f0e"
        }
    
    def compute_distance_from_earth(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute distance from Earth for orbital visualization.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Data with distance_from_earth field added
        """
        enhanced_data = []
        
        for neo in data:
            neo_copy = neo.copy()
            
            # Calculate distance based on semi-major axis (AU)
            semi_major_axis = neo.get("semi_major_axis", 0)
            if semi_major_axis > 0:
                # Distance from Earth's orbit (1 AU)
                neo_copy["distance_from_earth"] = abs(semi_major_axis - 1.0)
            else:
                neo_copy["distance_from_earth"] = 0
            
            enhanced_data.append(neo_copy)
        
        return enhanced_data
    
    def generate_2d_orbital_map(self, data: List[Dict[str, Any]], 
                              output_file: Optional[str] = None,
                              title: str = "2D Orbital Map") -> Optional[str]:
        """
        Generate a 2D orbital map visualization.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            title: Plot title
            
        Returns:
            Path to generated file or None if failed
        """
        if not self.has_matplotlib:
            self.logger.error("Matplotlib not available for 2D visualization")
            return None
        
        # Prepare data
        enhanced_data = self.compute_distance_from_earth(data)
        
        # Extract orbital parameters
        semi_major_axes = []
        inclinations = []
        distances = []
        is_anomalous = []
        designations = []
        
        for neo in enhanced_data:
            sma = neo.get("semi_major_axis", 0)
            inc = neo.get("inclination", 0)
            dist = neo.get("distance_from_earth", 0)
            
            if sma > 0 and inc >= 0:  # Valid orbital data
                semi_major_axes.append(sma)
                inclinations.append(inc)
                distances.append(dist)
                
                # Determine if anomalous
                anomalous = (
                    neo.get("ai_validated_anomaly", False) or 
                    neo.get("category") == "ISO Candidate" or 
                    neo.get("delta_v_anomaly_score", 0) > 1.5
                )
                is_anomalous.append(anomalous)
                designations.append(neo.get("Designation", "Unknown"))
        
        if not semi_major_axes:
            self.logger.warning("No valid orbital data for 2D visualization")
            return None
        
        # Create the plot
        plt.figure(figsize=self.config["figure_size"], dpi=self.config["dpi"])
        
        # Create scatter plot with color coding
        colors = [self.config["anomaly_color"] if anom else self.config["normal_color"] 
                 for anom in is_anomalous]
        
        scatter = plt.scatter(semi_major_axes, inclinations, 
                            c=distances, cmap='viridis', 
                            alpha=0.7, edgecolors=colors, 
                            linewidth=2, s=60)
        
        # Add Earth's orbit reference
        plt.axvline(x=1.0, color='blue', linestyle='--', alpha=0.5, label='Earth Orbit (1 AU)')
        
        # Formatting
        plt.xlabel("Semi-Major Axis (AU)", fontsize=12)
        plt.ylabel("Inclination (Degrees)", fontsize=12)
        
        # Calculate statistics for title
        avg_distance = (sum(distances) / len(distances)) if distances else 0
        total_neos = len(enhanced_data)
        anomalous_count = sum(is_anomalous)
        
        full_title = f"{title}\nTotal NEOs: {total_neos} | Anomalous: {anomalous_count} | Avg Distance: {avg_distance:.2f} AU"
        plt.title(full_title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Distance from Earth (AU)", fontsize=11)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.config["normal_color"], label='Normal NEOs'),
            Patch(facecolor=self.config["anomaly_color"], label='Anomalous NEOs'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Earth Orbit')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"orbital_map_2d_{timestamp}.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"2D orbital map saved to {output_path}")
        return str(output_path)
    
    def generate_3d_orbital_map(self, data: List[Dict[str, Any]], 
                              output_file: Optional[str] = None,
                              title: str = "3D Orbital Map") -> Optional[str]:
        """
        Generate a 3D interactive orbital map visualization.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            title: Plot title
            
        Returns:
            Path to generated file or None if failed
        """
        if not self.has_plotly:
            self.logger.error("Plotly not available for 3D visualization")
            return None
        
        # Prepare data
        enhanced_data = self.compute_distance_from_earth(data)
        
        # Extract orbital parameters
        plot_data = {
            'x': [],
            'y': [],
            'z': [],
            'color': [],
            'text': [],
            'size': [],
            'symbol': []
        }
        
        for neo in enhanced_data:
            sma = neo.get("semi_major_axis", 0)
            ecc = neo.get("eccentricity", 0)
            inc = neo.get("inclination", 0)
            dist = neo.get("distance_from_earth", 0)
            
            if sma > 0:  # Valid orbital data
                plot_data['x'].append(sma)
                plot_data['y'].append(ecc)
                plot_data['z'].append(inc)
                plot_data['color'].append(dist)
                
                # Create hover text
                designation = neo.get("Designation", "Unknown")
                anomaly_conf = neo.get("anomaly_confidence", 0)
                priority = neo.get("priority_score", 0)
                
                hover_text = (
                    f"Designation: {designation}<br>"
                    f"Semi-Major Axis: {sma:.3f} AU<br>"
                    f"Eccentricity: {ecc:.3f}<br>"
                    f"Inclination: {inc:.2f}°<br>"
                    f"Distance from Earth: {dist:.3f} AU<br>"
                    f"Anomaly Confidence: {anomaly_conf:.2f}<br>"
                    f"Priority Score: {priority:.2f}"
                )
                plot_data['text'].append(hover_text)
                
                # Determine size and symbol based on anomaly status
                anomalous = (
                    neo.get("ai_validated_anomaly", False) or 
                    neo.get("category") == "ISO Candidate" or 
                    neo.get("delta_v_anomaly_score", 0) > 1.5
                )
                
                plot_data['size'].append(8 if anomalous else 5)
                plot_data['symbol'].append('diamond' if anomalous else 'circle')
        
        if not plot_data['x']:
            self.logger.warning("No valid orbital data for 3D visualization")
            return None
        
        # Create the 3D plot
        fig = go.Figure()
        
        # Add NEO scatter plot
        fig.add_trace(go.Scatter3d(
            x=plot_data['x'],
            y=plot_data['y'],
            z=plot_data['z'],
            mode='markers',
            marker=dict(
                size=plot_data['size'],
                color=plot_data['color'],
                colorscale='Viridis',
                colorbar=dict(title="Distance from Earth (AU)"),
                symbol=plot_data['symbol'],
                line=dict(width=1, color='white')
            ),
            text=plot_data['text'],
            hovertemplate='%{text}<extra></extra>',
            name="NEOs"
        ))
        
        # Add Earth reference point
        fig.add_trace(go.Scatter3d(
            x=[1], y=[0], z=[0],
            mode='markers',
            marker=dict(size=12, color='blue', symbol='circle'),
            name="Earth Orbit Reference",
            hovertemplate='Earth Orbit<br>1 AU<extra></extra>'
        ))
        
        # Calculate statistics for title
        avg_distance = (sum(plot_data['color']) / len(plot_data['color'])) if plot_data['color'] else 0
        total_neos = len(enhanced_data)
        anomalous_count = sum(1 for neo in enhanced_data if (
            neo.get("ai_validated_anomaly", False) or 
            neo.get("category") == "ISO Candidate" or 
            neo.get("delta_v_anomaly_score", 0) > 1.5
        ))
        
        full_title = f"{title}<br>Total NEOs: {total_neos} | Anomalous: {anomalous_count} | Avg Distance: {avg_distance:.2f} AU"
        
        # Update layout
        fig.update_layout(
            title=dict(text=full_title, x=0.5, font=dict(size=16)),
            scene=dict(
                xaxis_title='Semi-Major Axis (AU)',
                yaxis_title='Eccentricity',
                zaxis_title='Inclination (Degrees)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=700,
            showlegend=True
        )
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"orbital_map_3d_{timestamp}.html"
        
        output_path = self.output_dir / output_file
        fig.write_html(output_path)
        
        self.logger.info(f"3D orbital map saved to {output_path}")
        return str(output_path)
    
    def generate_anomaly_distribution_chart(self, data: List[Dict[str, Any]], 
                                          output_file: Optional[str] = None) -> Optional[str]:
        """
        Generate a chart showing anomaly confidence distribution.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            
        Returns:
            Path to generated file or None if failed
        """
        if not self.has_matplotlib:
            self.logger.error("Matplotlib not available for distribution chart")
            return None
        
        # Extract anomaly confidence values
        confidences = []
        verified_status = []
        
        for neo in data:
            conf = neo.get("anomaly_confidence", 0)
            if conf > 0:  # Only include NEOs with anomaly data
                confidences.append(conf)
                verified_status.append(conf > 10)  # Verification threshold
        
        if not confidences:
            self.logger.warning("No anomaly confidence data available")
            return None
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.config["dpi"])
        
        # Histogram of anomaly confidence
        ax1.hist(confidences, bins=30, alpha=0.7, color=self.config["anomaly_color"], 
                edgecolor='black', linewidth=0.5)
        ax1.axvline(x=10, color='red', linestyle='--', linewidth=2, 
                   label='Verification Threshold (>10)')
        ax1.set_xlabel('Anomaly Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Anomaly Confidence Values', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparing verified vs unverified
        verified_conf = [c for c, v in zip(confidences, verified_status) if v]
        unverified_conf = [c for c, v in zip(confidences, verified_status) if not v]
        
        box_data = []
        labels = []
        
        if verified_conf:
            box_data.append(verified_conf)
            labels.append(f'Verified\n(n={len(verified_conf)})')
        
        if unverified_conf:
            box_data.append(unverified_conf)
            labels.append(f'Unverified\n(n={len(unverified_conf)})')
        
        if box_data:
            bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = [self.config["verified_color"], self.config["unverified_color"]]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax2.set_ylabel('Anomaly Confidence')
        ax2.set_title('Anomaly Confidence by Verification Status', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"anomaly_distribution_{timestamp}.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Anomaly distribution chart saved to {output_path}")
        return str(output_path)
    
    def generate_category_breakdown_chart(self, data: List[Dict[str, Any]], 
                                        output_file: Optional[str] = None) -> Optional[str]:
        """
        Generate a pie chart showing category breakdown.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            
        Returns:
            Path to generated file or None if failed
        """
        if not self.has_matplotlib:
            self.logger.error("Matplotlib not available for category chart")
            return None
        
        # Count categories
        categories = {}
        for neo in data:
            category = neo.get("category") or neo.get("dynamic_category", "Uncategorized")
            categories[category] = categories.get(category, 0) + 1
        
        if not categories:
            self.logger.warning("No category data available")
            return None
        
        # Create the pie chart
        plt.figure(figsize=self.config["figure_size"], dpi=self.config["dpi"])
        
        labels = list(categories.keys())
        sizes = list(categories.values())
        colors = plt.cm.Set3([i / (len(labels) - 1) for i in range(len(labels))] if len(labels) > 1 else [0])
        
        # Create pie chart with percentage labels
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 10})
        
        # Enhance text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title(f'NEO Category Distribution\nTotal NEOs: {sum(sizes)}', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Equal aspect ratio ensures pie is circular
        plt.axis('equal')
        
        # Add a legend with counts
        legend_labels = [f'{label} ({count})' for label, count in zip(labels, sizes)]
        plt.legend(wedges, legend_labels, title="Categories", loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"category_breakdown_{timestamp}.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Category breakdown chart saved to {output_path}")
        return str(output_path)
    
    def generate_priority_ranking_chart(self, data: List[Dict[str, Any]], 
                                      top_n: int = 20,
                                      output_file: Optional[str] = None) -> Optional[str]:
        """
        Generate a horizontal bar chart showing priority rankings.
        
        Args:
            data: List of NEO data dictionaries
            top_n: Number of top priorities to show
            output_file: Optional output file path
            
        Returns:
            Path to generated file or None if failed
        """
        if not self.has_matplotlib:
            self.logger.error("Matplotlib not available for priority chart")
            return None
        
        # Filter and sort by priority
        priority_data = [
            neo for neo in data 
            if neo.get("priority_score", 0) > 0
        ]
        priority_data.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        
        if not priority_data:
            self.logger.warning("No priority data available")
            return None
        
        # Take top N
        top_data = priority_data[:top_n]
        
        # Prepare plot data
        designations = [neo.get("Designation", "Unknown")[:15] for neo in top_data]
        priorities = [neo.get("priority_score", 0) for neo in top_data]
        verified = [neo.get("anomaly_confidence", 0) > 10 for neo in top_data]
        
        # Create colors based on verification status
        colors = [self.config["verified_color"] if v else self.config["unverified_color"] 
                 for v in verified]
        
        # Create horizontal bar chart
        plt.figure(figsize=(12, max(8, len(top_data) * 0.4)), dpi=self.config["dpi"])
        
        bars = plt.barh(range(len(designations)), priorities, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        plt.xlabel('Priority Score', fontsize=12)
        plt.ylabel('NEO Designation', fontsize=12)
        plt.title(f'Top {len(top_data)} Mission Priority Targets', fontsize=14, fontweight='bold')
        
        # Set y-axis labels
        plt.yticks(range(len(designations)), designations)
        
        # Add value labels on bars
        for i, (bar, priority) in enumerate(zip(bars, priorities)):
            plt.text(bar.get_width() + 0.01 * max(priorities), bar.get_y() + bar.get_height()/2, 
                    f'{priority:.2f}', ha='left', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.config["verified_color"], label='Verified Anomalies'),
            Patch(facecolor=self.config["unverified_color"], label='Unverified Anomalies')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Invert y-axis so highest priority is at top
        plt.gca().invert_yaxis()
        
        # Save the plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"priority_ranking_{timestamp}.png"
        
        output_path = self.output_dir / output_file
        plt.savefig(output_path, dpi=self.config["dpi"], bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Priority ranking chart saved to {output_path}")
        return str(output_path)
    
    def generate_all_visualizations(self, data: List[Dict[str, Any]], 
                                  base_filename: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        Generate all available visualizations.
        
        Args:
            data: List of NEO data dictionaries
            base_filename: Base filename for output files
            
        Returns:
            Dictionary mapping visualization types to their file paths
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"aneos_viz_{timestamp}"
        
        visualizations = {}
        
        # Generate each visualization type
        if self.has_matplotlib:
            visualizations["orbital_map_2d"] = self.generate_2d_orbital_map(
                data, f"{base_filename}_orbital_2d.png"
            )
            
            visualizations["anomaly_distribution"] = self.generate_anomaly_distribution_chart(
                data, f"{base_filename}_anomaly_dist.png"
            )
            
            visualizations["category_breakdown"] = self.generate_category_breakdown_chart(
                data, f"{base_filename}_categories.png"
            )
            
            visualizations["priority_ranking"] = self.generate_priority_ranking_chart(
                data, output_file=f"{base_filename}_priorities.png"
            )
        
        if self.has_plotly:
            visualizations["orbital_map_3d"] = self.generate_3d_orbital_map(
                data, f"{base_filename}_orbital_3d.html"
            )
        
        # Filter out None results
        visualizations = {k: v for k, v in visualizations.items() if v is not None}
        
        self.logger.info(f"Generated {len(visualizations)} visualizations with base filename: {base_filename}")
        return visualizations