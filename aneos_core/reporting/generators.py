#!/usr/bin/env python3
"""
Professional Report Generators for aNEOS Core

Provides comprehensive report generation capabilities with multiple output formats,
professional formatting, and academic rigor suitable for scientific analysis.
"""

import os
import json
import textwrap
import shutil
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ReportGenerator:
    """
    Professional report generator with multi-format output capabilities.
    
    Provides structured report generation for NEO analysis results with
    support for summary, detailed, priority, and anomaly reports.
    """
    
    def __init__(self, output_dir: str = "reports", logger: Optional[logging.Logger] = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for report output
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        self.console = Console() if HAS_RICH else None
        
        # Configuration for report formatting
        self.config = {
            "line_width": shutil.get_terminal_size(fallback=(80, 20)).columns,
            "border_char": "=",
            "separator_char": "-",
            "indent": "  ",
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "date_format": "%Y-%m-%d"
        }
        
        # Color codes for console output
        self.colors = {
            "header": "35",    # Magenta
            "title": "33",     # Yellow
            "success": "32",   # Green
            "warning": "31",   # Red
            "info": "36",      # Cyan
            "data": "34",      # Blue
            "emphasis": "1"    # Bold
        }
    
    def colorize(self, text: str, color_code: str) -> str:
        """Apply ANSI color codes to text."""
        return f"\033[{color_code}m{text}\033[0m"
    
    def wrap_text(self, text: str, width: Optional[int] = None) -> str:
        """Wrap text to terminal width."""
        width = width or self.config["line_width"]
        return textwrap.fill(text, width=width)
    
    def print_wrapped(self, text: str, color_code: Optional[str] = None):
        """Print text with wrapping and optional color."""
        wrapped = self.wrap_text(text)
        if color_code:
            print(self.colorize(wrapped, color_code))
        else:
            print(wrapped)
    
    def create_border(self, char: str = None, width: int = None) -> str:
        """Create a border line."""
        char = char or self.config["border_char"]
        width = width or self.config["line_width"]
        return char * width
    
    def create_separator(self, char: str = None, width: int = None) -> str:
        """Create a separator line."""
        char = char or self.config["separator_char"]
        width = width or self.config["line_width"]
        return char * width
    
    def format_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime(self.config["timestamp_format"])
    
    def format_date(self) -> str:
        """Get formatted date."""
        return date.today().strftime(self.config["date_format"])
    
    def calculate_statistics(self, data: List[Dict[str, Any]], 
                           field: str, anomalous_only: bool = False) -> Dict[str, Any]:
        """
        Calculate statistics for a specific field.
        
        Args:
            data: List of NEO data dictionaries
            field: Field name to analyze
            anomalous_only: Whether to filter for anomalous NEOs only
            
        Returns:
            Dictionary of statistics
        """
        if anomalous_only:
            filtered_data = [
                item for item in data 
                if item.get("ai_validated_anomaly", False) or 
                   item.get("category") == "ISO Candidate" or 
                   item.get("delta_v_anomaly_score", 0) > 1.5
            ]
        else:
            filtered_data = data
        
        values = [
            item.get(field) for item in filtered_data 
            if item.get(field) is not None and item.get(field) != 0
        ]
        
        if not values:
            return {
                "count": 0,
                "mean": 0,
                "min": 0,
                "max": 0,
                "total": len(filtered_data)
            }
        
        if HAS_NUMPY:
            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
                "total": len(filtered_data)
            }
        else:
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "total": len(filtered_data)
            }
    
    def categorize_by_dynamic_tas(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Categorize NEOs by Dynamic TAS values.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Dictionary of category counts
        """
        categories = {}
        
        for neo in data:
            dyn_tas = neo.get("Dynamic TAS") or neo.get("dynamic_tas", 0)
            dyn_cat = neo.get("Dynamic Category") or neo.get("dynamic_category")
            
            if dyn_cat and str(dyn_cat).lower() != "unknown":
                category = str(dyn_cat)
            elif dyn_tas is not None and dyn_tas != 0:
                if dyn_tas < 0.5:
                    category = "Within Normal Range"
                elif dyn_tas < 1.0:
                    category = "Slightly Anomalous"
                elif dyn_tas < 2.0:
                    category = "Moderately Anomalous"
                elif dyn_tas < 3.0:
                    category = "Highly Anomalous"
                else:
                    category = "Extremely Anomalous / Potentially Artificial"
            else:
                category = "Uncategorized"
            
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def verify_anomaly_mechanics(self, anomaly: Dict[str, Any]) -> bool:
        """
        Verify if an anomaly meets verification criteria.
        
        An anomaly is considered verified if its anomaly confidence > 10,
        indicating significant deviation from expected values.
        
        Args:
            anomaly: NEO data dictionary
            
        Returns:
            True if anomaly is verified, False otherwise
        """
        return anomaly.get("anomaly_confidence", 0) > 10
    
    def rank_mission_priorities(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank NEOs by mission priority based on anomaly scores and characteristics.
        
        Args:
            data: List of NEO data dictionaries
            
        Returns:
            Sorted list of NEOs by priority score
        """
        for neo in data:
            # Calculate priority score
            delta_v_score = neo.get("delta_v_anomaly_score", 0) or neo.get("anomaly_confidence", 0)
            orbital_score = 1 if neo.get("orbital_deviation_flag", False) else 0
            tisserand_score = 0  # Placeholder for future implementation
            
            priority_score = (
                delta_v_score * 2 +
                tisserand_score * 1.5 +
                orbital_score
            )
            
            neo["priority_score"] = priority_score
        
        return sorted(data, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    def generate_summary_report(self, data: List[Dict[str, Any]], 
                              output_file: Optional[str] = None,
                              anomalous_only: bool = True) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            anomalous_only: Whether to focus on anomalous NEOs only
            
        Returns:
            Report content as string
        """
        # Filter data if anomalous only
        if anomalous_only:
            filtered_data = [
                item for item in data 
                if item.get("ai_validated_anomaly", False) or 
                   item.get("category") == "ISO Candidate" or 
                   item.get("delta_v_anomaly_score", 0) > 1.5
            ]
            report_title = "aNEO ANOMALOUS ANALYSIS SUMMARY"
            data_description = "Anomalous NEOs Only"
        else:
            filtered_data = data
            report_title = "aNEO COMPREHENSIVE ANALYSIS SUMMARY"
            data_description = "All NEOs"
        
        # Calculate statistics
        raw_tas_stats = self.calculate_statistics(filtered_data, "Raw TAS")
        dynamic_tas_stats = self.calculate_statistics(filtered_data, "Dynamic TAS")
        category_counts = self.categorize_by_dynamic_tas(filtered_data)
        
        # Rank by priority
        ranked_data = self.rank_mission_priorities(filtered_data.copy())
        
        # Build report
        border = self.create_border()
        separator = self.create_separator()
        timestamp = self.format_timestamp()
        
        report_lines = [
            border,
            f"🚀 {report_title} 🚀".center(len(border)),
            border,
            f"Generated: {timestamp}",
            f"Data Scope: {data_description}",
            "",
            "EXECUTIVE SUMMARY:",
            separator,
            f"Total NEOs Analyzed: {raw_tas_stats['total']}",
            f"NEOs with Raw TAS Data: {raw_tas_stats['count']}",
            f"NEOs with Dynamic TAS Data: {dynamic_tas_stats['count']}",
            "",
            "RAW TAS STATISTICS:",
            f"  Average: {raw_tas_stats['mean']:.2f}",
            f"  Highest: {raw_tas_stats['max']:.2f}",
            f"  Lowest: {raw_tas_stats['min']:.2f}",
            "",
            "DYNAMIC TAS STATISTICS:",
            f"  Average: {dynamic_tas_stats['mean']:.2f}",
            f"  Highest: {dynamic_tas_stats['max']:.2f}",
            f"  Lowest: {dynamic_tas_stats['min']:.2f}",
            "",
            "DYNAMIC CATEGORY DISTRIBUTION:",
            separator
        ]
        
        # Add category counts
        for category, count in sorted(category_counts.items()):
            report_lines.append(f"  {category}: {count}")
        
        # Add top 10 priority targets
        report_lines.extend([
            "",
            "TOP 10 MISSION PRIORITY TARGETS:",
            separator
        ])
        
        for i, neo in enumerate(ranked_data[:10], 1):
            designation = neo.get("Designation", "N/A")
            priority = neo.get("priority_score", 0)
            verified = "[Verified]" if self.verify_anomaly_mechanics(neo) else "[Unverified]"
            report_lines.append(f"  {i:2}. {designation} - Priority: {priority:.2f} {verified}")
        
        # Add verification explanation
        report_lines.extend([
            "",
            "VERIFICATION STATUS EXPLANATION:",
            separator,
            "  [Verified]   : Anomaly confidence > 10 (significant deviation)",
            "  [Unverified] : Anomaly confidence ≤ 10 (requires further review)",
            "",
            border
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            self.logger.info(f"Summary report saved to {output_path}")
        
        return report_content
    
    def generate_detailed_report(self, data: List[Dict[str, Any]], 
                               output_file: Optional[str] = None,
                               anomalous_only: bool = True) -> str:
        """
        Generate a detailed report with full NEO information.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            anomalous_only: Whether to focus on anomalous NEOs only
            
        Returns:
            Report content as string
        """
        # Filter data if anomalous only
        if anomalous_only:
            filtered_data = [
                item for item in data 
                if item.get("ai_validated_anomaly", False) or 
                   item.get("category") == "ISO Candidate" or 
                   item.get("delta_v_anomaly_score", 0) > 1.5
            ]
            report_title = "aNEO DETAILED ANALYSIS REPORT (Anomalous NEOs)"
        else:
            filtered_data = data
            report_title = "aNEO DETAILED ANALYSIS REPORT (All NEOs)"
        
        # Build report
        border = self.create_border()
        separator = self.create_separator()
        timestamp = self.format_timestamp()
        
        report_lines = [
            border,
            f"📝 {report_title} 📝".center(len(border)),
            border,
            f"Generated: {timestamp}",
            f"Total Entries: {len(filtered_data)}",
            "",
        ]
        
        # Add detailed information for each NEO
        for i, neo in enumerate(filtered_data, 1):
            report_lines.extend([
                f"NEO #{i}: {neo.get('Designation', 'N/A')}",
                separator
            ])
            
            # Core identification
            report_lines.append("IDENTIFICATION:")
            report_lines.append(f"  Designation: {neo.get('Designation', 'N/A')}")
            
            # Observation period
            if neo.get("Observation Start") and neo.get("Observation End"):
                report_lines.append(f"  Observation Period: {neo['Observation Start']} to {neo['Observation End']}")
            
            # TAS values
            report_lines.append("THREAT ASSESSMENT:")
            report_lines.append(f"  Raw TAS: {neo.get('Raw TAS', 'N/A')}")
            report_lines.append(f"  Dynamic TAS: {neo.get('Dynamic TAS', 'N/A')}")
            
            # Classification
            report_lines.append("CLASSIFICATION:")
            report_lines.append(f"  Previous: {neo.get('previous_classification', 'N/A')}")
            report_lines.append(f"  Current: {neo.get('category', neo.get('dynamic_category', 'N/A'))}")
            
            # Anomaly information
            if neo.get("ai_validated_anomaly", False):
                report_lines.append("ANOMALY ANALYSIS:")
                report_lines.append(f"  Confidence: {neo.get('anomaly_confidence', 0):.2f}")
                report_lines.append(f"  Delta-V Score: {neo.get('delta_v_anomaly_score', 0):.2f}")
                report_lines.append(f"  Expected Delta-V: {neo.get('expected_delta_v', 0):.2f}")
                verified = "[Verified]" if self.verify_anomaly_mechanics(neo) else "[Unverified]"
                report_lines.append(f"  Status: {verified}")
            
            # Mission priority
            if neo.get("priority_score"):
                report_lines.append("MISSION PRIORITY:")
                report_lines.append(f"  Priority Score: {neo.get('priority_score', 0):.2f}")
            
            # Reclassification reasons
            if neo.get("reclassification_reasons"):
                report_lines.append("RECLASSIFICATION:")
                reasons = neo.get("reclassification_reasons", "").split("; ")
                for reason in reasons:
                    if reason.strip():
                        report_lines.append(f"  - {reason.strip()}")
            
            # Additional data
            report_lines.append("ADDITIONAL DATA:")
            excluded_keys = {
                'Designation', 'Observation Start', 'Observation End', 
                'Raw TAS', 'Dynamic TAS', 'previous_classification', 
                'category', 'dynamic_category', 'ai_validated_anomaly',
                'anomaly_confidence', 'delta_v_anomaly_score', 
                'expected_delta_v', 'priority_score', 'reclassification_reasons'
            }
            
            for key, value in neo.items():
                if key not in excluded_keys and value is not None:
                    report_lines.append(f"  {key}: {value}")
            
            report_lines.append("")
        
        report_lines.append(border)
        report_content = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            self.logger.info(f"Detailed report saved to {output_path}")
        
        return report_content
    
    def generate_priority_report(self, data: List[Dict[str, Any]], 
                               output_file: Optional[str] = None) -> str:
        """
        Generate a mission priority report.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            
        Returns:
            Report content as string
        """
        # Filter for priority targets (anomalous NEOs)
        filtered_data = [
            item for item in data 
            if item.get("ai_validated_anomaly", False) or 
               item.get("category") == "ISO Candidate" or 
               item.get("delta_v_anomaly_score", 0) > 1.5
        ]
        
        # Rank by priority
        ranked_data = self.rank_mission_priorities(filtered_data)
        
        # Build report
        border = self.create_border()
        separator = self.create_separator()
        timestamp = self.format_timestamp()
        
        report_lines = [
            border,
            "🚀 MISSION PRIORITY TARGET REPORT 🚀".center(len(border)),
            border,
            f"Generated: {timestamp}",
            f"Priority Targets Identified: {len(ranked_data)}",
            "",
            "MISSION PRIORITY RANKING:",
            separator,
            f"{'Rank':<6} {'Designation':<15} {'Priority':<10} {'Category':<25} {'Status':<12}",
            separator
        ]
        
        # Add ranked entries
        for i, neo in enumerate(ranked_data, 1):
            designation = neo.get("Designation", "N/A")[:14]
            priority = f"{neo.get('priority_score', 0):.2f}"
            category = (neo.get('category') or neo.get('dynamic_category', 'N/A'))[:24]
            verified = "[Verified]" if self.verify_anomaly_mechanics(neo) else "[Unverified]"
            
            report_lines.append(f"{i:<6} {designation:<15} {priority:<10} {category:<25} {verified:<12}")
        
        report_lines.extend([
            separator,
            "",
            "PRIORITY SCORING METHODOLOGY:",
            "  Priority Score = (Delta-V Anomaly Score × 2) + (Tisserand Score × 1.5) + Orbital Score",
            "",
            "VERIFICATION CRITERIA:",
            "  [Verified]   : Anomaly confidence > 10 (high significance)",
            "  [Unverified] : Anomaly confidence ≤ 10 (requires validation)",
            "",
            border
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            self.logger.info(f"Priority report saved to {output_path}")
        
        return report_content
    
    def generate_anomaly_report(self, data: List[Dict[str, Any]], 
                              output_file: Optional[str] = None) -> str:
        """
        Generate a detailed anomaly analysis report.
        
        Args:
            data: List of NEO data dictionaries
            output_file: Optional output file path
            
        Returns:
            Report content as string
        """
        # Filter for anomalous NEOs only
        anomalous_data = [
            item for item in data 
            if item.get("ai_validated_anomaly", False)
        ]
        
        # Sort by anomaly confidence
        sorted_data = sorted(
            anomalous_data, 
            key=lambda x: x.get("anomaly_confidence", 0), 
            reverse=True
        )
        
        # Build report
        border = self.create_border()
        separator = self.create_separator()
        timestamp = self.format_timestamp()
        
        report_lines = [
            border,
            "🔥 DELTA-V ANOMALY ANALYSIS REPORT 🔥".center(len(border)),
            border,
            f"Generated: {timestamp}",
            f"Validated Anomalies: {len(sorted_data)}",
            "",
            "ANOMALY ANALYSIS SUMMARY:",
            separator
        ]
        
        if sorted_data:
            # Calculate anomaly statistics
            confidences = [neo.get("anomaly_confidence", 0) for neo in sorted_data]
            if HAS_NUMPY:
                report_lines.extend([
                    f"  Average Confidence: {np.mean(confidences):.2f}",
                    f"  Highest Confidence: {np.max(confidences):.2f}",
                    f"  Lowest Confidence: {np.min(confidences):.2f}",
                    f"  Standard Deviation: {np.std(confidences):.2f}"
                ])
            else:
                report_lines.extend([
                    f"  Average Confidence: {sum(confidences) / len(confidences):.2f}",
                    f"  Highest Confidence: {max(confidences):.2f}",
                    f"  Lowest Confidence: {min(confidences):.2f}"
                ])
            
            # Verification status breakdown
            verified_count = sum(1 for neo in sorted_data if self.verify_anomaly_mechanics(neo))
            unverified_count = len(sorted_data) - verified_count
            
            report_lines.extend([
                "",
                "VERIFICATION STATUS:",
                f"  Verified Anomalies: {verified_count}",
                f"  Unverified Anomalies: {unverified_count}",
                "",
                "DETAILED ANOMALY LISTING:",
                separator,
                f"{'Designation':<15} {'ΔV':<10} {'Expected':<10} {'Confidence':<12} {'Status':<12}",
                separator
            ])
            
            # Add detailed anomaly entries
            for neo in sorted_data:
                designation = neo.get("Designation", "N/A")[:14]
                delta_v = f"{neo.get('delta_v', 0):.2f}"
                expected = f"{neo.get('expected_delta_v', 0):.2f}"
                confidence = f"{neo.get('anomaly_confidence', 0):.2f}"
                verified = "[Verified]" if self.verify_anomaly_mechanics(neo) else "[Unverified]"
                
                report_lines.append(f"{designation:<15} {delta_v:<10} {expected:<10} {confidence:<12} {verified:<12}")
        
        else:
            report_lines.append("  No validated anomalies detected in dataset.")
        
        report_lines.extend([
            separator,
            "",
            "ANOMALY DETECTION METHODOLOGY:",
            "  - AI-driven orbital anomaly validation using Random Forest regression",
            "  - Comparison of observed vs. expected Delta-V values",
            "  - Dynamic threshold adjustment based on dataset characteristics",
            "  - Slingshot effect filtering to reduce false positives",
            "",
            "CONFIDENCE SCORING:",
            "  Confidence = |observed_ΔV - expected_ΔV| / (expected_ΔV + ε)",
            "  Verification threshold: confidence > 10",
            "",
            border
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_path = self.output_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            self.logger.info(f"Anomaly report saved to {output_path}")
        
        return report_content
    
    def generate_all_reports(self, data: List[Dict[str, Any]], 
                           base_filename: Optional[str] = None) -> Dict[str, str]:
        """
        Generate all report types and return their contents.
        
        Args:
            data: List of NEO data dictionaries
            base_filename: Base filename for output files
            
        Returns:
            Dictionary mapping report types to their content
        """
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"aneos_report_{timestamp}"
        
        reports = {}
        
        # Generate each report type
        reports["summary"] = self.generate_summary_report(
            data, f"{base_filename}_summary.txt"
        )
        
        reports["detailed"] = self.generate_detailed_report(
            data, f"{base_filename}_detailed.txt"
        )
        
        reports["priority"] = self.generate_priority_report(
            data, f"{base_filename}_priority.txt"
        )
        
        reports["anomaly"] = self.generate_anomaly_report(
            data, f"{base_filename}_anomaly.txt"
        )
        
        self.logger.info(f"All reports generated with base filename: {base_filename}")
        
        return reports


class ConsoleReporter:
    """
    Enhanced console reporting with rich formatting and progress tracking.
    """
    
    def __init__(self):
        """Initialize console reporter with rich formatting if available."""
        self.has_rich = HAS_RICH
        self.console = Console() if HAS_RICH else None
        
        # Color codes for fallback formatting
        self.colors = {
            "header": "35",    # Magenta
            "title": "33",     # Yellow
            "success": "32",   # Green
            "warning": "31",   # Red
            "info": "36",      # Cyan
            "data": "34",      # Blue
            "emphasis": "1"    # Bold
        }
    
    def colorize(self, text: str, color_code: str) -> str:
        """Apply ANSI color codes to text for fallback formatting."""
        return f"\033[{color_code}m{text}\033[0m"
    
    def print_beautified_console_summary(self, data: List[Dict[str, Any]], 
                                       statistics: Dict[str, Any],
                                       categories: Dict[str, int]) -> None:
        """
        Print a beautified console summary with rich formatting.
        
        Args:
            data: List of NEO data dictionaries
            statistics: Statistics dictionary
            categories: Category counts dictionary
        """
        # Filter for anomalous NEOs
        anomalous_data = [
            item for item in data 
            if item.get("ai_validated_anomaly", False) or 
               item.get("category") == "ISO Candidate" or 
               item.get("delta_v_anomaly_score", 0) > 1.5
        ]
        
        if self.has_rich and self.console:
            self._print_rich_summary(anomalous_data, statistics, categories)
        else:
            self._print_fallback_summary(anomalous_data, statistics, categories)
    
    def _print_rich_summary(self, data: List[Dict[str, Any]], 
                          statistics: Dict[str, Any],
                          categories: Dict[str, int]) -> None:
        """Print summary using Rich formatting."""
        from rich.table import Table
        from rich.panel import Panel
        
        # Main title panel
        title_panel = Panel(
            "🚀 FINAL ANALYSIS SUMMARY (Anomalous NEOs Only) 🚀",
            style="bold magenta",
            expand=False
        )
        self.console.print(title_panel)
        
        # Statistics table
        stats_table = Table(title="Analysis Statistics", style="cyan")
        stats_table.add_column("Metric", style="bold blue")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Anomalous NEOs", str(len(data)))
        stats_table.add_row("Average Raw TAS", f"{statistics.get('raw_tas_mean', 0):.2f}")
        stats_table.add_row("Highest Raw TAS", f"{statistics.get('raw_tas_max', 0):.2f}")
        stats_table.add_row("Lowest Raw TAS", f"{statistics.get('raw_tas_min', 0):.2f}")
        stats_table.add_row("Average Dynamic TAS", f"{statistics.get('dynamic_tas_mean', 0):.2f}")
        stats_table.add_row("Highest Dynamic TAS", f"{statistics.get('dynamic_tas_max', 0):.2f}")
        stats_table.add_row("Lowest Dynamic TAS", f"{statistics.get('dynamic_tas_min', 0):.2f}")
        
        self.console.print(stats_table)
        
        # Category distribution table
        if categories:
            cat_table = Table(title="Dynamic Category Distribution", style="yellow")
            cat_table.add_column("Category", style="bold blue")
            cat_table.add_column("Count", style="green")
            
            for category, count in sorted(categories.items()):
                cat_table.add_row(category, str(count))
            
            self.console.print(cat_table)
        
        # Top priority targets
        ranked_data = sorted(data, key=lambda x: x.get("priority_score", 0), reverse=True)
        if ranked_data:
            priority_table = Table(title="Top 10 Mission Priority Targets", style="red")
            priority_table.add_column("Rank", style="bold")
            priority_table.add_column("Designation", style="blue")
            priority_table.add_column("Priority Score", style="green")
            priority_table.add_column("Status", style="yellow")
            
            for i, neo in enumerate(ranked_data[:10], 1):
                designation = neo.get("Designation", "N/A")
                priority = f"{neo.get('priority_score', 0):.2f}"
                verified = "[Verified]" if neo.get("anomaly_confidence", 0) > 10 else "[Unverified]"
                
                priority_table.add_row(str(i), designation, priority, verified)
            
            self.console.print(priority_table)
        
        # Status explanation panel
        explanation = Panel(
            """[bold]Status Explanation:[/bold]
[green][Verified][/green]   : Anomaly confidence > 10 (significant deviation from expected ΔV)
[yellow][Unverified][/yellow] : Anomaly confidence ≤ 10 (requires further review)""",
            title="Verification Criteria",
            style="blue"
        )
        self.console.print(explanation)
    
    def _print_fallback_summary(self, data: List[Dict[str, Any]], 
                              statistics: Dict[str, Any],
                              categories: Dict[str, int]) -> None:
        """Print summary using fallback ANSI formatting."""
        border = "=" * 60
        separator = "-" * 60
        
        print(self.colorize(f"\n{border}", self.colors["header"]))
        print(self.colorize("🚀 FINAL ANALYSIS SUMMARY (Anomalous NEOs Only) 🚀".center(60), self.colors["title"]))
        print(self.colorize(f"{border}", self.colors["header"]))
        
        # Statistics
        print(self.colorize(f"Total Anomalous NEOs: {len(data)} 🚀", self.colors["success"]))
        print(self.colorize(f"Average Raw TAS: {statistics.get('raw_tas_mean', 0):.2f}", self.colors["success"]))
        print(self.colorize(f"Highest Raw TAS: {statistics.get('raw_tas_max', 0):.2f}", self.colors["success"]))
        print(self.colorize(f"Lowest Raw TAS: {statistics.get('raw_tas_min', 0):.2f}", self.colors["success"]))
        
        print(self.colorize("\nDynamic Analysis:", self.colors["info"]))
        print(self.colorize(f"Highest Dynamic TAS: {statistics.get('dynamic_tas_max', 'N/A')}", self.colors["info"]))
        print(self.colorize(f"Lowest Dynamic TAS: {statistics.get('dynamic_tas_min', 'N/A')}", self.colors["info"]))
        
        # Categories
        if categories:
            print(self.colorize("\nDynamic Category Counts:", self.colors["info"]))
            for category, count in sorted(categories.items()):
                print(self.colorize(f"  {category}: {count}", self.colors["info"]))
        
        # Top priorities
        ranked_data = sorted(data, key=lambda x: x.get("priority_score", 0), reverse=True)
        if ranked_data:
            print(self.colorize("\nTop 10 Mission Priority Anomalous NEOs:", self.colors["title"]))
            for i, neo in enumerate(ranked_data[:10], 1):
                designation = neo.get("Designation", "N/A")
                priority = neo.get("priority_score", 0)
                verified = "[Verified]" if neo.get("anomaly_confidence", 0) > 10 else "[Unverified]"
                print(self.colorize(f"  {i:2}. {designation} - Priority Score: {priority:.2f} {verified}", self.colors["success"]))
        
        # Status explanation
        print(self.colorize("\nStatus Explanation:", self.colors["data"]))
        print(self.colorize("  [Verified]   : Anomaly confidence > 10 (significant deviation)", self.colors["data"]))
        print(self.colorize("  [Unverified] : Anomaly confidence ≤ 10 (requires further review)", self.colors["data"]))
        print(self.colorize(f"{border}\n", self.colors["header"]))
    
    def print_progress_update(self, message: str, step: int = None, total: int = None) -> None:
        """
        Print a progress update message.
        
        Args:
            message: Progress message
            step: Current step number (optional)
            total: Total steps (optional)
        """
        if step is not None and total is not None:
            progress_text = f"[{step}/{total}] {message}"
        else:
            progress_text = message
        
        if self.has_rich and self.console:
            self.console.print(f"[blue]INFO[/blue] {progress_text}")
        else:
            print(self.colorize(f"INFO: {progress_text}", self.colors["info"]))
    
    def print_success(self, message: str) -> None:
        """Print a success message."""
        if self.has_rich and self.console:
            self.console.print(f"[green]✅ {message}[/green]")
        else:
            print(self.colorize(f"✅ {message}", self.colors["success"]))
    
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        if self.has_rich and self.console:
            self.console.print(f"[yellow]⚠️  {message}[/yellow]")
        else:
            print(self.colorize(f"⚠️  {message}", self.colors["warning"]))
    
    def print_error(self, message: str) -> None:
        """Print an error message."""
        if self.has_rich and self.console:
            self.console.print(f"[red]❌ {message}[/red]")
        else:
            print(self.colorize(f"❌ {message}", self.colors["warning"]))
    
    def print_anomaly_summary(self, data: List[Dict[str, Any]], top_n: int = 10) -> None:
        """
        Print top N anomaly report with professional formatting.
        
        Replicates the functionality from legacy reporting_neos_ng_v3.0.py
        with enhanced formatting and verification status.
        
        Args:
            data: List of NEO data dictionaries
            top_n: Number of top anomalies to display
        """
        # Filter for validated anomalies
        anomalous_data = [
            item for item in data 
            if item.get("ai_validated_anomaly", False)
        ]
        
        # Sort by anomaly confidence
        sorted_data = sorted(
            anomalous_data, 
            key=lambda x: x.get("anomaly_confidence", 0), 
            reverse=True
        )[:top_n]
        
        if self.has_rich and self.console:
            self._print_rich_anomaly_summary(sorted_data, top_n)
        else:
            self._print_fallback_anomaly_summary(sorted_data, top_n)
    
    def _print_rich_anomaly_summary(self, data: List[Dict[str, Any]], top_n: int) -> None:
        """Print anomaly summary using Rich formatting."""
        from rich.table import Table
        from rich.panel import Panel
        
        # Title panel
        title_panel = Panel(
            f"🔥 TOP {top_n} ΔV ANOMALY REPORT (Anomalous NEOs Only) 🔥",
            style="bold red",
            expand=False
        )
        self.console.print(title_panel)
        
        if data:
            # Create anomaly table
            anomaly_table = Table(title="Validated Anomalies", style="red")
            anomaly_table.add_column("Designation", style="bold blue", width=15)
            anomaly_table.add_column("ΔV (km/s)", style="green", width=10)
            anomaly_table.add_column("Expected ΔV", style="cyan", width=12)
            anomaly_table.add_column("Confidence", style="yellow", width=12)
            anomaly_table.add_column("Category", style="magenta", width=20)
            anomaly_table.add_column("Status", style="red", width=12)
            
            for neo in data:
                designation = neo.get("Designation", "N/A")[:14]
                delta_v = f"{neo.get('delta_v', 0):.2f}"
                expected = f"{neo.get('expected_delta_v', 0):.2f}"
                confidence = f"{neo.get('anomaly_confidence', 0):.2f}"
                category = (neo.get('category') or neo.get('dynamic_category', 'N/A'))[:19]
                verified = "[Verified]" if neo.get("anomaly_confidence", 0) > 10 else "[Unverified]"
                
                anomaly_table.add_row(designation, delta_v, expected, confidence, category, verified)
            
            self.console.print(anomaly_table)
        else:
            self.console.print("[yellow]No validated anomalies found in dataset[/yellow]")
    
    def _print_fallback_anomaly_summary(self, data: List[Dict[str, Any]], top_n: int) -> None:
        """Print anomaly summary using fallback ANSI formatting."""
        border = "=" * 60
        separator = "-" * 60
        
        print(self.colorize(border, self.colors["header"]))
        print(self.colorize(f"🔥 TOP {top_n} ΔV ANOMALY REPORT (Anomalous NEOs Only) 🔥".center(60), self.colors["title"]))
        print(self.colorize(border, self.colors["header"]))
        
        if data:
            print(self.colorize("DESIGNATION  |  ΔV (km/s)  |  Expected ΔV  |  Anomaly Confidence  |  Category", self.colors["info"]))
            print(separator)
            
            for neo in data:
                designation = neo.get("Designation", "N/A")[:12]
                delta_v = neo.get("delta_v", 0)
                expected = neo.get("expected_delta_v", 0)
                confidence = neo.get("anomaly_confidence", 0)
                category = (neo.get('category') or neo.get('dynamic_category', 'N/A'))[:15]
                verified = "[Verified]" if confidence > 10 else "[Unverified]"
                
                line = (f"{designation:<12} | "
                        f"{delta_v:9.2f} | "
                        f"{expected:11.2f} | "
                        f"{confidence:18.2f} | "
                        f"{category:<15} {verified}")
                print(self.colorize(line, self.colors["warning"]))
        else:
            print(self.colorize("No validated anomalies found in dataset.", self.colors["info"]))
        
        print(self.colorize(border, self.colors["header"]))
    
    def print_introduction(self) -> None:
        """
        Print professional introduction banner for the aNEOS system.
        
        Replicates the introduction from legacy reporting_neos_ng_v3.0.py
        with enhanced formatting.
        """
        intro_text = (
            "Welcome to the aNEOS Reporting System v4.0 - Enhanced Integration.\n\n"
            "This professional reporting system analyzes Near Earth Object (NEO) data by:\n"
            "  1. Loading and enriching multi-source NEO data\n"
            "  2. Applying AI-driven anomaly detection and validation\n"
            "  3. Performing comprehensive orbital mechanics analysis\n"
            "  4. Categorizing objects with mission priority ranking\n"
            "  5. Generating professional reports with academic rigor\n"
            "  6. Providing rich console output with verification status\n\n"
            "Verification Status:\n"
            "  [Verified]   : Anomaly confidence > 10 (significant deviation)\n"
            "  [Unverified] : Anomaly confidence ≤ 10 (requires further review)\n\n"
            "Ready for professional NEO analysis and reporting."
        )
        
        if self.has_rich and self.console:
            from rich.panel import Panel
            intro_panel = Panel(
                intro_text,
                title="aNEOS Professional Reporting System",
                style="bold cyan",
                expand=False
            )
            self.console.print(intro_panel)
        else:
            terminal_width = 80
            wrapped_text = ""
            for line in intro_text.split('\n'):
                if line.strip():
                    wrapped_text += self._wrap_text_simple(line, terminal_width) + '\n'
                else:
                    wrapped_text += '\n'
            
            print(self.colorize("-" * terminal_width, self.colors["info"]))
            print(self.colorize(wrapped_text, self.colors["info"]))
            print(self.colorize("-" * terminal_width, self.colors["info"]))
    
    def _wrap_text_simple(self, text: str, width: int) -> str:
        """Simple text wrapping for fallback formatting."""
        if len(text) <= width:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)