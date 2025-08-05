#!/usr/bin/env python3
"""
Professional Reporting Suite for aNEOS Core

Comprehensive professional reporting system that integrates all reporting capabilities
from legacy reporting_neos_ng_v3.0.py with enhanced features, AI validation, and
academic rigor suitable for scientific analysis.
"""

import os
import json
import time
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from .generators import ReportGenerator, ConsoleReporter
from .analytics import NEOClassificationSystem, MissionPriorityCalculator
from .ai_validation import AIAnomalyValidator
from .progress import ProgressTracker, OperationTimer

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProfessionalReportingSuite:
    """
    Complete professional reporting suite for aNEOS analysis.
    
    Integrates all reporting capabilities with AI-driven anomaly validation,
    mission priority ranking, and comprehensive report generation matching
    the quality and academic rigor of the legacy system.
    """
    
    def __init__(self, 
                 output_dir: str = "reports",
                 logger: Optional[logging.Logger] = None,
                 enable_ai_validation: bool = True):
        """
        Initialize professional reporting suite.
        
        Args:
            output_dir: Directory for report output
            logger: Optional logger instance
            enable_ai_validation: Whether to enable AI-driven validation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        self.enable_ai_validation = enable_ai_validation
        
        # Initialize reporting components
        self.report_generator = ReportGenerator(output_dir, logger)
        self.console_reporter = ConsoleReporter()
        self.classification_system = NEOClassificationSystem(logger)
        self.priority_calculator = MissionPriorityCalculator(logger)
        
        # Initialize AI validation if enabled
        if enable_ai_validation:
            self.ai_validator = AIAnomalyValidator(logger)
        else:
            self.ai_validator = None
        
        # Progress tracking
        self.progress_tracker = ProgressTracker()
        self.current_operation = None
        
        # Analysis state
        self.analysis_state = {
            "data_loaded": False,
            "ai_model_trained": False,
            "anomalies_validated": False,
            "classifications_complete": False,
            "priorities_calculated": False,
            "reports_generated": False
        }
        
        # Statistics and metrics
        self.analysis_metrics = {
            "total_objects": 0,
            "anomalous_objects": 0,
            "verified_anomalies": 0,
            "unverified_anomalies": 0,
            "processing_time": 0.0,
            "ai_model_performance": {},
            "category_distribution": {},
            "priority_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        }
    
    def print_introduction(self) -> None:
        """Print professional introduction banner."""
        self.console_reporter.print_introduction()
    
    def process_comprehensive_analysis(self, 
                                     neo_data: List[Dict[str, Any]],
                                     show_progress: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive analysis with professional reporting.
        
        Replicates the 13-step process from legacy reporting_neos_ng_v3.0.py
        with enhanced capabilities and academic rigor.
        
        Args:
            neo_data: List of NEO data dictionaries
            show_progress: Whether to show progress updates
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        if show_progress:
            self.print_introduction()
        
        # Initialize progress tracking (13 steps like legacy system)
        total_steps = 13
        if HAS_TQDM and show_progress:
            meta_pbar = tqdm(total=total_steps, desc="Overall Process", dynamic_ncols=True, position=0, leave=True)
        else:
            meta_pbar = None
        
        try:
            # Step 1: Data Loading and Validation
            if show_progress:
                self.console_reporter.print_progress_update("Loading and validating NEO data...", 1, total_steps)
            validated_data = self._validate_and_prepare_data(neo_data)
            self.analysis_metrics["total_objects"] = len(validated_data)
            self.analysis_state["data_loaded"] = True
            if meta_pbar: meta_pbar.update(1)
            
            # Step 2: Data Enrichment and Quality Assessment
            if show_progress:
                self.console_reporter.print_progress_update("Enriching data and assessing quality...", 2, total_steps)
            enriched_data = self._enrich_and_assess_quality(validated_data)
            if meta_pbar: meta_pbar.update(1)
            
            # Step 3: Handle Incomplete Data
            if show_progress:
                self.console_reporter.print_progress_update("Handling incomplete data...", 3, total_steps)
            processed_data = self._handle_incomplete_data(enriched_data)
            if meta_pbar: meta_pbar.update(1)
            
            # Step 4: Legacy Dynamic Grouping
            if show_progress:
                self.console_reporter.print_progress_update("Applying dynamic grouping and categorization...", 4, total_steps)
            grouped_data = self._apply_dynamic_grouping(processed_data)
            if meta_pbar: meta_pbar.update(1)
            
            # Step 5: Enhanced Classification
            if show_progress:
                self.console_reporter.print_progress_update("Performing comprehensive classification...", 5, total_steps)
            classified_data = self._perform_comprehensive_classification(grouped_data)
            self.analysis_state["classifications_complete"] = True
            if meta_pbar: meta_pbar.update(1)
            
            # Step 6: Statistical Analysis
            if show_progress:
                self.console_reporter.print_progress_update("Calculating comprehensive statistics...", 6, total_steps)
            statistics = self._calculate_comprehensive_statistics(classified_data)
            if meta_pbar: meta_pbar.update(1)
            
            # Step 7: Dynamic Epoch Segmentation
            if show_progress:
                self.console_reporter.print_progress_update("Performing dynamic epoch segmentation...", 7, total_steps)
            segmented_data = self._perform_epoch_segmentation(classified_data)
            if meta_pbar: meta_pbar.update(1)
            
            # Step 8: AI-Based Anomaly Validation
            if show_progress:
                self.console_reporter.print_progress_update("Training AI model and validating anomalies...", 8, total_steps)
            if self.enable_ai_validation and self.ai_validator:
                validated_anomalies = self._train_and_validate_anomalies(segmented_data)
                self.analysis_state["ai_model_trained"] = True
                self.analysis_state["anomalies_validated"] = True
            else:
                validated_anomalies = segmented_data
            if meta_pbar: meta_pbar.update(1)
            
            # Step 9: Filter Anomalous Objects
            if show_progress:
                self.console_reporter.print_progress_update("Filtering and categorizing anomalous objects...", 9, total_steps)
            anomalous_objects = self._filter_anomalous_objects(validated_anomalies)
            self._update_anomaly_metrics(anomalous_objects)
            if meta_pbar: meta_pbar.update(1)
            
            # Step 10: Mission Priority Ranking
            if show_progress:
                self.console_reporter.print_progress_update("Calculating mission priority rankings...", 10, total_steps)
            priority_ranked = self._calculate_mission_priorities(validated_anomalies)
            self.analysis_state["priorities_calculated"] = True
            if meta_pbar: meta_pbar.update(1)
            
            # Step 11: Generate Professional Reports
            if show_progress:
                self.console_reporter.print_progress_update("Generating professional reports...", 11, total_steps)
            reports = self._generate_all_reports(priority_ranked, statistics)
            self.analysis_state["reports_generated"] = True
            if meta_pbar: meta_pbar.update(1)
            
            # Step 12: Generate Visualizations
            if show_progress:
                self.console_reporter.print_progress_update("Creating visualizations and orbital maps...", 12, total_steps)
            visualizations = self._generate_visualizations(priority_ranked)
            if meta_pbar: meta_pbar.update(1)
            
            # Step 13: Display Professional Console Summary
            if show_progress:
                self.console_reporter.print_progress_update("Displaying comprehensive analysis summary...", 13, total_steps)
            self._display_professional_summary(anomalous_objects, statistics)
            if meta_pbar: meta_pbar.update(1)
            
            # Close progress bar
            if meta_pbar:
                meta_pbar.close()
            
            # Calculate final metrics
            end_time = time.time()
            self.analysis_metrics["processing_time"] = end_time - start_time
            
            # Compile comprehensive results
            results = {
                "processed_data": priority_ranked,
                "anomalous_objects": anomalous_objects,
                "statistics": statistics,
                "reports": reports,
                "visualizations": visualizations,
                "analysis_state": self.analysis_state.copy(),
                "analysis_metrics": self.analysis_metrics.copy(),
                "processing_time": self.analysis_metrics["processing_time"]
            }
            
            if show_progress:
                self.console_reporter.print_success(f"Comprehensive analysis completed in {self.analysis_metrics['processing_time']:.1f} seconds")
            
            return results
            
        except Exception as e:
            if meta_pbar:
                meta_pbar.close()
            self.logger.error(f"Error in comprehensive analysis: {e}")
            self.console_reporter.print_error(f"Analysis failed: {e}")
            raise
    
    def _validate_and_prepare_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and prepare input data."""
        if not data:
            raise ValueError("No data provided for analysis")
        
        validated_data = []
        for item in data:
            if isinstance(item, dict) and item.get("Designation"):
                validated_data.append(item.copy())
        
        if not validated_data:
            raise ValueError("No valid NEO data found in input")
        
        self.logger.info(f"Validated {len(validated_data)} NEO objects")
        return validated_data
    
    def _enrich_and_assess_quality(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich data and assess quality metrics."""
        enriched_data = []
        
        for item in data:
            enriched_item = item.copy()
            
            # Assess data completeness
            required_fields = ["Designation", "Raw TAS", "Dynamic TAS", "delta_v"]
            available_fields = sum(1 for field in required_fields if enriched_item.get(field) is not None)
            completeness = available_fields / len(required_fields)
            
            enriched_item["data_completeness"] = completeness
            enriched_item["enrichment_timestamp"] = datetime.now().isoformat()
            
            enriched_data.append(enriched_item)
        
        return enriched_data
    
    def _handle_incomplete_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle incomplete data with professional fallbacks."""
        processed_data = []
        
        for item in data:
            processed_item = item.copy()
            
            # Handle missing string fields
            string_fields = ["Designation", "Observation Start", "Observation End", "Dynamic Category"]
            for field in string_fields:
                if not processed_item.get(field):
                    processed_item[field] = "unknown"
            
            # Handle missing numeric fields
            numeric_fields = ["Raw TAS", "Dynamic TAS", "delta_v", "Close Approaches", 
                            "semi_major_axis", "eccentricity", "inclination"]
            for field in numeric_fields:
                if processed_item.get(field) is None:
                    processed_item[field] = 0
            
            # Mark incomplete data
            critical_fields = ["Designation", "Observation Start", "Observation End"]
            processed_item["incomplete_data"] = any(
                processed_item.get(field) == "unknown" for field in critical_fields
            )
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def _apply_dynamic_grouping(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply dynamic grouping based on TAS values."""
        grouped_data = []
        
        for item in data:
            grouped_item = item.copy()
            
            # Apply legacy dynamic grouping logic
            dyn_tas = item.get("Dynamic TAS", 0)
            dyn_cat = item.get("Dynamic Category", "")
            
            if dyn_cat and str(dyn_cat).lower() != "unknown":
                grouped_item["dynamic_category"] = dyn_cat
            elif dyn_tas is not None and dyn_tas != 0:
                if dyn_tas < 0.5:
                    grouped_item["dynamic_category"] = "Within Normal Range"
                elif dyn_tas < 1.0:
                    grouped_item["dynamic_category"] = "Slightly Anomalous"
                elif dyn_tas < 2.0:
                    grouped_item["dynamic_category"] = "Moderately Anomalous"
                elif dyn_tas < 3.0:
                    grouped_item["dynamic_category"] = "Highly Anomalous"
                else:
                    grouped_item["dynamic_category"] = "Extremely Anomalous / Potentially Artificial"
            else:
                grouped_item["dynamic_category"] = "Uncategorized"
            
            grouped_data.append(grouped_item)
        
        return grouped_data
    
    def _perform_comprehensive_classification(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform comprehensive NEO classification."""
        classified_data = []
        
        for item in data:
            # Get previous classification
            previous_class = item.get("dynamic_category", "Unknown")
            
            # Perform comprehensive classification
            classification_result = self.classification_system.classify_neo_comprehensive(
                item, previous_class
            )
            
            # Merge classification results
            classified_item = item.copy()
            classified_item.update({
                "category": classification_result["primary_category"],
                "previous_classification": classification_result["previous_classification"],
                "classification_confidence": classification_result["classification_confidence"],
                "is_verified_anomaly": classification_result["is_verified_anomaly"],
                "verification_status": classification_result["verification_status"],
                "reclassification_reasons": "; ".join(classification_result["reclassification_reasons"]),
                "orbital_assessment": classification_result["orbital_assessment"],
                "velocity_assessment": classification_result["velocity_assessment"],
                "physical_assessment": classification_result["physical_assessment"]
            })
            
            classified_data.append(classified_item)
        
        return classified_data
    
    def _calculate_comprehensive_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the dataset."""
        statistics = {}
        
        # Basic counts
        statistics["total_objects"] = len(data)
        
        # TAS statistics
        raw_tas_stats = self.report_generator.calculate_statistics(data, "Raw TAS", False)
        dynamic_tas_stats = self.report_generator.calculate_statistics(data, "Dynamic TAS", False)
        
        statistics.update({
            "raw_tas_mean": raw_tas_stats["mean"],
            "raw_tas_max": raw_tas_stats["max"],
            "raw_tas_min": raw_tas_stats["min"],
            "dynamic_tas_mean": dynamic_tas_stats["mean"],
            "dynamic_tas_max": dynamic_tas_stats["max"],
            "dynamic_tas_min": dynamic_tas_stats["min"]
        })
        
        # Category distribution
        category_counts = {}
        for item in data:
            category = item.get("dynamic_category", "Uncategorized")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        statistics["category_distribution"] = category_counts
        self.analysis_metrics["category_distribution"] = category_counts
        
        return statistics
    
    def _perform_epoch_segmentation(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform dynamic epoch segmentation (placeholder for enhanced implementation)."""
        # For now, return data as-is with epoch information
        segmented_data = []
        
        for item in data:
            segmented_item = item.copy()
            segmented_item["epoch_segment"] = "epoch_full"
            segmented_data.append(segmented_item)
        
        return segmented_data
    
    def _train_and_validate_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Train AI model and validate anomalies."""
        if not self.ai_validator:
            return data
        
        # Train the model with spinner (replicating legacy behavior)
        model = self.ai_validator.run_with_spinner(
            self.ai_validator.train_orbital_anomaly_model, data
        )
        
        if model:
            # Validate anomalies
            validated_data = self.ai_validator.validate_orbital_anomalies(data, model)
            
            # Detect and filter slingshot effects
            filtered_data = self.ai_validator.detect_slingshot_effect(validated_data)
            
            # Store model performance
            self.analysis_metrics["ai_model_performance"] = self.ai_validator.get_model_performance_report()
            
            self.console_reporter.print_success("AI-Based ΔV anomaly validation and slingshot filtering complete! 🤖")
            
            return filtered_data
        else:
            self.console_reporter.print_warning("AI model training failed, using fallback validation")
            return self.ai_validator.validate_orbital_anomalies(data)
    
    def _filter_anomalous_objects(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter for anomalous objects only."""
        anomalous_objects = [
            item for item in data
            if (item.get("ai_validated_anomaly", False) or
                item.get("category") == "ISO Candidate" or
                item.get("delta_v_anomaly_score", 0) > 1.5)
        ]
        
        return anomalous_objects
    
    def _update_anomaly_metrics(self, anomalous_objects: List[Dict[str, Any]]) -> None:
        """Update anomaly-related metrics."""
        self.analysis_metrics["anomalous_objects"] = len(anomalous_objects)
        
        verified_count = sum(
            1 for obj in anomalous_objects 
            if obj.get("anomaly_confidence", 0) > 10
        )
        
        self.analysis_metrics["verified_anomalies"] = verified_count
        self.analysis_metrics["unverified_anomalies"] = len(anomalous_objects) - verified_count
    
    def _calculate_mission_priorities(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate comprehensive mission priorities."""
        priority_ranked = self.priority_calculator.rank_neos_by_priority(data)
        
        # Update priority distribution metrics
        for item in priority_ranked:
            tier = item.get("priority_tier", "LOW")
            self.analysis_metrics["priority_distribution"][tier] += 1
        
        return priority_ranked
    
    def _generate_all_reports(self, data: List[Dict[str, Any]], statistics: Dict[str, Any]) -> Dict[str, str]:
        """Generate all professional reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"aneos_professional_report_{timestamp}"
        
        reports = self.report_generator.generate_all_reports(data, base_filename)
        
        self.console_reporter.print_success(f"Professional reports generated: {len(reports)} report types")
        
        return reports
    
    def _generate_visualizations(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate visualizations and orbital maps."""
        # Placeholder for visualization generation
        # This would integrate with the visualizers module
        visualizations = {
            "orbital_map_2d": "placeholder_path",
            "orbital_map_3d": "placeholder_path"
        }
        
        return visualizations
    
    def _display_professional_summary(self, anomalous_objects: List[Dict[str, Any]], 
                                    statistics: Dict[str, Any]) -> None:
        """Display professional console summary."""
        # Display beautified console summary
        self.console_reporter.print_beautified_console_summary(
            anomalous_objects, statistics, statistics.get("category_distribution", {})
        )
        
        # Display anomaly summary
        if anomalous_objects:
            self.console_reporter.print_anomaly_summary(anomalous_objects)
        
        # Display distance analysis (if available)
        if anomalous_objects:
            sorted_by_distance = sorted(
                anomalous_objects,
                key=lambda x: x.get("distance_from_earth", 0),
                reverse=True
            )[:10]
            
            if any(obj.get("distance_from_earth") for obj in sorted_by_distance):
                self.console_reporter.print_progress_update("Top 10 aNEOs by Distance from Earth:")
                for i, neo in enumerate(sorted_by_distance, 1):
                    designation = neo.get("Designation", "N/A")
                    distance = neo.get("distance_from_earth", 0)
                    print(f"  {i:2}. {designation} - Distance: {distance:.2f} AU")


def create_professional_suite(output_dir: str = "reports",
                            logger: Optional[logging.Logger] = None,
                            enable_ai_validation: bool = True) -> ProfessionalReportingSuite:
    """Create a professional reporting suite instance."""
    return ProfessionalReportingSuite(output_dir, logger, enable_ai_validation)