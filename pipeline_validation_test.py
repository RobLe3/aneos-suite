#!/usr/bin/env python3
"""
Pipeline Validation Test - ZETA SWARM: PIPELINE VALIDATION & MENU CLEANUP SPECIALIST

Tests the complete aNEOS pipeline functionality using artificial test signatures
to validate that the system can properly detect and process artificial NEOs
through all stages: First-Stage → Multi-Stage → Expert Review.

This script identifies why the pipeline currently produces ZERO candidates
and provides calibration recommendations for realistic operation.
"""

import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import test signatures
from artificial_test_signatures import generate_complete_test_suite, ArtificialSignature

# Try to import rich for better output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

@dataclass 
class ValidationResult:
    """Result from pipeline validation test."""
    signature: ArtificialSignature
    reached_stage: str
    actual_score: float
    expected_stage: str
    passed: bool
    processing_time: float
    error_message: Optional[str] = None
    stage_details: Dict[str, Any] = None

class PipelineValidator:
    """Validates pipeline functionality with artificial signatures."""
    
    def __init__(self):
        """Initialize validator with pipeline components."""
        self.results: List[ValidationResult] = []
        self.pipeline_available = False
        self.scoring_available = False
        self._check_component_availability()
    
    def _check_component_availability(self):
        """Check which pipeline components are available."""
        try:
            from aneos_core.pipeline.automatic_review_pipeline import AutomaticReviewPipeline
            self.pipeline_available = True
            print("✅ Automatic Review Pipeline available")
        except ImportError as e:
            print(f"❌ Pipeline not available: {e}")
        
        try:
            from aneos_core.analysis.advanced_scoring import AdvancedScoreCalculator
            self.scoring_available = True  
            print("✅ Advanced Scoring System available")
        except ImportError as e:
            print(f"❌ Scoring system not available: {e}")
    
    async def validate_single_signature(self, signature: ArtificialSignature) -> ValidationResult:
        """Validate a single artificial signature through the pipeline."""
        start_time = time.time()
        
        try:
            if not self.pipeline_available:
                return ValidationResult(
                    signature=signature,
                    reached_stage="unavailable", 
                    actual_score=0.0,
                    expected_stage=signature.expected_stage,
                    passed=False,
                    processing_time=0.0,
                    error_message="Pipeline components not available"
                )
            
            # Convert signature to NEO data format
            neo_data = signature.to_neo_data_format()
            
            # Test with XVIII SWARM scoring first
            score = await self._test_xviii_swarm_scoring(neo_data)
            
            # Test full pipeline if scoring passes threshold
            reached_stage = await self._test_full_pipeline(neo_data, score)
            
            processing_time = time.time() - start_time
            passed = self._evaluate_result(signature.expected_stage, reached_stage, score)
            
            return ValidationResult(
                signature=signature,
                reached_stage=reached_stage,
                actual_score=score,
                expected_stage=signature.expected_stage,
                passed=passed,
                processing_time=processing_time,
                stage_details={
                    'xviii_swarm_score': score,
                    'threshold_analysis': self._analyze_thresholds(score)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                signature=signature,
                reached_stage="error",
                actual_score=0.0,
                expected_stage=signature.expected_stage, 
                passed=False,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _test_xviii_swarm_scoring(self, neo_data: Dict[str, Any]) -> float:
        """Test XVIII SWARM scoring system."""
        try:
            if not self.scoring_available:
                return 0.0
                
            from aneos_core.analysis.advanced_scoring import (
                AdvancedScoreCalculator, 
                AdvancedScoringConfig
            )
            
            # Load configuration
            config_path = PROJECT_ROOT / "aneos_core/config/advanced_scoring_weights.json"
            
            # Initialize calculator with config path
            calculator = AdvancedScoreCalculator(config_path if config_path.exists() else None)
            
            # Calculate score
            score_result = await calculator.calculate_score(neo_data)
            
            return score_result.overall_score if score_result else 0.0
            
        except Exception as e:
            print(f"⚠️  XVIII SWARM scoring error: {e}")
            return 0.0
    
    async def _test_full_pipeline(self, neo_data: Dict[str, Any], score: float) -> str:
        """Test object through full pipeline to see how far it gets."""
        try:
            from aneos_core.pipeline.automatic_review_pipeline import (
                AutomaticReviewPipeline,
                PipelineConfig
            )
            
            # Create pipeline with current configuration
            config = PipelineConfig()
            pipeline = AutomaticReviewPipeline(config)
            
            # Test against first-stage threshold
            if score < config.first_stage.score_threshold:
                return "filtered"
            
            # Passed first stage
            if score < config.multi_stage.score_threshold:
                return "first_stage"
            
            # Passed multi-stage
            if score < config.expert_review.score_threshold:
                return "multi_stage"
            
            # Reached expert review
            return "expert_review"
            
        except Exception as e:
            print(f"⚠️  Pipeline test error: {e}")
            return "error"
    
    def _evaluate_result(self, expected: str, actual: str, score: float) -> bool:
        """Evaluate if the result matches expectations."""
        stage_hierarchy = {
            "filtered": 0,
            "first_stage": 1, 
            "multi_stage": 2,
            "expert_review": 3
        }
        
        expected_level = stage_hierarchy.get(expected, 0)
        actual_level = stage_hierarchy.get(actual, 0)
        
        # Pass if reached expected stage or higher
        return actual_level >= expected_level
    
    def _analyze_thresholds(self, score: float) -> Dict[str, Any]:
        """Analyze score against current thresholds."""
        try:
            from aneos_core.pipeline.automatic_review_pipeline import PipelineConfig
            config = PipelineConfig()
            
            return {
                'score': score,
                'first_stage_threshold': config.first_stage.score_threshold,
                'multi_stage_threshold': config.multi_stage.score_threshold,
                'expert_review_threshold': config.expert_review.score_threshold,
                'passes_first_stage': score >= config.first_stage.score_threshold,
                'passes_multi_stage': score >= config.multi_stage.score_threshold,
                'passes_expert_review': score >= config.expert_review.score_threshold
            }
        except:
            return {'score': score, 'thresholds_unavailable': True}
    
    async def run_validation(self, signatures: List[ArtificialSignature]) -> List[ValidationResult]:
        """Run validation on all signatures."""
        if console:
            console.print("\n🔬 [bold]PIPELINE VALIDATION TEST[/bold]")
            console.print("=" * 60)
        else:
            print("\n🔬 PIPELINE VALIDATION TEST")
            print("=" * 60)
        
        results = []
        
        for i, signature in enumerate(signatures):
            if console:
                console.print(f"\n[cyan]Testing {signature.designation}[/cyan] ({i+1}/{len(signatures)})")
                console.print(f"Category: [yellow]{signature.category}[/yellow], Expected: [green]{signature.expected_stage}[/green]")
            else:
                print(f"\nTesting {signature.designation} ({i+1}/{len(signatures)})")
                print(f"Category: {signature.category}, Expected: {signature.expected_stage}")
            
            result = await self.validate_single_signature(signature)
            results.append(result)
            
            # Show immediate result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            if console:
                console.print(f"Result: [{'green' if result.passed else 'red'}]{status}[/] - "
                             f"Reached: [blue]{result.reached_stage}[/blue], Score: [yellow]{result.actual_score:.3f}[/yellow]")
            else:
                print(f"Result: {status} - Reached: {result.reached_stage}, Score: {result.actual_score:.3f}")
        
        self.results = results
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.results:
            return {"error": "No validation results available"}
        
        # Calculate statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        # Category breakdown
        category_stats = {}
        stage_stats = {}
        
        for result in self.results:
            category = result.signature.category
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'passed': 0}
            category_stats[category]['total'] += 1
            if result.passed:
                category_stats[category]['passed'] += 1
            
            stage = result.reached_stage
            if stage not in stage_stats:
                stage_stats[stage] = 0
            stage_stats[stage] += 1
        
        # Score analysis
        scores = [r.actual_score for r in self.results if r.actual_score > 0]
        score_stats = {
            'average_score': sum(scores) / len(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'zero_scores': sum(1 for r in self.results if r.actual_score == 0)
        }
        
        # Threshold analysis
        threshold_issues = []
        for result in self.results:
            if result.stage_details and 'threshold_analysis' in result.stage_details:
                analysis = result.stage_details['threshold_analysis']
                if not analysis.get('passes_first_stage', False) and result.signature.category != 'natural':
                    threshold_issues.append({
                        'designation': result.signature.designation,
                        'category': result.signature.category, 
                        'score': result.actual_score,
                        'issue': 'Failed first-stage threshold despite artificial characteristics'
                    })
        
        return {
            'validation_summary': {
                'total_signatures': total,
                'passed': passed,
                'failed': failed,
                'success_rate': passed / total if total > 0 else 0
            },
            'category_performance': category_stats,
            'stage_distribution': stage_stats,
            'score_analysis': score_stats,
            'threshold_issues': threshold_issues,
            'pipeline_status': {
                'pipeline_available': self.pipeline_available,
                'scoring_available': self.scoring_available
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for zero candidate issue
        zero_scores = sum(1 for r in self.results if r.actual_score == 0)
        if zero_scores > len(self.results) * 0.5:
            recommendations.append("CRITICAL: Over 50% of signatures scored zero - scoring system may be non-functional")
        
        # Check if artificial signatures are being filtered
        artificial_filtered = sum(1 for r in self.results 
                                if r.signature.category != 'natural' and r.reached_stage == 'filtered')
        if artificial_filtered > 0:
            recommendations.append(f"MAJOR: {artificial_filtered} artificial signatures were filtered - thresholds too high")
        
        # Check if obvious signatures reach expert review
        obvious_failures = sum(1 for r in self.results 
                              if r.signature.category == 'obvious' and r.reached_stage != 'expert_review')
        if obvious_failures > 0:
            recommendations.append(f"HIGH: {obvious_failures} obvious artificial signatures failed to reach expert review")
        
        # Check if natural controls are properly filtered
        natural_passed = sum(1 for r in self.results 
                           if r.signature.category == 'natural' and r.reached_stage != 'filtered')
        if natural_passed > 0:
            recommendations.append(f"MEDIUM: {natural_passed} natural objects passed filters - may need debris detection")
        
        # System availability issues
        if not self.pipeline_available:
            recommendations.append("CRITICAL: Pipeline components not available - check imports and dependencies")
        
        if not self.scoring_available:
            recommendations.append("CRITICAL: Scoring system not available - check XVIII SWARM implementation")
        
        return recommendations
    
    def display_results(self):
        """Display validation results in formatted table."""
        if not self.results:
            print("No validation results to display")
            return
        
        if console:
            # Rich table display
            table = Table(title="🔬 Pipeline Validation Results")
            table.add_column("Designation", style="cyan")
            table.add_column("Category", style="yellow")
            table.add_column("Expected", style="green")
            table.add_column("Actual", style="blue")
            table.add_column("Score", style="magenta")
            table.add_column("Status", style="white")
            
            for result in self.results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                status_style = "green" if result.passed else "red"
                
                table.add_row(
                    result.signature.designation,
                    result.signature.category,
                    result.expected_stage,
                    result.reached_stage,
                    f"{result.actual_score:.3f}",
                    f"[{status_style}]{status}[/]"
                )
            
            console.print(table)
        else:
            # Simple text display
            print("\n📊 VALIDATION RESULTS")
            print("-" * 80)
            print(f"{'Designation':<15} {'Category':<10} {'Expected':<12} {'Actual':<12} {'Score':<8} {'Status'}")
            print("-" * 80)
            
            for result in self.results:
                status = "PASS" if result.passed else "FAIL"
                print(f"{result.signature.designation:<15} {result.signature.category:<10} "
                      f"{result.expected_stage:<12} {result.reached_stage:<12} "
                      f"{result.actual_score:<8.3f} {status}")

async def main():
    """Main validation test function."""
    print("🚀 ZETA SWARM: PIPELINE VALIDATION & MENU CLEANUP SPECIALIST")
    print("🎯 MISSION: Validate complete pipeline with artificial test signatures")
    print("=" * 70)
    
    # Generate test signatures
    print("\n📋 PHASE 1: Generating Artificial Test Signatures")
    signatures = generate_complete_test_suite()
    
    # Initialize validator
    print(f"\n🔧 PHASE 2: Initializing Pipeline Validator")
    validator = PipelineValidator()
    
    # Run validation
    print(f"\n🧪 PHASE 3: Running Pipeline Validation")
    results = await validator.run_validation(signatures)
    
    # Display results
    print(f"\n📊 PHASE 4: Validation Results")
    validator.display_results()
    
    # Generate report
    print(f"\n📋 PHASE 5: Validation Report")
    report = validator.generate_validation_report()
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = PROJECT_ROOT / f"pipeline_validation_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        # Prepare report for JSON serialization
        serializable_report = json.loads(json.dumps(report, default=str))
        json.dump(serializable_report, f, indent=2)
    
    print(f"💾 Detailed report saved to: {report_file}")
    
    # Display summary
    if console:
        summary_panel = Panel(
            f"Total Signatures: {report['validation_summary']['total_signatures']}\n"
            f"Passed: {report['validation_summary']['passed']}\n" 
            f"Failed: {report['validation_summary']['failed']}\n"
            f"Success Rate: {report['validation_summary']['success_rate']:.1%}\n"
            f"\nRecommendations: {len(report['recommendations'])}",
            title="[bold red]🎯 Validation Summary[/bold red]",
            border_style="red"
        )
        console.print(summary_panel)
    else:
        print("\n🎯 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Total Signatures: {report['validation_summary']['total_signatures']}")
        print(f"Passed: {report['validation_summary']['passed']}")
        print(f"Failed: {report['validation_summary']['failed']}")
        print(f"Success Rate: {report['validation_summary']['success_rate']:.1%}")
    
    # Display recommendations
    print(f"\n🎯 CRITICAL RECOMMENDATIONS")
    print("=" * 40)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())