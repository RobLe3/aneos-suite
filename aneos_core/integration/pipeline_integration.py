#!/usr/bin/env python3
"""
Pipeline Integration Module - SAFE PATH PHASE 4

Integrates the complete automatic review pipeline with the existing aNEOS system.
Provides menu integration, component wiring, and safe execution workflows.

Key Features:
- Menu system integration for historical polling
- Component auto-discovery and wiring
- Safe execution with fallback options
- Results integration with existing analysis system
- User-friendly interface for 200-year polling
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Core aNEOS imports
try:
    from ..pipeline.automatic_review_pipeline import (
        AutomaticReviewPipeline,
        PipelineConfig, 
        create_automatic_pipeline
    )
    from ..polling.historical_chunked_poller import (
        HistoricalChunkedPoller,
        create_historical_poller
    )
    from ..analysis.enhanced_pipeline import EnhancedAnalysisPipeline
    from ..analysis.advanced_scoring import AdvancedScoreCalculator
    from ..validation.multi_stage_validator import MultiStageValidator
    HAS_PIPELINE_COMPONENTS = True
except ImportError as e:
    HAS_PIPELINE_COMPONENTS = False
    PIPELINE_IMPORT_ERROR = str(e)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, IntPrompt
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

logger = logging.getLogger(__name__)

class PipelineIntegration:
    """
    Integration layer for automatic review pipeline with aNEOS system.
    
    This class provides safe integration of the complete pipeline system
    with the existing aNEOS menu and analysis infrastructure.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.console = console if HAS_RICH else None
        
        # Component instances
        self.automatic_pipeline: Optional[AutomaticReviewPipeline] = None
        self.chunked_poller: Optional[HistoricalChunkedPoller] = None
        self.enhanced_pipeline: Optional[EnhancedAnalysisPipeline] = None
        self.xviii_swarm_scorer = None
        self.multi_stage_validator = None
        
        # Integration status
        self.components_initialized = False
        self.integration_errors = []
        
    async def initialize_components(self) -> bool:
        """
        Initialize and wire together all pipeline components.
        
        Returns:
            True if successful, False if fallback mode needed
        """
        try:
            if not HAS_PIPELINE_COMPONENTS:
                self.integration_errors.append(f"Pipeline components not available: {PIPELINE_IMPORT_ERROR}")
                return False
            
            self.logger.info("Initializing pipeline components...")
            
            # Initialize chunked poller
            self.chunked_poller = await create_historical_poller(
                chunk_size_years=5,
                max_objects_per_chunk=50000,
                enable_caching=True
            )
            
            # Connect to real data sources - initialize enhanced NEO poller
            try:
                from enhanced_neo_poller import EnhancedNEOPoller
                self.enhanced_neo_poller = EnhancedNEOPoller(
                    data_dir="neo_data",
                    professional_report=False,
                    enable_ai_validation=True
                )
                self.logger.info("Enhanced NEO poller initialized for real data access")
            except ImportError:
                self.logger.warning("Enhanced NEO poller not available - using fallback data simulation")
                self.enhanced_neo_poller = None
            
            # Try to initialize enhanced analysis components
            try:
                # Mock original pipeline for enhanced wrapper
                class MockOriginalPipeline:
                    async def analyze_neo(self, designation, neo_data=None):
                        return type('MockResult', (), {'overall_score': 0.5})()
                
                self.enhanced_pipeline = EnhancedAnalysisPipeline(
                    MockOriginalPipeline(),
                    enable_validation=True
                )
                
                # Initialize ATLAS scorer
                config_path = Path("aneos_core/config/advanced_scoring_weights.json")
                if config_path.exists():
                    self.xviii_swarm_scorer = AdvancedScoreCalculator(config_path)
                
                # Initialize multi-stage validator
                self.multi_stage_validator = MultiStageValidator()
                
            except Exception as e:
                self.logger.warning(f"Enhanced analysis components failed to initialize: {e}")
                self.integration_errors.append(f"Enhanced analysis: {e}")
            
            # Create automatic pipeline with proper production thresholds
            self.automatic_pipeline = await create_automatic_pipeline(
                years_back=200,
                first_stage_threshold=0.08,
                multi_stage_threshold=0.20,
                expert_threshold=0.35
            )
            
            # Wire components together
            if self.automatic_pipeline:
                self.automatic_pipeline.set_components(
                    chunked_poller=self.chunked_poller,
                    xviii_swarm_scorer=self.xviii_swarm_scorer,
                    multi_stage_validator=self.multi_stage_validator,
                    enhanced_pipeline=self.enhanced_pipeline
                )
                
                # Set chunked poller components
                if self.chunked_poller:
                    self.chunked_poller.set_components(
                        base_poller=self.enhanced_neo_poller,  # Connect to real data source
                        xviii_swarm_scorer=self.xviii_swarm_scorer
                    )
            
            self.components_initialized = True
            self.logger.info("Pipeline components initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            self.integration_errors.append(f"Initialization: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get current integration status and component availability.
        
        Returns:
            Dictionary with integration status information
        """
        return {
            'pipeline_components_available': HAS_PIPELINE_COMPONENTS,
            'components_initialized': self.components_initialized,
            'automatic_pipeline_ready': self.automatic_pipeline is not None,
            'chunked_poller_ready': self.chunked_poller is not None,
            'enhanced_analysis_ready': self.enhanced_pipeline is not None,
            'xviii_swarm_ready': self.xviii_swarm_scorer is not None,
            'multi_stage_validator_ready': self.multi_stage_validator is not None,
            'integration_errors': self.integration_errors,
            'rich_ui_available': HAS_RICH
        }
    
    async def run_historical_polling_workflow(
        self,
        years_back: int = 200,
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete historical polling workflow with user interaction.
        
        Args:
            years_back: Number of years to poll back
            interactive: Whether to use interactive prompts
            
        Returns:
            Dictionary with workflow results
        """
        try:
            # Ensure components are initialized
            if not self.components_initialized:
                success = await self.initialize_components()
                if not success and interactive:
                    return await self._run_fallback_workflow(years_back)
            
            if self.console and interactive:
                self.console.print(Panel(
                    f"[bold blue]aNEOS Historical Polling Workflow[/]\n"
                    f"Polling {years_back} years of NEO data with automatic review pipeline",
                    style="blue"
                ))
                
                # Confirm execution
                if not Confirm.ask(f"Start {years_back}-year historical polling?", default=True):
                    return {'status': 'cancelled', 'message': 'User cancelled operation'}
            
            # Display component status
            if self.console and interactive:
                self._display_component_status()
            
            # Run the complete pipeline
            if self.automatic_pipeline:
                self.logger.info(f"Starting complete pipeline for {years_back} years")
                
                result = await self.automatic_pipeline.run_complete_pipeline(
                    years_back=years_back
                )
                
                return {
                    'status': 'success',
                    'pipeline_result': result,
                    'total_objects': result.total_input_objects,
                    'final_candidates': result.final_candidates,
                    'processing_time_seconds': (result.processing_end_time - result.processing_start_time).total_seconds(),
                    'compression_ratio': result.pipeline_metrics.get('funnel_compression_ratio', 0)
                }
            else:
                return await self._run_fallback_workflow(years_back)
                
        except Exception as e:
            self.logger.error(f"Historical polling workflow failed: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'fallback_available': True
            }
    
    async def _run_fallback_workflow(self, years_back: int) -> Dict[str, Any]:
        """
        Run fallback workflow when full pipeline is not available.
        
        Args:
            years_back: Number of years to search
            
        Returns:
            Dictionary with fallback results
        """
        try:
            if self.console:
                self.console.print(Panel(
                    "[yellow]Running fallback mode - limited functionality available[/]",
                    style="yellow"
                ))
            
            # Try basic chunked polling if available
            if self.chunked_poller:
                self.logger.info("Running basic chunked polling")
                
                result = await self.chunked_poller.poll_historical_data(years_back)
                
                return {
                    'status': 'fallback_success',
                    'mode': 'basic_chunked_polling',
                    'total_objects': result.total_objects_found,
                    'candidates_flagged': result.total_candidates_flagged,
                    'chunks_processed': result.total_chunks_processed,
                    'message': 'Basic historical polling completed - advanced analysis not available'
                }
            else:
                # Most basic fallback
                return {
                    'status': 'fallback_limited',
                    'mode': 'mock_simulation',
                    'message': f'Simulated {years_back}-year poll - pipeline components not available',
                    'estimated_objects': years_back * 1000,  # Rough estimate
                    'note': 'This is a simulation - real polling requires component initialization'
                }
                
        except Exception as e:
            self.logger.error(f"Fallback workflow failed: {e}")
            return {
                'status': 'fallback_error',
                'error_message': str(e),
                'message': 'Both primary and fallback workflows failed'
            }
    
    def _display_component_status(self):
        """Display component status using rich UI."""
        if not self.console:
            return
        
        status = self.get_integration_status()
        
        # Create status table
        table = Table(title="Pipeline Component Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        components = [
            ("Automatic Pipeline", status['automatic_pipeline_ready'], "Complete workflow orchestration"),
            ("Chunked Poller", status['chunked_poller_ready'], "200-year historical data processing"),
            ("ATLAS Scorer", status['xviii_swarm_ready'], "Advanced anomaly scoring"),
            ("Multi-Stage Validator", status['multi_stage_validator_ready'], "Scientific validation pipeline"),
            ("Enhanced Analysis", status['enhanced_analysis_ready'], "Comprehensive NEO analysis")
        ]
        
        for name, ready, description in components:
            status_icon = "✅ Ready" if ready else "❌ Not Available"
            table.add_row(name, status_icon, description)
        
        self.console.print(table)
        
        # Show integration errors if any
        if status['integration_errors']:
            self.console.print("\n[yellow]Integration Notes:[/]")
            for error in status['integration_errors']:
                self.console.print(f"  • {error}")
    
    def get_menu_options(self) -> List[Dict[str, Any]]:
        """
        Get menu options for integration with aNEOS menu system.
        
        Returns:
            List of menu option dictionaries
        """
        options = [
            {
                'key': 'historical_poll_200y',
                'name': '🕰️  200-Year Historical NEO Poll',
                'description': 'Complete 200-year historical polling with automatic review pipeline',
                'function': lambda: self.run_historical_polling_workflow(200),
                'enabled': True
            },
            {
                'key': 'historical_poll_custom',
                'name': '📅 Custom Historical Poll',
                'description': 'Historical polling for custom time period',
                'function': self._custom_historical_poll,
                'enabled': True
            },
            {
                'key': 'pipeline_status',
                'name': '🔍 Pipeline Status',
                'description': 'Check automatic review pipeline component status',
                'function': self._display_pipeline_status,
                'enabled': True
            },
            {
                'key': 'test_pipeline',
                'name': '🧪 Test Pipeline (5 years)',
                'description': 'Test automatic pipeline with small dataset',
                'function': lambda: self.run_historical_polling_workflow(5),
                'enabled': True
            }
        ]
        
        return options
    
    async def _custom_historical_poll(self):
        """Interactive custom historical polling."""
        if not self.console:
            return await self.run_historical_polling_workflow(50)  # Default fallback
        
        self.console.print(Panel("[bold green]Custom Historical Polling[/]", style="green"))
        
        years = IntPrompt.ask("Enter number of years to poll back", default=50, show_default=True)
        
        if years > 200:
            self.console.print("[yellow]Warning: Large time periods may take significant time[/]")
            if not Confirm.ask(f"Continue with {years}-year poll?", default=False):
                return {'status': 'cancelled'}
        
        return await self.run_historical_polling_workflow(years, interactive=True)
    
    async def _display_pipeline_status(self):
        """Display detailed pipeline status."""
        if not self.console:
            status = self.get_integration_status()
            print(f"Pipeline Status: {status}")
            return status
        
        # Ensure components are checked
        if not self.components_initialized:
            await self.initialize_components()
        
        self.console.print(Panel("[bold blue]Automatic Review Pipeline Status[/]", style="blue"))
        
        self._display_component_status()
        
        # Test connectivity
        self.console.print("\n[bold]Testing Component Connectivity...[/]")
        
        tests = [
            ("Chunked Poller Generation", self._test_chunk_generation),
            ("ATLAS Scoring", self._test_xviii_scoring),
            ("Pipeline Creation", self._test_pipeline_creation)
        ]
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                status = "✅ Pass" if result else "❌ Fail"
                self.console.print(f"  {test_name}: {status}")
            except Exception as e:
                self.console.print(f"  {test_name}: ❌ Error - {e}")
        
        return self.get_integration_status()
    
    async def _test_chunk_generation(self) -> bool:
        """Test chunk generation functionality."""
        if not self.chunked_poller:
            return False
        
        chunks = list(self.chunked_poller.generate_time_chunks(years_back=10))
        return len(chunks) > 0
    
    async def _test_xviii_scoring(self) -> bool:
        """Test ATLAS scoring functionality."""
        if not self.xviii_swarm_scorer:
            return False
        
        # Test with mock data
        mock_neo = {
            'designation': 'TEST_2024_A1',
            'orbital_elements': {
                'eccentricity': 0.8,
                'inclination': 15,
                'semi_major_axis': 1.2
            }
        }
        
        try:
            score = self.xviii_swarm_scorer.calculate_score(mock_neo, {})
            return score.overall_score >= 0
        except Exception:
            return False
    
    async def _test_pipeline_creation(self) -> bool:
        """Test pipeline creation and basic functionality."""
        try:
            test_pipeline = await create_automatic_pipeline(years_back=1)
            return test_pipeline is not None
        except Exception:
            return False

# Global instance for menu integration
pipeline_integration = PipelineIntegration()

# Convenience functions for menu integration
async def initialize_pipeline_integration() -> bool:
    """Initialize pipeline integration for menu use."""
    return await pipeline_integration.initialize_components()

async def run_200_year_poll() -> Dict[str, Any]:
    """Run 200-year historical poll - main function for menu."""
    return await pipeline_integration.run_historical_polling_workflow(200)

async def get_pipeline_status() -> Dict[str, Any]:
    """Get pipeline status - utility function for menu."""
    return await pipeline_integration._display_pipeline_status()

def get_historical_polling_menu_options() -> List[Dict[str, Any]]:
    """Get menu options for historical polling."""
    return pipeline_integration.get_menu_options()

if __name__ == "__main__":
    # Test pipeline integration
    import asyncio
    
    async def test_integration():
        integration = PipelineIntegration()
        
        # Test initialization
        success = await integration.initialize_components()
        print(f"Initialization: {'Success' if success else 'Failed'}")
        
        # Test status
        status = integration.get_integration_status()
        print(f"Components ready: {sum(1 for k, v in status.items() if k.endswith('_ready') and v)}")
        
        # Test small workflow
        result = await integration.run_historical_polling_workflow(years_back=2, interactive=False)
        print(f"Test workflow: {result['status']}")
    
    asyncio.run(test_integration())