"""
Integration Module for aNEOS Automatic Review System

This module provides integration layers for connecting the automatic
review pipeline with the existing aNEOS menu system and components.
"""

from .pipeline_integration import (
    PipelineIntegration,
    pipeline_integration,
    initialize_pipeline_integration,
    run_200_year_poll,
    get_pipeline_status,
    get_historical_polling_menu_options
)

__all__ = [
    'PipelineIntegration',
    'pipeline_integration', 
    'initialize_pipeline_integration',
    'run_200_year_poll',
    'get_pipeline_status',
    'get_historical_polling_menu_options'
]