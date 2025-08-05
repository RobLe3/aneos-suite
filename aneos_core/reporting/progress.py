#!/usr/bin/env python3
"""
Progress Tracking and Status Updates for aNEOS Core

Provides comprehensive progress tracking capabilities with rich formatting,
status updates, and operation monitoring for professional reporting.
"""

import time
import threading
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from contextlib import contextmanager
import logging

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProgressTracker:
    """
    Professional progress tracking with rich formatting and status updates.
    
    Provides comprehensive progress monitoring for long-running operations
    with fallback support for environments without rich formatting.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize progress tracker.
        
        Args:
            logger: Optional logger instance
        """
        self.has_rich = HAS_RICH
        self.has_tqdm = HAS_TQDM
        self.console = Console() if HAS_RICH else None
        self.logger = logger or logging.getLogger(__name__)
        
        # Active progress bars and spinners
        self.active_progress = {}
        self.active_tasks = {}
        
        # Status tracking
        self.operation_start_time = None
        self.current_operation = None
        self.operation_history = []
        
        # Color codes for fallback formatting
        self.colors = {
            "success": "32",   # Green
            "warning": "33",   # Yellow
            "error": "31",     # Red
            "info": "36",      # Cyan
            "progress": "35",  # Magenta
            "emphasis": "1"    # Bold
        }
    
    def colorize(self, text: str, color_code: str) -> str:
        """Apply ANSI color codes to text for fallback formatting."""
        return f"\033[{color_code}m{text}\033[0m"
    
    def print_status(self, message: str, status_type: str = "info") -> None:
        """
        Print a status message with appropriate formatting.
        
        Args:
            message: Status message to display
            status_type: Type of status (info, success, warning, error, progress)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if self.has_rich and self.console:
            color_map = {
                "info": "blue",
                "success": "green",
                "warning": "yellow",
                "error": "red",
                "progress": "magenta"
            }
            color = color_map.get(status_type, "white")
            
            status_icons = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
                "progress": "🔄"
            }
            icon = status_icons.get(status_type, "•")
            
            self.console.print(f"[dim]{timestamp}[/dim] [{color}]{icon} {message}[/{color}]")
        else:
            # Fallback formatting
            status_prefixes = {
                "info": "INFO",
                "success": "SUCCESS",
                "warning": "WARNING",
                "error": "ERROR",
                "progress": "PROGRESS"
            }
            prefix = status_prefixes.get(status_type, "INFO")
            color = self.colors.get(status_type, self.colors["info"])
            
            formatted_message = self.colorize(f"[{timestamp}] {prefix}: {message}", color)
            print(formatted_message)
        
        # Always log to logger
        log_levels = {
            "info": logging.INFO,
            "success": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "progress": logging.INFO
        }
        level = log_levels.get(status_type, logging.INFO)
        self.logger.log(level, message)
    
    def start_operation(self, operation_name: str, description: str = "") -> None:
        """
        Start tracking a new operation.
        
        Args:
            operation_name: Name of the operation
            description: Optional description
        """
        self.operation_start_time = datetime.now()
        self.current_operation = operation_name
        
        full_message = f"Starting operation: {operation_name}"
        if description:
            full_message += f" - {description}"
        
        self.print_status(full_message, "progress")
        
        # Add to operation history
        self.operation_history.append({
            "name": operation_name,
            "description": description,
            "start_time": self.operation_start_time,
            "end_time": None,
            "duration": None,
            "status": "running"
        })
    
    def complete_operation(self, success: bool = True, message: str = "") -> None:
        """
        Complete the current operation.
        
        Args:
            success: Whether operation completed successfully
            message: Optional completion message
        """
        if not self.current_operation or not self.operation_start_time:
            return
        
        end_time = datetime.now()
        duration = end_time - self.operation_start_time
        
        # Update operation history
        if self.operation_history:
            self.operation_history[-1].update({
                "end_time": end_time,
                "duration": duration,
                "status": "completed" if success else "failed"
            })
        
        status_type = "success" if success else "error"
        duration_str = self._format_duration(duration)
        
        completion_message = f"Operation '{self.current_operation}' {'completed' if success else 'failed'} in {duration_str}"
        if message:
            completion_message += f" - {message}"
        
        self.print_status(completion_message, status_type)
        
        self.current_operation = None
        self.operation_start_time = None
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for display."""
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    @contextmanager
    def progress_bar(self, description: str, total: int = 100, 
                    show_percentage: bool = True, show_eta: bool = True):
        """
        Context manager for progress bar operations.
        
        Args:
            description: Description of the operation
            total: Total number of steps
            show_percentage: Whether to show percentage
            show_eta: Whether to show estimated time remaining
        """
        if self.has_rich and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn() if show_percentage else TextColumn(""),
                TimeElapsedColumn(),
                TimeRemainingColumn() if show_eta else TextColumn(""),
                console=self.console,
                transient=False
            ) as progress:
                task = progress.add_task(description, total=total)
                yield ProgressBarManager(progress, task)
        
        elif self.has_tqdm:
            with tqdm(total=total, desc=description, dynamic_ncols=True) as pbar:
                yield TqdmProgressManager(pbar)
        
        else:
            # Fallback progress manager
            yield FallbackProgressManager(description, total, self.print_status)
    
    @contextmanager
    def spinner(self, description: str):
        """
        Context manager for spinner operations.
        
        Args:
            description: Description of the operation
        """
        if self.has_rich and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task(description, total=None)
                yield SpinnerManager(progress, task)
        else:
            yield FallbackSpinnerManager(description, self.print_status)
    
    def print_operation_summary(self) -> None:
        """Print a summary of all operations."""
        if not self.operation_history:
            self.print_status("No operations recorded", "info")
            return
        
        if self.has_rich and self.console:
            table = Table(title="Operation Summary")
            table.add_column("Operation", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Duration", style="yellow")
            table.add_column("Description", style="blue")
            
            for op in self.operation_history:
                status_style = "green" if op["status"] == "completed" else "red" if op["status"] == "failed" else "yellow"
                duration_str = self._format_duration(op["duration"]) if op["duration"] else "Running..."
                
                table.add_row(
                    op["name"],
                    Text(op["status"].title(), style=status_style),
                    duration_str,
                    op["description"] or "N/A"
                )
            
            self.console.print(table)
        else:
            # Fallback summary
            self.print_status("=== Operation Summary ===", "info")
            for op in self.operation_history:
                duration_str = self._format_duration(op["duration"]) if op["duration"] else "Running..."
                status = op["status"].upper()
                self.print_status(f"{op['name']}: {status} ({duration_str})", "info")
    
    def wait_with_progress(self, delay: int, description: str = "Waiting") -> None:
        """
        Wait with a visual progress indicator.
        
        Args:
            delay: Number of seconds to wait
            description: Description of what we're waiting for
        """
        with self.progress_bar(description, total=delay, show_eta=False) as progress:
            for i in range(delay):
                time.sleep(1)
                progress.update(1)


class ProgressBarManager:
    """Manager for Rich progress bars."""
    
    def __init__(self, progress, task):
        self.progress = progress
        self.task = task
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress bar."""
        if description:
            self.progress.update(self.task, advance=advance, description=description)
        else:
            self.progress.update(self.task, advance=advance)
    
    def set_total(self, total: int):
        """Set new total for progress bar."""
        self.progress.update(self.task, total=total)


class TqdmProgressManager:
    """Manager for tqdm progress bars."""
    
    def __init__(self, pbar):
        self.pbar = pbar
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress bar."""
        if description:
            self.pbar.set_description(description)
        self.pbar.update(advance)
    
    def set_total(self, total: int):
        """Set new total for progress bar."""
        self.pbar.total = total


class FallbackProgressManager:
    """Fallback progress manager for environments without rich or tqdm."""
    
    def __init__(self, description: str, total: int, print_func: Callable):
        self.description = description
        self.total = total
        self.current = 0
        self.print_func = print_func
        self.last_percentage = -1
    
    def update(self, advance: int = 1, description: Optional[str] = None):
        """Update progress."""
        self.current += advance
        percentage = int((self.current / self.total) * 100)
        
        # Only print every 10% to avoid spam
        if percentage >= self.last_percentage + 10 or percentage == 100:
            desc = description or self.description
            self.print_func(f"{desc}: {percentage}%", "progress")
            self.last_percentage = percentage
    
    def set_total(self, total: int):
        """Set new total."""
        self.total = total


class SpinnerManager:
    """Manager for Rich spinners."""
    
    def __init__(self, progress, task):
        self.progress = progress
        self.task = task
    
    def update_description(self, description: str):
        """Update spinner description."""
        self.progress.update(self.task, description=description)


class FallbackSpinnerManager:
    """Fallback spinner manager."""
    
    def __init__(self, description: str, print_func: Callable):
        self.description = description
        self.print_func = print_func
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def _spin(self):
        """Spinner animation."""
        spinner_chars = ['|', '/', '-', '\\']
        idx = 0
        while self.running:
            sys.stdout.write(f"\r{spinner_chars[idx % len(spinner_chars)]} {self.description}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1
        sys.stdout.write(f"\r✓ {self.description} - Complete\n")
        sys.stdout.flush()
    
    def update_description(self, description: str):
        """Update spinner description."""
        self.description = description
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)


class OperationTimer:
    """
    Simple operation timer for measuring execution time.
    """
    
    def __init__(self, operation_name: str, tracker: Optional[ProgressTracker] = None):
        """
        Initialize operation timer.
        
        Args:
            operation_name: Name of the operation being timed
            tracker: Optional progress tracker instance
        """
        self.operation_name = operation_name
        self.tracker = tracker
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        if self.tracker:
            self.tracker.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        success = exc_type is None
        
        if self.tracker:
            duration = self.end_time - self.start_time
            message = f"Completed in {self.tracker._format_duration(duration)}"
            self.tracker.complete_operation(success, message)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get operation duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Convenience function for quick progress tracking
def create_progress_tracker(logger: Optional[logging.Logger] = None) -> ProgressTracker:
    """
    Create a progress tracker instance.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        ProgressTracker instance
    """
    return ProgressTracker(logger)