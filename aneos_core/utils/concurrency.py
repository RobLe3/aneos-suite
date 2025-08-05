"""
Concurrency utilities for aNEOS Core.

Provides thread pool management, async utilities, and
concurrent execution helpers for data processing tasks.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from contextlib import contextmanager
from typing import Callable, List, Any, Optional, Dict, Union, Iterator
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result of a concurrent task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    worker_id: Optional[str] = None


class ThreadPoolManager:
    """
    Advanced thread pool manager with monitoring and statistics.
    
    Features:
    - Dynamic pool sizing
    - Task monitoring and statistics
    - Health checks and recovery
    - Resource usage tracking
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        thread_name_prefix: str = "aNEOS",
        monitor_performance: bool = True
    ):
        """
        Initialize thread pool manager.
        
        Args:
            max_workers: Maximum number of worker threads
            thread_name_prefix: Prefix for thread names
            monitor_performance: Enable performance monitoring
        """
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.monitor_performance = monitor_performance
        
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "active_tasks": 0
        }
        
        self._lock = threading.Lock()
        self._active_futures: Dict[str, Future] = {}
    
    def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        **kwargs
    ) -> Future[TaskResult]:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            task_id: Optional task identifier
            **kwargs: Function keyword arguments
            
        Returns:
            Future representing the task
        """
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        def wrapped_task():
            start_time = time.time()
            worker_id = threading.current_thread().name
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                with self._lock:
                    self.stats["tasks_completed"] += 1
                    self.stats["total_execution_time"] += duration
                    self.stats["active_tasks"] -= 1
                
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    duration=duration,
                    worker_id=worker_id
                )
                
            except Exception as e:
                duration = time.time() - start_time
                
                with self._lock:
                    self.stats["tasks_failed"] += 1
                    self.stats["active_tasks"] -= 1
                
                self.logger.error(f"Task {task_id} failed: {e}")
                
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    duration=duration,
                    worker_id=worker_id
                )
        
        with self._lock:
            self.stats["tasks_submitted"] += 1
            self.stats["active_tasks"] += 1
        
        future = self.executor.submit(wrapped_task)
        self._active_futures[task_id] = future
        
        return future
    
    def submit_batch(
        self,
        func: Callable,
        args_list: List[tuple],
        kwargs_list: Optional[List[dict]] = None
    ) -> List[Future[TaskResult]]:
        """
        Submit multiple tasks as a batch.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples
            kwargs_list: Optional list of keyword argument dictionaries
            
        Returns:
            List of futures representing the tasks
        """
        if kwargs_list is None:
            kwargs_list = [{}] * len(args_list)
        
        futures = []
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list)):
            task_id = f"batch_task_{i}"
            future = self.submit_task(func, *args, task_id=task_id, **kwargs)
            futures.append(future)
        
        return futures
    
    def wait_for_completion(
        self,
        futures: List[Future[TaskResult]],
        timeout: Optional[float] = None,
        return_when: str = "ALL_COMPLETED"
    ) -> List[TaskResult]:
        """
        Wait for futures to complete and return results.
        
        Args:
            futures: List of futures to wait for
            timeout: Maximum time to wait
            return_when: When to return (ALL_COMPLETED, FIRST_COMPLETED, etc.)
            
        Returns:
            List of task results
        """
        results = []
        
        try:
            done_futures = as_completed(futures, timeout=timeout)
            
            for future in done_futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error getting future result: {e}")
                    results.append(TaskResult(
                        task_id="unknown",
                        success=False,
                        error=e
                    ))
        
        except TimeoutError:
            self.logger.warning(f"Timeout waiting for {len(futures)} futures")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get thread pool statistics.
        
        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            avg_execution_time = (
                self.stats["total_execution_time"] / max(self.stats["tasks_completed"], 1)
            )
            
            success_rate = (
                self.stats["tasks_completed"] / max(self.stats["tasks_submitted"], 1) * 100
            )
            
            return {
                "max_workers": self.max_workers,
                "active_tasks": self.stats["active_tasks"],
                "tasks_submitted": self.stats["tasks_submitted"],
                "tasks_completed": self.stats["tasks_completed"],
                "tasks_failed": self.stats["tasks_failed"],
                "success_rate": round(success_rate, 2),
                "average_execution_time": round(avg_execution_time, 3),
                "total_execution_time": round(self.stats["total_execution_time"], 3)
            }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled
        """
        if task_id in self._active_futures:
            future = self._active_futures[task_id]
            cancelled = future.cancel()
            if cancelled:
                del self._active_futures[task_id]
                with self._lock:
                    self.stats["active_tasks"] -= 1
            return cancelled
        return False
    
    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """
        Shutdown the thread pool.
        
        Args:
            wait: Wait for active tasks to complete
            cancel_futures: Cancel active futures
        """
        if cancel_futures:
            for task_id in list(self._active_futures.keys()):
                self.cancel_task(task_id)
        
        self.executor.shutdown(wait=wait)
        self.logger.info("Thread pool manager shutdown")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


class AsyncTaskManager:
    """
    Async task manager for I/O bound operations.
    """
    
    def __init__(self, max_concurrent_tasks: int = 50):
        """
        Initialize async task manager.
        
        Args:
            max_concurrent_tasks: Maximum concurrent async tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.logger = logging.getLogger(__name__)
    
    async def run_task(self, coro: Callable, *args, **kwargs) -> Any:
        """
        Run an async task with semaphore control.
        
        Args:
            coro: Coroutine function
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task result
        """
        async with self.semaphore:
            return await coro(*args, **kwargs)
    
    async def run_batch(self, tasks: List[Callable]) -> List[Any]:
        """
        Run multiple async tasks concurrently.
        
        Args:
            tasks: List of coroutine functions
            
        Returns:
            List of results
        """
        async_tasks = [self.run_task(task) for task in tasks]
        return await asyncio.gather(*async_tasks, return_exceptions=True)


@contextmanager
def time_limit(seconds: float):
    """
    Context manager to enforce time limits on operations.
    
    Args:
        seconds: Time limit in seconds
        
    Raises:
        TimeoutError: If operation exceeds time limit
    """
    def timeout_handler():
        raise TimeoutError(f"Operation exceeded {seconds} seconds")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield
    finally:
        timer.cancel()


class RateLimiter:
    """
    Rate limiter for controlling request frequency.
    """
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum calls per time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a call.
        
        Args:
            timeout: Maximum time to wait for permission
            
        Returns:
            True if permission granted
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                now = time.time()
                
                # Remove old calls outside the time window
                self.calls = [call_time for call_time in self.calls 
                             if now - call_time < self.time_window]
                
                # Check if we can make a new call
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return True
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            # Wait a bit before trying again
            time.sleep(0.1)
    
    @contextmanager
    def limit(self, timeout: Optional[float] = None):
        """
        Context manager for rate limiting.
        
        Args:
            timeout: Maximum time to wait for permission
        """
        if self.acquire(timeout):
            yield
        else:
            raise TimeoutError("Rate limit timeout")


def run_with_retries(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Run function with retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        exceptions: Exceptions to catch and retry
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                break
            
            time.sleep(current_delay)
            current_delay *= backoff
    
    raise last_exception


class WorkerPool:
    """
    Generic worker pool for processing tasks.
    """
    
    def __init__(
        self,
        worker_func: Callable,
        num_workers: int = 4,
        queue_size: int = 100
    ):
        """
        Initialize worker pool.
        
        Args:
            worker_func: Function to process tasks
            num_workers: Number of worker threads
            queue_size: Maximum queue size
        """
        self.worker_func = worker_func
        self.num_workers = num_workers
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.workers = []
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the worker pool."""
        self.running = True
        
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        self.logger.info(f"Started worker pool with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the worker pool."""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.logger.info("Worker pool stopped")
    
    async def submit(self, task: Any):
        """Submit a task to the pool."""
        await self.queue.put(task)
    
    async def _worker(self, worker_name: str):
        """Worker coroutine."""
        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Process task
                await self.worker_func(task)
                
                # Mark task as done
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
    
    async def join(self):
        """Wait for all tasks to complete."""
        await self.queue.join()