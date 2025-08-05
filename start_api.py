#!/usr/bin/env python3
"""
aNEOS API Startup Script

Production-ready startup script for the aNEOS API server with proper
initialization sequence, health checks, and graceful shutdown handling.
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import uvicorn
    from aneos_api.app import create_app, get_aneos_app
    from aneos_api.database import init_database, get_database_status
    HAS_DEPENDENCIES = True
except ImportError as e:
    HAS_DEPENDENCIES = False
    print(f"Missing dependencies: {e}")
    print("Please install required packages: pip install -r requirements.txt")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ANEOSServer:
    """aNEOS API Server with proper lifecycle management."""
    
    def __init__(self):
        self.app = None
        self.server = None
        self.shutdown_event = asyncio.Event()
        
    async def startup_checks(self) -> bool:
        """Perform startup health checks and initialization."""
        logger.info("🚀 Starting aNEOS API Server...")
        
        # Check dependencies
        if not HAS_DEPENDENCIES:
            logger.error("❌ Missing required dependencies")
            return False
            
        # Check database
        logger.info("🗄️  Checking database connection...")
        db_status = get_database_status()
        if not db_status.get('available', False):
            logger.warning(f"⚠️  Database not available: {db_status.get('error', 'Unknown error')}")
        else:
            logger.info("✅ Database connection successful")
            
        # Initialize database if needed
        logger.info("🔧 Initializing database...")
        init_result = init_database()
        if init_result:
            logger.info("✅ Database initialized successfully")
        else:
            logger.warning("⚠️  Database initialization failed")
            
        # Create FastAPI app
        logger.info("🌐 Creating FastAPI application...")
        try:
            self.app = create_app()
            logger.info("✅ FastAPI application created")
        except Exception as e:
            logger.error(f"❌ Failed to create FastAPI app: {e}")
            return False
            
        # Verify aNEOS services
        logger.info("🔍 Initializing aNEOS services...")
        aneos_app = get_aneos_app()
        try:
            await aneos_app.initialize_services()
            logger.info("✅ aNEOS services initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize aNEOS services: {e}")
            return False
            
        return True
        
    async def run_server(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run the server with proper configuration."""
        if not await self.startup_checks():
            logger.error("❌ Startup checks failed, exiting...")
            return False
            
        # Server configuration
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True,
            use_colors=True,
            loop="asyncio"
        )
        
        # Create server
        self.server = uvicorn.Server(config)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"📶 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        logger.info(f"🌟 Starting server on {host}:{port}")
        logger.info(f"📖 API Documentation: http://{host}:{port}/docs")
        logger.info(f"📊 Dashboard: http://{host}:{port}/dashboard")
        logger.info(f"❤️  Health Check: http://{host}:{port}/health")
        
        try:
            await self.server.serve()
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
            return False
            
        logger.info("🛑 Server stopped")
        return True
        
    async def shutdown(self):
        """Graceful shutdown sequence."""
        logger.info("🛑 Initiating graceful shutdown...")
        
        if self.server:
            logger.info("🔌 Stopping HTTP server...")
            self.server.should_exit = True
            
        # Shutdown aNEOS services
        logger.info("🔧 Shutting down aNEOS services...")
        try:
            aneos_app = get_aneos_app()
            await aneos_app.shutdown_services()
            logger.info("✅ aNEOS services shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️  Error during service shutdown: {e}")
            
        self.shutdown_event.set()
        logger.info("✅ Graceful shutdown complete")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="aNEOS API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    if not HAS_DEPENDENCIES:
        logger.error("❌ Cannot start server: missing dependencies")
        sys.exit(1)
        
    # Development mode
    if args.dev:
        logger.info("🔧 Starting in development mode with auto-reload")
        uvicorn.run(
            "aneos_api.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level.lower()
        )
        return
        
    # Production mode
    server = ANEOSServer()
    
    try:
        asyncio.run(server.run_server(
            host=args.host,
            port=args.port,
            workers=args.workers
        ))
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()