import asyncio
import signal
from loguru import logger
from backtest import run_backtest, parse_args
from strategy.base_strategy import Strategy
from dashboard.dashboard import Dashboard

# Global flag for shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    _shutdown_requested = True

async def run_with_graceful_shutdown(args):
    """Run the application with graceful shutdown handling."""
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create task for the backtest
        backtest_task = asyncio.create_task(run_backtest(args))
        
        # Wait for either completion or shutdown signal
        while not backtest_task.done() and not _shutdown_requested:
            await asyncio.sleep(0.1)
        
        if _shutdown_requested and not backtest_task.done():
            logger.info("Cancelling backtest...")
            backtest_task.cancel()
            try:
                await backtest_task
            except asyncio.CancelledError:
                logger.info("Backtest cancelled successfully")
        
        return await backtest_task
    except asyncio.CancelledError:
        logger.info("Application shutdown complete")
    finally:
        logger.info("Cleanup complete")

def main():
    """Main entry point for the application."""
    try:
        args = parse_args()
        logger.info("Starting application...")
        asyncio.run(run_with_graceful_shutdown(args))
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main() 