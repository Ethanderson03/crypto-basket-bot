import asyncio
import os
from dotenv import load_dotenv
from loguru import logger
import signal
from typing import Optional

from strategy import Strategy
from data.ox_price_feed import OXPriceFeed

# Load environment variables
load_dotenv()

class TradingSystem:
    """
    Trading system that manages the strategy lifecycle.
    """
    
    def __init__(self):
        self.strategy: Optional[Strategy] = None
        self._shutdown_requested = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self._shutdown_requested = True
        if self.strategy:
            asyncio.create_task(self.strategy.stop())
    
    async def start(self):
        """Start the trading system."""
        try:
            # Initialize price feed
            price_feed = OXPriceFeed(
                mode="paper",
                is_test=bool(os.getenv("OX_USE_TESTNET", "true").lower() == "true")
            )
            
            # Initialize strategy
            self.strategy = Strategy(
                trading_pairs=[
                    "BTC-USDT",  # OX.FUN market codes
                    "ETH-USDT",
                    "SOL-USDT",
                    "BNB-USDT",
                    "XRP-USDT"
                ],
                base_collateral=float(os.getenv("BASE_COLLATERAL", "10000")),
                update_interval=60.0,  # 1 minute
                min_trade_interval=300.0  # 5 minutes
            )
            
            # Set price feed
            self.strategy.price_feed = price_feed
            
            # Start strategy
            logger.info("Starting trading system")
            await self.strategy.start()
            
        except Exception as e:
            logger.error(f"Trading system error: {e}")
            raise
        
        finally:
            if self.strategy:
                await self.strategy.stop()
            logger.info("Trading system stopped")

async def main():
    """Main entry point."""
    # Configure logging
    logger.add(
        "logs/trading_{time}.log",
        rotation="1 day",
        retention="7 days",
        level=os.getenv("LOG_LEVEL", "INFO")
    )
    
    # Validate API keys
    if not os.getenv("OX_API_KEY") or not os.getenv("OX_API_SECRET"):
        logger.error("OX.FUN API credentials not found in .env file")
        return
    
    # Create and start trading system
    trading_system = TradingSystem()
    await trading_system.start()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Run the main function
    asyncio.run(main()) 