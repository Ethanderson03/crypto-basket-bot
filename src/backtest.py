import asyncio
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
from data.ccxt_price_feed import CCXTPriceFeed
from strategy.sentiment_strategy import SentimentStrategy
from dotenv import load_dotenv
import os
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run crypto trading backtest')
    parser.add_argument('--use-cache', action='store_true', default=True,
                      help='Use cached data if available (default: True)')
    parser.add_argument('--no-cache', dest='use_cache', action='store_false',
                      help='Force fetch new data')
    parser.add_argument('--cache-dir', type=str, default='data/cache',
                      help='Directory to store cached data (default: data/cache)')
    return parser.parse_args()

async def run_backtest(args):
    """Run the backtest."""
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Starting backtest setup...")
        
        # Configure backtest parameters
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)
        symbols = ["BTC/USDT", "ETH/USDT", "DOGE/USDT", "SHIB/USDT", "PEPE/USDT"]
        logger.info(f"Configured backtest for {len(symbols)} symbols from {start_date} to {end_date}")
        logger.info(f"Using cached data: {args.use_cache}")
        
        # Initialize price feed
        logger.info("Initializing price feed...")
        price_feed = CCXTPriceFeed(
            mode="backtest",
            backtest_start=start_date,
            backtest_end=end_date,
            use_cached_data=args.use_cache,
            cache_dir=args.cache_dir
        )
        
        # Initialize price feed with symbols
        logger.info("Loading historical data...")
        await price_feed.initialize(symbols)
        logger.info("Historical data loaded successfully")
        
        # Get base collateral from env or use default
        base_collateral = os.getenv("BASE_COLLATERAL")
        try:
            base_collateral = float(base_collateral.split("#")[0].strip())
        except (AttributeError, ValueError, IndexError):
            base_collateral = 10000.0
            logger.warning(f"Using default base collateral: {base_collateral}")
        
        # Initialize strategy
        logger.info("Initializing trading strategy...")
        strategy = SentimentStrategy(
            symbols=symbols,
            price_feed=price_feed,
            base_collateral=base_collateral
        )
        
        # Run simulation
        logger.info("Starting simulation loop...")
        current_time = start_date
        iteration = 0
        while current_time < end_date:
            try:
                # Log progress every 24 hours
                if iteration % 24 == 0:
                    progress = (current_time - start_date) / (end_date - start_date) * 100
                    logger.info(f"Simulation progress: {progress:.1f}% ({current_time})")
                
                # Update strategy
                await strategy.update()
                
                # Advance time
                await price_feed.advance_time(3600)  # 1 hour steps
                current_time += timedelta(hours=1)
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in backtest loop: {e}")
                break
        
        # Print results
        logger.info("Backtest completed successfully")
        logger.info(f"Final portfolio value: {strategy.get_portfolio_value()}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise
    finally:
        logger.info("Cleaning up resources...")
        await price_feed.close()
        logger.info("Cleanup completed")

def main():
    """Main entry point."""
    try:
        args = parse_args()
        logger.info("Starting backtest process...")
        asyncio.run(run_backtest(args))
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 