from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt.async_support as ccxt  # Use async version
import asyncio
import logging
from loguru import logger
import os
import pickle
import json

class CCXTPriceFeed:
    """
    Price feed handler using CCXT library supporting multiple exchanges.
    """
    
    def __init__(
        self,
        mode: str = "paper",  # 'paper' or 'backtest'
        use_cache: bool = True,
        cache_dir: str = "data/cache",
        backtest_start: Optional[datetime] = None,
        backtest_end: Optional[datetime] = None
    ):
        """
        Initialize the price feed.
        
        Args:
            mode: 'paper' for paper trading, 'backtest' for backtesting
            use_cache: Whether to use cached data if available
            cache_dir: Directory to store cached data
            backtest_start: Start date for backtesting
            backtest_end: End date for backtesting
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        
        # Create cache directory if it doesn't exist
        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Initialize exchanges
        self.exchanges = {
            'kucoin': ccxt.kucoin({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
        }
        
        # Historical data cache
        self.historical_data = {}
        
        # Backtesting state
        self.current_time = backtest_start if mode == 'backtest' and backtest_start else None
    
    def _get_cache_path(self, symbol: str) -> str:
        """Get cache file path for a symbol."""
        return os.path.join(self.cache_dir, f"{symbol.replace('/', '_')}.json")
    
    def _save_to_cache(self, symbol: str, candles: List[Dict]) -> None:
        """Save candles to cache file."""
        try:
            if not self.use_cache:
                return
            
            cache_path = self._get_cache_path(symbol)
            
            # Convert timestamps to strings for JSON serialization
            serializable_candles = []
            for candle in candles:
                candle_copy = candle.copy()
                if isinstance(candle_copy['timestamp'], datetime):
                    candle_copy['timestamp'] = candle_copy['timestamp'].isoformat()
                serializable_candles.append(candle_copy)
            
            with open(cache_path, 'w') as f:
                json.dump(serializable_candles, f)
            
            logger.info(f"Saved {len(candles)} candles to cache for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    def _load_from_cache(self, symbol: str) -> List[Dict]:
        """Load candles from cache file."""
        try:
            if not self.use_cache:
                return []
            
            cache_path = self._get_cache_path(symbol)
            if not os.path.exists(cache_path):
                return []
            
            with open(cache_path, 'r') as f:
                candles = json.load(f)
            
            # Convert timestamp strings back to datetime
            for candle in candles:
                if isinstance(candle['timestamp'], str):
                    candle['timestamp'] = datetime.fromisoformat(candle['timestamp'])
            
            logger.info(f"Loaded {len(candles)} candles from cache for {symbol}")
            return candles
            
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            return []
    
    async def initialize(self, symbols: List[str], start_date: datetime = None, end_date: datetime = None, timeframe: str = '1h'):
        """Initialize the price feed with historical data."""
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        
        # Load historical data for each symbol
        for symbol in symbols:
            await self._load_historical_data(symbol)
    
    async def _load_historical_data(self, symbol: str) -> None:
        """Load historical data for a symbol."""
        try:
            logger.info(f"Starting to load historical data for {symbol}")
            
            # Try to load from cache first
            cached_candles = self._load_from_cache(symbol)
            if cached_candles:
                logger.info(f"Found {len(cached_candles)} cached candles for {symbol}")
                # Convert to DataFrame and sort by timestamp
                df = pd.DataFrame(cached_candles)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                self.historical_data[symbol] = df
                logger.info(f"Successfully loaded cached data for {symbol}")
                return
            
            # Try each exchange until we get data
            for exchange_id, exchange in self.exchanges.items():
                try:
                    logger.info(f"Trying {exchange_id} for {symbol}...")
                    
                    # Get historical data in chunks to avoid rate limits
                    all_candles = []
                    current_date = self.backtest_start if self.mode == 'backtest' else datetime(2023, 1, 1)
                    end_date = self.backtest_end if self.mode == 'backtest' else datetime(2024, 1, 1)
                    
                    if not current_date or not end_date:
                        raise ValueError("Backtest start and end dates must be set for backtest mode")
                    
                    while current_date < end_date:
                        chunk_end = min(current_date + timedelta(days=30), end_date)
                        
                        logger.info(f"Fetching {symbol} data from {current_date} to {chunk_end}")
                        
                        # Get candles for this chunk
                        candles = await exchange.fetch_ohlcv(
                            symbol,
                            timeframe='1h',
                            since=int(current_date.timestamp() * 1000),
                            limit=1000
                        )
                        
                        # Convert to our format
                        formatted_candles = []
                        for candle in candles:
                            timestamp = datetime.fromtimestamp(candle[0] / 1000)
                            if timestamp >= current_date and timestamp < chunk_end:
                                formatted_candles.append({
                                    'timestamp': timestamp,
                                    'open': float(candle[1]),
                                    'high': float(candle[2]),
                                    'low': float(candle[3]),
                                    'close': float(candle[4]),
                                    'volume': float(candle[5])
                                })
                        
                        logger.info(f"Got {len(formatted_candles)} candles for {symbol} from {current_date} to {chunk_end}")
                        all_candles.extend(formatted_candles)
                        
                        current_date = chunk_end
                        await asyncio.sleep(exchange.rateLimit / 1000)  # Respect rate limits
                        
                    if all_candles:
                        logger.info(f"Successfully loaded {len(all_candles)} total candles for {symbol} from {exchange_id}")
                        
                        # Save to cache
                        self._save_to_cache(symbol, all_candles)
                        
                        # Convert to DataFrame and sort by timestamp
                        df = pd.DataFrame(all_candles)
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        self.historical_data[symbol] = df
                        
                        logger.info(f"Successfully processed and stored data for {symbol}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to get data from {exchange_id} for {symbol}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {str(e)}")
            raise
    
    async def get_current_price(self, symbol: str, current_time: Optional[datetime] = None) -> float:
        """Get current price for a symbol."""
        try:
            if self.mode == 'backtest':
                if symbol not in self.historical_data:
                    logger.warning(f"No historical data available for {symbol}")
                    return 0.0
                
                df = self.historical_data[symbol]
                if df.empty:
                    logger.warning(f"Empty historical data for {symbol}")
                    return 0.0
                
                # Get the price at the current backtest time
                timestamp = current_time if current_time is not None else self.current_time
                if timestamp is None:
                    logger.warning(f"No current time set for backtest")
                    return 0.0
                
                # Sort index to ensure correct time-based lookup
                df = df.sort_index()
                
                # Get the last price up to the current timestamp
                prices_up_to_now = df[df.index <= timestamp]
                if len(prices_up_to_now) == 0:
                    logger.warning(f"No historical data available for {symbol} at {timestamp}")
                    return 0.0
                    
                last_price = prices_up_to_now['close'].iloc[-1]
                logger.debug(f"Got backtest price for {symbol} at {timestamp}: {last_price}")
                return float(last_price)
            
            # Try each exchange
            for exchange in self.exchanges.values():
                try:
                    ticker = await exchange.fetch_ticker(symbol)
                    return float(ticker['last'])
                except Exception as e:
                    logger.warning(f"Failed to get price from {exchange.id}: {str(e)}")
                    continue
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return 0.0
    
    def _get_backtest_price(self, symbol: str) -> float:
        """Get price for backtesting."""
        try:
            if symbol not in self.historical_data:
                return 0.0
            
            df = self.historical_data[symbol]
            if df.empty:
                return 0.0
            
            return float(df.iloc[-1]['close'])
            
        except Exception as e:
            logger.error(f"Error getting backtest price for {symbol}: {str(e)}")
            return 0.0
    
    async def get_historical_candles(
        self,
        symbol: str,
        limit: int = 100,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get historical candles for a symbol."""
        try:
            if symbol not in self.historical_data:
                return []
            
            df = self.historical_data[symbol]
            if df.empty:
                return []
            
            # Filter data up to end_time if provided
            if end_time is not None:
                df = df[df.index <= end_time]
            
            # Get the last 'limit' candles
            df = df.tail(limit)
            
            # Convert to list of dictionaries
            candles = []
            for idx, row in df.iterrows():
                candle = {
                    'timestamp': idx,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting historical candles for {symbol}: {str(e)}")
            return []
    
    async def get_order_book(self, symbol: str) -> Dict:
        """Get current order book for a symbol."""
        try:
            # In backtest mode, simulate a simple order book
            if self.mode == 'backtest':
                current_price = await self.get_current_price(symbol)
                if current_price == 0:
                    return {'bids': [], 'asks': []}
                
                # Simulate some spread around the current price
                spread = current_price * 0.001  # 0.1% spread
                
                # Generate some fake orders around the current price
                bids = [
                    [current_price - spread * i, 1.0] for i in range(1, 6)
                ]
                asks = [
                    [current_price + spread * i, 1.0] for i in range(1, 6)
                ]
                
                return {
                    'bids': bids,  # [[price, size], ...]
                    'asks': asks   # [[price, size], ...]
                }
            
            # For paper trading, get real order book
            for exchange in self.exchanges.values():
                try:
                    order_book = await exchange.fetch_order_book(symbol)
                    return order_book
                except Exception as e:
                    logger.warning(f"Failed to get order book from {exchange.id}: {str(e)}")
                    continue
            
            return {'bids': [], 'asks': []}
            
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {str(e)}")
            return {'bids': [], 'asks': []} 
    
    async def advance_time(self, seconds: float):
        """Advance time in backtest mode."""
        if self.mode == 'backtest' and self.current_time:
            old_time = self.current_time
            self.current_time += timedelta(seconds=seconds)
            logger.debug(f"Advanced backtest time from {old_time} to {self.current_time} ({seconds} seconds)")
            
            # Log available data range for debugging
            for symbol in self.historical_data:
                df = self.historical_data[symbol]
                if not df.empty:
                    logger.debug(f"Data range for {symbol}: {df.index.min()} to {df.index.max()}")
                    # Log if current time is within data range
                    if df.index.min() <= self.current_time <= df.index.max():
                        logger.debug(f"Current time {self.current_time} is within data range for {symbol}")
                    else:
                        logger.warning(f"Current time {self.current_time} is outside data range for {symbol}")
        else:
            logger.warning("Cannot advance time: not in backtest mode or current_time not set") 