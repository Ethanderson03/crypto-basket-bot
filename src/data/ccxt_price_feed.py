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

class CCXTPriceFeed:
    """
    Price feed handler using CCXT library supporting multiple exchanges.
    """
    
    def __init__(
        self,
        mode: str = "paper",  # 'paper' or 'backtest'
        backtest_start: Optional[datetime] = None,
        backtest_end: Optional[datetime] = None,
        exchanges: List[str] = ["kucoin", "gate", "huobi"],  # Default exchanges that work in most regions
        use_cached_data: bool = True,
        cache_dir: str = "src/data/cache"
    ):
        """
        Initialize the price feed.
        
        Args:
            mode: 'paper' for paper trading, 'backtest' for backtesting
            backtest_start: Start date for backtesting
            backtest_end: End date for backtesting
            exchanges: List of CCXT exchange ids to use
            use_cached_data: Whether to use cached data if available
            cache_dir: Directory to store cached data
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.use_cached_data = use_cached_data
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if use_cached_data:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize exchanges
        self.exchanges = {}
        self.exchange_ids = exchanges
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self.current_time: Optional[datetime] = None
        
        if mode == 'backtest' and backtest_start:
            self.current_time = backtest_start
    
    def _get_cache_path(self, symbol: str) -> str:
        """Get the cache file path for a symbol."""
        # Replace / with _ in symbol name for file path
        safe_symbol = symbol.replace('/', '_')
        return os.path.join(self.cache_dir, f"{safe_symbol}.pkl")
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save data to cache."""
        if not self.use_cached_data:
            return
        
        try:
            cache_path = self._get_cache_path(symbol)
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Saved {len(df)} candles to cache for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        if not self.use_cached_data:
            return None
        
        try:
            cache_path = self._get_cache_path(symbol)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                logger.info(f"Loaded {len(df)} candles from cache for {symbol}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
        return None
    
    async def _init_exchanges(self):
        """Initialize exchange connections."""
        for exchange_id in self.exchange_ids:
            try:
                # Get the exchange class
                exchange_class = getattr(ccxt, exchange_id)
                # Initialize exchange
                self.exchanges[exchange_id] = exchange_class({
                    'enableRateLimit': True,  # Enable built-in rate limiter
                })
                await self.exchanges[exchange_id].load_markets()
                self.logger.info(f"Initialized {exchange_id} connection")
            except Exception as e:
                self.logger.error(f"Error initializing {exchange_id}: {e}")
    
    async def initialize(self, symbols: List[str]):
        """Initialize data feeds for given symbols."""
        try:
            # Initialize exchange connections
            await self._init_exchanges()
            
            if self.mode == 'backtest':
                await self._load_historical_data(symbols)
            else:
                await self._initialize_real_time_feeds(symbols)
            
            self.logger.info(f"Initialized price feed for {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error initializing price feed: {e}")
            raise
    
    async def _load_historical_data(self, symbols: List[str]):
        """Load historical data from cache or exchanges."""
        logger.info(f"Starting to load historical data for {len(symbols)} symbols")
        
        for symbol in symbols:
            logger.info(f"Loading data for {symbol}...")
            try:
                # Try loading from cache first
                df = self._load_from_cache(symbol)
                if df is not None and not self.use_cached_data:
                    self.historical_data[symbol] = df
                    continue
                
                # Initialize empty list to store all candles
                all_candles = []
                best_exchange = None
                
                # Try each exchange until we get data
                for exchange_id, exchange in self.exchanges.items():
                    logger.info(f"Trying {exchange_id} for {symbol}...")
                    try:
                        # Calculate time chunks (exchanges often limit single requests)
                        current_start = self.backtest_start
                        while current_start < self.backtest_end:
                            chunk_end = min(
                                current_start + timedelta(days=30),  # 30-day chunks
                                self.backtest_end
                            )
                            
                            logger.info(f"Fetching {symbol} data from {current_start} to {chunk_end}")
                            
                            # Fetch candles
                            candles = await exchange.fetch_ohlcv(
                                symbol,
                                timeframe='1d',
                                since=int(current_start.timestamp() * 1000),
                                limit=None  # Let exchange decide limit
                            )
                            
                            if candles:
                                logger.info(f"Got {len(candles)} candles for {symbol}")
                                all_candles.extend(candles)
                                best_exchange = exchange_id
                            else:
                                logger.warning(f"No candles returned for {symbol} in this chunk")
                            
                            current_start = chunk_end
                            
                            # Respect rate limits
                            await exchange.sleep(exchange.rateLimit / 1000)
                        
                        if all_candles:
                            logger.info(f"Successfully loaded data for {symbol} from {exchange_id}")
                            break  # Stop trying other exchanges if we got data
                            
                    except Exception as e:
                        logger.warning(f"Failed to get data from {exchange_id} for {symbol}: {str(e)}")
                        continue
                
                if not all_candles:
                    logger.warning(f"No historical data found for {symbol} on any exchange")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    all_candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                
                self.historical_data[symbol] = df
                logger.info(f"Successfully processed {len(df)} candles for {symbol} from {best_exchange}")
                
                # Save to cache
                self._save_to_cache(symbol, df)
                
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {str(e)}")
                raise
        
        logger.info("Completed loading historical data for all symbols")
    
    async def _initialize_real_time_feeds(self, symbols: List[str]):
        """Initialize real-time price tracking."""
        # For paper trading, we'll poll prices periodically
        for exchange_id, exchange in self.exchanges.items():
            asyncio.create_task(self._poll_prices(exchange, symbols))
    
    async def _poll_prices(self, exchange: ccxt.Exchange, symbols: List[str]):
        """Poll current prices periodically."""
        while True:
            try:
                for symbol in symbols:
                    ticker = await exchange.fetch_ticker(symbol)
                    self.current_prices[symbol] = ticker['last']
                
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                self.logger.error(f"Error polling prices: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        if self.mode == 'backtest':
            return self._get_backtest_price(symbol)
        return self.current_prices.get(symbol, 0.0)
    
    def _get_backtest_price(self, symbol: str) -> float:
        """Get price for current backtest timestamp."""
        if symbol not in self.historical_data or not self.current_time:
            return 0.0
        
        df = self.historical_data[symbol]
        try:
            # Get all prices up to current time and take the last one
            # This handles duplicate timestamps by using the latest value
            mask = df.index <= self.current_time
            if not mask.any():
                return 0.0
            prices = df[mask]['close']
            return float(prices.iloc[-1])
        except Exception as e:
            logger.error(f"Error getting backtest price for {symbol}: {e}")
            return 0.0
    
    async def advance_time(self, seconds: float):
        """Advance time in backtest mode."""
        if self.mode == 'backtest' and self.current_time:
            self.current_time += timedelta(seconds=seconds)
    
    async def close(self):
        """Clean up resources."""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except:
                pass  # Some exchanges might not have close method 
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Get order book data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels to fetch
            
        Returns:
            Dict containing bids and asks with prices and volumes
        """
        if self.mode == 'backtest':
            # For backtesting, simulate a basic order book around the current price
            current_price = self._get_backtest_price(symbol)
            if current_price == 0.0:
                return {'bids': [], 'asks': []}
            
            # Generate simulated order book levels
            spread = current_price * 0.001  # 0.1% spread
            bids = [[current_price - spread * i, 1.0 / (i + 1)] for i in range(limit)]
            asks = [[current_price + spread * i, 1.0 / (i + 1)] for i in range(limit)]
            
            return {
                'bids': bids,  # [[price, volume], ...]
                'asks': asks   # [[price, volume], ...]
            }
        
        # For paper trading, try to get real order book data
        for exchange_id, exchange in self.exchanges.items():
            try:
                order_book = await exchange.fetch_order_book(symbol, limit=limit)
                return {
                    'bids': order_book['bids'],
                    'asks': order_book['asks']
                }
            except Exception as e:
                logger.warning(f"Failed to get order book from {exchange_id} for {symbol}: {e}")
                continue
        
        # If all exchanges fail, return empty order book
        return {'bids': [], 'asks': []} 