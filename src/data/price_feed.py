from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from loguru import logger
import asyncio
import aiohttp

class PriceFeed:
    """
    Price feed handler that supports both historical and real-time data.
    """
    
    def __init__(
        self,
        exchange_id: str = "binance",
        mode: str = "paper",  # 'paper' or 'backtest'
        backtest_start: Optional[datetime] = None,
        backtest_end: Optional[datetime] = None
    ):
        """
        Initialize the price feed.
        
        Args:
            exchange_id: ID of the exchange to use
            mode: 'paper' for paper trading, 'backtest' for backtesting
            backtest_start: Start date for backtesting
            backtest_end: End date for backtesting
        """
        self.exchange_id = exchange_id
        self.mode = mode
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        
        # Initialize exchange
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self.order_books: Dict[str, Dict] = {}
        self.funding_rates: Dict[str, float] = {}
        
        # Backtesting state
        self.current_time: Optional[datetime] = None
        if mode == 'backtest' and backtest_start:
            self.current_time = backtest_start
    
    async def initialize(self, symbols: List[str]):
        """Initialize data feeds for given symbols."""
        try:
            if self.mode == 'backtest':
                await self._load_historical_data(symbols)
            else:
                await self._initialize_real_time_feeds(symbols)
                
            logger.info(f"Initialized price feed for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error initializing price feed: {e}")
            raise
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        if self.mode == 'backtest':
            return self._get_backtest_price(symbol)
        return self.current_prices.get(symbol, 0.0)
    
    async def get_order_book(self, symbol: str) -> Dict:
        """Get current order book for a symbol."""
        if self.mode == 'backtest':
            return self._generate_synthetic_order_book(symbol)
        return self.order_books.get(symbol, {'bids': [], 'asks': []})
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a symbol."""
        if self.mode == 'backtest':
            return self._get_backtest_funding_rate(symbol)
        return self.funding_rates.get(symbol, 0.0)
    
    async def advance_time(self, seconds: float):
        """Advance time in backtest mode."""
        if self.mode == 'backtest' and self.current_time:
            self.current_time += timedelta(seconds=seconds)
    
    async def _load_historical_data(self, symbols: List[str]):
        """Load historical data for backtesting."""
        for symbol in symbols:
            try:
                # Fetch OHLCV data
                ohlcv = await self._fetch_historical_ohlcv(
                    symbol,
                    self.backtest_start,
                    self.backtest_end
                )
                
                # Create DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                self.historical_data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error loading historical data for {symbol}: {e}")
    
    async def _initialize_real_time_feeds(self, symbols: List[str]):
        """Initialize real-time data feeds."""
        # This would typically connect to exchange websocket feeds
        # For paper trading, we'll use REST API polling for simplicity
        await self._update_market_data(symbols)
        
        # Start background update task
        asyncio.create_task(self._background_updates(symbols))
    
    async def _background_updates(self, symbols: List[str]):
        """Background task to update market data."""
        while True:
            try:
                await self._update_market_data(symbols)
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _update_market_data(self, symbols: List[str]):
        """Update market data for all symbols."""
        try:
            # Update prices
            tickers = await self._fetch_tickers(symbols)
            for symbol, ticker in tickers.items():
                self.current_prices[symbol] = ticker['last']
            
            # Update order books (limit to avoid rate limits)
            for symbol in symbols[:3]:  # Only update a few symbols per second
                order_book = await self._fetch_order_book(symbol)
                self.order_books[symbol] = order_book
            
            # Update funding rates
            funding_rates = await self._fetch_funding_rates(symbols)
            self.funding_rates.update(funding_rates)
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def _get_backtest_price(self, symbol: str) -> float:
        """Get price for current backtest timestamp."""
        if symbol not in self.historical_data or not self.current_time:
            return 0.0
            
        df = self.historical_data[symbol]
        return float(df.loc[:self.current_time]['close'].iloc[-1])
    
    def _generate_synthetic_order_book(self, symbol: str) -> Dict:
        """Generate synthetic order book for backtesting."""
        price = self._get_backtest_price(symbol)
        if price == 0:
            return {'bids': [], 'asks': []}
            
        # Generate synthetic liquidity
        spread = price * 0.0002  # 0.02% spread
        base_size = price * 10  # $10 worth of base currency
        
        bids = [
            [price - spread/2 - i*spread/10, base_size/(price - spread/2 - i*spread/10)]
            for i in range(10)
        ]
        
        asks = [
            [price + spread/2 + i*spread/10, base_size/(price + spread/2 + i*spread/10)]
            for i in range(10)
        ]
        
        return {
            'bids': bids,
            'asks': asks
        }
    
    def _get_backtest_funding_rate(self, symbol: str) -> float:
        """Generate synthetic funding rate for backtesting."""
        # Simplified funding rate based on price momentum
        if symbol not in self.historical_data or not self.current_time:
            return 0.0
            
        df = self.historical_data[symbol]
        recent_data = df.loc[:self.current_time].tail(24)  # Last 24 hours
        if len(recent_data) < 24:
            return 0.0
            
        momentum = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
        return min(max(momentum * 0.01, -0.01), 0.01)  # Cap at ±1%
    
    async def _fetch_historical_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> List:
        """Fetch historical OHLCV data."""
        try:
            since = int(start.timestamp() * 1000)
            all_ohlcv = []
            
            while True:
                ohlcv = await self._async_fetch_ohlcv(
                    symbol,
                    '1h',
                    since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if since > end.timestamp() * 1000:
                    break
                    
                await asyncio.sleep(self.exchange.rateLimit / 1000)
            
            return all_ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    async def _async_fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int
    ) -> List:
        """Async wrapper for CCXT's fetchOHLCV."""
        return await self.exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since,
            limit
        )
    
    async def _fetch_tickers(self, symbols: List[str]) -> Dict:
        """Fetch current tickers for symbols."""
        return await self.exchange.fetch_tickers(symbols)
    
    async def _fetch_order_book(self, symbol: str) -> Dict:
        """Fetch current order book for symbol."""
        return await self.exchange.fetch_order_book(symbol)
    
    async def _fetch_funding_rates(self, symbols: List[str]) -> Dict:
        """Fetch current funding rates for symbols."""
        rates = {}
        for symbol in symbols:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                rates[symbol] = ticker.get('fundingRate', 0)
            except Exception as e:
                logger.error(f"Error fetching funding rate for {symbol}: {e}")
                rates[symbol] = 0
        return rates 