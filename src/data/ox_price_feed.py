from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import asyncio
import aiohttp
import websockets
import hmac
import base64
import hashlib
import json
import time
import os
from dotenv import load_dotenv
import logging

class OXPriceFeed:
    """
    Price feed handler for OX.FUN API supporting both historical and real-time data.
    """
    
    def __init__(
        self,
        mode: str = "paper",  # 'paper' or 'backtest'
        backtest_start: Optional[datetime] = None,
        backtest_end: Optional[datetime] = None,
        is_test: bool = True  # True for testnet, False for mainnet
    ):
        """
        Initialize the price feed.
        
        Args:
            mode: 'paper' for paper trading, 'backtest' for backtesting
            backtest_start: Start date for backtesting
            backtest_end: End date for backtesting
            is_test: Whether to use testnet
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        self.mode = mode
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.is_test = is_test
        
        # API endpoints
        self.ws_url = "wss://stgapi.ox.fun/v2/websocket" if is_test else "wss://api.ox.fun/v2/websocket"
        self.rest_url = "https://stgapi.ox.fun/v3" if is_test else "https://api.ox.fun/v3"
        
        # Load API credentials
        load_dotenv()
        self.api_key = os.getenv("OX_API_KEY")
        self.api_secret = os.getenv("OX_API_SECRET")
        
        # Initialize aiohttp session
        self.session = None
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self.order_books: Dict[str, Dict] = {}
        self.funding_rates: Dict[str, float] = {}
        
        # WebSocket connection
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.subscribed_channels: Dict[str, bool] = {}
        
        # Backtesting state
        self.current_time: Optional[datetime] = None
        if mode == 'backtest' and backtest_start:
            self.current_time = backtest_start
        
        # Rate limiting state
        self._last_request_time = time.time()
        self._request_times = []  # Rolling window of request timestamps
        self._request_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests
    
    async def initialize(self, symbols: List[str]):
        """Initialize data feeds for given symbols."""
        try:
            self.session = aiohttp.ClientSession()
            
            if self.mode == 'backtest':
                await self._load_historical_data(symbols)
            else:
                await self._initialize_real_time_feeds(symbols)
                
            self.logger.info(f"Initialized price feed for {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error initializing price feed: {e}")
            if self.session:
                await self.session.close()
            raise
    
    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None
    
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
                # Initialize empty list to store all candles
                all_candles = []
                
                # Calculate number of 7-day chunks needed
                current_start = self.backtest_start
                while current_start < self.backtest_end:
                    # Calculate chunk end (7 days or remaining time)
                    chunk_end = min(
                        current_start + timedelta(days=7),
                        self.backtest_end
                    )
                    
                    # Convert symbol format (e.g., BTC-USDT -> BTCUSDT)
                    market_code = f"{symbol.replace('-', '')}-SWAP-LIN"
                    
                    # Fetch candle data for this chunk
                    candles = await self._fetch_historical_candles(
                        market_code,
                        int(current_start.timestamp() * 1000),
                        int(chunk_end.timestamp() * 1000)
                    )
                    
                    if candles:
                        all_candles.extend(candles)
                    else:
                        self.logger.warning(f"No candles returned for {symbol} from {current_start} to {chunk_end}")
                    
                    # Move to next chunk
                    current_start = chunk_end
                
                if not all_candles:
                    self.logger.warning(f"No historical data found for {symbol}")
                    continue
                
                # Create DataFrame
                df = pd.DataFrame(all_candles)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                
                self.historical_data[symbol] = df
                self.logger.info(f"Loaded {len(df)} candles for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error loading historical data for {symbol}: {e}")
                raise
    
    async def _initialize_real_time_feeds(self, symbols: List[str]):
        """Initialize real-time data feeds using WebSocket."""
        try:
            # Connect to WebSocket
            self.ws = await websockets.connect(self.ws_url)
            
            # Authenticate if API keys are available
            if self.api_key and self.api_secret:
                await self._authenticate()
            
            # Subscribe to channels for each symbol
            for symbol in symbols:
                # Subscribe to ticker
                await self._subscribe([f"ticker:{symbol}"])
                
                # Subscribe to order book
                await self._subscribe([f"depth:{symbol}"])
                
                # Subscribe to funding rates
                await self._subscribe([f"funding:{symbol}"])
            
            # Start background task to handle WebSocket messages
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            logger.error(f"Error initializing real-time feeds: {e}")
            raise
    
    async def _authenticate(self):
        """Authenticate WebSocket connection."""
        try:
            timestamp = str(int(time.time() * 1000))
            sig_payload = (timestamp + 'GET/auth/self/verify').encode('utf-8')
            signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    sig_payload,
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')
            
            auth_message = {
                "op": "login",
                "tag": "auth",
                "data": {
                    "apiKey": self.api_key,
                    "timestamp": timestamp,
                    "signature": signature
                }
            }
            
            await self.ws.send(json.dumps(auth_message))
            response = await self.ws.recv()
            
            if not json.loads(response).get('success', False):
                raise Exception("Authentication failed")
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise
    
    async def _subscribe(self, channels: List[str]):
        """Subscribe to WebSocket channels."""
        try:
            message = {
                "op": "subscribe",
                "args": channels
            }
            
            await self.ws.send(json.dumps(message))
            response = await self.ws.recv()
            
            if json.loads(response).get('success', False):
                for channel in channels:
                    self.subscribed_channels[channel] = True
            else:
                raise Exception(f"Subscription failed for channels: {channels}")
                
        except Exception as e:
            logger.error(f"Subscription error: {e}")
            raise
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            while True:
                message = await self.ws.recv()
                data = json.loads(message)
                
                if 'event' in data:
                    # Handle subscription responses
                    continue
                
                channel = data.get('channel', '')
                
                if channel.startswith('ticker:'):
                    symbol = channel.split(':')[1]
                    self.current_prices[symbol] = float(data['data']['last'])
                    
                elif channel.startswith('depth:'):
                    symbol = channel.split(':')[1]
                    self.order_books[symbol] = {
                        'bids': [[float(p), float(q)] for p, q in data['data']['bids']],
                        'asks': [[float(p), float(q)] for p, q in data['data']['asks']]
                    }
                    
                elif channel.startswith('funding:'):
                    symbol = channel.split(':')[1]
                    self.funding_rates[symbol] = float(data['data']['fundingRate'])
                
        except Exception as e:
            logger.error(f"Error handling WebSocket messages: {e}")
            raise
    
    async def _respect_rate_limits(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        
        # Clean up request times older than 5 minutes
        self._request_times = [t for t in self._request_times if current_time - t < 300]
        
        # Check 5-minute limit (2500 requests)
        if len(self._request_times) >= 2500:
            sleep_time = 300 - (current_time - self._request_times[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limit approaching, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Ensure 1 request per second minimum spacing
        time_since_last = current_time - self._last_request_time
        if time_since_last < 1:
            await asyncio.sleep(1 - time_since_last)
        
        self._last_request_time = time.time()
        self._request_times.append(self._last_request_time)
    
    async def _fetch_historical_candles(
        self,
        market_code: str,
        start_time: int,
        end_time: int,
    ) -> List[Dict]:
        """Fetch historical candle data with rate limiting"""
        async with self._request_semaphore:  # Limit concurrent requests
            try:
                await self._respect_rate_limits()  # Apply rate limiting before request
                
                endpoint = "/v3/candles"
                params = {
                    "marketCode": market_code,
                    "startTime": start_time,
                    "endTime": end_time,
                    "interval": "1d"  # Daily candles
                }
                
                self.logger.debug(f"Fetching candles for {market_code} from {start_time} to {end_time}")
                signature = self._generate_signature(endpoint, params)
                
                headers = {
                    "apiKey": self.api_key,
                    "signature": signature,
                    "timestamp": str(int(time.time() * 1000))
                }
                
                url = f"{self.rest_url}{endpoint}"
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 429:  # Rate limit exceeded
                        retry_after = int(response.headers.get('Retry-After', 60))
                        self.logger.warning(f"Rate limit exceeded, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._fetch_historical_candles(market_code, start_time, end_time)
                    
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("data", [])
                    
            except Exception as e:
                self.logger.error(f"Error fetching historical candles: {str(e)}")
                raise
    
    def _generate_signature(self, endpoint: str, params: Dict) -> str:
        """Generate API request signature."""
        try:
            # Sort parameters alphabetically and create parameter string
            sorted_params = sorted(params.items())
            param_str = '&'.join([f"{k}={v}" for k, v in sorted_params])
            
            # Get current timestamp
            timestamp = str(int(time.time() * 1000))
            
            # Create signature message
            message = (
                'GET\n' +
                endpoint + '\n' +
                param_str + '\n' +
                timestamp
            )
            
            # Generate HMAC SHA256 signature
            signature = hmac.new(
                base64.b64decode(self.api_secret),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Error generating signature: {e}")
            raise
    
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
        if symbol not in self.historical_data or not self.current_time:
            return 0.0
            
        df = self.historical_data[symbol]
        recent_data = df.loc[:self.current_time].tail(24)  # Last 24 hours
        if len(recent_data) < 24:
            return 0.0
            
        momentum = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
        return min(max(momentum * 0.01, -0.01), 0.01)  # Cap at ±1% 