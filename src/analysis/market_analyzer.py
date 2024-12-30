import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from loguru import logger
from src.data.ccxt_price_feed import CCXTPriceFeed
from src.analysis.order_book_analyzer import OrderBookAnalyzer
from src.analysis.sentiment_analyzer import SentimentAnalyzer

class MarketAnalyzer:
    def __init__(self, short_window: int = 24, long_window: int = 72):
        self.short_window = short_window
        self.long_window = long_window
        self.market_state = {
            'direction': 0.0,
            'coin_directions': {},
            'sentiment': 1.0,
            'fear_greed': 50.0,
            'basket_weights': {},
            'correlations': {},
            'is_oversold': False,
            'is_overbought': False,
            'volatility_multiplier': 1.0,
            'last_update': None
        }
    
    async def update(self, price_feed: CCXTPriceFeed, current_time: datetime) -> None:
        """Update market state with latest data."""
        logger.info("Starting market direction analysis...")
        
        # Get symbols from price feed
        symbols = list(price_feed.historical_data.keys())
        
        # Calculate basket weights based on market caps
        basket_weights = await self._calculate_basket_weights(price_feed, current_time)
        logger.info(f"Using basket weights: {basket_weights}")
        
        # Calculate direction for each coin
        coin_directions = {}
        for symbol in symbols:
            logger.info(f"\nAnalyzing direction for {symbol}...")
            
            # Get historical candles
            candles = await price_feed.get_historical_candles(
                    symbol,
                limit=self.long_window,
                end_time=current_time
            )
            logger.info(f"Got {len(candles)} candles for {symbol}")
            
            # Extract prices
            prices = [candle['close'] for candle in candles]
            logger.info(f"Price range for {symbol}: {max(prices)} to {min(prices)}")
            
            # Calculate EMAs
            ema_short = pd.Series(prices).ewm(span=self.short_window).mean().iloc[-1]
            ema_long = pd.Series(prices).ewm(span=self.long_window).mean().iloc[-1]
            
            logger.info(f"{symbol} EMAs:")
            logger.info(f"  Short ({self.short_window}): {ema_short}")
            logger.info(f"  Long ({self.long_window}): {ema_long}")
            
            # Calculate raw direction
            raw_direction = (ema_short / ema_long - 1) * 10
            logger.info(f"{symbol} raw direction: {raw_direction}")
            
            # Calculate momentum
            momentum = self._calculate_momentum(prices)
            logger.info(f"{symbol} momentum: {momentum}")
            
            # Adjust direction based on momentum
            direction = raw_direction * (1 + momentum * 0.2)
            logger.info(f"{symbol} direction after momentum: {direction}")
            
            coin_directions[symbol] = direction
            
        # Calculate market direction as weighted average
        market_direction = 0.0
        for symbol, direction in coin_directions.items():
            market_direction += direction * basket_weights[symbol]
            
        logger.info(f"\nFinal market direction: {market_direction}")
        logger.info(f"Individual directions: {coin_directions}")
        
        # Update market state
        logger.info(f"Market direction: {market_direction}, Coin directions: {coin_directions}")
        
        # Calculate sentiment multiplier (1.0 for now)
        sentiment_multiplier = 1.0
        logger.info(f"Sentiment multiplier: {sentiment_multiplier}")
        
        # Get Fear & Greed Index (50 for now)
        fear_greed = 50.0
        logger.info(f"Fear & Greed Index: {fear_greed}")
        
        # Calculate correlations
        logger.info(f"Basket weights: {basket_weights}")
        correlations = await self._calculate_correlations(price_feed, symbols, current_time)
        
        # Update market state
        self.market_state.update({
            'direction': market_direction,
            'coin_directions': coin_directions,
            'sentiment': sentiment_multiplier,
            'fear_greed': fear_greed,
            'basket_weights': basket_weights,
            'correlations': correlations,
            'is_oversold': market_direction < -0.5,
            'is_overbought': market_direction > 0.5,
            'volatility_multiplier': self._calculate_volatility_multiplier(prices),
            'last_update': current_time
        })
        
        logger.info(f"Updated market state: {self.market_state}")
        
    async def get_trading_signals(self, price_feed, current_time) -> Dict[str, float]:
        """Generate trading signals based on market state."""
        signals = {}
        
        for symbol in self.market_state['coin_directions'].keys():
            coin_direction = self.market_state['coin_directions'][symbol]
            base_signal = coin_direction * self.market_state['sentiment'] * 330  # Increased multiplier
            
            # Log signal calculation
            logger.debug(f"\nSignal calculation for {symbol}:")
            logger.debug(f"Coin direction: {coin_direction:.4f}")
            logger.debug(f"Sentiment: {self.market_state['sentiment']:.4f}")
            logger.debug(f"Base signal: {base_signal:.4f}")
            
            if abs(base_signal) >= 0.0000000000000000000000001:  # Very low threshold for testing
                signals[symbol] = base_signal
                logger.info(f"Generated signal for {symbol}: {base_signal:.4f}")
            else:
                logger.info(f"Signal for {symbol} ({base_signal:.4f}) below threshold, not generating")
        
        return signals
        
    async def _calculate_basket_weights(self, price_feed: CCXTPriceFeed, current_time: datetime) -> Dict[str, float]:
        """Calculate basket weights based on market caps."""
        weights = {}
        total_mcap = 0.0
        
        # Get symbols from price feed
        symbols = list(price_feed.historical_data.keys())
        
        # Calculate market caps
        for symbol in symbols:
            price = await price_feed.get_current_price(symbol, current_time)
            # Use price as proxy for market cap (simplified)
            mcap = price
            total_mcap += mcap
            weights[symbol] = mcap
            
        # Normalize weights
        if total_mcap > 0:
            for symbol in symbols:
                weights[symbol] /= total_mcap
                
        return weights
        
    def _calculate_momentum(self, prices: List[float], window: int = 14) -> float:
        """Calculate momentum indicator."""
        if len(prices) < window:
            return 0.0
            
        # Calculate rate of change
        roc = (prices[-1] - prices[-window]) / prices[-window]
        return roc
        
    def _calculate_volatility_multiplier(self, prices: List[float], window: int = 14) -> float:
        """Calculate volatility multiplier."""
        if len(prices) < window:
            return 1.0
            
        # Calculate standard deviation
        std = pd.Series(prices[-window:]).std()
        mean = pd.Series(prices[-window:]).mean()
        
        # Normalize volatility
        volatility = std / mean
        
        # Convert to multiplier (lower volatility = higher multiplier)
        multiplier = 1.0 / (1.0 + volatility)
        
        return multiplier
        
    async def _calculate_correlations(self, price_feed: CCXTPriceFeed, symbols: List[str], current_time: datetime) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between symbols."""
        correlations = {}
        
        # Get historical prices for all symbols
        prices = {}
        for symbol in symbols:
            candles = await price_feed.get_historical_candles(
                symbol,
                limit=self.long_window,
                end_time=current_time
            )
            prices[symbol] = [candle['close'] for candle in candles]
            
        # Calculate correlations
        for symbol1 in symbols:
            correlations[symbol1] = {}
            for symbol2 in symbols:
                if len(prices[symbol1]) == len(prices[symbol2]):
                    corr = pd.Series(prices[symbol1]).corr(pd.Series(prices[symbol2]))
                else:
                    corr = 0.0
                correlations[symbol1][symbol2] = corr
                
        return correlations 