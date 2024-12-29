import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger

class MarketAnalyzer:
    """Handles market analysis and signal generation."""
    
    def __init__(self, price_feed, sentiment_analyzer):
        self.price_feed = price_feed
        self.sentiment_analyzer = sentiment_analyzer
        
        # EMA parameters
        self.short_window = 20  # 20-hour EMA
        self.long_window = 50   # 50-hour EMA
        
        # Market state cache
        self.market_state = {
            'direction': 0.0,
            'sentiment': 0.0,
            'volatility': 0.0,
            'last_update': None
        }
    
    async def analyze_market_direction(self) -> float:
        """Returns market direction score (-1 to 1)."""
        try:
            # Get historical prices for major coins
            symbols = ['BTC/USDT', 'ETH/USDT']
            prices = {}
            
            for symbol in symbols:
                candles = await self.price_feed.get_historical_candles(
                    symbol,
                    limit=self.long_window
                )
                if candles:
                    df = pd.DataFrame(candles)
                    prices[symbol] = df['close'].values
            
            if not prices:
                return 0.0
            
            # Calculate EMAs for each symbol
            directions = []
            for symbol, price_data in prices.items():
                if len(price_data) >= self.long_window:
                    ema_short = pd.Series(price_data).ewm(span=self.short_window).mean().iloc[-1]
                    ema_long = pd.Series(price_data).ewm(span=self.long_window).mean().iloc[-1]
                    
                    # Calculate direction (-1 to 1)
                    direction = (ema_short / ema_long - 1) * 10  # Scale the signal
                    direction = max(min(direction, 1), -1)  # Clamp between -1 and 1
                    directions.append(direction)
            
            # Average direction across symbols
            market_direction = np.mean(directions) if directions else 0.0
            self.market_state['direction'] = market_direction
            return market_direction
            
        except Exception as e:
            logger.error(f"Error analyzing market direction: {e}")
            return 0.0
    
    async def get_sentiment_multiplier(self) -> float:
        """Returns sentiment-based position multiplier (0.5 to 1.5)."""
        try:
            # Get current Fear & Greed Index
            fear_greed = await self.sentiment_analyzer.get_fear_greed_index(datetime.now())
            
            # Convert to 0-1 range and adjust to 0.5-1.5 range
            sentiment_score = fear_greed / 100.0
            multiplier = 0.5 + sentiment_score
            
            self.market_state['sentiment'] = sentiment_score
            return multiplier
            
        except Exception as e:
            logger.error(f"Error getting sentiment multiplier: {e}")
            return 1.0  # Neutral multiplier on error
    
    async def analyze_onchain_metrics(self) -> float:
        """Returns on-chain signal modifier (-0.5 to 0.5)."""
        try:
            # TODO: Implement real on-chain metrics
            # For now, return mock data based on market direction
            direction = self.market_state.get('direction', 0)
            sentiment = self.market_state.get('sentiment', 0.5)
            
            # Combine signals with some randomness
            modifier = (direction * 0.3 + (sentiment - 0.5) * 0.2) + np.random.normal(0, 0.1)
            return max(min(modifier, 0.5), -0.5)
            
        except Exception as e:
            logger.error(f"Error analyzing on-chain metrics: {e}")
            return 0.0
    
    async def analyze_market_depth(self, symbol: str, size: float) -> Dict:
        """
        Analyze order book depth and expected slippage for a given trade size.
        
        Args:
            symbol: Trading pair symbol
            size: Trade size in base currency
            
        Returns:
            Dict containing market depth metrics
        """
        try:
            order_book = await self.price_feed.get_order_book(symbol)
            
            if not order_book or not order_book['bids'] or not order_book['asks']:
                logger.warning(f"Empty order book for {symbol}")
                return {
                    'bid_depth': 0.0,
                    'ask_depth': 0.0,
                    'spread': 0.0,
                    'slippage': 0.0,
                    'is_liquid': False
                }
            
            # Calculate bid and ask depth
            bid_depth = sum(bid[1] for bid in order_book['bids'])
            ask_depth = sum(ask[1] for ask in order_book['asks'])
            
            # Calculate spread
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            spread = (best_ask - best_bid) / best_bid
            
            # Estimate slippage for the given size
            slippage = 0.0
            remaining_size = size
            weighted_price = 0.0
            
            for price, volume in order_book['asks']:
                if remaining_size <= 0:
                    break
                filled = min(remaining_size, volume)
                weighted_price += price * filled
                remaining_size -= filled
            
            if weighted_price > 0:
                slippage = (weighted_price / size - best_ask) / best_ask
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'spread': spread,
                'slippage': slippage,
                'is_liquid': bid_depth > size and ask_depth > size
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market depth: {e}")
            return {
                'bid_depth': 0.0,
                'ask_depth': 0.0,
                'spread': 0.0,
                'slippage': 0.0,
                'is_liquid': False
            }
    
    async def get_market_state(self) -> Dict:
        """Get complete market state analysis."""
        try:
            # Update market state if needed
            if (not self.market_state['last_update'] or 
                datetime.now() - self.market_state['last_update'] > timedelta(hours=1)):
                
                direction = await self.analyze_market_direction()
                sentiment = await self.get_sentiment_multiplier()
                onchain = await self.analyze_onchain_metrics()
                
                self.market_state.update({
                    'direction': direction,
                    'sentiment': sentiment,
                    'onchain': onchain,
                    'last_update': datetime.now()
                })
            
            return self.market_state.copy()
            
        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            return self.market_state.copy() 