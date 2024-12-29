from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger
import requests
from datetime import datetime, timedelta

class MarketAnalyzer:
    """
    Handles market analysis and signal generation for crypto perpetual futures trading.
    
    This class is responsible for analyzing market sentiment, direction, and various
    other metrics to generate trading signals.
    """
    
    def __init__(
        self,
        fear_greed_api_url: str = "https://api.alternative.me/fng/",
        ema_periods: List[int] = [20, 50, 200],
        sentiment_lookback_days: int = 30,
        top_coins: List[str] = ["BTC", "ETH", "BNB", "SOL", "XRP"]
    ):
        """
        Initialize the MarketAnalyzer.
        
        Args:
            fear_greed_api_url: URL for the Fear & Greed Index API
            ema_periods: List of periods for EMA calculations
            sentiment_lookback_days: Days to look back for sentiment analysis
            top_coins: List of top coins to analyze
        """
        self.fear_greed_api_url = fear_greed_api_url
        self.ema_periods = ema_periods
        self.sentiment_lookback_days = sentiment_lookback_days
        self.top_coins = top_coins
        self.price_data: Dict[str, pd.DataFrame] = {}
        
    def analyze_market_direction(self) -> float:
        """
        Analyzes overall market direction using EMAs across top coins.
        
        Returns:
            float: Market direction score between -1 (bearish) and 1 (bullish)
        """
        direction_scores = []
        
        for coin in self.top_coins:
            if coin not in self.price_data:
                logger.warning(f"No price data available for {coin}")
                continue
                
            df = self.price_data[coin]
            
            # Calculate EMAs
            ema_signals = []
            for period in self.ema_periods:
                ema = df['close'].ewm(span=period).mean()
                # Compare current price to EMA
                signal = 1 if df['close'].iloc[-1] > ema.iloc[-1] else -1
                ema_signals.append(signal)
            
            # Weight shorter EMAs more heavily
            weights = np.array([3, 2, 1])
            coin_score = np.average(ema_signals, weights=weights)
            direction_scores.append(coin_score)
        
        if not direction_scores:
            logger.error("No direction scores could be calculated")
            return 0.0
            
        # Average across all coins
        return float(np.mean(direction_scores))
    
    def get_sentiment_multiplier(self) -> float:
        """
        Calculates position size multiplier based on Fear & Greed Index.
        
        Returns:
            float: Sentiment multiplier between 0.5 and 1.5
        """
        try:
            response = requests.get(self.fear_greed_api_url)
            data = response.json()
            fear_greed_value = float(data['data'][0]['value'])
            
            # Convert 0-100 scale to 0.5-1.5 multiplier
            # Using contrarian approach: high fear = larger positions
            normalized = (100 - fear_greed_value) / 100  # Invert scale
            multiplier = 0.5 + normalized
            
            return min(1.5, max(0.5, multiplier))
            
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return 1.0  # Neutral multiplier on error
    
    def analyze_onchain_metrics(self) -> float:
        """
        Analyzes on-chain metrics for signal generation.
        
        Returns:
            float: On-chain signal modifier between -0.5 and 0.5
        """
        try:
            metrics = self._fetch_onchain_metrics()
            
            # Analyze whale movements
            whale_signal = self._analyze_whale_movements(metrics['whale_transactions'])
            
            # Network activity
            activity_signal = self._analyze_network_activity(metrics['daily_active_addresses'])
            
            # Exchange flows
            flow_signal = self._analyze_exchange_flows(metrics['exchange_flows'])
            
            # Combine signals with weights
            weights = {
                'whale': 0.5,
                'activity': 0.3,
                'flow': 0.2
            }
            
            combined_signal = (
                whale_signal * weights['whale'] +
                activity_signal * weights['activity'] +
                flow_signal * weights['flow']
            )
            
            # Ensure output is between -0.5 and 0.5
            return float(max(-0.5, min(0.5, combined_signal)))
            
        except Exception as e:
            logger.error(f"Error analyzing on-chain metrics: {e}")
            return 0.0
    
    def analyze_market_depth(self, symbol: str, size: float) -> Dict:
        """
        Analyzes order book depth and expected slippage.
        
        Args:
            symbol: Trading pair symbol
            size: Order size to analyze
            
        Returns:
            dict: Analysis results including slippage and liquidity metrics
        """
        try:
            order_book = self._fetch_order_book(symbol)
            
            # Calculate spread
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Analyze depth
            cumulative_bids = self._calculate_cumulative_depth(order_book['bids'])
            cumulative_asks = self._calculate_cumulative_depth(order_book['asks'])
            
            # Calculate expected slippage
            slippage = self._estimate_slippage(size, cumulative_bids, cumulative_asks)
            
            # Check if there's sufficient liquidity
            sufficient_liquidity = (
                cumulative_bids[-1] >= size and
                cumulative_asks[-1] >= size
            )
            
            return {
                'expected_slippage': float(slippage),
                'spread': float(spread),
                'sufficient_liquidity': sufficient_liquidity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market depth: {e}")
            return {
                'expected_slippage': float('inf'),
                'spread': float('inf'),
                'sufficient_liquidity': False
            }
    
    def _fetch_onchain_metrics(self) -> Dict:
        """Mock function to fetch on-chain metrics."""
        # In a real implementation, this would fetch data from blockchain APIs
        return {
            'whale_transactions': [],
            'daily_active_addresses': [],
            'exchange_flows': []
        }
    
    def _analyze_whale_movements(self, transactions: List) -> float:
        """Analyzes whale transaction patterns."""
        # Implementation would analyze large transactions
        return 0.0
    
    def _analyze_network_activity(self, active_addresses: List) -> float:
        """Analyzes network activity trends."""
        # Implementation would analyze network usage
        return 0.0
    
    def _analyze_exchange_flows(self, flows: List) -> float:
        """Analyzes exchange inflow/outflow patterns."""
        # Implementation would analyze exchange flows
        return 0.0
    
    def _fetch_order_book(self, symbol: str) -> Dict:
        """Mock function to fetch order book data."""
        # In real implementation, this would fetch from exchange API
        return {
            'bids': [[0, 0]],
            'asks': [[0, 0]]
        }
    
    def _calculate_cumulative_depth(self, levels: List[List[float]]) -> np.ndarray:
        """Calculates cumulative depth from order book levels."""
        return np.cumsum([level[1] for level in levels])
    
    def _estimate_slippage(
        self,
        size: float,
        cumulative_bids: np.ndarray,
        cumulative_asks: np.ndarray
    ) -> float:
        """Estimates slippage for given order size."""
        # Implementation would calculate expected slippage
        return 0.0 