from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from decimal import Decimal
import pandas as pd

class PositionManager:
    """
    Manages position sizing and portfolio composition for crypto perpetual futures trading.
    """
    
    def __init__(
        self,
        base_leverage: float = 2.0,
        max_position_size: float = 0.15,  # 15% max per position
        min_position_size: float = 0.05,  # 5% min per position
        volatility_lookback: int = 30,    # 30 days for volatility calc
        funding_threshold: float = 0.01,  # 1% daily funding threshold
        min_market_cap: float = 1e9,     # $1B minimum market cap
        correlation_threshold: float = 0.8 # Maximum correlation allowed
    ):
        """
        Initialize the PositionManager.
        
        Args:
            base_leverage: Base leverage for positions
            max_position_size: Maximum position size as fraction of portfolio
            min_position_size: Minimum position size as fraction of portfolio
            volatility_lookback: Days to look back for volatility calculation
            funding_threshold: Maximum acceptable funding rate
            min_market_cap: Minimum market cap for eligible coins
            correlation_threshold: Maximum allowed correlation between positions
        """
        self.base_leverage = base_leverage
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.volatility_lookback = volatility_lookback
        self.funding_threshold = funding_threshold
        self.min_market_cap = min_market_cap
        self.correlation_threshold = correlation_threshold
        
        # Internal state
        self.positions: Dict[str, Dict] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.market_caps: Dict[str, float] = {}
        self.funding_rates: Dict[str, float] = {}
        
    def calculate_base_weights(self) -> Dict[str, float]:
        """
        Calculate optimal coin weights based on market cap, volatility, and correlation.
        
        Returns:
            Dict mapping coin symbols to their optimal weights
        """
        try:
            eligible_coins = self._filter_eligible_coins()
            if not eligible_coins:
                logger.warning("No eligible coins found")
                return {}
            
            # Calculate volatility scores
            volatility_scores = self._calculate_volatility_scores(eligible_coins)
            
            # Calculate market cap scores
            market_cap_scores = self._calculate_market_cap_scores(eligible_coins)
            
            # Calculate correlation penalties
            correlation_penalties = self._calculate_correlation_penalties(eligible_coins)
            
            # Combine scores
            final_scores = {}
            for coin in eligible_coins:
                final_scores[coin] = (
                    volatility_scores.get(coin, 0) *
                    market_cap_scores.get(coin, 0) *
                    (1 - correlation_penalties.get(coin, 0))
                )
            
            # Normalize to get weights
            total_score = sum(final_scores.values())
            if total_score == 0:
                return {}
                
            weights = {
                coin: score / total_score
                for coin, score in final_scores.items()
            }
            
            # Apply position size constraints
            weights = self._apply_position_constraints(weights)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating base weights: {e}")
            return {}
    
    def adjust_for_funding(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust position weights based on funding rates.
        
        Args:
            weights: Base position weights
            
        Returns:
            Adjusted weights accounting for funding rates
        """
        try:
            if not weights:
                return {}
            
            adjusted_weights = {}
            total_adjustment = 0
            
            for coin, weight in weights.items():
                funding_rate = self.funding_rates.get(coin, 0)
                
                # Calculate funding adjustment
                if abs(funding_rate) > self.funding_threshold:
                    # Reduce position size for high funding
                    adjustment = max(0.5, 1 - abs(funding_rate) / self.funding_threshold)
                else:
                    # Increase position size for favorable funding
                    adjustment = min(1.5, 1 + (self.funding_threshold - abs(funding_rate)) / self.funding_threshold)
                
                adjusted_weights[coin] = weight * adjustment
                total_adjustment += adjusted_weights[coin]
            
            # Renormalize weights
            if total_adjustment > 0:
                return {
                    coin: weight / total_adjustment
                    for coin, weight in adjusted_weights.items()
                }
            return weights
            
        except Exception as e:
            logger.error(f"Error adjusting for funding rates: {e}")
            return weights
    
    def calculate_execution_price(
        self,
        symbol: str,
        size: float,
        side: str,
        order_book: Dict
    ) -> float:
        """
        Calculate expected execution price including spread and slippage.
        
        Args:
            symbol: Trading pair symbol
            size: Order size in base currency
            side: 'buy' or 'sell'
            order_book: Current order book state
            
        Returns:
            Expected execution price
        """
        try:
            levels = order_book['asks'] if side == 'buy' else order_book['bids']
            remaining_size = Decimal(str(size))
            total_cost = Decimal('0')
            
            for price, amount in levels:
                price = Decimal(str(price))
                amount = Decimal(str(amount))
                
                if remaining_size <= 0:
                    break
                    
                fill_size = min(remaining_size, amount)
                total_cost += fill_size * price
                remaining_size -= fill_size
            
            if remaining_size > 0:
                return float('inf')
                
            return float(total_cost / Decimal(str(size)))
            
        except Exception as e:
            logger.error(f"Error calculating execution price: {e}")
            return float('inf')
    
    def _filter_eligible_coins(self) -> List[str]:
        """Filter coins based on market cap and other criteria."""
        return [
            coin for coin, cap in self.market_caps.items()
            if cap >= self.min_market_cap and coin in self.market_data
        ]
    
    def _calculate_volatility_scores(self, coins: List[str]) -> Dict[str, float]:
        """Calculate volatility-based scores for each coin."""
        scores = {}
        for coin in coins:
            if coin not in self.market_data:
                continue
                
            df = self.market_data[coin]
            if len(df) < self.volatility_lookback:
                continue
                
            # Calculate daily returns
            returns = df['close'].pct_change().dropna()
            
            # Calculate annualized volatility
            volatility = returns.std() * np.sqrt(365)
            
            # Convert to score (higher volatility = lower score)
            scores[coin] = 1 / (1 + volatility)
            
        return scores
    
    def _calculate_market_cap_scores(self, coins: List[str]) -> Dict[str, float]:
        """Calculate market cap based scores."""
        if not coins:
            return {}
            
        scores = {}
        max_cap = max(self.market_caps[coin] for coin in coins)
        
        for coin in coins:
            scores[coin] = self.market_caps[coin] / max_cap
            
        return scores
    
    def _calculate_correlation_penalties(self, coins: List[str]) -> Dict[str, float]:
        """Calculate correlation-based penalties."""
        penalties = {coin: 0.0 for coin in coins}
        
        for i, coin1 in enumerate(coins):
            if coin1 not in self.market_data:
                continue
                
            returns1 = self.market_data[coin1]['close'].pct_change().dropna()
            
            for coin2 in coins[i+1:]:
                if coin2 not in self.market_data:
                    continue
                    
                returns2 = self.market_data[coin2]['close'].pct_change().dropna()
                
                # Calculate correlation on overlapping periods
                correlation = returns1.corr(returns2)
                
                if correlation > self.correlation_threshold:
                    penalty = (correlation - self.correlation_threshold) / (1 - self.correlation_threshold)
                    penalties[coin1] = max(penalties[coin1], penalty)
                    penalties[coin2] = max(penalties[coin2], penalty)
        
        return penalties
    
    def _apply_position_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum position size constraints."""
        adjusted_weights = {}
        remaining_weight = 1.0
        
        # First pass: apply maximum constraints
        for coin, weight in weights.items():
            adjusted_weights[coin] = min(weight, self.max_position_size)
            remaining_weight -= adjusted_weights[coin]
        
        # Second pass: apply minimum constraints
        coins_below_min = [
            coin for coin, weight in adjusted_weights.items()
            if weight < self.min_position_size
        ]
        
        if coins_below_min:
            # Remove coins below minimum
            for coin in coins_below_min:
                remaining_weight += adjusted_weights.pop(coin)
            
            # Redistribute remaining weight
            if adjusted_weights:
                total_weight = sum(adjusted_weights.values())
                for coin in adjusted_weights:
                    adjusted_weights[coin] *= (1 + remaining_weight / total_weight)
        
        return adjusted_weights 