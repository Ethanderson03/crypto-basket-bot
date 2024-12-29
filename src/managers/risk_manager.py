from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from decimal import Decimal
import pandas as pd

class RiskManager:
    """
    Handles risk monitoring and portfolio protection for crypto perpetual futures trading.
    """
    
    def __init__(
        self,
        min_health_ratio: float = 1.5,
        health_ratio_thresholds: Dict[float, float] = {
            1.4: 0.0,   # Start monitoring
            1.3: 0.25,  # 25% reduction
            1.2: 0.50,  # 50% reduction
            1.1: 1.00   # Full deleveraging
        },
        max_position_size: float = 0.15,  # 15% of portfolio
        min_free_collateral: float = 0.30,  # 30% minimum free collateral
        max_drawdown: float = 0.25,      # 25% maximum drawdown
        correlation_limit: float = 0.8    # Maximum portfolio correlation
    ):
        """
        Initialize the RiskManager.
        
        Args:
            min_health_ratio: Minimum required health ratio
            health_ratio_thresholds: Dict mapping health ratios to required position reductions
            max_position_size: Maximum position size as fraction of portfolio
            min_free_collateral: Minimum required free collateral
            max_drawdown: Maximum allowed drawdown
            correlation_limit: Maximum allowed correlation between positions
        """
        self.min_health_ratio = min_health_ratio
        self.health_ratio_thresholds = health_ratio_thresholds
        self.max_position_size = max_position_size
        self.min_free_collateral = min_free_collateral
        self.max_drawdown = max_drawdown
        self.correlation_limit = correlation_limit
        
        # Internal state
        self.positions: Dict[str, Dict] = {}
        self.collateral: float = 0
        self.market_prices: Dict[str, float] = {}
        self.historical_pnl: List[float] = []
        self.position_correlations: Dict[str, Dict[str, float]] = {}
    
    def calculate_portfolio_health(self) -> float:
        """
        Calculate current portfolio health ratio.
        
        Returns:
            float: Current portfolio health ratio
        """
        try:
            if not self.positions:
                return float('inf')
            
            total_position_value = 0
            total_margin_used = 0
            
            for symbol, position in self.positions.items():
                current_price = self.market_prices.get(symbol)
                if not current_price:
                    logger.warning(f"No price data for {symbol}")
                    continue
                
                position_size = abs(position['size'])
                position_value = position_size * current_price
                margin_used = position_value / position['leverage']
                
                total_position_value += position_value
                total_margin_used += margin_used
            
            if total_margin_used == 0:
                return float('inf')
                
            return self.collateral / total_margin_used
            
        except Exception as e:
            logger.error(f"Error calculating portfolio health: {e}")
            return 0.0
    
    def get_liquidation_prices(self) -> Dict[str, float]:
        """
        Calculate liquidation prices for all positions.
        
        Returns:
            Dict mapping symbols to their liquidation prices
        """
        try:
            liquidation_prices = {}
            
            for symbol, position in self.positions.items():
                current_price = self.market_prices.get(symbol)
                if not current_price:
                    continue
                
                position_size = position['size']
                leverage = position['leverage']
                margin = abs(position_size * current_price / leverage)
                
                # Calculate maintenance margin requirement (simplified)
                maintenance_margin = margin * 0.05  # 5% maintenance margin
                
                # Calculate liquidation price
                if position_size > 0:  # Long position
                    liquidation_price = current_price * (1 - (margin - maintenance_margin) / (position_size * current_price))
                else:  # Short position
                    liquidation_price = current_price * (1 + (margin - maintenance_margin) / (abs(position_size) * current_price))
                
                liquidation_prices[symbol] = float(liquidation_price)
            
            return liquidation_prices
            
        except Exception as e:
            logger.error(f"Error calculating liquidation prices: {e}")
            return {}
    
    def check_correlation_risks(self) -> Dict:
        """
        Analyze correlation risks in the portfolio.
        
        Returns:
            Dict containing correlation risk metrics
        """
        try:
            high_correlation_pairs = []
            max_correlation = 0.0
            avg_correlation = 0.0
            correlation_count = 0
            
            symbols = list(self.positions.keys())
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = self.position_correlations.get(symbol1, {}).get(symbol2, 0)
                    
                    if correlation > self.correlation_limit:
                        high_correlation_pairs.append({
                            'pair': (symbol1, symbol2),
                            'correlation': correlation
                        })
                    
                    max_correlation = max(max_correlation, correlation)
                    avg_correlation += correlation
                    correlation_count += 1
            
            if correlation_count > 0:
                avg_correlation /= correlation_count
            
            return {
                'high_correlation_pairs': high_correlation_pairs,
                'max_correlation': float(max_correlation),
                'average_correlation': float(avg_correlation),
                'correlation_risk_level': 'high' if high_correlation_pairs else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error checking correlation risks: {e}")
            return {
                'high_correlation_pairs': [],
                'max_correlation': 0.0,
                'average_correlation': 0.0,
                'correlation_risk_level': 'unknown'
            }
    
    def calculate_max_position_size(
        self,
        symbol: str,
        current_price: float,
        order_book: Dict
    ) -> float:
        """
        Calculate maximum safe position size based on multiple constraints.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            order_book: Current order book state
            
        Returns:
            float: Maximum allowed position size in base currency
        """
        try:
            # Calculate various limits
            portfolio_limit = self.collateral * self.max_position_size
            
            # Calculate free collateral limit
            used_collateral = sum(
                abs(pos['size']) * self.market_prices.get(sym, 0) / pos['leverage']
                for sym, pos in self.positions.items()
            )
            free_collateral = self.collateral - used_collateral
            free_collateral_limit = free_collateral * (1 - self.min_free_collateral)
            
            # Calculate market depth limit
            depth_limit = self._calculate_depth_limit(order_book)
            
            # Calculate liquidation risk limit
            liquidation_limit = self._calculate_liquidation_risk_limit(
                symbol,
                current_price
            )
            
            # Take minimum of all limits
            max_size = min(
                portfolio_limit / current_price,
                free_collateral_limit / current_price,
                depth_limit,
                liquidation_limit
            )
            
            return float(max_size)
            
        except Exception as e:
            logger.error(f"Error calculating maximum position size: {e}")
            return 0.0
    
    def _calculate_depth_limit(self, order_book: Dict) -> float:
        """Calculate position size limit based on market depth."""
        try:
            # Sum up available liquidity within acceptable price impact
            max_price_impact = 0.02  # 2% maximum price impact
            total_liquidity = 0
            
            for price, size in order_book['bids']:
                if price < order_book['bids'][0][0] * (1 - max_price_impact):
                    break
                total_liquidity += size
                
            return float(total_liquidity)
            
        except Exception as e:
            logger.error(f"Error calculating depth limit: {e}")
            return 0.0
    
    def _calculate_liquidation_risk_limit(
        self,
        symbol: str,
        current_price: float
    ) -> float:
        """Calculate position size limit based on liquidation risk."""
        try:
            # Calculate maximum position size that keeps liquidation price
            # at least 20% away from current price
            min_distance = 0.20  # 20% minimum distance to liquidation
            leverage = 2.0  # Default leverage
            
            if symbol in self.positions:
                leverage = self.positions[symbol]['leverage']
            
            # For a long position:
            # liquidation_price = entry_price * (1 - (margin - maintenance_margin) / (position_size * entry_price))
            # Solving for position_size:
            maintenance_margin_rate = 0.05  # 5%
            available_margin = self.collateral * (1 - self.min_free_collateral)
            
            max_size = (available_margin * (1 - maintenance_margin_rate)) / (current_price * min_distance)
            
            return float(max_size)
            
        except Exception as e:
            logger.error(f"Error calculating liquidation risk limit: {e}")
            return 0.0 