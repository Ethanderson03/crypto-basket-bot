import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

class RiskManager:
    """Handles risk monitoring and portfolio protection."""
    
    def __init__(self, price_feed):
        self.price_feed = price_feed
        
        # Risk parameters
        self.min_health_ratio = 1.5
        self.max_leverage = 2.0
        self.min_free_collateral = 0.3  # 30% minimum free collateral
        
        # Liquidation protection levels
        self.liquidation_levels = {
            1.4: 0.0,   # Start monitoring
            1.3: 0.25,  # 25% reduction
            1.2: 0.5,   # 50% reduction
            1.1: 1.0    # Full deleveraging
        }
        
        # Portfolio state
        self.positions: Dict[str, Dict] = {}
        self.portfolio_value = 0.0
        self.health_ratio = 2.0
    
    async def calculate_portfolio_health(self) -> float:
        """Returns current portfolio health ratio."""
        try:
            if not self.positions:
                return 2.0  # Maximum health when no positions
            
            total_position_value = 0.0
            total_collateral = 0.0
            
            for symbol, position in self.positions.items():
                current_price = await self.price_feed.get_current_price(symbol)
                
                position_value = position['size'] * current_price
                collateral = position['collateral']
                
                total_position_value += position_value
                total_collateral += collateral
            
            if total_position_value == 0:
                return 2.0
            
            self.health_ratio = total_collateral / total_position_value
            return self.health_ratio
            
        except Exception as e:
            logger.error(f"Error calculating portfolio health: {e}")
            return 1.0
    
    async def get_liquidation_prices(self) -> Dict[str, float]:
        """Returns liquidation prices for all positions."""
        try:
            liquidation_prices = {}
            
            for symbol, position in self.positions.items():
                current_price = await self.price_feed.get_current_price(symbol)
                collateral = position['collateral']
                size = position['size']
                
                if size > 0:
                    # Calculate liquidation price with buffer
                    maintenance_margin = collateral * 0.5  # 50% maintenance margin
                    liquidation_price = current_price * (1 - maintenance_margin / (size * current_price))
                    liquidation_prices[symbol] = liquidation_price
            
            return liquidation_prices
            
        except Exception as e:
            logger.error(f"Error calculating liquidation prices: {e}")
            return {}
    
    async def check_correlation_risks(self) -> Dict:
        """Returns correlation risk metrics."""
        try:
            if len(self.positions) < 2:
                return {'max_correlation': 0.0, 'risk_level': 'low'}
            
            # Get historical prices for correlation calculation
            prices = {}
            for symbol in self.positions:
                candles = await self.price_feed.get_historical_candles(symbol, limit=100)
                if candles:
                    prices[symbol] = [float(c['close']) for c in candles]
            
            # Calculate correlations
            correlations = []
            symbols = list(prices.keys())
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    if len(prices[symbols[i]]) == len(prices[symbols[j]]):
                        corr = np.corrcoef(prices[symbols[i]], prices[symbols[j]])[0, 1]
                        correlations.append(abs(corr))
            
            max_correlation = max(correlations) if correlations else 0.0
            
            return {
                'max_correlation': max_correlation,
                'risk_level': 'high' if max_correlation > 0.8 else 'medium' if max_correlation > 0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error checking correlation risks: {e}")
            return {'max_correlation': 0.0, 'risk_level': 'unknown'}
    
    async def calculate_max_position_size(
        self,
        symbol: str,
        current_price: float,
        order_book: Dict
    ) -> float:
        """Calculate maximum allowed position size."""
        try:
            # Get market depth metrics
            bid_depth = sum(float(bid[1]) for bid in order_book.get('bids', [])[:10])
            ask_depth = sum(float(ask[1]) for ask in order_book.get('asks', [])[:10])
            market_depth = min(bid_depth, ask_depth)
            
            # Calculate size limits
            portfolio_limit = self.portfolio_value * self.max_leverage * 0.15  # 15% max per position
            depth_limit = market_depth * 0.1  # 10% of available liquidity
            
            # Consider current health ratio
            health_factor = min(self.health_ratio / self.min_health_ratio, 1.0)
            
            # Take minimum of all constraints
            max_size = min(
                portfolio_limit,
                depth_limit
            ) * health_factor
            
            return max_size
            
        except Exception as e:
            logger.error(f"Error calculating max position size: {e}")
            return 0.0
    
    async def check_position_change(self, symbol: str, size_delta: float) -> bool:
        """Check if a position change is allowed by risk limits."""
        try:
            # Get current portfolio health
            health_ratio = await self.calculate_portfolio_health()
            
            # Check if we need to deleverage
            for level, reduction in sorted(self.liquidation_levels.items(), reverse=True):
                if health_ratio <= level:
                    if size_delta > 0:  # Don't allow position increases when health is low
                        return False
                    if reduction > 0:  # Force position reduction
                        return True
            
            # Check correlation risks
            correlation_risks = await self.check_correlation_risks()
            if correlation_risks['risk_level'] == 'high' and size_delta > 0:
                return False
            
            # Get current price and order book
            current_price = await self.price_feed.get_current_price(symbol)
            order_book = await self.price_feed.get_order_book(symbol)
            
            # Calculate maximum allowed size
            max_size = await self.calculate_max_position_size(
                symbol,
                current_price,
                order_book
            )
            
            # Check if new position would exceed limits
            current_size = self.positions.get(symbol, {}).get('size', 0)
            new_size = current_size + size_delta
            
            return 0 <= new_size <= max_size
            
        except Exception as e:
            logger.error(f"Error checking position change: {e}")
            return False
    
    def update_portfolio_state(self, positions: Dict[str, Dict], portfolio_value: float):
        """Update internal portfolio state."""
        self.positions = positions
        self.portfolio_value = portfolio_value 