import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

class PositionManager:
    """Manages position sizing and portfolio composition."""
    
    def __init__(self, market_analyzer, risk_manager):
        self.market_analyzer = market_analyzer
        self.risk_manager = risk_manager
        
        # Position sizing parameters
        self.base_position_size = 1.0
        self.max_leverage = 2.0
        self.min_position_pct = 0.05  # 5% of portfolio
        self.max_position_pct = 0.15  # 15% of portfolio
        
        # Portfolio state
        self.current_weights: Dict[str, float] = {}
        self.target_weights: Dict[str, float] = {}
    
    async def calculate_base_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Returns optimal coin weights."""
        try:
            weights = {}
            total_weight = 0.0
            
            for symbol in symbols:
                # Get market analysis
                depth = await self.market_analyzer.analyze_market_depth(
                    symbol,
                    self.base_position_size
                )
                
                # Get market state
                market_state = await self.market_analyzer.get_market_state()
                
                # Calculate base weight
                weight = 1.0
                
                # Adjust for liquidity
                if depth['sufficient_liquidity']:
                    weight *= (1.0 - depth['spread'])  # Reduce weight for high spread
                else:
                    weight *= 0.5  # Significantly reduce weight for low liquidity
                
                # Adjust for market direction
                direction_factor = 1.0 + market_state['direction']  # Range: 0.0-2.0
                weight *= direction_factor
                
                # Adjust for sentiment
                sentiment_factor = market_state['sentiment']  # Range: 0.5-1.5
                weight *= sentiment_factor
                
                weights[symbol] = max(weight, 0.0)
                total_weight += weights[symbol]
            
            # Normalize weights
            if total_weight > 0:
                for symbol in weights:
                    weights[symbol] /= total_weight
                    
                    # Apply position size limits
                    weights[symbol] = max(
                        min(weights[symbol], self.max_position_pct),
                        self.min_position_pct
                    )
            
            self.target_weights = weights
            return weights.copy()
            
        except Exception as e:
            logger.error(f"Error calculating base weights: {e}")
            return {symbol: 1.0 / len(symbols) for symbol in symbols}
    
    async def adjust_for_funding(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Adjusts weights based on funding rates."""
        try:
            adjusted_weights = weights.copy()
            
            for symbol in weights:
                # Get funding rate (mock for now)
                funding_rate = 0.01 * np.random.normal()  # Mock funding rate
                
                # Adjust weight based on funding rate
                if abs(funding_rate) > 0.01:  # ±0.01% threshold
                    # Reduce position size for high funding
                    adjustment = 1.0 - (abs(funding_rate) / 0.02)  # Linear reduction
                    adjusted_weights[symbol] *= max(adjustment, 0.5)
            
            # Renormalize weights
            total = sum(adjusted_weights.values())
            if total > 0:
                for symbol in adjusted_weights:
                    adjusted_weights[symbol] /= total
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error adjusting for funding: {e}")
            return weights
    
    async def calculate_execution_price(
        self,
        symbol: str,
        size: float,
        side: str,
        order_book: Dict
    ) -> float:
        """Calculate expected execution price including spread and slippage."""
        try:
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                return 0.0
            
            # Get relevant side of the book
            levels = order_book['asks'] if side == 'buy' else order_book['bids']
            
            remaining_size = size
            weighted_price = 0.0
            total_filled = 0.0
            
            for level in levels:
                price = float(level[0])
                available = float(level[1])
                
                fill_size = min(remaining_size, available)
                weighted_price += price * fill_size
                total_filled += fill_size
                
                remaining_size -= fill_size
                if remaining_size <= 0:
                    break
            
            if total_filled > 0:
                return weighted_price / total_filled
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating execution price: {e}")
            return 0.0
    
    async def rebalance_portfolio(self, current_positions: Dict[str, float]) -> Dict[str, Dict]:
        """Calculate required position changes to match target weights."""
        try:
            changes = {}
            
            # Calculate total portfolio value (assuming USDT positions)
            total_value = sum(current_positions.values())
            
            if total_value == 0:
                return {}
            
            # Calculate current weights
            self.current_weights = {
                symbol: pos / total_value 
                for symbol, pos in current_positions.items()
            }
            
            # Calculate required changes
            for symbol in self.target_weights:
                current = self.current_weights.get(symbol, 0.0)
                target = self.target_weights[symbol]
                
                if abs(target - current) > 0.01:  # 1% threshold
                    size_change = (target - current) * total_value
                    
                    # Check if change is allowed by risk manager
                    if await self.risk_manager.check_position_change(symbol, size_change):
                        changes[symbol] = {
                            'size': abs(size_change),
                            'side': 'buy' if size_change > 0 else 'sell'
                        }
            
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating rebalance: {e}")
            return {} 