import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

class RiskManager:
    def __init__(self, max_position_size: float = 0.1, max_leverage: float = 3.0):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.positions = {}
        self.portfolio_value = 0.0
        self.current_time = None
        
    def update(self, positions: Dict[str, Dict], portfolio_value: float, current_time: datetime) -> Dict:
        """Update risk metrics based on current positions and portfolio value."""
        self.positions = positions
        self.portfolio_value = portfolio_value
        self.current_time = current_time
        
        metrics = {
            'total_exposure': self.calculate_total_exposure(),
            'max_drawdown': self.calculate_max_drawdown(),
            'position_concentration': self.calculate_position_concentration(),
            'leverage_ratio': self.calculate_leverage_ratio()
        }
        
        return metrics
        
    def check_trade(self, symbol: str, size: float, side: str, price: float) -> bool:
        """Check if a trade meets risk management criteria."""
        try:
            # Calculate trade value
            trade_value = abs(size * price)
            
            # Check position size limit
            if trade_value / self.portfolio_value > self.max_position_size:
                logger.warning(f"Trade rejected: Position size {trade_value / self.portfolio_value:.2%} exceeds limit {self.max_position_size:.2%}")
                return False
            
            # Calculate total exposure including new trade
            total_exposure = self.calculate_total_exposure()
            new_exposure = total_exposure + trade_value
            
            # Check leverage limit
            if new_exposure / self.portfolio_value > self.max_leverage:
                logger.warning(f"Trade rejected: Leverage {new_exposure / self.portfolio_value:.2f}x exceeds limit {self.max_leverage:.2f}x")
                return False
            
            # Check if we have an existing position
            if symbol in self.positions:
                existing_position = self.positions[symbol]
                existing_side = existing_position['side']
                
                # Don't allow increasing position in opposite direction
                if existing_side != side and size > 0:
                    logger.warning(f"Trade rejected: Cannot increase position in opposite direction for {symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade: {str(e)}")
            return False
            
    def calculate_total_exposure(self) -> float:
        """Calculate total exposure across all positions."""
        total = 0.0
        for symbol, position in self.positions.items():
            total += abs(position['collateral'])
        return total
        
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        # In a real implementation, this would track historical equity curve
        return 0.0
        
    def calculate_position_concentration(self) -> float:
        """Calculate position concentration ratio."""
        if not self.positions:
            return 0.0
            
        exposures = []
        for position in self.positions.values():
            exposures.append(abs(position['collateral']))
            
        if not exposures:
            return 0.0
            
        max_exposure = max(exposures)
        total_exposure = sum(exposures)
        
        return max_exposure / total_exposure if total_exposure > 0 else 0.0
        
    def calculate_leverage_ratio(self) -> float:
        """Calculate current leverage ratio."""
        total_exposure = self.calculate_total_exposure()
        return total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0.0 