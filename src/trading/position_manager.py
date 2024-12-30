from datetime import datetime
from loguru import logger
import numpy as np
from typing import Tuple

class PositionManager:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.portfolio_value = initial_balance
        self.positions = {}  # {symbol: {'size': float, 'entry_price': float, 'side': str, 'timestamp': datetime}}
        self.trades = []
        self.max_position_size = 0.2  # 20% of portfolio per position
        self.min_position_size = 0.02  # 2% of portfolio per position
        self.max_leverage = 3.0
        self.min_leverage = 0.5
        
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value including all positions."""
        total = self.portfolio_value
        for symbol, position in self.positions.items():
            total += position['collateral']
        return total
        
    def calculate_position_size(self, symbol: str, signal: float, current_price: float, total_value: float) -> Tuple[float, str]:
        """Calculate the position size based on the signal strength and available capital."""
        # Cap the leverage based on signal strength
        max_leverage = min(abs(signal), self.max_leverage)
        leverage = max(1.0, max_leverage)
        
        # Calculate base position size (% of portfolio)
        position_size_usd = total_value * self.max_position_size * leverage
        
        # Convert to units of the asset
        size = position_size_usd / current_price if current_price > 0 else 0
        
        # Determine trade direction
        side = 'long' if signal > 0 else 'short'
        
        logger.info(f"Calculated position size for {symbol}: {size:.4f} units ({side}) at {current_price}")
        logger.info(f"Signal strength: {signal}, Leverage: {leverage}")
        
        return size, side
        
    async def update_position(self, symbol: str, size: float, side: str, price: float, timestamp: datetime):
        """Update position for a given symbol."""
        # Calculate position value
        position_value = abs(size * price)
        
        # Check if we have enough portfolio value
        if position_value > self.portfolio_value:
            logger.warning(f"Insufficient portfolio value for {symbol} trade")
            return
        
        # Update or create position
        if symbol in self.positions:
            # Update existing position
            current_pos = self.positions[symbol]
            new_size = current_pos['size'] + size
            
            if new_size == 0:
                # Position closed
                self.positions.pop(symbol)
                logger.info(f"Closed position for {symbol}")
            else:
                # Update position
                avg_price = (current_pos['entry_price'] * current_pos['size'] + price * size) / new_size
                self.positions[symbol] = {
                    'size': new_size,
                    'entry_price': avg_price,
                    'side': 'long' if new_size > 0 else 'short',
                    'timestamp': timestamp,
                    'collateral': -position_value  # Negative because it's taken from portfolio value
                }
        else:
            # Create new position
            self.positions[symbol] = {
                'size': size,
                'entry_price': price,
                'side': side,
                'timestamp': timestamp,
                'collateral': -position_value  # Negative because it's taken from portfolio value
            }
        
        # Update portfolio value
        self.portfolio_value -= position_value  # Subtract the position value from portfolio
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': price,
            'value': position_value,
            'portfolio_value': self.get_total_portfolio_value()
        }
        self.trades.append(trade)
        
        logger.info(f"Updated position for {symbol}: {self.positions[symbol]}")
        logger.info(f"New portfolio value: ${self.get_total_portfolio_value():,.2f}")
        
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get the current value of a position."""
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol]['size'] * current_price
        
    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for a position."""
        if symbol not in self.positions:
            return 0.0
        position = self.positions[symbol]
        return (current_price - position['entry_price']) * position['size'] 