from loguru import logger

class RiskManager:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_position_size = 0.2  # 20% of portfolio per position
        self.max_leverage = 3.0
        self.max_drawdown = 0.25  # 25% max drawdown
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.positions = {}
        self.trades = []
        
    def update(self, positions: dict, portfolio_value: float, current_time):
        """Update risk metrics based on current positions and portfolio value."""
        self.positions = positions
        self.current_balance = portfolio_value
        
        # Calculate current drawdown
        drawdown = (self.initial_balance - portfolio_value) / self.initial_balance
        
        # Update position sizes based on current portfolio value
        for symbol, position in positions.items():
            position_value = abs(position['size'] * position['entry_price'])
            position_size_pct = position_value / portfolio_value
            
            # Check if position size exceeds limits
            if position_size_pct > self.max_position_size:
                logger.warning(f"Position size for {symbol} ({position_size_pct:.2%}) exceeds maximum allowed ({self.max_position_size:.2%})")
            
            # Check leverage
            if position.get('leverage', 1.0) > self.max_leverage:
                logger.warning(f"Leverage for {symbol} ({position['leverage']:.2f}x) exceeds maximum allowed ({self.max_leverage:.2f}x)")
        
        # Check overall portfolio risk
        if drawdown > self.max_drawdown:
            logger.warning(f"Current drawdown ({drawdown:.2%}) exceeds maximum allowed ({self.max_drawdown:.2%})")
            
        return {
            'drawdown': drawdown,
            'portfolio_value': portfolio_value,
            'position_sizes': {symbol: abs(pos['size'] * pos['entry_price']) / portfolio_value for symbol, pos in positions.items()},
            'leverages': {symbol: pos.get('leverage', 1.0) for symbol, pos in positions.items()}
        }
        
    def check_trade(self, symbol: str, size: float, side: str, price: float, portfolio_value: float) -> bool:
        """Check if a trade meets risk management criteria."""
        # Calculate position value
        position_value = abs(size * price)
        position_size_pct = position_value / portfolio_value
        
        # Check position size limit
        if position_size_pct > self.max_position_size:
            logger.warning(f"Rejecting trade: Position size ({position_size_pct:.2%}) exceeds maximum allowed ({self.max_position_size:.2%})")
            return False
            
        # Check if adding this position would exceed portfolio risk limits
        total_exposure = sum(abs(pos['size'] * pos['entry_price']) for pos in self.positions.values())
        total_exposure_pct = (total_exposure + position_value) / portfolio_value
        
        if total_exposure_pct > 0.8:  # 80% max total exposure
            logger.warning(f"Rejecting trade: Total exposure ({total_exposure_pct:.2%}) would exceed maximum allowed (80%)")
            return False
            
        return True 