from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from loguru import logger
import asyncio
from datetime import datetime, timedelta

from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.order_book_analyzer import OrderBookAnalyzer
from src.trading.position_manager import PositionManager
from src.risk.risk_manager import RiskManager
from src.execution.order_executor import OrderExecutor

class Strategy:
    """
    Main strategy class that orchestrates the crypto perpetual futures trading system.
    """
    
    def __init__(
        self,
        trading_pairs: List[str],
        base_collateral: float,
        update_interval: float = 60.0,  # 1 minute update interval
        min_trade_interval: float = 300.0,  # 5 minutes between trades
        max_leverage: float = 2.0,
        risk_free_rate: float = 0.03,  # 3% risk-free rate for Sharpe calculation
        performance_window: int = 30    # 30 days for performance metrics
    ):
        """
        Initialize the trading strategy.
        
        Args:
            trading_pairs: List of trading pairs to trade
            base_collateral: Initial collateral in quote currency
            update_interval: Interval between market updates in seconds
            min_trade_interval: Minimum time between trades in seconds
            max_leverage: Maximum allowed leverage
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            performance_window: Days to look back for performance metrics
        """
        self.trading_pairs = trading_pairs
        self.base_collateral = base_collateral
        self.update_interval = update_interval
        self.min_trade_interval = min_trade_interval
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        self.performance_window = performance_window
        
        # Initialize components
        self.market_analyzer = MarketAnalyzer()
        self.order_book_analyzer = OrderBookAnalyzer()
        self.position_manager = PositionManager(base_leverage=max_leverage)
        self.risk_manager = RiskManager()
        self.order_executor = OrderExecutor()
        
        # Internal state
        self.positions: Dict[str, Dict] = {}
        self.last_trade_time: Dict[str, datetime] = {}
        self.performance_metrics: Dict[str, float] = {}
        self.is_running: bool = False
        self.new_trades: List[Dict] = []  # New trades for backtesting
        self.price_feed = None  # Will be set externally
        
        # Performance tracking
        self.equity_history: List[float] = []
        self.trade_history: List[Dict] = []
        
        logger.info(f"Strategy initialized with {len(trading_pairs)} trading pairs")
    
    async def start(self):
        """Start the trading strategy."""
        self.is_running = True
        
        try:
            # Initialize market data and connections
            await self._initialize()
            
            # Main trading loop
            while self.is_running:
                await self._update_cycle()
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the trading strategy."""
        self.is_running = False
        
        # Close all positions
        await self._close_all_positions()
    
    async def _initialize(self):
        """Initialize market data and connections."""
        try:
            # Initialize market data feeds
            for pair in self.trading_pairs:
                # Initialize price feeds
                await self._initialize_price_feed(pair)
                
                # Initialize order book feeds
                await self._initialize_order_book_feed(pair)
                
                # Initialize funding rate monitoring
                await self._initialize_funding_rate_feed(pair)
            
            # Initialize risk monitoring
            self.risk_manager.collateral = self.base_collateral
            
            logger.info("Strategy initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    async def _update_cycle(self):
        """Run a single update cycle."""
        try:
            # Clear new trades list at the start of each cycle
            self.new_trades = []
            
            # Update market data
            await self._update_market_data()
            
            # Check portfolio health
            health_ratio = self.risk_manager.calculate_portfolio_health()
            if health_ratio < self.risk_manager.min_health_ratio:
                await self._handle_low_health(health_ratio)
                return
            
            # Analyze market conditions
            market_direction = self.market_analyzer.analyze_market_direction()
            sentiment_multiplier = self.market_analyzer.get_sentiment_multiplier()
            onchain_signal = self.market_analyzer.analyze_onchain_metrics()
            
            # Calculate position adjustments
            target_weights = self.position_manager.calculate_base_weights()
            adjusted_weights = self.position_manager.adjust_for_funding(target_weights)
            
            # Execute position changes
            for pair in self.trading_pairs:
                await self._update_position(
                    pair,
                    adjusted_weights.get(pair, 0),
                    market_direction,
                    sentiment_multiplier
                )
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Update cycle error: {e}")
    
    async def _update_position(
        self,
        pair: str,
        target_weight: float,
        market_direction: float,
        sentiment_multiplier: float
    ):
        """
        Update position for a single trading pair.
        
        Args:
            pair: Trading pair symbol
            target_weight: Target position weight
            market_direction: Market direction score (-1 to 1)
            sentiment_multiplier: Position size multiplier based on sentiment
        """
        try:
            current_position = self.positions.get(pair, {'size': 0})
            current_size = current_position.get('size', 0)
            
            # Skip if too soon since last trade
            last_trade = self.last_trade_time.get(pair, datetime.min)
            if (datetime.now() - last_trade).total_seconds() < self.min_trade_interval:
                return
            
            # Calculate target position size
            position_size = (
                target_weight *
                self.base_collateral *
                sentiment_multiplier *
                market_direction
            )
            
            # Check if adjustment is needed
            size_diff = position_size - current_size
            if abs(size_diff) < self.base_collateral * 0.01:  # 1% minimum change
                return
            
            # Get order book
            order_book = await self._fetch_order_book(pair)
            
            # Check execution conditions
            side = 'buy' if size_diff > 0 else 'sell'
            conditions = self.order_executor.analyze_execution_conditions(
                pair,
                side,
                abs(size_diff),
                order_book
            )
            
            if not conditions['recommendation']['should_execute']:
                logger.warning(f"Poor execution conditions for {pair}")
                return
            
            # Execute the trade
            if conditions['recommendation']['split_orders']:
                await self._execute_split_orders(
                    pair,
                    side,
                    abs(size_diff),
                    conditions['recommendation']['recommended_splits']
                )
            else:
                await self._execute_order(pair, side, abs(size_diff))
            
            self.last_trade_time[pair] = datetime.now()
            
        except Exception as e:
            logger.error(f"Position update error for {pair}: {e}")
    
    async def _handle_low_health(self, health_ratio: float):
        """Handle low portfolio health situation."""
        try:
            # Get required reduction from thresholds
            reduction = 0.0
            for threshold, required_reduction in sorted(
                self.risk_manager.health_ratio_thresholds.items(),
                reverse=True
            ):
                if health_ratio <= threshold:
                    reduction = max(reduction, required_reduction)
            
            if reduction > 0:
                # Reduce positions proportionally
                for pair, position in self.positions.items():
                    reduction_size = position['size'] * reduction
                    if reduction_size != 0:
                        side = 'sell' if position['size'] > 0 else 'buy'
                        await self._execute_order(pair, side, abs(reduction_size))
                
                logger.warning(f"Reduced positions by {reduction*100}% due to low health ratio")
            
        except Exception as e:
            logger.error(f"Error handling low health ratio: {e}")
    
    async def _execute_split_orders(
        self,
        pair: str,
        side: str,
        size: float,
        splits: List[Dict]
    ):
        """Execute a series of split orders."""
        for split in splits:
            try:
                await self._execute_order(pair, side, split['size'])
                await asyncio.sleep(split['delay_seconds'])
            except Exception as e:
                logger.error(f"Split order execution error: {e}")
                break
    
    async def _execute_order(self, pair: str, side: str, size: float):
        """Execute a single order."""
        # This would integrate with your exchange API
        logger.info(f"Executing {side} order for {size} {pair}")
        pass
    
    async def _close_all_positions(self):
        """Close all open positions."""
        for pair, position in self.positions.items():
            if position['size'] != 0:
                side = 'sell' if position['size'] > 0 else 'buy'
                await self._execute_order(pair, side, abs(position['size']))
    
    def _update_performance_metrics(self):
        """Update strategy performance metrics."""
        try:
            # Calculate daily returns
            daily_returns = []  # This would be calculated from historical PnL
            
            if not daily_returns:
                return
            
            # Calculate metrics
            returns = np.array(daily_returns)
            avg_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio
            excess_returns = avg_return - self.risk_free_rate / 252
            sharpe = excess_returns / volatility if volatility != 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            self.performance_metrics.update({
                'sharpe_ratio': float(sharpe),
                'volatility': float(volatility),
                'max_drawdown': float(max_drawdown),
                'avg_daily_return': float(avg_return)
            })
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _initialize_price_feed(self, pair: str):
        """Initialize price feed for a trading pair."""
        pass
    
    async def _initialize_order_book_feed(self, pair: str):
        """Initialize order book feed for a trading pair."""
        pass
    
    async def _initialize_funding_rate_feed(self, pair: str):
        """Initialize funding rate feed for a trading pair."""
        pass
    
    async def _update_market_data(self):
        """Update all market data."""
        pass
    
    async def _fetch_order_book(self, pair: str) -> Dict:
        """Fetch current order book for a trading pair."""
        return {'bids': [], 'asks': []} 