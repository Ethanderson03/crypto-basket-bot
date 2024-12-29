import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger

from data.ccxt_price_feed import CCXTPriceFeed
from data.sentiment_analyzer import SentimentAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from trading.position_manager import PositionManager
from risk.risk_manager import RiskManager
from analysis.order_book_analyzer import OrderBookAnalyzer
from execution.order_executor import OrderExecutor

class BacktestRunner:
    """Runs backtests for the sentiment trading strategy."""
    
    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        use_cache: bool = True
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        
        # Initialize components
        self.price_feed = CCXTPriceFeed(
            mode='backtest',
            backtest_start=start_date,
            backtest_end=end_date,
            use_cached_data=use_cache
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_analyzer = MarketAnalyzer(self.price_feed, self.sentiment_analyzer)
        self.risk_manager = RiskManager(self.price_feed)
        self.position_manager = PositionManager(self.market_analyzer, self.risk_manager)
        self.order_book_analyzer = OrderBookAnalyzer()
        self.order_executor = OrderExecutor(self.price_feed, self.order_book_analyzer)
        
        # Backtest state
        self.portfolio_value = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.metrics: Dict = {}
    
    async def initialize_backtest(self):
        """Initialize backtest data and state."""
        logger.info("Initializing backtest...")
        
        # Update risk manager with initial state
        self.risk_manager.update_portfolio_state(
            self.positions,
            self.portfolio_value
        )
        
        logger.info(f"Starting backtest with {len(self.symbols)} symbols:")
        for symbol in self.symbols:
            logger.info(f"  - {symbol}")
    
    async def execute_trades(self, timestamp: datetime, changes: Dict[str, Dict]):
        """Execute trades based on position changes."""
        for symbol, change in changes.items():
            size = change['size']
            side = change['side']
            
            # Get order book
            order_book = await self.price_feed.get_order_book(symbol)
            
            # Analyze execution conditions
            conditions = await self.order_executor.analyze_execution_conditions(
                symbol,
                side,
                size,
                order_book
            )
            
            if conditions['recommendation']['should_execute']:
                # Calculate execution price
                execution = await self.order_executor.estimate_total_cost(
                    symbol,
                    side,
                    size,
                    order_book
                )
                
                # Update positions and portfolio
                price = execution['base_asset']['average_price']
                trade_value = execution['quote_asset']['total_cost']
                
                if side == 'buy':
                    if trade_value <= self.portfolio_value:
                        self.portfolio_value -= trade_value
                        self.positions[symbol] = {
                            'size': self.positions.get(symbol, {}).get('size', 0) + size,
                            'collateral': trade_value
                        }
                else:  # sell
                    self.portfolio_value += trade_value
                    current_size = self.positions.get(symbol, {}).get('size', 0)
                    if current_size >= size:
                        new_size = current_size - size
                        if new_size > 0:
                            self.positions[symbol]['size'] = new_size
                            # Reduce collateral proportionally
                            self.positions[symbol]['collateral'] *= new_size / current_size
                        else:
                            del self.positions[symbol]
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'price': price,
                    'value': trade_value,
                    'portfolio_value': self.get_total_portfolio_value()
                })
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value including positions."""
        total = self.portfolio_value
        for symbol, position in self.positions.items():
            total += position['collateral']
        return total
    
    def calculate_metrics(self):
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        df = pd.DataFrame(self.trades)
        
        # Calculate returns
        df['return'] = df['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_return = (self.get_total_portfolio_value() - self.initial_balance) / self.initial_balance
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = df['return'].dropna()
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(365) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown
        df['cummax'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['cummax'] - df['portfolio_value']) / df['cummax']
        max_drawdown = df['drawdown'].max()
        
        # Calculate win rate
        profitable_trades = len(df[df['value'] > 0])
        
        self.metrics = {
            'total_trades': len(self.trades),
            'win_rate': profitable_trades / len(self.trades),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return self.metrics
    
    async def run(self):
        """Run the backtest."""
        await self.initialize_backtest()
        
        current_time = self.start_date
        
        while current_time <= self.end_date:
            try:
                # Update market state
                market_state = await self.market_analyzer.get_market_state()
                
                # Calculate position weights
                weights = await self.position_manager.calculate_base_weights(self.symbols)
                
                # Adjust for funding rates
                adjusted_weights = await self.position_manager.adjust_for_funding(weights)
                
                # Calculate required position changes
                changes = await self.position_manager.rebalance_portfolio(
                    {symbol: pos['size'] for symbol, pos in self.positions.items()}
                )
                
                # Execute trades
                if changes:
                    await self.execute_trades(current_time, changes)
                
                # Update risk manager
                self.risk_manager.update_portfolio_state(
                    self.positions,
                    self.portfolio_value
                )
                
                # Move to next time step (1 hour intervals)
                current_time += timedelta(hours=1)
                
            except Exception as e:
                logger.error(f"Error in backtest iteration: {e}")
                continue
        
        # Calculate final metrics
        metrics = self.calculate_metrics()
        
        logger.info("Backtest completed!")
        logger.info("Final metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.2%}" if isinstance(value, float) else f"  {key}: {value}")
        
        return {
            'metrics': metrics,
            'trades': self.trades,
            'final_positions': self.positions,
            'final_portfolio_value': self.get_total_portfolio_value()
        }

async def main():
    # Define backtest parameters
    symbols = [
        'BTC/USDT',
        'ETH/USDT',
        'BNB/USDT',
        'SOL/USDT',
        'DOGE/USDT',
        'SHIB/USDT',
        'PEPE/USDT'
    ]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    initial_balance = 10000.0
    
    # Run backtest
    backtest = BacktestRunner(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        use_cache=True
    )
    
    results = await backtest.run()
    
    # Save results to CSV
    trades_df = pd.DataFrame(results['trades'])
    trades_df.to_csv('backtest_trades.csv', index=False)
    
    # Print summary
    print("\nBacktest Summary:")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    print(f"Total Return: {results['metrics']['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Total Trades: {results['metrics']['total_trades']}")
    print(f"Win Rate: {results['metrics']['win_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(main()) 