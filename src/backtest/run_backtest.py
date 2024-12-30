import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger
import subprocess

from src.data.ccxt_price_feed import CCXTPriceFeed
from src.analysis.market_analyzer import MarketAnalyzer
from src.trading.position_manager import PositionManager
from src.risk.risk_manager import RiskManager
from src.analysis.order_book_analyzer import OrderBookAnalyzer

class BacktestRunner:
    def __init__(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 10000.0,
        timeframe: str = '1h'
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.timeframe = timeframe
        
        # Initialize components
        self.price_feed = CCXTPriceFeed(mode='backtest')
        self.market_analyzer = MarketAnalyzer()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(max_position_size=0.2, max_leverage=3.0)
        
        # Initialize state
        self.current_time = start_date
        self.portfolio_value = initial_balance
        self.trades = []
        self.positions = {}
        
    async def run(self):
        """Run the backtest simulation."""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        logger.info("Initializing price feed with historical data...")
        await self.price_feed.initialize(self.symbols, self.start_date, self.end_date, self.timeframe)
        logger.info("Price feed initialized successfully")

        metrics_history = []
        
        while self.current_time <= self.end_date:
            logger.debug("\n" + "=" * 50)
            logger.debug(f"Processing timestamp: {self.current_time}")
            logger.debug("=" * 50)

            # Update market state
            await self.market_analyzer.update(self.price_feed, self.current_time)

            # Log market state
            logger.debug("\nMarket State:")
            logger.debug(f"Direction: {self.market_analyzer.market_state['direction']}")
            logger.debug(f"Sentiment: {self.market_analyzer.market_state['sentiment']}")
            logger.debug(f"Fear & Greed: {self.market_analyzer.market_state['fear_greed']}")
            logger.debug(f"Is Oversold: {self.market_analyzer.market_state['is_oversold']}")
            logger.debug(f"Is Overbought: {self.market_analyzer.market_state['is_overbought']}")

            # Log current prices
            logger.debug("\nCurrent Prices:")
            for symbol in self.symbols:
                price = await self.price_feed.get_current_price(symbol, self.current_time)
                logger.debug(f"{symbol}: ${price:,.2f}")

            # Get trading signals
            signals = await self.market_analyzer.get_trading_signals(self.price_feed, self.current_time)

            # Execute trades based on signals
            if signals:
                await self.execute_trades(signals, self.current_time)

            # Calculate and record metrics
            metrics = await self.calculate_metrics()
            metrics_history.append(metrics)

            # Move to next timestamp
            self.current_time += timedelta(hours=1)

        return metrics_history
    
    async def execute_deleveraging(
        self,
        deleveraging: Dict[str, float],
        timestamp: datetime
    ) -> None:
        """Execute deleveraging trades."""
        try:
            for symbol, size in deleveraging.items():
                position = self.position_manager.positions.get(symbol)
                if position and size > 0:
                    await self.execute_trade(
                        timestamp,
                        symbol,
                        'close',
                        size
                    )
                    
        except Exception as e:
            logger.error(f"Error executing deleveraging: {str(e)}")
    
    async def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        size: float
    ) -> None:
        """Execute a trade and record it."""
        try:
            # Get order book
            order_book = await self.price_feed.get_order_book(symbol)
            
            # Execute order
            execution = await self.order_executor.execute_order(
                symbol,
                side,
                size
            )
            
            if execution['success']:
                # Record trade
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'price': execution['executed_price'],
                    'value': size * execution['executed_price'],
                    'slippage_bps': execution['cost_metrics']['slippage_bps'],
                    'portfolio_value': self.position_manager.portfolio_value
                }
                self.trades.append(trade)
                
                # Update position
                self.position_manager.update_position(
                    symbol,
                    size,
                    side,
                    execution['executed_price']
                )
                
                # Update risk manager
                price_history = [
                    candle['close']
                    for candle in await self.price_feed.get_historical_candles(symbol, limit=100)
                ]
                self.risk_manager.update_position(
                    symbol,
                    size,
                    side,
                    execution['executed_price'],
                    price_history
                )
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
    
    def update_portfolio_metrics(self, timestamp: datetime) -> None:
        """Update portfolio metrics."""
        try:
            self.portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': self.position_manager.portfolio_value,
                'free_collateral': self.position_manager.free_collateral,
                'health_ratio': self.risk_manager.calculate_portfolio_health()
            })
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {str(e)}")
    
    async def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}
        
        # Calculate total portfolio value
        total_value = await self.get_total_portfolio_value()
        
        # Calculate total return
        total_return = (total_value - self.initial_balance) / self.initial_balance
        metrics['total_return'] = total_return
        
        # Calculate other metrics
        metrics['portfolio_value'] = total_value
        metrics['timestamp'] = self.current_time
        metrics['num_trades'] = len(self.trades)
        
        # Log metrics
        logger.debug("\nMetrics:")
        logger.debug(f"Total Return: {total_return:.2%}")
        logger.debug(f"Portfolio Value: ${total_value:,.2f}")
        logger.debug(f"Number of Trades: {len(self.trades)}")
        
        return metrics
    
    async def execute_trades(self, signals: Dict[str, float], current_time: datetime):
        """Execute trades based on signals."""
        logger.info("\nExecuting trades...")
        
        for symbol, signal in signals.items():
            logger.debug(f"\nProcessing signal for {symbol}: {signal}")
            
            # Get current price
            price = await self.price_feed.get_current_price(symbol)
            logger.debug(f"Current price for {symbol}: ${price:.2f}")
            
            # Calculate position size
            total_value = self.get_total_portfolio_value()
            size, side = self.position_manager.calculate_position_size(symbol, signal, price, total_value)
            
            logger.debug(f"Calculated size: {size:.4f}")
            logger.debug(f"Side: {side}")
            
            # Execute trade
            if size > 0:
                logger.info(f"\033[92m💰 EXECUTING TRADE: {symbol} {side.upper()} {size:.4f} @ ${price:.2f} (Value: ${abs(size * price):.2f})\033[0m")
                await self.position_manager.update_position(symbol, size, side, price, current_time)
                self.trades.append({
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'price': price,
                    'timestamp': current_time
                })

    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value including positions."""
        total = self.initial_balance
        
        for symbol, position in self.positions.items():
            current_price = self.price_feed.get_current_price_sync(symbol)
            size = position['size']
            side = position['side']
            entry_price = position['entry_price']
            
            # Calculate position P&L
            if side == 'long':
                pnl = size * (current_price - entry_price)
            else:  # short
                pnl = size * (entry_price - current_price)
                
            total += pnl
            
            logger.debug(f"\nPosition value for {symbol}:")
            logger.debug(f"Size: {size:.4f}")
            logger.debug(f"Current Price: ${current_price:.2f}")
            logger.debug(f"P&L: ${pnl:.2f}")
        
        logger.debug(f"\nTotal Portfolio Value: ${total:.2f}")
        return total

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
    start_date = datetime(2023, 12, 1)
    end_date = datetime(2024, 1, 1)
    initial_balance = 10000.0
    
    logger.info(f"Starting backtest simulation from {start_date} to {end_date}")
    logger.info(f"Trading symbols: {symbols}")
    logger.info(f"Initial balance: ${initial_balance:,.2f}")
    
    try:
        # Run backtest
        backtest = BacktestRunner(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            timeframe='1h'
        )
        
        metrics_history = await backtest.run()
        
        # Calculate final metrics
        final_metrics = await backtest.calculate_metrics()
        
        # Save trades to CSV for dashboard
        trades_df = pd.DataFrame(backtest.trades)
        trades_df.to_csv('backtest_trades.csv', index=False)
        
        # Print summary
        print("\nBacktest Summary:")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Portfolio Value: ${backtest.position_manager.portfolio_value:,.2f}")
        print(f"Total Return: {final_metrics['total_return']:,.2%}")
        print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {final_metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {final_metrics['total_trades']}")
        print(f"Win Rate: {final_metrics['win_rate']:.2%}")
        
        # Launch dashboard
        subprocess.Popen(['python', 'src/dashboard/run_dashboard.py'])
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Ensure all resources are cleaned up
        await asyncio.sleep(0.1)  # Give time for connections to close

if __name__ == "__main__":
    asyncio.run(main()) 