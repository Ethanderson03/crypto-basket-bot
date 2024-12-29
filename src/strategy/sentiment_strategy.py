from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from data import SentimentAnalyzer, PriceFeed

class SentimentStrategy:
    """Trading strategy based on market sentiment analysis."""
    
    def __init__(self, price_feed: PriceFeed, initial_balance: float = 10000.0):
        self.price_feed = price_feed
        self.sentiment_analyzer = SentimentAnalyzer()
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trades: List[Dict] = []
        
        # Strategy parameters
        self.sentiment_threshold = 0.6  # Bullish above this
        self.position_size = 0.2  # Use 20% of balance per position
        self.stop_loss = 0.05  # 5% stop loss
        self.take_profit = 0.1  # 10% take profit
    
    async def update(self, timestamp: datetime, symbols: List[str]):
        """Update strategy state and execute trades based on sentiment."""
        # Get current prices for all tracked symbols
        current_prices = {
            symbol: await self.price_feed.get_current_price(symbol)
            for symbol in symbols
        }
        
        # Track initial portfolio value
        initial_portfolio_value = self.get_portfolio_value(current_prices)
        
        for symbol in symbols:
            # Get sentiment score
            sentiment = await self.sentiment_analyzer.get_combined_sentiment(symbol, timestamp)
            current_price = current_prices[symbol]
            
            # First check if we need to exit any existing positions
            if symbol in self.positions:
                quantity = self.positions[symbol]
                entry_price = next(
                    (t['price'] for t in reversed(self.trades) 
                     if t['symbol'] == symbol and t['side'] == 'buy'),
                    None
                )
                
                if entry_price:
                    profit_pct = (current_price - entry_price) / entry_price
                    
                    # Exit if sentiment turns bearish or we hit take profit/stop loss
                    if (sentiment < self.sentiment_threshold or 
                        profit_pct <= -self.stop_loss or 
                        profit_pct >= self.take_profit):
                        
                        # Calculate exit value
                        exit_value = quantity * current_price
                        self.balance += exit_value
                        
                        # Record trade with portfolio value
                        self.trades.append({
                            'timestamp': timestamp,
                            'symbol': symbol,
                            'side': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'value': exit_value,
                            'portfolio_value': self.get_portfolio_value(current_prices)
                        })
                        
                        # Remove position
                        del self.positions[symbol]
                        continue  # Skip entry check for this symbol
            
            # Then check if we should enter a new position
            if symbol not in self.positions and sentiment > self.sentiment_threshold:
                # Calculate position size with validation
                max_position_value = self.balance * self.position_size
                if max_position_value <= 0:
                    logger.warning(f"Insufficient balance for trade: {self.balance}")
                    continue
                    
                # Calculate quantity and validate
                quantity = max_position_value / current_price
                if quantity <= 0:
                    logger.warning(f"Invalid quantity calculated: {quantity} for {symbol}")
                    continue
                    
                # Recalculate final position value based on actual quantity
                position_value = quantity * current_price
                if position_value > self.balance:
                    logger.warning(f"Trade value {position_value} exceeds available balance {self.balance}")
                    continue
                
                # Enter long position
                self.positions[symbol] = quantity
                self.balance -= position_value
                
                # Record trade
                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': quantity,
                    'price': current_price,
                    'value': position_value,
                    'portfolio_value': self.get_portfolio_value(current_prices)
                })
        
        # Log portfolio value change
        final_portfolio_value = self.get_portfolio_value(current_prices)
        value_change = final_portfolio_value - initial_portfolio_value
        logger.info(f"Portfolio value changed by {value_change:.2f} to {final_portfolio_value:.2f}")
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions."""
        position_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in current_prices.items()
        )
        return self.balance + position_value
    
    def get_metrics(self) -> Dict:
        """Calculate strategy performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0
            }
        
        # Track open positions and their average entry prices
        open_positions: Dict[str, Dict] = {}  # symbol -> {quantity: float, avg_price: float}
        completed_trades = []
        
        for trade in self.trades:
            symbol = trade['symbol']
            if trade['side'] == 'buy':
                # Update or create position with new average entry price
                if symbol not in open_positions:
                    open_positions[symbol] = {
                        'quantity': trade['quantity'],
                        'avg_price': trade['price']
                    }
                else:
                    # Calculate new average entry price
                    current = open_positions[symbol]
                    total_quantity = current['quantity'] + trade['quantity']
                    current['avg_price'] = (
                        (current['quantity'] * current['avg_price'] + 
                         trade['quantity'] * trade['price']) / total_quantity
                    )
                    current['quantity'] = total_quantity
            
            elif trade['side'] == 'sell' and symbol in open_positions:
                position = open_positions[symbol]
                # Calculate profit for this exit
                entry_value = trade['quantity'] * position['avg_price']
                exit_value = trade['value']
                profit = (exit_value - entry_value) / entry_value
                
                completed_trades.append({
                    'symbol': symbol,
                    'profit': profit,
                    'quantity': trade['quantity']
                })
                
                # Update remaining position
                position['quantity'] -= trade['quantity']
                if position['quantity'] <= 0:
                    del open_positions[symbol]
                
        total_trades = len(completed_trades)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0,
                'open_positions': len(open_positions)
            }
        
        # Weight profits by position size
        total_quantity = sum(t['quantity'] for t in completed_trades)
        weighted_profit = sum(
            t['profit'] * (t['quantity'] / total_quantity)
            for t in completed_trades
        )
        
        winning_trades = len([t for t in completed_trades if t['profit'] > 0])
        win_rate = winning_trades / total_trades
        
        # Calculate max drawdown using portfolio values
        portfolio_values = []
        current_value = 10000.0  # Initial balance
        for trade in self.trades:
            if trade['side'] == 'buy':
                current_value -= trade['value']
            else:
                current_value += trade['value']
            portfolio_values.append(current_value)
        
        max_drawdown = 0.0
        peak = portfolio_values[0]
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': weighted_profit,
            'max_drawdown': max_drawdown,
            'open_positions': len(open_positions)
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Get the list of all trades."""
        return self.trades
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions 