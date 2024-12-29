from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
from ..data.sentiment_analyzer import SentimentAnalyzer
from ..data.price_feed import PriceFeed

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
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions."""
        position_value = sum(
            self.positions.get(symbol, 0) * price 
            for symbol, price in current_prices.items()
        )
        return self.balance + position_value
    
    async def update(self, timestamp: datetime, symbols: List[str]) -> None:
        """Update strategy state and execute trades based on sentiment."""
        try:
            # Get current prices for all symbols
            current_prices = {
                symbol: await self.price_feed.get_current_price(symbol)
                for symbol in symbols
            }
            
            for symbol in symbols:
                # Get sentiment score
                sentiment = await self.sentiment_analyzer.get_combined_sentiment(symbol, timestamp)
                current_price = current_prices[symbol]
                position = self.positions.get(symbol, 0)
                
                # Check exit conditions first
                if position > 0:
                    # Calculate profit/loss
                    entry_price = next(
                        (t['price'] for t in reversed(self.trades) 
                         if t['symbol'] == symbol and t['side'] == 'buy'),
                        current_price
                    )
                    pnl = (current_price - entry_price) / entry_price
                    
                    # Exit on stop loss or take profit
                    if pnl <= -self.stop_loss or pnl >= self.take_profit:
                        await self._execute_trade(symbol, 'sell', position, current_price, timestamp)
                        continue
                
                # Entry logic based on sentiment
                if sentiment > self.sentiment_threshold and position == 0:
                    # Calculate position size
                    amount_to_invest = self.balance * self.position_size
                    quantity = amount_to_invest / current_price
                    
                    if quantity > 0:
                        await self._execute_trade(symbol, 'buy', quantity, current_price, timestamp)
                
                # Exit on bearish sentiment
                elif sentiment < (1 - self.sentiment_threshold) and position > 0:
                    await self._execute_trade(symbol, 'sell', position, current_price, timestamp)
        
        except Exception as e:
            logger.error(f"Error updating strategy: {e}")
    
    async def _execute_trade(self, symbol: str, side: str, quantity: float, price: float, timestamp: datetime) -> None:
        """Execute a trade and update positions and balance."""
        try:
            trade_value = quantity * price
            
            if side == 'buy':
                if trade_value <= self.balance:
                    self.balance -= trade_value
                    self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                else:
                    logger.warning(f"Insufficient balance for trade: {trade_value} > {self.balance}")
                    return
            else:  # sell
                self.balance += trade_value
                self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            
            # Record trade
            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'value': trade_value
            })
            
            logger.info(f"Executed {side} trade: {quantity} {symbol} @ {price}")
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def get_trade_history(self) -> List[Dict]:
        """Get list of all executed trades."""
        return self.trades
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_metrics(self) -> Dict:
        """Calculate and return strategy performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0
            }
        
        total_trades = len(self.trades)
        profitable_trades = sum(1 for t in self.trades if t['side'] == 'sell' and t['value'] > 0)
        
        metrics = {
            'total_trades': total_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0.0,
            'avg_profit': (self.balance - 10000) / 10000 if total_trades > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history."""
        if not self.trades:
            return 0.0
        
        peak = 10000  # Initial balance
        max_drawdown = 0.0
        
        for trade in self.trades:
            if trade['side'] == 'sell':
                current_value = trade['value']
                drawdown = (peak - current_value) / peak
                max_drawdown = max(max_drawdown, drawdown)
                peak = max(peak, current_value)
        
        return max_drawdown 