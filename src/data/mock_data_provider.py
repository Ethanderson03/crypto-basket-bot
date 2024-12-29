import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

class MockDataProvider:
    """
    Provides mock data for backtesting when real API data is not available.
    """
    
    def __init__(self, start_date: datetime, end_date: datetime):
        """
        Initialize the mock data provider.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
        """
        self.start_date = start_date
        self.end_date = end_date
        
    def generate_price_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic price data for testing.
        
        Args:
            symbols: List of trading pairs
            
        Returns:
            Dict mapping symbols to DataFrames with OHLCV data
        """
        data = {}
        days = (self.end_date - self.start_date).days
        
        for symbol in symbols:
            # Generate daily timestamps
            timestamps = [self.start_date + timedelta(days=i) for i in range(days)]
            
            # Generate random walk prices
            base = 100.0  # Base price
            volatility = 0.02  # Daily volatility
            returns = np.random.normal(0, volatility, days)
            prices = base * np.exp(np.cumsum(returns))
            
            # Generate OHLCV data
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices,
                'high': prices * (1 + np.random.uniform(0, 0.01, days)),
                'low': prices * (1 - np.random.uniform(0, 0.01, days)),
                'close': prices,
                'volume': np.random.uniform(1000, 10000, days)
            })
            
            df.set_index('timestamp', inplace=True)
            data[symbol] = df
            
        return data
    
    def generate_fear_greed_index(self) -> float:
        """Generate a mock Fear & Greed index value."""
        return np.random.uniform(20, 80)  # Random value between 20-80
    
    def generate_order_book(self, symbol: str, mid_price: float) -> Dict:
        """
        Generate a mock order book.
        
        Args:
            symbol: Trading pair
            mid_price: Current mid price
            
        Returns:
            Dict with bids and asks
        """
        spread = mid_price * 0.001  # 0.1% spread
        
        bids = []
        asks = []
        
        # Generate 10 levels on each side
        for i in range(10):
            bid_price = mid_price - spread - (i * spread)
            ask_price = mid_price + spread + (i * spread)
            
            bid_size = np.random.uniform(0.1, 1.0)
            ask_size = np.random.uniform(0.1, 1.0)
            
            bids.append([bid_price, bid_size])
            asks.append([ask_price, ask_size])
            
        return {
            'bids': bids,
            'asks': asks
        } 