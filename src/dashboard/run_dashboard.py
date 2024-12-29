import asyncio
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from data.ccxt_price_feed import CCXTPriceFeed
from data.sentiment_analyzer import SentimentAnalyzer
from strategy.sentiment_strategy import SentimentStrategy
from dashboard.dashboard import Dashboard

def initialize_strategy():
    """Initialize the trading strategy components."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
    price_feed = CCXTPriceFeed(use_cached_data=True)
    strategy = SentimentStrategy(price_feed=price_feed)
    return strategy, symbols

async def run_strategy_iteration(
    strategy: SentimentStrategy,
    symbols: List[str],
    timestamp: datetime
) -> Dict:
    """Run one iteration of the strategy and return state for dashboard."""
    await strategy.update(timestamp, symbols)
    
    # Get current prices for all symbols
    current_prices = {
        symbol: await strategy.price_feed.get_current_price(symbol)
        for symbol in symbols
    }
    
    # Get sentiment data for visualization
    sentiment_data = {}
    for symbol in symbols:
        sentiment = await strategy.sentiment_analyzer.get_combined_sentiment(symbol, timestamp)
        
        if symbol not in sentiment_data:
            sentiment_data[symbol] = {
                'prices': [],
                'sentiments': [],
                'timestamps': []
            }
        
        sentiment_data[symbol]['prices'].append(current_prices[symbol])
        sentiment_data[symbol]['sentiments'].append(sentiment)
        sentiment_data[symbol]['timestamps'].append(timestamp)
    
    # Return complete state for dashboard
    return {
        'metrics': strategy.get_metrics(),
        'trades': strategy.get_trade_history(),
        'positions': strategy.get_current_positions(),
        'sentiment_data': sentiment_data
    }

async def run_backtest(strategy: SentimentStrategy, symbols: List[str], dashboard: Dashboard):
    """Run the backtest and update the dashboard."""
    # Simulation parameters
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    current_time = start_date
    
    progress_bar = st.progress(0)
    total_steps = (end_date - start_date).total_seconds() / 3600  # 1 hour intervals
    current_step = 0
    
    while current_time <= end_date:
        # Run strategy iteration
        state = await run_strategy_iteration(strategy, symbols, current_time)
        
        # Update dashboard
        dashboard.render(state)
        
        # Update progress
        current_step += 1
        progress = min(current_step / total_steps, 1.0)
        progress_bar.progress(progress)
        
        # Move to next time step (1 hour intervals)
        current_time += timedelta(hours=1)
        
        # Add small delay to prevent overwhelming the system
        await asyncio.sleep(0.1)

def main():
    """Main entry point for the dashboard application."""
    st.set_page_config(
        page_title="Crypto Market Sentiment Backtest",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Crypto Market Sentiment Backtest")
    st.write("This dashboard shows the results of running a sentiment-based trading strategy.")
    
    # Initialize components
    strategy, symbols = initialize_strategy()
    dashboard = Dashboard()
    
    # Add start button
    if st.button("Start Backtest"):
        asyncio.run(run_backtest(strategy, symbols, dashboard))

if __name__ == "__main__":
    main() 