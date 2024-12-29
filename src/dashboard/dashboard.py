import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List

class Dashboard:
    """Dashboard for visualizing trading strategy performance."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'start_time' not in st.session_state:
            st.session_state.start_time = datetime.now()
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
    
    def update_metrics(self, metrics: Dict):
        """Update dashboard metrics."""
        st.session_state.last_update = datetime.now()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Trades",
                f"{metrics['total_trades']}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Average Profit",
                f"{metrics['avg_profit']:.1%}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.1%}",
                delta=None
            )
    
    def plot_portfolio_performance(self, trades: List[Dict]):
        """Plot portfolio value over time."""
        if not trades:
            st.warning("No trades available to plot")
            return
        
        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate cumulative portfolio value
        df['cumulative_value'] = df['value'].cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            )
        )
        
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Time',
            yaxis_title='Portfolio Value (USD)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_sentiment_vs_price(self, symbol: str, prices: List[float], sentiments: List[float], timestamps: List[datetime]):
        """Plot price and sentiment over time for a symbol."""
        if not prices or not sentiments:
            st.warning(f"No data available for {symbol}")
            return
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=prices,
                name="Price",
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=sentiments,
                name="Sentiment",
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=f'{symbol} Price vs Sentiment',
            xaxis_title='Time',
            height=400
        )
        
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_recent_trades(self, trades: List[Dict]):
        """Display recent trades in a table."""
        if not trades:
            st.warning("No trades to display")
            return
        
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        st.subheader("Recent Trades")
        st.dataframe(
            df[['timestamp', 'symbol', 'side', 'quantity', 'price', 'value']],
            use_container_width=True
        )
    
    def show_current_positions(self, positions: Dict[str, float]):
        """Display current positions."""
        if not positions:
            st.warning("No open positions")
            return
        
        st.subheader("Current Positions")
        df = pd.DataFrame([
            {"symbol": symbol, "quantity": quantity}
            for symbol, quantity in positions.items()
            if quantity > 0
        ])
        
        if not df.empty:
            st.dataframe(df, use_container_width=True)
    
    def render(self, strategy_state: Dict):
        """Render the complete dashboard."""
        st.title("Crypto Sentiment Trading Dashboard")
        
        # Update metrics
        self.update_metrics(strategy_state['metrics'])
        
        # Portfolio performance chart
        self.plot_portfolio_performance(strategy_state['trades'])
        
        # Create two columns for positions and trades
        col1, col2 = st.columns(2)
        
        with col1:
            self.show_current_positions(strategy_state['positions'])
        
        with col2:
            self.show_recent_trades(strategy_state['trades'][-10:])  # Show last 10 trades
        
        # Sentiment analysis for each symbol
        if 'sentiment_data' in strategy_state:
            for symbol, data in strategy_state['sentiment_data'].items():
                self.plot_sentiment_vs_price(
                    symbol,
                    data['prices'],
                    data['sentiments'],
                    data['timestamps']
                )
        
        # Add update time
        st.sidebar.markdown(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        st.sidebar.markdown(f"Running since: {st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')}") 