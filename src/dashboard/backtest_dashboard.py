import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List

def create_backtest_dashboard(trades: List[Dict], metrics: Dict, initial_balance: float = 10000.0):
    """Create a summary dashboard for backtest results."""
    st.set_page_config(page_title="Backtest Results", layout="wide")
    st.title("Crypto Sentiment Trading Strategy - Backtest Results")
    
    # Convert trades to DataFrame
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate cumulative portfolio value
    df['cumulative_value'] = df['value'].cumsum() + initial_balance
    
    # Key Metrics Section
    st.header("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{((df['cumulative_value'].iloc[-1] - initial_balance) / initial_balance):.1%}",
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
            "Total Trades",
            f"{metrics['total_trades']}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.1%}",
            delta=None
        )
    
    # Portfolio Performance Chart
    st.header("Portfolio Performance")
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
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (USD)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade Analysis
    st.header("Trade Analysis")
    
    # Group trades by symbol
    symbol_metrics = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        profitable_trades = len(symbol_df[symbol_df['value'] > 0])
        total_trades = len(symbol_df)
        
        symbol_metrics[symbol] = {
            'total_trades': total_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': symbol_df['value'].sum()
        }
    
    # Create metrics table
    metrics_df = pd.DataFrame.from_dict(symbol_metrics, orient='index')
    metrics_df.columns = ['Total Trades', 'Win Rate', 'Total PnL']
    metrics_df['Win Rate'] = metrics_df['Win Rate'].map('{:.1%}'.format)
    metrics_df['Total PnL'] = metrics_df['Total PnL'].map('${:,.2f}'.format)
    
    st.dataframe(metrics_df)
    
    # Trade Distribution
    st.header("Trade Distribution")
    
    # Create trade size distribution chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=df['value'].abs(),
            name='Trade Size Distribution',
            nbinsx=30
        )
    )
    
    fig.update_layout(
        title='Trade Size Distribution',
        xaxis_title='Trade Size (USD)',
        yaxis_title='Frequency',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.warning("This script should be imported and used with backtest results.") 