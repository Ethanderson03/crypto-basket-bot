import pandas as pd
from dashboard import Dashboard
import webbrowser
import threading

def main():
    # Load backtest results
    try:
        trades_df = pd.read_csv('backtest_trades.csv')
        
        # Calculate metrics
        total_trades = len(trades_df)
        win_rate = len(trades_df[trades_df['value'] > 0]) / total_trades if total_trades > 0 else 0
        avg_profit = trades_df['value'].mean() if total_trades > 0 else 0
        portfolio_value = trades_df['portfolio_value'].iloc[-1] if total_trades > 0 else 10000.0
        
        # Initialize and run dashboard
        dashboard = Dashboard()
        dashboard.update_metrics({
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'portfolio_value': portfolio_value
        })
        
        # Open browser automatically
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:8050')).start()
        
        # Run the dashboard
        dashboard.run()
        
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main() 