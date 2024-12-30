import dash
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
from dash.dependencies import Input, Output
import threading
import webbrowser

class Dashboard:
    """Dashboard for visualizing trading strategy performance using Dash."""
    
    def __init__(self, port=8050):
        """Initialize the dashboard."""
        self.port = port
        self.app = dash.Dash(__name__)
        self.latest_metrics = {}
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1('Crypto Trading Dashboard', style={'textAlign': 'center'}),
            
            # Metrics Cards
            html.Div([
                html.Div([
                    html.H4('Total Trades'),
                    html.H2(id='total-trades', children='0')
                ], className='metric-card'),
                html.Div([
                    html.H4('Win Rate'),
                    html.H2(id='win-rate', children='0%')
                ], className='metric-card'),
                html.Div([
                    html.H4('Average Profit'),
                    html.H2(id='avg-profit', children='0%')
                ], className='metric-card'),
                html.Div([
                    html.H4('Portfolio Value'),
                    html.H2(id='portfolio-value', children='$0')
                ], className='metric-card'),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
            
            # Charts
            html.Div([
                dcc.Graph(id='portfolio-chart'),
                dcc.Graph(id='trades-chart'),
            ]),
            
            # Interval component for updates
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # in milliseconds
                n_intervals=0
            )
        ])
        
        # Add custom CSS
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>Trading Dashboard</title>
                {%favicon%}
                {%css%}
                <style>
                    .metric-card {
                        background-color: #f8f9fa;
                        border-radius: 10px;
                        padding: 20px;
                        text-align: center;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        flex: 1;
                        margin: 0 10px;
                    }
                    .metric-card h4 {
                        color: #666;
                        margin: 0;
                    }
                    .metric-card h2 {
                        color: #333;
                        margin: 10px 0 0 0;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def setup_callbacks(self):
        """Set up the dashboard callbacks."""
        @self.app.callback(
            [Output('total-trades', 'children'),
             Output('win-rate', 'children'),
             Output('avg-profit', 'children'),
             Output('portfolio-value', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(_):
            if not self.latest_metrics:
                return '0', '0%', '0%', '$0'
            
            return (
                str(self.latest_metrics.get('total_trades', 0)),
                f"{self.latest_metrics.get('win_rate', 0):.1%}",
                f"{self.latest_metrics.get('avg_profit', 0):.1%}",
                f"${self.latest_metrics.get('portfolio_value', 0):,.2f}"
            )
    
    def update_metrics(self, metrics: Dict):
        """Update dashboard metrics."""
        self.latest_metrics = metrics
    
    def run(self):
        """Run the dashboard server."""
        # Open browser in a separate thread
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{self.port}')).start()
        # Run the server
        self.app.run_server(debug=False, port=self.port) 