"""Dashboard package for visualizing trading strategy results."""

from .backtest_dashboard import create_backtest_dashboard
from .dashboard import Dashboard

__all__ = ['Dashboard', 'create_backtest_dashboard'] 