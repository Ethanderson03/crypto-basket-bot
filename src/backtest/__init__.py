"""Backtest package for running trading strategy simulations."""

from .backtest import run_backtest, parse_args
from .run_backtest import BacktestRunner

__all__ = ['run_backtest', 'parse_args', 'BacktestRunner'] 