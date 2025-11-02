"""Backtesting framework."""

from src.backtesting.strategy_combiner import StrategyCombiner, StrategyConfig
from src.backtesting.data_loader import load_data, select_close_series
from src.backtesting.legs import LegResult, backtest_leg
from src.backtesting.portfolio_combo import CombinedPortfolio, combine_portfolios
from src.backtesting.performance_analyzer import PerformanceAnalyzer

__all__ = [
    'StrategyCombiner',
    'StrategyConfig',
    'load_data',
    'select_close_series',
    'LegResult',
    'backtest_leg',
    'CombinedPortfolio',
    'combine_portfolios',
    'PerformanceAnalyzer',
]

