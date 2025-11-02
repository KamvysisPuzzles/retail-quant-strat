"""Quantitative strategy backtesting library."""

from lib.strategies import STRATEGIES, register_strategy
from lib.strategy_combiner import StrategyCombiner, StrategyConfig
from lib.performance_analyzer import PerformanceAnalyzer
from lib.portfolio_combo import CombinedPortfolio, combine_portfolios
from lib.legs import LegResult, backtest_leg
from lib.data_loader import load_data, select_close_series

__all__ = [
    'STRATEGIES',
    'register_strategy',
    'StrategyCombiner',
    'StrategyConfig',
    'PerformanceAnalyzer',
    'CombinedPortfolio',
    'combine_portfolios',
    'LegResult',
    'backtest_leg',
    'load_data',
    'select_close_series',
]



