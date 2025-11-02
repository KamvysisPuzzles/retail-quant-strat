# Quantitative Trading Strategy

A comprehensive quantitative trading strategy framework with backtesting, portfolio combination, and automated signal alerting.

## Structure

```
quant_strat/
├── notebooks/              # Jupyter notebooks for analysis and backtesting
├── src/                    # Core trading strategy code
│   ├── strategies/         # Strategy implementations
│   └── backtesting/        # Backtesting framework
├── alerts/                 # Alerting system (Telegram notifications)
├── outputs/                # Generated outputs (plots, analysis)
└── .github/workflows/      # GitHub Actions for automated alerts
```

## Features

- **Multiple Strategy Support**: Implement and combine various trading strategies (3EMA, MACD-V, RSI, Aroon, etc.)
- **Portfolio Combination**: Combine multiple strategies with custom weights and rebalancing
- **Performance Analysis**: Comprehensive metrics including Sharpe, Sortino, drawdown, and trade analysis
- **Automated Alerts**: Telegram notifications for strategy signals via GitHub Actions
- **Safe Asset Support**: Hold safe assets (e.g., GLD) instead of cash when strategies signal exits

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd quant_strat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

```python
from src.backtesting.strategy_combiner import StrategyCombiner, StrategyConfig

# Define strategies
strategies = [
    StrategyConfig(
        id="tqqq_strategy",
        symbol="TQQQ",
        strategy_name="tqqq_3ema_macdv_aroon",
        params={"ema1": 12, "ema2": 89, "ema3": 125, ...},
        weight=0.5,
    ),
    # Add more strategies...
]

# Run backtest
combiner = StrategyCombiner(
    strategies=strategies,
    initial_capital=100_000.0,
    fees=0.0005,
    start_date="2018-01-01",
    safe_asset='GLD'
)
combiner.run()

# Analyze results
from src.backtesting.performance_analyzer import PerformanceAnalyzer
analyzer = PerformanceAnalyzer(combiner.combined_portfolio)
analyzer.print_metrics_table()
```


- `sma_cross` - Simple Moving Average crossover
- `tqqq_3ema_macdv_aroon` - 3EMA + MACD-V + Aroon ensemble
- `3ema_macdv` - 3EMA + MACD-V ensemble
- `3ema_macd_rsi` - 3EMA + MACD-V + RSI ensemble
- `macd_rsi` - MACD-V + RSI ensemble
- `macdv` - Pure MACD-V strategy
- `rsi` - Pure RSI strategy
- `rsi_aroon` - RSI + Aroon ensemble
- `aroon` - Pure Aroon strategy
- `btc_3ema_macdv_aroon` - BTC-optimized version
