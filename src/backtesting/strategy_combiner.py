"""
Strategy Combiner - Object-oriented approach for combining multiple strategies.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from src.backtesting.legs import LegResult, backtest_leg
from src.strategies.strategies import STRATEGIES
from src.backtesting.portfolio_combo import CombinedPortfolio, combine_portfolios, calculate_rebalance_dates
from src.backtesting.data_loader import load_data
from src.backtesting.signal_based_allocation import calculate_signal_based_weights


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    id: str
    symbol: str
    strategy_name: str
    params: Dict
    weight: float  # Allocation weight (0.0 to 1.0)


class StrategyCombiner:
    """
    Main class for combining and backtesting multiple strategies.
    
    This class follows the approach:
    1. Generate signals for each strategy
    2. Simulate each strategy separately (100% allocation per strategy)
    3. Combine portfolios with specified weights
    4. Analyze combined performance
    """
    
    def __init__(
        self,
        strategies: List[StrategyConfig],
        initial_capital: float = 100_000.0,
        fees: float = 0.0005,
        slippage: float = 0.0005,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        align_to_symbol: Optional[str] = None,
        rebalance_freq: Optional[str] = None,
        safe_asset: Optional[str] = None,
        rsi_ranking: Optional[Any] = None  # Deprecated - ignored, kept for backward compatibility
    ):
        """
        Initialize StrategyCombiner.
        
        Args:
            strategies: List of StrategyConfig objects
            initial_capital: Initial capital for backtesting
            fees: Trading fees (as fraction, e.g. 0.0005 = 0.05%)
            slippage: Slippage (as fraction)
            start_date: Start date for data loading (optional, can be set later)
            end_date: End date for data loading (optional)
            align_to_symbol: Optional symbol to align all data to (useful for aligning
                            24/7 crypto markets to traditional market trading days, e.g., 'TQQQ')
            rebalance_freq: Optional rebalancing frequency. Options:
                - None: No rebalancing (weights applied to returns directly)
                - 'D': Daily rebalancing
                - 'W': Weekly rebalancing
                - 'M': Monthly rebalancing
                - 'Q': Quarterly rebalancing
                - 'Y' or 'A': Yearly rebalancing
            safe_asset: Optional ticker symbol for safe asset (e.g., 'TLT', 'GLD', 'SHY').
                       When strategies have sell signals, this asset will be held instead of cash.
            rsi_ranking: DEPRECATED - This parameter is ignored. Signal-based allocation is now used
                        automatically when rebalance_freq is set. Only strategies with buy signals
                        receive allocation, with a maximum of 1/3 per strategy.
        """
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.start_date = start_date
        self.end_date = end_date
        self.align_to_symbol = align_to_symbol
        self.rebalance_freq = rebalance_freq
        self.safe_asset = safe_asset
        # rsi_ranking is deprecated but kept for backward compatibility
        self.rsi_ranking = None  # Always None - signal-based allocation is used instead
        
        # Will be populated during execution
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.legs: Optional[Dict[str, LegResult]] = None
        self.combined_portfolio: Optional[CombinedPortfolio] = None
        self.safe_asset_data: Optional[pd.DataFrame] = None
        self.rsi_weights_df: Optional[pd.DataFrame] = None  # Dynamic weights (signal-based)
        self.rsi_values_df: Optional[pd.DataFrame] = None  # Not used (kept for compatibility)
        
        # Validate strategy names
        self._validate_strategies()
    
    def _validate_strategies(self):
        """Validate that all strategy names exist in STRATEGIES registry."""
        for strat in self.strategies:
            if strat.strategy_name not in STRATEGIES:
                raise ValueError(
                    f"Strategy '{strat.strategy_name}' not found. "
                    f"Available strategies: {list(STRATEGIES.keys())}"
                )
        
        # Validate weights sum to approximately 1.0
        total_weight = sum(s.weight for s in self.strategies)
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Strategy weights sum to {total_weight:.4f}, expected 1.0")
    
    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, align_to_symbol: Optional[str] = None):
        """
        Load stock data for all symbols used by strategies.
        
        Args:
            start_date: Start date (overrides instance start_date if provided)
            end_date: End date (overrides instance end_date if provided)
            align_to_symbol: Symbol to align data to (overrides instance align_to_symbol if provided)
        """
        symbols = [s.symbol for s in self.strategies if s.weight > 0]
        symbols = sorted(set(symbols))
        
        # Include safe asset in symbols list if provided
        if self.safe_asset and self.safe_asset not in symbols:
            symbols.append(self.safe_asset)
        
        start = start_date or self.start_date
        if start is None:
            raise ValueError("start_date must be provided either in __init__ or load_data()")
        
        align_symbol = align_to_symbol or self.align_to_symbol
        all_data = load_data(symbols, start, end_date or self.end_date, align_to_symbol=align_symbol)
        
        # Separate safe asset data from strategy data
        if self.safe_asset and self.safe_asset in all_data:
            self.safe_asset_data = all_data[self.safe_asset].copy()
            self.data = {k: v for k, v in all_data.items() if k != self.safe_asset}
            print(f"✅ Loaded safe asset data: {self.safe_asset}")
        else:
            self.safe_asset_data = None
            self.data = all_data
        
        self.start_date = start
        if end_date:
            self.end_date = end_date
        if align_to_symbol:
            self.align_to_symbol = align_to_symbol
        
        print(f"✅ Loaded data for {len(self.data)} symbols")
    
    def generate_signals_and_backtest(self):
        """
        Generate signals for each strategy and backtest individually.
        Each strategy is backtested with 100% of the portfolio.
        """
        if self.data is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        self.legs = {}
        config = {
            'initial_capital': self.initial_capital,
            'fee_bps': self.fees * 10000,  # Convert to basis points
            'slip_bps': self.slippage * 10000,
            'execution_price': 'close',
            'risk_free_rate': 0.0
        }
        
        for strat_config in self.strategies:
            if strat_config.weight == 0:
                continue  # Skip strategies with zero weight
            
            leg_id = strat_config.id
            symbol = strat_config.symbol
            strategy_name = strat_config.strategy_name
            params = strat_config.params
            
            # Get strategy function
            if strategy_name not in STRATEGIES:
                print(f"Warning: Strategy '{strategy_name}' not found, skipping {leg_id}")
                continue
            
            strategy_func = STRATEGIES[strategy_name]
            
            # Get data for this symbol
            if symbol not in self.data:
                print(f"Warning: Data for {symbol} not found, skipping {leg_id}")
                continue
            
            price_data = self.data[symbol].copy()
            
            # Generate strategy signals/exposure
            try:
                strategy_output = strategy_func(price_data, params)
            except Exception as e:
                print(f"Error generating signals for {leg_id}: {e}")
                continue
            
            # Backtest this leg with 100% allocation
            try:
                # Get safe asset data if configured (align it to this leg's index)
                safe_asset_prices = None
                if self.safe_asset_data is not None:
                    # Align safe asset data to this strategy's price data index
                    safe_asset_prices = self.safe_asset_data.reindex(price_data.index)
                    # Forward fill missing values (e.g., weekends/holidays)
                    safe_asset_prices = safe_asset_prices.ffill()
                    # Backfill any remaining NaN at the start
                    safe_asset_prices = safe_asset_prices.bfill()
                
                leg_result = backtest_leg(
                    symbol=symbol,
                    prices=price_data,
                    strategy_output=strategy_output,
                    config=config,
                    leg_id=leg_id,
                    strategy_name=strategy_name,
                    params=params,
                    safe_asset_prices=safe_asset_prices,
                    safe_asset_ticker=self.safe_asset
                )
                self.legs[leg_id] = leg_result
                safe_asset_msg = f" (with safe asset: {self.safe_asset})" if self.safe_asset else ""
                print(f"✅ Backtested {leg_id} ({symbol}, {strategy_name}){safe_asset_msg}")
            except Exception as e:
                print(f"Error backtesting {leg_id}: {e}")
                continue
    
    def combine_portfolios(self):
        """
        Combine individual strategy portfolios with specified weights.
        
        This creates a CombinedPortfolio object that combines the returns
        from each strategy according to their weights.
        If rebalancing is enabled (rebalance_freq is not None), signal-based
        allocation is used: only strategies with buy signals receive allocation,
        with a maximum of 1/3 per strategy.
        """
        if self.legs is None or len(self.legs) == 0:
            raise ValueError("No legs to combine. Call generate_signals_and_backtest() first")
        
        # Create weights dictionary (used as fallback if rebalancing is not enabled)
        weights = {s.id: s.weight for s in self.strategies if s.id in self.legs}
        
        # Check if weights are all equal (indicating they weren't meaningfully set)
        # If so, default to equal weights
        if len(weights) > 0:
            weight_values = list(weights.values())
            if len(set(weight_values)) == 1:
                # All weights are the same - use equal weights instead
                equal_weight = 1.0 / len(weights)
                weights = {k: equal_weight for k in weights.keys()}
            else:
                # Normalize weights (in case they don't sum to 1.0)
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate dynamic weights using signal-based allocation on a DAILY basis
        # Every trading day:
        #   - If at least one leg has a buy signal, split allocation equally across those legs
        #   - If no legs have a buy signal, keep equal allocation across all legs
        dynamic_weights_df = None
        rebalance_freq_for_combo = self.rebalance_freq

        # Get common index across all legs (trading days)
        indices = [leg.returns.index for leg in self.legs.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
        
        if len(common_index) > 0:
            try:
                # Use all trading days for signal-based allocation
                dynamic_weights_df = calculate_signal_based_weights(
                    legs=self.legs,
                    dates=common_index
                )
                self.rsi_weights_df = dynamic_weights_df
                # Rebalance daily based on these weights
                rebalance_freq_for_combo = 'D'
                print(f"✅ Calculated daily signal-based dynamic weights for {len(common_index)} trading days")
                print("   - If at least one strategy has a buy signal, capital is split equally across those legs.")
                print("   - If no strategies have buy signals, capital is split equally across all legs.")
            except Exception as e:
                print(f"Warning: Failed to calculate signal-based weights: {e}")
                print("Falling back to static weights")
                dynamic_weights_df = None
                rebalance_freq_for_combo = self.rebalance_freq
        
        # Combine portfolios
        self.combined_portfolio = combine_portfolios(
            legs=self.legs,
            weights=weights,
            initial_capital=self.initial_capital,
            rebalance_freq=rebalance_freq_for_combo,
            dynamic_weights_df=dynamic_weights_df
        )
        
        if dynamic_weights_df is not None:
            print(f"✅ Combined {len(self.legs)} strategies into portfolio (with signal-based dynamic weights)")
        else:
            print(f"✅ Combined {len(self.legs)} strategies into portfolio")
    
    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Run complete backtest pipeline:
        1. Load data
        2. Generate signals and backtest each strategy
        3. Combine portfolios
        
        Args:
            start_date: Start date (optional, uses instance start_date if not provided)
            end_date: End date (optional)
        """
        if start_date or not hasattr(self, 'data') or self.data is None:
            self.load_data(start_date, end_date)
        
        self.generate_signals_and_backtest()
        self.combine_portfolios()
        
        print("\n✅ Strategy combination backtest complete!")
    
    def get_combined_returns(self) -> pd.Series:
        """Get combined portfolio returns."""
        if self.combined_portfolio is None:
            raise ValueError("Portfolio not combined yet. Call combine_portfolios() or run()")
        return self.combined_portfolio.returns
    
    def get_combined_equity(self) -> pd.Series:
        """Get combined portfolio equity curve."""
        if self.combined_portfolio is None:
            raise ValueError("Portfolio not combined yet. Call combine_portfolios() or run()")
        return self.combined_portfolio.equity
    
    def get_leg_results(self) -> Dict[str, LegResult]:
        """Get individual leg results."""
        if self.legs is None:
            raise ValueError("Legs not backtested yet. Call generate_signals_and_backtest() or run()")
        return self.legs

