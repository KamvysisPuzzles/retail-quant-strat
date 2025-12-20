"""Portfolio combination utilities for combining multiple strategy portfolios."""
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Optional
from src.backtesting.legs import LegResult


def calculate_rebalance_dates(common_index: pd.DatetimeIndex, rebalance_freq: str) -> pd.DatetimeIndex:
    """
    Calculate rebalancing dates consistently across the codebase.
    
    This function ensures that rebalance dates calculated for RSI ranking
    match exactly with rebalance dates used in portfolio combination.
    
    Args:
        common_index: Common date index from all legs
        rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y', 'A')
    
    Returns:
        DatetimeIndex of rebalancing dates
    """
    if rebalance_freq == 'D':
        return common_index  # Every day
    else:
        # Use pandas date offset to find rebalancing dates
        freq_map = {
            'W': 'W',
            'M': 'MS',  # Month start
            'Q': 'QS',  # Quarter start
            'Y': 'YS',  # Year start
            'A': 'YS',  # Year start (annual)
        }
        
        freq = freq_map.get(rebalance_freq.upper(), rebalance_freq)
        try:
            rebalance_dates = pd.date_range(
                start=common_index[0],
                end=common_index[-1],
                freq=freq
            )
            # Filter to only include dates in common_index
            rebalance_dates = common_index.intersection(rebalance_dates)
            # Always include first date
            if len(rebalance_dates) == 0 or rebalance_dates[0] != common_index[0]:
                rebalance_dates = rebalance_dates.union([common_index[0]])
                rebalance_dates = rebalance_dates.sort_values()
            return rebalance_dates
        except Exception as e:
            # Fallback to first and last date
            return pd.DatetimeIndex([common_index[0], common_index[-1]])


class CombinedPortfolio:
    """
    Represents a combined portfolio from multiple strategy legs.
    """
    def __init__(
        self,
        legs: Dict[str, LegResult],
        weights: Dict[str, float],
        initial_capital: float,
        rebalance_freq: Optional[str] = None,
        dynamic_weights_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize combined portfolio.
        
        Args:
            legs: Dictionary mapping leg_id to LegResult
            weights: Dictionary mapping leg_id to weight (must sum to 1.0)
                     Used as fallback if dynamic_weights_df is not provided
            initial_capital: Initial capital for portfolio
            rebalance_freq: Optional rebalancing frequency. Options:
                - None: No rebalancing (weights applied to returns directly)
                - 'D': Daily rebalancing
                - 'W': Weekly rebalancing
                - 'M': Monthly rebalancing
                - 'Q': Quarterly rebalancing
                - 'Y' or 'A': Yearly rebalancing
                - 'WOM-1MON' or similar: Custom pandas offset aliases
            dynamic_weights_df: Optional DataFrame with dynamic weights over time.
                                Index should be dates, columns should be leg_ids.
                                If provided, weights will be looked up from this DataFrame
                                at each rebalancing date instead of using static weights.
        """
        self.legs = legs
        self.weights = weights
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.dynamic_weights_df = dynamic_weights_df
        
        # Validate static weights sum to 1.0 (used as fallback)
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Combine returns and equity curves
        self._combine_results()
    
    def _combine_results(self):
        """Combine individual leg results into portfolio results."""
        if not self.legs:
            raise ValueError("No legs provided")
        
        # Get common index from returns (returns have the same index as equity curves)
        indices = [leg.returns.index for leg in self.legs.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
        
        if len(common_index) == 0:
            raise ValueError("No overlapping time periods across legs")
        
        # If rebalancing is enabled, use dollar-based allocation tracking
        if self.rebalance_freq is not None:
            self._combine_with_rebalancing(common_index)
        else:
            # Standard approach: weight returns directly (no rebalancing)
            self._combine_without_rebalancing(common_index)
    
    def _combine_without_rebalancing(self, common_index):
        """Combine returns without rebalancing (weighted returns approach)."""
        combined_returns = pd.Series(0.0, index=common_index)
        
        for leg_id, leg in self.legs.items():
            weight = self.weights.get(leg_id, 0.0)
            if weight > 0:
                # Align returns to common index - avoid reindex if indices match exactly
                if leg.returns.index.equals(common_index):
                    leg_returns = leg.returns
                else:
                    # Only reindex if indices don't match exactly
                    leg_returns = leg.returns.reindex(common_index).fillna(0.0)
                # Weight and add returns
                combined_returns += weight * leg_returns
        
        self.returns = combined_returns
        
        # Reconstruct equity curve from combined returns
        # Start with initial capital and compound returns
        self.equity = (1.0 + combined_returns).cumprod() * self.initial_capital
    
    def _combine_with_rebalancing(self, common_index):
        """Combine with periodic rebalancing to target weights."""
        # Align all leg returns to common index
        # Include ALL legs, not just those with static weight > 0, because
        # dynamic weights may allocate to different legs over time
        leg_returns = {}
        for leg_id, leg in self.legs.items():
            if leg.returns.index.equals(common_index):
                leg_returns[leg_id] = leg.returns
            else:
                leg_returns[leg_id] = leg.returns.reindex(common_index).fillna(0.0)
        
        # Initialize tracking arrays
        portfolio_value = pd.Series(0.0, index=common_index)
        allocations = pd.DataFrame(0.0, index=common_index, columns=list(leg_returns.keys()))
        actual_weights = pd.DataFrame(0.0, index=common_index, columns=list(leg_returns.keys()))
        
        # Determine rebalancing dates using shared function for consistency
        try:
            rebalance_dates = calculate_rebalance_dates(common_index, self.rebalance_freq)
        except Exception as e:
            print(f"Warning: Could not parse rebalance frequency '{self.rebalance_freq}', using no rebalancing. Error: {e}")
            self._combine_without_rebalancing(common_index)
            return
        
        # Track portfolio through time with rebalancing
        # Initialize with static weights, but will be overridden by dynamic weights if available
        initial_weights = {}
        if self.dynamic_weights_df is not None and len(self.dynamic_weights_df) > 0:
            # Use first dynamic weight if available
            first_date = self.dynamic_weights_df.index[0]
            for leg_id in leg_returns.keys():
                if leg_id in self.dynamic_weights_df.columns:
                    initial_weights[leg_id] = self.dynamic_weights_df.loc[first_date, leg_id]
                else:
                    initial_weights[leg_id] = self.weights.get(leg_id, 0.0)
        else:
            # Use static weights
            for leg_id in leg_returns.keys():
                initial_weights[leg_id] = self.weights.get(leg_id, 0.0)
        
        current_allocation = {leg_id: initial_weights.get(leg_id, 0.0) * self.initial_capital 
                             for leg_id in leg_returns.keys()}
        portfolio_value.iloc[0] = self.initial_capital
        
        for i, date in enumerate(common_index):
            if i == 0:
                # Initial allocation
                for leg_id in leg_returns.keys():
                    allocations.loc[date, leg_id] = current_allocation[leg_id]
                    actual_weights.loc[date, leg_id] = initial_weights.get(leg_id, 0.0)
                continue
            
            prev_date = common_index[i-1]
            
            # Calculate new portfolio value based on previous allocations and returns
            # This includes ALL legs, even those with 0 allocation, to preserve portfolio value
            total_value = 0.0
            for leg_id in leg_returns.keys():
                if prev_date in leg_returns[leg_id].index:
                    # Apply return to allocation
                    prev_allocation = allocations.loc[prev_date, leg_id]
                    leg_return = leg_returns[leg_id].loc[prev_date]
                    new_allocation = prev_allocation * (1.0 + leg_return)
                    current_allocation[leg_id] = new_allocation
                    total_value += new_allocation
            
            # Check if this is a rebalancing date
            if date in rebalance_dates:
                # Rebalance: redistribute total value according to target weights
                # Use dynamic weights if available, otherwise use static weights
                if self.dynamic_weights_df is not None:
                    # Check if we have weights for this exact date
                    if date in self.dynamic_weights_df.index:
                        # Get dynamic weights for this date
                        for leg_id in leg_returns.keys():
                            if leg_id in self.dynamic_weights_df.columns:
                                target_weight = self.dynamic_weights_df.loc[date, leg_id]
                            else:
                                target_weight = self.weights.get(leg_id, 0.0)
                            current_allocation[leg_id] = total_value * target_weight
                    else:
                        # If exact date not found, try to find the most recent weight before this date
                        # This handles cases where rebalance dates might not match exactly
                        available_dates = self.dynamic_weights_df.index[self.dynamic_weights_df.index <= date]
                        if len(available_dates) > 0:
                            # Use the most recent available weight
                            weight_date = available_dates[-1]
                            for leg_id in leg_returns.keys():
                                if leg_id in self.dynamic_weights_df.columns:
                                    target_weight = self.dynamic_weights_df.loc[weight_date, leg_id]
                                else:
                                    target_weight = self.weights.get(leg_id, 0.0)
                                current_allocation[leg_id] = total_value * target_weight
                        else:
                            # No dynamic weights available, use static weights
                            for leg_id in leg_returns.keys():
                                target_weight = self.weights.get(leg_id, 0.0)
                                current_allocation[leg_id] = total_value * target_weight
                else:
                    # Use static weights
                    for leg_id in leg_returns.keys():
                        target_weight = self.weights.get(leg_id, 0.0)
                        current_allocation[leg_id] = total_value * target_weight
            
            # Store current state
            portfolio_value.loc[date] = total_value
            for leg_id in leg_returns.keys():
                allocations.loc[date, leg_id] = current_allocation[leg_id]
                if total_value > 0:
                    actual_weights.loc[date, leg_id] = current_allocation[leg_id] / total_value
                else:
                    actual_weights.loc[date, leg_id] = self.weights.get(leg_id, 0.0)
        
        # Calculate returns from portfolio value
        self.equity = portfolio_value
        self.returns = portfolio_value.pct_change().fillna(0.0)
        
        # Store allocation tracking
        self.allocations = allocations
        self.actual_weights = actual_weights
        
        # Create combined portfolio DataFrame
        self.portfolio_df = pd.DataFrame({
            'portfolio_returns': self.returns,
            'portfolio_equity': self.equity
        }, index=common_index)
        
        # Create weights DataFrame (showing allocation over time)
        if hasattr(self, 'actual_weights'):
            # Use actual weights from rebalancing if available
            self.weights_df = self.actual_weights.copy()
        else:
            # For non-rebalanced portfolios, show target weights
            weights_df_data = {}
            for leg_id, leg in self.legs.items():
                if self.weights.get(leg_id, 0.0) > 0:
                    weights_df_data[leg_id] = pd.Series(
                        self.weights[leg_id], 
                        index=common_index
                    )
            self.weights_df = pd.DataFrame(weights_df_data, index=common_index)


def combine_portfolios(
    legs: Dict[str, LegResult],
    weights: Dict[str, float],
    initial_capital: float,
    rebalance_freq: Optional[str] = None,
    dynamic_weights_df: Optional[pd.DataFrame] = None
) -> CombinedPortfolio:
    """
    Combine multiple strategy portfolios with specified weights.
    
    This function combines returns from individual strategies by:
    1. Taking each strategy's returns
    2. Weighting them according to the provided weights
    3. Summing the weighted returns
    4. Calculating the combined equity curve
    
    If rebalance_freq is specified, the portfolio will be rebalanced periodically
    back to target weights, ensuring allocations stay aligned with targets.
    
    Args:
        legs: Dictionary mapping leg_id to LegResult (from individual strategy backtests)
        weights: Dictionary mapping leg_id to allocation weight (must sum to 1.0)
                 Used as fallback if dynamic_weights_df is not provided
        initial_capital: Initial capital amount
        rebalance_freq: Optional rebalancing frequency. Options:
            - None: No rebalancing (weights applied to returns directly)
            - 'D': Daily rebalancing
            - 'W': Weekly rebalancing
            - 'M': Monthly rebalancing
            - 'Q': Quarterly rebalancing
            - 'Y' or 'A': Yearly rebalancing
        dynamic_weights_df: Optional DataFrame with dynamic weights over time.
                           Index should be dates, columns should be leg_ids.
                           If provided, weights will be looked up from this DataFrame
                           at each rebalancing date instead of using static weights.
    
    Returns:
        CombinedPortfolio object with combined returns and equity curve
    """
    return CombinedPortfolio(
        legs, 
        weights, 
        initial_capital, 
        rebalance_freq=rebalance_freq,
        dynamic_weights_df=dynamic_weights_df
    )

