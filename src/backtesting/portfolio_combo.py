"""Portfolio combination utilities for combining multiple strategy portfolios."""
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Optional
from src.backtesting.legs import LegResult


class CombinedPortfolio:
    """
    Represents a combined portfolio from multiple strategy legs.
    """
    def __init__(
        self,
        legs: Dict[str, LegResult],
        weights: Dict[str, float],
        initial_capital: float,
        rebalance_freq: Optional[str] = None
    ):
        """
        Initialize combined portfolio.
        
        Args:
            legs: Dictionary mapping leg_id to LegResult
            weights: Dictionary mapping leg_id to weight (must sum to 1.0)
            initial_capital: Initial capital for portfolio
            rebalance_freq: Optional rebalancing frequency. Options:
                - None: No rebalancing (weights applied to returns directly)
                - 'D': Daily rebalancing
                - 'W': Weekly rebalancing
                - 'M': Monthly rebalancing
                - 'Q': Quarterly rebalancing
                - 'Y' or 'A': Yearly rebalancing
                - 'WOM-1MON' or similar: Custom pandas offset aliases
        """
        self.legs = legs
        self.weights = weights
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        
        # Validate weights sum to 1.0
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
        leg_returns = {}
        for leg_id, leg in self.legs.items():
            weight = self.weights.get(leg_id, 0.0)
            if weight > 0:
                if leg.returns.index.equals(common_index):
                    leg_returns[leg_id] = leg.returns
                else:
                    leg_returns[leg_id] = leg.returns.reindex(common_index).fillna(0.0)
        
        # Initialize tracking arrays
        portfolio_value = pd.Series(0.0, index=common_index)
        allocations = pd.DataFrame(0.0, index=common_index, columns=list(leg_returns.keys()))
        actual_weights = pd.DataFrame(0.0, index=common_index, columns=list(leg_returns.keys()))
        
        # Determine rebalancing dates
        if self.rebalance_freq == 'D':
            rebalance_dates = common_index  # Every day
        else:
            # Use pandas date offset to find rebalancing dates
            # Group by the frequency and take the first date of each period
            freq_map = {
                'W': 'W',
                'M': 'MS',  # Month start
                'Q': 'QS',  # Quarter start
                'Y': 'YS',  # Year start
                'A': 'YS',  # Year start (annual)
            }
            
            freq = freq_map.get(self.rebalance_freq.upper(), self.rebalance_freq)
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
            except Exception as e:
                print(f"Warning: Could not parse rebalance frequency '{self.rebalance_freq}', using no rebalancing. Error: {e}")
                self._combine_without_rebalancing(common_index)
                return
        
        # Track portfolio through time with rebalancing
        current_allocation = {leg_id: self.weights.get(leg_id, 0.0) * self.initial_capital 
                             for leg_id in leg_returns.keys()}
        portfolio_value.iloc[0] = self.initial_capital
        
        for i, date in enumerate(common_index):
            if i == 0:
                # Initial allocation
                for leg_id in leg_returns.keys():
                    allocations.loc[date, leg_id] = current_allocation[leg_id]
                    actual_weights.loc[date, leg_id] = self.weights.get(leg_id, 0.0)
                continue
            
            prev_date = common_index[i-1]
            
            # Calculate new portfolio value based on previous allocations and returns
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
    rebalance_freq: Optional[str] = None
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
        initial_capital: Initial capital amount
        rebalance_freq: Optional rebalancing frequency. Options:
            - None: No rebalancing (weights applied to returns directly)
            - 'D': Daily rebalancing
            - 'W': Weekly rebalancing
            - 'M': Monthly rebalancing
            - 'Q': Quarterly rebalancing
            - 'Y' or 'A': Yearly rebalancing
    
    Returns:
        CombinedPortfolio object with combined returns and equity curve
    """
    return CombinedPortfolio(legs, weights, initial_capital, rebalance_freq=rebalance_freq)

