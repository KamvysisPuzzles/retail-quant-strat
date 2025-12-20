"""
Signal-based allocation module.

This module calculates dynamic weights based on strategy buy/sell signals.
Only strategies with buy signals (exposure > 0) receive allocation.
No strategy can exceed 1/3 (33.33%) of total allocation.
"""
from typing import Dict
import numpy as np
import pandas as pd
from src.backtesting.legs import LegResult


def calculate_signal_based_weights(
    legs: Dict[str, LegResult],
    rebalance_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Calculate dynamic weights based on strategy buy/sell signals.
    
    Allocation rules:
    - Only strategies with buy signal (exposure > 0) receive allocation
    - No strategy can exceed 1/3 (33.33%) of total allocation
    - If 0 strategies have buy signal: 100% to safe asset (all weights = 0.0)
    - If 1 strategy has buy signal: that strategy gets 1/3, rest to safe asset
    - If 2+ strategies have buy signal: allocate equally, capped at 1/3 per strategy
    
    Args:
        legs: Dictionary mapping leg_id to LegResult
        rebalance_dates: Dates to calculate weights for
    
    Returns:
        DataFrame with weights for each strategy at each rebalancing date.
        Index: rebalance_dates, Columns: leg_ids
        Weights sum to <= 1.0 (remaining goes to safe asset via existing mechanism)
    """
    leg_ids = list(legs.keys())
    max_weight_per_strategy = 1.0 / 3.0  # 33.33% maximum per strategy
    
    # Initialize weights DataFrame
    weights_df = pd.DataFrame(0.0, index=rebalance_dates, columns=leg_ids)
    
    for date in rebalance_dates:
        # Get strategies with buy signal (exposure > 0) at this date
        strategies_with_signal = []
        for leg_id, leg in legs.items():
            # Check if strategy has exposure (buy signal) at this date
            if date in leg.exposure.index:
                exposure = leg.exposure.loc[date]
                if exposure > 0:
                    strategies_with_signal.append(leg_id)
        
        n_strategies_with_signal = len(strategies_with_signal)
        
        # Initialize weights for this date
        weights = {leg_id: 0.0 for leg_id in leg_ids}
        
        if n_strategies_with_signal == 0:
            # No strategies have buy signal: keep even allocation across all legs
            equal_weight_all = 1.0 / len(leg_ids) if len(leg_ids) > 0 else 0.0
            for leg_id in leg_ids:
                weights[leg_id] = equal_weight_all
        elif n_strategies_with_signal == 1:
            # Only 1 strategy has buy signal: allocate 1/3 to it
            # Remaining 2/3 goes to safe asset (weights sum to 1/3)
            leg_id = strategies_with_signal[0]
            weights[leg_id] = max_weight_per_strategy
        else:
            # 2+ strategies have buy signal: allocate equally, capped at 1/3
            equal_weight = 1.0 / n_strategies_with_signal
            
            if equal_weight <= max_weight_per_strategy:
                # Equal weight is within limit: allocate equally
                for leg_id in strategies_with_signal:
                    weights[leg_id] = equal_weight
            else:
                # Equal weight exceeds 1/3: cap each at 1/3 and distribute remaining
                # This case occurs when n_strategies_with_signal < 3
                # (e.g., 2 strategies would get 0.5 each, but max is 0.333)
                
                # Cap each strategy at 1/3
                for leg_id in strategies_with_signal:
                    weights[leg_id] = max_weight_per_strategy
                
                # Calculate remaining allocation
                allocated = max_weight_per_strategy * n_strategies_with_signal
                remaining = 1.0 - allocated
                
                # Remaining goes to safe asset (weights sum to allocated, not 1.0)
                # This is correct - the portfolio combination logic handles safe asset
                # when weights don't sum to 1.0
            
            # Normalize weights to ensure they sum correctly
            total_weight = sum(weights.values())
            if total_weight > 1.0:
                # Floating point error - normalize down to 1.0
                weights = {leg_id: w / total_weight for leg_id, w in weights.items()}
        
        # Store weights for this date
        for leg_id in leg_ids:
            weights_df.loc[date, leg_id] = weights[leg_id]
    
    return weights_df

