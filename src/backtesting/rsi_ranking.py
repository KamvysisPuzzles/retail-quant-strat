"""
Dynamic strategy ranking module.

This module contains all logic for ranking strategies using either:
- RSI (Relative Strength Index) as a momentum indicator
- Rolling Sortino ratio as a risk-adjusted return metric

Ranking logic:
1. Calculate RSI or Sortino for all strategies
2. Rank ALL strategies by the selected metric (higher = higher rank)
3. Select top N ranked strategies (based on weight_distribution)
4. Allocate weights to top strategies (regardless of buy/sell signals)
5. Remaining strategies get 0% allocation (or min_weight if specified)

Note: If a top strategy has a sell signal, it still receives allocation but will
hold the safe asset (via the existing safe_asset mechanism in backtest_leg).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
try:
    import talib
except ImportError:
    talib = None
from src.backtesting.legs import LegResult


@dataclass
class RSIRankingConfig:
    """Configuration for dynamic strategy ranking.
    
    Supports both RSI and Sortino-based ranking methods.
    
    Note: rsi_period and sortino_window are in the same frequency as your data. 
    For daily data, rsi_period=14 means 14 days. For quarterly rebalancing, 
    this does NOT mean 14 quarters - it still means 14 days of daily data.
    """
    ranking_method: str = 'rsi'  # 'rsi' or 'sortino'
    rsi_period: int = 14  # Period for RSI calculation (in same frequency as data: days for daily data)
    sortino_window: int = 63  # Rolling window for Sortino calculation (in same frequency as data: days for daily data)
    rsi_source: str = 'equity'  # 'equity' or 'returns' (only used for RSI method)
    risk_free_rate: float = 0.02  # Risk-free rate for Sortino calculation (annualized, e.g., 0.02 = 2%)
    freq_per_year: int = 252  # Trading days per year for annualization (252 for daily data)
    weight_distribution: str = 'top_4'  # 'linear', 'exponential', 'equal_top_n', 'top_3', or 'top_4'
    min_weight: float = 0.05  # Minimum allocation per strategy (5%)
    equal_top_n: Optional[int] = None  # For 'equal_top_n' distribution, number of top strategies


def calculate_rsi_for_strategies(
    legs: Dict[str, LegResult],
    date: pd.Timestamp,
    rsi_period: int,
    rsi_source: str = 'equity',
    use_data_before_date: bool = True
) -> Dict[str, float]:
    """
    Calculate RSI for all strategies at a given date.
    
    Note: rsi_period is in the same frequency as your data. For daily data,
    rsi_period=14 means 14 days (not 14 quarters or 14 months).
    
    Args:
        legs: Dictionary mapping leg_id to LegResult
        date: Date to calculate RSI up to (exclusive if use_data_before_date=True, inclusive otherwise)
        rsi_period: Period for RSI calculation (in same frequency as data: days for daily data)
        rsi_source: 'equity' or 'returns' - what to calculate RSI on
        use_data_before_date: If True, use data strictly before the date (to avoid look-ahead bias).
                             If False, use data up to and including the date.
    
    Returns:
        Dictionary mapping leg_id to RSI value (NaN if insufficient data)
    """
    if talib is None:
        raise ImportError("talib is required for RSI calculation. Install with: pip install TA-Lib")
    
    rsi_values = {}
    
    for leg_id, leg in legs.items():
        # Get data up to (but not including) the specified date to avoid look-ahead bias
        # This ensures RSI is calculated using only data available before the rebalance decision
        if rsi_source == 'equity':
            # Use equity curve, normalize to start at 1.0 for RSI calculation
            if use_data_before_date:
                data_series = leg.equity_curve[leg.equity_curve.index < date]
            else:
                data_series = leg.equity_curve[leg.equity_curve.index <= date]
            if len(data_series) > 0:
                # Normalize equity curve to start at 1.0
                first_value = data_series.iloc[0]
                if first_value > 0:
                    normalized = data_series / first_value
                else:
                    normalized = data_series
            else:
                rsi_values[leg_id] = np.nan
                continue
        else:  # rsi_source == 'returns'
            # Use returns directly
            if use_data_before_date:
                data_series = leg.returns[leg.returns.index < date]
            else:
                data_series = leg.returns[leg.returns.index <= date]
            normalized = data_series
        
        # Need at least rsi_period + 1 data points for RSI calculation
        if len(normalized) < rsi_period + 1:
            rsi_values[leg_id] = np.nan
            continue
        
        # Calculate RSI using talib
        try:
            # Convert to numpy array for talib
            values = normalized.values.astype(float)
            rsi = talib.RSI(values, timeperiod=rsi_period)
            
            # Get the last valid RSI value
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                rsi_values[leg_id] = float(rsi[-1])
            else:
                rsi_values[leg_id] = np.nan
        except Exception as e:
            # If RSI calculation fails, set to NaN
            rsi_values[leg_id] = np.nan
    
    return rsi_values


def get_strategies_with_buy_signal(
    legs: Dict[str, LegResult],
    date: pd.Timestamp
) -> List[str]:
    """
    Get list of strategy leg_ids that have a buy signal (in position) at the given date.
    
    Args:
        legs: Dictionary mapping leg_id to LegResult
        date: Date to check for buy signals
    
    Returns:
        List of leg_ids that have exposure > 0 at the given date
    """
    strategies_with_signal = []
    for leg_id, leg in legs.items():
        # Check if strategy has exposure (buy signal) at this date
        if date in leg.exposure.index:
            exposure = leg.exposure.loc[date]
            if exposure > 0:
                strategies_with_signal.append(leg_id)
    return strategies_with_signal


def rank_strategies_by_rsi(
    rsi_values: Dict[str, float],
    strategies_with_signal: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Rank strategies by RSI value (higher RSI = higher rank).
    
    If strategies_with_signal is provided, only rank strategies that have buy signals.
    
    Args:
        rsi_values: Dictionary mapping leg_id to RSI value
        strategies_with_signal: Optional list of leg_ids that have buy signals.
                              If provided, only these strategies will be ranked.
    
    Returns:
        List of (leg_id, rsi_value) tuples sorted by RSI (descending)
    """
    # Filter by buy signals first if provided
    if strategies_with_signal is not None:
        rsi_values = {leg_id: rsi for leg_id, rsi in rsi_values.items() 
                     if leg_id in strategies_with_signal}
    
    # Filter out NaN values and sort by RSI (descending)
    valid_rsi = [(leg_id, rsi) for leg_id, rsi in rsi_values.items() if not np.isnan(rsi)]
    valid_rsi.sort(key=lambda x: x[1], reverse=True)
    
    return valid_rsi


def calculate_sortino_for_strategies(
    legs: Dict[str, LegResult],
    date: pd.Timestamp,
    window: int,
    risk_free_rate: float = 0.02,
    freq_per_year: int = 252,
    use_data_before_date: bool = True
) -> Dict[str, float]:
    """
    Calculate rolling Sortino ratio for all strategies at a given date.
    
    Sortino = (annualized excess return) / (annualized downside deviation)
    
    Args:
        legs: Dictionary mapping leg_id to LegResult
        date: Date to calculate Sortino up to (exclusive if use_data_before_date=True, inclusive otherwise)
        window: Rolling window for Sortino calculation (in same frequency as data: days for daily data)
        risk_free_rate: Annualized risk-free rate (default: 0.02 = 2%)
        freq_per_year: Trading periods per year for annualization (default: 252 for daily data)
        use_data_before_date: If True, use data strictly before the date (to avoid look-ahead bias).
                             If False, use data up to and including the date.
    
    Returns:
        Dictionary mapping leg_id to Sortino ratio (NaN if insufficient data)
    """
    sortino_values = {}
    
    for leg_id, leg in legs.items():
        # Get returns up to (but not including) the specified date to avoid look-ahead bias
        if use_data_before_date:
            returns_series = leg.returns[leg.returns.index < date]
        else:
            returns_series = leg.returns[leg.returns.index <= date]
        
        # Need at least window data points for rolling Sortino
        if len(returns_series) < window:
            sortino_values[leg_id] = np.nan
            continue
        
        # Get the last window returns
        window_returns = returns_series.iloc[-window:]
        
        # Calculate Sortino ratio
        try:
            # Calculate excess returns (daily)
            daily_rf_rate = risk_free_rate / freq_per_year
            excess_returns = window_returns - daily_rf_rate
            
            # Calculate downside deviation (only negative returns)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                # Mean excess return
                mean_excess = excess_returns.mean()
                
                # Downside deviation (annualized)
                downside_std = downside_returns.std() * np.sqrt(freq_per_year)
                
                # Sortino ratio (annualized)
                if downside_std > 0:
                    sortino = (mean_excess * freq_per_year) / downside_std
                else:
                    # No downside deviation - if positive returns, Sortino is infinite
                    sortino = np.inf if mean_excess > 0 else np.nan
            else:
                # No negative returns or no downside deviation
                mean_excess = excess_returns.mean()
                if mean_excess > 0:
                    # All positive returns - Sortino is infinite
                    sortino = np.inf
                else:
                    # All negative or zero returns
                    sortino = np.nan
            
            sortino_values[leg_id] = float(sortino) if not np.isinf(sortino) else np.inf
            
        except Exception as e:
            # If Sortino calculation fails, set to NaN
            sortino_values[leg_id] = np.nan
    
    return sortino_values


def rank_strategies_by_sortino(
    sortino_values: Dict[str, float],
    strategies_with_signal: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Rank strategies by Sortino ratio (higher Sortino = higher rank).
    
    If strategies_with_signal is provided, only rank strategies that have buy signals.
    
    Args:
        sortino_values: Dictionary mapping leg_id to Sortino ratio
        strategies_with_signal: Optional list of leg_ids that have buy signals.
                              If provided, only these strategies will be ranked.
    
    Returns:
        List of (leg_id, sortino_value) tuples sorted by Sortino (descending)
    """
    # Filter by buy signals first if provided
    if strategies_with_signal is not None:
        sortino_values = {leg_id: sortino for leg_id, sortino in sortino_values.items() 
                         if leg_id in strategies_with_signal}
    
    # Filter out NaN values and sort by Sortino (descending)
    # Handle infinity values (treat as highest rank)
    valid_sortino = []
    for leg_id, sortino in sortino_values.items():
        if not np.isnan(sortino):
            valid_sortino.append((leg_id, sortino))
    
    # Sort by Sortino (descending), with infinity values first
    valid_sortino.sort(key=lambda x: (x[1] != np.inf, -x[1] if x[1] != np.inf else 0), reverse=True)
    
    return valid_sortino


def rankings_to_weights(
    ranked_strategies: List[Tuple[str, float]],
    weight_distribution: str = 'exponential',
    min_weight: float = 0.05,
    equal_top_n: Optional[int] = None
) -> Dict[str, float]:
    """
    Convert strategy rankings to allocation weights.
    
    Args:
        ranked_strategies: List of (leg_id, rsi_value) tuples sorted by rank (best first)
        weight_distribution: 'linear', 'exponential', 'equal_top_n', 'top_3', or 'top_4'
        min_weight: Minimum allocation per strategy (used for strategies not in top 3 for 'top_3' distribution)
        equal_top_n: For 'equal_top_n' distribution, number of top strategies
    
    Returns:
        Dictionary mapping leg_id to weight (sums to 1.0)
    """
    if len(ranked_strategies) == 0:
        return {}
    
    n_strategies = len(ranked_strategies)
    weights = {}
    
    if weight_distribution == 'top_3':
        # Top 3 strategies get 50%, 30%, 20% respectively
        # Remaining strategies get 0% (or min_weight if specified)
        top_3_weights = [0.50, 0.30, 0.20]
        n_top = min(3, n_strategies)
        
        # Allocate to top 3
        for i in range(n_top):
            leg_id, _ = ranked_strategies[i]
            weights[leg_id] = top_3_weights[i]
        
        # Remaining strategies get 0% (or min_weight if specified)
        n_remaining = n_strategies - n_top
        if n_remaining > 0:
            if min_weight > 0:
                # Allocate min_weight to remaining strategies
                total_min_weight = min_weight * n_remaining
                # Adjust top 3 weights to make room for min_weight
                available_for_top = 1.0 - total_min_weight
                if available_for_top > 0:
                    # Scale top 3 weights proportionally
                    scale_factor = available_for_top / sum(top_3_weights[:n_top])
                    for i in range(n_top):
                        leg_id, _ = ranked_strategies[i]
                        weights[leg_id] = weights[leg_id] * scale_factor
                    
                    # Assign min_weight to remaining
                    for i in range(n_top, n_strategies):
                        leg_id, _ = ranked_strategies[i]
                        weights[leg_id] = min_weight
                else:
                    # min_weight is too high, fall back to equal weights
                    for leg_id, _ in ranked_strategies:
                        weights[leg_id] = 1.0 / n_strategies
            else:
                # Remaining strategies get 0%
                for i in range(n_top, n_strategies):
                    leg_id, _ = ranked_strategies[i]
                    weights[leg_id] = 0.0
        
        # Normalize to ensure sum is exactly 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback to equal weights if something went wrong
            for leg_id, _ in ranked_strategies:
                weights[leg_id] = 1.0 / n_strategies
    
    elif weight_distribution == 'top_4':
        # Top 4 strategies get 45%, 35%, 15%, 5% respectively
        # Remaining strategies get 0% (or min_weight if specified)
        top_4_weights = [0.55, 0.30, 0.10, 0.05]
        n_top = min(4, n_strategies)
        
        # Allocate to top 4
        for i in range(n_top):
            leg_id, _ = ranked_strategies[i]
            weights[leg_id] = top_4_weights[i]
        
        # Remaining strategies get 0% (or min_weight if specified)
        n_remaining = n_strategies - n_top
        if n_remaining > 0:
            if min_weight > 0:
                # Allocate min_weight to remaining strategies
                total_min_weight = min_weight * n_remaining
                # Adjust top 4 weights to make room for min_weight
                available_for_top = 1.0 - total_min_weight
                if available_for_top > 0:
                    # Scale top 4 weights proportionally
                    scale_factor = available_for_top / sum(top_4_weights[:n_top])
                    for i in range(n_top):
                        leg_id, _ = ranked_strategies[i]
                        weights[leg_id] = weights[leg_id] * scale_factor
                    
                    # Assign min_weight to remaining
                    for i in range(n_top, n_strategies):
                        leg_id, _ = ranked_strategies[i]
                        weights[leg_id] = min_weight
                else:
                    # min_weight is too high, fall back to equal weights
                    for leg_id, _ in ranked_strategies:
                        weights[leg_id] = 1.0 / n_strategies
            else:
                # Remaining strategies get 0%
                for i in range(n_top, n_strategies):
                    leg_id, _ = ranked_strategies[i]
                    weights[leg_id] = 0.0
        
        # Normalize to ensure sum is exactly 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback to equal weights if something went wrong
            for leg_id, _ in ranked_strategies:
                weights[leg_id] = 1.0 / n_strategies
    
    elif weight_distribution == 'exponential':
        # Exponential distribution: top gets 50%, 2nd gets 25%, 3rd gets 15%, 4th gets 10%, etc.
        # Formula: weight[i] = 0.5 * (0.5 ** i) for i=0,1,2,3...
        # But we need to ensure they sum to 1.0 and respect min_weight
        total_available = 1.0 - (min_weight * n_strategies)
        if total_available < 0:
            # If min_weight is too high, just use equal weights
            for leg_id, _ in ranked_strategies:
                weights[leg_id] = 1.0 / n_strategies
        else:
            # Calculate exponential weights
            exp_weights = []
            for i in range(n_strategies):
                if i == 0:
                    exp_weights.append(0.5)
                else:
                    exp_weights.append(0.5 * (0.5 ** i))
            
            # Normalize to sum to total_available
            exp_sum = sum(exp_weights)
            if exp_sum > 0:
                exp_weights = [w * total_available / exp_sum for w in exp_weights]
            else:
                exp_weights = [total_available / n_strategies] * n_strategies
            
            # Add min_weight to each and assign
            for i, (leg_id, _) in enumerate(ranked_strategies):
                weights[leg_id] = exp_weights[i] + min_weight
    
    elif weight_distribution == 'linear':
        # Linear distribution: equal spacing
        # For 4 strategies: 40%, 30%, 20%, 10%
        total_available = 1.0 - (min_weight * n_strategies)
        if total_available < 0:
            # If min_weight is too high, just use equal weights
            for leg_id, _ in ranked_strategies:
                weights[leg_id] = 1.0 / n_strategies
        else:
            # Calculate linear weights (decreasing by equal amounts)
            # Sum of n, n-1, n-2, ... = n*(n+1)/2
            sum_ranks = n_strategies * (n_strategies + 1) / 2
            for i, (leg_id, _) in enumerate(ranked_strategies):
                rank_weight = (n_strategies - i) / sum_ranks * total_available
                weights[leg_id] = rank_weight + min_weight
    
    elif weight_distribution == 'equal_top_n':
        # Equal weights for top N strategies, rest get min_weight
        if equal_top_n is None or equal_top_n <= 0:
            equal_top_n = max(1, n_strategies // 2)  # Default to top half
        
        equal_top_n = min(equal_top_n, n_strategies)
        n_others = n_strategies - equal_top_n
        
        # Calculate weight for top N
        total_for_top = 1.0 - (min_weight * n_others)
        if total_for_top < 0:
            # If min_weight is too high, just use equal weights
            for leg_id, _ in ranked_strategies:
                weights[leg_id] = 1.0 / n_strategies
        else:
            weight_per_top = total_for_top / equal_top_n
            for i, (leg_id, _) in enumerate(ranked_strategies):
                if i < equal_top_n:
                    weights[leg_id] = weight_per_top
                else:
                    weights[leg_id] = min_weight
    else:
        # Unknown distribution, use equal weights
        for leg_id, _ in ranked_strategies:
            weights[leg_id] = 1.0 / n_strategies
    
    # Normalize to ensure sum is exactly 1.0 (handle floating point errors)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        # Fallback to equal weights
        weights = {leg_id: 1.0 / n_strategies for leg_id, _ in ranked_strategies}
    
    return weights


def calculate_dynamic_weights(
    legs: Dict[str, LegResult],
    rebalance_dates: pd.DatetimeIndex,
    config: RSIRankingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate dynamic weights using RSI momentum ranking on cumulative returns.
    
    RSI is calculated on the equity curve (cumulative returns) to measure long-term momentum.
    Strategies with RSI > 50 are considered to have positive momentum and receive allocation.
    Allocation is based on RSI ranking: higher RSI = higher allocation priority.
    
    Key features:
    - RSI calculated on cumulative returns (equity curve) - measures long-term momentum
    - RSI threshold of 50 (not 70/30) - separates positive vs negative momentum
    - Only strategies with RSI > 50 receive allocation
    - Equal allocation among momentum strategies (RSI > 50)
    - If only 1 strategy has RSI > 50: allocate 50% to it, 50% to cash
    - Remaining allocation is held in safe asset/cash
    
    Examples:
    - 5 strategies with RSI > 50: Each gets 20% (equal)
    - 3 strategies with RSI > 50: Each gets 33.33% (equal)
    - 2 strategies with RSI > 50: Each gets 50% (equal)
    - 1 strategy with RSI > 50: Gets 50%, 50% in cash
    - 0 strategies with RSI > 50: 100% in cash
    
    Args:
        legs: Dictionary mapping leg_id to LegResult
        rebalance_dates: Dates to calculate weights for
        config: Ranking configuration (must use RSI method with rsi_source='equity')
    
    Returns:
        Tuple of (weights_df, metric_df) DataFrames
        - weights_df: DataFrame with weights for each strategy at each rebalancing date
        - metric_df: DataFrame with RSI values for each strategy at each rebalancing date
    """
    leg_ids = list(legs.keys())
    
    # Ensure we're using RSI method on equity (cumulative returns)
    if config.ranking_method != 'rsi':
        raise ValueError("This function requires ranking_method='rsi' for momentum signal")
    
    if config.rsi_source != 'equity':
        print("Warning: rsi_source should be 'equity' for momentum signal on cumulative returns")
    
    min_data_length = config.rsi_period + 1
    metric_name = 'RSI'
    rsi_threshold = 50.0  # Momentum threshold (not overbought/oversold)
    
    # Track earliest valid date per leg
    leg_earliest_dates = {}
    for leg_id, leg in legs.items():
        leg_data = leg.equity_curve  # Use equity curve (cumulative returns)
        if len(leg_data) >= min_data_length:
            leg_min_date = leg_data.index[min_data_length - 1]
            leg_earliest_dates[leg_id] = leg_min_date
        else:
            leg_earliest_dates[leg_id] = None
    
    # Find the overall earliest date where at least one leg has enough data
    valid_earliest_dates = [d for d in leg_earliest_dates.values() if d is not None]
    if len(valid_earliest_dates) > 0:
        earliest_valid_date = min(valid_earliest_dates)
    else:
        earliest_valid_date = None
    
    # Initialize DataFrames
    weights_df = pd.DataFrame(0.0, index=rebalance_dates, columns=leg_ids)
    metric_df = pd.DataFrame(np.nan, index=rebalance_dates, columns=leg_ids)
    
    # Track how many dates had valid RSI values
    valid_dates_count = 0
    
    for date in rebalance_dates:
        # Calculate RSI on equity curve (cumulative returns) for all strategies
        # Use data strictly BEFORE the rebalance date to avoid look-ahead bias
        rsi_values = calculate_rsi_for_strategies(
            legs=legs,
            date=date,
            rsi_period=config.rsi_period,
            rsi_source='equity',  # Always use equity curve for momentum
            use_data_before_date=True
        )
        
        # Store RSI values
        for leg_id, rsi_value in rsi_values.items():
            metric_df.loc[date, leg_id] = rsi_value
        
        # Filter strategies with positive momentum (RSI > 50)
        strategies_with_momentum = [
            leg_id for leg_id, rsi in rsi_values.items() 
            if not np.isnan(rsi) and rsi > rsi_threshold
        ]
        
        n_momentum = len(strategies_with_momentum)
        
        # Initialize weights
        weights = {leg_id: 0.0 for leg_id in leg_ids}
        
        if n_momentum >= 2:
            # 2+ strategies with positive momentum: allocate equally
            equal_weight = 1.0 / n_momentum
            for leg_id in strategies_with_momentum:
                weights[leg_id] = equal_weight
        elif n_momentum == 1:
            # Only 1 strategy with positive momentum: allocate 50% to it, 50% to cash
            for leg_id in strategies_with_momentum:
                weights[leg_id] = 0.50
        # else: n_momentum == 0, all weights remain 0.0 (hold 100% cash)
        
        # Normalize weights to handle floating point errors
        total_weight = sum(weights.values())
        if total_weight > 1.0:
            # Floating point error - normalize down to 1.0
            weights = {leg_id: w / total_weight for leg_id, w in weights.items()}
        elif total_weight > 0.5 and total_weight < 1.0:
            # Floating point error - normalize up to 1.0
            weights = {leg_id: w / total_weight for leg_id, w in weights.items()}
        # If total_weight == 0.5 or 0.0, keep as is (intentional cash allocation)
        
        # Store weights
        for leg_id in leg_ids:
            weights_df.loc[date, leg_id] = weights[leg_id]
        
        if n_momentum > 0:
            valid_dates_count += 1
    
    # Print debugging information
    if earliest_valid_date is not None:
        print(f"   RSI momentum calculation requires {min_data_length} data points per leg.")
        print(f"   Using RSI threshold of {rsi_threshold} (positive momentum signal).")
        print(f"   Earliest valid RSI date per leg:")
        for leg_id, leg_date in leg_earliest_dates.items():
            if leg_date is not None:
                leg_data = legs[leg_id].equity_curve
                print(f"     {leg_id}: {leg_date.date()} ({len(leg_data)} total data points)")
            else:
                leg_data = legs[leg_id].equity_curve
                print(f"     {leg_id}: Insufficient data (only {len(leg_data)} points, need {min_data_length})")
        print(f"   First valid RSI date (earliest across all legs): {earliest_valid_date.date()}")
        print(f"   Valid RSI calculated for {valid_dates_count} of {len(rebalance_dates)} rebalancing dates.")
    else:
        print(f"   Warning: No legs have enough data for RSI calculation (need {min_data_length} data points)")
    
    return weights_df, metric_df

