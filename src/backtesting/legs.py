from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass
class LegResult:
    id: str
    symbol: str
    strategy_name: str
    params: Dict
    close: pd.Series
    exposure: pd.Series
    returns: pd.Series
    equity_curve: pd.Series
    stats: Dict
    portfolio: Optional[vbt.Portfolio] = None  # Store portfolio object for vectorbt metric access
    entries: Optional[pd.Series] = None  # Store entries for period-specific portfolio creation
    exits: Optional[pd.Series] = None  # Store exits for period-specific portfolio creation
    close_for_exec: Optional[pd.Series] = None  # Store execution prices for period-specific portfolio creation
    fees: float = 0.0005  # Store fees for period-specific portfolio creation
    slippage: float = 0.0005  # Store slippage for period-specific portfolio creation


def bps_to_fraction(bps: float) -> float:
    return float(bps) / 1e4


def backtest_leg(symbol: str, prices: pd.DataFrame, strategy_output, config: Dict, leg_id: str, strategy_name: str, params: Dict, safe_asset_prices: Optional[pd.DataFrame] = None, safe_asset_ticker: Optional[str] = None) -> LegResult:
    close = prices["close"].astype(float)
    
    # Handle both exposure Series and (entries, exits) tuple
    if isinstance(strategy_output, tuple) and len(strategy_output) == 2:
        # Strategy returns (entries, exits) directly
        entries, exits = strategy_output
        entries = entries.reindex(close.index).fillna(False).astype(bool)
        exits = exits.reindex(close.index).fillna(False).astype(bool)
        # Create exposure for return value
        exposure = pd.Series(0, index=close.index, dtype=int)
        in_position = False
        for idx in close.index:
            if exits.loc[idx] and in_position:
                in_position = False
            elif entries.loc[idx] and not in_position:
                in_position = True
            exposure.loc[idx] = 1 if in_position else 0
    else:
        # Legacy: strategy returns exposure Series
        exposure = strategy_output.reindex(close.index).fillna(0).astype(float)
        prev = exposure.shift(1).fillna(0)
        entries = (exposure > prev)
        exits = (exposure < prev)

    close_for_exec = close.copy()
    if config.get("execution_price", "close") == "open" and "open" in prices.columns:
        close_for_exec = prices["open"].astype(float)

    fees = bps_to_fraction(config.get("fee_bps", 0.0))
    slippage = bps_to_fraction(config.get("slip_bps", 0.0))
    
    # Use initial_capital from config (default 100000 to match reference notebook)
    initial_capital = config.get("initial_capital", 100_000.0)

    # Don't pass size parameter - let vectorbt use default (all available cash)
    # This matches the original notebook behavior exactly where size is not specified
    pf = vbt.Portfolio.from_signals(
        close=close_for_exec,
        entries=entries,
        exits=exits,
        # size parameter omitted - vectorbt defaults to using all available cash when entries/exits provided
        fees=fees,
        slippage=slippage,
        init_cash=initial_capital,
        cash_sharing=False,
        freq='D'
    )

    equity = pf.value()
    # Use vectorbt's returns() method instead of equity.pct_change() to match original notebook
    rets_raw = pf.returns()
    
    # Ensure returns index matches equity index exactly
    # If indices don't match, align returns to equity index
    if not rets_raw.index.equals(equity.index):
        # Reindex returns to match equity index, filling missing values with 0.0
        rets = rets_raw.reindex(equity.index).fillna(0.0)
    else:
        # Fill NaN values (typically first period) with 0.0
        rets = rets_raw.fillna(0.0)
    
    # Substitute safe asset returns when out of position (exposure == 0)
    if safe_asset_prices is not None and safe_asset_ticker is not None:
        # Align safe asset prices to returns index
        safe_asset_close = safe_asset_prices['close'].reindex(rets.index)
        # Forward fill and backfill to handle any missing values
        safe_asset_close = safe_asset_close.ffill().bfill()
        
        # Calculate safe asset returns
        safe_asset_rets = safe_asset_close.pct_change().fillna(0.0)
        
        # Ensure exposure is aligned to returns index
        exposure_aligned = exposure.reindex(rets.index).fillna(0).astype(int)
        exposure_prev = exposure_aligned.shift(1).fillna(0).astype(int)
        
        # Detect transitions: when we switch between strategy and safe asset
        # Transition TO safe asset: exit strategy (vectorbt already applied fees), now buy safe asset
        transition_to_safe = (exposure_prev > 0) & (exposure_aligned == 0)
        # Transition FROM safe asset: sell safe asset, then enter strategy (vectorbt will apply fees on entry)
        transition_from_safe = (exposure_prev == 0) & (exposure_aligned > 0)
        
        # Total transaction cost (fees + slippage) for buying/selling safe asset
        total_cost = fees + slippage
        
        # Adjust safe asset returns to account for transaction costs
        safe_asset_rets_adjusted = safe_asset_rets.copy()
        
        # When buying safe asset (after strategy exit), reduce return by transaction cost
        if transition_to_safe.any():
            safe_asset_rets_adjusted[transition_to_safe] = safe_asset_rets_adjusted[transition_to_safe] - total_cost
        
        # When selling safe asset (before strategy entry), reduce return by transaction cost
        if transition_from_safe.any():
            safe_asset_rets_adjusted[transition_from_safe] = safe_asset_rets_adjusted[transition_from_safe] - total_cost
        
        # Strategy returns already include fees/slippage from vectorbt for strategy entry/exit
        # But we need to account for selling safe asset when transitioning to strategy
        # Since we're selling safe asset first, we need to reduce the strategy return on transition day
        rets_adjusted = rets.copy()
        if transition_from_safe.any():
            # On days we sell safe asset to enter strategy, reduce strategy return by transaction cost
            # (vectorbt already accounts for strategy entry fees, so we're adding safe asset sale fees)
            rets_adjusted[transition_from_safe] = rets_adjusted[transition_from_safe] - total_cost
        
        # Replace 0 returns (cash) with safe asset returns when out of position
        # When exposure == 0, use safe asset returns; otherwise use strategy returns
        final_rets = pd.Series(
            np.where(exposure_aligned.values == 0, safe_asset_rets_adjusted.values, rets_adjusted.values),
            index=rets.index
        )
        
        # Reconstruct equity curve from modified returns
        final_equity = (1.0 + final_rets).cumprod() * initial_capital
        
        # Use the modified returns and equity
        rets = final_rets
        equity = final_equity

    stats = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe": float(pf.sharpe_ratio()) if hasattr(pf, "sharpe_ratio") else None,
        "max_drawdown": float(pf.max_drawdown()) if hasattr(pf, "max_drawdown") else None,
    }

    return LegResult(
        id=leg_id,
        symbol=symbol,
        strategy_name=strategy_name,
        params=params,
        close=close,
        exposure=exposure,
        returns=rets.rename(leg_id),
        equity_curve=equity.rename(leg_id),
        stats=stats,
        portfolio=pf,  # Store portfolio for vectorbt metric access
        entries=entries,  # Store entries for period-specific portfolio creation
        exits=exits,  # Store exits for period-specific portfolio creation
        close_for_exec=close_for_exec,  # Store execution prices for period-specific portfolio creation
        fees=fees,  # Store fees for period-specific portfolio creation
        slippage=slippage,  # Store slippage for period-specific portfolio creation
    )

