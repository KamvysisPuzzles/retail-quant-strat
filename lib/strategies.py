from typing import Callable, Dict, Tuple, Union
import numpy as np
import pandas as pd
try:
    import talib
except ImportError:
    talib = None
    print("Warning: talib not installed. TQQQ strategy requires talib. Install with: pip install TA-Lib")
try:
    import vectorbt as vbt
except ImportError:
    vbt = None


StrategyFunc = Callable[[pd.DataFrame, Dict], Union[pd.Series, Tuple[pd.Series, pd.Series]]]

STRATEGIES: Dict[str, StrategyFunc] = {}


def register_strategy(name: str, func: StrategyFunc) -> None:
    STRATEGIES[name] = func


def sma_cross(prices: pd.DataFrame, params: Dict) -> pd.Series:
    close = prices["close"].astype(float)
    fast = int(params.get("fast", 50))
    slow = int(params.get("slow", 200))
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    signal = (ma_fast > ma_slow).astype(int)
    return signal.rename("exposure")


def tqqq_3ema_macdv_aroon(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    TQQQ 3EMA + MACD-V + Aroon ensemble strategy with OR logic.
    Matches the implementation from TQQQ_3EMA_MACDV_Aroon.ipynb.
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults matching the notebook
    ema1 = int(params.get("ema1", 12))
    ema2 = int(params.get("ema2", 89))
    ema3 = int(params.get("ema3", 125))
    macd_fast = int(params.get("macd_fast", 25))
    macd_slow = int(params.get("macd_slow", 30))
    macd_signal = int(params.get("macd_signal", 85))
    volume_threshold = float(params.get("volume_threshold", 0.0))
    aroon_length = int(params.get("aroon_length", 66))
    
    # Calculate EMAs using vectorbt if available, otherwise pandas
    if vbt is not None:
        ema1_ma = vbt.MA.run(close, ema1, ewm=True).ma
        ema2_ma = vbt.MA.run(close, ema2, ewm=True).ma
        ema3_ma = vbt.MA.run(close, ema3, ewm=True).ma
        
        # 3EMA Signals using vectorbt's crossover methods
        entries_3ema = (
            ema1_ma.vbt.crossed_above(ema2_ma) |
            ema1_ma.vbt.crossed_above(ema3_ma) |
            ema2_ma.vbt.crossed_above(ema3_ma)
        )
        exits_3ema = (
            ema1_ma.vbt.crossed_below(ema2_ma) |
            ema1_ma.vbt.crossed_below(ema3_ma) |
            ema2_ma.vbt.crossed_below(ema3_ma)
        )
        # Convert vectorbt signals to pandas Series
        entries_3ema = pd.Series(np.asarray(entries_3ema).ravel(), index=close.index, dtype=bool)
        exits_3ema = pd.Series(np.asarray(exits_3ema).ravel(), index=close.index, dtype=bool)
    else:
        # Fallback to pandas if vectorbt not available
        ema1_ma = close.ewm(span=ema1, adjust=False).mean()
        ema2_ma = close.ewm(span=ema2, adjust=False).mean()
        ema3_ma = close.ewm(span=ema3, adjust=False).mean()
        
        # 3EMA Signals (crossover-based)
        entries_3ema = (
            (ema1_ma > ema2_ma) & (ema1_ma.shift(1) <= ema2_ma.shift(1)) |
            (ema1_ma > ema3_ma) & (ema1_ma.shift(1) <= ema3_ma.shift(1)) |
            (ema2_ma > ema3_ma) & (ema2_ma.shift(1) <= ema3_ma.shift(1))
        )
        exits_3ema = (
            (ema1_ma < ema2_ma) & (ema1_ma.shift(1) >= ema2_ma.shift(1)) |
            (ema1_ma < ema3_ma) & (ema1_ma.shift(1) >= ema3_ma.shift(1)) |
            (ema2_ma < ema3_ma) & (ema2_ma.shift(1) >= ema3_ma.shift(1))
        )
    
    # MACD-V Signals using talib
    close_array = close.values
    macd, macd_signal_line, _ = talib.MACD(close_array, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    macd_series = pd.Series(macd, index=close.index)
    signal_series = pd.Series(macd_signal_line, index=close.index)
    
    # Volume confirmation - match notebook exactly
    if "volume" in prices.columns:
        volume_data = prices["volume"].astype(float)
        volume_aligned = volume_data.reindex(close.index)
        volume_sma = volume_aligned.rolling(window=20).mean()
        # Match reference notebook exactly: simple division (NaN values handled by fillna(False) later)
        volume_ratio = volume_aligned / volume_sma
    else:
        # No volume data - create volume_ratio with all True (no volume filter)
        volume_ratio = pd.Series(1.0, index=close.index)
    
    # MACD-V entries/exits
    macd_bullish_cross = (macd_series > signal_series) & (macd_series.shift(1) <= signal_series.shift(1))
    macd_bearish_cross = (macd_series < signal_series) & (macd_series.shift(1) >= signal_series.shift(1))
    volume_confirm = volume_ratio > volume_threshold
    entries_macdv = (macd_bullish_cross & volume_confirm).fillna(False)
    exits_macdv = (macd_bearish_cross & volume_confirm).fillna(False)
    
    # Aroon Signals using talib
    # Match original notebook: uses close_array for both high and low parameters
    aroon_up, aroon_down = talib.AROON(close_array, close_array, timeperiod=aroon_length)
    aroon_up_series = pd.Series(aroon_up, index=close.index)
    aroon_down_series = pd.Series(aroon_down, index=close.index)
    
    # Use vectorbt's crossover methods if available
    if vbt is not None:
        entries_aroon = aroon_up_series.vbt.crossed_above(aroon_down_series)
        exits_aroon = aroon_up_series.vbt.crossed_below(aroon_down_series)
        # Convert to pandas Series
        entries_aroon = pd.Series(np.asarray(entries_aroon).ravel(), index=close.index, dtype=bool)
        exits_aroon = pd.Series(np.asarray(exits_aroon).ravel(), index=close.index, dtype=bool)
    else:
        entries_aroon = (aroon_up_series > aroon_down_series) & (aroon_up_series.shift(1) <= aroon_down_series.shift(1))
        exits_aroon = (aroon_up_series < aroon_down_series) & (aroon_up_series.shift(1) >= aroon_down_series.shift(1))
    
    # Combine with OR logic
    entries_final = entries_3ema | entries_macdv | entries_aroon
    exits_final = exits_3ema | exits_macdv | exits_aroon
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries_final).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits_final).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def three_ema_macdv(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    3EMA + MACD-V ensemble strategy with OR logic (without Aroon).
    Identical to tqqq_3ema_macdv_aroon but excludes Aroon signals.
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults matching the notebook
    ema1 = int(params.get("ema1", 12))
    ema2 = int(params.get("ema2", 89))
    ema3 = int(params.get("ema3", 125))
    macd_fast = int(params.get("macd_fast", 25))
    macd_slow = int(params.get("macd_slow", 30))
    macd_signal = int(params.get("macd_signal", 85))
    volume_threshold = float(params.get("volume_threshold", 0.0))
    
    # Calculate EMAs using vectorbt if available, otherwise pandas
    if vbt is not None:
        ema1_ma = vbt.MA.run(close, ema1, ewm=True).ma
        ema2_ma = vbt.MA.run(close, ema2, ewm=True).ma
        ema3_ma = vbt.MA.run(close, ema3, ewm=True).ma
        
        # 3EMA Signals using vectorbt's crossover methods
        entries_3ema = (
            ema1_ma.vbt.crossed_above(ema2_ma) |
            ema1_ma.vbt.crossed_above(ema3_ma) |
            ema2_ma.vbt.crossed_above(ema3_ma)
        )
        exits_3ema = (
            ema1_ma.vbt.crossed_below(ema2_ma) |
            ema1_ma.vbt.crossed_below(ema3_ma) |
            ema2_ma.vbt.crossed_below(ema3_ma)
        )
        # Convert vectorbt signals to pandas Series
        entries_3ema = pd.Series(np.asarray(entries_3ema).ravel(), index=close.index, dtype=bool)
        exits_3ema = pd.Series(np.asarray(exits_3ema).ravel(), index=close.index, dtype=bool)
    else:
        # Fallback to pandas if vectorbt not available
        ema1_ma = close.ewm(span=ema1, adjust=False).mean()
        ema2_ma = close.ewm(span=ema2, adjust=False).mean()
        ema3_ma = close.ewm(span=ema3, adjust=False).mean()
        
        # 3EMA Signals (crossover-based)
        entries_3ema = (
            (ema1_ma > ema2_ma) & (ema1_ma.shift(1) <= ema2_ma.shift(1)) |
            (ema1_ma > ema3_ma) & (ema1_ma.shift(1) <= ema3_ma.shift(1)) |
            (ema2_ma > ema3_ma) & (ema2_ma.shift(1) <= ema3_ma.shift(1))
        )
        exits_3ema = (
            (ema1_ma < ema2_ma) & (ema1_ma.shift(1) >= ema2_ma.shift(1)) |
            (ema1_ma < ema3_ma) & (ema1_ma.shift(1) >= ema3_ma.shift(1)) |
            (ema2_ma < ema3_ma) & (ema2_ma.shift(1) >= ema3_ma.shift(1))
        )
    
    # MACD-V Signals using talib
    close_array = close.values
    macd, macd_signal_line, _ = talib.MACD(close_array, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    macd_series = pd.Series(macd, index=close.index)
    signal_series = pd.Series(macd_signal_line, index=close.index)
    
    # Volume confirmation - match notebook exactly
    if "volume" in prices.columns:
        volume_data = prices["volume"].astype(float)
        volume_aligned = volume_data.reindex(close.index)
        volume_sma = volume_aligned.rolling(window=20).mean()
        # Match reference notebook exactly: simple division (NaN values handled by fillna(False) later)
        volume_ratio = volume_aligned / volume_sma
    else:
        # No volume data - create volume_ratio with all True (no volume filter)
        volume_ratio = pd.Series(1.0, index=close.index)
    
    # MACD-V entries/exits
    macd_bullish_cross = (macd_series > signal_series) & (macd_series.shift(1) <= signal_series.shift(1))
    macd_bearish_cross = (macd_series < signal_series) & (macd_series.shift(1) >= signal_series.shift(1))
    volume_confirm = volume_ratio > volume_threshold
    entries_macdv = (macd_bullish_cross & volume_confirm).fillna(False)
    exits_macdv = (macd_bearish_cross & volume_confirm).fillna(False)
    
    # Combine 3EMA and MACD-V with OR logic (NO AROON)
    entries_final = entries_3ema | entries_macdv
    exits_final = exits_3ema | exits_macdv
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries_final).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits_final).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def three_ema_macd_rsi(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    3EMA + MACD-V + RSI ensemble strategy with OR logic.
    Combines trend-following (3EMA), momentum (MACD-V), and mean-reversion (RSI) signals.
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults
    ema1 = int(params.get("ema1", 39))
    ema2 = int(params.get("ema2", 98))
    ema3 = int(params.get("ema3", 327))
    macd_fast = int(params.get("macd_fast", 22))
    macd_slow = int(params.get("macd_slow", 33))
    macd_signal = int(params.get("macd_signal", 37))
    volume_threshold = float(params.get("volume_threshold", 0.0))
    rsi_period = int(params.get("rsi_period", 14))
    oversold_threshold = float(params.get("oversold_threshold", 30))
    overbought_threshold = float(params.get("overbought_threshold", 70))
    
    # Calculate EMAs using vectorbt if available, otherwise pandas
    if vbt is not None:
        ema1_ma = vbt.MA.run(close, ema1, ewm=True).ma
        ema2_ma = vbt.MA.run(close, ema2, ewm=True).ma
        ema3_ma = vbt.MA.run(close, ema3, ewm=True).ma
        
        # 3EMA Signals using vectorbt's crossover methods
        entries_3ema = (
            ema1_ma.vbt.crossed_above(ema2_ma) |
            ema1_ma.vbt.crossed_above(ema3_ma) |
            ema2_ma.vbt.crossed_above(ema3_ma)
        )
        exits_3ema = (
            ema1_ma.vbt.crossed_below(ema2_ma) |
            ema1_ma.vbt.crossed_below(ema3_ma) |
            ema2_ma.vbt.crossed_below(ema3_ma)
        )
        # Convert vectorbt signals to pandas Series
        entries_3ema = pd.Series(np.asarray(entries_3ema).ravel(), index=close.index, dtype=bool)
        exits_3ema = pd.Series(np.asarray(exits_3ema).ravel(), index=close.index, dtype=bool)
    else:
        # Fallback to pandas if vectorbt not available
        ema1_ma = close.ewm(span=ema1, adjust=False).mean()
        ema2_ma = close.ewm(span=ema2, adjust=False).mean()
        ema3_ma = close.ewm(span=ema3, adjust=False).mean()
        
        # 3EMA Signals (crossover-based)
        entries_3ema = (
            (ema1_ma > ema2_ma) & (ema1_ma.shift(1) <= ema2_ma.shift(1)) |
            (ema1_ma > ema3_ma) & (ema1_ma.shift(1) <= ema3_ma.shift(1)) |
            (ema2_ma > ema3_ma) & (ema2_ma.shift(1) <= ema3_ma.shift(1))
        )
        exits_3ema = (
            (ema1_ma < ema2_ma) & (ema1_ma.shift(1) >= ema2_ma.shift(1)) |
            (ema1_ma < ema3_ma) & (ema1_ma.shift(1) >= ema3_ma.shift(1)) |
            (ema2_ma < ema3_ma) & (ema2_ma.shift(1) >= ema3_ma.shift(1))
        )
    
    # MACD-V Signals using talib
    close_array = close.values
    macd, macd_signal_line, _ = talib.MACD(close_array, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    macd_series = pd.Series(macd, index=close.index)
    signal_series = pd.Series(macd_signal_line, index=close.index)
    
    # Volume confirmation
    if "volume" in prices.columns:
        volume_data = prices["volume"].astype(float)
        volume_aligned = volume_data.reindex(close.index)
        volume_sma = volume_aligned.rolling(window=20).mean()
        volume_ratio = volume_aligned / volume_sma
    else:
        # No volume data - create volume_ratio with all True (no volume filter)
        volume_ratio = pd.Series(1.0, index=close.index)
    
    # MACD-V entries/exits
    macd_bullish_cross = (macd_series > signal_series) & (macd_series.shift(1) <= signal_series.shift(1))
    macd_bearish_cross = (macd_series < signal_series) & (macd_series.shift(1) >= signal_series.shift(1))
    volume_confirm = volume_ratio > volume_threshold
    entries_macdv = (macd_bullish_cross & volume_confirm).fillna(False)
    exits_macdv = (macd_bearish_cross & volume_confirm).fillna(False)
    
    # RSI Signals using talib
    rsi = talib.RSI(close_array, timeperiod=rsi_period)
    rsi_series = pd.Series(rsi, index=close.index)
    
    # RSI mean-reversion signals
    # Entry: RSI crosses below oversold threshold
    # Exit: RSI crosses above overbought threshold
    entries_rsi = (rsi_series < oversold_threshold) & (rsi_series.shift(1) >= oversold_threshold)
    exits_rsi = (rsi_series > overbought_threshold) & (rsi_series.shift(1) <= overbought_threshold)
    
    # Convert RSI signals to Series
    entries_rsi = pd.Series(entries_rsi.fillna(False), index=close.index, dtype=bool)
    exits_rsi = pd.Series(exits_rsi.fillna(False), index=close.index, dtype=bool)
    
    # Combine with OR logic: 3EMA OR MACD-V OR RSI
    entries_final = entries_3ema | entries_macdv | entries_rsi
    exits_final = exits_3ema | exits_macdv | exits_rsi
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries_final).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits_final).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def macd_rsi(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    MACD-V + RSI ensemble strategy with OR logic.
    Combines momentum (MACD-V) and mean-reversion (RSI) signals.
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults
    macd_fast = int(params.get("macd_fast", 22))
    macd_slow = int(params.get("macd_slow", 33))
    macd_signal = int(params.get("macd_signal", 37))
    volume_threshold = float(params.get("volume_threshold", 0.0))
    rsi_period = int(params.get("rsi_period", 14))
    oversold_threshold = float(params.get("oversold_threshold", 30))
    overbought_threshold = float(params.get("overbought_threshold", 70))
    
    # MACD-V Signals using talib
    close_array = close.values
    macd, macd_signal_line, _ = talib.MACD(close_array, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    macd_series = pd.Series(macd, index=close.index)
    signal_series = pd.Series(macd_signal_line, index=close.index)
    
    # Volume confirmation
    if "volume" in prices.columns:
        volume_data = prices["volume"].astype(float)
        volume_aligned = volume_data.reindex(close.index)
        volume_sma = volume_aligned.rolling(window=20).mean()
        volume_ratio = volume_aligned / volume_sma
    else:
        # No volume data - create volume_ratio with all True (no volume filter)
        volume_ratio = pd.Series(1.0, index=close.index)
    
    # MACD-V entries/exits
    macd_bullish_cross = (macd_series > signal_series) & (macd_series.shift(1) <= signal_series.shift(1))
    macd_bearish_cross = (macd_series < signal_series) & (macd_series.shift(1) >= signal_series.shift(1))
    volume_confirm = volume_ratio > volume_threshold
    entries_macdv = (macd_bullish_cross & volume_confirm).fillna(False)
    exits_macdv = (macd_bearish_cross & volume_confirm).fillna(False)
    
    # Ensure MACD-V signals are properly indexed Series
    entries_macdv = pd.Series(entries_macdv, index=close.index, dtype=bool)
    exits_macdv = pd.Series(exits_macdv, index=close.index, dtype=bool)
    
    # RSI Signals using talib
    rsi = talib.RSI(close_array, timeperiod=rsi_period)
    rsi_series = pd.Series(rsi, index=close.index)
    
    # RSI mean-reversion signals
    # Entry: RSI crosses below oversold threshold
    # Exit: RSI crosses above overbought threshold
    entries_rsi = (rsi_series < oversold_threshold) & (rsi_series.shift(1) >= oversold_threshold)
    exits_rsi = (rsi_series > overbought_threshold) & (rsi_series.shift(1) <= overbought_threshold)
    
    # Convert RSI signals to Series with proper indexing
    entries_rsi = pd.Series(entries_rsi.fillna(False), index=close.index, dtype=bool)
    exits_rsi = pd.Series(exits_rsi.fillna(False), index=close.index, dtype=bool)
    
    # Combine with OR logic: MACD-V OR RSI
    # Ensure both Series are aligned before combining
    entries_final = entries_macdv | entries_rsi
    exits_final = exits_macdv | exits_rsi
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries_final).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits_final).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def macdv(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    Pure MACD-V strategy with volume confirmation.
    Uses only MACD-V signals (no other indicators).
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults
    macd_fast = int(params.get("macd_fast", 22))
    macd_slow = int(params.get("macd_slow", 33))
    macd_signal = int(params.get("macd_signal", 37))
    volume_threshold = float(params.get("volume_threshold", 0.0))
    
    # MACD-V Signals using talib
    close_array = close.values
    macd, macd_signal_line, _ = talib.MACD(close_array, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    macd_series = pd.Series(macd, index=close.index)
    signal_series = pd.Series(macd_signal_line, index=close.index)
    
    # Volume confirmation
    if "volume" in prices.columns:
        volume_data = prices["volume"].astype(float)
        volume_aligned = volume_data.reindex(close.index)
        volume_sma = volume_aligned.rolling(window=20).mean()
        volume_ratio = volume_aligned / volume_sma
    else:
        # No volume data - create volume_ratio with all True (no volume filter)
        volume_ratio = pd.Series(1.0, index=close.index)
    
    # MACD-V entries/exits
    macd_bullish_cross = (macd_series > signal_series) & (macd_series.shift(1) <= signal_series.shift(1))
    macd_bearish_cross = (macd_series < signal_series) & (macd_series.shift(1) >= signal_series.shift(1))
    volume_confirm = volume_ratio > volume_threshold
    entries = (macd_bullish_cross & volume_confirm).fillna(False)
    exits = (macd_bearish_cross & volume_confirm).fillna(False)
    
    # Ensure signals are properly indexed Series
    entries = pd.Series(entries, index=close.index, dtype=bool)
    exits = pd.Series(exits, index=close.index, dtype=bool)
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def rsi(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    Pure RSI strategy for mean-reversion trading.
    Uses only RSI signals (no other indicators).
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold_threshold = float(params.get("oversold_threshold", 30))
    overbought_threshold = float(params.get("overbought_threshold", 70))
    
    # RSI Signals using talib
    close_array = close.values
    rsi_values = talib.RSI(close_array, timeperiod=rsi_period)
    rsi_series = pd.Series(rsi_values, index=close.index)
    
    # RSI mean-reversion signals
    # Entry: RSI crosses below oversold threshold (oversold bounce)
    # Exit: RSI crosses above overbought threshold (overbought reversal)
    entries = (rsi_series < oversold_threshold) & (rsi_series.shift(1) >= oversold_threshold)
    exits = (rsi_series > overbought_threshold) & (rsi_series.shift(1) <= overbought_threshold)
    
    # Convert signals to Series with proper indexing
    entries = pd.Series(entries.fillna(False), index=close.index, dtype=bool)
    exits = pd.Series(exits.fillna(False), index=close.index, dtype=bool)
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def rsi_aroon(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    RSI + Aroon ensemble strategy with OR logic.
    Combines mean-reversion (RSI) and trend strength (Aroon) signals.
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults
    rsi_period = int(params.get("rsi_period", 14))
    oversold_threshold = float(params.get("oversold_threshold", 30))
    overbought_threshold = float(params.get("overbought_threshold", 70))
    aroon_length = int(params.get("aroon_length", 66))
    
    # RSI Signals using talib
    close_array = close.values
    rsi_values = talib.RSI(close_array, timeperiod=rsi_period)
    rsi_series = pd.Series(rsi_values, index=close.index)
    
    # RSI mean-reversion signals
    # Entry: RSI crosses below oversold threshold
    # Exit: RSI crosses above overbought threshold
    entries_rsi = (rsi_series < oversold_threshold) & (rsi_series.shift(1) >= oversold_threshold)
    exits_rsi = (rsi_series > overbought_threshold) & (rsi_series.shift(1) <= overbought_threshold)
    
    # Convert RSI signals to Series with proper indexing
    entries_rsi = pd.Series(entries_rsi.fillna(False), index=close.index, dtype=bool)
    exits_rsi = pd.Series(exits_rsi.fillna(False), index=close.index, dtype=bool)
    
    # Aroon Signals using talib
    # Match original notebook: uses close_array for both high and low parameters
    aroon_up, aroon_down = talib.AROON(close_array, close_array, timeperiod=aroon_length)
    aroon_up_series = pd.Series(aroon_up, index=close.index)
    aroon_down_series = pd.Series(aroon_down, index=close.index)
    
    # Use vectorbt's crossover methods if available
    if vbt is not None:
        entries_aroon = aroon_up_series.vbt.crossed_above(aroon_down_series)
        exits_aroon = aroon_up_series.vbt.crossed_below(aroon_down_series)
        # Convert to pandas Series
        entries_aroon = pd.Series(np.asarray(entries_aroon).ravel(), index=close.index, dtype=bool)
        exits_aroon = pd.Series(np.asarray(exits_aroon).ravel(), index=close.index, dtype=bool)
    else:
        entries_aroon = (aroon_up_series > aroon_down_series) & (aroon_up_series.shift(1) <= aroon_down_series.shift(1))
        exits_aroon = (aroon_up_series < aroon_down_series) & (aroon_up_series.shift(1) >= aroon_down_series.shift(1))
    
    # Ensure Aroon signals are properly indexed Series
    entries_aroon = pd.Series(entries_aroon, index=close.index, dtype=bool)
    exits_aroon = pd.Series(exits_aroon, index=close.index, dtype=bool)
    
    # Combine with OR logic: RSI OR Aroon
    entries_final = entries_rsi | entries_aroon
    exits_final = exits_rsi | exits_aroon
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries_final).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits_final).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def aroon(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    Pure Aroon strategy for trend-following trading.
    Uses only Aroon signals (no other indicators).
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with defaults
    aroon_length = int(params.get("aroon_length", 66))
    
    # Aroon Signals using talib
    close_array = close.values
    # Match original notebook: uses close_array for both high and low parameters
    aroon_up, aroon_down = talib.AROON(close_array, close_array, timeperiod=aroon_length)
    aroon_up_series = pd.Series(aroon_up, index=close.index)
    aroon_down_series = pd.Series(aroon_down, index=close.index)
    
    # Use vectorbt's crossover methods if available
    if vbt is not None:
        entries = aroon_up_series.vbt.crossed_above(aroon_down_series)
        exits = aroon_up_series.vbt.crossed_below(aroon_down_series)
        # Convert to pandas Series
        entries = pd.Series(np.asarray(entries).ravel(), index=close.index, dtype=bool)
        exits = pd.Series(np.asarray(exits).ravel(), index=close.index, dtype=bool)
    else:
        entries = (aroon_up_series > aroon_down_series) & (aroon_up_series.shift(1) <= aroon_down_series.shift(1))
        exits = (aroon_up_series < aroon_down_series) & (aroon_up_series.shift(1) >= aroon_down_series.shift(1))
    
    # Ensure signals are properly indexed Series
    entries = pd.Series(entries, index=close.index, dtype=bool)
    exits = pd.Series(exits, index=close.index, dtype=bool)
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


def btc_3ema_macdv_aroon(prices: pd.DataFrame, params: Dict) -> pd.Series:
    """
    BTC 3EMA + MACD-V + Aroon ensemble strategy with OR logic.
    Same logic as tqqq_3ema_macdv_aroon but with BTC-optimized default parameters.
    """
    if talib is None:
        raise ImportError("talib is required for this strategy. Install with: pip install TA-Lib")
    
    close = prices["close"].astype(float)
    
    # Extract parameters with BTC-optimized defaults
    ema1 = int(params.get("ema1", 10))
    ema2 = int(params.get("ema2", 64))
    ema3 = int(params.get("ema3", 126))
    macd_fast = int(params.get("macd_fast", 32))
    macd_slow = int(params.get("macd_slow", 40))
    macd_signal = int(params.get("macd_signal", 9))
    volume_threshold = float(params.get("volume_threshold", 0.0))
    aroon_length = int(params.get("aroon_length", 66))
    
    # Calculate EMAs using vectorbt if available, otherwise pandas
    if vbt is not None:
        ema1_ma = vbt.MA.run(close, ema1, ewm=True).ma
        ema2_ma = vbt.MA.run(close, ema2, ewm=True).ma
        ema3_ma = vbt.MA.run(close, ema3, ewm=True).ma
        
        # 3EMA Signals using vectorbt's crossover methods
        entries_3ema = (
            ema1_ma.vbt.crossed_above(ema2_ma) |
            ema1_ma.vbt.crossed_above(ema3_ma) |
            ema2_ma.vbt.crossed_above(ema3_ma)
        )
        exits_3ema = (
            ema1_ma.vbt.crossed_below(ema2_ma) |
            ema1_ma.vbt.crossed_below(ema3_ma) |
            ema2_ma.vbt.crossed_below(ema3_ma)
        )
        # Convert vectorbt signals to pandas Series
        entries_3ema = pd.Series(np.asarray(entries_3ema).ravel(), index=close.index, dtype=bool)
        exits_3ema = pd.Series(np.asarray(exits_3ema).ravel(), index=close.index, dtype=bool)
    else:
        # Fallback to pandas if vectorbt not available
        ema1_ma = close.ewm(span=ema1, adjust=False).mean()
        ema2_ma = close.ewm(span=ema2, adjust=False).mean()
        ema3_ma = close.ewm(span=ema3, adjust=False).mean()
        
        # 3EMA Signals (crossover-based)
        entries_3ema = (
            (ema1_ma > ema2_ma) & (ema1_ma.shift(1) <= ema2_ma.shift(1)) |
            (ema1_ma > ema3_ma) & (ema1_ma.shift(1) <= ema3_ma.shift(1)) |
            (ema2_ma > ema3_ma) & (ema2_ma.shift(1) <= ema3_ma.shift(1))
        )
        exits_3ema = (
            (ema1_ma < ema2_ma) & (ema1_ma.shift(1) >= ema2_ma.shift(1)) |
            (ema1_ma < ema3_ma) & (ema1_ma.shift(1) >= ema3_ma.shift(1)) |
            (ema2_ma < ema3_ma) & (ema2_ma.shift(1) >= ema3_ma.shift(1))
        )
    
    # MACD-V Signals using talib
    close_array = close.values
    macd, macd_signal_line, _ = talib.MACD(close_array, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    macd_series = pd.Series(macd, index=close.index)
    signal_series = pd.Series(macd_signal_line, index=close.index)
    
    # Volume confirmation - match notebook exactly
    if "volume" in prices.columns:
        volume_data = prices["volume"].astype(float)
        volume_aligned = volume_data.reindex(close.index)
        volume_sma = volume_aligned.rolling(window=20).mean()
        # Match reference notebook exactly: simple division (NaN values handled by fillna(False) later)
        volume_ratio = volume_aligned / volume_sma
    else:
        # No volume data - create volume_ratio with all True (no volume filter)
        volume_ratio = pd.Series(1.0, index=close.index)
    
    # MACD-V entries/exits
    macd_bullish_cross = (macd_series > signal_series) & (macd_series.shift(1) <= signal_series.shift(1))
    macd_bearish_cross = (macd_series < signal_series) & (macd_series.shift(1) >= signal_series.shift(1))
    volume_confirm = volume_ratio > volume_threshold
    entries_macdv = (macd_bullish_cross & volume_confirm).fillna(False)
    exits_macdv = (macd_bearish_cross & volume_confirm).fillna(False)
    
    # Aroon Signals using talib
    # Match original notebook: uses close_array for both high and low parameters
    aroon_up, aroon_down = talib.AROON(close_array, close_array, timeperiod=aroon_length)
    aroon_up_series = pd.Series(aroon_up, index=close.index)
    aroon_down_series = pd.Series(aroon_down, index=close.index)
    
    # Use vectorbt's crossover methods if available
    if vbt is not None:
        entries_aroon = aroon_up_series.vbt.crossed_above(aroon_down_series)
        exits_aroon = aroon_up_series.vbt.crossed_below(aroon_down_series)
        # Convert to pandas Series
        entries_aroon = pd.Series(np.asarray(entries_aroon).ravel(), index=close.index, dtype=bool)
        exits_aroon = pd.Series(np.asarray(exits_aroon).ravel(), index=close.index, dtype=bool)
    else:
        entries_aroon = (aroon_up_series > aroon_down_series) & (aroon_up_series.shift(1) <= aroon_down_series.shift(1))
        exits_aroon = (aroon_up_series < aroon_down_series) & (aroon_up_series.shift(1) >= aroon_down_series.shift(1))
    
    # Combine with OR logic
    entries_final = entries_3ema | entries_macdv | entries_aroon
    exits_final = exits_3ema | exits_macdv | exits_aroon
    
    # Convert to arrays first (matching original notebook behavior)
    entries_array = pd.Series(np.asarray(entries_final).ravel(), index=close.index, dtype=bool)
    exits_array = pd.Series(np.asarray(exits_final).ravel(), index=close.index, dtype=bool)
    
    # Return entries/exits directly (NOT exposure) to match original notebook
    # The backtest_leg function will handle these directly
    return (entries_array, exits_array)


register_strategy("sma_cross", sma_cross)
register_strategy("tqqq_3ema_macdv_aroon", tqqq_3ema_macdv_aroon)
register_strategy("btc_3ema_macdv_aroon", btc_3ema_macdv_aroon)
register_strategy("3ema_macdv", three_ema_macdv)
register_strategy("3ema_macd_rsi", three_ema_macd_rsi)
register_strategy("macd_rsi", macd_rsi)
register_strategy("macdv", macdv)
register_strategy("rsi", rsi)
register_strategy("rsi_aroon", rsi_aroon)
register_strategy("aroon", aroon)


