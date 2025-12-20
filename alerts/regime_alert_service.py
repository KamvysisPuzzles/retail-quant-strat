import os
import sys
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from hmmlearn import hmm

# Configuration - matches notebook settings
SYMBOL = "QQQ"
LEVERAGED_SYMBOL = "TQQQ"
VIX_SYMBOL = "^VIX"
RATES_SYMBOL = "^TNX"
FEATURE_SET_CONFIG = 'base'  # 'base' or 'extended'
CIRCUIT_BREAKER_ENABLED = False
CIRCUIT_BREAKER_THRESHOLD = -0.05

# Optional Risk Management Filters (set to None to disable)
MIN_REGIME_CONFIDENCE = None  # Minimum confidence to hold leveraged asset
MAX_VOLATILITY_FOR_LEVERAGE = None  # Max 20-day volatility to hold leveraged asset
MIN_VIX_FOR_LEVERAGE = None # If VIX > this, avoid leverage (use None to disable)

# Training window (use last 252 trading days for training)
TRAIN_WINDOW = 252

STATE_FILE = "regime_state.json"

def send_telegram(message):
    """Send message via Telegram bot"""
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable is not set")
        return False
    
    if not chat_id:
        print("Error: TELEGRAM_CHAT_ID environment variable is not set")
        return False
    
    token = token.strip()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"Error: Telegram API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False

def load_state():
    """Load previous regime state"""
    state_path = Path(__file__).parent / STATE_FILE
    if state_path.exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    """Save current regime state"""
    state_path = Path(__file__).parent / STATE_FILE
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

def calculate_features(qqq_data, vix_data, rates_data):
    """
    Calculate all features for the HMM-RF strategy
    
    Args:
        qqq_data: DataFrame with 'close' column
        vix_data: Series or DataFrame with 'close' column (aligned VIX prices)
        rates_data: Series or DataFrame with 'close' column (aligned rates)
    """
    # Validate input data
    if qqq_data.empty or len(qqq_data) < 2:
        raise ValueError(f"QQQ data must have at least 2 rows, got {len(qqq_data)}")
    if 'close' not in qqq_data.columns:
        raise ValueError("QQQ data missing 'close' column")
    if qqq_data['close'].isna().all():
        raise ValueError("All QQQ close prices are NaN")
    
    features_df = pd.DataFrame(index=qqq_data.index)
    
    # Base features (all properly lagged by 1 day)
    # Check if we have valid data before calculating returns
    valid_close = qqq_data['close'].dropna()
    if len(valid_close) < 2:
        raise ValueError(f"Need at least 2 valid close prices, got {len(valid_close)}")
    
    # Calculate rt (log returns) with validation
    close_series = qqq_data['close']
    if close_series.isna().all():
        raise ValueError("Cannot calculate returns: All QQQ close prices are NaN")
    if close_series.notna().sum() < 2:
        raise ValueError(f"Cannot calculate returns: Need at least 2 non-null values, got {close_series.notna().sum()}")
    
    features_df['rt'] = np.log(close_series / close_series.shift(1)).shift(1)
    
    # Calculate daily_return with error handling
    try:
        # Ensure we have valid data before calling pct_change
        if close_series.isna().all() or close_series.notna().sum() < 1:
            raise ValueError(f"Cannot calculate pct_change: insufficient valid data. "
                           f"Total rows: {len(close_series)}, Non-null: {close_series.notna().sum()}")
        features_df['daily_return'] = close_series.pct_change().shift(1)
    except (ValueError, IndexError) as e:
        error_msg = str(e)
        if "argmax of an empty sequence" in error_msg or "empty sequence" in error_msg:
            raise ValueError(f"Cannot calculate returns: QQQ close data issue. "
                           f"Data length: {len(qqq_data)}, "
                           f"Non-null values: {qqq_data['close'].notna().sum()}, "
                           f"First few values: {qqq_data['close'].head().tolist()}")
        raise
    features_df['rv_20'] = features_df['rt'].rolling(window=20).std() * np.sqrt(252)
    features_df['rv_5'] = features_df['rt'].rolling(window=5).std() * np.sqrt(252)
    features_df['mom_21'] = qqq_data['close'].pct_change(21).shift(1)
    
    # Handle vix_data and rates_data as either Series or DataFrame
    if isinstance(vix_data, pd.Series):
        features_df['vix'] = vix_data.shift(1)
    else:
        features_df['vix'] = vix_data['close'].shift(1) if 'close' in vix_data.columns else vix_data.shift(1)
    
    if isinstance(rates_data, pd.Series):
        features_df['rates'] = rates_data.shift(1)
    else:
        features_df['rates'] = rates_data['close'].shift(1) if 'close' in rates_data.columns else rates_data.shift(1)
    
    # Extended features
    if FEATURE_SET_CONFIG == 'extended':
        # Volatility features
        features_df['vol_term_structure'] = features_df['rv_5'] - features_df['rv_20']
        features_df['vol_of_vol'] = features_df['rt'].rolling(window=20).std().rolling(window=10).std() * np.sqrt(252)
        features_df['mom_5'] = qqq_data['close'].pct_change(5).shift(1)
        
        # Price structure
        ma_20 = qqq_data['close'].rolling(window=20).mean()
        ma_50 = qqq_data['close'].rolling(window=50).mean()
        features_df['price_ma20_ratio'] = (qqq_data['close'].shift(1) / ma_20.shift(1))
        features_df['price_ma50_ratio'] = (qqq_data['close'].shift(1) / ma_50.shift(1))
        features_df['ma20_ma50_ratio'] = (ma_20.shift(1) / ma_50.shift(1))
        
        # RSI
        delta = qqq_data['close'].diff().shift(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # VIX indicators
        vix_ma = features_df['vix'].rolling(window=20).mean()
        features_df['vix_ma_ratio'] = features_df['vix'] / vix_ma
        features_df['vix_percentile'] = features_df['vix'].rolling(window=252).rank(pct=True)
        
        # Rates
        rates_ma = features_df['rates'].rolling(window=60).mean()
        features_df['rates_ma_diff'] = features_df['rates'] - rates_ma
        
        # Risk features
        rolling_max = qqq_data['close'].rolling(window=20).max().shift(1)
        features_df['drawdown_20'] = (qqq_data['close'].shift(1) - rolling_max) / rolling_max
        features_df['return_skew'] = features_df['rt'].rolling(window=20).skew()
        features_df['vol_autocorr'] = features_df['rt'].rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x) == 20 and x.std() > 0 else 0
        )
        
        # Trend features
        ma_200 = qqq_data['close'].rolling(window=200).mean()
        features_df['price_ma200_ratio'] = (qqq_data['close'].shift(1) / ma_200.shift(1))
        features_df['ma_slope_20'] = (ma_20.shift(1) - ma_20.shift(6)) / ma_20.shift(6)
        features_df['ma_slope_50'] = (ma_50.shift(1) - ma_50.shift(11)) / ma_50.shift(11)
        
        ema_12 = qqq_data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = qqq_data['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        features_df['macd_signal_diff'] = (macd_line - macd_signal).shift(1)
        features_df['macd_histogram'] = macd_histogram.shift(1)
        
        # Trend score
        trend_components = pd.DataFrame(index=features_df.index)
        trend_components['price_pos'] = (features_df['price_ma50_ratio'] - 1.0) * 2
        trend_components['price_pos_200'] = (features_df['price_ma200_ratio'] - 1.0) * 2
        trend_components['momentum'] = features_df['mom_21'] * 10
        trend_components['ma_slope'] = features_df['ma_slope_50'] * 100
        trend_components['macd'] = features_df['macd_signal_diff'] * 100
        
        features_df['trend_score'] = (
            0.25 * trend_components['price_pos'] +
            0.15 * trend_components['price_pos_200'] +
            0.30 * trend_components['momentum'] +
            0.15 * trend_components['ma_slope'] +
            0.15 * trend_components['macd']
        )
    
    return features_df

def download_symbol_data(symbol, start_date, end_date=None, max_attempts=3):
    """
    Download symbol data with retry logic and error handling.
    
    Args:
        symbol: Stock symbol to download
        start_date: Start date for data
        end_date: End date for data (None for latest)
        max_attempts: Maximum number of download attempts
        
    Returns:
        DataFrame with downloaded data
        
    Raises:
        ValueError if download fails after all attempts
    """
    download_attempts = [
        {'auto_adjust': True, 'actions': True},
        {'auto_adjust': False, 'actions': False},
        {'auto_adjust': True, 'actions': False},
    ]
    
    for attempt_num, params in enumerate(download_attempts[:max_attempts], 1):
        try:
            print(f"  Attempt {attempt_num}/{max_attempts}: Downloading {symbol}...")
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date, 
                progress=False, 
                auto_adjust=params['auto_adjust'],
                actions=params['actions'],
                timeout=30
            )
            
            # Check if we got valid data
            if data is not None and not data.empty:
                print(f"  ✓ Successfully downloaded {symbol} ({len(data)} rows)")
                return data
                
        except Exception as e:
            error_str = str(e)
            # Handle specific yfinance errors
            if "timezone" in error_str.lower() or "delisted" in error_str.lower():
                print(f"  ⚠ Timezone/delisting warning for {symbol}, trying alternative method...")
            else:
                print(f"  ✗ Attempt {attempt_num} failed: {error_str[:100]}")
            
            # On last attempt, try Ticker object as fallback
            if attempt_num == len(download_attempts[:max_attempts]):
                try:
                    print(f"  Trying alternative download method (Ticker) for {symbol}...")
                    ticker = yf.Ticker(symbol)
                    # Try with period instead of dates if date range fails
                    try:
                        data = ticker.history(start=start_date, end=end_date, auto_adjust=params['auto_adjust'])
                    except:
                        # Fallback to period-based download
                        data = ticker.history(period="1y", auto_adjust=params['auto_adjust'])
                    
                    if data is not None and not data.empty:
                        # Filter to requested date range if needed
                        if start_date:
                            data = data[data.index >= pd.to_datetime(start_date)]
                        print(f"  ✓ Successfully downloaded {symbol} via Ticker ({len(data)} rows)")
                        return data
                except Exception as e2:
                    print(f"  ✗ Alternative method also failed: {str(e2)[:100]}")
            continue
    
    # All attempts failed
    error_msg = f"Failed to download data for {symbol} after {max_attempts} attempts.\n"
    error_msg += "Possible causes:\n"
    error_msg += "  - Network connectivity issues\n"
    error_msg += f"  - Symbol {symbol} may be temporarily unavailable\n"
    error_msg += "  - yfinance API issues or rate limiting\n"
    error_msg += "  - Market is closed (try again during market hours)\n"
    error_msg += "  - Symbol may have changed or been delisted"
    raise ValueError(error_msg)

def get_regime_prediction():
    """
    Get current regime prediction using HMM-RF model.
    
    IMPORTANT - Date Logic and Look-Ahead Bias Prevention:
    - All features are calculated with .shift(1) to ensure we only use data available
      before market open on the prediction date
    - If the last row in features_df is date T, its features use data from date T-1
    - We predict the regime for date T using features from T-1
    - This ensures no look-ahead bias: we never use same-day returns or prices
    
    Example:
    - If last available data is Friday, features_df[-1] is Friday's row
    - Friday's features are based on Thursday's data (due to .shift(1))
    - We predict Friday's regime using Thursday's metrics
    - This is correct because Thursday's data is available before Friday's market open
    """
    
    # Load data - use enough history for feature calculation
    # IMPORTANT: We need extra days because:
    # - Rolling windows create NaN at start (rv_20 needs 20 days, vix_percentile needs 252 days)
    # - .shift(1) operations create NaN in first row
    # - Market holidays reduce trading days (500 calendar days ≈ 350-360 trading days)
    # - After dropna(), we need at least 253 valid rows for training
    # So we request 500 calendar days to ensure we have enough trading days after feature calculation
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
    end_date = None  # Get latest data
    
    print("Loading market data...")
    
    # Download QQQ data with retry logic
    qqq_data = download_symbol_data(SYMBOL, start_date, end_date)
    
    # Clean up column names
    if isinstance(qqq_data.columns, pd.MultiIndex):
        qqq_data.columns = qqq_data.columns.droplevel(1)
    qqq_data.columns = qqq_data.columns.str.lower()
    
    # Validate QQQ data
    if qqq_data.empty:
        raise ValueError(f"No data downloaded for {SYMBOL} - DataFrame is empty")
    if 'close' not in qqq_data.columns:
        raise ValueError(f"Missing 'close' column in {SYMBOL} data")
    if qqq_data['close'].isna().all():
        raise ValueError(f"All 'close' values are NaN for {SYMBOL}")
    if len(qqq_data) < 50:
        raise ValueError(f"Insufficient data for {SYMBOL}: only {len(qqq_data)} rows")
    
    # Download VIX data with retry logic
    vix_data = download_symbol_data(VIX_SYMBOL, start_date, end_date)
    
    # Clean up column names
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.droplevel(1)
    vix_data.columns = vix_data.columns.str.lower()
    
    # Validate VIX data
    if 'close' not in vix_data.columns:
        raise ValueError(f"Missing 'close' column in {VIX_SYMBOL} data")
    
    # Download rates data with retry logic
    rates_data = download_symbol_data(RATES_SYMBOL, start_date, end_date)
    
    # Clean up column names
    if isinstance(rates_data.columns, pd.MultiIndex):
        rates_data.columns = rates_data.columns.droplevel(1)
    rates_data.columns = rates_data.columns.str.lower()
    
    # Validate rates data
    if 'close' not in rates_data.columns:
        raise ValueError(f"Missing 'close' column in {RATES_SYMBOL} data")
    
    # Align data
    aligned_index = qqq_data.index
    vix_aligned = vix_data['close'].reindex(aligned_index).ffill().bfill()
    rates_aligned = rates_data['close'].reindex(aligned_index).ffill().bfill()
    
    # Validate aligned data
    if vix_aligned.isna().all():
        raise ValueError(f"All VIX values are NaN after alignment")
    if rates_aligned.isna().all():
        raise ValueError(f"All rates values are NaN after alignment")
    
    # Calculate features
    print("Calculating features...")
    print(f"  Raw QQQ data: {len(qqq_data)} rows")
    features_df = calculate_features(qqq_data, vix_aligned, rates_aligned)
    print(f"  Features before dropna: {len(features_df)} rows")
    
    # Check which features have NaN values
    nan_counts = features_df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"  NaN counts by feature:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {count} NaN values")
    
    # Drop rows with NaN values
    features_df = features_df.dropna()
    print(f"  Features after dropna: {len(features_df)} rows")
    
    # Validate we have enough data
    # We need at least TRAIN_WINDOW days for training, ideally TRAIN_WINDOW + 1 for prediction
    if len(features_df) < TRAIN_WINDOW:
        error_msg = f"Not enough data after feature calculation. Need at least {TRAIN_WINDOW} days, got {len(features_df)}.\n"
        error_msg += f"  Raw QQQ data: {len(qqq_data)} rows\n"
        error_msg += f"  Features before dropna: {features_before_dropna} rows\n"
        error_msg += f"  Features after dropna: {len(features_df)} rows\n"
        error_msg += f"  Rows lost to NaN: {features_before_dropna - len(features_df)}\n"
        error_msg += f"  Date range: {qqq_data.index[0].date()} to {qqq_data.index[-1].date()}\n"
        error_msg += "\nWhy data is lost:\n"
        error_msg += "  - Rolling window features create NaN values:\n"
        error_msg += "    * rv_20 needs 20 days + 1 shift = ~21 rows lost\n"
        error_msg += "    * mom_21 needs 21 days + 1 shift = ~22 rows lost\n"
        error_msg += "    * vix_percentile needs 252 days = ~252 rows lost\n"
        error_msg += "  - .shift(1) operations create NaN in first row\n"
        error_msg += "  - Market holidays reduce trading days (400 calendar days ≈ 280-290 trading days)\n"
        error_msg += "\nSolution: Download more historical data by increasing the start_date range"
        raise ValueError(error_msg)
    
    # Adjust training window if we have exactly TRAIN_WINDOW days
    # In this case, we'll train on TRAIN_WINDOW - 1 days and predict on the last day
    actual_train_window = min(TRAIN_WINDOW, len(features_df) - 1)
    if actual_train_window < TRAIN_WINDOW:
        print(f"  ⚠ Warning: Only {len(features_df)} days available. Training on {actual_train_window} days instead of {TRAIN_WINDOW}")
    
    # Define feature sets
    base_features = ['rt', 'rv_20', 'mom_21', 'rv_5', 'vix', 'rates']
    extended_features = base_features + [
        'vol_term_structure', 'vol_of_vol', 'mom_5',
        'price_ma20_ratio', 'price_ma50_ratio', 'ma20_ma50_ratio',
        'rsi', 'vix_ma_ratio', 'vix_percentile', 'rates_ma_diff',
        'drawdown_20', 'return_skew', 'vol_autocorr',
        'price_ma200_ratio', 'ma_slope_20', 'ma_slope_50',
        'macd_signal_diff', 'macd_histogram', 'trend_score'
    ]
    
    if FEATURE_SET_CONFIG == 'base':
        feature_names = base_features
    else:
        feature_names = extended_features
    
    # Filter to only existing features
    feature_names = [f for f in feature_names if f in features_df.columns]
    
    # Use last actual_train_window days for training, predict for next trading day
    # IMPORTANT: features_df[-1] contains features calculated from the PREVIOUS day's data (due to .shift(1))
    # So if the last row is date T, its features use data from T-1, and we predict regime for date T
    # This ensures no look-ahead bias - we only use data available before market open on date T
    
    # Adjust slicing based on available data
    if len(features_df) >= TRAIN_WINDOW + 1:
        # Ideal case: we have enough data for full training window
        train_data = features_df.iloc[-TRAIN_WINDOW-1:-1]  # Last TRAIN_WINDOW days (excluding last row)
    else:
        # Fallback: use all data except the last row for training
        train_data = features_df.iloc[:-1]  # All rows except last
    
    today_features = features_df.iloc[-1:]  # Last row: features from previous day, used to predict this day's regime
    
    # The prediction_date is the date of the last row in features_df
    # This row's features are based on the previous day's data (properly lagged)
    # So we're predicting the regime for prediction_date using data from prediction_date - 1 day
    prediction_date = today_features.index[0]
    print(f"Training on {len(train_data)} days (requested: {TRAIN_WINDOW}, available: {len(features_df)})")
    print(f"Features date: {prediction_date.date()} (features calculated from previous day's data)")
    print(f"Predicting regime for: {prediction_date.date()} (using previous day's metrics)")
    
    # Scale features (fit on training, transform today)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[feature_names].values)
    X_today = scaler.transform(today_features[feature_names].values)
    
    # Train HMM
    print("Training HMM...")
    hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    hmm_model.fit(X_train)
    
    # Get HMM predictions
    hmm_train_states = hmm_model.predict(X_train)
    hmm_today_probs = hmm_model.predict_proba(X_today)[0]
    
    # Label regimes based on volatility
    train_state_vols = []
    for state in range(2):
        state_mask = hmm_train_states == state
        if state_mask.sum() > 0:
            avg_vol = train_data.loc[train_data.index[state_mask], 'rv_20'].mean()
            train_state_vols.append((state, avg_vol))
    
    if len(train_state_vols) == 2:
        train_state_vols.sort(key=lambda x: x[1])
        calm_state = train_state_vols[0][0]
        turbulent_state = train_state_vols[1][0]
        hmm_today_regime = 0 if np.argmax(hmm_today_probs) == calm_state else 1
        hmm_prob_calm = hmm_today_probs[calm_state]
        hmm_prob_turbulent = hmm_today_probs[turbulent_state]
    else:
        hmm_today_regime = 0 if np.argmax(hmm_today_probs) == 0 else 1
        hmm_prob_calm = hmm_today_probs[0]
        hmm_prob_turbulent = hmm_today_probs[1]
    
    # Prepare RF features
    rf_features_train = train_data[feature_names].copy()
    hmm_train_probs = hmm_model.predict_proba(X_train)
    if len(train_state_vols) == 2:
        rf_features_train['hmm_prob_calm'] = hmm_train_probs[:, calm_state]
        rf_features_train['hmm_prob_turbulent'] = hmm_train_probs[:, turbulent_state]
        rf_target_train = np.where(hmm_train_states == calm_state, 0, 1)
    else:
        rf_features_train['hmm_prob_calm'] = hmm_train_probs[:, 0]
        rf_features_train['hmm_prob_turbulent'] = hmm_train_probs[:, 1]
        rf_target_train = np.where(hmm_train_states == 0, 0, 1)
    
    # Train RF
    print("Training Random Forest...")
    rf_feature_names = feature_names + ['hmm_prob_calm', 'hmm_prob_turbulent']
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(rf_features_train[rf_feature_names].values, rf_target_train)
    
    # Predict for today
    rf_features_today = today_features[feature_names].copy()
    rf_features_today['hmm_prob_calm'] = hmm_prob_calm
    rf_features_today['hmm_prob_turbulent'] = hmm_prob_turbulent
    X_rf_today = rf_features_today[rf_feature_names].values
    
    rf_today_regime = rf_model.predict(X_rf_today)[0]
    rf_today_probs = rf_model.predict_proba(X_rf_today)[0]
    regime_confidence = rf_today_probs[rf_today_regime]
    
    # Apply circuit breaker if enabled
    final_regime = rf_today_regime
    circuit_breaker_triggered = False
    
    if CIRCUIT_BREAKER_ENABLED:
        # Circuit breaker: if the day before the prediction date had a large drop,
        # force the prediction date to be turbulent
        # Note: today_features['daily_return'] is already shifted by 1 day
        # So if prediction_date is T, daily_return is the return from T-2 to T-1
        # This is the return from the day before the prediction date, which is what we want to check
        prediction_date = today_features.index[0]
        day_before_return = today_features['daily_return'].iloc[0]
        if day_before_return < CIRCUIT_BREAKER_THRESHOLD:
            final_regime = 1  # Force turbulent
            circuit_breaker_triggered = True
    
    # Determine holding with risk management filters
    current_vol = today_features['rv_20'].iloc[0] if 'rv_20' in today_features.columns else None
    current_vix = today_features['vix'].iloc[0] if 'vix' in today_features.columns else None
    risk_filters_triggered = []
    
    if final_regime == 0:  # Predicted Calm -> potentially TQQQ
        # Apply risk management filters
        use_leverage = True
        
        if MIN_REGIME_CONFIDENCE and regime_confidence < MIN_REGIME_CONFIDENCE:
            use_leverage = False
            risk_filters_triggered.append(f"Low confidence ({regime_confidence:.1%} < {MIN_REGIME_CONFIDENCE:.0%})")
        
        if MAX_VOLATILITY_FOR_LEVERAGE and current_vol and current_vol > MAX_VOLATILITY_FOR_LEVERAGE:
            use_leverage = False
            risk_filters_triggered.append(f"High volatility ({current_vol:.1%} > {MAX_VOLATILITY_FOR_LEVERAGE:.0%})")
        
        if MIN_VIX_FOR_LEVERAGE and current_vix and current_vix > MIN_VIX_FOR_LEVERAGE:
            use_leverage = False
            risk_filters_triggered.append(f"High VIX ({current_vix:.1f} > {MIN_VIX_FOR_LEVERAGE:.1f})")
        
        if use_leverage:
            holding = LEVERAGED_SYMBOL  # TQQQ
            regime_name = "Calm"
        else:
            holding = SYMBOL  # QQQ (risk filter triggered)
            regime_name = "Calm (Risk Filter)"
    else:  # Turbulent
        holding = SYMBOL  # QQQ
        regime_name = "Turbulent"
    
    # Get current prices
    latest_date = today_features.index[0]
    qqq_price = qqq_data['close'].iloc[-1]
    
    # Get TQQQ price
    try:
        tqqq_data = yf.download(LEVERAGED_SYMBOL, period="5d", progress=False, auto_adjust=True)
        if isinstance(tqqq_data.columns, pd.MultiIndex):
            tqqq_data.columns = tqqq_data.columns.droplevel(1)
        tqqq_data.columns = tqqq_data.columns.str.lower()
        tqqq_price = tqqq_data['close'].iloc[-1]
    except:
        tqqq_price = None
    
    return {
        'date': latest_date,
        'regime': regime_name,
        'regime_code': final_regime,
        'holding': holding,
        'confidence': regime_confidence,
        'qqq_price': qqq_price,
        'tqqq_price': tqqq_price,
        'circuit_breaker': circuit_breaker_triggered,
        'vix': current_vix,
        'volatility_20d': current_vol,
        'risk_filters_triggered': risk_filters_triggered,
    }

def main():
    try:
        print("="*60)
        print("HMM-RF Regime Alert Service")
        print("="*60)
        
        previous_state = load_state()
        prediction = get_regime_prediction()
        
        # Format date - prediction['date'] is the date we're predicting the regime for
        # The features used for this prediction are from the previous trading day (due to .shift(1))
        pred_date = prediction['date']
        if hasattr(pred_date, 'date'):
            prediction_date_str = pred_date.date().strftime('%Y-%m-%d')
            # The features date is the previous trading day (data used for prediction)
            # Since features are shifted, we need to get the previous trading day
            # For now, we'll use the prediction date as the reference
            features_date_str = prediction_date_str  # This represents the date whose data was used
        else:
            prediction_date_str = str(pred_date)
            features_date_str = prediction_date_str
        
        # The prediction is for the date in prediction['date']
        # If this is a past date, we might want to predict for the next trading day
        # But for now, we'll use the prediction date as-is
        today_str = prediction_date_str
        
        # Check if regime changed
        prev_regime = previous_state.get('regime', None)
        prev_holding = previous_state.get('holding', None)
        regime_changed = prev_regime != prediction['regime']
        holding_changed = prev_holding != prediction['holding']
        
        # Build message
        if regime_changed or holding_changed:
            message = "🚨 REGIME CHANGE ALERT\n\n"
            if prev_regime:
                message += f"Previous: {prev_regime} → {prediction['regime']}\n"
                message += f"Holding: {prev_holding} → {prediction['holding']}\n\n"
        else:
            message = "📊 Daily Regime Update\n\n"
        
        message += f"📅 Prediction Date: {today_str}\n"
        message += f"📊 Features Date: {features_date_str}\n"
        message += f"🎯 Regime: {prediction['regime']}\n"
        message += f"💼 Recommended Holding: {prediction['holding']}\n\n"
        
        message += f"Market Data (as of close of previous day):\n"
        message += f"  {SYMBOL} Price: ${prediction['qqq_price']:.2f}\n"
        if prediction['tqqq_price']:
            message += f"  {LEVERAGED_SYMBOL} Price: ${prediction['tqqq_price']:.2f}\n"
        if prediction['vix']:
            message += f"  VIX: {prediction['vix']:.2f}\n"
        if prediction['volatility_20d']:
            message += f"  20D Volatility: {prediction['volatility_20d']:.2%}\n"
        
        if prediction['circuit_breaker']:
            message += f"\n⚠️ Circuit Breaker Triggered (>{abs(CIRCUIT_BREAKER_THRESHOLD)*100:.0f}% drop)\n"
        
        if prediction.get('risk_filters_triggered'):
            message += f"\n⚠️ Risk Filters Triggered:\n"
            for filter_msg in prediction['risk_filters_triggered']:
                message += f"  • {filter_msg}\n"
            message += f"  → Holding {SYMBOL} instead of {LEVERAGED_SYMBOL} for safety\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}"
        
        # Send alert
        if send_telegram(message):
            save_state({
                'regime': prediction['regime'],
                'holding': prediction['holding'],
                'date': today_str,
                'features_date': features_date_str,
                'confidence': prediction['confidence']
            })
            if regime_changed or holding_changed:
                print(f"✅ Successfully sent regime change alert")
            else:
                print(f"✅ Successfully sent daily regime update")
        else:
            print("❌ Failed to send Telegram alert")
            
    except Exception as e:
        error_msg = f"❌ Error in regime alert service: {str(e)}\n\n"
        error_msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}"
        send_telegram(error_msg)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

