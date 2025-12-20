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
FEATURE_SET_CONFIG = 'extended'  # 'base' or 'extended'
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
    """Calculate all features for the HMM-RF strategy"""
    features_df = pd.DataFrame(index=qqq_data.index)
    
    # Base features (all properly lagged by 1 day)
    features_df['rt'] = np.log(qqq_data['close'] / qqq_data['close'].shift(1)).shift(1)
    features_df['daily_return'] = qqq_data['close'].pct_change().shift(1)
    features_df['rv_20'] = features_df['rt'].rolling(window=20).std() * np.sqrt(252)
    features_df['rv_5'] = features_df['rt'].rolling(window=5).std() * np.sqrt(252)
    features_df['mom_21'] = qqq_data['close'].pct_change(21).shift(1)
    features_df['vix'] = vix_data['close'].shift(1)
    features_df['rates'] = rates_data['close'].shift(1)
    
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

def get_regime_prediction():
    """Get current regime prediction using HMM-RF model"""
    
    # Load data - use enough history for feature calculation
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    end_date = None  # Get latest data
    
    print("Loading market data...")
    qqq_data = yf.download(SYMBOL, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if isinstance(qqq_data.columns, pd.MultiIndex):
        qqq_data.columns = qqq_data.columns.droplevel(1)
    qqq_data.columns = qqq_data.columns.str.lower()
    
    vix_data = yf.download(VIX_SYMBOL, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.droplevel(1)
    vix_data.columns = vix_data.columns.str.lower()
    
    rates_data = yf.download(RATES_SYMBOL, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if isinstance(rates_data.columns, pd.MultiIndex):
        rates_data.columns = rates_data.columns.droplevel(1)
    rates_data.columns = rates_data.columns.str.lower()
    
    # Align data
    aligned_index = qqq_data.index
    vix_aligned = vix_data['close'].reindex(aligned_index).ffill().bfill()
    rates_aligned = rates_data['close'].reindex(aligned_index).ffill().bfill()
    
    # Calculate features
    print("Calculating features...")
    features_df = calculate_features(qqq_data, vix_aligned, rates_aligned)
    features_df = features_df.dropna()
    
    if len(features_df) < TRAIN_WINDOW + 1:
        raise ValueError(f"Not enough data. Need at least {TRAIN_WINDOW + 1} days, got {len(features_df)}")
    
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
    
    # Use last TRAIN_WINDOW days for training, predict for today (last row)
    # Note: features_df[-1] contains features calculated from yesterday's data
    # These features are used to predict today's regime (properly lagged)
    train_data = features_df.iloc[-TRAIN_WINDOW-1:-1]  # Last TRAIN_WINDOW days (excluding last row)
    today_features = features_df.iloc[-1:]  # Last row: features from yesterday, used to predict today
    
    # The date we're predicting for is actually "today" (next trading day after last data point)
    prediction_date = today_features.index[0]
    print(f"Training on {len(train_data)} days")
    print(f"Features date: {prediction_date.date()} (yesterday's data)")
    print(f"Predicting regime for: Next trading day")
    
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
        # Check yesterday's return
        yesterday_return = today_features['daily_return'].iloc[0]
        if yesterday_return < CIRCUIT_BREAKER_THRESHOLD:
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
        
        # Format date - this is the date of the features (yesterday), but we're predicting for today
        pred_date = prediction['date']
        if hasattr(pred_date, 'date'):
            features_date_str = pred_date.date().strftime('%Y-%m-%d')
            # Today is the next trading day
            today_str = datetime.now().strftime('%Y-%m-%d')
        else:
            features_date_str = str(pred_date)
            today_str = datetime.now().strftime('%Y-%m-%d')
        
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
        message += f"💼 Recommended Holding: {prediction['holding']}\n"
        message += f"📈 Confidence: {prediction['confidence']:.1%}\n\n"
        
        message += f"Market Data:\n"
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

