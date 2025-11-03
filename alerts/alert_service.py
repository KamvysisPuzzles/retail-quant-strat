import os
import sys
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path so we can import src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backtesting.strategy_combiner import StrategyCombiner, StrategyConfig

STATE_FILE = "signal_state.json" 

STRATEGIES_CONFIG = [
    StrategyConfig(
        id="tqqq_ensemble",
        symbol="TQQQ",
        strategy_name="tqqq_3ema_macdv_aroon",
        params={
            "ema1": 12,
            "ema2": 89,
            "ema3": 125,
            "macd_fast": 25,
            "macd_slow": 30,
            "macd_signal": 85,
            "volume_threshold": 0.0,
            "aroon_length": 66,
        },
        weight=0.5,
    ),
    StrategyConfig(
        id="btc_ensemble",
        symbol="BTC-USD",
        strategy_name="3ema_macdv",
        params={
            "ema1": 10,
            "ema2": 64,
            "ema3": 126,
            "macd_fast": 32,
            "macd_slow": 40,
            "macd_signal": 94,
        },
        weight=0.5,
    ),
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def send_telegram(message):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable is not set")
        return False
    
    if not chat_id:
        print("Error: TELEGRAM_CHAT_ID environment variable is not set")
        return False
    
    # Clean up token (remove any whitespace that might have been accidentally added)
    token = token.strip()
    
    # Validate token format (Telegram bot tokens have format: numbers:letters_with_underscores)
    if ':' not in token or not token.split(':', 1)[0].isdigit():
        print("Warning: TELEGRAM_BOT_TOKEN format appears invalid. Expected format: '123456789:ABCdef...'")
        print(f"Token starts with: {token[:10]}... (first 10 chars)")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    
    try:
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            return True
        else:
            print(f"Error: Telegram API returned status code {response.status_code}")
            try:
                error_data = response.json()
                print(f"Telegram API error: {error_data}")
            except:
                print(f"Telegram API response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error sending Telegram message: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def get_current_signals():
    # Use enough history for indicator warm-up (max period is EMA 125 = ~125 days)
    # Adding buffer for safety - use ~200 days to ensure stable indicators
    start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
    
    # Use previous day's date as end_date to ensure we only use complete daily candles
    # This avoids issues with partial/incomplete data for the current day
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    combiner = StrategyCombiner(
        strategies=STRATEGIES_CONFIG,
        initial_capital=100_000.0,
        fees=0.0005,
        slippage=0.0005,
        start_date=start_date,
        end_date=end_date,
        align_to_symbol='TQQQ',
        rebalance_freq='Q',
        safe_asset='GLD'
    )
    
    combiner.run()
    legs = combiner.get_leg_results()
    
    current_state = {}
    signals = []
    
    for leg_id, leg in legs.items():
        exposure = leg.exposure
        if len(exposure) == 0:
            print(f"Warning: No exposure data for {leg_id}")
            continue
            
        current_position = int(exposure.iloc[-1])
        current_state[leg_id] = current_position
        
        # Get the date of the latest signal for debugging
        latest_date = exposure.index[-1] if hasattr(exposure.index[-1], 'strftime') else str(exposure.index[-1])
        
        signal_type = "BUY" if current_position == 1 else "SELL"
        signals.append({
            'leg_id': leg_id,
            'symbol': leg.symbol,
            'strategy': leg.strategy_name,
            'position': current_position,
            'signal': signal_type,
            'price': float(leg.close.iloc[-1]) if len(leg.close) > 0 else 0.0,
            'date': latest_date
        })
        
        # Debug output
        print(f"{leg_id}: {signal_type} signal (position={current_position}) on {latest_date}")
    
    return current_state, signals

def main():
    try:
        previous_state = load_state()
        current_state, signals = get_current_signals()
        
        # Identify which signals have flipped
        changed_signals = []
        for sig in signals:
            leg_id = sig['leg_id']
            prev_pos = previous_state.get(leg_id, -1)
            curr_pos = sig['position']
            
            if prev_pos != curr_pos and prev_pos != -1:
                changed_signals.append(sig)
        
        # Always send a message - use different header if signals flipped
        has_changes = len(changed_signals) > 0
        
        if has_changes:
            # Signal flipped - use alert header
            message = "🚨 Signal Alert - Signals Changed\n\n"
            message += "Changed Signals:\n"
            for sig in changed_signals:
                prev_pos = previous_state.get(sig['leg_id'], -1)
                prev_signal = "BUY" if prev_pos == 1 else "SELL"
                date_str = sig.get('date', 'N/A')
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                message += f"{sig['symbol']} ({sig['strategy']}): {prev_signal} → {sig['signal']}\n"
                message += f"Price: ${sig['price']:.2f} (as of {date_str})\n\n"
            
            if len(changed_signals) < len(signals):
                message += "All Current Signals:\n"
                for sig in signals:
                    date_str = sig.get('date', 'N/A')
                    if hasattr(date_str, 'strftime'):
                        date_str = date_str.strftime('%Y-%m-%d')
                    message += f"{sig['symbol']} ({sig['strategy']}): {sig['signal']}\n"
                    message += f"Price: ${sig['price']:.2f} (as of {date_str})\n\n"
        else:
            # No changes - daily status update
            message = "📊 Daily Signal Update\n\n"
            message += "Current Positions:\n\n"
            for sig in signals:
                date_str = sig.get('date', 'N/A')
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                message += f"{sig['symbol']} ({sig['strategy']}): {sig['signal']}\n"
                message += f"Price: ${sig['price']:.2f} (as of {date_str})\n\n"
        
        message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}"
        
        if send_telegram(message):
            save_state(current_state)
            if has_changes:
                print(f"Successfully sent alert with {len(changed_signals)} changed signal(s)")
            else:
                print(f"Successfully sent daily update with {len(signals)} signal(s)")
        else:
            print("Failed to send Telegram alert")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

