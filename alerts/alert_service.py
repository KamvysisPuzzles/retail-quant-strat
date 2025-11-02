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
    
    if not token or not chat_id:
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    
    try:
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except:
        return False

def get_current_signals():
    start_date = (datetime.now() - timedelta(days=250)).strftime('%Y-%m-%d')
    
    combiner = StrategyCombiner(
        strategies=STRATEGIES_CONFIG,
        initial_capital=100_000.0,
        fees=0.0005,
        slippage=0.0005,
        start_date=start_date,
        end_date=None,
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
        current_position = int(exposure.iloc[-1]) if len(exposure) > 0 else 0
        current_state[leg_id] = current_position
        
        signal_type = "BUY" if current_position == 1 else "SELL"
        signals.append({
            'leg_id': leg_id,
            'symbol': leg.symbol,
            'strategy': leg.strategy_name,
            'position': current_position,
            'signal': signal_type,
            'price': float(leg.close.iloc[-1]) if len(leg.close) > 0 else 0.0
        })
    
    return current_state, signals

def main():
    try:
        previous_state = load_state()
        current_state, signals = get_current_signals()
        
        changed_signals = []
        for sig in signals:
            leg_id = sig['leg_id']
            prev_pos = previous_state.get(leg_id, -1)
            curr_pos = sig['position']
            
            if prev_pos != curr_pos and prev_pos != -1:
                changed_signals.append(sig)
        
        if changed_signals:
            message = "🚨 Signal Alert\n\n"
            for sig in changed_signals:
                message += f"{sig['symbol']} ({sig['strategy']}): {sig['signal']}\n"
                message += f"Price: ${sig['price']:.2f}\n\n"
            message += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}"
            
            if send_telegram(message):
                save_state(current_state)
            else:
                print("Failed to send Telegram alert")
        else:
            save_state(current_state)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

