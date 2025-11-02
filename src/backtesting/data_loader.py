"""Data loading utilities for backtesting."""
import pandas as pd
import yfinance as yf
from typing import Dict, Optional


def load_data(
    symbols: list, 
    start_date: str, 
    end_date: Optional[str] = None,
    interval: str = '1d',
    align_to_symbol: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load stock data for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (None for latest)
        interval: Data interval ('1d' for daily)
        align_to_symbol: Optional symbol to align all other symbols to (useful for aligning
                        24/7 crypto markets to traditional market trading days). If specified,
                        all symbols' data will be reindexed to match this symbol's trading days.
    
    Returns:
        Dictionary mapping symbol to DataFrame with columns: open, high, low, close, volume
    """
    data_dict = {}
    
    for symbol in symbols:
        try:
            stock_data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            if stock_data.empty:
                print(f"Warning: No data downloaded for {symbol}")
                continue
            
            # Normalize column names (handle MultiIndex or single index)
            if isinstance(stock_data.columns, pd.MultiIndex):
                # Extract the first level or use Close/Open/etc directly
                stock_data.columns = stock_data.columns.droplevel(1)
            
            # Ensure required columns exist (case-insensitive)
            required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            available_cols = [col for col in stock_data.columns if col in required_cols]
            
            if 'Close' not in available_cols:
                print(f"Warning: Close price not found for {symbol}")
                continue
            
            # Normalize to lowercase column names
            stock_data.columns = stock_data.columns.str.lower()
            
            # Ensure all required columns exist, fill missing ones
            if 'open' not in stock_data.columns:
                stock_data['open'] = stock_data['close']
            if 'high' not in stock_data.columns:
                stock_data['high'] = stock_data['close']
            if 'low' not in stock_data.columns:
                stock_data['low'] = stock_data['close']
            if 'volume' not in stock_data.columns:
                stock_data['volume'] = 0.0
            
            # Select only required columns
            stock_data = stock_data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Remove any rows with NaN in close price
            stock_data = stock_data.dropna(subset=['close'])
            
            data_dict[symbol] = stock_data
            
            print(f"Loaded {len(stock_data)} records for {symbol} "
                  f"({stock_data.index[0].date()} to {stock_data.index[-1].date()})")
            
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            continue
    
    # Align all symbols to a reference symbol's trading days if specified
    if align_to_symbol and align_to_symbol in data_dict:
        reference_index = data_dict[align_to_symbol].index
        print(f"\nAligning all data to {align_to_symbol} trading days ({len(reference_index)} days)...")
        
        for symbol, df in data_dict.items():
            if symbol == align_to_symbol:
                continue  # Skip the reference symbol itself
            
            original_len = len(df)
            # Reindex to reference index, forward-filling price data
            # This ensures 24/7 markets (like crypto) are aligned to market trading days
            df_aligned = df.reindex(reference_index)
            
            # Forward fill missing values (use last known price for weekends/holidays)
            df_aligned = df_aligned.ffill()
            
            # Backfill any remaining NaN values at the start (use first available price)
            df_aligned = df_aligned.bfill()
            
            # Remove any remaining NaN rows (shouldn't happen after ffill/bfill, but just in case)
            df_aligned = df_aligned.dropna(subset=['close'])
            
            data_dict[symbol] = df_aligned
            print(f"  {symbol}: {original_len} records → {len(df_aligned)} records "
                  f"(aligned to {align_to_symbol})")
    
    return data_dict


def select_close_series(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.Series:
    """
    Extract close price series from downloaded data.
    Handles both MultiIndex and single-level column structures.
    
    Args:
        df: DataFrame from yfinance download
        ticker: Optional ticker symbol (for MultiIndex columns)
    
    Returns:
        Close price Series
    """
    if isinstance(df.columns, pd.MultiIndex):
        if ticker and ('Close', ticker) in df.columns:
            s = df[('Close', ticker)]
        else:
            cols = [c for c in df.columns if 'Close' in str(c)]
            if not cols:
                raise KeyError("Close not found in DataFrame")
            s = df[cols[0]]
    else:
        if 'close' in df.columns:
            s = df['close']
        elif 'Close' in df.columns:
            s = df['Close']
        else:
            raise KeyError("Close not found in DataFrame")
    
    return s.astype(float).squeeze()

