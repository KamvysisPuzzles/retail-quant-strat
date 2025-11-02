"""Performance analysis and visualization utilities."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from src.backtesting.legs import LegResult
from src.backtesting.portfolio_combo import CombinedPortfolio
import vectorbt as vbt


class PerformanceAnalyzer:
    """Analyzes and visualizes performance of combined portfolios."""
    
    def __init__(self, combined_portfolio: CombinedPortfolio):
        """
        Initialize analyzer.
        
        Args:
            combined_portfolio: CombinedPortfolio object to analyze
        """
        self.portfolio = combined_portfolio
        self.returns = combined_portfolio.returns
        self.equity = combined_portfolio.equity
    
    def calculate_metrics(
        self, 
        risk_free_rate: float = 0.0,
        freq_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            risk_free_rate: Risk-free rate (annual)
            freq_per_year: Trading days per year
        
        Returns:
            Dictionary of performance metrics
        """
        ret = self.returns.fillna(0.0)
        
        # Basic return metrics
        total_return = (1.0 + ret).prod() - 1.0
        years = len(ret) / freq_per_year
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        
        # Volatility
        volatility = ret.std() * np.sqrt(freq_per_year)
        
        # Sharpe ratio - use vectorbt's method if available (for single-leg portfolios)
        # Otherwise calculate manually using standard formula
        sharpe = np.nan
        if hasattr(self.portfolio, 'legs') and len(self.portfolio.legs) == 1:
            # Single leg - try to use vectorbt's sharpe_ratio method
            leg = list(self.portfolio.legs.values())[0]
            if leg.portfolio is not None and hasattr(leg.portfolio, 'sharpe_ratio'):
                try:
                    sharpe = float(leg.portfolio.sharpe_ratio(freq='D'))
                except:
                    pass
        
        # If still NaN, calculate manually
        if np.isnan(sharpe):
            # Sharpe = (annualized excess return) / (annualized volatility)
            # Standard formula: Sharpe = (mean(excess) / std) * sqrt(periods_per_year)
            excess_returns = ret - (risk_free_rate / freq_per_year)
            if ret.std() > 0:
                sharpe = (excess_returns.mean() / ret.std()) * np.sqrt(freq_per_year)
            else:
                sharpe = np.nan
        
        # Sortino ratio - use vectorbt's method if available (for single-leg portfolios)
        # Otherwise calculate manually
        sortino = np.nan
        if hasattr(self.portfolio, 'legs') and len(self.portfolio.legs) == 1:
            # Single leg - try to use vectorbt's sortino_ratio method
            leg = list(self.portfolio.legs.values())[0]
            if leg.portfolio is not None and hasattr(leg.portfolio, 'sortino_ratio'):
                try:
                    sortino = float(leg.portfolio.sortino_ratio(freq='D'))
                except:
                    pass
        
        # If still NaN, calculate manually
        if np.isnan(sortino):
            # Sortino = (annualized excess return) / (annualized downside deviation)
            excess_returns = ret - (risk_free_rate / freq_per_year)
            downside_excess_returns = excess_returns[excess_returns < 0]
            if len(downside_excess_returns) > 0 and downside_excess_returns.std() > 0:
                # Standard formula: Sortino = (mean(excess) / downside_std) * sqrt(periods_per_year)
                mean_excess = excess_returns.mean()
                sortino = (mean_excess / downside_excess_returns.std()) * np.sqrt(freq_per_year)
            else:
                # No negative returns or no downside deviation
                mean_excess = excess_returns.mean()
                sortino = np.nan if mean_excess <= 0 else np.inf
        
        # Drawdown
        roll_max = (1.0 + ret).cumprod().cummax()
        underwater = (1.0 + ret).cumprod() / roll_max - 1.0
        max_drawdown = underwater.min()
        
        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Trade metrics calculation - match notebook exactly
        # Win rate = percentage of winning trades
        # Profit Factor = sum of wins / abs(sum of losses)
        win_rate = np.nan
        profit_factor = np.nan
        total_trades = 0
        
        # Try to get trade metrics from individual leg trades (matches notebook logic)
        if hasattr(self.portfolio, 'legs'):
            total_winners = 0
            total_gains = 0.0
            total_losses = 0.0
            
            for leg in self.portfolio.legs.values():
                if leg.portfolio is not None and hasattr(leg.portfolio, 'trades'):
                    try:
                        trades = leg.portfolio.trades
                        if len(trades) > 0:
                            # Get trade returns - matches notebook exactly
                            # Notebook: tr = trades.returns.values if hasattr(trades.returns, 'values') else np.array(trades.returns)
                            if hasattr(trades, 'returns'):
                                tr_raw = trades.returns.values if hasattr(trades.returns, 'values') else np.array(trades.returns)
                                # Notebook uses len(tr) for counting, so tr should be 1D
                                # Convert to 1D array (ravel handles both 1D and 2D correctly)
                                tr = np.asarray(tr_raw).ravel()
                                if len(tr) > 0:  # Use len() like notebook does
                                    # Count positive returns (winning trades) - matches notebook exactly
                                    pos = tr[tr > 0]
                                    neg = tr[tr < 0]
                                    total_winners += len(pos)
                                    total_trades += len(tr)
                                    # Calculate gains and losses for profit factor
                                    gains = pos.sum() if len(pos) > 0 else 0.0
                                    losses = abs(neg.sum()) if len(neg) > 0 else 0.0
                                    total_gains += gains
                                    total_losses += losses
                    except Exception as e:
                        # Debug: print error if needed
                        # print(f"Error calculating trade metrics: {e}")
                        pass
            
            if total_trades > 0:
                win_rate = (total_winners / total_trades) * 100.0
            
            if total_losses > 0:
                profit_factor = total_gains / total_losses
            elif total_gains > 0:
                profit_factor = np.inf
            else:
                profit_factor = np.nan
        
        # If win_rate still NaN (no trades available), set to 0
        if np.isnan(win_rate):
            win_rate = 0.0
        
        # Trades per year
        trades_per_year = total_trades / years if years > 0 else 0.0
        
        # Best and worst periods
        best_day = ret.max()
        worst_day = ret.min()
        
        # Monthly returns analysis (ME = Month End for pandas 2.0+)
        try:
            monthly_returns = self.equity.resample('ME').last().pct_change().dropna()
        except ValueError:
            # Fallback for older pandas versions
            monthly_returns = self.equity.resample('M').last().pct_change().dropna()
        best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0.0
        worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0.0
        avg_monthly = monthly_returns.mean() if len(monthly_returns) > 0 else 0.0
        positive_months = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0.0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(cagr),  # Also include as annualized_return for consistency
            'cagr': float(cagr),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'profit_factor': float(profit_factor) if not np.isinf(profit_factor) and not np.isnan(profit_factor) else np.nan,
            'trades_per_year': float(trades_per_year),
            'best_day': float(best_day),
            'worst_day': float(worst_day),
            'best_month': float(best_month),
            'worst_month': float(worst_month),
            'avg_monthly_return': float(avg_monthly),
            'positive_months_pct': float(positive_months),
            'total_days': len(ret),
            'years': years
        }
    
    def plot_equity_curve(self, benchmark: Optional[pd.Series] = None, figsize=(14, 8)):
        """Plot equity curve with optional benchmark."""
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        
        # Equity curve
        axes[0].plot(self.equity.index, self.equity.values, label='Combined Portfolio', 
                    linewidth=2, color='darkgreen')
        axes[0].axhline(y=self.portfolio.initial_capital, color='gray', 
                       linestyle='--', linewidth=1, alpha=0.5, label='Initial Capital')
        
        if benchmark is not None:
            # Normalize benchmark to same starting value
            benchmark_normalized = benchmark / benchmark.iloc[0] * self.portfolio.initial_capital
            axes[0].plot(benchmark.index, benchmark_normalized.values, 
                        label='Benchmark', linewidth=1.5, color='blue', alpha=0.7)
        
        axes[0].set_ylabel('Portfolio Value ($)', fontweight='bold')
        axes[0].set_title('Combined Portfolio Equity Curve', fontweight='bold', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        returns = self.returns.fillna(0.0)
        roll_max = (1.0 + returns).cumprod().cummax()
        drawdown = (1.0 + returns).cumprod() / roll_max - 1.0
        
        axes[1].fill_between(drawdown.index, drawdown.values, 0, 
                            color='red', alpha=0.3, label='Drawdown')
        axes[1].plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        axes[1].set_ylabel('Drawdown (%)', fontweight='bold')
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].set_title('Drawdown Analysis', fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_leg_comparison(self, figsize=(14, 10)):
        """Plot comparison of individual strategy legs."""
        if not self.portfolio.legs:
            print("No legs to compare")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        
        # Normalize all equity curves to same starting value
        for leg_id, leg in self.portfolio.legs.items():
            weight = self.portfolio.weights.get(leg_id, 0.0)
            if weight > 0:
                leg_normalized = leg.equity_curve / leg.equity_curve.iloc[0] * self.portfolio.initial_capital
                axes[0].plot(leg.equity_curve.index, leg_normalized.values, 
                           label=f'{leg_id} (weight: {weight:.1%})', linewidth=1.5, alpha=0.7)
        
        # Combined portfolio
        axes[0].plot(self.equity.index, self.equity.values, 
                    label='Combined Portfolio', linewidth=2.5, color='black')
        axes[0].set_ylabel('Portfolio Value ($)', fontweight='bold')
        axes[0].set_title('Individual Strategy vs Combined Portfolio', fontweight='bold', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Weights over time (currently static, but shown for consistency)
        for leg_id, weight in self.portfolio.weights.items():
            if weight > 0:
                axes[1].axhline(y=weight, label=f'{leg_id} weight', linewidth=1.5, alpha=0.7)
        
        axes[1].set_ylabel('Allocation Weight', fontweight='bold')
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].set_title('Strategy Allocation Weights', fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
    
    def print_metrics_table(self, risk_free_rate: float = 0.0):
        """Print formatted metrics table matching notebook format."""
        metrics = self.calculate_metrics(risk_free_rate)
        
        print("\n📊 Performance Metrics:")
        print(f"  Total Return:      {metrics['total_return']*100:>10.2f}%")
        print(f"  Annualized Return: {metrics['annualized_return']*100:>10.2f}%")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>10.3f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>10.3f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']*100:>10.2f}%")
        print(f"  Volatility:        {metrics['volatility']*100:>10.2f}%")
        print(f"  Total Trades:     {metrics['total_trades']:>10}")
        print(f"  Win Rate:         {metrics['win_rate']:>10.1f}%")
        pf = metrics.get('profit_factor', np.nan)
        if not np.isnan(pf) and not np.isinf(pf):
            print(f"  Profit Factor:     {pf:>10.2f}")
        elif np.isinf(pf):
            print(f"  Profit Factor:        Inf")
        else:
            print(f"  Profit Factor:          N/A")
        print(f"  Trades per Year:   {metrics['trades_per_year']:>10.1f}")
        print("=" * 70)
    
    def print_leg_metrics(self):
        """Print metrics for individual strategy legs."""
        if not self.portfolio.legs:
            print("No legs to analyze")
            return
        
        print("\n" + "="*70)
        print("INDIVIDUAL STRATEGY PERFORMANCE")
        print("="*70)
        
        for leg_id, leg in self.portfolio.legs.items():
            weight = self.portfolio.weights.get(leg_id, 0.0)
            
            # Calculate basic metrics for this leg
            ret = leg.returns.fillna(0.0)
            total_return = (1.0 + ret).prod() - 1.0
            sharpe = leg.stats.get('sharpe', np.nan)
            max_dd = leg.stats.get('max_drawdown', np.nan)
            
            print(f"\n{leg_id} (Weight: {weight:.1%}):")
            print(f"  Symbol:              {leg.symbol}")
            print(f"  Strategy:            {leg.strategy_name}")
            print(f"  Total Return:        {total_return:>10.2%}")
            print(f"  Sharpe Ratio:        {sharpe:>10.2f}")
            print(f"  Max Drawdown:        {max_dd:>10.2%}")
        
        print("="*70)
    
    def plot_trade_entries_and_moving_sharpe(self, prices: Optional[pd.Series] = None, 
                                            window: int = 63, figsize=(16, 12)):
        """
        Plot price chart with trade entry/exit markers and moving Sharpe ratio.
        
        Args:
            prices: Price series to plot (if None, uses first leg's price data)
            window: Rolling window for moving Sharpe calculation (default 63 days = ~3 months)
            figsize: Figure size
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 1, 1], sharex=True)
        
        # Get price data if not provided
        if prices is None and hasattr(self.portfolio, 'legs') and len(self.portfolio.legs) > 0:
            # Use first leg's price data
            first_leg = list(self.portfolio.legs.values())[0]
            if hasattr(first_leg, 'portfolio') and first_leg.portfolio is not None:
                try:
                    prices = first_leg.portfolio.close
                except:
                    pass
        
        # Plot 1: Price with trade entries/exits
        if prices is not None:
            axes[0].plot(prices.index, prices.values, label='Price', linewidth=1.5, color='black', alpha=0.7)
            axes[0].set_ylabel('Price ($)', fontweight='bold')
            axes[0].set_title('Price Chart with Trade Entries/Exits', fontweight='bold', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            
            # Get entry and exit signals from all legs
            for leg in self.portfolio.legs.values():
                if hasattr(leg, 'portfolio') and leg.portfolio is not None:
                    try:
                        entries = leg.portfolio.orders.records_readable
                        if len(entries) > 0:
                            # Get entry signals (buy orders)
                            buy_orders = entries[entries['Side'] == 'Buy']
                            if len(buy_orders) > 0:
                                entry_dates = pd.to_datetime(buy_orders['Timestamp'])
                                entry_prices = buy_orders['Price']
                                axes[0].scatter(entry_dates, entry_prices, 
                                              marker='^', color='green', s=100, 
                                              alpha=0.7, label='Entry', zorder=5)
                            
                            # Get exit signals (sell orders)
                            sell_orders = entries[entries['Side'] == 'Sell']
                            if len(sell_orders) > 0:
                                exit_dates = pd.to_datetime(sell_orders['Timestamp'])
                                exit_prices = sell_orders['Price']
                                axes[0].scatter(exit_dates, exit_prices, 
                                              marker='v', color='red', s=100, 
                                              alpha=0.7, label='Exit', zorder=5)
                    except:
                        pass
            
            axes[0].legend(loc='best')
        else:
            axes[0].text(0.5, 0.5, 'Price data not available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_ylabel('Price ($)', fontweight='bold')
        
        # Plot 2: Moving Sharpe Ratio
        ret = self.returns.fillna(0.0)
        risk_free_rate = 0.0  # Can be parameterized later
        
        # Calculate rolling Sharpe ratio
        rolling_window = window
        rolling_sharpe = pd.Series(index=ret.index, dtype=float)
        
        for i in range(rolling_window, len(ret)):
            window_returns = ret.iloc[i-rolling_window:i]
            if window_returns.std() > 0:
                excess_returns = window_returns - (risk_free_rate / 252)
                sharpe = (excess_returns.mean() / window_returns.std()) * np.sqrt(252)
                rolling_sharpe.iloc[i] = sharpe
        
        axes[1].plot(rolling_sharpe.index, rolling_sharpe.values, 
                    linewidth=1.5, color='blue', label=f'Rolling Sharpe ({window} days)')
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')
        axes[1].axhline(y=2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 2')
        axes[1].set_ylabel('Sharpe Ratio', fontweight='bold')
        axes[1].set_title('Rolling Sharpe Ratio', fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, 
                            where=(rolling_sharpe.values > 0), alpha=0.2, color='green')
        axes[1].fill_between(rolling_sharpe.index, rolling_sharpe.values, 0, 
                            where=(rolling_sharpe.values <= 0), alpha=0.2, color='red')
        
        # Plot 3: Equity curve
        axes[2].plot(self.equity.index, self.equity.values, 
                    linewidth=2, color='darkgreen', label='Portfolio Equity')
        axes[2].set_ylabel('Portfolio Value ($)', fontweight='bold')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].set_title('Portfolio Equity Curve', fontweight='bold')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_train_val_comparison(self, train_analyzer, val_analyzer, risk_free_rate: float = 0.0, full_analyzer=None):
        """
        Print side-by-side comparison of train, validation, and full set metrics.
        
        Args:
            train_analyzer: PerformanceAnalyzer for training set
            val_analyzer: PerformanceAnalyzer for validation set
            risk_free_rate: Risk-free rate
            full_analyzer: Optional PerformanceAnalyzer for full set (train + validation combined)
        """
        train_metrics = train_analyzer.calculate_metrics(risk_free_rate)
        val_metrics = val_analyzer.calculate_metrics(risk_free_rate)
        
        # Calculate full set metrics if provided
        full_metrics = None
        if full_analyzer is not None:
            full_metrics = full_analyzer.calculate_metrics(risk_free_rate)
        
        print("\n" + "="*110)
        if full_analyzer is not None:
            print("TRAIN vs VALIDATION vs FULL SET METRICS COMPARISON")
            print("="*110)
            print(f"{'METRIC':<25} {'TRAINING':<18} {'VALIDATION':<18} {'FULL SET':<18} {'CHANGE':<18}")
            print("-"*110)
        else:
            print("TRAIN vs VALIDATION METRICS COMPARISON")
            print("="*110)
            print(f"{'METRIC':<25} {'TRAINING':<20} {'VALIDATION':<20} {'CHANGE':<20}")
            print("-"*110)
        
        # Return metrics
        print(f"\n{'RETURN METRICS:':<25}")
        tr_train = train_metrics['total_return'] * 100
        tr_val = val_metrics['total_return'] * 100
        tr_chg = ((tr_val - tr_train) / abs(tr_train) * 100) if tr_train != 0 else np.nan
        if full_metrics is not None:
            tr_full = full_metrics['total_return'] * 100
            print(f"{'Total Return':<25} {tr_train:>16.2f}% {tr_val:>16.2f}% {tr_full:>16.2f}% {tr_chg:>16.1f}%" if not np.isnan(tr_chg) else f"{'Total Return':<25} {tr_train:>16.2f}% {tr_val:>16.2f}% {tr_full:>16.2f}% {'N/A':>18}")
        else:
            print(f"{'Total Return':<25} {tr_train:>18.2f}% {tr_val:>18.2f}% {tr_chg:>18.1f}%" if not np.isnan(tr_chg) else f"{'Total Return':<25} {tr_train:>18.2f}% {tr_val:>18.2f}% {'N/A':>20}")
        
        ar_train = train_metrics['annualized_return'] * 100
        ar_val = val_metrics['annualized_return'] * 100
        ar_chg = ((ar_val - ar_train) / abs(ar_train) * 100) if ar_train != 0 else np.nan
        if full_metrics is not None:
            ar_full = full_metrics['annualized_return'] * 100
            print(f"{'Annualized Return':<25} {ar_train:>16.2f}% {ar_val:>16.2f}% {ar_full:>16.2f}% {ar_chg:>16.1f}%" if not np.isnan(ar_chg) else f"{'Annualized Return':<25} {ar_train:>16.2f}% {ar_val:>16.2f}% {ar_full:>16.2f}% {'N/A':>18}")
        else:
            print(f"{'Annualized Return':<25} {ar_train:>18.2f}% {ar_val:>18.2f}% {ar_chg:>18.1f}%" if not np.isnan(ar_chg) else f"{'Annualized Return':<25} {ar_train:>18.2f}% {ar_val:>18.2f}% {'N/A':>20}")
        
        # Risk metrics
        print(f"\n{'RISK METRICS:':<25}")
        dd_train = train_metrics['max_drawdown'] * 100
        dd_val = val_metrics['max_drawdown'] * 100
        dd_chg = ((dd_val - dd_train) / abs(dd_train) * 100) if dd_train != 0 else np.nan
        if full_metrics is not None:
            dd_full = full_metrics['max_drawdown'] * 100
            print(f"{'Max Drawdown':<25} {dd_train:>16.2f}% {dd_val:>16.2f}% {dd_full:>16.2f}% {dd_chg:>16.1f}%" if not np.isnan(dd_chg) else f"{'Max Drawdown':<25} {dd_train:>16.2f}% {dd_val:>16.2f}% {dd_full:>16.2f}% {'N/A':>18}")
        else:
            print(f"{'Max Drawdown':<25} {dd_train:>18.2f}% {dd_val:>18.2f}% {dd_chg:>18.1f}%" if not np.isnan(dd_chg) else f"{'Max Drawdown':<25} {dd_train:>18.2f}% {dd_val:>18.2f}% {'N/A':>20}")
        
        vol_train = train_metrics['volatility'] * 100
        vol_val = val_metrics['volatility'] * 100
        vol_chg = ((vol_val - vol_train) / abs(vol_train) * 100) if vol_train != 0 else np.nan
        if full_metrics is not None:
            vol_full = full_metrics['volatility'] * 100
            print(f"{'Volatility':<25} {vol_train:>16.2f}% {vol_val:>16.2f}% {vol_full:>16.2f}% {vol_chg:>16.1f}%" if not np.isnan(vol_chg) else f"{'Volatility':<25} {vol_train:>16.2f}% {vol_val:>16.2f}% {vol_full:>16.2f}% {'N/A':>18}")
        else:
            print(f"{'Volatility':<25} {vol_train:>18.2f}% {vol_val:>18.2f}% {vol_chg:>18.1f}%" if not np.isnan(vol_chg) else f"{'Volatility':<25} {vol_train:>18.2f}% {vol_val:>18.2f}% {'N/A':>20}")
        
        # Risk-adjusted metrics
        print(f"\n{'RISK-ADJUSTED METRICS:':<25}")
        sr_train = train_metrics['sharpe_ratio']
        sr_val = val_metrics['sharpe_ratio']
        sr_chg = ((sr_val - sr_train) / abs(sr_train) * 100) if sr_train != 0 and not np.isnan(sr_train) else np.nan
        if full_metrics is not None:
            sr_full = full_metrics['sharpe_ratio']
            print(f"{'Sharpe Ratio':<25} {sr_train:>16.3f} {sr_val:>16.3f} {sr_full:>16.3f} {sr_chg:>16.1f}%" if not np.isnan(sr_chg) else f"{'Sharpe Ratio':<25} {sr_train:>16.3f} {sr_val:>16.3f} {sr_full:>16.3f} {'N/A':>18}")
        else:
            print(f"{'Sharpe Ratio':<25} {sr_train:>18.3f} {sr_val:>18.3f} {sr_chg:>18.1f}%" if not np.isnan(sr_chg) else f"{'Sharpe Ratio':<25} {sr_train:>18.3f} {sr_val:>18.3f} {'N/A':>20}")
        
        sor_train = train_metrics['sortino_ratio']
        sor_val = val_metrics['sortino_ratio']
        sor_chg = ((sor_val - sor_train) / abs(sor_train) * 100) if sor_train != 0 and not np.isnan(sor_train) else np.nan
        if full_metrics is not None:
            sor_full = full_metrics['sortino_ratio']
            print(f"{'Sortino Ratio':<25} {sor_train:>16.3f} {sor_val:>16.3f} {sor_full:>16.3f} {sor_chg:>16.1f}%" if not np.isnan(sor_chg) else f"{'Sortino Ratio':<25} {sor_train:>16.3f} {sor_val:>16.3f} {sor_full:>16.3f} {'N/A':>18}")
        else:
            print(f"{'Sortino Ratio':<25} {sor_train:>18.3f} {sor_val:>18.3f} {sor_chg:>18.1f}%" if not np.isnan(sor_chg) else f"{'Sortino Ratio':<25} {sor_train:>18.3f} {sor_val:>18.3f} {'N/A':>20}")
        
        # Trade metrics
        print(f"\n{'TRADE METRICS:':<25}")
        if full_metrics is not None:
            print(f"{'Total Trades':<25} {train_metrics['total_trades']:>16} {val_metrics['total_trades']:>16} {full_metrics['total_trades']:>16} {'N/A':>18}")
        else:
            print(f"{'Total Trades':<25} {train_metrics['total_trades']:>18} {val_metrics['total_trades']:>18} {'N/A':>20}")
        
        wr_train = train_metrics['win_rate']
        wr_val = val_metrics['win_rate']
        wr_chg = ((wr_val - wr_train) / abs(wr_train) * 100) if wr_train != 0 else np.nan
        if full_metrics is not None:
            wr_full = full_metrics['win_rate']
            print(f"{'Win Rate':<25} {wr_train:>16.1f}% {wr_val:>16.1f}% {wr_full:>16.1f}% {wr_chg:>16.1f}%" if not np.isnan(wr_chg) else f"{'Win Rate':<25} {wr_train:>16.1f}% {wr_val:>16.1f}% {wr_full:>16.1f}% {'N/A':>18}")
        else:
            print(f"{'Win Rate':<25} {wr_train:>18.1f}% {wr_val:>18.1f}% {wr_chg:>18.1f}%" if not np.isnan(wr_chg) else f"{'Win Rate':<25} {wr_train:>18.1f}% {wr_val:>18.1f}% {'N/A':>20}")
        
        pf_train = train_metrics.get('profit_factor', np.nan)
        pf_val = val_metrics.get('profit_factor', np.nan)
        if full_metrics is not None:
            pf_full = full_metrics.get('profit_factor', np.nan)
            pf_full_str = f"{pf_full:>16.2f}" if not np.isnan(pf_full) and not np.isinf(pf_full) else "Inf" if np.isinf(pf_full) else "N/A"
            pf_train_str = f"{pf_train:>16.2f}" if not np.isnan(pf_train) and not np.isinf(pf_train) else "Inf" if np.isinf(pf_train) else "N/A"
            pf_val_str = f"{pf_val:>16.2f}" if not np.isnan(pf_val) and not np.isinf(pf_val) else "Inf" if np.isinf(pf_val) else "N/A"
            print(f"{'Profit Factor':<25} {pf_train_str:>16} {pf_val_str:>16} {pf_full_str:>16} {'N/A':>18}")
        else:
            pf_train_str = f"{pf_train:>18.2f}" if not np.isnan(pf_train) and not np.isinf(pf_train) else "Inf" if np.isinf(pf_train) else "N/A"
            pf_val_str = f"{pf_val:>18.2f}" if not np.isnan(pf_val) and not np.isinf(pf_val) else "Inf" if np.isinf(pf_val) else "N/A"
            print(f"{'Profit Factor':<25} {pf_train_str:>18} {pf_val_str:>18} {'N/A':>20}")
        
        if full_analyzer is not None:
            print("="*110)
        else:
            print("="*90)
    
    def plot_rolling_sortino_and_trades(self, train_end_date: Optional[pd.Timestamp] = None,
                                       window: int = 63, figsize=(18, 14)):
        """
        Plot rolling Sortino ratio over time and trade entries/exits for each strategy leg separately.
        
        Args:
            train_end_date: Date where training period ends (for split indicator)
            window: Rolling window for Sortino calculation (default 63 days = ~3 months)
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        
        # Calculate number of legs
        num_legs = len(self.portfolio.legs) if hasattr(self.portfolio, 'legs') else 0
        
        # Create subplot layout: Sortino on top, then one subplot per leg
        if num_legs > 0:
            gs = fig.add_gridspec(num_legs + 2, 1, height_ratios=[1.5, 1] + [1] * num_legs, hspace=0.3)
        else:
            gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.3)
        
        # Plot 1: Rolling Sortino Ratio
        ax_sortino = fig.add_subplot(gs[0, 0])
        
        ret = self.returns.fillna(0.0)
        risk_free_rate = 0.0
        
        # Calculate rolling Sortino ratio
        rolling_window = window
        rolling_sortino = pd.Series(index=ret.index, dtype=float)
        
        for i in range(rolling_window, len(ret)):
            window_returns = ret.iloc[i-rolling_window:i]
            if len(window_returns) > 0:
                excess_returns = window_returns - (risk_free_rate / 252)
                downside_excess_returns = excess_returns[excess_returns < 0]
                if len(downside_excess_returns) > 0 and downside_excess_returns.std() > 0:
                    mean_excess = excess_returns.mean()
                    sortino = (mean_excess / downside_excess_returns.std()) * np.sqrt(252)
                    rolling_sortino.iloc[i] = sortino
        
        ax_sortino.plot(rolling_sortino.index, rolling_sortino.values, 
                       linewidth=2, color='purple', label=f'Rolling Sortino ({window} days)')
        ax_sortino.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax_sortino.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sortino = 1')
        ax_sortino.axhline(y=2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Sortino = 2')
        
        # Add train/validation split indicator
        if train_end_date is not None:
            ax_sortino.axvline(x=train_end_date, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Train/Val Split')
            # Add text annotation
            y_max = rolling_sortino.max() if len(rolling_sortino.dropna()) > 0 else 2
            y_min = rolling_sortino.min() if len(rolling_sortino.dropna()) > 0 else -2
            y_pos = y_max * 0.9 if not np.isnan(y_max) else 2
            ax_sortino.text(train_end_date, y_pos, 
                           'Validation\nStarts', ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                           fontsize=10, fontweight='bold')
        
        ax_sortino.set_ylabel('Sortino Ratio', fontweight='bold')
        ax_sortino.set_title('Rolling Sortino Ratio Over Time', fontweight='bold', fontsize=14)
        ax_sortino.legend(loc='best')
        ax_sortino.grid(True, alpha=0.3)
        ax_sortino.fill_between(rolling_sortino.index, rolling_sortino.values, 0, 
                               where=(rolling_sortino.values > 0), alpha=0.2, color='green')
        ax_sortino.fill_between(rolling_sortino.index, rolling_sortino.values, 0, 
                               where=(rolling_sortino.values <= 0), alpha=0.2, color='red')
        
        # Plot 2: Combined portfolio equity for reference
        ax_equity = fig.add_subplot(gs[1, 0], sharex=ax_sortino)
        ax_equity.plot(self.equity.index, self.equity.values, 
                      linewidth=2, color='darkgreen', label='Combined Portfolio Equity')
        
        if train_end_date is not None:
            ax_equity.axvline(x=train_end_date, color='red', linestyle='--', 
                             linewidth=2, alpha=0.7)
        
        ax_equity.set_ylabel('Portfolio Value ($)', fontweight='bold')
        ax_equity.set_title('Combined Portfolio Equity', fontweight='bold')
        ax_equity.legend(loc='best')
        ax_equity.grid(True, alpha=0.3)
        
        # Plot 3+: Individual strategy legs with trade entries/exits
        if num_legs > 0:
            leg_list = list(self.portfolio.legs.items())
            for idx, (leg_id, leg) in enumerate(leg_list):
                ax_leg = fig.add_subplot(gs[idx + 2, 0], sharex=ax_sortino)
                
                # Get price data for this leg - check if it's a FullPeriodLeg
                prices = None
                train_leg = None
                val_leg = None
                
                # Check if this is a full period leg (has train_leg and val_leg attributes)
                if hasattr(leg, 'train_leg') and hasattr(leg, 'val_leg'):
                    train_leg = leg.train_leg
                    val_leg = leg.val_leg
                    # Combine price data from both periods
                    if hasattr(train_leg, 'portfolio') and train_leg.portfolio is not None:
                        try:
                            train_prices = train_leg.portfolio.close
                            if hasattr(val_leg, 'portfolio') and val_leg.portfolio is not None:
                                val_prices = val_leg.portfolio.close
                                prices = pd.concat([train_prices, val_prices])
                        except:
                            pass
                elif hasattr(leg, 'portfolio') and leg.portfolio is not None:
                    try:
                        prices = leg.portfolio.close
                    except:
                        pass
                
                # Plot price if available
                if prices is not None:
                    ax_leg.plot(prices.index, prices.values, 
                               linewidth=1.5, color='black', alpha=0.6, label='Price')
                    
                    # Get trade signals from this leg
                    # If it's a full period leg, combine trades from both periods
                    if train_leg and val_leg:
                        # Combine trades from both train and validation periods
                        try:
                            train_orders = train_leg.portfolio.orders.records_readable if hasattr(train_leg.portfolio, 'orders') else pd.DataFrame()
                            val_orders = val_leg.portfolio.orders.records_readable if hasattr(val_leg.portfolio, 'orders') else pd.DataFrame()
                            
                            all_orders = pd.concat([train_orders, val_orders]).reset_index(drop=True) if len(train_orders) > 0 and len(val_orders) > 0 else (train_orders if len(train_orders) > 0 else val_orders)
                            
                            if len(all_orders) > 0:
                                # Entry signals (buy orders)
                                buy_orders = all_orders[all_orders['Side'] == 'Buy']
                                if len(buy_orders) > 0:
                                    entry_dates = pd.to_datetime(buy_orders['Timestamp'])
                                    entry_prices = buy_orders['Price']
                                    ax_leg.scatter(entry_dates, entry_prices, 
                                                 marker='^', color='green', s=80, 
                                                 alpha=0.7, label='Entry', zorder=5)
                                
                                # Exit signals (sell orders)
                                sell_orders = all_orders[all_orders['Side'] == 'Sell']
                                if len(sell_orders) > 0:
                                    exit_dates = pd.to_datetime(sell_orders['Timestamp'])
                                    exit_prices = sell_orders['Price']
                                    ax_leg.scatter(exit_dates, exit_prices, 
                                                 marker='v', color='red', s=80, 
                                                 alpha=0.7, label='Exit', zorder=5)
                        except Exception as e:
                            # Fallback: try getting orders from the leg's portfolio
                            try:
                                orders = leg.portfolio.orders.records_readable if hasattr(leg, 'portfolio') and hasattr(leg.portfolio, 'orders') else pd.DataFrame()
                                if len(orders) > 0:
                                    buy_orders = orders[orders['Side'] == 'Buy']
                                    if len(buy_orders) > 0:
                                        entry_dates = pd.to_datetime(buy_orders['Timestamp'])
                                        entry_prices = buy_orders['Price']
                                        ax_leg.scatter(entry_dates, entry_prices, 
                                                     marker='^', color='green', s=80, 
                                                     alpha=0.7, label='Entry', zorder=5)
                                    
                                    sell_orders = orders[orders['Side'] == 'Sell']
                                    if len(sell_orders) > 0:
                                        exit_dates = pd.to_datetime(sell_orders['Timestamp'])
                                        exit_prices = sell_orders['Price']
                                        ax_leg.scatter(exit_dates, exit_prices, 
                                                     marker='v', color='red', s=80, 
                                                     alpha=0.7, label='Exit', zorder=5)
                            except:
                                pass
                    else:
                        # Regular leg - get orders directly
                        try:
                            orders = leg.portfolio.orders.records_readable
                            if len(orders) > 0:
                                buy_orders = orders[orders['Side'] == 'Buy']
                                if len(buy_orders) > 0:
                                    entry_dates = pd.to_datetime(buy_orders['Timestamp'])
                                    entry_prices = buy_orders['Price']
                                    ax_leg.scatter(entry_dates, entry_prices, 
                                                 marker='^', color='green', s=80, 
                                                 alpha=0.7, label='Entry', zorder=5)
                                
                                sell_orders = orders[orders['Side'] == 'Sell']
                                if len(sell_orders) > 0:
                                    exit_dates = pd.to_datetime(sell_orders['Timestamp'])
                                    exit_prices = sell_orders['Price']
                                    ax_leg.scatter(exit_dates, exit_prices, 
                                                 marker='v', color='red', s=80, 
                                                 alpha=0.7, label='Exit', zorder=5)
                        except:
                            pass
                    
                    # Add train/validation split indicator
                    if train_end_date is not None:
                        ax_leg.axvline(x=train_end_date, color='red', linestyle='--', 
                                      linewidth=2, alpha=0.7)
                    
                    weight = self.portfolio.weights.get(leg_id, 0.0)
                    ax_leg.set_ylabel('Price ($)', fontweight='bold')
                    ax_leg.set_title(f'{leg_id} ({leg.symbol}, {leg.strategy_name}) - Weight: {weight:.1%}', 
                                    fontweight='bold', fontsize=11)
                    ax_leg.legend(loc='best', fontsize=8)
                    ax_leg.grid(True, alpha=0.3)
                else:
                    # No price data - show equity curve
                    if hasattr(leg, 'equity_curve'):
                        equity_data = leg.equity_curve
                        ax_leg.plot(equity_data.index, equity_data.values,
                                   linewidth=1.5, color='blue', label='Equity')
                        if train_end_date is not None:
                            ax_leg.axvline(x=train_end_date, color='red', linestyle='--', 
                                          linewidth=2, alpha=0.7)
                        weight = self.portfolio.weights.get(leg_id, 0.0)
                        ax_leg.set_ylabel('Equity ($)', fontweight='bold')
                        ax_leg.set_title(f'{leg_id} ({leg.symbol}, {leg.strategy_name}) - Weight: {weight:.1%}', 
                                        fontweight='bold', fontsize=11)
                        ax_leg.legend(loc='best', fontsize=8)
                        ax_leg.grid(True, alpha=0.3)
                    else:
                        ax_leg.text(0.5, 0.5, f'No data for {leg_id}', 
                                   ha='center', va='center', transform=ax_leg.transAxes)
                        ax_leg.set_ylabel('N/A', fontweight='bold')
                        ax_leg.set_title(f'{leg_id}', fontweight='bold', fontsize=11)
        
        # Set x-axis label on last subplot
        if num_legs > 0:
            fig.axes[-1].set_xlabel('Date', fontweight='bold')
        else:
            ax_equity.set_xlabel('Date', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

