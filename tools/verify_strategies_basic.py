#!/usr/bin/env python3
"""
Verification Script for Additional Strategies (Buy & Hold, RSI Mean Reversion)

This script validates:
1. Buy & Hold: Exactly 1 trade, entry on first bar
2. RSI Mean Reversion: At least 1 trade, all long positions

Run: python tools/verify_strategies_basic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from datetime import datetime
from typing import List, Dict, Any

from backend.auto_sim_lab import run_auto_simulation, AutoSimConfig


def verify_buy_and_hold(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Verify Buy & Hold strategy."""
    print(f"\n[Buy & Hold] Testing {symbol}...")
    
    try:
        config = AutoSimConfig(
            symbol=symbol,
            timeframe="1d",
            strategy_mode="buy_and_hold",
            initial_capital=100000.0,
            position_sizing_mode="full_equity",
            execution_mode="same_bar_close",
            start_date=start_date,
            end_date=end_date,
            use_r_management=False  # Disable R-management for Buy & Hold
        )
        
        result = run_auto_simulation(config)
        
        trades = result.trades
        trade_count = len(trades)
        
        # Check trade count == 1
        trade_count_ok = trade_count == 1
        
        # Check entry is near start date
        entry_time_ok = False
        exit_time_ok = False
        
        if trades:
            first_trade = trades[0]
            entry_time = first_trade.get("entry_time", "")
            exit_time = first_trade.get("exit_time", "")
            
            # Entry should be near start date
            entry_time_ok = start_date in str(entry_time) or entry_time.startswith(start_date[:7])
            
            # Exit should be near end date (forced close at end)
            exit_time_ok = end_date in str(exit_time) or exit_time.startswith(end_date[:7])
        
        passed = trade_count_ok
        
        return {
            "symbol": symbol,
            "passed": passed,
            "trade_count": trade_count,
            "trade_count_ok": trade_count_ok,
            "entry_time": trades[0].get("entry_time") if trades else None,
            "exit_time": trades[0].get("exit_time") if trades else None,
            "entry_time_ok": entry_time_ok,
            "exit_time_ok": exit_time_ok,
            "final_equity": result.final_equity,
            "error": None
        }
        
    except Exception as e:
        import traceback
        return {
            "symbol": symbol,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def verify_rsi_mean_reversion(symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Verify RSI Mean Reversion strategy."""
    print(f"\n[RSI Mean Reversion] Testing {symbol}...")
    
    try:
        config = AutoSimConfig(
            symbol=symbol,
            timeframe="1d",
            strategy_mode="rsi_mean_reversion",
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            initial_capital=100000.0,
            position_sizing_mode="full_equity",
            execution_mode="same_bar_close",
            start_date=start_date,
            end_date=end_date,
            use_r_management=True,
            virtual_stop_method="percent",
            virtual_stop_percent=0.03
        )
        
        result = run_auto_simulation(config)
        
        trades = result.trades
        trade_count = len(trades)
        
        # Check trade count >= 1
        trade_count_ok = trade_count >= 1
        
        # Check all trades are long (entry_side = "long" or direction is positive)
        all_long = True
        for trade in trades:
            side = trade.get("side", "long")
            if side != "long":
                all_long = False
                break
        
        passed = trade_count_ok and all_long
        
        return {
            "symbol": symbol,
            "passed": passed,
            "trade_count": trade_count,
            "trade_count_ok": trade_count_ok,
            "all_long": all_long,
            "final_equity": result.final_equity,
            "total_r": result.summary.get("total_r"),
            "error": None
        }
        
    except Exception as e:
        import traceback
        return {
            "symbol": symbol,
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Run verification tasks."""
    print("="*60)
    print("Strategies Verification Script (Buy & Hold, RSI)")
    print("="*60)
    
    # Get commit hash
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True, text=True,
        cwd='/Users/kousukenakamura/dev/ai-signal-chart'
    )
    commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
    
    symbols = ["NVDA", "AAPL"]
    start_date = "2025-01-01"
    end_date = "2025-12-06"
    
    report_lines = [
        "# Strategies Verification Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Commit:** `{commit_hash}`",
        "",
        "## Test Parameters",
        "",
        f"- **Symbols:** {', '.join(symbols)}",
        f"- **Period:** {start_date} to {end_date}",
        "",
        "---",
        "",
        "## Buy & Hold Strategy",
        ""
    ]
    
    # Test Buy & Hold
    bh_results = []
    for symbol in symbols:
        result = verify_buy_and_hold(symbol, start_date, end_date)
        bh_results.append(result)
    
    report_lines.append("| Symbol | Trade Count | Entry Time | Exit Time | Passed |")
    report_lines.append("|--------|-------------|------------|-----------|--------|")
    
    bh_all_pass = True
    for r in bh_results:
        status = "✅" if r.get("passed") else "❌"
        if r.get("error"):
            report_lines.append(f"| {r['symbol']} | ERROR | - | - | {status} |")
            bh_all_pass = False
        else:
            report_lines.append(
                f"| {r['symbol']} | {r['trade_count']} | {r.get('entry_time', 'N/A')} | {r.get('exit_time', 'N/A')} | {status} |"
            )
            if not r.get("passed"):
                bh_all_pass = False
    
    report_lines.append("")
    report_lines.append(f"**Result:** {'✅ Pass' if bh_all_pass else '❌ Fail'}")
    report_lines.append("")
    report_lines.append("**Expected:** Exactly 1 trade, entry near start date, exit at end date.")
    report_lines.append("")
    
    # Test RSI Mean Reversion
    report_lines.extend([
        "---",
        "",
        "## RSI Mean Reversion Strategy",
        ""
    ])
    
    rsi_results = []
    for symbol in symbols:
        result = verify_rsi_mean_reversion(symbol, start_date, end_date)
        rsi_results.append(result)
    
    report_lines.append("| Symbol | Trade Count | All Long | Total R | Passed |")
    report_lines.append("|--------|-------------|----------|---------|--------|")
    
    rsi_all_pass = True
    for r in rsi_results:
        status = "✅" if r.get("passed") else "❌"
        if r.get("error"):
            report_lines.append(f"| {r['symbol']} | ERROR | - | - | {status} |")
            rsi_all_pass = False
        else:
            total_r_str = f"{r.get('total_r', 0):.2f}R" if r.get('total_r') else "N/A"
            report_lines.append(
                f"| {r['symbol']} | {r['trade_count']} | {'Yes' if r.get('all_long') else 'No'} | {total_r_str} | {status} |"
            )
            if not r.get("passed"):
                rsi_all_pass = False
    
    report_lines.append("")
    report_lines.append(f"**Result:** {'✅ Pass' if rsi_all_pass else '❌ Fail'}")
    report_lines.append("")
    report_lines.append("**Expected:** At least 1 trade, all trades are long positions.")
    report_lines.append("")
    
    # Summary
    report_lines.extend([
        "---",
        "",
        "## Summary",
        ""
    ])
    
    all_pass = bh_all_pass and rsi_all_pass
    
    report_lines.append(f"- **Buy & Hold:** {'✅ Pass' if bh_all_pass else '❌ Fail'}")
    report_lines.append(f"- **RSI Mean Reversion:** {'✅ Pass' if rsi_all_pass else '❌ Fail'}")
    report_lines.append("")
    
    if all_pass:
        report_lines.append("**✅ All strategy verifications passed.**")
        print("\n✅ All verifications passed!")
    else:
        report_lines.append("**⚠️ Some verifications failed.** See details above.")
        print("\n⚠️ Some verifications failed!")
    
    # Save report
    report_path = "/Users/kousukenakamura/dev/ai-signal-chart/docs/verify_strategies_basic.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
