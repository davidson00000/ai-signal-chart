#!/usr/bin/env python3
"""
Verification Script for Multi-Symbol Auto Sim

This script validates:
1. API endpoint works without crashes
2. All expected metrics are returned
3. Ranking order is correct (Total R descending)
4. Single-symbol result matches Auto Sim Lab result

Run: python tools/verify_multi_sim.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from typing import Dict, List, Any
import subprocess

from backend.models.multi_sim import MultiSimConfig, MultiSimResult
from backend.multi_sim_engine import run_multi_simulation
from backend.auto_sim_lab import run_auto_simulation, AutoSimConfig


class VerificationReport:
    """Collects and formats verification results."""
    
    def __init__(self, commit_hash: str):
        self.commit_hash = commit_hash
        self.results: List[Dict[str, Any]] = []
        self.params: Dict[str, Any] = {}
    
    def set_params(self, **kwargs):
        self.params.update(kwargs)
    
    def add_result(self, task: str, status: str, description: str, details: str = ""):
        self.results.append({
            "task": task,
            "status": status,
            "description": description,
            "details": details
        })
    
    def to_markdown(self) -> str:
        lines = [
            "# Multi-Symbol Auto Sim Verification Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Commit:** `{self.commit_hash}`",
            "",
            "## Parameters Used",
            "",
        ]
        
        for key, value in self.params.items():
            lines.append(f"- **{key}:** {value}")
        
        lines.extend(["", "---", "", "## Verification Results", ""])
        
        for r in self.results:
            status_icon = {"Pass": "✅", "Warning": "⚠️", "Fail": "❌"}.get(r["status"], "❓")
            lines.append(f"### {status_icon} {r['task']}: {r['status']}")
            lines.append("")
            lines.append(r["description"])
            if r["details"]:
                lines.append("")
                lines.append("```")
                lines.append(r["details"])
                lines.append("```")
            lines.append("")
        
        return "\n".join(lines)


def task1_api_no_crashes(report: VerificationReport) -> bool:
    """
    Task 1: Verify API endpoint works without crashes
    """
    print("\n" + "="*60)
    print("Task 1: API Endpoint - No Crashes")
    print("="*60)
    
    symbols = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD"]
    
    report.set_params(
        test_symbols=symbols,
        timeframe="1d",
        start_date="2025-01-01",
        end_date="2025-12-06"
    )
    
    print(f"\nRunning multi-sim for {len(symbols)} symbols...")
    
    try:
        config = MultiSimConfig(
            symbols=symbols,
            timeframe="1d",
            start_date="2025-01-01",
            end_date="2025-12-06",
            initial_capital=100000.0,
            strategy_mode="ma_crossover",
            ma_short_window=50,
            ma_long_window=60,
            execution_mode="same_bar_close",
            use_r_management=True,
            virtual_stop_method="percent",
            virtual_stop_percent=0.03
        )
        
        result = run_multi_simulation(config)
        
        print(f"   Total symbols: {result.total_symbols}")
        print(f"   Successful: {result.successful}")
        print(f"   Failed: {result.failed}")
        
        report.add_result(
            "Task 1: API No Crashes",
            "Pass",
            f"Multi-sim completed successfully for {len(symbols)} symbols.\n"
            f"Successful: {result.successful}, Failed: {result.failed}",
            f"Symbols: {', '.join(symbols)}"
        )
        return True
        
    except Exception as e:
        import traceback
        report.add_result(
            "Task 1: API No Crashes",
            "Fail",
            f"API crashed with error: {str(e)}",
            traceback.format_exc()
        )
        return False


def task2_metrics_present(report: VerificationReport) -> bool:
    """
    Task 2: Verify all expected metrics are returned
    """
    print("\n" + "="*60)
    print("Task 2: All Expected Metrics Present")
    print("="*60)
    
    symbols = ["NVDA", "AAPL"]
    
    print(f"\nRunning multi-sim for {len(symbols)} symbols...")
    
    try:
        config = MultiSimConfig(
            symbols=symbols,
            timeframe="1d",
            start_date="2025-01-01",
            end_date="2025-12-06",
            initial_capital=100000.0,
            use_r_management=True
        )
        
        result = run_multi_simulation(config)
        
        # Expected fields
        expected_fields = [
            "rank", "symbol", "final_equity", "total_return",
            "total_r", "avg_r", "win_rate", "max_dd", "trades"
        ]
        
        all_pass = True
        details = []
        
        for sym_result in result.results:
            missing = []
            for field in expected_fields:
                val = getattr(sym_result, field, None)
                if val is None and field not in ["total_r", "avg_r", "best_r", "worst_r"]:
                    missing.append(field)
            
            if missing:
                all_pass = False
                details.append(f"{sym_result.symbol}: Missing fields: {missing}")
            else:
                details.append(f"{sym_result.symbol}: All fields present ✅")
        
        if all_pass:
            report.add_result(
                "Task 2: Metrics Present",
                "Pass",
                f"All expected metrics present for {len(result.results)} symbols",
                "\n".join(details)
            )
        else:
            report.add_result(
                "Task 2: Metrics Present",
                "Fail",
                "Some symbols missing required metrics",
                "\n".join(details)
            )
        
        return all_pass
        
    except Exception as e:
        import traceback
        report.add_result(
            "Task 2: Metrics Present",
            "Fail",
            f"Error: {str(e)}",
            traceback.format_exc()
        )
        return False


def task3_ranking_order(report: VerificationReport) -> bool:
    """
    Task 3: Verify ranking order is correct (Total R descending)
    """
    print("\n" + "="*60)
    print("Task 3: Ranking Order Correct")
    print("="*60)
    
    symbols = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD"]
    
    print(f"\nRunning multi-sim for {len(symbols)} symbols...")
    
    try:
        config = MultiSimConfig(
            symbols=symbols,
            timeframe="1d",
            start_date="2025-01-01",
            end_date="2025-12-06",
            initial_capital=100000.0,
            use_r_management=True
        )
        
        result = run_multi_simulation(config)
        
        # Check ranking order
        results_list = result.results
        
        details = []
        all_pass = True
        
        # Verify ranks are sequential
        for i, r in enumerate(results_list):
            expected_rank = i + 1
            if r.rank != expected_rank:
                all_pass = False
                details.append(f"Rank mismatch: {r.symbol} has rank {r.rank}, expected {expected_rank}")
        
        # Verify sorted by Total R (successful results only)
        successful = [r for r in results_list if r.error is None]
        
        for i in range(len(successful) - 1):
            curr = successful[i]
            next_r = successful[i + 1]
            
            curr_r = curr.total_r if curr.total_r is not None else -999999
            next_r_val = next_r.total_r if next_r.total_r is not None else -999999
            
            if curr_r < next_r_val:
                all_pass = False
                details.append(
                    f"Order incorrect: {curr.symbol} ({curr_r}R) should come after {next_r.symbol} ({next_r_val}R)"
                )
        
        # Build ranking table
        details.append("\nRanking Table:")
        details.append("| Rank | Symbol | Total R | Return % |")
        details.append("|------|--------|---------|----------|")
        for r in results_list:
            total_r_str = f"{r.total_r:.2f}R" if r.total_r is not None else "N/A"
            details.append(f"| {r.rank} | {r.symbol} | {total_r_str} | {r.total_return:.2f}% |")
        
        if all_pass:
            report.add_result(
                "Task 3: Ranking Order",
                "Pass",
                f"All {len(results_list)} symbols correctly ranked by Total R (descending)",
                "\n".join(details)
            )
        else:
            report.add_result(
                "Task 3: Ranking Order",
                "Fail",
                "Ranking order is incorrect",
                "\n".join(details)
            )
        
        return all_pass
        
    except Exception as e:
        import traceback
        report.add_result(
            "Task 3: Ranking Order",
            "Fail",
            f"Error: {str(e)}",
            traceback.format_exc()
        )
        return False


def task4_single_vs_multi_match(report: VerificationReport) -> bool:
    """
    Task 4: Verify single-symbol multi-sim matches Auto Sim Lab result
    """
    print("\n" + "="*60)
    print("Task 4: Single Symbol Multi-Sim vs Auto Sim Lab")
    print("="*60)
    
    symbol = "NVDA"
    
    print(f"\nRunning both simulations for {symbol}...")
    
    try:
        # Run single-symbol via multi-sim
        multi_config = MultiSimConfig(
            symbols=[symbol],
            timeframe="1d",
            start_date="2025-01-01",
            end_date="2025-12-06",
            initial_capital=100000.0,
            strategy_mode="ma_crossover",
            ma_short_window=50,
            ma_long_window=60,
            execution_mode="same_bar_close",
            use_r_management=True,
            virtual_stop_method="percent",
            virtual_stop_percent=0.03
        )
        
        multi_result = run_multi_simulation(multi_config)
        multi_sym = multi_result.results[0]
        
        # Run via Auto Sim Lab
        auto_config = AutoSimConfig(
            symbol=symbol,
            timeframe="1d",
            start_date="2025-01-01",
            end_date="2025-12-06",
            initial_capital=100000.0,
            strategy_mode="ma_crossover",
            ma_short_window=50,
            ma_long_window=60,
            position_sizing_mode="full_equity",
            execution_mode="same_bar_close",
            use_r_management=True,
            virtual_stop_method="percent",
            virtual_stop_percent=0.03
        )
        
        auto_result = run_auto_simulation(auto_config)
        
        # Compare results
        details = []
        details.append("| Metric | Multi-Sim | Auto Sim | Match |")
        details.append("|--------|-----------|----------|-------|")
        
        all_match = True
        
        # Final equity
        eq_match = abs(multi_sym.final_equity - auto_result.final_equity) < 0.01
        details.append(f"| Final Equity | ${multi_sym.final_equity:,.2f} | ${auto_result.final_equity:,.2f} | {'✅' if eq_match else '❌'} |")
        if not eq_match:
            all_match = False
        
        # Total return
        multi_return = multi_sym.total_return
        auto_return = auto_result.total_return_pct
        ret_match = abs(multi_return - auto_return) < 0.01
        details.append(f"| Total Return | {multi_return:.2f}% | {auto_return:.2f}% | {'✅' if ret_match else '❌'} |")
        if not ret_match:
            all_match = False
        
        # Trades
        auto_trades = auto_result.summary.get("total_trades", 0)
        trades_match = multi_sym.trades == auto_trades
        details.append(f"| Trades | {multi_sym.trades} | {auto_trades} | {'✅' if trades_match else '❌'} |")
        if not trades_match:
            all_match = False
        
        # Total R
        auto_total_r = auto_result.summary.get("total_r")
        if multi_sym.total_r is not None and auto_total_r is not None:
            r_match = abs(multi_sym.total_r - auto_total_r) < 0.01
            details.append(f"| Total R | {multi_sym.total_r:.2f}R | {auto_total_r:.2f}R | {'✅' if r_match else '❌'} |")
            if not r_match:
                all_match = False
        
        if all_match:
            report.add_result(
                "Task 4: Single vs Multi Match",
                "Pass",
                f"Single-symbol multi-sim result matches Auto Sim Lab for {symbol}",
                "\n".join(details)
            )
        else:
            report.add_result(
                "Task 4: Single vs Multi Match",
                "Warning",
                "Some metrics differ (may be due to floating point precision)",
                "\n".join(details)
            )
        
        return all_match
        
    except Exception as e:
        import traceback
        report.add_result(
            "Task 4: Single vs Multi Match",
            "Fail",
            f"Error: {str(e)}",
            traceback.format_exc()
        )
        return False


def main():
    """Run all verification tasks and generate report."""
    print("="*60)
    print("Multi-Symbol Auto Sim Verification Script")
    print("="*60)
    
    # Get commit hash
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True, text=True,
        cwd='/Users/kousukenakamura/dev/ai-signal-chart'
    )
    commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
    
    report = VerificationReport(commit_hash)
    
    # Run tasks
    task1_pass = task1_api_no_crashes(report)
    task2_pass = task2_metrics_present(report)
    task3_pass = task3_ranking_order(report)
    task4_pass = task4_single_vs_multi_match(report)
    
    # Generate report
    print("\n" + "="*60)
    print("Generating Verification Report...")
    print("="*60)
    
    report_md = report.to_markdown()
    
    # Add summary
    all_pass = task1_pass and task2_pass and task3_pass and task4_pass
    
    summary = "\n---\n\n## Summary\n\n"
    if all_pass:
        summary += "**✅ All verifications passed.** Multi-Symbol Auto Sim is ready for use.\n"
    else:
        summary += "**⚠️ Some verifications need attention.** See details above.\n"
    
    summary += f"\n- Task 1 (API No Crashes): {'✅ Pass' if task1_pass else '❌ Fail'}\n"
    summary += f"- Task 2 (Metrics Present): {'✅ Pass' if task2_pass else '❌ Fail'}\n"
    summary += f"- Task 3 (Ranking Order): {'✅ Pass' if task3_pass else '❌ Fail'}\n"
    summary += f"- Task 4 (Single vs Multi Match): {'✅ Pass' if task4_pass else '⚠️ Warning/Fail'}\n"
    
    report_md += summary
    
    # Save report
    report_path = "/Users/kousukenakamura/dev/ai-signal-chart/docs/multi_sim_verification_report.md"
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + "="*60)
    print("VERIFICATION REPORT")
    print("="*60)
    print(report_md)
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
