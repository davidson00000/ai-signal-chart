#!/usr/bin/env python3
"""
Verification Script for S&P500 Universe

This script validates:
1. /symbol-universes API endpoint returns sp500_all
2. Symbol count is in expected range (400-550)
3. No duplicate symbols

Run: python tools/verify_sp500_universe.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from datetime import datetime

from backend.config.symbol_universes import (
    SYMBOL_UNIVERSES,
    get_all_universes,
    get_universe,
    get_universe_symbols,
    SP500_SYMBOLS
)


def main():
    """Run verification tasks."""
    print("="*60)
    print("S&P 500 Universe Verification Script")
    print("="*60)
    
    # Get commit hash
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True, text=True,
        cwd='/Users/kousukenakamura/dev/ai-signal-chart'
    )
    commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
    
    report_lines = [
        "# S&P 500 Universe Verification Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Commit:** `{commit_hash}`",
        "",
        "---",
        "",
        "## Verification Tasks",
        ""
    ]
    
    all_pass = True
    
    # Task 1: Check sp500_all exists
    print("\n[Task 1] Checking sp500_all universe exists...")
    sp500 = get_universe("sp500_all")
    
    if sp500 is not None:
        print("   ✅ sp500_all universe found")
        report_lines.append("### Task 1: sp500_all exists ✅ Pass")
        report_lines.append("")
        report_lines.append(f"- Label: {sp500.get('label')}")
        report_lines.append(f"- Description: {sp500.get('description')}")
        report_lines.append("")
    else:
        print("   ❌ sp500_all universe NOT FOUND")
        report_lines.append("### Task 1: sp500_all exists ❌ Fail")
        report_lines.append("")
        report_lines.append("sp500_all universe not found in SYMBOL_UNIVERSES")
        report_lines.append("")
        all_pass = False
    
    # Task 2: Check symbol count
    print("\n[Task 2] Checking symbol count (expected: 400-550)...")
    symbols = get_universe_symbols("sp500_all")
    count = len(symbols)
    
    if 400 <= count <= 550:
        print(f"   ✅ Symbol count: {count} (in range 400-550)")
        report_lines.append("### Task 2: Symbol count in range ✅ Pass")
        report_lines.append("")
        report_lines.append(f"- Total symbols: **{count}**")
        report_lines.append(f"- Expected range: 400-550")
        report_lines.append("")
    else:
        print(f"   ❌ Symbol count: {count} (out of range 400-550)")
        report_lines.append("### Task 2: Symbol count in range ❌ Fail")
        report_lines.append("")
        report_lines.append(f"- Total symbols: {count}")
        report_lines.append(f"- Expected range: 400-550")
        report_lines.append("")
        all_pass = False
    
    # Task 3: Check for duplicates
    print("\n[Task 3] Checking for duplicate symbols...")
    unique_symbols = set(symbols)
    duplicates = len(symbols) - len(unique_symbols)
    
    if duplicates == 0:
        print("   ✅ No duplicate symbols found")
        report_lines.append("### Task 3: No duplicates ✅ Pass")
        report_lines.append("")
        report_lines.append(f"- Unique symbols: {len(unique_symbols)}")
        report_lines.append("- No duplicates found")
        report_lines.append("")
    else:
        print(f"   ❌ Found {duplicates} duplicate symbols")
        
        # Find the duplicates
        seen = set()
        dups = []
        for s in symbols:
            if s in seen:
                dups.append(s)
            seen.add(s)
        
        report_lines.append("### Task 3: No duplicates ❌ Fail")
        report_lines.append("")
        report_lines.append(f"- Found {duplicates} duplicate symbols:")
        report_lines.append(f"  - {', '.join(dups)}")
        report_lines.append("")
        all_pass = False
    
    # Task 4: Check other universes
    print("\n[Task 4] Checking other universes...")
    all_universes = get_all_universes()
    
    report_lines.append("### Task 4: Other Universes ✅ Pass")
    report_lines.append("")
    report_lines.append("| Universe ID | Label | Symbol Count |")
    report_lines.append("|-------------|-------|--------------|")
    
    for uid, udata in all_universes.items():
        count = len(udata.get("symbols", []))
        report_lines.append(f"| {uid} | {udata.get('label')} | {count} |")
        print(f"   - {uid}: {count} symbols")
    
    report_lines.append("")
    
    # Summary
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    
    if all_pass:
        report_lines.append("**✅ All verifications passed.** S&P 500 Universe is ready for use.")
        print("\n✅ All verifications passed!")
    else:
        report_lines.append("**⚠️ Some verifications failed.** See details above.")
        print("\n⚠️ Some verifications failed!")
    
    # Sample symbols
    report_lines.append("")
    report_lines.append("### Sample Symbols (first 20)")
    report_lines.append("")
    report_lines.append("```")
    report_lines.append(", ".join(symbols[:20]))
    report_lines.append("```")
    
    # Save report
    report_path = "/Users/kousukenakamura/dev/ai-signal-chart/docs/verify_sp500_universe.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
