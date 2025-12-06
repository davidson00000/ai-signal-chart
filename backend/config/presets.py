"""
Symbol Preset Loader

Loads and manages per-symbol strategy presets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Path to presets file
PRESETS_FILE = Path(__file__).parent / "symbol_presets.json"

# Default preset (used when symbol not found)
DEFAULT_PRESET = {
    "predictor": "rule_v2",
    "timeframe": "1d",
    "lookback": 200,
    "min_hold_days": 3,
    "position_risk_pct": 0.05
}


def _load_presets_file() -> Dict[str, Any]:
    """Load presets from JSON file."""
    if not PRESETS_FILE.exists():
        return {"_default": DEFAULT_PRESET}
    
    try:
        with open(PRESETS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"_default": DEFAULT_PRESET}


def load_symbol_preset(symbol: str) -> Dict[str, Any]:
    """
    Returns preset dict for the given symbol, or a sensible default
    if the symbol is not explicitly defined.
    
    Args:
        symbol: Stock symbol (e.g., "NVDA")
        
    Returns:
        Preset dictionary with keys: predictor, timeframe, lookback, min_hold_days, position_risk_pct
    """
    presets = _load_presets_file()
    
    # Try exact match
    if symbol in presets:
        return presets[symbol]
    
    # Fall back to default
    return presets.get("_default", DEFAULT_PRESET)


def save_symbol_preset(symbol: str, preset: Dict[str, Any]) -> bool:
    """
    Save or update a preset for a symbol.
    
    Args:
        symbol: Stock symbol
        preset: Preset dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        presets = _load_presets_file()
        presets[symbol] = preset
        
        with open(PRESETS_FILE, "w") as f:
            json.dump(presets, f, indent=2)
        
        return True
    except Exception:
        return False


def get_all_presets() -> Dict[str, Any]:
    """Return all presets."""
    return _load_presets_file()
