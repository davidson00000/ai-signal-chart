"""
Technical indicators calculation
Migrated from main_legacy.py - DO NOT MODIFY LOGIC (bug-fixed version)
"""
from typing import List, Optional


def simple_moving_average(values: List[float], window: int) -> List[Optional[float]]:
    """
    単純移動平均。
    window に満たない最初の部分は None を返す（NaN は JSON でエラーになるため使用しない）。
    
    Args:
        values: List of prices (close prices typically)
        window: Moving average window size
        
    Returns:
        List of MA values with None for insufficient data points
        
    Note:
        This function is migrated from main_legacy.py without any logic changes.
        It has been tested and verified to work correctly.
        DO NOT MODIFY THE CALCULATION LOGIC.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    ma: List[Optional[float]] = []
    sum_val = 0.0
    for i, v in enumerate(values):
        sum_val += v
        if i >= window:
            sum_val -= values[i - window]
        if i >= window - 1:
            ma.append(sum_val / window)
        else:
            ma.append(None)
    return ma
