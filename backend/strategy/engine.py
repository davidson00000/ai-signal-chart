import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Any, Optional
from backend.strategy.models import JsonStrategySpec, RuleGroup, RuleCondition, ValueRef, IndicatorSpec
from backend.backtester import BacktestEngine
from backend.models.backtest import BacktestRequest

class JsonStrategyEngine:
    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame, indicators: List[IndicatorSpec]) -> pd.DataFrame:
        """
        Calculate indicators and add them to the DataFrame.
        """
        # Ensure we work on a copy to avoid side effects if reused
        df = df.copy()
        
        for ind in indicators:
            try:
                if ind.type == "sma":
                    if not ind.period:
                        continue
                    # pandas_ta SMA
                    sma = ta.sma(df[ind.source], length=ind.period)
                    df[ind.id] = sma
                    
                elif ind.type == "ema":
                    if not ind.period:
                        continue
                    ema = ta.ema(df[ind.source], length=ind.period)
                    df[ind.id] = ema
                    
                elif ind.type == "rsi":
                    if not ind.period:
                        continue
                    rsi = ta.rsi(df[ind.source], length=ind.period)
                    df[ind.id] = rsi
                    
                elif ind.type == "bollinger":
                    if not ind.period or not ind.std_dev:
                        continue
                    # bbands returns a DataFrame with lower, mid, upper columns
                    bb = ta.bbands(df[ind.source], length=ind.period, std=ind.std_dev)
                    if bb is not None:
                        # Map pandas_ta output names to our ID convention
                        # pandas_ta default names: BBL_length_std, BBM_length_std, BBU_length_std
                        # We might want to standardize: id_lower, id_mid, id_upper
                        # For simplicity in v0.1, let's just assume simple access or single value?
                        # Bollinger is complex because it returns 3 values.
                        # Let's flatten it: {id}_lower, {id}_mid, {id}_upper
                        df[f"{ind.id}_lower"] = bb.iloc[:, 0]
                        df[f"{ind.id}_mid"] = bb.iloc[:, 1]
                        df[f"{ind.id}_upper"] = bb.iloc[:, 2]
                        
            except Exception as e:
                print(f"Error calculating indicator {ind.id}: {e}")
                
        return df

    def _get_value(self, row: pd.Series, ref: ValueRef) -> float:
        """
        Resolve a ValueRef to a float value from the row or constant.
        """
        if ref.value is not None:
            return ref.value
        if ref.ref is not None:
            if ref.ref in row:
                return float(row[ref.ref])
            else:
                # Fallback or error? For now return NaN or 0
                return 0.0
        return 0.0

    def evaluate_condition(self, row: pd.Series, condition: RuleCondition) -> bool:
        """
        Evaluate a single comparison condition.
        """
        left_val = self._get_value(row, condition.left)
        right_val = self._get_value(row, condition.right)
        
        op = condition.op
        if op == "==":
            return left_val == right_val
        elif op == "!=":
            return left_val != right_val
        elif op == ">":
            return left_val > right_val
        elif op == ">=":
            return left_val >= right_val
        elif op == "<":
            return left_val < right_val
        elif op == "<=":
            return left_val <= right_val
        return False

    def evaluate_rule_group(self, row: pd.Series, group: RuleGroup) -> bool:
        """
        Recursively evaluate a RuleGroup.
        """
        # If group has 'all', all conditions must be true (AND)
        if group.all:
            for item in group.all:
                if isinstance(item, RuleGroup):
                    if not self.evaluate_rule_group(row, item):
                        return False
                elif isinstance(item, RuleCondition):
                    if not self.evaluate_condition(row, item):
                        return False
            return True
            
        # If group has 'any', at least one condition must be true (OR)
        if group.any:
            for item in group.any:
                if isinstance(item, RuleGroup):
                    if self.evaluate_rule_group(row, item):
                        return True
                elif isinstance(item, RuleCondition):
                    if self.evaluate_condition(row, item):
                        return True
            return False
            
        # Empty group is considered True? or False? 
        # Let's say True to be safe, or maybe it shouldn't happen.
        return True

    def generate_signals(self, df: pd.DataFrame, strategy: JsonStrategySpec) -> pd.DataFrame:
        """
        Generate 'signal' column (1 for Buy, -1 for Sell, 0 for Hold)
        based on entry and exit rules.
        """
        df['signal'] = 0
        
        # We need to iterate row by row or use apply. 
        # apply is slower but easier for complex recursive logic.
        # For v0.1 performance is secondary to correctness.
        
        def apply_rules(row):
            # Check Entry
            is_entry = False
            for rule in strategy.entry_rules:
                if self.evaluate_rule_group(row, rule):
                    is_entry = True
                    break
            
            # Check Exit
            is_exit = False
            for rule in strategy.exit_rules:
                if self.evaluate_rule_group(row, rule):
                    is_exit = True
                    break
            
            if is_entry:
                return 1
            elif is_exit:
                return -1
            return 0

        # Optimization: Only apply to rows where we have enough data (after max period)
        # But for now, just apply to all
        df['signal_raw'] = df.apply(apply_rules, axis=1)
        
        # Logic for position holding
        # If signal_raw is 1, we want to be Long.
        # If signal_raw is -1, we want to be Flat (or Short if supported).
        # If signal_raw is 0, we keep previous state? 
        # Usually 'entry rule' triggers a new position, 'exit rule' closes it.
        # So:
        #   State 0 (Flat) -> Entry True -> State 1 (Long)
        #   State 1 (Long) -> Exit True -> State 0 (Flat)
        #   State 1 (Long) -> Entry True -> State 1 (Long) [Re-entry or hold]
        
        # Let's implement a simple state machine loop for signals
        signals = []
        position = 0 # 0: Flat, 1: Long, -1: Short
        
        for i, row in df.iterrows():
            raw_sig = row['signal_raw']
            
            if position == 0:
                if raw_sig == 1:
                    position = 1
                    signals.append(1) # Buy Signal
                elif raw_sig == -1 and strategy.position.direction == "both":
                     position = -1
                     signals.append(-1) # Sell Signal (Short)
                else:
                    signals.append(0)
            elif position == 1:
                if raw_sig == -1:
                    position = 0
                    signals.append(-1) # Sell Signal (Close Long)
                else:
                    signals.append(0) # Hold
            elif position == -1:
                if raw_sig == 1: # Exit Short or Flip to Long?
                    # Simple: Exit Short
                    position = 0
                    signals.append(1) # Buy Signal (Close Short)
                else:
                    signals.append(0)
                    
        df['signal'] = signals
        return df

    def run(self, df: pd.DataFrame, strategy: JsonStrategySpec, 
            initial_capital: float, commission_rate: float, position_size: float) -> Dict[str, Any]:
        """
        Run the strategy backtest.
        """
        # 1. Calculate Indicators
        df = self.calculate_indicators(df, strategy.indicators)
        
        # 2. Generate Signals (This part is actually handled inside BacktestEngine usually, 
        # but BacktestEngine expects a Strategy Class with a `generate_signals` method.
        # We can either:
        # A) Create a dynamic Strategy class adapter
        # B) Pre-calculate signals column and use a "SignalColumnStrategy"
        
        # Let's go with B (Pre-calculate signals) and make a simple Adapter Strategy
        df = self.generate_signals(df, strategy)
        
        # 3. Run Backtest using existing Engine
        # We need a dummy strategy object that BacktestEngine accepts.
        # BacktestEngine.run_backtest(df, strategy_instance)
        # The strategy_instance needs a generate_signals(df) method.
        
        class PreCalculatedSignalStrategy:
            def generate_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
                # Signals are already in 'signal' column, just return df
                return dataframe

        adapter_strategy = PreCalculatedSignalStrategy()
        
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            position_size=position_size
        )
        
        result = engine.run_backtest(df, adapter_strategy)
        return result
