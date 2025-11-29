# Strategy Lab: Future Extensibility Notes

## Current Implementation (v0.1)
Strategy Lab v0.1 implements a batch optimization framework for the **MA Cross** strategy.
It allows users to:
1. Define a study with multiple symbols
2. Set a parameter search grid (Short/Long MA windows)
3. Run optimization in batch
4. View a leaderboard of best parameters per symbol

## Future Roadmap & Extension Points

### 1. New Strategy Types
Currently, `strategy_type` is hardcoded to `ma_cross` in the UI and partially in the backend logic.
To add new strategies (e.g., RSI, Bollinger Bands, MACD):

- **Backend**:
  - Update `GridSearchOptimizer.optimize` to handle new strategy classes.
  - Update `StrategyLabBatchRequest` model to include strategy-specific parameter ranges (e.g., `rsi_period_min`, `bb_std_dev`).
- **Frontend**:
  - Update `StrategyLabBatchRequest` interface.
  - Add dynamic form fields in the Strategy Lab UI based on selected strategy.

### 2. Automated Strategy Generation
The ultimate goal is to automatically generate trading rules.
- **Genetic Algorithms (GA)**:
  - Instead of a fixed grid search, use GA to evolve strategy parameters and even rule combinations.
  - Libraries like `DEAP` or `PyGAD` can be integrated into `backend/optimizer.py`.
- **Rule Trees**:
  - Represent strategies as decision trees (e.g., `IF RSI < 30 AND Price > MA(200) THEN BUY`).
  - Use Genetic Programming to evolve these trees.

### 3. Meta-Strategy & Portfolio Optimization
- **Portfolio Optimization**:
  - Instead of optimizing per symbol, optimize a portfolio of symbols to maximize Sharpe Ratio while minimizing correlation.
  - Use `PyPortfolioOpt` or similar libraries.
- **Ensemble Methods**:
  - Combine multiple strategies (e.g., MA Cross + RSI) into a single meta-strategy.
  - Use voting or weighted average mechanisms for signal generation.

### 4. Advanced Optimization Techniques
- **Bayesian Optimization**:
  - Use `Optuna` to efficiently search large parameter spaces without exhaustive grid search.
  - Already partially prepared in `backend/optimizer.py` structure.
- **Walk-Forward Analysis**:
  - Implement walk-forward validation to avoid overfitting.
  - Split data into multiple train/test periods.

## Architecture for v0.2
Refactor `GridSearchOptimizer` into an abstract base class `BaseOptimizer` to support multiple optimization engines (Grid, Bayesian, Genetic).
Create a `StrategyFactory` to dynamically instantiate strategies based on configuration.
