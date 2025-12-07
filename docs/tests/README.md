# Testing Guide

## Overview

This directory contains test documentation and test plans for the ai-signal-chart project.

## Test Files

### Automated Tests
- **Location**: `backend/tests/`
- **Framework**: pytest

### Test Documentation
1. **`explainability_test_plan.md`** - Comprehensive test plan for Explainability Layer
   - Architecture documentation
   - Test specifications (Indicator Integrity, Rule Trigger Consistency, Confidence Coherence)
   - Browser verification procedures
   - Future work guidelines

2. **`SUMMARY.md`** - Quick reference for running tests and understanding test coverage

## Running Tests

### Quick Start

From the project root:

```bash
# Activate virtual environment (if using)
source .venv/bin/activate

# Run all tests
pytest backend/tests/ -v

# Run specific test file
pytest backend/tests/test_explainability.py -v

# Run with coverage
pytest backend/tests/ --cov=backend --cov-report=html
```

### Test Categories

Currently implemented:
- ✅ **Explainability Tests** - 10 test cases covering 3 strategies

Planned:
- ⏳ E2E UI tests (Playwright)
- ⏳ Performance/stress tests
- ⏳ Integration tests for data feeds

## Test Structure

```
ai-signal-chart/
├── conftest.py              # Pytest configuration (adds project root to sys.path)
├── pytest.ini               # Pytest settings
├── backend/
│   ├── __init__.py          # Makes backend a package
│   └── tests/
│       ├── __init__.py      # Makes tests a package
│       └── test_explainability.py  # Explainability tests
└── docs/
    └── tests/
        ├── README.md        # This file
        ├── SUMMARY.md       # Test execution quick reference
        └── explainability_test_plan.md  # Detailed test plan
```

## Import Convention

Tests use **absolute imports**:
```python
from backend.strategies.ma_cross import MACrossStrategy
from backend.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
```

This ensures tests can run from the project root without sys.path hacks.

## Adding New Tests

When adding a new test file:

1. Create file in `backend/tests/` with `test_` prefix
2. Use absolute imports (`from backend.xxx import ...`)
3. Run from project root: `pytest backend/tests/test_yourfile.py -v`
4. Document the test in this directory if needed

## Continuous Integration

To integrate with CI/CD (e.g., GitHub Actions):

```yaml
- name: Run Tests
  run: |
    pip install -r requirements.txt
    pytest backend/tests/ -v --cov=backend --cov-report=xml
```

## Troubleshooting

**ModuleNotFoundError: No module named 'backend'**
- Ensure you're running pytest from the **project root**
- Check that `conftest.py` exists at project root
- Verify virtual environment is activated

**Tests not discovered**
- Check file naming: must start with `test_`
- Check function naming: must start with `test_`
- Check class naming: must start with `Test`

For more details, see:
- [pytest documentation](https://docs.pytest.org/)
- `SUMMARY.md` - Quick reference
- `explainability_test_plan.md` - Detailed test documentation
