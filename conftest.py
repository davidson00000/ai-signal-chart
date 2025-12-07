"""
Pytest configuration for ai-signal-chart

This conftest.py ensures that the project root is in sys.path,
allowing tests to import modules from 'backend' package.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
