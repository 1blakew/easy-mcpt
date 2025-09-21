"""
Core Framework Components
========================

Contains the fundamental building blocks of the MCPT framework:
- bar_permute: Core permutation functionality
- strategy_base: Abstract base classes for strategies
- mcpt_runner: High-level testing interface
"""

from .bar_permute import get_permutation
from .strategy_base import TradingStrategy, WalkForwardStrategy, StrategyResult, MCPTTester
from .mcpt_runner import MCPTRunner

__all__ = [
    'get_permutation',
    'TradingStrategy',
    'WalkForwardStrategy',
    'StrategyResult',
    'MCPTTester',
    'MCPTRunner'
]