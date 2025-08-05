"""Portfolio Management Module

This module provides portfolio management capabilities including:
- Balance tracking and position management
- Portfolio-aware trading decisions
- Asset allocation and rebalancing
- Performance tracking and analytics
"""

from .manager import PortfolioManager, Position

__all__ = [
    'PortfolioManager',
    'Position'
]