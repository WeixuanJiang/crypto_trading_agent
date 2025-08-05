"""Trading Module

This module provides trading execution and tracking capabilities including:
- Trade execution and order management
- Trade history tracking and analytics
- P&L calculation and performance metrics
- AWS integration for enhanced tracking
"""

from .tracker import TradeTracker, Trade
from .aws_tracker import EnhancedTradeTracker, create_trade_tracker

__all__ = [
    'TradeTracker',
    'Trade',
    'EnhancedTradeTracker',
    'create_trade_tracker'
]