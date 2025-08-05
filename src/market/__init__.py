"""Market Data and Analysis Module

This module provides comprehensive market data management and technical analysis including:
- Real-time and historical market data fetching
- Advanced technical indicators and oscillators
- Market sentiment analysis and pattern recognition
- Price action analysis and trend detection
"""

from .data_manager import MarketDataManager
from .analysis import MarketAnalyzer
from .indicators import TechnicalIndicators
from .data import MarketDataManager as DataManager

__all__ = [
    'MarketDataManager',
    'MarketAnalyzer', 
    'TechnicalIndicators',
    'DataManager'
]