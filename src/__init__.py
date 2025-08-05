"""Crypto Trading Agent - Production Ready Trading System

A sophisticated cryptocurrency trading agent that combines Large Language Model (LLM) 
analysis with advanced quantitative trading methods for automated trading on KuCoin exchange.

Main Features:
- LLM-powered trading strategies with AWS Bedrock integration
- Advanced technical analysis and market sentiment analysis
- Comprehensive risk management and portfolio optimization
- Real-time performance monitoring and alerting
- Modular architecture for easy customization and extension
"""

# Core imports
from .core import Config, get_logger, TradingAgentError, ValidationError, APIError
from .data import DatabaseManager, Trade, Position, MarketData, PerformanceMetrics
from .market import DataManager
from .strategy import LLMStrategy, FastLLMStrategy, FullLLMStrategy, LLMTradingStrategy
from .llm import BedrockLLMClient
from .risk import RiskManager
from .portfolio import PortfolioManager, Position as PortfolioPosition
from .trading import TradeTracker, Trade as TradeRecord, EnhancedTradeTracker, create_trade_tracker
from .monitoring import PerformanceMonitor, PerformanceMetrics as MonitoringMetrics, performance_monitor
from .notifications import NotificationService
from .utils import InputValidator, ConfigManager, config

__version__ = "2.0.0"
__author__ = "Crypto Trading Agent Team"
__description__ = "A sophisticated cryptocurrency trading agent with LLM integration"

__all__ = [
    # Core
    'Config',
    'get_logger',
    'TradingAgentError',
    'ValidationError',
    'APIError',
    
    # Data
    'DatabaseManager', 'Trade', 'Position', 'MarketData', 'PerformanceMetrics',
    
    # Market
    'DataManager',
    
    # Strategy
    'LLMStrategy', 'FastLLMStrategy', 'FullLLMStrategy', 'LLMTradingStrategy',
    
    # LLM
    'BedrockLLMClient',
    
    # Risk
    'RiskManager',
    
    # Portfolio
    'PortfolioManager', 'PortfolioPosition',
    
    # Trading
    'TradeTracker', 'TradeRecord', 'EnhancedTradeTracker', 'create_trade_tracker',
    
    # Monitoring
    'PerformanceMonitor', 'MonitoringMetrics', 'performance_monitor',
    
    # Notifications
    'NotificationService',
    
    # Utils
    'InputValidator', 'ConfigManager', 'config'
]