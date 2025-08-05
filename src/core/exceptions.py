"""Custom exceptions for the Crypto Trading Agent"""

from typing import Optional, Dict, Any


class TradingAgentError(Exception):
    """Base exception for all trading agent errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ValidationError(TradingAgentError):
    """Raised when input validation fails"""
    pass


class APIError(TradingAgentError):
    """Raised when API calls fail"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message, error_code="API_ERROR")
        self.status_code = status_code
        self.response_data = response_data or {}


class ConfigurationError(TradingAgentError):
    """Raised when configuration is invalid"""
    pass


class InsufficientFundsError(TradingAgentError):
    """Raised when account has insufficient funds for trading"""
    pass


class RiskManagementError(TradingAgentError):
    """Raised when risk management rules are violated"""
    pass


class MarketDataError(TradingAgentError):
    """Raised when market data is unavailable or invalid"""
    pass


class StrategyError(TradingAgentError):
    """Raised when trading strategy encounters an error"""
    pass


class DatabaseError(TradingAgentError):
    """Raised when database operations fail"""
    pass