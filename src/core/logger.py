"""Centralized logging configuration for the Crypto Trading Agent"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import config


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class TradingLogger:
    """Enhanced logging system for trading operations"""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logging(cls, force_reset=False):
        """Setup global logging configuration"""
        if cls._initialized and not force_reset:
            return
        
        # Reset if forced
        if force_reset:
            cls._loggers.clear()
            cls._initialized = False
        
        # Create logs directory
        log_dir = Path(config.logging.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.logging.log_level.upper()))
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """Get or create a logger with specified configuration"""
        if not cls._initialized:
            cls.setup_logging()
        
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.logging.log_level.upper()))
        
        # Avoid duplicate handlers
        if logger.handlers:
            cls._loggers[name] = logger
            return logger
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        colored_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with rotation (if enabled)
        if config.logging.log_to_file:
            log_file_path = log_file or config.logging.log_file_path
            
            try:
                # Use RotatingFileHandler instead of TimedRotatingFileHandler
                # to avoid file locking issues with multiple processes
                file_handler = RotatingFileHandler(
                    log_file_path,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=config.logging.log_backup_count,
                    delay=False  # Open file immediately
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(detailed_formatter)
                logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                # If file handler fails, continue with console logging only
                print(f"Warning: Could not setup file logging: {e}")
                print("Continuing with console logging only.")
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Get a logger instance"""
    return TradingLogger.get_logger(name, log_file)


# Create specialized loggers
main_logger = get_logger("main")
strategy_logger = get_logger("strategy")
market_data_logger = get_logger("market_data")
risk_logger = get_logger("risk_management")
trade_logger = get_logger("trade_tracker")
api_logger = get_logger("api")
performance_logger = get_logger("performance")


class TradeLoggerMixin:
    """Mixin class to add logging capabilities to trading components"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__.lower())
    
    def log_trade(self, action: str, symbol: str, price: float, quantity: float, confidence: float):
        """Log trade execution"""
        self.logger.info(
            f"Trade executed: {action} {quantity:.6f} {symbol} @ {price:.6f} (confidence: {confidence:.2f})"
        )
    
    def log_analysis(self, symbol: str, action: str, confidence: float, details: dict):
        """Log analysis results"""
        self.logger.info(
            f"Analysis: {symbol} -> {action} (confidence: {confidence:.2f}) - {details}"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context"""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        self.logger.info(f"Performance metrics: {metrics}")