"""
Centralized logging configuration for the Crypto Trading Agent
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logger(name: str, level: str = None) -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Get configuration from environment variables
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')
    
    log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    log_file_path = os.getenv('LOG_FILE_PATH', 'trading_agent.log')
    max_log_file_size_mb = int(os.getenv('MAX_LOG_FILE_SIZE_MB', '10'))
    log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    # Create logs directory if it doesn't exist and logging to file is enabled
    if log_to_file:
        log_dir = os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with rotation (if enabled)
    if log_to_file:
        # Use the configured log file path or create one with date
        if log_file_path == 'trading_agent.log':
            log_file_path = os.path.join("logs", f"trading_agent_{datetime.now().strftime('%Y%m%d')}.log")
        
        try:
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=max_log_file_size_mb * 1024 * 1024,  # Convert MB to bytes
                backupCount=log_backup_count,
                delay=True  # Delay file opening until first write
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # If file handler fails, continue with console logging only
            print(f"Warning: Could not setup file logging for {name}: {e}")
            print("Continuing with console logging only.")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create module-specific loggers
main_logger = setup_logger("main")
strategy_logger = setup_logger("strategy")
market_data_logger = setup_logger("market_data")
risk_logger = setup_logger("risk_management")
trade_logger = setup_logger("trade_tracker")