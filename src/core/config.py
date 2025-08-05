"""Centralized configuration management for the Crypto Trading Agent"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    trading_pairs: List[str] = field(default_factory=lambda: ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT'])
    default_interval_minutes: int = 60
    min_interval_minutes: int = 5
    max_interval_minutes: int = 1440
    min_confidence_threshold: float = 0.6
    min_account_balance: float = 10.0
    min_trade_size_usdt: float = 1.0
    enable_live_trading: bool = False
    historical_data_limit: int = 500
    kline_interval: str = '1hour'
    max_daily_trades: int = 1000


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size_percent: float = 2.0
    position_sizing_method: str = 'fixed_percent'
    max_portfolio_exposure_percent: float = 20.0
    max_open_positions: int = 5
    max_correlation_threshold: float = 0.7
    default_stop_loss_percent: float = 5.0
    use_atr_stop_loss: bool = True
    atr_multiplier: float = 2.0
    risk_reward_ratio: float = 2.0
    use_trailing_stop: bool = False
    trailing_stop_percent: float = 3.0
    var_confidence_level: float = 0.95
    max_drawdown_threshold: float = 15.0


@dataclass
class TechnicalConfig:
    """Technical analysis configuration"""
    rsi_period: int = 14
    rsi_short_period: int = 7
    rsi_oversold_strong: int = 25
    rsi_oversold_weak: int = 35
    rsi_overbought_weak: int = 65
    rsi_overbought_strong: int = 75
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std_dev: int = 2
    bb_min_width: float = 0.04
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    atr_period: int = 14
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])
    williams_r_period: int = 14
    cci_period: int = 20
    mfi_period: int = 14
    adx_period: int = 14
    aroon_period: int = 14


@dataclass
class LLMConfig:
    """LLM configuration"""
    model_id: str = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
    max_tokens: int = 1000
    temperature: float = 0.3
    fast_mode: bool = True
    aws_region: str = 'us-east-1'
    cross_region_inference: bool = True
    enable_sentiment_analysis: bool = True
    enable_news_analysis: bool = True
    news_lookback_hours: int = 24


@dataclass
class APIConfig:
    """API configuration"""
    kucoin_api_key: str = ''
    kucoin_api_secret: str = ''
    kucoin_api_passphrase: str = ''
    sandbox_mode: bool = False
    max_retries: int = 5
    retry_delay_seconds: float = 2.0
    backoff_factor: float = 2.0
    connection_timeout: int = 30
    read_timeout: int = 30
    requests_per_second: int = 10
    requests_per_minute: int = 600


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_file_path: str = 'logs/trading_agent.log'
    max_log_file_size_mb: int = 10
    log_backup_count: int = 5
    log_trades: bool = True
    log_analysis: bool = True
    log_errors: bool = True
    log_performance: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str = 'data/trading_agent.db'
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backup_files: int = 7
    connection_pool_size: int = 5


class Config:
    """Centralized configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None, load_env: bool = True):
        if load_env:
            load_dotenv()
        
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.technical = TechnicalConfig()
        self.llm = LLMConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.database = DatabaseConfig()
        
        self._load_from_env()
        
        if config_file:
            self._load_from_file(config_file)
        else:
            # Try to load from saved settings file
            settings_file = 'config/settings.json'
            if os.path.exists(settings_file):
                self._load_from_file(settings_file)
        
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Trading configuration
        if os.getenv('TRADING_PAIRS'):
            self.trading.trading_pairs = os.getenv('TRADING_PAIRS').split(',')
        self.trading.default_interval_minutes = int(os.getenv('TRADING_INTERVAL_MINUTES', '60'))
        self.trading.min_confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
        self.trading.min_account_balance = float(os.getenv('MINIMUM_BALANCE', '10.0'))
        self.trading.min_trade_size_usdt = float(os.getenv('MIN_TRADE_SIZE_USDT', '0.0'))
        self.trading.enable_live_trading = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
        self.trading.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '1000'))
        
        # Risk configuration
        self.risk.max_position_size_percent = float(os.getenv('MAX_POSITION_SIZE', '0.02')) * 100
        self.risk.max_portfolio_exposure_percent = float(os.getenv('MAX_PORTFOLIO_EXPOSURE', '0.20')) * 100
        self.risk.default_stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.05')) * 100
        self.risk.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', '2.0'))
        
        # Technical configuration
        self.technical.rsi_period = int(os.getenv('RSI_PERIOD', '14'))
        self.technical.macd_fast = int(os.getenv('MACD_FAST', '12'))
        self.technical.macd_slow = int(os.getenv('MACD_SLOW', '26'))
        self.technical.bb_period = int(os.getenv('BB_PERIOD', '20'))
        
        # LLM configuration
        self.llm.model_id = os.getenv('LLM_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
        self.llm.fast_mode = os.getenv('FAST_MODE', 'true').lower() == 'true'
        self.llm.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        # API configuration
        self.api.kucoin_api_key = os.getenv('KUCOIN_API_KEY', '')
        self.api.kucoin_api_secret = os.getenv('KUCOIN_API_SECRET', '')
        self.api.kucoin_api_passphrase = os.getenv('KUCOIN_API_PASSPHRASE', '')
        
        # Logging configuration
        self.logging.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.logging.log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
        self.logging.log_file_path = os.getenv('LOG_FILE_PATH', 'logs/trading_agent.log')
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            for section, data in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
        
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate API credentials for live trading
        if self.trading.enable_live_trading:
            if not all([self.api.kucoin_api_key, self.api.kucoin_api_secret, self.api.kucoin_api_passphrase]):
                raise ConfigurationError("API credentials required for live trading")
        
        # Validate trading pairs
        if not self.trading.trading_pairs:
            raise ConfigurationError("At least one trading pair must be configured")
        
        # Validate risk parameters
        if self.risk.max_position_size_percent <= 0 or self.risk.max_position_size_percent > 100:
            raise ConfigurationError("Max position size must be between 0 and 100 percent")
        
        if self.risk.default_stop_loss_percent <= 0 or self.risk.default_stop_loss_percent > 50:
            raise ConfigurationError("Stop loss percentage must be between 0 and 50 percent")
        
        # Validate confidence threshold
        if self.trading.min_confidence_threshold < 0 or self.trading.min_confidence_threshold > 1:
            raise ConfigurationError("Confidence threshold must be between 0 and 1")
    
    def get_trading_pairs(self) -> List[str]:
        """Get configured trading pairs"""
        return self.trading.trading_pairs
    
    def get_api_credentials(self) -> tuple:
        """Get API credentials"""
        return (
            self.api.kucoin_api_key,
            self.api.kucoin_api_secret,
            self.api.kucoin_api_passphrase
        )
    
    def is_live_trading_enabled(self) -> bool:
        """Check if live trading is enabled"""
        return self.trading.enable_live_trading
    
    def export_config(self, filename: str):
        """Export current configuration to file"""
        config_dict = {
            'trading': self.trading.__dict__,
            'risk': self.risk.__dict__,
            'technical': self.technical.__dict__,
            'llm': self.llm.__dict__,
            'api': {k: v for k, v in self.api.__dict__.items() if 'key' not in k.lower()},  # Exclude sensitive data
            'logging': self.logging.__dict__,
            'database': self.database.__dict__
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def reload_config(self, config_file: str = 'config/settings.json'):
        """Reload configuration from file"""
        try:
            if os.path.exists(config_file):
                self._load_from_file(config_file)
                self._validate_config()
                return True
            else:
                raise ConfigurationError(f"Configuration file not found: {config_file}")
        except Exception as e:
            raise ConfigurationError(f"Failed to reload configuration: {e}")


# Global configuration instance
config = Config()