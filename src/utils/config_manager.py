"""
Centralized configuration management for the Crypto Trading Agent

This module provides a single source of truth for all configuration settings,
with validation, type checking, and environment-specific overrides.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json

@dataclass
class TradingConfig:
    """Trading-specific configuration"""
    trading_pairs: List[str] = field(default_factory=lambda: os.getenv('TRADING_PAIRS', 'BTC-USDT,ETH-USDT,ADA-USDT,DOT-USDT').split(','))
    default_interval_minutes: int = int(os.getenv('TRADING_INTERVAL_MINUTES', '60'))
    min_interval_minutes: int = int(os.getenv('MIN_INTERVAL_MINUTES', '5'))
    max_interval_minutes: int = int(os.getenv('MAX_INTERVAL_MINUTES', '1440'))
    min_confidence_threshold: float = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
    min_account_balance: float = float(os.getenv('MINIMUM_BALANCE', '10.0'))
    enable_live_trading: bool = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
    historical_data_limit: int = int(os.getenv('HISTORICAL_DATA_LIMIT', '500'))
    kline_interval: str = os.getenv('KLINE_INTERVAL', '1hour')
    max_daily_trades: int = int(os.getenv('MAX_DAILY_TRADES', '1000'))

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size_percent: float = float(os.getenv('MAX_POSITION_SIZE', '0.02')) * 100
    position_sizing_method: str = os.getenv('POSITION_SIZING_METHOD', 'fixed_percent')
    max_portfolio_exposure_percent: float = float(os.getenv('MAX_PORTFOLIO_EXPOSURE', '0.20')) * 100
    max_open_positions: int = int(os.getenv('MAX_OPEN_POSITIONS', '5'))
    max_correlation_threshold: float = float(os.getenv('MAX_CORRELATION_THRESHOLD', '0.7'))
    default_stop_loss_percent: float = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.05')) * 100
    use_atr_stop_loss: bool = os.getenv('USE_ATR_STOP_LOSS', 'true').lower() == 'true'
    atr_multiplier: float = float(os.getenv('ATR_MULTIPLIER', '2.0'))
    risk_reward_ratio: float = float(os.getenv('RISK_REWARD_RATIO', '2.0'))

@dataclass
class TechnicalConfig:
    """Technical analysis configuration"""
    rsi_period: int = int(os.getenv('RSI_PERIOD', '14'))
    rsi_short_period: int = int(os.getenv('RSI_SHORT_PERIOD', '7'))
    rsi_oversold_strong: int = int(os.getenv('RSI_OVERSOLD_STRONG', '25'))
    rsi_oversold_weak: int = int(os.getenv('RSI_OVERSOLD_WEAK', '35'))
    rsi_overbought_weak: int = int(os.getenv('RSI_OVERBOUGHT_WEAK', '65'))
    rsi_overbought_strong: int = int(os.getenv('RSI_OVERBOUGHT_STRONG', '75'))
    macd_fast: int = int(os.getenv('MACD_FAST', '12'))
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std_dev: int = 2
    bb_min_width: float = 0.04
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    atr_period: int = 14
    sma_20_period: int = 20
    sma_50_period: int = 50
    sma_200_period: int = 200

@dataclass
class LLMConfig:
    """LLM configuration"""
    model_id: str = 'us.anthropic.claude-3-5-haiku-20241022-v1:0'
    max_tokens: int = 1000
    temperature: float = 0.3
    fast_mode: bool = True
    aws_region: str = 'us-east-1'
    cross_region_inference: bool = True

@dataclass
class APIConfig:
    """API configuration"""
    kucoin_api_key: str = ''
    kucoin_api_secret: str = ''
    kucoin_api_passphrase: str = ''

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        
        # Initialize configurations
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.technical = TechnicalConfig()
        self.llm = LLMConfig()
        self.api = APIConfig()
        
        # Load from environment variables
        self._load_from_env()
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        
        # Validate configuration
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Trading config
        if os.getenv('TRADING_PAIRS'):
            self.trading.trading_pairs = os.getenv('TRADING_PAIRS').split(',')
        self.trading.min_confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', self.trading.min_confidence_threshold))
        self.trading.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', self.trading.max_daily_trades))
        self.trading.trading_interval_minutes = int(os.getenv('TRADING_INTERVAL_MINUTES', self.trading.default_interval_minutes))
        self.trading.min_account_balance = float(os.getenv('MINIMUM_BALANCE', self.trading.min_account_balance))
        
        # Risk config
        self.risk.max_position_size_percent = float(os.getenv('MAX_POSITION_SIZE', self.risk.max_position_size_percent)) * 100
        self.risk.max_portfolio_exposure_percent = float(os.getenv('MAX_PORTFOLIO_RISK', self.risk.max_portfolio_exposure_percent)) * 100
        self.risk.default_stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENTAGE', self.risk.default_stop_loss_percent)) * 100
        self.risk.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', self.risk.risk_reward_ratio))
        self.risk.atr_multiplier = float(os.getenv('ATR_MULTIPLIER', self.risk.atr_multiplier))
        
        # Technical config
        self.technical.rsi_period = int(os.getenv('RSI_PERIOD', self.technical.rsi_period))
        self.technical.rsi_short_period = int(os.getenv('RSI_SHORT_PERIOD', self.technical.rsi_short_period))
        self.technical.macd_fast = int(os.getenv('MACD_FAST', self.technical.macd_fast))
        self.technical.macd_slow = int(os.getenv('MACD_SLOW', self.technical.macd_slow))
        self.technical.macd_signal = int(os.getenv('MACD_SIGNAL', self.technical.macd_signal))
        
        # LLM config
        self.llm.model_id = os.getenv('LLM_MODEL_ID', self.llm.model_id)
        self.llm.max_tokens = int(os.getenv('LLM_MAX_TOKENS', self.llm.max_tokens))
        self.llm.temperature = float(os.getenv('LLM_TEMPERATURE', self.llm.temperature))
        self.llm.fast_mode = os.getenv('FAST_MODE', str(self.llm.fast_mode)).lower() == 'true'
        self.llm.aws_region = os.getenv('AWS_REGION', self.llm.aws_region)
        self.llm.cross_region_inference = os.getenv('AWS_CROSS_REGION_INFERENCE', str(self.llm.cross_region_inference)).lower() == 'true'
        
        # API config
        self.api.kucoin_api_key = os.getenv('KUCOIN_API_KEY', '')
        self.api.kucoin_api_secret = os.getenv('KUCOIN_API_SECRET', '')
        self.api.kucoin_api_passphrase = os.getenv('KUCOIN_API_PASSPHRASE', '')
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            if 'trading' in config_data:
                for key, value in config_data['trading'].items():
                    if hasattr(self.trading, key):
                        setattr(self.trading, key, value)
            
            if 'risk' in config_data:
                for key, value in config_data['risk'].items():
                    if hasattr(self.risk, key):
                        setattr(self.risk, key, value)
            
            # Similar for other config sections...
            
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate API credentials
        if not self.api.kucoin_api_key:
            errors.append("KUCOIN_API_KEY is required")
        if not self.api.kucoin_api_secret:
            errors.append("KUCOIN_API_SECRET is required")
        if not self.api.kucoin_api_passphrase:
            errors.append("KUCOIN_API_PASSPHRASE is required")
        
        # Validate trading config
        if self.trading.min_confidence_threshold < 0 or self.trading.min_confidence_threshold > 1:
            errors.append("MIN_CONFIDENCE_THRESHOLD must be between 0 and 1")
        
        if self.trading.min_account_balance <= 0:
            errors.append("MINIMUM_BALANCE must be positive")
        
        # Validate risk config
        if self.risk.max_position_size_percent <= 0 or self.risk.max_position_size_percent > 100:
            errors.append("MAX_POSITION_SIZE must be between 0 and 100%")
        
        if self.risk.default_stop_loss_percent <= 0 or self.risk.default_stop_loss_percent > 50:
            errors.append("STOP_LOSS_PERCENTAGE must be between 0 and 50%")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def get_trading_pairs(self) -> List[str]:
        """Get list of trading pairs"""
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
            'llm': self.llm.__dict__
            # Note: API credentials are not exported for security
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration exported to {filename}")

# Global configuration instance
config = ConfigManager()