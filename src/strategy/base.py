"""Base strategy class and interfaces"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

from ..core.logger import get_logger, TradeLoggerMixin
from ..core.exceptions import StrategyError, ValidationError
from ..data.models import MarketData, AnalysisResult


class StrategyType(Enum):
    """Strategy type classification"""
    TECHNICAL = "technical"
    LLM = "llm"
    HYBRID = "hybrid"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"


class SignalStrength(Enum):
    """Signal strength classification"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


@dataclass
class StrategySignal:
    """Trading signal from strategy"""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-100
    strength: SignalStrength
    entry_price: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    position_size: Optional[float]  # Percentage of portfolio
    reasoning: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    valid_until: Optional[datetime]


@dataclass
class StrategyResult:
    """Complete strategy analysis result"""
    strategy_name: str
    strategy_type: StrategyType
    symbol: str
    timestamp: datetime
    signal: StrategySignal
    analysis: AnalysisResult
    performance_metrics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    execution_priority: int  # 1-10, higher = more urgent


class StrategyConfig:
    """Base configuration for strategies"""
    
    def __init__(self, **kwargs):
        # Common parameters
        self.max_position_size = kwargs.get('max_position_size', 0.1)  # 10% max
        self.min_confidence = kwargs.get('min_confidence', 60.0)
        self.risk_tolerance = kwargs.get('risk_tolerance', 'medium')
        self.time_horizon = kwargs.get('time_horizon', 'short')  # short, medium, long
        
        # Risk management
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = kwargs.get('take_profit_pct', 0.04)  # 4%
        self.max_drawdown = kwargs.get('max_drawdown', 0.05)  # 5%
        
        # Signal filtering
        self.min_volume_ratio = kwargs.get('min_volume_ratio', 0.5)
        self.max_volatility = kwargs.get('max_volatility', 0.1)  # 10%
        
        # Timing
        self.signal_timeout_minutes = kwargs.get('signal_timeout_minutes', 60)
        self.cooldown_minutes = kwargs.get('cooldown_minutes', 30)
        
        # Custom parameters
        self.custom_params = {k: v for k, v in kwargs.items() 
                            if k not in self.__dict__}
    
    def get(self, key: str, default=None):
        """Get configuration parameter"""
        return getattr(self, key, self.custom_params.get(key, default))
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_params[key] = value


class BaseStrategy(ABC, TradeLoggerMixin):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, strategy_type: StrategyType, config: Optional[StrategyConfig] = None):
        super().__init__()
        self.name = name
        self.strategy_type = strategy_type
        self.config = config or StrategyConfig()
        self.logger = get_logger(f'strategy_{name.lower()}')
        
        # Strategy state
        self.is_active = True
        self.last_signal_time = None
        self.performance_history = []
        self.error_count = 0
        self.max_errors = 5
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.total_return = 0.0
        
        self.log_info(f"Strategy {self.name} initialized")
    
    @abstractmethod
    def analyze(self, symbol: str, market_data: List[MarketData], 
               additional_data: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Analyze market data and generate trading signal"""
        pass
    
    @abstractmethod
    def validate_signal(self, signal: StrategySignal, market_data: List[MarketData]) -> bool:
        """Validate if signal meets strategy criteria"""
        pass
    
    def can_generate_signal(self) -> bool:
        """Check if strategy can generate new signals"""
        if not self.is_active:
            return False
        
        if self.error_count >= self.max_errors:
            self.log_warning(f"Strategy {self.name} disabled due to too many errors")
            return False
        
        # Check cooldown period
        if self.last_signal_time:
            cooldown = datetime.now() - self.last_signal_time
            if cooldown.total_seconds() < self.config.cooldown_minutes * 60:
                return False
        
        return True
    
    def generate_signal(self, symbol: str, market_data: List[MarketData], 
                       additional_data: Optional[Dict[str, Any]] = None) -> Optional[StrategyResult]:
        """Generate trading signal with error handling"""
        try:
            if not self.can_generate_signal():
                return None
            
            if len(market_data) < self.get_min_data_points():
                raise ValidationError(f"Insufficient data: {len(market_data)} < {self.get_min_data_points()}")
            
            # Generate signal
            result = self.analyze(symbol, market_data, additional_data)
            
            # Validate signal
            if not self.validate_signal(result.signal, market_data):
                self.log_debug(f"Signal validation failed for {symbol}")
                return None
            
            # Update tracking
            self.last_signal_time = datetime.now()
            self.total_signals += 1
            
            self.log_info(f"Generated {result.signal.action} signal for {symbol} with {result.signal.confidence:.1f}% confidence")
            return result
        
        except Exception as e:
            self.error_count += 1
            self.log_error(f"Signal generation failed for {symbol}: {e}")
            
            if self.error_count >= self.max_errors:
                self.is_active = False
                self.log_error(f"Strategy {self.name} deactivated due to repeated errors")
            
            return None
    
    def update_performance(self, signal_result: Dict[str, Any]):
        """Update strategy performance metrics"""
        try:
            self.performance_history.append({
                'timestamp': datetime.now(),
                'result': signal_result
            })
            
            # Update success rate
            if signal_result.get('successful', False):
                self.successful_signals += 1
            
            # Update return
            pnl = signal_result.get('pnl', 0)
            self.total_return += pnl
            
            self.log_debug(f"Performance updated: {self.get_success_rate():.1f}% success rate")
        
        except Exception as e:
            self.log_warning(f"Failed to update performance: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        return {
            'name': self.name,
            'type': self.strategy_type.value,
            'is_active': self.is_active,
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'success_rate': self.get_success_rate(),
            'total_return': self.total_return,
            'error_count': self.error_count,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'avg_confidence': self._calculate_avg_confidence(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_signals == 0:
            return 0.0
        return (self.successful_signals / self.total_signals) * 100
    
    def reset_errors(self):
        """Reset error count and reactivate strategy"""
        self.error_count = 0
        self.is_active = True
        self.log_info(f"Strategy {self.name} errors reset and reactivated")
    
    def get_min_data_points(self) -> int:
        """Get minimum data points required for analysis"""
        return 50  # Default minimum
    
    def _create_signal(self, action: str, confidence: float, reasoning: List[str],
                      entry_price: Decimal, stop_loss: Optional[Decimal] = None,
                      take_profit: Optional[Decimal] = None, 
                      position_size: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> StrategySignal:
        """Helper method to create strategy signal"""
        # Determine signal strength
        if confidence >= 85:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 75:
            strength = SignalStrength.STRONG
        elif confidence >= 65:
            strength = SignalStrength.MODERATE
        elif confidence >= 55:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK
        
        # Calculate position size if not provided
        if position_size is None:
            position_size = self._calculate_position_size(confidence)
        
        # Calculate levels if not provided
        if stop_loss is None:
            stop_loss = self._calculate_stop_loss(entry_price, action)
        
        if take_profit is None:
            take_profit = self._calculate_take_profit(entry_price, action)
        
        # Signal validity
        valid_until = datetime.now().replace(
            minute=datetime.now().minute + self.config.signal_timeout_minutes
        )
        
        return StrategySignal(
            action=action,
            confidence=confidence,
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reasoning=reasoning,
            metadata=metadata or {},
            timestamp=datetime.now(),
            valid_until=valid_until
        )
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        # Scale position size with confidence
        base_size = self.config.max_position_size
        confidence_factor = confidence / 100.0
        
        # Apply risk tolerance
        risk_multiplier = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }.get(self.config.risk_tolerance, 1.0)
        
        position_size = base_size * confidence_factor * risk_multiplier
        return min(position_size, self.config.max_position_size)
    
    def _calculate_stop_loss(self, entry_price: Decimal, action: str) -> Decimal:
        """Calculate stop loss level"""
        stop_loss_amount = entry_price * Decimal(str(self.config.stop_loss_pct))
        
        if action == 'buy':
            return entry_price - stop_loss_amount
        else:  # sell
            return entry_price + stop_loss_amount
    
    def _calculate_take_profit(self, entry_price: Decimal, action: str) -> Decimal:
        """Calculate take profit level"""
        take_profit_amount = entry_price * Decimal(str(self.config.take_profit_pct))
        
        if action == 'buy':
            return entry_price + take_profit_amount
        else:  # sell
            return entry_price - take_profit_amount
    
    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence of recent signals"""
        if not self.performance_history:
            return 0.0
        
        recent_history = self.performance_history[-20:]  # Last 20 signals
        confidences = [h['result'].get('confidence', 0) for h in recent_history]
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for strategy"""
        if len(self.performance_history) < 10:
            return 0.0
        
        try:
            returns = [h['result'].get('return', 0) for h in self.performance_history]
            
            if not returns:
                return 0.0
            
            mean_return = sum(returns) / len(returns)
            
            if len(returns) < 2:
                return 0.0
            
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return 0.0
            
            return mean_return / std_dev
        
        except Exception as e:
            self.log_warning(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.performance_history) < 2:
            return 0.0
        
        try:
            cumulative_returns = []
            cumulative = 0
            
            for h in self.performance_history:
                cumulative += h['result'].get('return', 0)
                cumulative_returns.append(cumulative)
            
            if not cumulative_returns:
                return 0.0
            
            peak = cumulative_returns[0]
            max_drawdown = 0
            
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
        
        except Exception as e:
            self.log_warning(f"Failed to calculate max drawdown: {e}")
            return 0.0
    
    def __str__(self) -> str:
        return f"{self.name} ({self.strategy_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', type='{self.strategy_type.value}', active={self.is_active})>"