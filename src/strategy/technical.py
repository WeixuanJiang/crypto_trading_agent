"""Technical analysis based trading strategies"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal
import numpy as np

from .base import BaseStrategy, StrategyType, StrategyResult, StrategyConfig
from ..core.logger import get_logger
from ..core.exceptions import StrategyError, ValidationError
from ..data.models import MarketData, AnalysisResult
from ..market.indicators import TechnicalIndicators, IndicatorConfig


class TechnicalStrategyConfig(StrategyConfig):
    """Configuration for technical strategies"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Technical indicator periods
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.macd_fast = kwargs.get('macd_fast', 12)
        self.macd_slow = kwargs.get('macd_slow', 26)
        self.macd_signal = kwargs.get('macd_signal', 9)
        self.bb_period = kwargs.get('bb_period', 20)
        self.bb_std = kwargs.get('bb_std', 2)
        self.ema_short = kwargs.get('ema_short', 12)
        self.ema_long = kwargs.get('ema_long', 26)
        
        # Signal thresholds
        self.rsi_oversold = kwargs.get('rsi_oversold', 30)
        self.rsi_overbought = kwargs.get('rsi_overbought', 70)
        self.volume_threshold = kwargs.get('volume_threshold', 1.5)
        self.trend_strength_min = kwargs.get('trend_strength_min', 0.6)
        
        # Pattern recognition
        self.enable_patterns = kwargs.get('enable_patterns', True)
        self.pattern_confidence_min = kwargs.get('pattern_confidence_min', 0.7)
        
        # Multi-timeframe analysis
        self.enable_mtf = kwargs.get('enable_mtf', False)
        self.mtf_timeframes = kwargs.get('mtf_timeframes', ['1hour', '4hour', '1day'])


class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(self, config: Optional[TechnicalStrategyConfig] = None):
        super().__init__("RSI_Strategy", StrategyType.TECHNICAL, config or TechnicalStrategyConfig())
        self.indicators = TechnicalIndicators()
    
    def analyze(self, symbol: str, market_data: List[MarketData], 
               additional_data: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Analyze using RSI strategy"""
        try:
            # Calculate RSI
            closes = [float(data.close) for data in market_data]
            rsi_values = self.indicators.rsi(closes, self.config.rsi_period)
            
            if len(rsi_values) == 0:
                raise StrategyError("Failed to calculate RSI")
            
            current_rsi = rsi_values[-1]
            current_price = Decimal(str(market_data[-1].close))
            
            # Generate signal
            action = "hold"
            confidence = 50.0
            reasoning = []
            
            # RSI oversold condition
            if current_rsi <= self.config.rsi_oversold:
                action = "buy"
                confidence = min(90, 60 + (self.config.rsi_oversold - current_rsi) * 2)
                reasoning.append(f"RSI oversold: {current_rsi:.1f} <= {self.config.rsi_oversold}")
            
            # RSI overbought condition
            elif current_rsi >= self.config.rsi_overbought:
                action = "sell"
                confidence = min(90, 60 + (current_rsi - self.config.rsi_overbought) * 2)
                reasoning.append(f"RSI overbought: {current_rsi:.1f} >= {self.config.rsi_overbought}")
            
            # Additional confirmation signals
            if len(rsi_values) >= 3:
                rsi_trend = self._analyze_rsi_trend(rsi_values[-3:])
                if rsi_trend:
                    confidence += 10
                    reasoning.append(f"RSI trend confirmation: {rsi_trend}")
            
            # Volume confirmation
            volume_factor = self._check_volume_confirmation(market_data)
            if volume_factor > 1:
                confidence += 5
                reasoning.append(f"Volume confirmation: {volume_factor:.1f}x average")
            
            # Create signal
            signal = self._create_signal(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                entry_price=current_price,
                metadata={
                    'rsi': current_rsi,
                    'rsi_period': self.config.rsi_period,
                    'volume_factor': volume_factor
                }
            )
            
            # Create analysis result
            analysis = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                indicators={
                    'rsi': current_rsi,
                    'rsi_values': rsi_values[-10:],  # Last 10 values
                },
                patterns=[],
                signals=[action],
                confidence=confidence,
                metadata={'strategy': 'RSI'}
            )
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                analysis=analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment=self._assess_risk(market_data, signal),
                execution_priority=self._calculate_priority(confidence, action)
            )
        
        except Exception as e:
            raise StrategyError(f"RSI analysis failed: {e}")
    
    def validate_signal(self, signal, market_data: List[MarketData]) -> bool:
        """Validate RSI signal"""
        # Minimum confidence check
        if signal.confidence < self.config.min_confidence:
            return False
        
        # No trading in low volume conditions
        if len(market_data) >= 2:
            current_volume = float(market_data[-1].volume)
            avg_volume = np.mean([float(d.volume) for d in market_data[-20:]])
            
            if current_volume < avg_volume * 0.5:  # Very low volume
                return False
        
        return True
    
    def _analyze_rsi_trend(self, rsi_values: List[float]) -> Optional[str]:
        """Analyze RSI trend for confirmation"""
        if len(rsi_values) < 3:
            return None
        
        if all(rsi_values[i] < rsi_values[i+1] for i in range(len(rsi_values)-1)):
            return "bullish_divergence"
        elif all(rsi_values[i] > rsi_values[i+1] for i in range(len(rsi_values)-1)):
            return "bearish_divergence"
        
        return None
    
    def _check_volume_confirmation(self, market_data: List[MarketData]) -> float:
        """Check volume confirmation"""
        if len(market_data) < 20:
            return 1.0
        
        current_volume = float(market_data[-1].volume)
        avg_volume = np.mean([float(d.volume) for d in market_data[-20:]])
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0


class MACDStrategy(BaseStrategy):
    """MACD-based trend following strategy"""
    
    def __init__(self, config: Optional[TechnicalStrategyConfig] = None):
        super().__init__("MACD_Strategy", StrategyType.TECHNICAL, config or TechnicalStrategyConfig())
        self.indicators = TechnicalIndicators()
    
    def analyze(self, symbol: str, market_data: List[MarketData], 
               additional_data: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Analyze using MACD strategy"""
        try:
            closes = [float(data.close) for data in market_data]
            
            # Calculate MACD
            macd_line, signal_line, histogram = self.indicators.macd(
                closes, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            
            if len(macd_line) == 0:
                raise StrategyError("Failed to calculate MACD")
            
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            current_histogram = histogram[-1]
            current_price = Decimal(str(market_data[-1].close))
            
            # Generate signal
            action = "hold"
            confidence = 50.0
            reasoning = []
            
            # MACD crossover signals
            if len(macd_line) >= 2:
                prev_macd = macd_line[-2]
                prev_signal = signal_line[-2]
                
                # Bullish crossover
                if prev_macd <= prev_signal and current_macd > current_signal:
                    action = "buy"
                    confidence = 75
                    reasoning.append("MACD bullish crossover")
                
                # Bearish crossover
                elif prev_macd >= prev_signal and current_macd < current_signal:
                    action = "sell"
                    confidence = 75
                    reasoning.append("MACD bearish crossover")
            
            # Histogram analysis
            if len(histogram) >= 3:
                histogram_trend = self._analyze_histogram_trend(histogram[-3:])
                if histogram_trend:
                    if action != "hold":
                        confidence += 10
                    reasoning.append(f"Histogram trend: {histogram_trend}")
            
            # Zero line analysis
            zero_line_signal = self._analyze_zero_line(current_macd, current_signal)
            if zero_line_signal:
                if action != "hold":
                    confidence += 5
                reasoning.append(f"Zero line: {zero_line_signal}")
            
            # Create signal
            signal = self._create_signal(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                entry_price=current_price,
                metadata={
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_histogram,
                    'fast_period': self.config.macd_fast,
                    'slow_period': self.config.macd_slow,
                    'signal_period': self.config.macd_signal
                }
            )
            
            # Create analysis result
            analysis = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                indicators={
                    'macd_line': current_macd,
                    'signal_line': current_signal,
                    'histogram': current_histogram,
                    'macd_values': macd_line[-10:],
                    'signal_values': signal_line[-10:]
                },
                patterns=[],
                signals=[action],
                confidence=confidence,
                metadata={'strategy': 'MACD'}
            )
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                analysis=analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment=self._assess_risk(market_data, signal),
                execution_priority=self._calculate_priority(confidence, action)
            )
        
        except Exception as e:
            raise StrategyError(f"MACD analysis failed: {e}")
    
    def validate_signal(self, signal, market_data: List[MarketData]) -> bool:
        """Validate MACD signal"""
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check for sufficient trend strength
        if len(market_data) >= 20:
            trend_strength = self._calculate_trend_strength(market_data[-20:])
            if trend_strength < self.config.trend_strength_min:
                return False
        
        return True
    
    def _analyze_histogram_trend(self, histogram_values: List[float]) -> Optional[str]:
        """Analyze MACD histogram trend"""
        if len(histogram_values) < 3:
            return None
        
        if all(histogram_values[i] < histogram_values[i+1] for i in range(len(histogram_values)-1)):
            return "increasing_momentum"
        elif all(histogram_values[i] > histogram_values[i+1] for i in range(len(histogram_values)-1)):
            return "decreasing_momentum"
        
        return None
    
    def _analyze_zero_line(self, macd: float, signal: float) -> Optional[str]:
        """Analyze MACD zero line position"""
        if macd > 0 and signal > 0:
            return "bullish_territory"
        elif macd < 0 and signal < 0:
            return "bearish_territory"
        
        return None
    
    def _calculate_trend_strength(self, market_data: List[MarketData]) -> float:
        """Calculate trend strength"""
        closes = [float(data.close) for data in market_data]
        
        if len(closes) < 2:
            return 0.0
        
        # Simple trend strength calculation
        price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        positive_changes = sum(1 for change in price_changes if change > 0)
        
        return positive_changes / len(price_changes)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, config: Optional[TechnicalStrategyConfig] = None):
        super().__init__("BB_Strategy", StrategyType.TECHNICAL, config or TechnicalStrategyConfig())
        self.indicators = TechnicalIndicators()
    
    def analyze(self, symbol: str, market_data: List[MarketData], 
               additional_data: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Analyze using Bollinger Bands strategy"""
        try:
            closes = [float(data.close) for data in market_data]
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.indicators.bollinger_bands(
                closes, self.config.bb_period, self.config.bb_std
            )
            
            if len(upper_band) == 0:
                raise StrategyError("Failed to calculate Bollinger Bands")
            
            current_price = float(market_data[-1].close)
            current_upper = upper_band[-1]
            current_middle = middle_band[-1]
            current_lower = lower_band[-1]
            
            # Calculate band position
            band_width = current_upper - current_lower
            price_position = (current_price - current_lower) / band_width if band_width > 0 else 0.5
            
            # Generate signal
            action = "hold"
            confidence = 50.0
            reasoning = []
            
            # Price near lower band (oversold)
            if price_position <= 0.1:
                action = "buy"
                confidence = 70 + (0.1 - price_position) * 200  # Higher confidence closer to band
                reasoning.append(f"Price near lower band: {price_position:.2f}")
            
            # Price near upper band (overbought)
            elif price_position >= 0.9:
                action = "sell"
                confidence = 70 + (price_position - 0.9) * 200
                reasoning.append(f"Price near upper band: {price_position:.2f}")
            
            # Band squeeze detection
            if len(upper_band) >= 20:
                band_squeeze = self._detect_band_squeeze(upper_band[-20:], lower_band[-20:])
                if band_squeeze:
                    confidence += 10
                    reasoning.append("Band squeeze detected - volatility breakout expected")
            
            # Mean reversion confirmation
            if action != "hold":
                mean_reversion_strength = abs(current_price - current_middle) / current_middle
                if mean_reversion_strength > 0.02:  # 2% deviation
                    confidence += 5
                    reasoning.append(f"Strong mean reversion signal: {mean_reversion_strength:.1%}")
            
            # Create signal
            signal = self._create_signal(
                action=action,
                confidence=min(confidence, 95),  # Cap confidence
                reasoning=reasoning,
                entry_price=Decimal(str(current_price)),
                metadata={
                    'upper_band': current_upper,
                    'middle_band': current_middle,
                    'lower_band': current_lower,
                    'band_position': price_position,
                    'band_width': band_width,
                    'bb_period': self.config.bb_period,
                    'bb_std': self.config.bb_std
                }
            )
            
            # Create analysis result
            analysis = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                indicators={
                    'bb_upper': current_upper,
                    'bb_middle': current_middle,
                    'bb_lower': current_lower,
                    'bb_position': price_position,
                    'bb_width': band_width
                },
                patterns=[],
                signals=[action],
                confidence=confidence,
                metadata={'strategy': 'BollingerBands'}
            )
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                analysis=analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment=self._assess_risk(market_data, signal),
                execution_priority=self._calculate_priority(confidence, action)
            )
        
        except Exception as e:
            raise StrategyError(f"Bollinger Bands analysis failed: {e}")
    
    def validate_signal(self, signal, market_data: List[MarketData]) -> bool:
        """Validate Bollinger Bands signal"""
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check volatility conditions
        if len(market_data) >= 20:
            volatility = self._calculate_volatility(market_data[-20:])
            if volatility < 0.01:  # Very low volatility
                return False
        
        return True
    
    def _detect_band_squeeze(self, upper_band: List[float], lower_band: List[float]) -> bool:
        """Detect Bollinger Band squeeze"""
        if len(upper_band) < 20 or len(lower_band) < 20:
            return False
        
        # Calculate current and historical band widths
        current_width = upper_band[-1] - lower_band[-1]
        avg_width = np.mean([upper_band[i] - lower_band[i] for i in range(-20, -1)])
        
        # Squeeze if current width is significantly smaller than average
        return current_width < avg_width * 0.8
    
    def _calculate_volatility(self, market_data: List[MarketData]) -> float:
        """Calculate price volatility"""
        closes = [float(data.close) for data in market_data]
        
        if len(closes) < 2:
            return 0.0
        
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        return np.std(returns) if returns else 0.0


class MultiIndicatorStrategy(BaseStrategy):
    """Strategy combining multiple technical indicators"""
    
    def __init__(self, config: Optional[TechnicalStrategyConfig] = None):
        super().__init__("Multi_Indicator_Strategy", StrategyType.TECHNICAL, config or TechnicalStrategyConfig())
        self.indicators = TechnicalIndicators()
        
        # Sub-strategies
        self.rsi_strategy = RSIStrategy(config)
        self.macd_strategy = MACDStrategy(config)
        self.bb_strategy = BollingerBandsStrategy(config)
    
    def analyze(self, symbol: str, market_data: List[MarketData], 
               additional_data: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Analyze using multiple indicators"""
        try:
            # Get signals from sub-strategies
            rsi_result = self.rsi_strategy.analyze(symbol, market_data, additional_data)
            macd_result = self.macd_strategy.analyze(symbol, market_data, additional_data)
            bb_result = self.bb_strategy.analyze(symbol, market_data, additional_data)
            
            # Combine signals
            signals = [rsi_result.signal, macd_result.signal, bb_result.signal]
            combined_signal = self._combine_signals(signals)
            
            # Merge analysis results
            combined_analysis = self._merge_analysis_results([
                rsi_result.analysis, macd_result.analysis, bb_result.analysis
            ])
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=combined_signal,
                analysis=combined_analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment=self._assess_risk(market_data, combined_signal),
                execution_priority=self._calculate_priority(combined_signal.confidence, combined_signal.action)
            )
        
        except Exception as e:
            raise StrategyError(f"Multi-indicator analysis failed: {e}")
    
    def validate_signal(self, signal, market_data: List[MarketData]) -> bool:
        """Validate combined signal"""
        # Higher confidence threshold for combined strategy
        return signal.confidence >= max(self.config.min_confidence, 70)
    
    def _combine_signals(self, signals):
        """Combine multiple signals into one"""
        # Count signal types
        buy_signals = sum(1 for s in signals if s.action == "buy")
        sell_signals = sum(1 for s in signals if s.action == "sell")
        hold_signals = sum(1 for s in signals if s.action == "hold")
        
        # Determine action
        if buy_signals > sell_signals and buy_signals > hold_signals:
            action = "buy"
        elif sell_signals > buy_signals and sell_signals > hold_signals:
            action = "sell"
        else:
            action = "hold"
        
        # Calculate combined confidence
        relevant_signals = [s for s in signals if s.action == action]
        if relevant_signals:
            confidence = np.mean([s.confidence for s in relevant_signals])
            # Boost confidence if multiple indicators agree
            if len(relevant_signals) > 1:
                confidence += 10 * (len(relevant_signals) - 1)
        else:
            confidence = 50.0
        
        # Combine reasoning
        reasoning = []
        for i, signal in enumerate(signals):
            strategy_names = ["RSI", "MACD", "BB"]
            reasoning.extend([f"{strategy_names[i]}: {r}" for r in signal.reasoning])
        
        # Use the first signal as template
        template_signal = signals[0]
        
        return self._create_signal(
            action=action,
            confidence=min(confidence, 95),
            reasoning=reasoning,
            entry_price=template_signal.entry_price,
            metadata={
                'combined_strategy': True,
                'sub_signals': {
                    'rsi': {'action': signals[0].action, 'confidence': signals[0].confidence},
                    'macd': {'action': signals[1].action, 'confidence': signals[1].confidence},
                    'bb': {'action': signals[2].action, 'confidence': signals[2].confidence}
                }
            }
        )
    
    def _merge_analysis_results(self, analyses):
        """Merge multiple analysis results"""
        # Use first analysis as template
        base_analysis = analyses[0]
        
        # Combine indicators
        combined_indicators = {}
        for analysis in analyses:
            combined_indicators.update(analysis.indicators)
        
        # Combine patterns
        combined_patterns = []
        for analysis in analyses:
            combined_patterns.extend(analysis.patterns)
        
        # Combine signals
        combined_signals = []
        for analysis in analyses:
            combined_signals.extend(analysis.signals)
        
        # Average confidence
        avg_confidence = np.mean([a.confidence for a in analyses])
        
        return AnalysisResult(
            symbol=base_analysis.symbol,
            timestamp=datetime.now(),
            indicators=combined_indicators,
            patterns=combined_patterns,
            signals=combined_signals,
            confidence=avg_confidence,
            metadata={'strategy': 'MultiIndicator', 'sub_strategies': ['RSI', 'MACD', 'BollingerBands']}
        )
    
    def get_min_data_points(self) -> int:
        """Get minimum data points for multi-indicator analysis"""
        return max(50, self.config.bb_period + 10)  # Ensure enough data for all indicators
    
    def _assess_risk(self, market_data: List[MarketData], signal) -> Dict[str, Any]:
        """Assess risk for the signal"""
        try:
            # Calculate volatility
            closes = [float(data.close) for data in market_data[-20:]]
            volatility = np.std(closes) / np.mean(closes) if closes else 0
            
            # Calculate volume trend
            volumes = [float(data.volume) for data in market_data[-10:]]
            volume_trend = "increasing" if volumes[-1] > np.mean(volumes[:-1]) else "decreasing"
            
            # Risk level based on volatility and confidence
            if volatility > 0.05 or signal.confidence < 70:
                risk_level = "high"
            elif volatility > 0.03 or signal.confidence < 80:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                'risk_level': risk_level,
                'volatility': volatility,
                'volume_trend': volume_trend,
                'confidence_factor': signal.confidence / 100,
                'recommended_position_size': signal.position_size
            }
        
        except Exception as e:
            self.log_warning(f"Risk assessment failed: {e}")
            return {'risk_level': 'unknown', 'error': str(e)}
    
    def _calculate_priority(self, confidence: float, action: str) -> int:
        """Calculate execution priority"""
        if action == "hold":
            return 1
        
        # Higher confidence = higher priority
        base_priority = int(confidence / 10)  # 0-10 scale
        
        # Boost priority for strong signals
        if confidence >= 85:
            base_priority += 2
        elif confidence >= 75:
            base_priority += 1
        
        return min(base_priority, 10)