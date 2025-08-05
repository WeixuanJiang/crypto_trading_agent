"""Technical indicators calculation module"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass

from ..core.logger import get_logger
from ..core.exceptions import ValidationError
from ..data.models import MarketData


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Moving Averages
    sma_short: int = 10
    sma_long: int = 50
    ema_short: int = 12
    ema_long: int = 26
    
    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    
    # Williams %R
    williams_period: int = 14
    
    # ATR
    atr_period: int = 14
    
    # Volume indicators
    volume_sma_period: int = 20
    
    # Momentum
    momentum_period: int = 10
    
    # CCI
    cci_period: int = 20
    cci_overbought: float = 100.0
    cci_oversold: float = -100.0


class TechnicalIndicators:
    """Technical indicators calculation engine"""
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self.logger = get_logger('indicators')
    
    def calculate_all(self, data: List[MarketData]) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        if len(data) < 50:  # Minimum data points for reliable calculations
            raise ValidationError("Insufficient data for technical analysis (minimum 50 points required)")
        
        try:
            # Convert to DataFrame for easier manipulation
            df = self._to_dataframe(data)
            
            indicators = {}
            
            # Price-based indicators
            indicators.update(self._calculate_moving_averages(df))
            indicators.update(self._calculate_rsi(df))
            indicators.update(self._calculate_macd(df))
            indicators.update(self._calculate_bollinger_bands(df))
            indicators.update(self._calculate_stochastic(df))
            indicators.update(self._calculate_williams_r(df))
            indicators.update(self._calculate_cci(df))
            indicators.update(self._calculate_momentum(df))
            
            # Volatility indicators
            indicators.update(self._calculate_atr(df))
            
            # Volume indicators
            indicators.update(self._calculate_volume_indicators(df))
            
            # Support/Resistance levels
            indicators.update(self._calculate_support_resistance(df))
            
            # Trend analysis
            indicators.update(self._analyze_trend(df))
            
            # Pattern recognition
            indicators.update(self._detect_patterns(df))
            
            self.logger.debug(f"Calculated {len(indicators)} technical indicators")
            return indicators
        
        except Exception as e:
            self.logger.error(f"Failed to calculate indicators: {e}")
            raise ValidationError(f"Technical analysis failed: {e}")
    
    def _to_dataframe(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert MarketData list to pandas DataFrame"""
        records = []
        for item in data:
            records.append({
                'timestamp': item.timestamp,
                'open': float(item.open_price),
                'high': float(item.high_price),
                'low': float(item.low_price),
                'close': float(item.close_price),
                'volume': float(item.volume)
            })
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate moving averages"""
        try:
            sma_short = df['close'].rolling(window=self.config.sma_short).mean()
            sma_long = df['close'].rolling(window=self.config.sma_long).mean()
            ema_short = df['close'].ewm(span=self.config.ema_short).mean()
            ema_long = df['close'].ewm(span=self.config.ema_long).mean()
            
            current_price = df['close'].iloc[-1]
            
            return {
                'sma_short': float(sma_short.iloc[-1]) if not pd.isna(sma_short.iloc[-1]) else None,
                'sma_long': float(sma_long.iloc[-1]) if not pd.isna(sma_long.iloc[-1]) else None,
                'ema_short': float(ema_short.iloc[-1]) if not pd.isna(ema_short.iloc[-1]) else None,
                'ema_long': float(ema_long.iloc[-1]) if not pd.isna(ema_long.iloc[-1]) else None,
                'sma_cross_signal': self._get_cross_signal(sma_short, sma_long),
                'ema_cross_signal': self._get_cross_signal(ema_short, ema_long),
                'price_above_sma_short': current_price > sma_short.iloc[-1] if not pd.isna(sma_short.iloc[-1]) else None,
                'price_above_sma_long': current_price > sma_long.iloc[-1] if not pd.isna(sma_long.iloc[-1]) else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate moving averages: {e}")
            return {}
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            
            return {
                'rsi': current_rsi,
                'rsi_overbought': current_rsi > self.config.rsi_overbought if current_rsi else None,
                'rsi_oversold': current_rsi < self.config.rsi_oversold if current_rsi else None,
                'rsi_signal': self._get_rsi_signal(current_rsi)
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate RSI: {e}")
            return {}
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = df['close'].ewm(span=self.config.macd_fast).mean()
            ema_slow = df['close'].ewm(span=self.config.macd_slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.config.macd_signal).mean()
            histogram = macd_line - signal_line
            
            current_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
            current_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
            current_histogram = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
            
            return {
                'macd': current_macd,
                'macd_signal': current_signal,
                'macd_histogram': current_histogram,
                'macd_bullish': current_macd > current_signal if current_macd and current_signal else None,
                'macd_cross_signal': self._get_cross_signal(macd_line, signal_line)
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate MACD: {e}")
            return {}
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        try:
            sma = df['close'].rolling(window=self.config.bb_period).mean()
            std = df['close'].rolling(window=self.config.bb_period).std()
            
            upper_band = sma + (std * self.config.bb_std_dev)
            lower_band = sma - (std * self.config.bb_std_dev)
            
            current_price = df['close'].iloc[-1]
            current_upper = float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else None
            current_lower = float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else None
            current_middle = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None
            
            # Calculate %B (position within bands)
            bb_percent = None
            if current_upper and current_lower:
                bb_percent = (current_price - current_lower) / (current_upper - current_lower)
            
            return {
                'bb_upper': current_upper,
                'bb_middle': current_middle,
                'bb_lower': current_lower,
                'bb_percent': float(bb_percent) if bb_percent else None,
                'bb_squeeze': self._detect_bb_squeeze(std),
                'price_above_bb_upper': current_price > current_upper if current_upper else None,
                'price_below_bb_lower': current_price < current_lower if current_lower else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate Bollinger Bands: {e}")
            return {}
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = df['low'].rolling(window=self.config.stoch_k_period).min()
            highest_high = df['high'].rolling(window=self.config.stoch_k_period).max()
            
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=self.config.stoch_d_period).mean()
            
            current_k = float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else None
            current_d = float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else None
            
            return {
                'stoch_k': current_k,
                'stoch_d': current_d,
                'stoch_overbought': current_k > self.config.stoch_overbought if current_k else None,
                'stoch_oversold': current_k < self.config.stoch_oversold if current_k else None,
                'stoch_cross_signal': self._get_cross_signal(k_percent, d_percent)
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate Stochastic: {e}")
            return {}
    
    def _calculate_williams_r(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Williams %R"""
        try:
            highest_high = df['high'].rolling(window=self.config.williams_period).max()
            lowest_low = df['low'].rolling(window=self.config.williams_period).min()
            
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            current_williams = float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None
            
            return {
                'williams_r': current_williams,
                'williams_overbought': current_williams > -20 if current_williams else None,
                'williams_oversold': current_williams < -80 if current_williams else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate Williams %R: {e}")
            return {}
    
    def _calculate_cci(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=self.config.cci_period).mean()
            mean_deviation = typical_price.rolling(window=self.config.cci_period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            current_cci = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else None
            
            return {
                'cci': current_cci,
                'cci_overbought': current_cci > self.config.cci_overbought if current_cci else None,
                'cci_oversold': current_cci < self.config.cci_oversold if current_cci else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate CCI: {e}")
            return {}
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Momentum indicators"""
        try:
            momentum = df['close'] / df['close'].shift(self.config.momentum_period) - 1
            roc = df['close'].pct_change(periods=self.config.momentum_period) * 100
            
            current_momentum = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else None
            current_roc = float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else None
            
            return {
                'momentum': current_momentum,
                'rate_of_change': current_roc,
                'momentum_bullish': current_momentum > 0 if current_momentum else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate Momentum: {e}")
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=self.config.atr_period).mean()
            
            current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            current_price = df['close'].iloc[-1]
            
            return {
                'atr': current_atr,
                'atr_percent': (current_atr / current_price * 100) if current_atr else None,
                'volatility_level': self._classify_volatility(current_atr, current_price) if current_atr else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate ATR: {e}")
            return {}
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        try:
            volume_sma = df['volume'].rolling(window=self.config.volume_sma_period).mean()
            current_volume = df['volume'].iloc[-1]
            avg_volume = float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else None
            
            # Volume trend
            volume_trend = df['volume'].rolling(window=5).mean().pct_change().iloc[-1]
            
            # On-Balance Volume (simplified)
            price_change = df['close'].diff()
            obv_direction = np.where(price_change > 0, df['volume'], 
                                   np.where(price_change < 0, -df['volume'], 0))
            obv = pd.Series(obv_direction).cumsum()
            
            return {
                'volume_avg': avg_volume,
                'volume_ratio': (current_volume / avg_volume) if avg_volume else None,
                'volume_trend': float(volume_trend) if not pd.isna(volume_trend) else None,
                'high_volume': (current_volume > avg_volume * 1.5) if avg_volume else None,
                'obv': float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate volume indicators: {e}")
            return {}
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        try:
            # Use recent highs and lows for support/resistance
            recent_data = df.tail(50)  # Last 50 periods
            
            # Find local maxima and minima
            highs = recent_data['high'].rolling(window=5, center=True).max()
            lows = recent_data['low'].rolling(window=5, center=True).min()
            
            resistance_levels = highs[highs == recent_data['high']].dropna().unique()
            support_levels = lows[lows == recent_data['low']].dropna().unique()
            
            # Get closest levels to current price
            current_price = df['close'].iloc[-1]
            
            nearest_resistance = None
            nearest_support = None
            
            if len(resistance_levels) > 0:
                resistance_above = resistance_levels[resistance_levels > current_price]
                if len(resistance_above) > 0:
                    nearest_resistance = float(min(resistance_above))
            
            if len(support_levels) > 0:
                support_below = support_levels[support_levels < current_price]
                if len(support_below) > 0:
                    nearest_support = float(max(support_below))
            
            return {
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'resistance_distance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
                'support_distance': ((current_price - nearest_support) / current_price * 100) if nearest_support else None
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate support/resistance: {e}")
            return {}
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trend"""
        try:
            # Short-term trend (last 10 periods)
            short_trend = df['close'].tail(10).pct_change().mean()
            
            # Medium-term trend (last 30 periods)
            medium_trend = df['close'].tail(30).pct_change().mean()
            
            # Long-term trend (last 50 periods)
            long_trend = df['close'].tail(50).pct_change().mean()
            
            # Trend strength
            price_changes = df['close'].pct_change().tail(20)
            trend_consistency = (price_changes > 0).sum() / len(price_changes)
            
            return {
                'short_trend': float(short_trend) if not pd.isna(short_trend) else None,
                'medium_trend': float(medium_trend) if not pd.isna(medium_trend) else None,
                'long_trend': float(long_trend) if not pd.isna(long_trend) else None,
                'trend_consistency': float(trend_consistency),
                'trend_direction': self._classify_trend(short_trend, medium_trend, long_trend)
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze trend: {e}")
            return {}
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns"""
        try:
            patterns = {}
            
            # Doji pattern (open ≈ close)
            last_candle = df.iloc[-1]
            body_size = abs(last_candle['close'] - last_candle['open'])
            candle_range = last_candle['high'] - last_candle['low']
            
            patterns['doji'] = (body_size / candle_range) < 0.1 if candle_range > 0 else False
            
            # Hammer pattern
            lower_shadow = last_candle['open'] - last_candle['low'] if last_candle['close'] > last_candle['open'] else last_candle['close'] - last_candle['low']
            upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            
            patterns['hammer'] = (lower_shadow > body_size * 2) and (upper_shadow < body_size * 0.5)
            
            # Shooting star pattern
            patterns['shooting_star'] = (upper_shadow > body_size * 2) and (lower_shadow < body_size * 0.5)
            
            # Engulfing patterns (simplified)
            if len(df) >= 2:
                prev_candle = df.iloc[-2]
                current_candle = df.iloc[-1]
                
                # Bullish engulfing
                patterns['bullish_engulfing'] = (
                    prev_candle['close'] < prev_candle['open'] and  # Previous red
                    current_candle['close'] > current_candle['open'] and  # Current green
                    current_candle['open'] < prev_candle['close'] and  # Gap down
                    current_candle['close'] > prev_candle['open']  # Engulfs previous
                )
                
                # Bearish engulfing
                patterns['bearish_engulfing'] = (
                    prev_candle['close'] > prev_candle['open'] and  # Previous green
                    current_candle['close'] < current_candle['open'] and  # Current red
                    current_candle['open'] > prev_candle['close'] and  # Gap up
                    current_candle['close'] < prev_candle['open']  # Engulfs previous
                )
            
            return patterns
        except Exception as e:
            self.logger.warning(f"Failed to detect patterns: {e}")
            return {}
    
    def _get_cross_signal(self, fast_line: pd.Series, slow_line: pd.Series) -> Optional[str]:
        """Detect crossover signals"""
        try:
            if len(fast_line) < 2 or len(slow_line) < 2:
                return None
            
            current_fast = fast_line.iloc[-1]
            current_slow = slow_line.iloc[-1]
            prev_fast = fast_line.iloc[-2]
            prev_slow = slow_line.iloc[-2]
            
            if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
                return None
            
            # Bullish cross (fast crosses above slow)
            if prev_fast <= prev_slow and current_fast > current_slow:
                return 'bullish_cross'
            
            # Bearish cross (fast crosses below slow)
            if prev_fast >= prev_slow and current_fast < current_slow:
                return 'bearish_cross'
            
            return None
        except Exception:
            return None
    
    def _get_rsi_signal(self, rsi: Optional[float]) -> Optional[str]:
        """Get RSI signal"""
        if rsi is None:
            return None
        
        if rsi > self.config.rsi_overbought:
            return 'overbought'
        elif rsi < self.config.rsi_oversold:
            return 'oversold'
        elif rsi > 50:
            return 'bullish'
        else:
            return 'bearish'
    
    def _detect_bb_squeeze(self, std_series: pd.Series) -> bool:
        """Detect Bollinger Band squeeze"""
        try:
            if len(std_series) < 20:
                return False
            
            current_std = std_series.iloc[-1]
            avg_std = std_series.tail(20).mean()
            
            return current_std < avg_std * 0.8
        except Exception:
            return False
    
    def _classify_volatility(self, atr: float, price: float) -> str:
        """Classify volatility level"""
        atr_percent = (atr / price) * 100
        
        if atr_percent < 1:
            return 'low'
        elif atr_percent < 3:
            return 'medium'
        else:
            return 'high'
    
    def _classify_trend(self, short: Optional[float], medium: Optional[float], long: Optional[float]) -> str:
        """Classify overall trend direction"""
        trends = [t for t in [short, medium, long] if t is not None]
        
        if not trends:
            return 'neutral'
        
        positive_count = sum(1 for t in trends if t > 0.001)
        negative_count = sum(1 for t in trends if t < -0.001)
        
        if positive_count >= 2:
            return 'bullish'
        elif negative_count >= 2:
            return 'bearish'
        else:
            return 'neutral'