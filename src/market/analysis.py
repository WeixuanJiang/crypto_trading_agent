"""Market analysis module for comprehensive market evaluation"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

from ..core.logger import get_logger, TradeLoggerMixin
from ..core.exceptions import ValidationError, MarketDataError
from ..data.models import MarketData, AnalysisResult
from .indicators import TechnicalIndicators, IndicatorConfig
from .data_manager import MarketDataManager, DataRequest


class MarketSentiment(Enum):
    """Market sentiment classification"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class TrendStrength(Enum):
    """Trend strength classification"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


@dataclass
class MarketCondition:
    """Market condition assessment"""
    sentiment: MarketSentiment
    trend_direction: str
    trend_strength: TrendStrength
    volatility_level: str
    volume_profile: str
    support_strength: float
    resistance_strength: float
    momentum_score: float
    confidence_score: float


@dataclass
class TradingSignal:
    """Trading signal with confidence and reasoning"""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-100
    entry_price: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    reasoning: List[str]
    risk_level: str
    time_horizon: str
    expected_return: Optional[float]


class MarketAnalyzer(TradeLoggerMixin):
    """Comprehensive market analyzer"""
    
    def __init__(self, data_manager: MarketDataManager, 
                 indicator_config: Optional[IndicatorConfig] = None):
        super().__init__()
        self.data_manager = data_manager
        self.indicators = TechnicalIndicators(indicator_config)
        self.logger = get_logger('market_analyzer')
        
        # Analysis weights for different factors
        self.weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'volatility': 0.15,
            'support_resistance': 0.15,
            'patterns': 0.10
        }
        
        self.log_info("MarketAnalyzer initialized")
    
    def analyze_symbol(self, symbol: str, timeframe: str = '1hour', 
                      lookback_periods: int = 100) -> AnalysisResult:
        """Perform comprehensive analysis of a symbol"""
        try:
            self.log_info(f"Starting analysis for {symbol} on {timeframe}")
            
            # Get market data
            request = DataRequest(
                symbol=symbol,
                interval=timeframe,
                limit=lookback_periods,
                use_cache=True
            )
            
            market_data = self.data_manager.get_klines(request)
            
            if len(market_data) < 50:
                raise ValidationError(f"Insufficient data for analysis: {len(market_data)} periods")
            
            # Calculate technical indicators
            indicators = self.indicators.calculate_all(market_data)
            
            # Assess market condition
            market_condition = self._assess_market_condition(market_data, indicators)
            
            # Generate trading signal
            trading_signal = self._generate_trading_signal(market_data, indicators, market_condition)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(indicators, market_condition)
            
            # Create analysis result
            result = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                timeframe=timeframe,
                current_price=market_data[-1].close_price,
                indicators=indicators,
                signals={
                    'action': trading_signal.action,
                    'confidence': trading_signal.confidence,
                    'entry_price': float(trading_signal.entry_price) if trading_signal.entry_price else None,
                    'stop_loss': float(trading_signal.stop_loss) if trading_signal.stop_loss else None,
                    'take_profit': float(trading_signal.take_profit) if trading_signal.take_profit else None,
                    'reasoning': trading_signal.reasoning,
                    'risk_level': trading_signal.risk_level,
                    'time_horizon': trading_signal.time_horizon,
                    'expected_return': trading_signal.expected_return
                },
                market_condition={
                    'sentiment': market_condition.sentiment.value,
                    'trend_direction': market_condition.trend_direction,
                    'trend_strength': market_condition.trend_strength.value,
                    'volatility_level': market_condition.volatility_level,
                    'volume_profile': market_condition.volume_profile,
                    'support_strength': market_condition.support_strength,
                    'resistance_strength': market_condition.resistance_strength,
                    'momentum_score': market_condition.momentum_score,
                    'confidence_score': market_condition.confidence_score
                },
                score=overall_score,
                confidence=trading_signal.confidence
            )
            
            self.log_info(f"Analysis completed for {symbol}: {trading_signal.action} with {trading_signal.confidence:.1f}% confidence")
            return result
        
        except Exception as e:
            self.log_error(f"Analysis failed for {symbol}: {e}")
            raise MarketDataError(f"Market analysis failed: {e}")
    
    def compare_symbols(self, symbols: List[str], timeframe: str = '1hour') -> List[AnalysisResult]:
        """Compare multiple symbols and rank by attractiveness"""
        try:
            results = []
            
            for symbol in symbols:
                try:
                    analysis = self.analyze_symbol(symbol, timeframe)
                    results.append(analysis)
                except Exception as e:
                    self.log_warning(f"Failed to analyze {symbol}: {e}")
                    continue
            
            # Sort by overall score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            self.log_info(f"Compared {len(results)} symbols successfully")
            return results
        
        except Exception as e:
            self.log_error(f"Symbol comparison failed: {e}")
            raise MarketDataError(f"Symbol comparison failed: {e}")
    
    def get_market_overview(self, symbols: List[str]) -> Dict[str, Any]:
        """Get overall market overview"""
        try:
            analyses = self.compare_symbols(symbols)
            
            if not analyses:
                return {'error': 'No valid analyses available'}
            
            # Calculate market statistics
            sentiments = [a.market_condition['sentiment'] for a in analyses]
            scores = [a.score for a in analyses]
            
            bullish_count = sum(1 for s in sentiments if 'bullish' in s)
            bearish_count = sum(1 for s in sentiments if 'bearish' in s)
            neutral_count = len(sentiments) - bullish_count - bearish_count
            
            overview = {
                'timestamp': datetime.now().isoformat(),
                'total_symbols': len(analyses),
                'market_sentiment': {
                    'bullish_count': bullish_count,
                    'bearish_count': bearish_count,
                    'neutral_count': neutral_count,
                    'bullish_percentage': (bullish_count / len(analyses)) * 100,
                    'overall_sentiment': self._determine_overall_sentiment(sentiments)
                },
                'performance_metrics': {
                    'average_score': np.mean(scores),
                    'median_score': np.median(scores),
                    'score_std': np.std(scores),
                    'top_performers': [{
                        'symbol': a.symbol,
                        'score': a.score,
                        'action': a.signals['action'],
                        'confidence': a.confidence
                    } for a in analyses[:5]],
                    'worst_performers': [{
                        'symbol': a.symbol,
                        'score': a.score,
                        'action': a.signals['action'],
                        'confidence': a.confidence
                    } for a in analyses[-5:]]
                },
                'trading_opportunities': {
                    'strong_buys': [a.symbol for a in analyses 
                                  if a.signals['action'] == 'buy' and a.confidence > 75],
                    'strong_sells': [a.symbol for a in analyses 
                                   if a.signals['action'] == 'sell' and a.confidence > 75],
                    'high_confidence_signals': [a.symbol for a in analyses if a.confidence > 80]
                }
            }
            
            self.log_info(f"Market overview generated for {len(analyses)} symbols")
            return overview
        
        except Exception as e:
            self.log_error(f"Failed to generate market overview: {e}")
            raise MarketDataError(f"Market overview failed: {e}")
    
    def _assess_market_condition(self, data: List[MarketData], indicators: Dict[str, Any]) -> MarketCondition:
        """Assess overall market condition"""
        try:
            # Determine sentiment
            sentiment = self._calculate_sentiment(indicators)
            
            # Analyze trend
            trend_direction = indicators.get('trend_direction', 'neutral')
            trend_strength = self._calculate_trend_strength(indicators)
            
            # Assess volatility
            volatility_level = indicators.get('volatility_level', 'medium')
            
            # Analyze volume
            volume_profile = self._analyze_volume_profile(indicators)
            
            # Calculate support/resistance strength
            support_strength = self._calculate_support_strength(indicators)
            resistance_strength = self._calculate_resistance_strength(indicators)
            
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(indicators)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(indicators)
            
            return MarketCondition(
                sentiment=sentiment,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                volatility_level=volatility_level,
                volume_profile=volume_profile,
                support_strength=support_strength,
                resistance_strength=resistance_strength,
                momentum_score=momentum_score,
                confidence_score=confidence_score
            )
        
        except Exception as e:
            self.log_warning(f"Failed to assess market condition: {e}")
            # Return neutral condition as fallback
            return MarketCondition(
                sentiment=MarketSentiment.NEUTRAL,
                trend_direction='neutral',
                trend_strength=TrendStrength.MODERATE,
                volatility_level='medium',
                volume_profile='normal',
                support_strength=0.5,
                resistance_strength=0.5,
                momentum_score=0.5,
                confidence_score=0.5
            )
    
    def _generate_trading_signal(self, data: List[MarketData], indicators: Dict[str, Any], 
                               condition: MarketCondition) -> TradingSignal:
        """Generate trading signal based on analysis"""
        try:
            current_price = data[-1].close_price
            reasoning = []
            
            # Initialize signal components
            trend_signal = 0
            momentum_signal = 0
            volume_signal = 0
            technical_signal = 0
            
            # Trend analysis
            if condition.trend_direction == 'bullish':
                trend_signal = 1
                reasoning.append("Bullish trend detected")
            elif condition.trend_direction == 'bearish':
                trend_signal = -1
                reasoning.append("Bearish trend detected")
            
            # Momentum analysis
            if indicators.get('rsi_oversold'):
                momentum_signal += 0.5
                reasoning.append("RSI oversold condition")
            elif indicators.get('rsi_overbought'):
                momentum_signal -= 0.5
                reasoning.append("RSI overbought condition")
            
            if indicators.get('macd_bullish'):
                momentum_signal += 0.3
                reasoning.append("MACD bullish crossover")
            elif indicators.get('macd_cross_signal') == 'bearish_cross':
                momentum_signal -= 0.3
                reasoning.append("MACD bearish crossover")
            
            # Volume analysis
            if indicators.get('high_volume') and trend_signal > 0:
                volume_signal = 0.5
                reasoning.append("High volume supporting uptrend")
            elif indicators.get('high_volume') and trend_signal < 0:
                volume_signal = -0.5
                reasoning.append("High volume supporting downtrend")
            
            # Technical levels
            if indicators.get('price_below_bb_lower'):
                technical_signal += 0.4
                reasoning.append("Price below Bollinger Band lower")
            elif indicators.get('price_above_bb_upper'):
                technical_signal -= 0.4
                reasoning.append("Price above Bollinger Band upper")
            
            # Moving average signals
            if indicators.get('sma_cross_signal') == 'bullish_cross':
                technical_signal += 0.3
                reasoning.append("SMA bullish crossover")
            elif indicators.get('sma_cross_signal') == 'bearish_cross':
                technical_signal -= 0.3
                reasoning.append("SMA bearish crossover")
            
            # Calculate overall signal
            overall_signal = (
                trend_signal * self.weights['trend'] +
                momentum_signal * self.weights['momentum'] +
                volume_signal * self.weights['volume'] +
                technical_signal * 0.4
            )
            
            # Determine action
            if overall_signal > 0.3:
                action = 'buy'
            elif overall_signal < -0.3:
                action = 'sell'
            else:
                action = 'hold'
            
            # Calculate confidence
            confidence = min(95, max(5, abs(overall_signal) * 100 * condition.confidence_score))
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(condition, indicators)
            
            # Calculate entry, stop loss, and take profit
            entry_price = current_price
            stop_loss, take_profit = self._calculate_levels(current_price, action, indicators, condition)
            
            # Estimate expected return
            expected_return = self._estimate_expected_return(action, current_price, take_profit, stop_loss)
            
            return TradingSignal(
                action=action,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                risk_level=risk_level,
                time_horizon=self._determine_time_horizon(condition),
                expected_return=expected_return
            )
        
        except Exception as e:
            self.log_warning(f"Failed to generate trading signal: {e}")
            # Return neutral signal as fallback
            return TradingSignal(
                action='hold',
                confidence=50.0,
                entry_price=data[-1].close_price,
                stop_loss=None,
                take_profit=None,
                reasoning=['Analysis incomplete'],
                risk_level='medium',
                time_horizon='short',
                expected_return=None
            )
    
    def _calculate_sentiment(self, indicators: Dict[str, Any]) -> MarketSentiment:
        """Calculate market sentiment from indicators"""
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI signals
        if indicators.get('rsi_oversold'):
            bullish_signals += 1
        elif indicators.get('rsi_overbought'):
            bearish_signals += 1
        
        # MACD signals
        if indicators.get('macd_bullish'):
            bullish_signals += 1
        elif indicators.get('macd_cross_signal') == 'bearish_cross':
            bearish_signals += 1
        
        # Trend signals
        trend_direction = indicators.get('trend_direction', 'neutral')
        if trend_direction == 'bullish':
            bullish_signals += 2
        elif trend_direction == 'bearish':
            bearish_signals += 2
        
        # Moving average signals
        if indicators.get('price_above_sma_long'):
            bullish_signals += 1
        elif indicators.get('price_above_sma_long') is False:
            bearish_signals += 1
        
        # Determine sentiment
        net_signal = bullish_signals - bearish_signals
        
        if net_signal >= 3:
            return MarketSentiment.VERY_BULLISH
        elif net_signal >= 1:
            return MarketSentiment.BULLISH
        elif net_signal <= -3:
            return MarketSentiment.VERY_BEARISH
        elif net_signal <= -1:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.NEUTRAL
    
    def _calculate_trend_strength(self, indicators: Dict[str, Any]) -> TrendStrength:
        """Calculate trend strength"""
        consistency = indicators.get('trend_consistency', 0.5)
        
        if consistency >= 0.8:
            return TrendStrength.VERY_STRONG
        elif consistency >= 0.7:
            return TrendStrength.STRONG
        elif consistency >= 0.6:
            return TrendStrength.MODERATE
        elif consistency >= 0.4:
            return TrendStrength.WEAK
        else:
            return TrendStrength.VERY_WEAK
    
    def _analyze_volume_profile(self, indicators: Dict[str, Any]) -> str:
        """Analyze volume profile"""
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        if volume_ratio >= 2.0:
            return 'very_high'
        elif volume_ratio >= 1.5:
            return 'high'
        elif volume_ratio >= 0.8:
            return 'normal'
        elif volume_ratio >= 0.5:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_support_strength(self, indicators: Dict[str, Any]) -> float:
        """Calculate support level strength"""
        support_distance = indicators.get('support_distance')
        if support_distance is None:
            return 0.5
        
        # Closer support = stronger
        if support_distance <= 2:
            return 0.9
        elif support_distance <= 5:
            return 0.7
        elif support_distance <= 10:
            return 0.5
        else:
            return 0.3
    
    def _calculate_resistance_strength(self, indicators: Dict[str, Any]) -> float:
        """Calculate resistance level strength"""
        resistance_distance = indicators.get('resistance_distance')
        if resistance_distance is None:
            return 0.5
        
        # Closer resistance = stronger
        if resistance_distance <= 2:
            return 0.9
        elif resistance_distance <= 5:
            return 0.7
        elif resistance_distance <= 10:
            return 0.5
        else:
            return 0.3
    
    def _calculate_momentum_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate momentum score"""
        score = 0.5  # Neutral baseline
        
        # RSI contribution
        rsi = indicators.get('rsi')
        if rsi:
            if rsi > 70:
                score -= 0.2
            elif rsi < 30:
                score += 0.2
            elif rsi > 50:
                score += 0.1
            else:
                score -= 0.1
        
        # MACD contribution
        if indicators.get('macd_bullish'):
            score += 0.15
        elif indicators.get('macd_bullish') is False:
            score -= 0.15
        
        # Momentum contribution
        momentum = indicators.get('momentum')
        if momentum:
            score += min(0.2, max(-0.2, momentum * 10))
        
        return max(0, min(1, score))
    
    def _calculate_confidence_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate analysis confidence score"""
        confidence_factors = []
        
        # Data quality factors
        if indicators.get('atr') is not None:
            confidence_factors.append(0.8)  # Good volatility data
        
        if indicators.get('volume_avg') is not None:
            confidence_factors.append(0.8)  # Good volume data
        
        # Signal clarity factors
        if indicators.get('trend_direction') != 'neutral':
            confidence_factors.append(0.9)  # Clear trend
        
        if indicators.get('rsi_signal') in ['overbought', 'oversold']:
            confidence_factors.append(0.7)  # Clear RSI signal
        
        # Pattern confirmation
        patterns = [k for k, v in indicators.items() if k.endswith('_cross') and v]
        if patterns:
            confidence_factors.append(0.8)  # Pattern confirmation
        
        if not confidence_factors:
            return 0.5
        
        return min(0.95, np.mean(confidence_factors))
    
    def _calculate_overall_score(self, indicators: Dict[str, Any], condition: MarketCondition) -> float:
        """Calculate overall analysis score"""
        try:
            scores = []
            
            # Trend score
            if condition.trend_direction == 'bullish':
                scores.append(0.8)
            elif condition.trend_direction == 'bearish':
                scores.append(0.2)
            else:
                scores.append(0.5)
            
            # Momentum score
            scores.append(condition.momentum_score)
            
            # Volume score
            volume_profile = condition.volume_profile
            if volume_profile in ['high', 'very_high']:
                scores.append(0.7)
            elif volume_profile == 'normal':
                scores.append(0.6)
            else:
                scores.append(0.4)
            
            # Technical score
            technical_score = 0.5
            if indicators.get('rsi_oversold'):
                technical_score += 0.2
            elif indicators.get('rsi_overbought'):
                technical_score -= 0.2
            
            if indicators.get('macd_bullish'):
                technical_score += 0.1
            elif indicators.get('macd_bullish') is False:
                technical_score -= 0.1
            
            scores.append(max(0, min(1, technical_score)))
            
            # Confidence weight
            weighted_score = np.average(scores, weights=[0.3, 0.25, 0.2, 0.25])
            final_score = weighted_score * condition.confidence_score
            
            return round(final_score * 100, 2)
        
        except Exception as e:
            self.log_warning(f"Failed to calculate overall score: {e}")
            return 50.0
    
    def _calculate_risk_level(self, condition: MarketCondition, indicators: Dict[str, Any]) -> str:
        """Calculate risk level for the trade"""
        risk_factors = 0
        
        # Volatility risk
        if condition.volatility_level in ['high', 'very_high']:
            risk_factors += 2
        elif condition.volatility_level == 'medium':
            risk_factors += 1
        
        # Trend strength risk
        if condition.trend_strength in [TrendStrength.WEAK, TrendStrength.VERY_WEAK]:
            risk_factors += 2
        
        # Volume risk
        if condition.volume_profile in ['low', 'very_low']:
            risk_factors += 1
        
        # Technical risk
        if indicators.get('bb_squeeze'):
            risk_factors += 1
        
        if risk_factors >= 4:
            return 'high'
        elif risk_factors >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_levels(self, current_price: Decimal, action: str, 
                        indicators: Dict[str, Any], condition: MarketCondition) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate stop loss and take profit levels"""
        try:
            atr = indicators.get('atr')
            if not atr:
                return None, None
            
            atr_decimal = Decimal(str(atr))
            
            if action == 'buy':
                # Stop loss below current price
                stop_loss = current_price - (atr_decimal * Decimal('2'))
                # Take profit above current price
                take_profit = current_price + (atr_decimal * Decimal('3'))
            elif action == 'sell':
                # Stop loss above current price
                stop_loss = current_price + (atr_decimal * Decimal('2'))
                # Take profit below current price
                take_profit = current_price - (atr_decimal * Decimal('3'))
            else:
                return None, None
            
            return stop_loss, take_profit
        
        except Exception as e:
            self.log_warning(f"Failed to calculate levels: {e}")
            return None, None
    
    def _estimate_expected_return(self, action: str, entry_price: Decimal, 
                                take_profit: Optional[Decimal], stop_loss: Optional[Decimal]) -> Optional[float]:
        """Estimate expected return for the trade"""
        try:
            if not take_profit or not stop_loss:
                return None
            
            if action == 'buy':
                profit_potential = float((take_profit - entry_price) / entry_price * 100)
                loss_potential = float((entry_price - stop_loss) / entry_price * 100)
            elif action == 'sell':
                profit_potential = float((entry_price - take_profit) / entry_price * 100)
                loss_potential = float((stop_loss - entry_price) / entry_price * 100)
            else:
                return None
            
            # Simple expected return calculation (assuming 60% win rate)
            expected_return = (0.6 * profit_potential) - (0.4 * loss_potential)
            return round(expected_return, 2)
        
        except Exception as e:
            self.log_warning(f"Failed to estimate expected return: {e}")
            return None
    
    def _determine_time_horizon(self, condition: MarketCondition) -> str:
        """Determine appropriate time horizon for the trade"""
        if condition.trend_strength in [TrendStrength.VERY_STRONG, TrendStrength.STRONG]:
            return 'medium'  # 1-7 days
        elif condition.volatility_level in ['high', 'very_high']:
            return 'short'  # Hours to 1 day
        else:
            return 'short'  # Default to short term
    
    def _determine_overall_sentiment(self, sentiments: List[str]) -> str:
        """Determine overall market sentiment from individual sentiments"""
        bullish_count = sum(1 for s in sentiments if 'bullish' in s)
        bearish_count = sum(1 for s in sentiments if 'bearish' in s)
        
        total = len(sentiments)
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        
        if bullish_ratio >= 0.6:
            return 'bullish'
        elif bearish_ratio >= 0.6:
            return 'bearish'
        else:
            return 'neutral'