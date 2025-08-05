"""LLM-powered trading strategies"""

import json
import re
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from dataclasses import asdict

from .base import BaseStrategy, StrategyType, StrategyResult, StrategyConfig
from ..core.logger import get_logger
from ..core.exceptions import StrategyError, ValidationError, APIError
from ..data.models import MarketData, AnalysisResult
from ..market.indicators import TechnicalIndicators
from ..llm.bedrock_client import BedrockLLMClient


class LLMStrategyConfig(StrategyConfig):
    """Configuration for LLM strategies"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # AWS Bedrock settings
        self.aws_region = kwargs.get('aws_region', None)
        self.enable_llm = kwargs.get('enable_llm', None)
        self.cross_region_inference = kwargs.get('cross_region_inference', True)
        self.model_id = kwargs.get('model_id', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.temperature = kwargs.get('temperature', 0.3)
        self.timeout_seconds = kwargs.get('timeout_seconds', 30)
        
        # Analysis settings
        self.include_technical_analysis = kwargs.get('include_technical_analysis', True)
        self.include_market_sentiment = kwargs.get('include_market_sentiment', True)
        self.include_news_analysis = kwargs.get('include_news_analysis', False)
        self.lookback_periods = kwargs.get('lookback_periods', [24, 168, 720])  # 1d, 1w, 1m hours
        
        # Response validation
        self.require_reasoning = kwargs.get('require_reasoning', True)
        self.min_reasoning_length = kwargs.get('min_reasoning_length', 50)
        self.max_retries = kwargs.get('max_retries', 3)
        
        # Performance optimization
        self.cache_responses = kwargs.get('cache_responses', True)
        self.cache_ttl_minutes = kwargs.get('cache_ttl_minutes', 15)
        self.batch_analysis = kwargs.get('batch_analysis', False)


class LLMStrategy(BaseStrategy):
    """Base LLM-powered trading strategy using AWS Bedrock"""
    
    def __init__(self, name: str, config: Optional[LLMStrategyConfig] = None):
        super().__init__(name, StrategyType.LLM, config or LLMStrategyConfig())
        self.indicators = TechnicalIndicators()
        self.response_cache = {}
        
        # Initialize AWS Bedrock client
        try:
            self.llm_client = BedrockLLMClient(
                aws_region=self.config.aws_region,
                enable_llm=self.config.enable_llm,
                cross_region_inference=self.config.cross_region_inference
            )
            self.log_info(f"Initialized AWS Bedrock client for {name}")
        except Exception as e:
            self.log_error(f"Failed to initialize Bedrock client: {e}")
            self.llm_client = None
        
    def set_llm_client(self, client):
        """Set LLM client for API calls"""
        self.llm_client = client
    
    def analyze(self, symbol: str, market_data: List[MarketData], 
               additional_data: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Analyze using AWS Bedrock LLM"""
        try:
            if not self.llm_client or not self.llm_client.is_available():
                self.log_warning(f"LLM client not available for {symbol}, using fallback")
                return self._get_fallback_result(symbol, market_data)
            
            # Prepare analysis context
            context = self._prepare_analysis_context(symbol, market_data, additional_data)
            
            # Calculate technical indicators for LLM analysis
            technical_data = self._calculate_technical_indicators(market_data)
            
            # Get market sentiment if news data is available
            news_data = additional_data.get('news_data', []) if additional_data else []
            sentiment_analysis = self.llm_client.analyze_market_sentiment(
                symbol, news_data, self.config.cache_ttl_minutes
            )
            
            # Perform comprehensive trading signal analysis
            market_context = {
                'sentiment': sentiment_analysis,
                'technical_summary': self._summarize_technical_data(technical_data),
                'market_conditions': context.get('market_conditions', {}),
                'timeframe': context.get('timeframe', 'medium')
            }
            
            trading_analysis = self.llm_client.analyze_trading_signals(
                symbol, technical_data, market_context
            )
            
            # Process and validate the LLM response
            return self._process_llm_response(symbol, market_data, trading_analysis, context)
        
        except Exception as e:
            self.log_error(f"LLM analysis failed for {symbol}: {e}")
            return self._get_fallback_result(symbol, market_data)
    
    def validate_signal(self, signal, market_data: List[MarketData]) -> bool:
        """Validate LLM-generated signal"""
        # Basic validation
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check reasoning quality
        if self.config.require_reasoning:
            if not signal.reasoning or len(' '.join(signal.reasoning)) < self.config.min_reasoning_length:
                self.log_warning("LLM signal rejected: insufficient reasoning")
                return False
        
        # Validate price levels
        current_price = float(market_data[-1].close)
        if signal.entry_price:
            price_diff = abs(float(signal.entry_price) - current_price) / current_price
            if price_diff > 0.05:  # 5% difference seems unreasonable
                self.log_warning(f"LLM signal rejected: unreasonable entry price difference {price_diff:.1%}")
                return False
        
        return True
    
    def _prepare_analysis_context(self, symbol: str, market_data: List[MarketData], 
                                 additional_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare context for LLM analysis"""
        context = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(market_data)
        }
        
        # Current market state
        latest_data = market_data[-1]
        context['current_price'] = float(latest_data.close)
        context['current_volume'] = float(latest_data.volume)
        context['price_change_24h'] = self._calculate_price_change(market_data, 24)
        
        # Technical indicators if enabled
        if self.config.include_technical_analysis:
            context['technical_analysis'] = self._calculate_technical_indicators(market_data)
        
        # Market sentiment if enabled
        if self.config.include_market_sentiment:
            context['market_sentiment'] = self._analyze_market_sentiment(market_data)
        
        # Additional data
        if additional_data:
            context['additional_data'] = additional_data
        
        return context
    
    def _calculate_technical_indicators(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Calculate technical indicators for LLM context"""
        closes = [float(data.close) for data in market_data]
        volumes = [float(data.volume) for data in market_data]
        
        indicators = {}
        
        try:
            # RSI
            rsi_values = self.indicators.rsi(closes, 14)
            if rsi_values:
                indicators['rsi'] = {
                    'current': rsi_values[-1],
                    'trend': 'increasing' if len(rsi_values) > 1 and rsi_values[-1] > rsi_values[-2] else 'decreasing',
                    'overbought': rsi_values[-1] > 70,
                    'oversold': rsi_values[-1] < 30
                }
            
            # MACD
            macd_line, signal_line, histogram = self.indicators.macd(closes, 12, 26, 9)
            if macd_line:
                indicators['macd'] = {
                    'line': macd_line[-1],
                    'signal': signal_line[-1],
                    'histogram': histogram[-1],
                    'bullish_crossover': len(macd_line) > 1 and macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1],
                    'bearish_crossover': len(macd_line) > 1 and macd_line[-2] >= signal_line[-2] and macd_line[-1] < signal_line[-1]
                }
            
            # Bollinger Bands
            upper, middle, lower = self.indicators.bollinger_bands(closes, 20, 2)
            if upper:
                current_price = closes[-1]
                indicators['bollinger_bands'] = {
                    'upper': upper[-1],
                    'middle': middle[-1],
                    'lower': lower[-1],
                    'position': (current_price - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5,
                    'squeeze': (upper[-1] - lower[-1]) / middle[-1] < 0.1
                }
            
            # Moving averages
            sma_20 = self.indicators.sma(closes, 20)
            sma_50 = self.indicators.sma(closes, 50)
            if sma_20 and sma_50:
                indicators['moving_averages'] = {
                    'sma_20': sma_20[-1],
                    'sma_50': sma_50[-1],
                    'golden_cross': sma_20[-1] > sma_50[-1],
                    'price_above_sma20': closes[-1] > sma_20[-1],
                    'price_above_sma50': closes[-1] > sma_50[-1]
                }
            
            # Volume analysis
            if len(volumes) >= 20:
                avg_volume = sum(volumes[-20:]) / 20
                indicators['volume'] = {
                    'current': volumes[-1],
                    'average_20': avg_volume,
                    'above_average': volumes[-1] > avg_volume,
                    'volume_ratio': volumes[-1] / avg_volume if avg_volume > 0 else 1
                }
        
        except Exception as e:
            self.log_warning(f"Technical indicator calculation failed: {e}")
        
        return indicators
    
    def _analyze_market_sentiment(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze market sentiment from price action"""
        try:
            closes = [float(data.close) for data in market_data]
            highs = [float(data.high) for data in market_data]
            lows = [float(data.low) for data in market_data]
            
            sentiment = {}
            
            # Price momentum
            if len(closes) >= 10:
                recent_change = (closes[-1] - closes[-10]) / closes[-10]
                sentiment['momentum'] = {
                    'direction': 'bullish' if recent_change > 0 else 'bearish',
                    'strength': abs(recent_change),
                    'change_10_periods': recent_change
                }
            
            # Volatility
            if len(closes) >= 20:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = (sum(r**2 for r in returns[-20:]) / 20) ** 0.5
                sentiment['volatility'] = {
                    'level': 'high' if volatility > 0.03 else 'medium' if volatility > 0.015 else 'low',
                    'value': volatility
                }
            
            # Support/Resistance levels
            if len(lows) >= 20 and len(highs) >= 20:
                recent_low = min(lows[-20:])
                recent_high = max(highs[-20:])
                current_price = closes[-1]
                
                sentiment['levels'] = {
                    'support': recent_low,
                    'resistance': recent_high,
                    'distance_to_support': (current_price - recent_low) / current_price,
                    'distance_to_resistance': (recent_high - current_price) / current_price
                }
            
            return sentiment
        
        except Exception as e:
            self.log_warning(f"Sentiment analysis failed: {e}")
            return {}
    
    def _calculate_price_change(self, market_data: List[MarketData], periods: int) -> float:
        """Calculate price change over specified periods"""
        if len(market_data) < periods + 1:
            return 0.0
        
        current_price = float(market_data[-1].close)
        past_price = float(market_data[-(periods + 1)].close)
        
        return (current_price - past_price) / past_price
    
    def _generate_prompt(self, symbol: str, context: Dict[str, Any]) -> str:
        """Generate prompt for LLM analysis"""
        prompt = f"""You are an expert cryptocurrency trader analyzing {symbol}. Based on the following market data and technical analysis, provide a trading recommendation.

Current Market Data:
- Symbol: {symbol}
- Current Price: ${context['current_price']:.6f}
- 24h Change: {context.get('price_change_24h', 0):.2%}
- Current Volume: {context['current_volume']:,.0f}
- Data Points: {context['data_points']}
"""
        
        # Add technical analysis
        if 'technical_analysis' in context:
            prompt += "\nTechnical Analysis:\n"
            ta = context['technical_analysis']
            
            if 'rsi' in ta:
                rsi = ta['rsi']
                prompt += f"- RSI: {rsi['current']:.1f} ({'Overbought' if rsi['overbought'] else 'Oversold' if rsi['oversold'] else 'Neutral'})\n"
            
            if 'macd' in ta:
                macd = ta['macd']
                prompt += f"- MACD: {macd['line']:.6f} (Signal: {macd['signal']:.6f}, Histogram: {macd['histogram']:.6f})\n"
                if macd['bullish_crossover']:
                    prompt += "  * Bullish MACD crossover detected\n"
                elif macd['bearish_crossover']:
                    prompt += "  * Bearish MACD crossover detected\n"
            
            if 'bollinger_bands' in ta:
                bb = ta['bollinger_bands']
                prompt += f"- Bollinger Bands Position: {bb['position']:.2f} (0=lower band, 1=upper band)\n"
                if bb['squeeze']:
                    prompt += "  * Band squeeze detected - volatility breakout expected\n"
            
            if 'moving_averages' in ta:
                ma = ta['moving_averages']
                prompt += f"- Price vs SMA20: {'Above' if ma['price_above_sma20'] else 'Below'}\n"
                prompt += f"- Price vs SMA50: {'Above' if ma['price_above_sma50'] else 'Below'}\n"
                if ma['golden_cross']:
                    prompt += "  * Golden cross pattern (SMA20 > SMA50)\n"
            
            if 'volume' in ta:
                vol = ta['volume']
                prompt += f"- Volume: {vol['volume_ratio']:.1f}x average ({'Above' if vol['above_average'] else 'Below'} average)\n"
        
        # Add market sentiment
        if 'market_sentiment' in context:
            prompt += "\nMarket Sentiment:\n"
            sentiment = context['market_sentiment']
            
            if 'momentum' in sentiment:
                mom = sentiment['momentum']
                prompt += f"- Momentum: {mom['direction'].title()} (10-period change: {mom['change_10_periods']:.2%})\n"
            
            if 'volatility' in sentiment:
                vol = sentiment['volatility']
                prompt += f"- Volatility: {vol['level'].title()} ({vol['value']:.1%})\n"
            
            if 'levels' in sentiment:
                levels = sentiment['levels']
                prompt += f"- Support: ${levels['support']:.6f} ({levels['distance_to_support']:.1%} below current)\n"
                prompt += f"- Resistance: ${levels['resistance']:.6f} ({levels['distance_to_resistance']:.1%} above current)\n"
        
        # Add instructions
        prompt += """\n\nPlease provide your analysis in the following JSON format:
{
    "action": "buy" | "sell" | "hold",
    "confidence": <number between 0-100>,
    "entry_price": <suggested entry price>,
    "stop_loss": <suggested stop loss price>,
    "take_profit": <suggested take profit price>,
    "position_size": <suggested position size as percentage 0-1>,
    "reasoning": ["reason 1", "reason 2", "reason 3"],
    "risk_level": "low" | "medium" | "high",
    "time_horizon": "short" | "medium" | "long",
    "key_factors": ["factor 1", "factor 2"],
    "market_outlook": "bullish" | "bearish" | "neutral"
}

Consider:
1. Current technical indicators and their signals
2. Market sentiment and momentum
3. Support and resistance levels
4. Volume confirmation
5. Risk management principles
6. Current market conditions

Provide clear, actionable reasoning for your recommendation."""
        
        return prompt
    
    def _summarize_technical_data(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize technical data for LLM analysis"""
        try:
            summary = {
                'trend_indicators': {},
                'momentum_indicators': {},
                'volatility_indicators': {},
                'volume_indicators': {},
                'support_resistance': {}
            }
            
            # Trend indicators
            if 'sma_trend' in technical_data:
                summary['trend_indicators']['sma_trend'] = technical_data['sma_trend']
            if 'ema_trend' in technical_data:
                summary['trend_indicators']['ema_trend'] = technical_data['ema_trend']
            if 'macd' in technical_data:
                summary['trend_indicators']['macd_signal'] = technical_data['macd']
            
            # Momentum indicators
            if 'rsi' in technical_data:
                summary['momentum_indicators']['rsi'] = technical_data['rsi']
            if 'stochastic' in technical_data:
                summary['momentum_indicators']['stochastic'] = technical_data['stochastic']
            if 'williams_r' in technical_data:
                summary['momentum_indicators']['williams_r'] = technical_data['williams_r']
            
            # Volatility indicators
            if 'bollinger_bands' in technical_data:
                summary['volatility_indicators']['bollinger_bands'] = technical_data['bollinger_bands']
            if 'atr' in technical_data:
                summary['volatility_indicators']['atr'] = technical_data['atr']
            
            # Volume indicators
            if 'volume_trend' in technical_data:
                summary['volume_indicators']['volume_trend'] = technical_data['volume_trend']
            if 'obv' in technical_data:
                summary['volume_indicators']['obv'] = technical_data['obv']
            
            # Support and resistance
            if 'support_levels' in technical_data:
                summary['support_resistance']['support'] = technical_data['support_levels']
            if 'resistance_levels' in technical_data:
                summary['support_resistance']['resistance'] = technical_data['resistance_levels']
            
            return summary
            
        except Exception as e:
            self.log_warning(f"Failed to summarize technical data: {e}")
            return {}
    
    def _get_fallback_result(self, symbol: str, market_data: List[MarketData]) -> StrategyResult:
        """Get fallback result when LLM is unavailable"""
        try:
            # Use basic technical analysis as fallback
            technical_data = self._calculate_technical_indicators(market_data)
            latest_data = market_data[-1] if market_data else None
            
            if not latest_data:
                raise StrategyError("No market data available")
            
            # Simple fallback logic based on RSI
            rsi_value = technical_data.get('rsi', {}).get('current', 50)
            
            if rsi_value < 30:
                action = 'buy'
                confidence = 60
                reasoning = ["RSI oversold condition detected (fallback analysis)"]
            elif rsi_value > 70:
                action = 'sell'
                confidence = 60
                reasoning = ["RSI overbought condition detected (fallback analysis)"]
            else:
                action = 'hold'
                confidence = 40
                reasoning = ["Neutral conditions detected (fallback analysis)"]
            
            # Get current price
            current_price = Decimal(str(latest_data.close))
            
            # Create signal
            signal = self._create_signal(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                entry_price=current_price,
                stop_loss=current_price * Decimal('0.95') if action == 'buy' else current_price * Decimal('1.05'),
                take_profit=current_price * Decimal('1.05') if action == 'buy' else current_price * Decimal('0.95'),
                position_size=0.02,  # Conservative 2%
                metadata={
                    'strategy_type': 'llm_fallback',
                    'rsi': rsi_value,
                    'fallback_reason': 'LLM unavailable'
                }
            )
            
            # Create analysis result
            analysis = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                indicators=technical_data,
                patterns=[],
                signals=[action],
                confidence=confidence,
                metadata={
                    'strategy': 'LLM_Fallback',
                    'rsi': rsi_value
                }
            )
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                analysis=analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment={'risk_level': 'medium', 'fallback_mode': True},
                execution_priority=3
            )
            
        except Exception as e:
            self.log_error(f"Fallback analysis failed for {symbol}: {e}")
            # Return minimal safe result
            signal = self._create_signal(
                action='hold',
                confidence=0,
                reasoning=["Analysis failed, holding position for safety"],
                entry_price=Decimal('0'),
                stop_loss=None,
                take_profit=None,
                position_size=0.0,
                metadata={'strategy_type': 'emergency_fallback'}
            )
            
            analysis = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                indicators={},
                patterns=[],
                signals=['hold'],
                confidence=0,
                metadata={'strategy': 'Emergency_Fallback'}
            )
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                analysis=analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment={'risk_level': 'high', 'emergency_mode': True},
                execution_priority=1
            )
    
    def _validate_llm_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate LLM response data"""
        try:
            # Check required fields
            required_fields = ['action', 'confidence']
            for field in required_fields:
                if field not in response_data:
                    self.log_warning(f"Missing required field: {field}")
                    return False
            
            # Validate action
            if response_data['action'] not in ['buy', 'sell', 'hold']:
                self.log_warning(f"Invalid action: {response_data['action']}")
                return False
            
            # Validate confidence
            confidence = float(response_data['confidence'])
            if not 0 <= confidence <= 1:
                self.log_warning(f"Invalid confidence: {confidence}")
                return False
            
            return True
            
        except Exception as e:
            self.log_warning(f"Response validation failed: {e}")
            return False
    
    def _calculate_position_size(self, confidence: float, volatility: float, 
                               risk_level: str, suggested_size: float = 0.02) -> float:
        """Calculate position size based on confidence, volatility and risk"""
        try:
            # Base position size from LLM suggestion
            base_size = min(suggested_size, 0.1)  # Cap at 10%
            
            # Adjust for confidence
            confidence_factor = confidence if confidence <= 1 else confidence / 100
            size = base_size * confidence_factor
            
            # Adjust for volatility
            if volatility > 0.05:  # High volatility
                size *= 0.5
            elif volatility > 0.03:  # Medium volatility
                size *= 0.75
            
            # Adjust for risk level
            risk_multipliers = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.6
            }
            size *= risk_multipliers.get(risk_level, 1.0)
            
            # Ensure minimum and maximum bounds
            return max(0.001, min(size, 0.1))  # Between 0.1% and 10%
            
        except Exception as e:
            self.log_warning(f"Position size calculation failed: {e}")
            return 0.02  # Default 2%
    
    def _process_llm_response(self, symbol: str, market_data: List[MarketData], 
                             response_data: Dict[str, Any], context: Dict[str, Any]) -> StrategyResult:
        """Process AWS Bedrock LLM response and create strategy result"""
        try:
            # Validate response data
            if not self._validate_llm_response(response_data):
                self.log_warning(f"Invalid LLM response for {symbol}, using fallback")
                return self._get_fallback_result(symbol, market_data)
            
            # Extract signal information
            action = response_data.get('action', 'hold').lower()
            confidence = float(response_data.get('confidence', 0.5))
            reasoning = response_data.get('reasoning', 'No reasoning provided')
            
            # Risk assessment
            risk_data = self._assess_llm_risk(response_data, market_data)
            
            # Calculate position sizing based on LLM recommendation and risk
            suggested_size = response_data.get('position_size', 0.02)
            position_size = self._calculate_position_size(
                confidence, 
                risk_data.get('volatility', 0.02),
                response_data.get('risk_level', 'medium'),
                suggested_size
            )
            
            # Get current price
            current_price = Decimal(str(market_data[-1].close))
            
            # Extract price levels from LLM response
            entry_price = Decimal(str(response_data.get('entry_price', current_price)))
            stop_loss = None
            take_profit = None
            
            if response_data.get('stop_loss'):
                stop_loss = Decimal(str(response_data['stop_loss']))
            if response_data.get('take_profit'):
                take_profit = Decimal(str(response_data['take_profit']))
            
            # Create signal
            signal = self._create_signal(
                action=action,
                confidence=int(confidence * 100),
                reasoning=[reasoning] if isinstance(reasoning, str) else reasoning,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                metadata={
                    'llm_model': self.config.model_id,
                    'risk_level': response_data.get('risk_level', 'medium'),
                    'time_horizon': response_data.get('time_horizon', 'short'),
                    'key_signals': response_data.get('key_signals', []),
                    'market_conditions': response_data.get('market_conditions', 'unknown'),
                    'sentiment_score': context.get('sentiment', {}).get('sentiment_score', 0.0)
                }
            )
            
            # Create analysis result
            analysis = AnalysisResult(
                symbol=symbol,
                timestamp=datetime.now(),
                indicators=context.get('technical_indicators', {}),
                patterns=response_data.get('patterns', []),
                signals=[action],
                confidence=int(confidence * 100),
                metadata={
                    'strategy': 'AWS_Bedrock_LLM',
                    'model': self.config.model_id,
                    'sentiment': context.get('sentiment', {}),
                    'llm_reasoning': reasoning
                }
            )
            
            # Calculate execution priority
            priority = self._calculate_llm_priority(response_data)
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                analysis=analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment=risk_data,
                execution_priority=priority
            )
            
        except Exception as e:
            self.log_error(f"Failed to process LLM response for {symbol}: {e}")
            return self._get_fallback_result(symbol, market_data)
    
    def _calculate_dynamic_position_size(self, confidence: float, risk_assessment: Dict[str, Any], 
                                        market_volatility: float) -> float:
        """Calculate dynamic position size based on multiple factors"""
        try:
            # Base size from confidence
            base_size = min(confidence / 100 * 0.1, 0.1)  # Max 10%
            
            # Adjust for risk level
            risk_multipliers = {'low': 1.2, 'medium': 1.0, 'high': 0.6}
            risk_level = risk_assessment.get('risk_level', 'medium')
            size = base_size * risk_multipliers.get(risk_level, 1.0)
            
            # Adjust for volatility
            if market_volatility > 0.05:
                size *= 0.5
            elif market_volatility > 0.03:
                size *= 0.75
            
            return max(0.001, min(size, 0.1))
            
        except Exception as e:
            self.log_warning(f"Dynamic position sizing failed: {e}")
            return 0.02
    
    def _extract_market_signals(self, technical_data: Dict[str, Any]) -> List[str]:
        """Extract key market signals from technical data"""
        signals = []
        
        try:
            # RSI signals
            if 'rsi' in technical_data:
                rsi = technical_data['rsi']
                if rsi.get('oversold'):
                    signals.append('RSI_OVERSOLD')
                elif rsi.get('overbought'):
                    signals.append('RSI_OVERBOUGHT')
            
            # MACD signals
            if 'macd' in technical_data:
                macd = technical_data['macd']
                if macd.get('bullish_crossover'):
                    signals.append('MACD_BULLISH_CROSS')
                elif macd.get('bearish_crossover'):
                    signals.append('MACD_BEARISH_CROSS')
            
            # Bollinger Bands signals
            if 'bollinger_bands' in technical_data:
                bb = technical_data['bollinger_bands']
                if bb.get('squeeze'):
                    signals.append('BB_SQUEEZE')
                elif bb.get('position', 0.5) > 0.8:
                    signals.append('BB_UPPER_BAND')
                elif bb.get('position', 0.5) < 0.2:
                    signals.append('BB_LOWER_BAND')
            
            # Moving average signals
            if 'moving_averages' in technical_data:
                ma = technical_data['moving_averages']
                if ma.get('golden_cross'):
                    signals.append('GOLDEN_CROSS')
                if ma.get('price_above_sma20') and ma.get('price_above_sma50'):
                    signals.append('PRICE_ABOVE_MAS')
            
            # Volume signals
            if 'volume' in technical_data:
                vol = technical_data['volume']
                if vol.get('above_average') and vol.get('volume_ratio', 1) > 2:
                    signals.append('HIGH_VOLUME')
            
            return signals
            
        except Exception as e:
            self.log_warning(f"Signal extraction failed: {e}")
            return []
    
    def _assess_llm_risk(self, response_data: Dict[str, Any], market_data: List[MarketData]) -> Dict[str, Any]:
        """Assess risk based on LLM response and market data"""
        try:
            # Calculate volatility from recent market data
            if len(market_data) >= 20:
                prices = [float(data.close) for data in market_data[-20:]]
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                volatility = np.std(returns) if returns else 0.02
            else:
                volatility = 0.02
            
            # Get risk level from LLM
            risk_level = response_data.get('risk_level', 'medium')
            confidence = response_data.get('confidence', 50)
            action = response_data.get('action', 'hold')
            
            # Calculate risk score
            risk_scores = {'low': 0.3, 'medium': 0.5, 'high': 0.8}
            risk_score = risk_scores.get(risk_level, 0.5)
            
            # Adjust for volatility
            if volatility > 0.05:
                risk_score = min(risk_score + 0.2, 1.0)
            
            # Risk factors assessment
            risk_factors = []
            if risk_level == 'high':
                risk_factors.append("LLM assessed high risk")
            if confidence < 70:
                risk_factors.append("Low confidence signal")
            if volatility > 0.03:
                risk_factors.append("High market volatility")
            if action != 'hold' and not response_data.get('reasoning'):
                risk_factors.append("Insufficient reasoning provided")
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'volatility': volatility,
                'risk_factors': risk_factors,
                'max_position_size': 0.1 if risk_level == 'low' else 0.05 if risk_level == 'medium' else 0.02,
                'recommended_stop_loss': volatility * 2,
                'confidence_threshold': 0.7 if risk_level == 'high' else 0.5,
                'llm_risk_assessment': risk_level,
                'confidence_factor': confidence / 100,
                'recommended_position_size': response_data.get('position_size', 0.05),
                'time_horizon': response_data.get('time_horizon', 'short')
            }
            
        except Exception as e:
            self.log_warning(f"Risk assessment failed: {e}")
            return {
                'risk_level': 'medium',
                'risk_score': 0.5,
                'volatility': 0.02,
                'risk_factors': ['Assessment failed'],
                'max_position_size': 0.05,
                'recommended_stop_loss': 0.04,
                'confidence_threshold': 0.5,
                'error': str(e)
            }
    
    def _calculate_llm_priority(self, response_data: Dict[str, Any]) -> int:
        """Calculate execution priority based on LLM response"""
        try:
            base_priority = 50
            
            # Adjust for confidence
            confidence = response_data.get('confidence', 0.5)
            if confidence > 1:  # Handle percentage format
                confidence = confidence / 100
            confidence_bonus = int((confidence - 0.5) * 50)
            
            # Adjust for action type
            action = response_data.get('action', 'hold')
            if action in ['buy', 'sell']:
                action_bonus = 20
            else:
                action_bonus = 0
            
            # Adjust for risk level
            risk_level = response_data.get('risk_level', 'medium')
            if risk_level == 'low':
                risk_bonus = 10
            elif risk_level == 'high':
                risk_bonus = -10
            else:
                risk_bonus = 0
            
            # Adjust for time horizon
            time_horizon = response_data.get('time_horizon', 'short')
            if time_horizon == 'immediate':
                time_bonus = 30
            elif time_horizon == 'short':
                time_bonus = 10
            else:
                time_bonus = 0
            
            # Adjust for signal strength
            key_signals = response_data.get('key_signals', [])
            signal_bonus = min(len(key_signals) * 5, 20)
            
            priority = base_priority + confidence_bonus + action_bonus + risk_bonus + time_bonus + signal_bonus
            return max(1, min(priority, 100))
            
        except Exception as e:
            self.log_warning(f"Priority calculation failed: {e}")
            return 50
    
    def _generate_cache_key(self, symbol: str, context: Dict[str, Any]) -> str:
        """Generate cache key for response caching"""
        # Create a hash of relevant context data
        cache_data = {
            'symbol': symbol,
            'price': context.get('current_price'),
            'volume': context.get('current_volume'),
            'data_points': context.get('data_points')
        }
        
        # Add technical indicators if available
        if 'technical_analysis' in context:
            ta = context['technical_analysis']
            if 'rsi' in ta:
                cache_data['rsi'] = round(ta['rsi']['current'], 1)
            if 'macd' in ta:
                cache_data['macd'] = round(ta['macd']['line'], 6)
        
        return f"{symbol}_{hash(str(sorted(cache_data.items())))}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached response is still valid"""
        age = datetime.now() - timestamp
        return age.total_seconds() < self.config.cache_ttl_minutes * 60
    
    def get_min_data_points(self) -> int:
        """Get minimum data points for LLM analysis"""
        return 50  # Need sufficient data for technical analysis


class FastLLMStrategy(LLMStrategy):
    """Fast LLM strategy with optimized prompts"""
    
    def __init__(self, config: Optional[LLMStrategyConfig] = None):
        config = config or LLMStrategyConfig()
        config.max_tokens = 1000  # Shorter responses
        config.temperature = 0.1  # More deterministic
        config.cache_ttl_minutes = 30  # Longer cache
        super().__init__("Fast_LLM_Strategy", config)
    
    def _generate_prompt(self, symbol: str, context: Dict[str, Any]) -> str:
        """Generate optimized prompt for fast analysis"""
        prompt = f"""Quick trading analysis for {symbol}:
Price: ${context['current_price']:.6f}
24h Change: {context.get('price_change_24h', 0):.2%}
"""
        
        # Add key technical indicators only
        if 'technical_analysis' in context:
            ta = context['technical_analysis']
            if 'rsi' in ta:
                prompt += f"RSI: {ta['rsi']['current']:.1f}\n"
            if 'macd' in ta and ta['macd'].get('bullish_crossover'):
                prompt += "MACD: Bullish crossover\n"
            elif 'macd' in ta and ta['macd'].get('bearish_crossover'):
                prompt += "MACD: Bearish crossover\n"
        
        prompt += """\nProvide JSON response:
{"action": "buy/sell/hold", "confidence": 0-100, "reasoning": ["brief reason"]}

Focus on: RSI levels, MACD signals, price momentum. Be concise."""
        
        return prompt


class FullLLMStrategy(LLMStrategy):
    """Comprehensive LLM strategy with detailed analysis"""
    
    def __init__(self, config: Optional[LLMStrategyConfig] = None):
        config = config or LLMStrategyConfig()
        config.max_tokens = 3000  # Longer responses
        config.temperature = 0.3  # More creative
        config.include_news_analysis = True
        config.min_reasoning_length = 100
        super().__init__("Full_LLM_Strategy", config)
    
    def _generate_prompt(self, symbol: str, context: Dict[str, Any]) -> str:
        """Generate comprehensive prompt for detailed analysis"""
        # Use the base prompt and extend it
        base_prompt = super()._generate_prompt(symbol, context)
        
        # Add additional analysis requirements
        extended_prompt = base_prompt + """\n\nAdditional Analysis Requirements:
1. Provide detailed market structure analysis
2. Consider multiple timeframe perspectives
3. Analyze volume patterns and their significance
4. Assess potential catalysts or market events
5. Compare with broader market trends
6. Provide risk-adjusted position sizing recommendations
7. Include alternative scenarios (bull/bear cases)

Provide comprehensive reasoning with at least 5 detailed points."""
        
        return extended_prompt