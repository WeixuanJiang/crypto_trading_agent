"""Hybrid trading strategies combining multiple approaches"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
import numpy as np
from dataclasses import asdict

from .base import BaseStrategy, StrategyType, StrategyResult, StrategyConfig, StrategySignal
from .technical import TechnicalStrategyConfig, RSIStrategy, MACDStrategy, BollingerBandsStrategy
from .llm import LLMStrategy, LLMStrategyConfig
from ..core.logger import get_logger
from ..core.exceptions import StrategyError, ValidationError
from ..data.models import MarketData, AnalysisResult


class HybridStrategyConfig(StrategyConfig):
    """Configuration for hybrid strategies"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Strategy weights
        self.technical_weight = kwargs.get('technical_weight', 0.6)
        self.llm_weight = kwargs.get('llm_weight', 0.4)
        self.sentiment_weight = kwargs.get('sentiment_weight', 0.2)
        
        # Consensus requirements
        self.require_consensus = kwargs.get('require_consensus', True)
        self.min_consensus_threshold = kwargs.get('min_consensus_threshold', 0.7)
        self.max_strategy_disagreement = kwargs.get('max_strategy_disagreement', 30)  # confidence points
        
        # Adaptive weighting
        self.enable_adaptive_weights = kwargs.get('enable_adaptive_weights', True)
        self.performance_lookback = kwargs.get('performance_lookback', 50)
        self.weight_adjustment_factor = kwargs.get('weight_adjustment_factor', 0.1)
        
        # Risk management
        self.enable_risk_override = kwargs.get('enable_risk_override', True)
        self.max_combined_risk = kwargs.get('max_combined_risk', 0.15)  # 15% max risk
        self.volatility_adjustment = kwargs.get('volatility_adjustment', True)
        
        # Signal filtering
        self.enable_signal_filtering = kwargs.get('enable_signal_filtering', True)
        self.min_signal_strength = kwargs.get('min_signal_strength', 2)  # Number of agreeing strategies
        self.confidence_boost_factor = kwargs.get('confidence_boost_factor', 1.2)
        
        # Sub-strategy configs
        self.technical_config = TechnicalStrategyConfig(**kwargs.get('technical_config', {}))
        self.llm_config = LLMStrategyConfig(**kwargs.get('llm_config', {}))


class TechnicalLLMHybrid(BaseStrategy):
    """Hybrid strategy combining technical analysis with LLM insights"""
    
    def __init__(self, config: Optional[HybridStrategyConfig] = None):
        super().__init__("Technical_LLM_Hybrid", StrategyType.HYBRID, config or HybridStrategyConfig())
        
        # Initialize sub-strategies
        self.technical_strategies = {
            'rsi': RSIStrategy(self.config.technical_config),
            'macd': MACDStrategy(self.config.technical_config),
            'bb': BollingerBandsStrategy(self.config.technical_config)
        }
        
        self.llm_strategy = LLMStrategy(self.config.llm_config)
        
        # Performance tracking for adaptive weights
        self.strategy_performance = {
            'technical': [],
            'llm': [],
            'combined': []
        }
        
        # Current adaptive weights
        self.current_weights = {
            'technical': self.config.technical_weight,
            'llm': self.config.llm_weight
        }
    
    def set_llm_client(self, client):
        """Set LLM client for the LLM strategy"""
        self.llm_strategy.set_llm_client(client)
    
    def analyze(self, symbol: str, market_data: List[MarketData], 
               additional_data: Optional[Dict[str, Any]] = None) -> StrategyResult:
        """Analyze using hybrid approach"""
        try:
            # Update adaptive weights if enabled
            if self.config.enable_adaptive_weights:
                self._update_adaptive_weights()
            
            # Get technical analysis signals
            technical_results = self._get_technical_signals(symbol, market_data, additional_data)
            
            # Get LLM analysis
            llm_result = self._get_llm_signal(symbol, market_data, additional_data)
            
            # Combine signals
            combined_signal = self._combine_signals(technical_results, llm_result, market_data)
            
            # Apply risk management
            if self.config.enable_risk_override:
                combined_signal = self._apply_risk_management(combined_signal, market_data)
            
            # Create comprehensive analysis
            combined_analysis = self._create_combined_analysis(
                symbol, technical_results, llm_result, combined_signal
            )
            
            return StrategyResult(
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=combined_signal,
                analysis=combined_analysis,
                performance_metrics=self.get_performance_metrics(),
                risk_assessment=self._assess_hybrid_risk(technical_results, llm_result, market_data),
                execution_priority=self._calculate_hybrid_priority(combined_signal, technical_results, llm_result)
            )
        
        except Exception as e:
            raise StrategyError(f"Hybrid analysis failed for {symbol}: {e}")
    
    def validate_signal(self, signal, market_data: List[MarketData]) -> bool:
        """Validate hybrid signal"""
        # Basic validation
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check consensus if required
        if self.config.require_consensus:
            consensus_score = signal.metadata.get('consensus_score', 0)
            if consensus_score < self.config.min_consensus_threshold:
                self.log_debug(f"Signal rejected: low consensus {consensus_score:.2f}")
                return False
        
        # Check strategy agreement
        strategy_disagreement = signal.metadata.get('strategy_disagreement', 0)
        if strategy_disagreement > self.config.max_strategy_disagreement:
            self.log_debug(f"Signal rejected: high disagreement {strategy_disagreement}")
            return False
        
        return True
    
    def _get_technical_signals(self, symbol: str, market_data: List[MarketData], 
                              additional_data: Optional[Dict[str, Any]]) -> Dict[str, StrategyResult]:
        """Get signals from all technical strategies"""
        results = {}
        
        for name, strategy in self.technical_strategies.items():
            try:
                if strategy.can_generate_signal():
                    result = strategy.analyze(symbol, market_data, additional_data)
                    if result and strategy.validate_signal(result.signal, market_data):
                        results[name] = result
                        self.log_debug(f"Technical {name}: {result.signal.action} ({result.signal.confidence:.1f}%)")
            except Exception as e:
                self.log_warning(f"Technical strategy {name} failed: {e}")
        
        return results
    
    def _get_llm_signal(self, symbol: str, market_data: List[MarketData], 
                       additional_data: Optional[Dict[str, Any]]) -> Optional[StrategyResult]:
        """Get signal from LLM strategy"""
        try:
            if self.llm_strategy.can_generate_signal():
                result = self.llm_strategy.analyze(symbol, market_data, additional_data)
                if result and self.llm_strategy.validate_signal(result.signal, market_data):
                    self.log_debug(f"LLM: {result.signal.action} ({result.signal.confidence:.1f}%)")
                    return result
        except Exception as e:
            self.log_warning(f"LLM strategy failed: {e}")
        
        return None
    
    def _combine_signals(self, technical_results: Dict[str, StrategyResult], 
                        llm_result: Optional[StrategyResult], 
                        market_data: List[MarketData]) -> StrategySignal:
        """Combine technical and LLM signals"""
        # Aggregate technical signals
        technical_signal = self._aggregate_technical_signals(technical_results)
        
        # Determine final action and confidence
        if llm_result is None:
            # Use only technical signals
            final_action = technical_signal['action']
            final_confidence = technical_signal['confidence']
            reasoning = technical_signal['reasoning']
            consensus_score = 1.0 if len(technical_results) > 0 else 0.0
            strategy_disagreement = 0
        else:
            # Combine technical and LLM signals
            final_action, final_confidence, reasoning, consensus_score, strategy_disagreement = \
                self._weighted_signal_combination(technical_signal, llm_result.signal)
        
        # Apply signal filtering
        if self.config.enable_signal_filtering:
            final_confidence = self._apply_signal_filtering(
                final_confidence, technical_results, llm_result
            )
        
        # Calculate position size and price levels
        current_price = Decimal(str(market_data[-1].close))
        position_size = self._calculate_hybrid_position_size(final_confidence, consensus_score)
        
        # Create combined signal
        signal = self._create_signal(
            action=final_action,
            confidence=final_confidence,
            reasoning=reasoning,
            entry_price=current_price,
            position_size=position_size,
            metadata={
                'hybrid_strategy': True,
                'technical_weight': self.current_weights['technical'],
                'llm_weight': self.current_weights['llm'],
                'consensus_score': consensus_score,
                'strategy_disagreement': strategy_disagreement,
                'technical_signals': {name: {
                    'action': result.signal.action,
                    'confidence': result.signal.confidence
                } for name, result in technical_results.items()},
                'llm_signal': {
                    'action': llm_result.signal.action,
                    'confidence': llm_result.signal.confidence
                } if llm_result else None,
                'adaptive_weights_used': self.config.enable_adaptive_weights
            }
        )
        
        return signal
    
    def _aggregate_technical_signals(self, technical_results: Dict[str, StrategyResult]) -> Dict[str, Any]:
        """Aggregate multiple technical signals"""
        if not technical_results:
            return {
                'action': 'hold',
                'confidence': 50.0,
                'reasoning': ['No technical signals available']
            }
        
        # Count actions
        actions = [result.signal.action for result in technical_results.values()]
        action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for action in actions:
            action_counts[action] += 1
        
        # Determine majority action
        majority_action = max(action_counts, key=action_counts.get)
        
        # Calculate weighted confidence
        relevant_signals = [result.signal for result in technical_results.values() 
                          if result.signal.action == majority_action]
        
        if relevant_signals:
            confidences = [signal.confidence for signal in relevant_signals]
            avg_confidence = np.mean(confidences)
            
            # Boost confidence if multiple strategies agree
            agreement_boost = min(10 * (len(relevant_signals) - 1), 20)
            final_confidence = min(avg_confidence + agreement_boost, 95)
        else:
            final_confidence = 50.0
        
        # Combine reasoning
        reasoning = []
        for name, result in technical_results.items():
            if result.signal.action == majority_action:
                reasoning.extend([f"{name.upper()}: {r}" for r in result.signal.reasoning[:2]])
        
        return {
            'action': majority_action,
            'confidence': final_confidence,
            'reasoning': reasoning
        }
    
    def _weighted_signal_combination(self, technical_signal: Dict[str, Any], 
                                   llm_signal: StrategySignal) -> Tuple[str, float, List[str], float, float]:
        """Combine technical and LLM signals using weighted approach"""
        tech_action = technical_signal['action']
        tech_confidence = technical_signal['confidence']
        llm_action = llm_signal.action
        llm_confidence = llm_signal.confidence
        
        # Calculate strategy disagreement
        if tech_action == llm_action:
            strategy_disagreement = abs(tech_confidence - llm_confidence)
            consensus_score = 1.0
        else:
            strategy_disagreement = max(tech_confidence, llm_confidence)
            consensus_score = 0.3  # Low consensus when strategies disagree
        
        # Weighted action selection
        tech_weight = self.current_weights['technical']
        llm_weight = self.current_weights['llm']
        
        # If actions agree, use the agreed action
        if tech_action == llm_action:
            final_action = tech_action
            # Weighted confidence
            final_confidence = (tech_confidence * tech_weight + llm_confidence * llm_weight)
            # Boost for agreement
            final_confidence = min(final_confidence * self.config.confidence_boost_factor, 95)
        else:
            # Actions disagree - use higher weighted confidence
            tech_weighted = tech_confidence * tech_weight
            llm_weighted = llm_confidence * llm_weight
            
            if tech_weighted > llm_weighted:
                final_action = tech_action
                final_confidence = tech_confidence * 0.8  # Reduce confidence due to disagreement
            else:
                final_action = llm_action
                final_confidence = llm_confidence * 0.8
            
            # If disagreement is too high, default to hold
            if strategy_disagreement > self.config.max_strategy_disagreement:
                final_action = 'hold'
                final_confidence = 50.0
        
        # Combine reasoning
        reasoning = []
        reasoning.extend([f"Technical: {r}" for r in technical_signal['reasoning'][:3]])
        reasoning.extend([f"LLM: {r}" for r in llm_signal.reasoning[:3]])
        reasoning.append(f"Consensus: {consensus_score:.1f}, Disagreement: {strategy_disagreement:.1f}")
        
        return final_action, final_confidence, reasoning, consensus_score, strategy_disagreement
    
    def _apply_signal_filtering(self, confidence: float, technical_results: Dict[str, StrategyResult], 
                               llm_result: Optional[StrategyResult]) -> float:
        """Apply signal filtering to adjust confidence"""
        # Count agreeing strategies
        all_signals = list(technical_results.values())
        if llm_result:
            all_signals.append(llm_result)
        
        if len(all_signals) < self.config.min_signal_strength:
            return confidence * 0.8  # Reduce confidence for weak signals
        
        # Check for strong agreement
        actions = [result.signal.action for result in all_signals]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        max_agreement = max(action_counts.values()) if action_counts else 0
        
        if max_agreement >= 3:  # Strong agreement
            return min(confidence * 1.1, 95)
        elif max_agreement >= 2:  # Moderate agreement
            return confidence
        else:  # Weak agreement
            return confidence * 0.9
    
    def _calculate_hybrid_position_size(self, confidence: float, consensus_score: float) -> float:
        """Calculate position size for hybrid strategy"""
        base_size = self.config.max_position_size
        
        # Adjust for confidence
        confidence_factor = confidence / 100.0
        
        # Adjust for consensus
        consensus_factor = consensus_score
        
        # Adjust for risk tolerance
        risk_multiplier = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5
        }.get(self.config.risk_tolerance, 1.0)
        
        position_size = base_size * confidence_factor * consensus_factor * risk_multiplier
        
        return min(position_size, self.config.max_position_size)
    
    def _apply_risk_management(self, signal: StrategySignal, market_data: List[MarketData]) -> StrategySignal:
        """Apply risk management overrides"""
        try:
            # Calculate current market volatility
            closes = [float(data.close) for data in market_data[-20:]]
            if len(closes) >= 2:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = np.std(returns)
            else:
                volatility = 0.0
            
            # Adjust position size for volatility
            if self.config.volatility_adjustment and volatility > 0.03:  # High volatility
                signal.position_size *= 0.7  # Reduce position size
                signal.reasoning.append(f"Position size reduced due to high volatility ({volatility:.1%})")
            
            # Check maximum combined risk
            total_risk = signal.position_size * (1 - signal.confidence / 100)
            if total_risk > self.config.max_combined_risk:
                reduction_factor = self.config.max_combined_risk / total_risk
                signal.position_size *= reduction_factor
                signal.reasoning.append(f"Position size reduced to limit total risk to {self.config.max_combined_risk:.1%}")
            
            # Override for very low confidence
            if signal.confidence < 60 and signal.action != 'hold':
                signal.action = 'hold'
                signal.confidence = 50.0
                signal.reasoning.append("Risk override: confidence too low for trading")
            
            return signal
        
        except Exception as e:
            self.log_warning(f"Risk management failed: {e}")
            return signal
    
    def _create_combined_analysis(self, symbol: str, technical_results: Dict[str, StrategyResult], 
                                 llm_result: Optional[StrategyResult], 
                                 combined_signal: StrategySignal) -> AnalysisResult:
        """Create comprehensive analysis result"""
        # Combine indicators from all strategies
        combined_indicators = {}
        
        for name, result in technical_results.items():
            for key, value in result.analysis.indicators.items():
                combined_indicators[f"{name}_{key}"] = value
        
        if llm_result:
            for key, value in llm_result.analysis.indicators.items():
                combined_indicators[f"llm_{key}"] = value
        
        # Combine patterns
        combined_patterns = []
        for result in technical_results.values():
            combined_patterns.extend(result.analysis.patterns)
        
        if llm_result:
            combined_patterns.extend(llm_result.analysis.patterns)
        
        # Combine signals
        all_signals = [combined_signal.action]
        for result in technical_results.values():
            all_signals.extend(result.analysis.signals)
        
        if llm_result:
            all_signals.extend(llm_result.analysis.signals)
        
        return AnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            indicators=combined_indicators,
            patterns=combined_patterns,
            signals=all_signals,
            confidence=combined_signal.confidence,
            metadata={
                'strategy': 'Hybrid',
                'sub_strategies': list(technical_results.keys()) + (['LLM'] if llm_result else []),
                'weights': self.current_weights,
                'consensus_score': combined_signal.metadata.get('consensus_score', 0),
                'strategy_disagreement': combined_signal.metadata.get('strategy_disagreement', 0)
            }
        )
    
    def _assess_hybrid_risk(self, technical_results: Dict[str, StrategyResult], 
                           llm_result: Optional[StrategyResult], 
                           market_data: List[MarketData]) -> Dict[str, Any]:
        """Assess risk for hybrid strategy"""
        try:
            # Aggregate risk assessments from sub-strategies
            risk_levels = []
            
            for result in technical_results.values():
                risk_level = result.risk_assessment.get('risk_level', 'medium')
                risk_levels.append(risk_level)
            
            if llm_result:
                risk_level = llm_result.risk_assessment.get('risk_level', 'medium')
                risk_levels.append(risk_level)
            
            # Determine overall risk level
            risk_counts = {'low': 0, 'medium': 0, 'high': 0}
            for level in risk_levels:
                risk_counts[level] += 1
            
            if risk_counts['high'] > 0:
                overall_risk = 'high'
            elif risk_counts['medium'] > risk_counts['low']:
                overall_risk = 'medium'
            else:
                overall_risk = 'low'
            
            # Calculate market volatility
            closes = [float(data.close) for data in market_data[-20:]]
            volatility = np.std(closes) / np.mean(closes) if len(closes) >= 2 else 0
            
            return {
                'risk_level': overall_risk,
                'sub_strategy_risks': risk_levels,
                'volatility': volatility,
                'consensus_risk': 'low' if len(set(risk_levels)) == 1 else 'high',
                'strategy_count': len(technical_results) + (1 if llm_result else 0),
                'recommended_position_size': min(0.1, self.config.max_position_size)
            }
        
        except Exception as e:
            self.log_warning(f"Hybrid risk assessment failed: {e}")
            return {'risk_level': 'high', 'error': str(e)}
    
    def _calculate_hybrid_priority(self, combined_signal: StrategySignal, 
                                  technical_results: Dict[str, StrategyResult], 
                                  llm_result: Optional[StrategyResult]) -> int:
        """Calculate execution priority for hybrid strategy"""
        try:
            base_priority = int(combined_signal.confidence / 10)
            
            # Boost for consensus
            consensus_score = combined_signal.metadata.get('consensus_score', 0)
            if consensus_score >= 0.8:
                base_priority += 2
            elif consensus_score >= 0.6:
                base_priority += 1
            
            # Boost for multiple agreeing strategies
            strategy_count = len(technical_results) + (1 if llm_result else 0)
            if strategy_count >= 3:
                base_priority += 1
            
            # Reduce for high disagreement
            disagreement = combined_signal.metadata.get('strategy_disagreement', 0)
            if disagreement > 20:
                base_priority -= 1
            
            return max(1, min(base_priority, 10))
        
        except Exception:
            return 5
    
    def _update_adaptive_weights(self):
        """Update strategy weights based on recent performance"""
        try:
            if len(self.strategy_performance['technical']) < 10:
                return  # Need more data
            
            # Calculate recent performance
            recent_tech = self.strategy_performance['technical'][-self.config.performance_lookback:]
            recent_llm = self.strategy_performance['llm'][-self.config.performance_lookback:]
            
            if not recent_tech or not recent_llm:
                return
            
            tech_performance = np.mean([p['return'] for p in recent_tech])
            llm_performance = np.mean([p['return'] for p in recent_llm])
            
            # Adjust weights based on relative performance
            if tech_performance > llm_performance:
                adjustment = self.config.weight_adjustment_factor
                self.current_weights['technical'] = min(0.8, self.current_weights['technical'] + adjustment)
                self.current_weights['llm'] = max(0.2, self.current_weights['llm'] - adjustment)
            else:
                adjustment = self.config.weight_adjustment_factor
                self.current_weights['llm'] = min(0.8, self.current_weights['llm'] + adjustment)
                self.current_weights['technical'] = max(0.2, self.current_weights['technical'] - adjustment)
            
            # Normalize weights
            total_weight = self.current_weights['technical'] + self.current_weights['llm']
            self.current_weights['technical'] /= total_weight
            self.current_weights['llm'] /= total_weight
            
            self.log_debug(f"Adaptive weights updated: Technical={self.current_weights['technical']:.2f}, LLM={self.current_weights['llm']:.2f}")
        
        except Exception as e:
            self.log_warning(f"Adaptive weight update failed: {e}")
    
    def update_performance(self, signal_result: Dict[str, Any]):
        """Update performance for adaptive weighting"""
        super().update_performance(signal_result)
        
        # Track sub-strategy performance
        if 'sub_strategy_results' in signal_result:
            sub_results = signal_result['sub_strategy_results']
            
            if 'technical' in sub_results:
                self.strategy_performance['technical'].append(sub_results['technical'])
            
            if 'llm' in sub_results:
                self.strategy_performance['llm'].append(sub_results['llm'])
        
        # Limit history size
        max_history = self.config.performance_lookback * 2
        for key in self.strategy_performance:
            if len(self.strategy_performance[key]) > max_history:
                self.strategy_performance[key] = self.strategy_performance[key][-max_history:]
    
    def get_min_data_points(self) -> int:
        """Get minimum data points for hybrid analysis"""
        return max(50, max(strategy.get_min_data_points() for strategy in self.technical_strategies.values()))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics for hybrid strategy"""
        base_metrics = super().get_performance_metrics()
        
        # Add hybrid-specific metrics
        base_metrics.update({
            'current_weights': self.current_weights.copy(),
            'adaptive_weighting_enabled': self.config.enable_adaptive_weights,
            'sub_strategy_performance': {
                'technical': len(self.strategy_performance['technical']),
                'llm': len(self.strategy_performance['llm'])
            },
            'consensus_requirements': {
                'enabled': self.config.require_consensus,
                'threshold': self.config.min_consensus_threshold
            }
        })
        
        return base_metrics