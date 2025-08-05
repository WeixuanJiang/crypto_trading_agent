"""AWS Bedrock LLM Client for trading strategies"""

import boto3
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from ..core.logger import get_logger
from ..core.exceptions import APIError, ConfigurationError


class BedrockLLMClient:
    """AWS Bedrock client for LLM operations"""
    
    def __init__(self, aws_region: str = None, enable_llm: bool = None, 
                 cross_region_inference: bool = None):
        """Initialize Bedrock client
        
        Args:
            aws_region: AWS region for primary client
            enable_llm: Whether to enable LLM functionality
            cross_region_inference: Whether to use cross-region fallback
        """
        load_dotenv()
        self.logger = get_logger(__name__)
        
        # Configuration from environment with fallbacks
        self.aws_region = aws_region or os.getenv('AWS_REGION', 'us-east-1')
        self.enable_llm = enable_llm if enable_llm is not None else (
            os.getenv('FAST_MODE', 'true').lower() != 'true'
        )
        self.cross_region_inference = cross_region_inference if cross_region_inference is not None else (
            os.getenv('AWS_CROSS_REGION_INFERENCE', 'true').lower() == 'true'
        )
        
        # Model configuration
        self.model_id = os.getenv('LLM_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '1000'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.3'))
        
        # Initialize clients
        self.bedrock_client = None
        self.fallback_clients = {}
        self.response_cache = {}
        
        if self.enable_llm:
            self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize primary and fallback Bedrock clients"""
        try:
            # Primary region client
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=self.aws_region
            )
            self.logger.info(f"Initialized primary Bedrock client in {self.aws_region}")
            
            # Cross-region fallback clients for better availability
            if self.cross_region_inference:
                fallback_regions = ['us-west-2', 'eu-west-1', 'ap-southeast-1']
                for region in fallback_regions:
                    if region != self.aws_region:
                        try:
                            self.fallback_clients[region] = boto3.client(
                                'bedrock-runtime',
                                region_name=region
                            )
                            self.logger.debug(f"Initialized fallback client for {region}")
                        except Exception as e:
                            self.logger.warning(f"Could not initialize fallback client for {region}: {e}")
            
        except Exception as e:
            self.logger.error(f"Could not initialize AWS Bedrock client: {e}")
            self.bedrock_client = None
            self.enable_llm = False
            raise ConfigurationError(f"Failed to initialize Bedrock client: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM client is available"""
        return self.enable_llm and self.bedrock_client is not None
    
    def analyze_market_sentiment(self, symbol: str, news_data: List[str], 
                               cache_ttl_minutes: int = 15) -> Dict[str, Any]:
        """Analyze market sentiment using AWS Bedrock Claude
        
        Args:
            symbol: Trading symbol
            news_data: List of news/information strings
            cache_ttl_minutes: Cache time-to-live in minutes
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.is_available():
            return self._get_fallback_sentiment()
        
        # Create cache key from news data
        news_key = hash(' '.join(news_data[:3])) if news_data else 0
        cache_key = f"{symbol}_{news_key}"
        
        # Check cache first
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            if self._is_cache_valid(cached_response['timestamp'], cache_ttl_minutes):
                self.logger.debug(f"Using cached sentiment analysis for {symbol}")
                return cached_response['data']
        
        # Generate prompt
        prompt = self._create_sentiment_prompt(symbol, news_data)
        
        # Call LLM with retries across regions
        try:
            response = self._call_llm_with_retry(prompt)
            analysis = self._parse_sentiment_response(response)
            
            # Cache the result
            self.response_cache[cache_key] = {
                'data': analysis,
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._get_fallback_sentiment()
    
    def analyze_trading_signals(self, symbol: str, technical_data: Dict[str, Any], 
                              market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading signals using LLM
        
        Args:
            symbol: Trading symbol
            technical_data: Technical analysis data
            market_context: Market context and additional data
            
        Returns:
            Dictionary with trading signal analysis
        """
        if not self.is_available():
            return self._get_fallback_trading_signals()
        
        # Generate comprehensive trading analysis prompt
        prompt = self._create_trading_prompt(symbol, technical_data, market_context)
        
        try:
            response = self._call_llm_with_retry(prompt)
            analysis = self._parse_trading_response(response)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Trading signal analysis failed for {symbol}: {e}")
            return self._get_fallback_trading_signals()
    
    def _create_sentiment_prompt(self, symbol: str, news_data: List[str]) -> str:
        """Create prompt for sentiment analysis"""
        news_text = ' '.join(news_data[:3]) if news_data else "No recent news available"
        
        return f"""
Analyze the market sentiment for {symbol} based on the following news and information:

{news_text}

Provide a JSON response with:
1. sentiment_score: float between -1 (very bearish) and 1 (very bullish)
2. confidence: float between 0 and 1
3. key_factors: list of main factors affecting sentiment
4. risk_level: string (low, medium, high)
5. recommendation: string (buy, sell, hold)
6. reasoning: string explaining the analysis

Response format: {{"sentiment_score": 0.5, "confidence": 0.8, "key_factors": [...], "risk_level": "medium", "recommendation": "buy", "reasoning": "..."}}
"""
    
    def _create_trading_prompt(self, symbol: str, technical_data: Dict[str, Any], 
                             market_context: Dict[str, Any]) -> str:
        """Create prompt for trading signal analysis"""
        return f"""
Analyze the trading opportunity for {symbol} based on the following data:

Technical Indicators:
{json.dumps(technical_data, indent=2)}

Market Context:
{json.dumps(market_context, indent=2)}

Provide a comprehensive JSON response with:
1. action: string (buy, sell, hold)
2. confidence: float between 0 and 1
3. entry_price: float (suggested entry price)
4. stop_loss: float (suggested stop loss level)
5. take_profit: float (suggested take profit level)
6. position_size: float between 0 and 1 (percentage of portfolio)
7. risk_level: string (low, medium, high)
8. time_horizon: string (short, medium, long)
9. key_signals: list of main technical signals
10. reasoning: string explaining the analysis
11. market_conditions: string describing current market state

Response format: {{"action": "buy", "confidence": 0.8, "entry_price": 50000, "stop_loss": 48000, "take_profit": 55000, "position_size": 0.05, "risk_level": "medium", "time_horizon": "short", "key_signals": [...], "reasoning": "...", "market_conditions": "..."}}
"""
    
    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic across regions"""
        clients_to_try = [self.bedrock_client]
        if self.cross_region_inference:
            clients_to_try.extend(self.fallback_clients.values())
        
        last_error = None
        
        for attempt in range(max_retries):
            for client in clients_to_try:
                try:
                    # Prepare the request body for Claude
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"You are an expert cryptocurrency market analyst. Provide accurate, data-driven analysis.\n\n{prompt}"
                            }
                        ]
                    })
                    
                    # Invoke the model
                    response = client.invoke_model(
                        modelId=self.model_id,
                        body=body,
                        contentType="application/json",
                        accept="application/json"
                    )
                    
                    # Parse the response
                    response_body = json.loads(response['body'].read())
                    content = response_body['content'][0]['text']
                    
                    self.logger.debug(f"LLM response received successfully (attempt {attempt + 1})")
                    return content
                    
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    last_error = f"AWS Bedrock error ({error_code}): {e}"
                    self.logger.warning(last_error)
                    continue
                    
                except Exception as e:
                    last_error = f"LLM call error: {e}"
                    self.logger.warning(last_error)
                    continue
        
        raise APIError(f"All LLM call attempts failed. Last error: {last_error}")
    
    def _parse_sentiment_response(self, response: str) -> Dict[str, Any]:
        """Parse sentiment analysis response"""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                # Fallback parsing if no clear JSON structure
                analysis = json.loads(response)
            
            # Validate required fields
            required_fields = ['sentiment_score', 'confidence', 'key_factors', 'risk_level', 'recommendation']
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse sentiment response: {e}")
            return self._get_fallback_sentiment()
    
    def _parse_trading_response(self, response: str) -> Dict[str, Any]:
        """Parse trading signal response"""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                analysis = json.loads(json_str)
            else:
                analysis = json.loads(response)
            
            # Validate required fields
            required_fields = ['action', 'confidence', 'risk_level', 'reasoning']
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse trading response: {e}")
            return self._get_fallback_trading_signals()
    
    def _get_fallback_sentiment(self) -> Dict[str, Any]:
        """Get fallback sentiment analysis"""
        return {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "key_factors": ["LLM analysis unavailable"],
            "risk_level": "medium",
            "recommendation": "hold",
            "reasoning": "LLM service unavailable, using neutral sentiment"
        }
    
    def _get_fallback_trading_signals(self) -> Dict[str, Any]:
        """Get fallback trading signals"""
        return {
            "action": "hold",
            "confidence": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "position_size": 0.0,
            "risk_level": "medium",
            "time_horizon": "medium",
            "key_signals": ["LLM analysis unavailable"],
            "reasoning": "LLM service unavailable, recommending hold",
            "market_conditions": "unknown"
        }
    
    def _is_cache_valid(self, timestamp: datetime, ttl_minutes: int) -> bool:
        """Check if cached response is still valid"""
        return datetime.now() - timestamp < timedelta(minutes=ttl_minutes)
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.logger.info("LLM response cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.response_cache),
            "cache_keys": list(self.response_cache.keys())
        }