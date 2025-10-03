"""LLM Factory for creating and managing different LLM providers"""

import os
from typing import Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv

from ..core.logger import get_logger
from ..core.exceptions import ConfigurationError


class LLMProvider(Enum):
    """Supported LLM providers"""
    BEDROCK = "bedrock"
    OPENAI = "openai"


class LLMFactory:
    """Factory for creating LLM clients"""

    @staticmethod
    def create_llm_client(provider: str = None, **kwargs) -> Any:
        """Create an LLM client based on provider

        Args:
            provider: LLM provider name (bedrock, openai). If None, reads from env.
            **kwargs: Additional arguments passed to the client constructor

        Returns:
            Initialized LLM client

        Raises:
            ConfigurationError: If provider is invalid or initialization fails
        """
        load_dotenv()
        logger = get_logger(__name__)

        # Determine provider from environment if not specified
        if provider is None:
            provider = os.getenv('LLM_PROVIDER', 'bedrock').lower()

        logger.info(f"Creating LLM client for provider: {provider}")

        try:
            if provider == LLMProvider.BEDROCK.value:
                from .bedrock_client import BedrockLLMClient
                return BedrockLLMClient(**kwargs)

            elif provider == LLMProvider.OPENAI.value:
                from .openai_client import OpenAILLMClient
                return OpenAILLMClient(**kwargs)

            else:
                raise ConfigurationError(
                    f"Unsupported LLM provider: {provider}. "
                    f"Supported providers: {[p.value for p in LLMProvider]}"
                )

        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import {provider} client. "
                f"Ensure required dependencies are installed: {e}"
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create {provider} client: {e}")

    @staticmethod
    def get_available_providers() -> list:
        """Get list of available LLM providers

        Returns:
            List of available provider names
        """
        available = []

        # Check Bedrock
        try:
            import boto3
            available.append(LLMProvider.BEDROCK.value)
        except ImportError:
            pass

        # Check OpenAI
        try:
            import openai
            available.append(LLMProvider.OPENAI.value)
        except ImportError:
            pass

        return available

    @staticmethod
    def get_recommended_provider() -> str:
        """Get recommended provider based on available credentials

        Returns:
            Recommended provider name
        """
        load_dotenv()

        # Check for OpenAI credentials
        if os.getenv('OPENAI_API_KEY'):
            try:
                import openai
                return LLMProvider.OPENAI.value
            except ImportError:
                pass

        # Check for AWS credentials
        if os.getenv('AWS_REGION') or os.getenv('AWS_ACCESS_KEY_ID'):
            try:
                import boto3
                return LLMProvider.BEDROCK.value
            except ImportError:
                pass

        # Default to Bedrock
        return LLMProvider.BEDROCK.value


class UnifiedLLMClient:
    """Unified interface for different LLM providers"""

    def __init__(self, provider: str = None, auto_fallback: bool = True, **kwargs):
        """Initialize unified LLM client

        Args:
            provider: Primary LLM provider (bedrock, openai)
            auto_fallback: Automatically fallback to another provider if primary fails
            **kwargs: Additional arguments passed to the client
        """
        self.logger = get_logger(__name__)
        self.auto_fallback = auto_fallback
        self.primary_provider = provider or os.getenv('LLM_PROVIDER', 'bedrock')
        self.current_provider = None
        self.client = None

        # Initialize primary client
        self._initialize_client(self.primary_provider, **kwargs)

        # Initialize fallback client if enabled
        self.fallback_client = None
        if auto_fallback:
            self._initialize_fallback_client(**kwargs)

    def _initialize_client(self, provider: str, **kwargs):
        """Initialize the primary client"""
        try:
            self.client = LLMFactory.create_llm_client(provider, **kwargs)
            self.current_provider = provider
            self.logger.info(f"Initialized primary LLM client: {provider}")
        except Exception as e:
            self.logger.error(f"Failed to initialize {provider} client: {e}")
            if not self.auto_fallback:
                raise
            self.client = None
            self.current_provider = None

    def _initialize_fallback_client(self, **kwargs):
        """Initialize a fallback client with a different provider"""
        available_providers = LLMFactory.get_available_providers()

        # Find a different provider
        fallback_provider = None
        for provider in available_providers:
            if provider != self.primary_provider:
                fallback_provider = provider
                break

        if fallback_provider:
            try:
                self.fallback_client = LLMFactory.create_llm_client(fallback_provider, **kwargs)
                self.logger.info(f"Initialized fallback LLM client: {fallback_provider}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize fallback client: {e}")
                self.fallback_client = None

    def is_available(self) -> bool:
        """Check if any LLM client is available"""
        if self.client and self.client.is_available():
            return True
        if self.fallback_client and self.fallback_client.is_available():
            return True
        return False

    def analyze_market_sentiment(self, symbol: str, news_data: list,
                               cache_ttl_minutes: int = 15) -> Dict[str, Any]:
        """Analyze market sentiment using available LLM client"""
        # Try primary client
        if self.client and self.client.is_available():
            try:
                return self.client.analyze_market_sentiment(symbol, news_data, cache_ttl_minutes)
            except Exception as e:
                self.logger.warning(f"Primary client failed for sentiment analysis: {e}")

        # Try fallback client
        if self.auto_fallback and self.fallback_client and self.fallback_client.is_available():
            try:
                self.logger.info("Using fallback LLM client for sentiment analysis")
                return self.fallback_client.analyze_market_sentiment(symbol, news_data, cache_ttl_minutes)
            except Exception as e:
                self.logger.error(f"Fallback client also failed: {e}")

        # Return neutral sentiment if all fail
        return self._get_fallback_sentiment()

    def analyze_trading_signals(self, symbol: str, technical_data: Dict[str, Any],
                              market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading signals using available LLM client"""
        # Try primary client
        if self.client and self.client.is_available():
            try:
                return self.client.analyze_trading_signals(symbol, technical_data, market_context)
            except Exception as e:
                self.logger.warning(f"Primary client failed for trading signals: {e}")

        # Try fallback client
        if self.auto_fallback and self.fallback_client and self.fallback_client.is_available():
            try:
                self.logger.info("Using fallback LLM client for trading signals")
                return self.fallback_client.analyze_trading_signals(symbol, technical_data, market_context)
            except Exception as e:
                self.logger.error(f"Fallback client also failed: {e}")

        # Return neutral signals if all fail
        return self._get_fallback_trading_signals()

    def clear_cache(self):
        """Clear cache for all clients"""
        if self.client:
            self.client.clear_cache()
        if self.fallback_client:
            self.fallback_client.clear_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from all clients"""
        stats = {
            'primary_provider': self.current_provider,
            'primary_client': None,
            'fallback_client': None
        }

        if self.client:
            stats['primary_client'] = self.client.get_cache_stats()
        if self.fallback_client:
            stats['fallback_client'] = self.fallback_client.get_cache_stats()

        return stats

    def _get_fallback_sentiment(self) -> Dict[str, Any]:
        """Get fallback sentiment analysis"""
        return {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "key_factors": ["LLM analysis unavailable"],
            "risk_level": "medium",
            "recommendation": "hold",
            "reasoning": "All LLM services unavailable, using neutral sentiment"
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
            "reasoning": "All LLM services unavailable, recommending hold",
            "market_conditions": "unknown"
        }
