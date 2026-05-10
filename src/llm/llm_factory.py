"""Factory for creating LLM clients"""

import os
from enum import Enum
from typing import Optional, Dict, Any

from .bedrock_client import BedrockLLMClient
from .openai_client import OpenAILLMClient
from .base_client import BaseLLMClient
from ..core.logger import get_logger
from ..core.exceptions import ConfigurationError


class LLMProvider(Enum):
    """Supported LLM providers"""
    BEDROCK = "bedrock"
    OPENAI = "openai"


class LLMFactory:
    """Factory for creating LLM clients based on provider"""

    @staticmethod
    def create_llm_client(provider: str = None, **kwargs) -> BaseLLMClient:
        """Create an LLM client based on provider

        Args:
            provider: LLM provider ('bedrock' or 'openai'). If None, reads from LLM_PROVIDER env var
            **kwargs: Additional arguments to pass to the client constructor

        Returns:
            BaseLLMClient: Initialized LLM client

        Raises:
            ConfigurationError: If provider is invalid or not configured
        """
        logger = get_logger(__name__)

        # Get provider from env if not specified
        if provider is None:
            provider = os.getenv('LLM_PROVIDER', 'bedrock').lower()

        # Validate provider
        try:
            provider_enum = LLMProvider(provider)
        except ValueError:
            raise ConfigurationError(
                f"Invalid LLM provider: {provider}. "
                f"Supported providers: {[p.value for p in LLMProvider]}"
            )

        # Create client based on provider
        if provider_enum == LLMProvider.BEDROCK:
            logger.info("Creating AWS Bedrock LLM client")
            return BedrockLLMClient(**kwargs)

        elif provider_enum == LLMProvider.OPENAI:
            logger.info("Creating OpenAI LLM client")
            return OpenAILLMClient(**kwargs)

        else:
            raise ConfigurationError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_available_providers() -> list:
        """Get list of available LLM providers"""
        return [provider.value for provider in LLMProvider]


class UnifiedLLMClient:
    """Unified client that can switch between providers at runtime"""

    def __init__(self, primary_provider: str = None, fallback_provider: str = None):
        """Initialize unified LLM client with primary and optional fallback

        Args:
            primary_provider: Primary LLM provider to use
            fallback_provider: Optional fallback provider if primary fails
        """
        self.logger = get_logger(__name__)

        # Create primary client
        self.primary_provider = primary_provider or os.getenv('LLM_PROVIDER', 'bedrock')
        self.primary_client = LLMFactory.create_llm_client(self.primary_provider)

        # Create fallback client if specified
        self.fallback_provider = fallback_provider
        self.fallback_client = None
        if fallback_provider:
            try:
                self.fallback_client = LLMFactory.create_llm_client(fallback_provider)
                self.logger.info(f"Fallback LLM client initialized: {fallback_provider}")
            except Exception as e:
                self.logger.warning(f"Could not initialize fallback client: {e}")

    def analyze_market_sentiment(self, symbol: str, news_data: list,
                                 cache_ttl_minutes: int = 15) -> Dict[str, Any]:
        """Analyze market sentiment with automatic fallback"""
        try:
            # Try primary client first
            if self.primary_client.is_available():
                return self.primary_client.analyze_market_sentiment(
                    symbol, news_data, cache_ttl_minutes
                )
        except Exception as e:
            self.logger.warning(f"Primary LLM failed for sentiment analysis: {e}")

        # Try fallback if available
        if self.fallback_client and self.fallback_client.is_available():
            self.logger.info(f"Using fallback provider: {self.fallback_provider}")
            try:
                return self.fallback_client.analyze_market_sentiment(
                    symbol, news_data, cache_ttl_minutes
                )
            except Exception as e:
                self.logger.error(f"Fallback LLM also failed: {e}")

        # Return neutral sentiment if all fail
        return self.primary_client._get_fallback_sentiment()

    def analyze_trading_signals(self, symbol: str, technical_data: Dict[str, Any],
                                market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading signals with automatic fallback"""
        try:
            # Try primary client first
            if self.primary_client.is_available():
                return self.primary_client.analyze_trading_signals(
                    symbol, technical_data, market_context
                )
        except Exception as e:
            self.logger.warning(f"Primary LLM failed for trading signals: {e}")

        # Try fallback if available
        if self.fallback_client and self.fallback_client.is_available():
            self.logger.info(f"Using fallback provider: {self.fallback_provider}")
            try:
                return self.fallback_client.analyze_trading_signals(
                    symbol, technical_data, market_context
                )
            except Exception as e:
                self.logger.error(f"Fallback LLM also failed: {e}")

        # Return neutral signals if all fail
        return self.primary_client._get_fallback_trading_signals()

    def is_available(self) -> bool:
        """Check if any LLM client is available"""
        return (self.primary_client.is_available() or
                (self.fallback_client and self.fallback_client.is_available()))

    def get_active_provider(self) -> str:
        """Get the currently active provider"""
        if self.primary_client.is_available():
            return self.primary_provider
        elif self.fallback_client and self.fallback_client.is_available():
            return self.fallback_provider
        return "none"

    def clear_cache(self):
        """Clear cache for all clients"""
        self.primary_client.clear_cache()
        if self.fallback_client:
            self.fallback_client.clear_cache()
