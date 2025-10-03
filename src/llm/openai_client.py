"""OpenAI LLM Client for trading strategies"""

import json
import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_client import BaseLLMClient
from ..core.exceptions import APIError, ConfigurationError


class OpenAILLMClient(BaseLLMClient):
    """OpenAI client for LLM operations"""

    def __init__(self, api_key: str = None, enable_llm: bool = None,
                 model_name: str = None):
        """Initialize OpenAI client

        Args:
            api_key: OpenAI API key
            enable_llm: Whether to enable LLM functionality
            model_name: Model to use (default: gpt-4o-mini)
        """
        if not OPENAI_AVAILABLE:
            raise ConfigurationError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        super().__init__()
        load_dotenv()

        # Configuration from environment with fallbacks
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.enable_llm = enable_llm if enable_llm is not None else (
            os.getenv('FAST_MODE', 'true').lower() != 'true'
        )

        # Model configuration
        self.model_name = model_name or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '1000'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.3'))

        # Initialize client
        self.client = None

        if self.enable_llm:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.logger.info(f"Initialized OpenAI client with model {self.model_name}")

            # Test the connection
            self._test_connection()

        except Exception as e:
            self.logger.error(f"Could not initialize OpenAI client: {e}")
            self.client = None
            self.enable_llm = False
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}")

    def _test_connection(self):
        """Test OpenAI API connection"""
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            self.logger.debug("OpenAI API connection test successful")
        except Exception as e:
            raise ConfigurationError(f"OpenAI API connection test failed: {e}")

    def is_available(self) -> bool:
        """Check if LLM client is available"""
        return self.enable_llm and self.client is not None

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI LLM with retry logic

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response text

        Raises:
            APIError: If all retry attempts fail
        """
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cryptocurrency market analyst. Provide accurate, data-driven analysis in JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}  # Enforce JSON response
                )

                content = response.choices[0].message.content
                self.logger.debug(f"OpenAI response received successfully (attempt {attempt + 1})")
                return content

            except Exception as e:
                last_error = f"OpenAI API error: {e}"
                self.logger.warning(f"{last_error} (attempt {attempt + 1}/{max_retries})")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    self.logger.debug(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

        raise APIError(f"All OpenAI LLM call attempts failed. Last error: {last_error}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics

        Note: OpenAI doesn't provide built-in usage tracking via SDK.
        This is a placeholder for custom usage tracking if needed.
        """
        return {
            "model": self.model_name,
            "cache_size": len(self.response_cache),
            "status": "operational" if self.is_available() else "unavailable"
        }
