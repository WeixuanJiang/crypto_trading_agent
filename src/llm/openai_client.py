"""OpenAI LLM Client for trading strategies"""

import json
import os
import time
from typing import Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .base_client import BaseLLMClient
from ..core.logger import get_logger
from ..core.exceptions import APIError, ConfigurationError


class OpenAILLMClient(BaseLLMClient):
    """OpenAI client for LLM operations"""

    def __init__(self, api_key: str = None, enable_llm: bool = None,
                 model_name: str = None, max_tokens: int = None,
                 temperature: float = None):
        """Initialize OpenAI client

        Args:
            api_key: OpenAI API key
            enable_llm: Whether to enable LLM functionality
            model_name: OpenAI model to use
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
        """
        super().__init__()

        # Configuration from environment with fallbacks
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.enable_llm = enable_llm if enable_llm is not None else (
            os.getenv('FAST_MODE', 'true').lower() != 'true'
        )

        # Model configuration
        self.model_name = model_name or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.max_tokens = max_tokens or int(os.getenv('LLM_MAX_TOKENS', '1000'))
        self.temperature = temperature if temperature is not None else float(
            os.getenv('LLM_TEMPERATURE', '0.3')
        )

        # Initialize client
        self.client = None
        if self.enable_llm:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            if not self.api_key:
                raise ConfigurationError("OpenAI API key not provided")

            self.client = OpenAI(api_key=self.api_key)

            # Test connection
            self._test_connection()
            self.logger.info(f"Initialized OpenAI client with model {self.model_name}")

        except Exception as e:
            self.logger.error(f"Could not initialize OpenAI client: {e}")
            self.client = None
            self.enable_llm = False
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}")

    def _test_connection(self):
        """Test OpenAI API connection"""
        try:
            # Make a minimal test call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            self.logger.debug("OpenAI connection test successful")
        except Exception as e:
            raise ConfigurationError(f"OpenAI connection test failed: {e}")

    def is_available(self) -> bool:
        """Check if LLM client is available"""
        return self.enable_llm and self.client is not None

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API with retry logic

        Args:
            prompt: The prompt to send to OpenAI

        Returns:
            Response text from OpenAI

        Raises:
            APIError: If all retry attempts fail
        """
        max_retries = 3
        base_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Call OpenAI API with JSON mode enforcement
                response: ChatCompletion = self.client.chat.completions.create(
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
                    response_format={"type": "json_object"}
                )

                # Extract content from response
                content = response.choices[0].message.content

                self.logger.debug(f"OpenAI response received successfully (attempt {attempt + 1})")
                return content

            except Exception as e:
                last_error = f"OpenAI API error: {e}"
                self.logger.warning(f"{last_error} (attempt {attempt + 1}/{max_retries})")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    self.logger.debug(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise APIError(f"All OpenAI API attempts failed. Last error: {last_error}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "provider": "openai",
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enabled": self.enable_llm,
            "available": self.is_available()
        }
