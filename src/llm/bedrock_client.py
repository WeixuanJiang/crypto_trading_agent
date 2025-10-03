"""AWS Bedrock LLM Client for trading strategies"""

import boto3
import json
import os
from typing import Dict, Any
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from .base_client import BaseLLMClient
from ..core.exceptions import APIError, ConfigurationError


class BedrockLLMClient(BaseLLMClient):
    """AWS Bedrock client for LLM operations"""

    def __init__(self, aws_region: str = None, enable_llm: bool = None,
                 cross_region_inference: bool = None):
        """Initialize Bedrock client

        Args:
            aws_region: AWS region for primary client
            enable_llm: Whether to enable LLM functionality
            cross_region_inference: Whether to use cross-region fallback
        """
        super().__init__()
        load_dotenv()

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

    def _call_llm(self, prompt: str) -> str:
        """Call AWS Bedrock LLM with cross-region retry logic

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response text

        Raises:
            APIError: If all retry attempts fail
        """
        max_retries = 3
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

                    self.logger.debug(f"Bedrock LLM response received successfully (attempt {attempt + 1})")
                    return content

                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    last_error = f"AWS Bedrock error ({error_code}): {e}"
                    self.logger.warning(last_error)
                    continue

                except Exception as e:
                    last_error = f"Bedrock LLM call error: {e}"
                    self.logger.warning(last_error)
                    continue

        raise APIError(f"All Bedrock LLM call attempts failed. Last error: {last_error}")
