"""
Example script demonstrating how to use different LLM providers
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import LLMFactory, UnifiedLLMClient, BedrockLLMClient, OpenAILLMClient


def example_bedrock_client():
    """Example using AWS Bedrock client directly"""
    print("\n" + "="*60)
    print("Example 1: AWS Bedrock Client")
    print("="*60)

    try:
        # Initialize Bedrock client
        client = BedrockLLMClient(enable_llm=True)

        if not client.is_available():
            print("❌ Bedrock client not available")
            return

        print("✅ Bedrock client initialized successfully")

        # Analyze market sentiment
        news_data = [
            "Bitcoin reaches new all-time high above $100,000",
            "Major institutional investors increase crypto holdings",
            "New regulatory framework provides clarity for digital assets"
        ]

        print("\nAnalyzing market sentiment for BTC-USDT...")
        sentiment = client.analyze_market_sentiment("BTC-USDT", news_data)

        print(f"\nSentiment Analysis Results:")
        print(f"  Sentiment Score: {sentiment['sentiment_score']}")
        print(f"  Confidence: {sentiment['confidence']}")
        print(f"  Recommendation: {sentiment['recommendation']}")
        print(f"  Risk Level: {sentiment['risk_level']}")
        print(f"  Reasoning: {sentiment.get('reasoning', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_openai_client():
    """Example using OpenAI client directly"""
    print("\n" + "="*60)
    print("Example 2: OpenAI Client")
    print("="*60)

    try:
        # Initialize OpenAI client
        client = OpenAILLMClient(enable_llm=True)

        if not client.is_available():
            print("❌ OpenAI client not available")
            return

        print("✅ OpenAI client initialized successfully")

        # Analyze market sentiment
        news_data = [
            "Ethereum completes major network upgrade",
            "DeFi protocols see record TVL growth",
            "Smart contract security improves with new auditing tools"
        ]

        print("\nAnalyzing market sentiment for ETH-USDT...")
        sentiment = client.analyze_market_sentiment("ETH-USDT", news_data)

        print(f"\nSentiment Analysis Results:")
        print(f"  Sentiment Score: {sentiment['sentiment_score']}")
        print(f"  Confidence: {sentiment['confidence']}")
        print(f"  Recommendation: {sentiment['recommendation']}")
        print(f"  Risk Level: {sentiment['risk_level']}")
        print(f"  Reasoning: {sentiment.get('reasoning', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_factory_pattern():
    """Example using LLM Factory"""
    print("\n" + "="*60)
    print("Example 3: LLM Factory Pattern")
    print("="*60)

    # Show available providers
    available = LLMFactory.get_available_providers()
    print(f"Available LLM providers: {available}")

    # Get recommended provider based on credentials
    recommended = LLMFactory.get_recommended_provider()
    print(f"Recommended provider: {recommended}")

    try:
        # Create client using factory
        client = LLMFactory.create_llm_client(provider=recommended, enable_llm=True)

        if client.is_available():
            print(f"✅ {recommended.upper()} client created successfully via factory")

            # Test trading signal analysis
            technical_data = {
                "rsi": 35.5,
                "macd": 0.002,
                "bb_position": 0.25,
                "trend": "bullish"
            }

            market_context = {
                "volume_24h": 15000000,
                "volatility": "medium",
                "market_cap": 500000000000
            }

            print("\nAnalyzing trading signals for BTC-USDT...")
            signals = client.analyze_trading_signals("BTC-USDT", technical_data, market_context)

            print(f"\nTrading Signal Analysis:")
            print(f"  Action: {signals['action']}")
            print(f"  Confidence: {signals['confidence']}")
            print(f"  Risk Level: {signals['risk_level']}")
            print(f"  Time Horizon: {signals.get('time_horizon', 'N/A')}")
            print(f"  Reasoning: {signals.get('reasoning', 'N/A')}")
        else:
            print(f"❌ {recommended.upper()} client not available")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_unified_client():
    """Example using Unified Client with auto-fallback"""
    print("\n" + "="*60)
    print("Example 4: Unified Client with Auto-Fallback")
    print("="*60)

    try:
        # Initialize unified client with auto-fallback
        client = UnifiedLLMClient(auto_fallback=True, enable_llm=True)

        if not client.is_available():
            print("❌ No LLM clients available")
            return

        print(f"✅ Unified client initialized (Primary: {client.current_provider})")

        # Analyze market sentiment - will automatically fallback if primary fails
        news_data = [
            "Market shows strong recovery signals",
            "Technical indicators suggest upward momentum",
            "Increased trading volume across major exchanges"
        ]

        print("\nAnalyzing market sentiment with auto-fallback...")
        sentiment = client.analyze_market_sentiment("BTC-USDT", news_data)

        print(f"\nSentiment Analysis Results:")
        print(f"  Sentiment Score: {sentiment['sentiment_score']}")
        print(f"  Confidence: {sentiment['confidence']}")
        print(f"  Recommendation: {sentiment['recommendation']}")
        print(f"  Risk Level: {sentiment['risk_level']}")

        # Get cache statistics
        cache_stats = client.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"  Primary Provider: {cache_stats['primary_provider']}")
        if cache_stats['primary_client']:
            print(f"  Primary Cache Size: {cache_stats['primary_client']['cache_size']}")
        if cache_stats['fallback_client']:
            print(f"  Fallback Cache Size: {cache_stats['fallback_client']['cache_size']}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all examples"""
    print("\n🤖 LLM Provider Examples for Crypto Trading Agent")
    print("="*60)

    # Check environment variables
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_aws = bool(os.getenv('AWS_REGION')) or bool(os.getenv('AWS_ACCESS_KEY_ID'))

    print("\nEnvironment Check:")
    print(f"  OpenAI API Key: {'✅ Set' if has_openai else '❌ Not set'}")
    print(f"  AWS Credentials: {'✅ Available' if has_aws else '❌ Not available'}")

    # Run examples based on available credentials
    if has_aws:
        example_bedrock_client()
    else:
        print("\n⚠️  Skipping Bedrock example (AWS credentials not available)")

    if has_openai:
        example_openai_client()
    else:
        print("\n⚠️  Skipping OpenAI example (API key not set)")

    if has_aws or has_openai:
        example_factory_pattern()
        example_unified_client()
    else:
        print("\n⚠️  No LLM providers available. Please set API credentials.")

    print("\n" + "="*60)
    print("✅ Examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
