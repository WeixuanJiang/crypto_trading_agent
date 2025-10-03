# LLM Integration Module

This module provides a flexible, provider-agnostic interface for integrating Large Language Models (LLMs) into the crypto trading agent. It supports multiple LLM providers with automatic fallback capabilities.

## Supported Providers

- **AWS Bedrock** (Claude 3.5 Haiku) - Default provider with cross-region inference
- **OpenAI** (GPT-4o-mini, GPT-4o, GPT-4 Turbo) - Alternative provider with JSON mode

## Architecture

### Class Hierarchy

```
BaseLLMClient (Abstract)
├── BedrockLLMClient
└── OpenAILLMClient

LLMFactory (Factory Pattern)
UnifiedLLMClient (Facade Pattern with Auto-Fallback)
```

### Key Components

1. **BaseLLMClient** - Abstract base class with shared functionality
   - Caching logic (15-minute TTL by default)
   - Prompt templates for sentiment and trading analysis
   - Response parsing and validation
   - Fallback handling

2. **BedrockLLMClient** - AWS Bedrock implementation
   - Cross-region inference for high availability
   - Automatic fallback across us-west-2, eu-west-1, ap-southeast-1
   - Claude 3.5 Haiku model integration

3. **OpenAILLMClient** - OpenAI implementation
   - JSON mode enforcement for structured responses
   - Exponential backoff retry logic
   - Support for multiple GPT models

4. **LLMFactory** - Factory for creating provider-specific clients
   - Provider detection based on available credentials
   - Automatic provider recommendation

5. **UnifiedLLMClient** - Unified interface with automatic failover
   - Transparent fallback between providers
   - Consolidated cache management

## Usage

### Option 1: Direct Client Usage

```python
from src.llm import OpenAILLMClient

# Initialize OpenAI client
client = OpenAILLMClient(enable_llm=True)

# Analyze market sentiment
news_data = ["Bitcoin reaches new highs", "Institutional adoption grows"]
sentiment = client.analyze_market_sentiment("BTC-USDT", news_data)

print(f"Sentiment: {sentiment['sentiment_score']}")
print(f"Recommendation: {sentiment['recommendation']}")
```

### Option 2: Factory Pattern

```python
from src.llm import LLMFactory

# Create client using factory (auto-detects best provider)
client = LLMFactory.create_llm_client(enable_llm=True)

# Or specify provider explicitly
client = LLMFactory.create_llm_client(provider="openai", enable_llm=True)

# Use the client
technical_data = {"rsi": 35, "macd": 0.002}
market_context = {"volume_24h": 15000000}

signals = client.analyze_trading_signals("BTC-USDT", technical_data, market_context)
print(f"Action: {signals['action']}")
print(f"Confidence: {signals['confidence']}")
```

### Option 3: Unified Client with Auto-Fallback

```python
from src.llm import UnifiedLLMClient

# Initialize with automatic fallback
client = UnifiedLLMClient(auto_fallback=True, enable_llm=True)

# Automatically uses primary provider, falls back if needed
sentiment = client.analyze_market_sentiment("BTC-USDT", news_data)

# Check which provider was used
stats = client.get_cache_stats()
print(f"Primary provider: {stats['primary_provider']}")
```

## Configuration

### Environment Variables

```bash
# Provider Selection
LLM_PROVIDER=openai  # or "bedrock"

# AWS Bedrock (for bedrock provider)
AWS_REGION=us-east-1
AWS_CROSS_REGION_INFERENCE=true
LLM_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0

# OpenAI (for openai provider)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o, gpt-4-turbo

# Common Settings
LLM_MAX_TOKENS=1000
LLM_TEMPERATURE=0.3
FAST_MODE=true  # Disable LLM for faster analysis
```

### Switching Providers

**Method 1: Environment Variable**
```bash
export LLM_PROVIDER=openai
python main.py
```

**Method 2: Programmatic**
```python
from src.llm import LLMFactory

# Use OpenAI
openai_client = LLMFactory.create_llm_client(provider="openai")

# Use Bedrock
bedrock_client = LLMFactory.create_llm_client(provider="bedrock")
```

## API Methods

### analyze_market_sentiment()

Analyzes market sentiment from news data.

**Parameters:**
- `symbol` (str): Trading symbol (e.g., "BTC-USDT")
- `news_data` (List[str]): List of news articles or text
- `cache_ttl_minutes` (int): Cache TTL in minutes (default: 15)

**Returns:**
```python
{
    "sentiment_score": 0.5,      # -1 (bearish) to 1 (bullish)
    "confidence": 0.8,            # 0 to 1
    "key_factors": [...],         # List of factors
    "risk_level": "medium",       # low, medium, high
    "recommendation": "buy",      # buy, sell, hold
    "reasoning": "..."            # Explanation
}
```

### analyze_trading_signals()

Analyzes trading signals based on technical data.

**Parameters:**
- `symbol` (str): Trading symbol
- `technical_data` (Dict): Technical indicators
- `market_context` (Dict): Market context data

**Returns:**
```python
{
    "action": "buy",              # buy, sell, hold
    "confidence": 0.8,            # 0 to 1
    "entry_price": 50000,         # Suggested entry
    "stop_loss": 48000,           # Stop loss level
    "take_profit": 55000,         # Take profit level
    "position_size": 0.05,        # 0 to 1 (% of portfolio)
    "risk_level": "medium",       # low, medium, high
    "time_horizon": "short",      # short, medium, long
    "key_signals": [...],         # List of signals
    "reasoning": "...",           # Explanation
    "market_conditions": "..."    # Market state
}
```

## Caching

All clients implement response caching to reduce API costs and latency:

- **Default TTL**: 15 minutes
- **Cache Key**: Generated from symbol and news data hash
- **Methods**:
  - `clear_cache()` - Clear all cached responses
  - `get_cache_stats()` - Get cache statistics

```python
# Clear cache
client.clear_cache()

# Get cache stats
stats = client.get_cache_stats()
print(f"Cached responses: {stats['cache_size']}")
```

## Error Handling

All clients provide graceful degradation:

1. **Retry Logic**: Automatic retries with exponential backoff
2. **Cross-Region Failover**: Bedrock tries multiple AWS regions
3. **Provider Fallback**: UnifiedLLMClient switches providers
4. **Neutral Fallback**: Returns neutral sentiment/signals if all fail

```python
# Example: Handling failures
try:
    sentiment = client.analyze_market_sentiment(symbol, news_data)
except Exception as e:
    # Client automatically returns neutral fallback
    print(f"LLM analysis unavailable: {e}")
```

## Performance

### Fast Mode vs Full Mode

- **Fast Mode** (`FAST_MODE=true`): Disables LLM, uses only technical analysis
- **Full Mode** (`FAST_MODE=false`): Includes LLM sentiment analysis

### Speed Comparison

| Provider | Average Latency | Notes |
|----------|----------------|-------|
| OpenAI (gpt-4o-mini) | ~2-3s | Fastest, JSON mode |
| AWS Bedrock (Claude 3.5 Haiku) | ~3-5s | Cross-region capable |
| Cached Response | <10ms | Any provider |

## Best Practices

1. **Use Caching**: Don't clear cache unnecessarily
2. **Set Appropriate TTL**: Balance freshness vs API costs
3. **Enable Fast Mode in Production**: Use for high-frequency trading
4. **Use Unified Client**: For automatic failover capabilities
5. **Monitor Cache Stats**: Track cache hit rates

## Examples

See `examples/llm_example.py` for comprehensive usage examples:

```bash
python examples/llm_example.py
```

## Testing

```bash
# Install OpenAI package
pip install openai>=1.0.0

# Set credentials
export OPENAI_API_KEY=sk-...
# or
export AWS_REGION=us-east-1

# Run tests
pytest tests/test_llm_clients.py
```

## Troubleshooting

### OpenAI Client Issues

**Problem**: `ConfigurationError: OpenAI package not installed`
```bash
pip install openai>=1.0.0
```

**Problem**: `ConfigurationError: OpenAI API key not provided`
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Bedrock Client Issues

**Problem**: `ConfigurationError: Failed to initialize Bedrock client`
```bash
# Configure AWS credentials
aws configure
# or
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
```

**Problem**: Cross-region inference not working
```bash
# Enable cross-region inference
export AWS_CROSS_REGION_INFERENCE=true
```

### General Issues

**Problem**: LLM always returns neutral sentiment
- Check `FAST_MODE=false` to enable LLM
- Verify API credentials are set
- Check logs for specific error messages

**Problem**: High API costs
- Enable caching (default)
- Increase `cache_ttl_minutes`
- Use `FAST_MODE=true` for frequent analysis
- Use gpt-4o-mini instead of gpt-4o

## License

Same as parent project.
