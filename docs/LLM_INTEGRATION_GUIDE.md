# LLM Integration Guide

This document explains how the LLM clients integrate with the Crypto Trading Agent application.

## Current Architecture

### Integration Flow

```
main.py (CryptoTradingAgent)
    ↓
LLMTradingStrategy
    ↓
Direct AWS Bedrock API Calls (boto3)
    ↓
Sentiment Analysis → Trading Decisions
```

### Current Implementation (Before Refactor)

The `LLMTradingStrategy` class directly embeds AWS Bedrock calls:

```python
# In main.py:21
from src.strategy.llm_strategy import LLMTradingStrategy

# In main.py:61-65
self.strategy = LLMTradingStrategy(
    aws_region=self.aws_region,
    enable_llm=not fast_mode,
    cross_region_inference=self.cross_region_inference
)

# LLMTradingStrategy has hardcoded boto3 calls
self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
```

**Limitations:**
- ❌ Tightly coupled to AWS Bedrock only
- ❌ Cannot switch to OpenAI without code changes
- ❌ Duplicates LLM logic across codebase
- ❌ No unified interface for different providers

## New Architecture (With Modular LLM Clients)

### Updated Integration Flow

```
main.py (CryptoTradingAgent)
    ↓
LLMTradingStrategy
    ↓
UnifiedLLMClient (Facade)
    ↓
├─ BedrockLLMClient (BaseLLMClient)
└─ OpenAILLMClient (BaseLLMClient)
    ↓
Sentiment Analysis → Trading Decisions
```

### Benefits

- ✅ Provider-agnostic design
- ✅ Easy switching between Bedrock and OpenAI
- ✅ Automatic failover across providers
- ✅ Centralized LLM logic (DRY principle)
- ✅ Better testability and maintainability

## Integration Points

### 1. Main Entry Point (`main.py`)

The `CryptoTradingAgent` class initializes the trading strategy:

```python
# Line 61-72 in main.py
self.strategy = LLMTradingStrategy(
    aws_region=self.aws_region,
    enable_llm=not fast_mode,
    cross_region_inference=self.cross_region_inference
)
```

**What it does:**
- Creates the trading strategy instance
- Passes LLM configuration (region, enable/disable, failover)
- Used throughout the trading cycle for analysis

### 2. Trading Strategy (`src/strategy/llm_strategy.py`)

The `LLMTradingStrategy` performs:

**Technical Analysis (Always Active):**
- 15+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price position analysis
- Market regime detection
- Pattern recognition

**LLM Analysis (Optional, when `enable_llm=True`):**
- Market sentiment from news (line 138: `analyze_market_sentiment()`)
- Currently hardcoded to AWS Bedrock (lines 106-130)
- Called during `generate_trading_strategy()` (line 848-850)

### 3. Analysis Cycle (`main.py:178-257`)

```python
def analyze_symbol(self, symbol: str, use_llm: bool = False) -> Dict:
    # Get market data
    df = self.market_data.get_historical_klines(symbol, '1hour', limit=limit)
    news_data = self.market_data.get_news_data(symbol)  # If use_llm

    # Generate strategy (calls LLM if enabled)
    strategy_result = self.strategy.generate_trading_strategy(symbol, df, news_data, use_llm)

    # Get portfolio-aware position recommendation
    position_recommendation = self.portfolio_manager.get_position_size_recommendation(...)

    return analysis
```

**Flow:**
1. Fetch historical market data
2. Fetch news data (if LLM enabled)
3. Generate trading strategy (technical + optional LLM)
4. Get position sizing recommendation
5. Display results
6. Execute trade (if conditions met)

## How to Refactor for Modular LLM Clients

### Option 1: Minimal Change (Dependency Injection)

**Modify `LLMTradingStrategy.__init__()` to accept an LLM client:**

```python
# src/strategy/llm_strategy.py

from ..llm import UnifiedLLMClient, LLMFactory

class LLMTradingStrategy:
    def __init__(self, aws_region: str = None, enable_llm: bool = None,
                 cross_region_inference: bool = None, llm_client=None):

        # ... existing code ...

        # NEW: Use injected client or create one
        if llm_client:
            self.llm_client = llm_client
        elif enable_llm:
            # Auto-create unified client with provider from env
            self.llm_client = UnifiedLLMClient(
                auto_fallback=True,
                enable_llm=enable_llm
            )
        else:
            self.llm_client = None

        # REMOVE: Old bedrock_client initialization
        # self.bedrock_client = boto3.client(...)

    def analyze_market_sentiment(self, symbol: str, news_data: List[str]) -> Dict:
        """Use LLM client instead of direct boto3 calls"""
        if not self.llm_client or not self.llm_client.is_available():
            return self._get_fallback_sentiment()

        # NEW: Use modular client
        return self.llm_client.analyze_market_sentiment(symbol, news_data)

        # REMOVE: All the boto3 API calls (lines 172-249)
```

**Update `main.py`:**

```python
# main.py

from src.llm import UnifiedLLMClient, LLMFactory

class CryptoTradingAgent:
    def __init__(self, auto_trading_enabled=False, fast_mode=None):
        # ... existing code ...

        # NEW: Create LLM client first
        self.llm_client = None
        if not fast_mode:
            try:
                self.llm_client = UnifiedLLMClient(
                    auto_fallback=True,
                    enable_llm=True
                )
                provider = os.getenv('LLM_PROVIDER', 'bedrock')
                logger.info(f"✓ LLM Client initialized (Provider: {provider})")
            except Exception as e:
                logger.warning(f"⚠ LLM initialization failed: {e}")

        # Initialize trading strategy WITH llm_client
        self.strategy = LLMTradingStrategy(
            aws_region=self.aws_region,
            enable_llm=not fast_mode,
            cross_region_inference=self.cross_region_inference,
            llm_client=self.llm_client  # NEW: Inject the client
        )
```

### Option 2: Configuration-Based (Cleaner)

**Use environment variable to control provider:**

```python
# main.py

class CryptoTradingAgent:
    def __init__(self, auto_trading_enabled=False, fast_mode=None):
        # ... existing code ...

        # Create strategy (it will auto-create LLM client based on LLM_PROVIDER env)
        self.strategy = LLMTradingStrategy(
            enable_llm=not fast_mode
        )
```

**Simplified `LLMTradingStrategy`:**

```python
# src/strategy/llm_strategy.py

from ..llm import LLMFactory

class LLMTradingStrategy:
    def __init__(self, enable_llm: bool = None):
        load_dotenv()

        self.enable_llm = enable_llm if enable_llm is not None else (
            os.getenv('FAST_MODE', 'true').lower() != 'true'
        )

        # Auto-create LLM client based on LLM_PROVIDER env var
        if self.enable_llm:
            try:
                provider = os.getenv('LLM_PROVIDER', 'bedrock')
                self.llm_client = LLMFactory.create_llm_client(
                    provider=provider,
                    enable_llm=True
                )
            except Exception as e:
                logger.error(f"Failed to create LLM client: {e}")
                self.llm_client = None
        else:
            self.llm_client = None

        # ... rest of initialization ...

    def analyze_market_sentiment(self, symbol: str, news_data: List[str]) -> Dict:
        if not self.llm_client or not self.llm_client.is_available():
            return self._get_fallback_sentiment()

        return self.llm_client.analyze_market_sentiment(symbol, news_data)
```

## Configuration

### Switching Between Providers

**Use Bedrock (Default):**
```bash
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
export FAST_MODE=false
python main.py
```

**Use OpenAI:**
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
export FAST_MODE=false
python main.py
```

**Disable LLM (Fast Mode):**
```bash
export FAST_MODE=true
python main.py
```

## Current Call Hierarchy

### Trading Cycle Flow

```
main.py: run_analysis_cycle()
  ↓
main.py: analyze_symbol(symbol, use_llm=True/False)
  ↓
market_data.get_historical_klines() → DataFrame
market_data.get_news_data() → List[str]
  ↓
strategy.generate_trading_strategy(symbol, df, news_data, use_llm)
  ├─ calculate_technical_indicators(df) → DataFrame with indicators
  ├─ detect_market_regime(df) → Dict (regime info)
  ├─ calculate_technical_signals(df) → Dict (signal scores)
  ├─ analyze_market_sentiment(symbol, news_data) → Dict [LLM CALL]
  └─ _combine_signals() → Dict (final decision)
  ↓
portfolio_manager.get_position_size_recommendation()
  ↓
execute_trade() if conditions met
```

### When LLM is Called

**Trigger Conditions:**
1. `FAST_MODE=false` (environment variable)
2. `use_llm=True` parameter in `analyze_symbol()`
3. `news_data` is available and not empty
4. LLM client is initialized and available

**Frequency:**
- Once per symbol per analysis cycle
- Results cached for 15 minutes (by default)
- In automated mode: every `TRADING_INTERVAL_MINUTES` (default: 15 min)

## Performance Impact

### Fast Mode (LLM Disabled)
- **Analysis Time**: ~2-3 seconds per symbol
- **Uses**: Technical indicators only
- **Recommended for**: High-frequency trading, production

### Full Mode with Bedrock
- **Analysis Time**: ~10-15 seconds per symbol
- **Uses**: Technical + AWS Bedrock sentiment
- **Recommended for**: Research, detailed analysis

### Full Mode with OpenAI
- **Analysis Time**: ~5-8 seconds per symbol
- **Uses**: Technical + OpenAI sentiment
- **Recommended for**: Cost-effective LLM analysis

## Testing the Integration

### Test with Bedrock
```bash
export LLM_PROVIDER=bedrock
export AWS_REGION=us-east-1
export FAST_MODE=false
python main.py

# In interactive menu, choose:
# 4. Analyze specific symbol
# Enter: BTC-USDT
```

### Test with OpenAI
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export FAST_MODE=false
python main.py

# Same as above
```

### Test Auto-Fallback
```bash
# Set primary to OpenAI, but don't set API key (will fail)
export LLM_PROVIDER=openai
export AWS_REGION=us-east-1  # Fallback to Bedrock
export FAST_MODE=false

# If using UnifiedLLMClient, it will automatically
# try OpenAI first, then fallback to Bedrock
python main.py
```

## Migration Checklist

To fully integrate the new LLM clients:

- [ ] Update `LLMTradingStrategy.__init__()` to use `LLMFactory` or accept injected client
- [ ] Replace `analyze_market_sentiment()` boto3 calls with `llm_client.analyze_market_sentiment()`
- [ ] Remove direct `boto3` client initialization
- [ ] Update `main.py` to optionally inject LLM client or rely on auto-creation
- [ ] Add `LLM_PROVIDER` to `.env.example` (already done ✅)
- [ ] Test with both providers
- [ ] Update logging messages to show active provider
- [ ] Update CLAUDE.md with new LLM integration details

## Next Steps

1. **Create refactored `LLMTradingStrategy`** - Would you like me to create a new version?
2. **Update `main.py`** - Modify initialization to use new clients
3. **Add provider switching CLI flag** - `python main.py --llm-provider openai`
4. **Create integration tests** - Test both providers work correctly
5. **Update documentation** - Reflect new architecture in README

Would you like me to implement any of these refactoring options?
