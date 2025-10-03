# LLM Architecture Overview

## Current vs New Architecture

### BEFORE (Current State)

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│                CryptoTradingAgent                       │
│                                                         │
│  __init__():                                           │
│    self.strategy = LLMTradingStrategy(                │
│        aws_region='us-east-1',                        │
│        enable_llm=True                                │
│    )                                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ creates
                     ↓
┌─────────────────────────────────────────────────────────┐
│           src/strategy/llm_strategy.py                  │
│              LLMTradingStrategy                         │
│                                                         │
│  __init__():                                           │
│    # HARDCODED BEDROCK                                │
│    self.bedrock_client = boto3.client(                │
│        'bedrock-runtime',                             │
│        region_name='us-east-1'                        │
│    )                                                  │
│                                                        │
│  analyze_market_sentiment():                          │
│    # DIRECT BOTO3 API CALLS                           │
│    body = json.dumps({...})                           │
│    response = self.bedrock_client.invoke_model(       │
│        modelId='claude-3-5-haiku',                    │
│        body=body                                      │
│    )                                                  │
│    # Parse response manually                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ calls directly
                     ↓
              ┌──────────────┐
              │ AWS Bedrock  │
              │   (boto3)    │
              │ ONLY OPTION  │
              └──────────────┘
```

**Problems:**
- 🔴 Cannot use OpenAI without rewriting code
- 🔴 Duplicated LLM logic (sentiment analysis, parsing, caching)
- 🔴 Tight coupling to AWS
- 🔴 Hard to test (requires AWS credentials)
- 🔴 No fallback to other providers

---

### AFTER (New Modular Architecture)

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│                CryptoTradingAgent                       │
│                                                         │
│  __init__():                                           │
│    # Option 1: Auto-create based on env var           │
│    self.strategy = LLMTradingStrategy(                │
│        enable_llm=True                                │
│    )  # Reads LLM_PROVIDER from .env                 │
│                                                        │
│    # Option 2: Dependency injection                   │
│    llm_client = UnifiedLLMClient()                    │
│    self.strategy = LLMTradingStrategy(                │
│        llm_client=llm_client                          │
│    )                                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ creates
                     ↓
┌─────────────────────────────────────────────────────────┐
│           src/strategy/llm_strategy.py                  │
│              LLMTradingStrategy                         │
│                                                         │
│  __init__(llm_client=None):                           │
│    if llm_client:                                     │
│        self.llm_client = llm_client                   │
│    else:                                              │
│        # Auto-create via factory                      │
│        provider = os.getenv('LLM_PROVIDER')           │
│        self.llm_client = LLMFactory.create_llm_client(│
│            provider=provider                          │
│        )                                              │
│                                                        │
│  analyze_market_sentiment():                          │
│    # SIMPLE DELEGATION                                │
│    return self.llm_client.analyze_market_sentiment(   │
│        symbol, news_data                              │
│    )                                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ uses
                     ↓
┌─────────────────────────────────────────────────────────┐
│              src/llm/llm_factory.py                     │
│                 LLMFactory                              │
│                                                         │
│  create_llm_client(provider='openai'):                 │
│    if provider == 'openai':                            │
│        return OpenAILLMClient()                        │
│    elif provider == 'bedrock':                         │
│        return BedrockLLMClient()                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ↓                         ↓
┌──────────────────┐    ┌──────────────────┐
│ OpenAILLMClient  │    │ BedrockLLMClient │
│   (OpenAI)       │    │   (AWS Bedrock)  │
│                  │    │                  │
│ _call_llm():     │    │ _call_llm():     │
│   openai.chat... │    │   boto3.invoke...│
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         │                       │
    Both inherit from            │
         │                       │
         └───────────┬───────────┘
                     ↓
         ┌───────────────────────┐
         │   BaseLLMClient       │
         │   (Abstract Base)     │
         │                       │
         │ - Caching logic       │
         │ - Prompt templates    │
         │ - Response parsing    │
         │ - Validation          │
         │ - Fallback handling   │
         └───────────────────────┘
```

**Benefits:**
- ✅ Switch providers via environment variable
- ✅ Shared logic in base class (DRY)
- ✅ Easy to add new providers
- ✅ Better testability (mock LLM client)
- ✅ Auto-fallback between providers

---

## Unified Client with Auto-Fallback

For maximum reliability, use `UnifiedLLMClient`:

```
┌─────────────────────────────────────────────────────────┐
│                UnifiedLLMClient                         │
│           (Automatic Failover)                          │
│                                                         │
│  __init__(auto_fallback=True):                         │
│    self.primary = OpenAILLMClient()                    │
│    self.fallback = BedrockLLMClient()                  │
│                                                         │
│  analyze_market_sentiment():                           │
│    try:                                                │
│        return self.primary.analyze_market_sentiment()  │
│    except Exception:                                   │
│        return self.fallback.analyze_market_sentiment() │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ Primary                      │ Fallback
         ↓                              ↓
  ┌─────────────┐              ┌─────────────┐
  │  OpenAI     │              │ AWS Bedrock │
  │  GPT-4o-mini│              │ Claude 3.5  │
  └─────────────┘              └─────────────┘

If OpenAI fails → Automatically uses Bedrock
```

---

## Data Flow Through the System

### 1. User Starts Analysis

```
User runs: python main.py
           │
           ↓
    CryptoTradingAgent.__init__()
           │
           ├─ Reads LLM_PROVIDER from .env
           │  (bedrock or openai)
           │
           ├─ Creates LLM client via factory
           │
           └─ Injects into LLMTradingStrategy
```

### 2. Analysis Cycle Begins

```
main.py: run_analysis_cycle()
    │
    ├─ Every TRADING_INTERVAL_MINUTES (default: 15 min)
    │
    └─ For each symbol in TRADING_PAIRS:
           │
           └─ analyze_symbol(symbol)
```

### 3. Symbol Analysis Pipeline

```
analyze_symbol("BTC-USDT")
    │
    ├─ [1] Fetch Market Data
    │      market_data.get_historical_klines()
    │      → Returns 500 candles (OHLCV data)
    │
    ├─ [2] Fetch News Data (if LLM enabled)
    │      market_data.get_news_data()
    │      → Returns recent news articles
    │
    ├─ [3] Generate Trading Strategy
    │      strategy.generate_trading_strategy(symbol, df, news)
    │      │
    │      ├─ A. Technical Analysis (Always)
    │      │    calculate_technical_indicators(df)
    │      │    → RSI, MACD, BB, Stochastic, etc.
    │      │    → 15+ indicators calculated
    │      │
    │      ├─ B. Market Regime Detection
    │      │    detect_market_regime(df)
    │      │    → Bull/Bear/Sideways/Volatile
    │      │
    │      ├─ C. Technical Signals
    │      │    calculate_technical_signals(df)
    │      │    → Buy/Sell/Hold scores
    │      │
    │      ├─ D. LLM Sentiment (Optional) 🤖
    │      │    IF use_llm AND news_data:
    │      │        llm_client.analyze_market_sentiment()
    │      │        │
    │      │        ├─ Check cache (15 min TTL)
    │      │        │  IF cached → return cached result
    │      │        │
    │      │        ├─ Create prompt from news
    │      │        │
    │      │        ├─ Call LLM API
    │      │        │  (OpenAI or Bedrock based on config)
    │      │        │
    │      │        ├─ Parse JSON response
    │      │        │  {
    │      │        │    sentiment_score: 0.7,
    │      │        │    confidence: 0.85,
    │      │        │    recommendation: "buy"
    │      │        │  }
    │      │        │
    │      │        └─ Cache result
    │      │
    │      └─ E. Combine Signals
    │           _combine_signals(technical, llm_sentiment)
    │           → Weighted combination
    │           → Final decision: BUY/SELL/HOLD
    │
    ├─ [4] Portfolio-Aware Position Sizing
    │      portfolio_manager.get_position_size_recommendation()
    │      → Considers current holdings
    │      → Calculates safe position size
    │
    ├─ [5] Risk Validation
    │      risk_manager.validate_trade()
    │      → Check exposure limits
    │      → Verify stop-loss levels
    │
    └─ [6] Execute Trade (if approved)
           execute_trade()
           → Place order on KuCoin
           → Record in trade tracker
```

---

## Configuration Matrix

| Scenario | LLM_PROVIDER | FAST_MODE | Result |
|----------|--------------|-----------|---------|
| Fast Trading | - | true | Technical only, 2-3s |
| OpenAI Analysis | openai | false | Technical + OpenAI, 5-8s |
| Bedrock Analysis | bedrock | false | Technical + Bedrock, 10-15s |
| Auto-Fallback | openai | false | OpenAI → Bedrock on fail |
| No LLM Available | - | false | Technical only (fallback) |

---

## Example: Complete Request Flow

```
1. User Input
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=sk-...
   python main.py

2. Initialization
   CryptoTradingAgent()
     └─ LLMTradingStrategy()
          └─ LLMFactory.create_llm_client('openai')
               └─ OpenAILLMClient()
                    └─ Tests connection ✓

3. Analysis Cycle (every 15 min)
   For BTC-USDT:

   [Market Data] ──────┐
   500 candles         │
   OHLCV data          │
                       ├──→ Technical Indicators
   [News Data] ────────┤    RSI: 35 (oversold)
   "BTC breaks $100k"  │    MACD: bullish cross
   "Institutions buy"  │    BB: near lower band
                       │
                       ├──→ LLM Sentiment Analysis
                       │    OpenAI GPT-4o-mini:
                       │    {
                       │      sentiment: 0.8 (bullish)
                       │      confidence: 0.9
                       │      recommendation: "buy"
                       │    }
                       │
                       └──→ Combined Signal
                            Technical: 0.7
                            LLM: 0.8
                            Final: 0.75 → BUY
                            Confidence: 85%

4. Position Sizing
   Portfolio: $10,000
   Current BTC: 0
   Recommendation: Buy $1,500 (15%)

5. Risk Check
   Max position: 25% ✓
   Available funds: $8,000 ✓
   Trade approved ✓

6. Execution
   Order: Buy 0.015 BTC @ $100,000
   Total: $1,500
   Status: Executed ✓

7. Tracking
   Record trade in JSON
   Update portfolio
   Log P&L
```

---

## Summary

### What This Is
This is a **CLI trading agent** (not a web app) that:
- Runs continuously in Docker or interactively
- Analyzes crypto markets every 15 minutes
- Uses LLM for sentiment analysis
- Executes trades on KuCoin exchange

### How LLM Integrates
The LLM provides **sentiment analysis** that is:
- Combined with technical indicators
- Used to make buy/sell decisions
- Optional (can be disabled with FAST_MODE)
- Cached to reduce costs
- Provider-agnostic (Bedrock or OpenAI)

### New Modular Design
The new LLM clients:
- ✅ Abstract away provider differences
- ✅ Enable easy switching between OpenAI/Bedrock
- ✅ Provide automatic failover
- ✅ Reduce code duplication
- ✅ Improve testability

To fully integrate, we need to **refactor `LLMTradingStrategy`** to use the new clients instead of direct boto3 calls.
