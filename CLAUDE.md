# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated cryptocurrency trading agent that combines AWS Bedrock Claude 3.5 Haiku LLM analysis with quantitative trading methods for automated trading on KuCoin exchange. The system implements a "buy low, sell high" strategy with comprehensive risk management, portfolio tracking, and P&L monitoring.

## Core Architecture

### Main Components

The application follows a modular architecture with clear separation of concerns:

1. **main.py** - `CryptoTradingAgent` orchestrator
   - Entry point and main trading loop
   - Coordinates all subsystems (market data, strategy, risk, portfolio)
   - Handles Docker/automated mode detection
   - Manages interactive CLI menu (non-Docker)
   - Command-line args: `--auto-trading`, `--paper-trading`, `--auto-mode`, `--check-aws`

2. **src/strategy/llm_strategy.py** - `LLMTradingStrategy` engine
   - AWS Bedrock integration with cross-region fallback
   - 15+ technical indicators (RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, MFI, ADX, Aroon, PSAR, VWAP, Keltner Channels)
   - Market regime detection (bull/bear trend, volatile/quiet sideways, transitional)
   - Adaptive strategy selection based on market conditions
   - Multi-timeframe analysis with dynamic weight adjustment
   - Pattern recognition (candlestick patterns, divergence, market structure)
   - Fast mode (no LLM) vs Full mode (with LLM sentiment analysis)

3. **src/portfolio/manager.py** - `PortfolioManager`
   - Balance-aware position sizing
   - Portfolio allocation management (max 25% per asset)
   - Rebalancing recommendations
   - P&L tracking per position
   - Buy/sell recommendations considering existing positions

4. **src/market/data.py** - `MarketDataManager`
   - KuCoin API integration for live price feeds
   - Historical OHLCV data retrieval
   - Market metrics calculation
   - News data integration (optional)

5. **src/risk/manager.py** - `RiskManager`
   - Position sizing (2% max portfolio risk per trade)
   - Stop loss management (5% default, ATR-based)
   - Portfolio exposure limits (10% per position, 20% total)
   - Trade validation and risk assessment

6. **src/trading/aws_tracker.py** - `TradeTracker`
   - Trade history persistence (JSON files)
   - P&L calculation (realized and unrealized)
   - Performance metrics (win rate, profit factor, Sharpe ratio)
   - Portfolio summary and position tracking

7. **src/core/config.py** - Centralized configuration
   - Environment variable management via python-dotenv
   - Dataclass-based config sections (TradingConfig, RiskConfig, TechnicalConfig, LLMConfig, etc.)
   - Config validation and export/import

### Data Flow

```
Market Data → Technical Analysis → LLM Analysis (optional) → Signal Combination → Risk Assessment → Trade Execution → Portfolio Tracking
```

1. Market data fetched from KuCoin (historical + real-time)
2. Calculate 15+ technical indicators
3. LLM sentiment analysis (if enabled, Full mode only)
4. Combine signals with dynamic weights based on market regime
5. Portfolio-aware position sizing via PortfolioManager
6. Risk validation via RiskManager
7. Trade execution (real or paper)
8. Track in TradeTracker and update PortfolioManager

## Development Commands

### Running the Application

```bash
# Interactive mode (local only, not Docker)
python main.py

# Paper trading (simulation, no real trades)
python main.py --paper-trading

# Auto trading (REAL MONEY - requires trading API permissions)
python main.py --auto-trading

# Automated mode (for EC2/server deployment)
python main.py --auto-mode --auto-trading

# Check AWS Bedrock status
python main.py --check-aws
```

### Docker Deployment

```bash
# Build and run (auto-detects Docker environment)
docker-compose up --build -d

# View logs
docker-compose logs -f crypto-trading-agent

# Stop
docker-compose down
```

### Testing

```bash
# Run tests (pytest configured in requirements.txt)
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Linting and code quality
black src/
flake8 src/
mypy src/
bandit -r src/
safety check
```

## Configuration Management

All configuration is centralized in `src/core/config.py` and loaded from:
1. Environment variables (highest priority)
2. `.env` file (via python-dotenv)
3. `config/settings.json` (optional override file)

### Key Environment Variables

**Trading:**
- `TRADING_PAIRS` - Comma-separated list (default: BTC-USDT,ETH-USDT,ADA-USDT,DOT-USDT)
- `TRADING_INTERVAL_MINUTES` - Analysis cycle interval (default: 15)
- `MIN_CONFIDENCE_THRESHOLD` - Minimum confidence to execute (default: 0.7)
- `MAX_DAILY_TRADES` - Daily trade limit (default: 1000)
- `FAST_MODE` - Enable fast mode without LLM (default: true)

**AWS/LLM:**
- `AWS_REGION` - Primary AWS region (default: us-east-1)
- `AWS_CROSS_REGION_INFERENCE` - Enable multi-region fallback (default: true)
- `LLM_MODEL_ID` - Bedrock model (default: us.anthropic.claude-3-5-haiku-20241022-v1:0)

**Risk Management:**
- `MAX_POSITION_SIZE` - Max % of portfolio per trade (default: 0.02 = 2%)
- `STOP_LOSS_PERCENTAGE` - Default stop loss (default: 0.05 = 5%)
- `MAX_PORTFOLIO_EXPOSURE` - Total portfolio exposure limit (default: 0.20 = 20%)

**API:**
- `KUCOIN_API_KEY`, `KUCOIN_API_SECRET`, `KUCOIN_API_PASSPHRASE` - Required for trading

## Key Design Patterns

### Configuration Pattern
- Dataclass-based configs in `src/core/config.py`
- Global singleton: `from src.core.config import config`
- Environment variable loading with validation
- Type-safe config access: `config.trading.min_confidence_threshold`

### Strategy Pattern
- `LLMTradingStrategy` implements comprehensive strategy logic
- Market regime detection drives adaptive strategy selection
- Different thresholds/weights for: bull_trend, bear_trend, volatile_sideways, quiet_sideways, transitional
- Dynamic signal weighting based on volatility and trend strength

### Portfolio-Aware Trading
- `PortfolioManager` tracks all positions and balances
- Buy recommendations limited by target allocation (15-25% per asset)
- Sell recommendations based on position size, P&L, and confidence
- Rebalancing suggestions when deviation > 5%

### Risk-First Approach
- All trades validated by `RiskManager` before execution
- Position sizing calculated dynamically based on volatility and confidence
- Multiple stop-loss strategies (fixed %, ATR-based, trailing)
- Portfolio-level risk tracking (max exposure, correlation limits)

### Trade Tracking
- `TradeTracker` persists all trades to JSON (trading_data/trade_history.json)
- Automatic P&L calculation linking buys and sells
- Portfolio summary with open positions
- Performance metrics (win rate, profit factor, best/worst trades)

## Mode Detection and Execution

### Fast Mode vs Full Mode
- **Fast Mode** (default, `FAST_MODE=true`): Technical analysis only, 3-5x faster, ~2-3s per symbol
- **Full Mode** (`FAST_MODE=false`): Includes AWS Bedrock LLM sentiment analysis, ~10-15s per symbol
- Toggle via environment variable or constructor parameter

### Docker Auto-Detection
The application automatically detects Docker environments and enables automated mode:
```python
is_docker = os.getenv('DOCKER_CONTAINER', 'false').lower() == 'true' or os.path.exists('/.dockerenv')
```
When in Docker: runs continuous trading cycles without interactive menu.

### Trading Modes
- **Paper Trading** (default): Simulates trades, no real money, all signals work normally
- **Auto Trading** (opt-in): Places real orders on KuCoin, requires `--auto-trading` flag

## Important Implementation Notes

### API Integration
- Uses `python-kucoin` library: `from kucoin.client import Client`
- Single unified client for all operations (market, trade, user)
- Retry mechanism with exponential backoff for API timeouts
- Rate limiting configured via environment variables

### Cross-Region Inference
When `AWS_CROSS_REGION_INFERENCE=true`, the LLM strategy tries multiple AWS regions:
1. Primary region (from `AWS_REGION`)
2. Fallback regions: us-west-2, eu-west-1, ap-southeast-1
3. Automatic failover on `ClientError` or timeout

### Portfolio Value Calculation
```python
portfolio_value = sum(position.market_value for position in positions) + usdt_balance
```
- Crypto positions valued at current market prices
- USDT balance added directly
- Used for allocation % and position sizing

### Signal Combination Logic
Signals are combined with dynamic weights that adjust based on:
- Market volatility (higher volatility → favor mean reversion indicators)
- Trend strength (stronger trends → favor trend-following indicators)
- Base weights favor price position (20%), RSI (12%), Bollinger Bands (12%)
- Multiple confirmation layers increase confidence

### Trade Execution Flow
1. Check confidence threshold (default 0.7)
2. Check daily trade limit
3. Update portfolio data
4. Get position recommendation from PortfolioManager
5. Validate trade size (min/max, increments)
6. Execute order (if auto-trading enabled) OR record paper trade
7. Record in TradeTracker
8. Update portfolio cache

## Data Persistence

### File Structure
```
trading_data/
  ├── trade_history.json          # All trades
  ├── portfolio_cache.json        # Portfolio state cache
  └── avg_prices.json             # Average purchase prices

logs/
  └── trading_agent.log           # Application logs
```

### Trade History Format
```json
{
  "id": "BTC-USDT_20250103_143022",
  "timestamp": "2025-01-03T14:30:22",
  "symbol": "BTC-USDT",
  "action": "BUY",
  "price": 45000.0,
  "size": 0.022,
  "value": 990.0,
  "confidence": 0.85,
  "order_id": "abc123",
  "status": "executed" | "paper_trade",
  "fees": 0.99,
  "portfolio_context": {...}
}
```

## Extending the System

### Adding New Technical Indicators
Add to `LLMTradingStrategy.calculate_technical_indicators()` in src/strategy/llm_strategy.py:
```python
df['your_indicator'] = ta.your_indicator(df['close'], length=14)
```
Then add signal logic in `calculate_technical_signals()` and update `_combine_signals()` weights.

### Adding New Trading Pairs
Update `.env`:
```bash
TRADING_PAIRS=BTC-USDT,ETH-USDT,SOL-USDT,MATIC-USDT
```

### Custom Risk Parameters
Modify `src/core/config.py` RiskConfig dataclass or override via environment variables.

### Integration with External Systems
- Trade notifications: Implement in `src/notifications/service.py`
- Database storage: Extend `src/data/database.py` and repositories
- Custom indicators: Add to `src/market/indicators.py`

## Security Considerations

- API credentials stored in `.env` (gitignored)
- Never commit `.env` or any file with real API keys
- Docker mounts `.env` as read-only volume
- Paper trading mode is default (safe)
- Auto trading requires explicit `--auto-trading` flag
- Minimum balance requirements (default $10 USDT)

## Monitoring and Debugging

### Logs
- Console output with emoji indicators for readability
- File logging to `logs/trading_agent.log` (rotated at 10MB, 5 backups)
- Log level configurable via `LOG_LEVEL` env var (default: INFO)

### Performance Metrics
View via interactive menu or programmatically:
```python
agent.show_performance_report(days=30)
stats = agent.get_trading_stats(days=30)
agent.show_trade_history(limit=10)
```

### Portfolio Summary
```python
agent.get_portfolio_summary()  # Full breakdown with positions
agent.show_open_positions()    # Focused view of open positions
```

## Common Workflows

### Development Cycle
1. Make changes to strategy/risk logic
2. Test in paper trading mode: `python main.py --paper-trading`
3. Run single analysis cycle (menu option 1)
4. Review output and adjust parameters
5. Run linting: `black src/ && flake8 src/`

### Deployment to Production
1. Set `FAST_MODE=true` for production speed
2. Configure `TRADING_INTERVAL_MINUTES` (e.g., 15)
3. Set `MIN_CONFIDENCE_THRESHOLD` conservatively (0.7-0.8)
4. Set `MAX_DAILY_TRADES` limit (e.g., 5-10)
5. Enable auto-trading: `python main.py --auto-mode --auto-trading`
6. Monitor logs and trade history

### Debugging Trading Decisions
1. Run single symbol analysis: menu option 4
2. Review `technical_signals` dictionary in strategy output
3. Check `buy_factors` and `sell_factors` counts
4. Verify `portfolio_context` in trade records
5. Examine signal weights in strategy response
