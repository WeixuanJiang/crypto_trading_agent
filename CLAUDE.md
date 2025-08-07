# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (copy example and edit)
cp .env.example .env
```

### Running the Application
```bash
# Run in interactive mode (default)
python main.py

# Run with auto trading enabled
python main.py --auto-trading

# Run in automated mode (for server/Docker deployment)
python main.py --auto-mode

# Run with Flask API server
python main.py

# Force paper trading mode (simulation only)
python main.py --paper-trading

# Check AWS service status
python main.py --check-aws
```

### Docker Commands
```bash
# Build and start the application in detached mode
docker-compose up --build -d

# Stop and remove the application
docker-compose down

# View real-time logs
docker-compose logs -f crypto-trading-agent

# Restart the application
docker-compose restart crypto-trading-agent
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test module
pytest tests/test_specific_module.py
```

### Linting and Formatting
```bash
# Run flake8 linter
flake8 src/

# Format code with black
black src/

# Run type checking with mypy
mypy src/

# Run security checks with bandit
bandit -r src/
```

## Architecture Overview

### Core Components

1. **Main Controller (`main.py` - `CryptoTradingAgent`)**
   - Orchestrates the trading logic
   - Handles API connections to KuCoin
   - Manages the trading cycle and execution
   - Provides a Flask API interface for monitoring and control

2. **Strategy System (`src/strategy/`)**
   - `base.py`: Abstract base classes for trading strategies
   - `llm.py`: LLM-powered trading strategies using AWS Bedrock
   - `technical.py`: Technical analysis trading strategies
   - `hybrid.py`: Combined LLM and technical analysis strategies

3. **Market Data Management (`src/market/`)**
   - `data_manager.py`: Fetches and processes market data
   - `analysis.py`: Market analysis and metrics calculation
   - `indicators.py`: Technical indicators calculation

4. **Risk Management (`src/risk/`)**
   - `manager.py`: Risk assessment and position sizing
   - Controls exposure limits and stop losses

5. **LLM Integration (`src/llm/`)**
   - `bedrock_client.py`: Client for AWS Bedrock Claude 3.5 Haiku
   - Cross-region inference for high availability

6. **Data Layer (`src/data/`)**
   - `database.py`: SQLite database connection and management
   - `repositories.py`: Data access objects for trading records
   - `models.py`: Data models for market data and analysis

7. **Portfolio Management (`src/portfolio/`)**
   - `manager.py`: Tracks positions and portfolio status
   - Calculates P&L and portfolio metrics

8. **Configuration (`src/core/`)**
   - `config.py`: Centralized configuration management
   - Environment variables and config files handling

### Data Flow

1. Market data is fetched from KuCoin API
2. Technical analysis indicators are calculated
3. LLM analyzes market data and provides recommendations (in Full mode)
4. Strategy signals are combined and validated
5. Risk management applies position sizing and constraints
6. Trade decisions are executed (or simulated in paper trading mode)
7. Results are tracked and reported via API or CLI

## Key Configuration

The application uses a hierarchical configuration system:

1. Default configuration in `src/core/config.py`
2. Environment variables (loaded from `.env` file)
3. Settings file (`config/settings.json`)

Important environment variables:
- `KUCOIN_API_KEY`, `KUCOIN_API_SECRET`, `KUCOIN_API_PASSPHRASE`: API credentials
- `TRADING_PAIRS`: Comma-separated list of trading pairs (default: BTC-USDT,ETH-USDT,ADA-USDT,DOT-USDT)
- `FAST_MODE`: Enable fast mode without LLM (default: true)
- `AWS_REGION`: AWS region for Bedrock (default: us-east-1)
- `AWS_CROSS_REGION_INFERENCE`: Enable cross-region failover (default: true)
- `TRADING_INTERVAL_MINUTES`: Minutes between trading cycles (default: 60)
- `MIN_CONFIDENCE_THRESHOLD`: Minimum confidence to execute trades (default: 0.7)

## Security Notes

- Never commit `.env` file with API keys
- API keys should have appropriate permissions (read-only for paper trading)
- Use paper trading mode for testing
- Set conservative trading limits (`MIN_CONFIDENCE_THRESHOLD`, `MAX_DAILY_TRADES`)