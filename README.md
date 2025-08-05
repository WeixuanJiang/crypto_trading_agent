# Crypto Trading Agent with LLM Integration

A sophisticated headless cryptocurrency trading agent that combines Large Language Model (LLM) analysis with advanced quantitative trading methods for automated trading on KuCoin exchange. Features an intelligent "buy low, sell high" strategy with comprehensive risk management and P&L tracking.

## Features
### 🤖 LLM-Powered Strategy
- **Fast Mode**: Quick analysis without LLM for speed (⚡ 3-5x faster)
- **Full Mode**: Complete analysis with AWS Bedrock Claude 3.5 Haiku sentiment analysis
- **Cross-Region Inference**: Automatic failover across AWS regions for high availability
- **Smart Caching**: LLM results cached to avoid redundant API calls
- **Intelligent Decision Making**: Combines AI insights with technical analysis
- **News Sentiment Analysis**: Real-time market sentiment from news sources
- **Buy Low, Sell High Strategy**: Advanced price position analysis for optimal entry/exit points

### 📊 Quantitative Analysis
- **Enhanced Technical Indicators**: SMA (20/50/200), EMA, MACD with histogram, RSI (14/7), Bollinger Bands, Stochastic, ATR
- **Advanced Oscillators**: Williams %R, Commodity Channel Index (CCI), Rate of Change (ROC), Money Flow Index (MFI)
- **Price Position Analysis**: Dynamic support/resistance levels and price positioning
- **Signal Generation**: Multi-timeframe technical analysis with weighted signal combination
- **Pattern Recognition**: Market trend and reversal pattern detection with confidence scoring
- **Backtesting Support**: Historical performance analysis with comprehensive metrics

### 🛡️ Risk Management
- **Dynamic Position Sizing**: Risk-based position sizing with 2% max portfolio risk per trade
- **Stop Loss Management**: ATR-based and percentage-based stops (default 5%)
- **Portfolio Risk Control**: Maximum 10% position size, 20% total exposure limits
- **Trade Validation**: Pre-trade risk checks with confidence thresholds
- **Real-time Monitoring**: Continuous portfolio exposure and P&L tracking
- **Safety Constraints**: Minimum balance requirements and daily trade limits

### 📈 Market Data Integration
- **Real-time Data**: Live price feeds from KuCoin
- **Historical Analysis**: OHLCV data with technical indicators
- **Market Metrics**: Volatility, volume analysis, and trend detection
- **News Integration**: Market sentiment from news sources

### 🚀 Auto Trading Features
- **Auto Trading**: Place real buy/sell orders based on AI predictions
- **Paper Trading**: Safe testing mode without real money
- **Trading Controls**: Enable/disable auto trading, adjust parameters on the fly
- **Safety Features**: Confidence thresholds, daily trade limits, emergency stops
- **Trade History**: Track all executed and paper trades

### 📊 Trading History & P&L Tracking
- **Enhanced Trade History**: Detailed trade logs with P&L display and status indicators
- **Automatic P&L Calculation**: Real-time profit/loss calculation for all trades
- **Portfolio Tracking**: Track positions, average prices, and unrealized P&L
- **Performance Metrics**: Win rate, profit factor, best/worst trades, and risk metrics
- **Persistent Storage**: Trade data saved to JSON files for historical analysis
- **Comprehensive Reports**: Detailed performance reports with trading statistics

## 🐳 Docker Deployment

The easiest way to run the Crypto Trading Agent is using Docker. This provides a consistent environment and eliminates dependency issues.

### Quick Start with Docker

```bash
# Clone and navigate to the project
git clone <repository-url>
cd crypto_trading_agent

# Create your .env file with API credentials
cp .env.example .env
# Edit .env with your KuCoin API credentials

# Build and run the application
docker-compose up --build -d
```

The application will automatically run as a headless trading bot in Docker environments and output logs to the console.

### Docker Commands
- `docker-compose up --build -d` - Build and start the application in detached mode
- `docker-compose down` - Stop and remove the application
- `docker-compose logs -f crypto-trading-agent` - View real-time logs
- `docker-compose restart crypto-trading-agent` - Restart the application
- `docker ps` - Check container status

### Docker Environment Detection
The application automatically detects Docker environments and enables automated mode:
- Checks for `DOCKER_CONTAINER=true` environment variable
- Checks for `/.dockerenv` file existence
- Automatically runs continuous trading cycles without user interaction

### 🖥️ Headless Operation
- **Console-Based Interface**: All output via structured logging
- **Automated Mode**: Continuous operation without user interaction
- **Docker-Optimized**: Automatic detection and configuration for containerized environments
- **Real-time Monitoring**: Comprehensive logging for trade execution and portfolio status

### Additional Features
- **Position Management**: Track open positions and calculate portfolio-wide P&L
- **Export Functionality**: Export trade history to Excel/CSV for external analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- KuCoin API credentials
- AWS account with Bedrock access (optional, for LLM features)
- AWS CLI configured or AWS credentials set up

### Setup

1. **Clone or create the project directory:**
```bash
mkdir crypto_trading_agent
cd crypto_trading_agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note**: The project uses the unified KuCoin API client:
```python
from kucoin.client import Client
```

3. **Set up environment variables:**

The application uses `python-dotenv` to automatically load environment variables from a `.env` file. Create a `.env` file in the project root with your API credentials:

```
KUCOIN_API_KEY=your_api_key_here
KUCOIN_API_SECRET=your_api_secret_here
KUCOIN_API_PASSPHRASE=your_api_passphrase_here
AWS_REGION=us-east-1
AWS_CROSS_REGION_INFERENCE=true
# AWS credentials will be handled via AWS CLI or IAM roles
```

The `.env` file is automatically loaded when the `CryptoTradingAgent` is initialized, so you don't need to manually set environment variables.

**Important**: For auto trading, ensure your KuCoin API key has trading permissions enabled.

### KuCoin API Setup

1. **Create KuCoin Account**: Sign up at [KuCoin](https://www.kucoin.com)
2. **Generate API Keys**:
   - Go to API Management in your account settings
   - Create new API key with trading permissions
   - Save your API Key, Secret, and Passphrase securely
3. **Set Permissions**: Ensure your API key has:
   - General permission
   - Trade permission
   - Transfer permission (if needed)

## Usage

### Quick Start

#### Paper Trading (Recommended for beginners)
```bash
# Clone and setup
git clone <repository>
cd crypto_trading_agent
pip install -r requirements.txt

# Create .env file with your credentials
cp .env.example .env
# Edit .env with your API keys

# Run in paper trading mode (default)
python main.py

# Or explicitly force paper trading
python main.py --paper-trading
```

#### Auto Trading (Advanced users)
```bash
# Ensure you have trading permissions on KuCoin
# Run with auto trading enabled
python main.py --auto-trading

# For server deployment with auto trading
python main.py --auto-mode --auto-trading

# Interactive mode - toggle auto trading in menu
python main.py
# Use option 6 to toggle auto trading on/off
```

## Performance Modes

### Fast Mode (⚡ Recommended)
- **Speed**: 3-5x faster analysis
- **Features**: Technical analysis + quantitative signals
- **Use Case**: Live trading, frequent analysis
- **Time**: ~2-3 seconds per symbol

### Full Mode (🧠 Comprehensive)
- **Speed**: Slower due to LLM calls
- **Features**: All Fast Mode + Claude 3.5 Haiku sentiment analysis
- **Use Case**: Research, detailed market analysis
- **Time**: ~10-15 seconds per symbol

```bash
# Test performance comparison
python performance_example.py
```

1. **Paper Trading (Safe Mode)**:
```python
from main import CryptoTradingAgent

# Initialize in paper trading mode (default)
agent = CryptoTradingAgent(auto_trading_enabled=False)
agent.run_analysis_cycle()
```

2. **Auto Trading (Live Mode)**:
```python
# Initialize with auto trading enabled
agent = CryptoTradingAgent(auto_trading_enabled=True)

# Set safety parameters
agent.set_trading_parameters(
    min_confidence_threshold=0.8,  # High confidence required
    max_daily_trades=5             # Limit daily trades
)

# Start trading
agent.run_analysis_cycle()
```

### Trading History & P&L Tracking

#### View Trade History
```python
from main import CryptoTradingAgent

agent = CryptoTradingAgent()

# Show recent trade history with P&L
agent.show_trade_history(limit=10)

# Get comprehensive trading statistics
stats = agent.get_trading_stats(days=30)
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"Total P&L: ${stats['total_pnl']:.2f}")
```

#### Portfolio Management
```python
# View current portfolio summary
agent.get_portfolio_summary()

# Calculate P&L for specific symbol
btc_pnl = agent.calculate_portfolio_pnl('BTC-USDT')
print(f"BTC P&L: ${btc_pnl:.2f}")

# Calculate total portfolio P&L
total_pnl = agent.calculate_portfolio_pnl()
print(f"Total P&L: ${total_pnl:.2f}")
```

#### Performance Reports
```python
# Generate detailed performance report
agent.show_performance_report(days=7)  # Last 7 days

# Export trade history to CSV
agent.trade_tracker.export_trades_to_csv()
```

### Running the Application

#### Command Line Options

The application supports several command-line arguments for flexible operation:

```bash
# Interactive mode (default - not available in Docker)
python main.py

# Enable automatic trading execution
python main.py --auto-trading

# Run in automated mode (for server/Docker deployment)
python main.py --auto-mode

# Combine automated mode with auto trading
python main.py --auto-mode --auto-trading

# Force paper trading mode (overrides auto-trading)
python main.py --paper-trading

# Check AWS service status
python main.py --check-aws

# View help and all available options
python main.py --help
```

#### Docker Automatic Mode
When running in Docker, the application automatically detects the containerized environment and runs in automated mode, eliminating the need for interactive input.

**Command Line Arguments:**
- `--auto-trading`: Enable automatic trade execution (real money)
- `--auto-mode`: Run in automated mode for server deployment
- `--paper-trading`: Force paper trading mode (simulation only)
- `--check-aws`: Check AWS service status and exit

### Interactive Mode

**Note**: Interactive mode is only available when running locally (not in Docker).

Run the main script to access the interactive menu:

```bash
python main.py
```

The interactive menu displays the current auto trading status and provides the following options:

**Menu Options:**
1. **Run single analysis cycle** - Analyze all configured trading pairs once
2. **Start automated trading** - Begin continuous trading with specified intervals
3. **Show portfolio status** - Display current account balance and positions
4. **Analyze specific symbol** - Deep analysis of a particular trading pair
5. **Check AWS status** - Verify AWS service connectivity
6. **Toggle auto trading** - Enable/disable automatic trade execution
7. **Exit** - Close the application

**Auto Trading Status:**
- 🚀 ENABLED: Real trades will be executed
- 📊 DISABLED: Analysis only mode (paper trading)

### Headless/Docker Mode
When running in Docker or with `--auto-mode`, the application runs continuously without user interaction, automatically executing trading cycles at configured intervals.

### Configuration

**Default Trading Pairs:**
- BTC-USDT
- ETH-USDT
- ADA-USDT
- DOT-USDT

You can modify the trading pairs in the `CryptoTradingAgent` class initialization.

### Safety Features

⚠️ **IMPORTANT SAFETY NOTES:**

1. **Paper Trading Mode**: By default, the agent runs in simulation mode and doesn't place actual trades
2. **Manual Activation**: To enable real trading, uncomment the order placement code in `execute_trade()` method
3. **Risk Limits**: Built-in safeguards prevent excessive position sizes and losses
4. **Minimum Balance**: Requires minimum $10 USDT balance for trading

## Trading Modes

### Paper Trading (Default)
- Safe testing mode that simulates trades
- No real money involved
- Perfect for testing strategies and parameters
- All analysis and signals work the same as live trading

### Auto Trading (Live Mode)
- Places real buy/sell orders on KuCoin
- Uses actual account balance
- Includes safety features:
  - Minimum confidence threshold (default: 0.7)
  - Daily trade limits (default: 10)
  - Position sizing controls
  - Emergency stop functionality

### Safety Features

#### Command Line Safety Controls
```bash
# Always start in safe paper trading mode
python main.py --paper-trading

# Enable auto trading only when ready
python main.py --auto-trading

# Check system status before trading
python main.py --check-aws
```

#### Programmatic Safety Controls
```python
# Set conservative parameters
agent.set_trading_parameters(
    min_confidence_threshold=0.8,  # Only trade on high confidence
    max_daily_trades=3             # Limit daily trades
)

# Control trading on the fly
agent.disable_auto_trading()  # Switch to paper mode
agent.enable_auto_trading()   # Resume live trading

# Monitor performance
stats = agent.get_trading_stats()
agent.show_trade_history()
```

#### Interactive Safety Controls
- **Menu Option 6**: Toggle auto trading on/off during runtime
- **Real-time Status**: Auto trading status displayed in menu header
- **Safe Defaults**: Application starts in analysis-only mode by default

## Architecture

### Core Components

#### 1. `main.py` - Trading Agent Controller
- **CryptoTradingAgent**: Main orchestrator class
- **Automated Trading Loops**: Continuous operation for Docker/headless mode
- **Trade Execution**: Order management and execution
- **Portfolio Monitoring**: Real-time account tracking

#### 2. `trading_strategy.py` - LLM Strategy Engine
- **LLMTradingStrategy**: AWS Bedrock Claude 3.5 Haiku integration
- **Technical Analysis**: Comprehensive indicator calculations
- **Signal Generation**: Advanced technical pattern recognition
- **Cross-Region Inference**: Multi-region AWS Bedrock support

#### 3. `market_data.py` - Data Management
- **MarketDataManager**: KuCoin API integration
- **Historical Data**: OHLCV data retrieval
- **Real-time Feeds**: Live price monitoring
- **Market Metrics**: Volatility and trend analysis

#### 4. `risk_management.py` - Risk Control
- **RiskManager**: Position sizing and validation
- **Stop Loss Management**: Multiple stop-loss strategies
- **Portfolio Metrics**: Risk exposure calculation
- **Trade Statistics**: Performance tracking

### Data Flow

1. **Market Data Collection** → Historical and real-time data from KuCoin
2. **Technical Analysis** → Calculate indicators and generate signals
3. **LLM Analysis** → Claude 3.5 Haiku sentiment and strategy analysis (Full Mode)
4. **Signal Combination** → Merge LLM and technical analysis signals
5. **Risk Assessment** → Validate trade against risk parameters
6. **Trade Execution** → Place orders (if enabled)
7. **Portfolio Monitoring** → Track performance and adjust
8. **Continuous Operation** → Automated cycles for headless/Docker deployment

## Strategy Details


### Buy Low, Sell High Strategy

The agent implements an intelligent "buy low, sell high" approach with:
- **Price Position Analysis**: Calculates relative price position within recent trading range
- **Dynamic Support/Resistance**: Automatically identifies key price levels
- **Multi-Oscillator Confirmation**: Uses RSI, Williams %R, CCI, and MFI for oversold/overbought signals
- **Trend Alignment**: Ensures trades align with longer-term trends using 200-period SMA
- **Confidence Scoring**: Weighted signal combination with higher confidence for extreme price positions

### LLM Integration

The agent uses AWS Bedrock Claude 3.5 Haiku to:
- Analyze market sentiment from news and social media
- Interpret complex market conditions
- Generate trading recommendations
- Provide cross-region inference for high availability
- Complement technical analysis with fundamental insights

### Enhanced Technical Indicators

- **Trend Following**: Multiple SMA periods (20/50/200), EMA crossovers
- **Momentum**: Enhanced RSI (14/7 periods), Stochastic oscillator, Williams %R
- **Volatility**: Bollinger Bands with position analysis, ATR
- **Volume**: Money Flow Index (MFI), Accumulation/Distribution line
- **Divergence**: MACD with histogram confirmation
- **Oscillators**: Commodity Channel Index (CCI), Rate of Change (ROC)

### Quantitative Analysis

- **Signal Generation**: Multi-timeframe technical analysis with 15+ indicators
- **Pattern Recognition**: Price action and trend analysis with price positioning
- **Signal Weighting**: Optimized weights favoring price position and oversold/overbought conditions
- **Adaptive Thresholds**: Dynamic buy/sell decision points based on market conditions
- **Confidence Scoring**: Advanced confidence calculation considering signal strength and alignment

### Risk Management

- **Position Sizing**: 2% maximum risk per trade with dynamic sizing
- **Stop Loss**: 5% maximum loss per position with ATR-based adjustments
- **Portfolio Exposure**: Maximum 10% per position, 20% total account exposure
- **Trade Validation**: Pre-trade checks including confidence thresholds and balance requirements
- **Real-time Monitoring**: Continuous P&L tracking and portfolio exposure management

## Performance Monitoring

### Key Metrics

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss estimation

### Logging

The agent provides comprehensive logging:
- Trade signals and execution details
- Risk management decisions
- Portfolio performance metrics
- Error handling and debugging information

## Testing & Validation

### Strategy Testing

The agent includes comprehensive testing tools to validate trading strategies:

**Features:**
- Simulates market data with realistic price movements
- Analyzes strategy performance and signal distribution
- Provides detailed statistics on buy/sell timing
- Shows confidence levels and price positioning
- Demonstrates key strategy improvements

**Sample Output:**
- Action distribution (buy vs sell signals)
- Average price positions for buy/sell signals
- Confidence scoring analysis
- Recent trading signals with explanations

**Features:**
- Tests P&L calculation accuracy
- Validates portfolio tracking functionality
- Demonstrates trade history management
- Shows performance metrics calculation

### Performance Validation

#### Strategy Performance Metrics
- **Signal Quality**: Analyzes buy signals at low price positions (typically 0.1-0.3)
- **Sell Timing**: Validates sell signals at high price positions (typically 0.7-0.9)
- **Confidence Distribution**: Ensures high-confidence signals (average 0.8+)
- **Risk Management**: Validates position sizing and stop-loss implementation

#### Backtesting Results
- **Win Rate Tracking**: Historical performance analysis
- **Drawdown Analysis**: Maximum loss periods
- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Gross profit to gross loss ratio

## Customization

### Adding New Trading Pairs

```python
# In main.py, modify the trading_pairs list
self.trading_pairs = ['BTC-USDT', 'ETH-USDT', 'YOUR-PAIR']
```

### Adjusting Risk Parameters

```python
# In risk_management.py, modify RiskManager settings
self.max_position_size = 0.02  # 2% of account per trade
self.max_portfolio_exposure = 0.20  # 20% total exposure
```

### Custom Indicators

Add new technical indicators in `trading_strategy.py`:

```python
def custom_indicator(self, df):
    # Your custom indicator logic
    return indicator_values
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API credentials
   - Check internet connection
   - Ensure API permissions are correct

2. **Insufficient Balance**
   - Minimum $10 USDT required
   - Check account balance
   - Verify trading permissions

3. **Missing Dependencies**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **LLM Features Disabled**
   - Set OPENAI_API_KEY environment variable
   - Verify OpenAI account has sufficient credits

### Debug Mode

Enable detailed logging by modifying the logging level in each module.

## Disclaimer

⚠️ **IMPORTANT DISCLAIMER:**

This trading agent is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software.

**Before using with real money:**

1. **Paper Trading**: Use paper trading mode extensively to validate performance
2. **Understand Risk Management**: Review all risk management features and position sizing
3. **Start Small**: Begin with minimal amounts to test real market conditions
4. **Monitor Closely**: Use the comprehensive P&L tracking and performance reports
5. **Strategy Validation**: Ensure the "buy low, sell high" signals align with your expectations
6. **Never Risk More**: Only invest money you can afford to lose completely

**Key Safety Features:**
- Comprehensive testing framework with validation scripts
- Paper trading mode for safe strategy testing
- Real-time P&L tracking and portfolio monitoring
- Advanced risk management with position sizing limits
- Confidence-based trade execution with safety thresholds

## License

This project is provided as-is for educational purposes. Use at your own risk.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Test with paper trading first
4. Ensure all dependencies are properly installed

---

**Happy Trading! 🚀**

*Remember: The best trading strategy is the one you understand completely.*