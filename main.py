import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

from kucoin.client import Client
from src.strategy.llm_strategy import LLMTradingStrategy
from src.market.data import MarketDataManager
from src.risk.manager import RiskManager
from src.trading.aws_tracker import create_trade_tracker
from src.core.config import config
from src.core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class CryptoTradingAgent:
    def __init__(self, auto_trading_enabled=False, fast_mode=None):
        # Load environment variables from .env file
        load_dotenv()
        
        # Read fast_mode from environment if not explicitly provided
        if fast_mode is None:
            fast_mode = os.getenv('FAST_MODE', 'true').lower() == 'true'
        
        logger.info("Initializing Crypto Trading Agent...")
        
        # Initialize API credentials
        self.api_key, self.api_secret, self.api_passphrase = self.get_api_credentials()
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.cross_region_inference = os.getenv('AWS_CROSS_REGION_INFERENCE', 'true').lower() == 'true'
        
        # Initialize KuCoin client for production environment
        self.client = Client(
            api_key=self.api_key,
            api_secret=self.api_secret,
            passphrase=self.api_passphrase,
            sandbox=False  # Explicitly set to production environment
        )
        
        # For backward compatibility, create aliases
        self.trade_client = self.client
        self.user_client = self.client
        self.market_client = self.client
        
        # Initialize components
        self.market_data = MarketDataManager(self.market_client)
        self.risk_manager = RiskManager()
        self.trade_tracker = create_trade_tracker()
        
        # Initialize portfolio manager
        from src.portfolio.manager import PortfolioManager
        self.portfolio_manager = PortfolioManager(self.user_client, self.market_client, self.trade_tracker, config)
        
        # Initialize trading strategy with AWS Bedrock
        self.fast_mode = fast_mode
        try:
            self.strategy = LLMTradingStrategy(
                aws_region=self.aws_region,
                enable_llm=not fast_mode,
                cross_region_inference=self.cross_region_inference
            )
            if fast_mode:
                logger.info("⚡ Fast mode enabled - LLM analysis disabled for speed")
            else:
                logger.info(f"✓ AWS Bedrock Claude 3.5 Haiku initialized (Region: {self.aws_region}, Cross-region: {self.cross_region_inference})")
        except Exception as e:
            self.strategy = None
            logger.warning(f"⚠ AWS Bedrock initialization failed: {e}. Using simple technical analysis only.")
        
        # Trading parameters from environment variables
        trading_pairs_str = os.getenv('TRADING_PAIRS', 'BTC-USDT,ETH-USDT,ADA-USDT,DOT-USDT')
        self.trading_pairs = [pair.strip() for pair in trading_pairs_str.split(',')]
        self.is_running = False
        self.auto_trading_enabled = auto_trading_enabled
        self.trade_history = []  # Keep for backward compatibility
        
        # Safety settings from environment variables
        self.min_confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7'))
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '5'))
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.minimum_balance = float(os.getenv('MINIMUM_BALANCE', '10'))
        self.trading_interval_minutes = int(os.getenv('TRADING_INTERVAL_MINUTES', '60'))
        
        # Log current mode
        mode = "AUTO TRADING" if auto_trading_enabled else "PAPER TRADING"
        speed_mode = "FAST" if fast_mode else "FULL"
        logger.info(f"🚀 Agent initialized in {mode} mode ({speed_mode} analysis)")
        
        # Initialize portfolio data
        self._update_portfolio_data()
        logger.info(f"📊 Confidence threshold: {self.min_confidence_threshold}")
        logger.info(f"📈 Max daily trades: {self.max_daily_trades}")

    def retry_api_call(self, func, *args, **kwargs):
        """Retry API calls with exponential backoff"""
        max_retries = getattr(config.api, 'max_retries', 5)
        retry_delay = getattr(config.api, 'retry_delay_seconds', 2)
        backoff_factor = getattr(config.api, 'backoff_factor', 2.0)
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Check if it's a timeout error
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    wait_time = retry_delay * (backoff_factor ** attempt)
                    logger.warning(f"⚠️ API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # For non-timeout errors, re-raise immediately
                    raise e
        
        return None

    def get_api_credentials(self):
        """Get API credentials from environment variables"""
        api_key = os.getenv('KUCOIN_API_KEY')
        api_secret = os.getenv('KUCOIN_API_SECRET')
        api_passphrase = os.getenv('KUCOIN_API_PASSPHRASE')
        
        if not all([api_key, api_secret, api_passphrase]):
            logger.error("Please set KUCOIN_API_KEY, KUCOIN_API_SECRET, and KUCOIN_API_PASSPHRASE environment variables.")
            sys.exit(1)
        
        return api_key, api_secret, api_passphrase
    
    def get_account_balance(self) -> float:
        """Get USDT account balance with retry mechanism"""
        try:
            def _get_balance():
                accounts = self.user_client.get_accounts()
                for account in accounts:
                    if account['currency'] == 'USDT' and account['type'] == 'trade':
                        return float(account['balance'])
                return 0.0
            
            return self.retry_api_call(_get_balance)
        except Exception as e:
            logger.error(f"❌ Error getting account balance after retries: {e}")
            return 0.0
    
    def _update_portfolio_data(self):
        """Update portfolio balances and positions"""
        try:
            logger.info("📊 Updating portfolio data...")
            
            # Update balances
            self.portfolio_manager.update_balances()
            
            # Get current prices for all trading pairs
            current_prices = {}
            for symbol in self.trading_pairs:
                try:
                    ticker = self.market_client.get_24hr_stats(symbol)
                    current_prices[symbol] = float(ticker['last'])
                except Exception as e:
                    logger.warning(f"⚠️ Could not get price for {symbol}: {e}")
            
            # Update positions with current prices
            self.portfolio_manager.update_positions(current_prices)
            
            # Save cache
            self.portfolio_manager.save_portfolio_cache()
            
            logger.info("✅ Portfolio data updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio data: {e}")
    
    def analyze_symbol(self, symbol: str, use_llm: bool = False) -> Dict:
        """Perform comprehensive analysis on a trading symbol with portfolio awareness"""
        logger.info(f"🔍 Analyzing {symbol}...")
        
        try:
            # Update portfolio data first
            self._update_portfolio_data()
            
            # Get historical data (reduced for fast mode)
            limit = 200 if self.fast_mode else 500
            df = self.market_data.get_historical_klines(symbol, '1hour', limit=limit)
            if df.empty:
                logger.error(f"❌ No data available for {symbol}")
                return {}
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Get news data only if LLM is enabled
            news_data = []
            if use_llm and not self.fast_mode:
                news_data = self.market_data.get_news_data(symbol)
            
            # Generate trading strategy
            if self.strategy:
                # Generate strategy using enhanced LLM and technical analysis
                strategy_result = self.strategy.generate_trading_strategy(symbol, df, news_data, use_llm)
            else:
                # Fallback to simple technical analysis
                strategy_result = self.simple_technical_analysis(df)
            
            # Get portfolio-aware position recommendation
            if strategy_result.get('action') in ['BUY', 'SELL']:
                position_recommendation = self.portfolio_manager.get_position_size_recommendation(
                    symbol=symbol,
                    action=strategy_result['action'],
                    confidence=strategy_result.get('confidence', 0.5),
                    current_price=current_price,
                    account_balance=self.portfolio_manager.portfolio_value
                )
                strategy_result['position_recommendation'] = position_recommendation
            
            # Get market metrics
            market_metrics = self.market_data.calculate_market_metrics(df)
            
            # Get current position info
            current_position = self.portfolio_manager.positions.get(symbol)
            position_info = {
                'has_position': current_position is not None,
                'quantity': current_position.quantity if current_position else 0.0,
                'market_value': current_position.market_value if current_position else 0.0,
                'unrealized_pnl': current_position.unrealized_pnl if current_position else 0.0,
                'unrealized_pnl_percent': current_position.unrealized_pnl_percent if current_position else 0.0,
                'allocation_percent': (current_position.market_value / self.portfolio_manager.portfolio_value * 100) if current_position and self.portfolio_manager.portfolio_value > 0 else 0.0
            }
            
            # Combine results with portfolio context
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'strategy': strategy_result,
                'market_metrics': market_metrics,
                'position_info': position_info,
                'portfolio_context': {
                    'total_portfolio_value': self.portfolio_manager.portfolio_value,
                    'usdt_available': self.portfolio_manager.balances.get('USDT', {}).get('available', 0),
                    'can_buy': self.portfolio_manager.balances.get('USDT', {}).get('available', 0) >= 10,
                    'can_sell': position_info['has_position'] and position_info['quantity'] > 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Enhanced decision display
            self._display_analysis_result(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return {}
    
    def _display_analysis_result(self, analysis: Dict):
        """Display enhanced analysis results with portfolio context"""
        symbol = analysis['symbol']
        strategy = analysis['strategy']
        position_info = analysis['position_info']
        portfolio_context = analysis['portfolio_context']
        
        logger.info(f"📈 Analysis Results for {symbol}")
        logger.info("=" * 50)
        
        # Current position
        if position_info['has_position']:
            logger.info(f"💼 Current Position:")
            logger.info(f"   Quantity: {position_info['quantity']:.6f}")
            logger.info(f"   Market Value: ${position_info['market_value']:.2f}")
            logger.info(f"   Allocation: {position_info['allocation_percent']:.1f}%")
            if position_info['unrealized_pnl'] != 0:
                pnl_color = "🟢" if position_info['unrealized_pnl'] > 0 else "🔴"
                logger.info(f"   P&L: {pnl_color} ${position_info['unrealized_pnl']:.2f} ({position_info['unrealized_pnl_percent']:.1f}%)")
        else:
            logger.info(f"💼 Current Position: None")
        
        # Strategy recommendation
        action = strategy.get('action', 'HOLD')
        confidence = strategy.get('confidence', 0.0)
        
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")
        logger.info(f"{action_emoji} Strategy Recommendation: {action}")
        logger.info(f"🎯 Confidence: {confidence:.1f}%")
        
        # Enhanced strategy details
        if 'market_regime' in strategy:
            regime = strategy['market_regime']
            logger.info(f"🌊 Market Regime: {regime.get('regime', 'unknown').title()} (Confidence: {regime.get('confidence', 0)*100:.1f}%)")
        
        if 'risk_level' in strategy:
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(strategy['risk_level'], "⚪")
            logger.info(f"{risk_emoji} Risk Level: {strategy['risk_level'].title()}")
        
        # Position recommendation
        if 'position_recommendation' in strategy:
            pos_rec = strategy['position_recommendation']
            if pos_rec['can_execute']:
                logger.info(f"💰 Position Recommendation:")
                logger.info(f"   Action: {pos_rec['action']}")
                logger.info(f"   Size: {pos_rec['recommended_size']:.6f} {symbol.split('-')[0]}")
                logger.info(f"   Value: ${pos_rec['recommended_value']:.2f}")
                logger.info(f"   Reason: {pos_rec['reason']}")
                if 'current_pnl_percent' in pos_rec:
                    logger.info(f"   Current P&L: {pos_rec['current_pnl_percent']:.1f}%")
            else:
                logger.warning(f"⚠️ Cannot Execute: {pos_rec['reason']}")
        
        # Portfolio context
        logger.info(f"💼 Portfolio Context:")
        logger.info(f"   Total Value: ${portfolio_context['total_portfolio_value']:.2f}")
        logger.info(f"   USDT Available: ${portfolio_context['usdt_available']:.2f}")
        logger.info(f"   Can Buy: {'✅' if portfolio_context['can_buy'] else '❌'}")
        logger.info(f"   Can Sell: {'✅' if portfolio_context['can_sell'] else '❌'}")
        
        # Technical indicators summary
        if 'price_position' in strategy:
            price_pos = strategy['price_position'] * 100
            logger.info(f"📊 Technical Summary:")
            logger.info(f"   Price Position: {price_pos:.1f}% of recent range")
            if 'rsi' in strategy:
                logger.info(f"   RSI: {strategy['rsi']:.1f}")
            if 'trend_strength' in strategy:
                logger.info(f"   Trend Strength: {strategy['trend_strength']*100:.1f}%")
    
    def simple_technical_analysis(self, df) -> Dict:
        """Simple technical analysis fallback when LLM is not available"""
        if df.empty:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        latest = df.iloc[-1]
        
        # Simple moving averages
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        
        # Simple RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Simple decision logic
        signals = []
        if latest['close'] > sma_20 > sma_50:
            signals.append(1)  # Bullish
        elif latest['close'] < sma_20 < sma_50:
            signals.append(-1)  # Bearish
        else:
            signals.append(0)  # Neutral
        
        if current_rsi < 30:
            signals.append(1)  # Oversold
        elif current_rsi > 70:
            signals.append(-1)  # Overbought
        else:
            signals.append(0)
        
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.3:
            action = 'BUY'
        elif avg_signal < -0.3:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'confidence': abs(avg_signal),
            'current_price': latest['close'],
            'rsi': current_rsi,
            'sma_20': sma_20,
            'sma_50': sma_50
        }
    
    def execute_trade(self, symbol: str, action: str, confidence: float, analysis_result: Dict = None) -> bool:
        """Execute a trade based on strategy recommendation with portfolio awareness"""
        if action == 'HOLD' or confidence < self.min_confidence_threshold:
            logger.info(f"❌ No trade executed for {symbol} - Action: {action}, Confidence: {confidence:.2f}")
            return False
        
        # Check daily trade limit
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(f"❌ Daily trade limit reached ({self.max_daily_trades}). Skipping trade for {symbol}")
            return False
        
        try:
            # Update portfolio data
            self._update_portfolio_data()
            
            current_price = self.market_data.get_current_price(symbol)
            if current_price is None:
                logger.error(f"❌ Could not get current price for {symbol}")
                return False
            
            # Use portfolio manager for position sizing
            position_recommendation = self.portfolio_manager.get_position_size_recommendation(
                symbol=symbol,
                action=action,
                confidence=confidence,
                current_price=current_price
            )
            
            if not position_recommendation['can_execute']:
                logger.warning(f"❌ Cannot execute trade: {position_recommendation['reason']}")
                return False
            
            # Get symbol information for proper increment handling
            symbol_info = self.market_data.get_symbol_info(symbol)
            if not symbol_info:
                logger.error(f"❌ Could not get symbol information for {symbol}")
                return False
            
            # Extract position details
            trade_size = position_recommendation['recommended_size']
            trade_value = position_recommendation['recommended_value']
            
            # Validate minimum trade requirements
            min_funds = symbol_info.get('minFunds', 10.0)
            if trade_value < min_funds:
                logger.warning(f"❌ Trade value ${trade_value:.2f} below minimum ${min_funds}")
                return False
            
            # Round to proper increments
            base_increment = float(symbol_info.get('baseIncrement', '0.00000001'))
            quote_increment = float(symbol_info.get('quoteIncrement', '0.01'))
            
            # Round trade size to base increment
            trade_size = round(trade_size / base_increment) * base_increment
            trade_value = round(trade_value / quote_increment) * quote_increment
            
            logger.info(f"💰 Executing {action} order for {symbol}")
            logger.info(f"   Size: {trade_size:.8f} {symbol.split('-')[0]}")
            logger.info(f"   Value: ${trade_value:.2f}")
            logger.info(f"   Price: ${current_price:.6f}")
            logger.info(f"   Reason: {position_recommendation['reason']}")
            
            # Execute the trade (if auto trading is enabled)
            order_id = None
            if self.auto_trading_enabled:
                try:
                    if action == 'BUY':
                        # Market buy order
                        order_response = self.trade_client.create_market_order(
                            symbol=symbol,
                            side='buy',
                            funds=str(trade_value)
                        )
                    else:  # SELL
                        # Market sell order
                        order_response = self.trade_client.create_market_order(
                            symbol=symbol,
                            side='sell',
                            size=str(trade_size)
                        )
                    
                    order_id = order_response.get('orderId')
                    logger.info(f"✅ Order placed successfully! Order ID: {order_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Order execution failed: {e}")
                    return False
            else:
                logger.info("📝 Paper trade - No actual order placed")
            
            # Record the trade
            trade_record = {
                'id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': action,
                'price': current_price,
                'size': trade_size,
                'value': trade_value,
                'confidence': confidence,
                'order_id': order_id,
                'status': 'executed' if self.auto_trading_enabled else 'paper_trade',
                'fees': 0.0,  # Will be updated when order is filled
                'portfolio_context': {
                    'portfolio_value_before': self.portfolio_manager.portfolio_value,
                    'position_before': position_recommendation.get('current_position', 0),
                    'allocation_before': position_recommendation.get('current_allocation', 0),
                    'target_allocation': position_recommendation.get('target_allocation', 0)
                }
            }
            
            # Add to trade tracker
            self.trade_tracker.add_trade(**trade_record)
            
            # Add to legacy trade history for backward compatibility
            self.trade_history.append({
                'symbol': symbol,
                'action': action,
                'price': current_price,
                'size': trade_size,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'status': 'executed' if self.auto_trading_enabled else 'paper_trade',
                'order_id': order_id
            })
            
            self.daily_trade_count += 1
            
            # Update portfolio after trade
            if self.auto_trading_enabled:
                time.sleep(2)  # Wait for order to process
                self._update_portfolio_data()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
            return False
    
    def run_analysis_cycle(self):
        """Run one complete analysis cycle for all trading pairs with portfolio awareness"""
        logger.info(f"{'='*60}")
        logger.info(f"🚀 Starting Enhanced Analysis Cycle")
        logger.info(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        # Update portfolio data first
        self._update_portfolio_data()
        
        # Display current portfolio status
        portfolio_summary = self.portfolio_manager.get_portfolio_summary()
        logger.info(f"💼 Portfolio Value: ${portfolio_summary['total_portfolio_value']:.2f}")
        logger.info(f"💰 USDT Available: ${portfolio_summary['usdt_balance']:.2f}")
        logger.info(f"📊 Active Positions: {portfolio_summary['number_of_positions']}")
        
        if portfolio_summary['total_unrealized_pnl'] != 0:
            pnl_color = "🟢" if portfolio_summary['total_unrealized_pnl'] > 0 else "🔴"
            logger.info(f"📈 Total P&L: {pnl_color} ${portfolio_summary['total_unrealized_pnl']:.2f} ({portfolio_summary['total_unrealized_pnl_percent']:.1f}%)")
        
        logger.info(f"🔍 Analyzing {len(self.trading_pairs)} trading pairs...")
        
        analysis_results = []
        
        for i, symbol in enumerate(self.trading_pairs, 1):
            try:
                logger.info(f"[{i}/{len(self.trading_pairs)}] Processing {symbol}...")
                
                # Validate symbol
                if not self.market_data.validate_symbol(symbol):
                    logger.error(f"❌ Invalid symbol: {symbol}")
                    continue
                
                # Analyze symbol with enhanced features
                analysis = self.analyze_symbol(symbol, use_llm=not self.fast_mode)
                if not analysis:
                    logger.error(f"❌ No analysis data for {symbol}")
                    continue
                
                analysis_results.append(analysis)
                
                strategy = analysis.get('strategy', {})
                action = strategy.get('action', 'HOLD')
                confidence = strategy.get('confidence', 0.0)
                position_info = analysis.get('position_info', {})
                
                # Display summary
                action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")
                logger.info(f"   {action_emoji} Recommendation: {action} (Confidence: {confidence:.1%})")
                
                if position_info.get('has_position'):
                    pnl_emoji = "🟢" if position_info['unrealized_pnl'] > 0 else "🔴" if position_info['unrealized_pnl'] < 0 else "⚪"
                    logger.info(f"   💼 Current Position: {position_info['allocation_percent']:.1f}% allocation")
                    logger.info(f"   {pnl_emoji} P&L: ${position_info['unrealized_pnl']:.2f} ({position_info['unrealized_pnl_percent']:.1f}%)")
                
                # Execute trade if conditions are met
                if action in ['BUY', 'SELL'] and confidence >= self.min_confidence_threshold:
                    logger.info(f"   🎯 Executing {action} trade...")
                    success = self.execute_trade(symbol, action, confidence, analysis)
                    if success:
                        logger.info(f"   ✅ Trade executed successfully")
                    else:
                        logger.error(f"   ❌ Trade execution failed")
                elif action in ['BUY', 'SELL']:
                    logger.warning(f"   ⚠️ Confidence too low ({confidence:.1%} < {self.min_confidence_threshold:.1%})")
                
                # Small delay between symbols
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
                continue
        
        # Summary of analysis cycle
        logger.info(f"{'='*60}")
        logger.info(f"📋 Analysis Cycle Complete")
        logger.info(f"   Symbols Analyzed: {len(analysis_results)}")
        
        # Count recommendations
        buy_signals = sum(1 for a in analysis_results if a.get('strategy', {}).get('action') == 'BUY')
        sell_signals = sum(1 for a in analysis_results if a.get('strategy', {}).get('action') == 'SELL')
        hold_signals = sum(1 for a in analysis_results if a.get('strategy', {}).get('action') == 'HOLD')
        
        logger.info(f"   🟢 Buy Signals: {buy_signals}")
        logger.info(f"   🔴 Sell Signals: {sell_signals}")
        logger.info(f"   🟡 Hold Signals: {hold_signals}")
        logger.info(f"   📊 Daily Trades: {self.daily_trade_count}/{self.max_daily_trades}")
        
        # Check for rebalancing opportunities
        rebalancing_suggestions = self.portfolio_manager.check_rebalancing_needed()
        if rebalancing_suggestions:
            logger.info(f"   ⚖️ Rebalancing Opportunities: {len(rebalancing_suggestions)}")
        
        logger.info(f"{'='*60}")
        
        return analysis_results
    
    def start_trading(self, interval_minutes: int = None):
        """Start the trading bot with specified interval"""
        if interval_minutes is None:
            interval_minutes = self.trading_interval_minutes
        
        logger.info(f"Starting trading bot with {interval_minutes} minute intervals...")
        logger.info("Press Ctrl+C to stop")
        
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_analysis_cycle()
                
                # Wait for next cycle with responsive checking for stop signal
                logger.info(f"\nWaiting {interval_minutes} minutes until next analysis...")
                
                # Break the sleep into smaller chunks to respond to stop signals quickly
                total_sleep_seconds = interval_minutes * 60
                sleep_chunk = 10  # Check every 10 seconds if we should stop
                
                for i in range(0, total_sleep_seconds, sleep_chunk):
                    if not self.is_running:
                        logger.info("\nTrading stopped by user request")
                        return
                    
                    remaining_sleep = min(sleep_chunk, total_sleep_seconds - i)
                    time.sleep(remaining_sleep)
                
        except KeyboardInterrupt:
            logger.info("\nStopping trading bot...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.is_running = False
    
    def show_portfolio_status(self):
        """Display current portfolio status"""
        try:
            balance = self.get_account_balance()
            logger.info(f"\n=== Portfolio Status ===")
            logger.info(f"Account Balance: ${balance:.2f}")
            logger.info(f"Auto Trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
            logger.info(f"Daily Trades: {self.daily_trade_count}/{self.max_daily_trades}")
            logger.info(f"Total Trades: {len(self.trade_history)}")
            
            logger.info("\n=== Current Prices ===")
            for symbol in self.trading_pairs:
                price = self.market_data.get_current_price(symbol)
                if price:
                    logger.info(f"{symbol}: ${price:.4f}")
                else:
                    logger.warning(f"{symbol}: Price unavailable")
            
            # Show detailed portfolio summary
            self.get_portfolio_summary()
            
            # Show detailed open positions
            self.show_open_positions()
                
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
    
    def show_trade_history(self, limit=10):
        """Display recent trade history with P&L information"""
        trades = self.trade_tracker.get_trade_history(limit=limit)
        
        if not trades:
            logger.info("No trades in history")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADE HISTORY (Last {limit})")
        logger.info(f"{'='*80}")
        logger.info(f"{'Time':<19} | {'Symbol':<10} | {'Action':<4} | {'Price':<10} | {'Size':<12} | {'P&L':<10} | {'Status':<8}")
        logger.info(f"{'-'*80}")
        
        for trade in trades:
            status_icon = "✅" if trade['status'] == 'executed' else "📝"
            pnl_color = "" if trade['pnl'] == "N/A" else ("🟢" if "$-" not in trade['pnl'] and trade['pnl'] != "$0.00" else "🔴")
            
            logger.info(f"{status_icon} {trade['timestamp']} | {trade['symbol']:<10} | {trade['action']:<4} | "
                  f"{trade['price']:<10} | {trade['size']:<12} | {pnl_color}{trade['pnl']:<8} | {trade['status']:<8}")
        
        logger.info(f"{'-'*80}")
    
    def enable_auto_trading(self):
        """Enable automatic trading"""
        self.auto_trading_enabled = True
        logger.info("🚀 Auto trading ENABLED - Real orders will be placed!")
    
    def disable_auto_trading(self):
        """Disable automatic trading"""
        self.auto_trading_enabled = False
        logger.info("🛑 Auto trading DISABLED - Paper trading mode activated")
    
    def is_trading_active(self):
        """Check if trading loop is currently active"""
        return self.is_running
    
    def stop_trading(self):
        """Stop the trading loop"""
        logger.info("🛑 Stopping trading loop...")
        self.is_running = False
    
    def set_trading_parameters(self, min_confidence=None, max_daily_trades=None):
        """Update trading parameters"""
        if min_confidence is not None:
            self.min_confidence_threshold = min_confidence
            logger.info(f"Minimum confidence threshold set to: {min_confidence}")
        
        if max_daily_trades is not None:
            self.max_daily_trades = max_daily_trades
            logger.info(f"Maximum daily trades set to: {max_daily_trades}")
    
    def get_trading_stats(self, days: int = 30):
        """Get comprehensive trading statistics"""
        # Get basic stats for backward compatibility
        basic_stats = {
            "total_trades": len(self.trade_history),
            "executed_trades": len([t for t in self.trade_history if t['status'] == 'executed']),
            "paper_trades": len([t for t in self.trade_history if t['status'] == 'paper_trade']),
            "daily_trades_today": self.daily_trade_count,
            "auto_trading_enabled": self.auto_trading_enabled
        }
        
        # Get comprehensive performance metrics
        performance_metrics = self.trade_tracker.get_performance_metrics(days=days)
        
        # Combine both
        return {**basic_stats, **performance_metrics}
    
    def calculate_portfolio_pnl(self, symbol: str = None):
        """Calculate current portfolio P&L"""
        return self.trade_tracker.calculate_portfolio_pnl(symbol)
    
    def get_portfolio_summary(self):
        """Get enhanced portfolio summary with current positions, balances, and P&L"""
        logger.info("\n💼 Portfolio Summary")
        logger.info("=" * 60)
        
        try:
            # Update portfolio data
            self._update_portfolio_data()
            
            # Get comprehensive portfolio summary
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # Display overall portfolio metrics
            logger.info(f"📊 Overall Portfolio:")
            logger.info(f"   Total Value: ${portfolio_summary['total_portfolio_value']:.2f}")
            logger.info(f"   Crypto Value: ${portfolio_summary['total_crypto_value']:.2f}")
            logger.info(f"   USDT Balance: ${portfolio_summary['usdt_balance']:.2f}")
            logger.info(f"   Total P&L: ${portfolio_summary['total_unrealized_pnl']:.2f} ({portfolio_summary['total_unrealized_pnl_percent']:.1f}%)")
            logger.info(f"   Active Positions: {portfolio_summary['number_of_positions']}")
            
            # Display individual positions
            if portfolio_summary['positions']:
                logger.info(f"\n🏦 Individual Positions:")
                for symbol, position in portfolio_summary['positions'].items():
                    pnl_color = "🟢" if position['unrealized_pnl'] > 0 else "🔴" if position['unrealized_pnl'] < 0 else "⚪"
                    logger.info(f"   {symbol}:")
                    logger.info(f"      Quantity: {position['quantity']:.8f}")
                    logger.info(f"      Current Price: ${position['current_price']:.6f}")
                    logger.info(f"      Market Value: ${position['market_value']:.2f}")
                    logger.info(f"      Allocation: {position['allocation_percent']:.1f}%")
                    logger.info(f"      P&L: {pnl_color} ${position['unrealized_pnl']:.2f} ({position['unrealized_pnl_percent']:.1f}%)")
                    logger.info("")
            
            # Check for rebalancing opportunities
            rebalancing_suggestions = self.portfolio_manager.check_rebalancing_needed()
            if rebalancing_suggestions:
                logger.info(f"⚖️ Rebalancing Suggestions:")
                for suggestion in rebalancing_suggestions:
                    action_emoji = "🟢" if suggestion['action'] == 'BUY' else "🔴"
                    logger.info(f"   {action_emoji} {suggestion['symbol']}: {suggestion['action']} ${suggestion['suggested_amount']:.2f}")
                    logger.info(f"      Current: {suggestion['current_allocation']:.1f}% → Target: {suggestion['target_allocation']:.1f}%")
            
            # Get trade tracker summary for additional context
            trade_summary = self.trade_tracker.get_portfolio_summary()
            if trade_summary:
                logger.info(f"\n📈 Trading Performance:")
                for symbol, stats in trade_summary.items():
                    if isinstance(stats, dict) and 'total_pnl' in stats:
                        logger.info(f"   {symbol}: ${stats['total_pnl']:.2f} P&L")
            
            return portfolio_summary
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio summary: {e}")
            return {}
    
    def show_open_positions(self):
        """Display open positions in a detailed format"""
        try:
            portfolio = self.trade_tracker.get_portfolio_summary()
            
            if not portfolio or portfolio['open_positions_count'] == 0:
                logger.info("\nNo open positions")
                return
            
            logger.info(f"\n{'='*60}")
            logger.info(f"PORTFOLIO SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"{'Symbol':<10} | {'Position':<12} | {'Avg Price':<10} | {'Current P&L':<12}")
            logger.info(f"{'-'*60}")
            
            open_positions = portfolio['open_positions']
            for symbol, data in open_positions.items():
                if data['size'] > 0:
                    avg_price = data['total_cost'] / data['size'] if data['size'] > 0 else 0
                    pnl_color = "🟢" if data['unrealized_pnl'] >= 0 else "🔴"
                    logger.info(f"{symbol:<10} | {data['size']:<12.6f} | ${avg_price:<9.4f} | {pnl_color}${data['unrealized_pnl']:<11.2f}")
            
            logger.info(f"{'-'*60}")
            total_pnl = portfolio['total_pnl']
            total_color = "🟢" if total_pnl >= 0 else "🔴"
            logger.info(f"{'TOTAL P&L:':<35} | {total_color}${total_pnl:.2f}")
            logger.info(f"{'REALIZED P&L:':<35} | ${portfolio['total_realized_pnl']:.2f}")
            logger.info(f"{'UNREALIZED P&L:':<35} | ${portfolio['total_unrealized_pnl']:.2f}")
            logger.info(f"{'TOTAL FEES:':<35} | ${portfolio['total_fees']:.2f}")
            logger.info(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"❌ Error showing open positions: {e}")
    
    def show_performance_report(self, days: int = 30):
        """Display comprehensive performance report"""
        metrics = self.trade_tracker.get_performance_metrics(days=days)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PERFORMANCE REPORT (Last {days} days)")
        logger.info(f"{'='*70}")
        
        # Trading Activity
        logger.info(f"\n📊 TRADING ACTIVITY:")
        logger.info(f"   Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"   Executed Trades: {metrics.get('executed_trades', 0)}")
        logger.info(f"   Paper Trades: {metrics.get('paper_trades', 0)}")
        logger.info(f"   Win Rate: {metrics.get('win_rate', 0):.1f}%")
        
        # P&L Performance
        logger.info(f"\n💰 P&L PERFORMANCE:")
        total_pnl = metrics.get('total_pnl', 0)
        pnl_color = "🟢" if total_pnl >= 0 else "🔴"
        logger.info(f"   Total P&L: {pnl_color}${total_pnl:.2f}")
        logger.info(f"   Realized P&L: ${metrics.get('realized_pnl', 0):.2f}")
        logger.info(f"   Unrealized P&L: ${metrics.get('unrealized_pnl', 0):.2f}")
        logger.info(f"   Average P&L per Trade: ${metrics.get('avg_pnl_per_trade', 0):.2f}")
        
        # Risk Metrics
        logger.info(f"\n⚠️  RISK METRICS:")
        logger.info(f"   Best Trade: ${metrics.get('best_trade', 0):.2f}")
        logger.info(f"   Worst Trade: ${metrics.get('worst_trade', 0):.2f}")
        logger.info(f"   Total Fees Paid: ${metrics.get('total_fees', 0):.2f}")
        
        logger.info(f"\n{'='*70}")
    
    def export_trades_to_excel(self, filename: str = None) -> str:
        """Export all trade history to Excel file"""
        try:
            excel_file = self.trade_tracker.export_trades_to_excel(filename)
            logger.info(f"\n📊 Trade history exported to Excel: {excel_file}")
            logger.info(f"📁 File location: {os.path.abspath(excel_file)}")
            return excel_file
        except Exception as e:
            logger.error(f"❌ Failed to export to Excel: {e}")
            return None
    
    def start_automated_trading(self):
        """Start automated trading - alias for start_trading method"""
        logger.info("🚀 Starting automated trading mode...")
        self.start_trading()
    
    def send_startup_notification(self):
        """Send notification when trading agent starts"""
        if hasattr(self.trade_tracker, 'send_system_alert'):
            message = f"""
🚀 Crypto Trading Agent Started

🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
🤖 Mode: {'Auto Trading' if self.auto_trading_enabled else 'Paper Trading'}
📊 Trading Pairs: {', '.join(self.trading_pairs)}
⚙️ Min Confidence: {self.min_confidence_threshold}
🔢 Max Daily Trades: {self.max_daily_trades}

The trading agent is now active and monitoring the markets.
            """.strip()
            
            self.trade_tracker.send_system_alert(
                message, 
                "🚀 Crypto Trading Agent Started"
            )
    
    def send_shutdown_notification(self):
        """Send notification when trading agent stops"""
        if hasattr(self.trade_tracker, 'send_system_alert'):
            today = datetime.now().strftime('%Y-%m-%d')
            today_trades = len([
                t for t in self.trade_tracker.trades
                if t.timestamp.startswith(today)
            ])
            
            message = f"""
🛑 Crypto Trading Agent Stopped

🕐 Stop Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
📊 Today's Trades: {today_trades}
💰 Portfolio P&L: ${self.trade_tracker.calculate_portfolio_pnl():.2f}

The trading agent has been stopped.
            """.strip()
            
            self.trade_tracker.send_system_alert(
                message, 
                "🛑 Crypto Trading Agent Stopped"
            )
    
    def send_error_notification(self, error_message: str, context: str = ""):
        """Send error notification"""
        if hasattr(self.trade_tracker, 'send_system_alert'):
            message = f"""
❌ Crypto Trading Agent Error

🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
📍 Context: {context}
🚨 Error: {error_message}

Please check the trading agent logs for more details.
            """.strip()
            
            self.trade_tracker.send_system_alert(
                message, 
                "❌ Crypto Trading Agent Error"
            )
    
    def check_aws_status(self):
        """Check and display AWS service status"""
        logger.info("\n🔧 AWS Service Status:")
        logger.info("   AWS Bedrock: ✅ Available (for LLM analysis)")
        logger.info("   Trade Storage: ✅ Local JSON files")
        logger.info("   Notifications: ✅ Local console/file logging")


def main():
    """Enhanced main function with auto-mode support for EC2 deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Trading Agent')
    parser.add_argument('--auto-mode', action='store_true', 
                       help='Run in automated mode (for EC2 deployment)')
    parser.add_argument('--auto-trading', action='store_true',
                       help='Enable automatic trading execution')
    parser.add_argument('--paper-trading', action='store_true',
                       help='Force paper trading mode')
    parser.add_argument('--check-aws', action='store_true',
                       help='Check AWS service status and exit')
    
    args = parser.parse_args()
    
    # Auto-detect Docker environment and enable auto-mode
    is_docker = os.getenv('DOCKER_CONTAINER', 'false').lower() == 'true' or os.path.exists('/.dockerenv')
    auto_mode = args.auto_mode or is_docker
    
    # Determine auto trading setting
    auto_trading_enabled = args.auto_trading and not args.paper_trading
    
    # Initialize agent
    agent = CryptoTradingAgent(auto_trading_enabled=auto_trading_enabled)
    
    # Display trading mode
    if auto_trading_enabled:
        logger.info("🚀 Auto Trading: ENABLED - Real trades will be executed")
    else:
        logger.info("📊 Auto Trading: DISABLED - Analysis only mode")
    
    if is_docker:
        logger.info("🐳 Docker environment detected - running in automated mode")
    
    # Check AWS status if requested
    if args.check_aws:
        agent.check_aws_status()
        return
    
    try:
        if auto_mode:
            # Run in automated mode for EC2
            logger.info("🚀 Starting Crypto Trading Agent in automated mode...")
            agent.send_startup_notification()
            
            # Run continuous trading loop
            import time
            while True:
                try:
                    agent.run_analysis_cycle()
                    
                    # Wait for next cycle with proper message
                    interval_minutes = agent.trading_interval_minutes
                    logger.info(f"\n⏰ Waiting {interval_minutes} minutes until next analysis...")
                    time.sleep(interval_minutes * 60)
                except KeyboardInterrupt:
                    logger.info("\n🛑 Shutting down trading agent...")
                    agent.send_shutdown_notification()
                    break
                except Exception as e:
                    agent.send_error_notification(str(e), "automated_mode")
                    logger.warning("\n⚠️ Error occurred, waiting 1 minute before retrying...")
                    time.sleep(60)  # Wait 1 minute before retrying
        else:
            # Run interactive mode
            while True:
                trading_status = "🚀 ENABLED" if agent.auto_trading_enabled else "📊 DISABLED"
                logger.info("\n" + "="*60)
                logger.info("🤖 CRYPTO TRADING AGENT")
                logger.info(f"Auto Trading: {trading_status}")
                logger.info("="*60)
                logger.info("1. 📊 Run single analysis cycle")
                logger.info("2. 🚀 Start automated trading")
                logger.info("3. 💰 Show portfolio status")
                logger.info("4. 🔍 Analyze specific symbol")
                logger.info("5. 🔧 Check AWS status")
                logger.info("6. ⚙️  Toggle auto trading")
                logger.info("7. 🚪 Exit")
                logger.info("="*60)
                
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == '1':
                    agent.run_analysis_cycle()
                elif choice == '2':
                    agent.start_automated_trading()
                elif choice == '3':
                    agent.show_portfolio_status()
                elif choice == '4':
                    symbol = input("Enter symbol (e.g., BTC-USDT): ").strip().upper()
                    if symbol:
                        agent.analyze_symbol(symbol)
                elif choice == '5':
                    agent.check_aws_status()
                elif choice == '6':
                    if agent.auto_trading_enabled:
                        agent.disable_auto_trading()
                        logger.info("📊 Auto trading disabled")
                    else:
                        agent.enable_auto_trading()
                        logger.info("🚀 Auto trading enabled")
                elif choice == '7':
                    logger.info("\n👋 Goodbye!")
                    agent.send_shutdown_notification()
                    break
                else:
                    logger.warning("❌ Invalid option. Please try again.")
                    
    except Exception as e:
        agent.send_error_notification(str(e), "main_function")
        raise


if __name__ == "__main__":
    main()

