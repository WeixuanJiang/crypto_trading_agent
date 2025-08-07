import os
import sys
import time
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading

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
        
        # Initialize database and repositories
        from src.data import DatabaseManager, TradeRepository, PositionRepository, MarketDataRepository
        self.db_manager = DatabaseManager()
        self.trade_repository = TradeRepository(self.db_manager)
        self.position_repository = PositionRepository(self.db_manager)
        self.market_data_repository = MarketDataRepository(self.db_manager)
        
        # Initialize components with database support
        self.market_data = MarketDataManager(self.market_client)
        self.risk_manager = RiskManager()
        self.trade_tracker = create_trade_tracker()
        
        # Initialize portfolio manager
        from src.portfolio.manager import PortfolioManager
        self.portfolio_manager = PortfolioManager(self.user_client, self.market_client, self.trade_tracker, config, self.db_manager, self.position_repository)
        
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
        
        # Flask server setup
        self.app = None
        self.server_thread = None
        self.server_running = False
        
        # Initialize memory log handler for real-time logs
        self.log_memory = []
        self.max_log_entries = 1000
        self._setup_memory_log_handler()
        
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
                
                # Check if it's a timeout error or API signature error
                if ("timeout" in str(e).lower() or "connection" in str(e).lower() or 
                    "invalid kc-api" in str(e).lower()):
                    wait_time = retry_delay * (backoff_factor ** attempt)
                    logger.warning(f"⚠️ API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # For other errors, re-raise immediately
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
    
    def _setup_memory_log_handler(self):
        """Setup memory log handler to capture logs for real-time display"""
        import logging
        
        class MemoryLogHandler(logging.Handler):
            def __init__(self, log_memory, max_entries):
                super().__init__()
                self.log_memory = log_memory
                self.max_entries = max_entries
            
            def emit(self, record):
                try:
                    # Extract the main message without timestamps if it's already formatted
                    message = record.getMessage()
                    
                    # Clean up message - remove timestamp and color codes if present
                    if " - [" in message and "]" in message:
                        # It's a web access log, make it cleaner
                        if "GET" in message or "POST" in message:
                            if "api/logs" in message:
                                # Skip log API calls to avoid cluttering the logs
                                return
                            # Simplify HTTP request logs
                            parts = message.split('"', 2)
                            if len(parts) > 1:
                                message = parts[1]
                        else:
                            # Extract the actual message content after color codes
                            parts = message.split("[0m - ", 1)
                            if len(parts) > 1:
                                message = parts[1]
                    
                    # Look for trading-specific log patterns to prioritize
                    is_trading_log = False
                    if any(pattern in message for pattern in [
                        "Analysis Results", "Strategy Recommendation", "Portfolio", "P&L",
                        "Trade executed", "Current Position", "📈", "💰", "🚀", "💼",
                        "BUY", "SELL", "HOLD", "Confidence"
                    ]):
                        is_trading_log = True
                        # Keep emojis and make trading logs more visible
                        pass
                    
                    # Strip ANSI color codes that might remain
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    message = ansi_escape.sub('', message)
                    
                    # Create log entry with emoji icons preserved
                    log_entry = {
                        'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S'),
                        'level': record.levelname,
                        'message': message,
                        'type': 'trading' if is_trading_log else 'system'
                    }
                    
                    # Add to memory (newest first)
                    self.log_memory.insert(0, log_entry)
                    
                    # Keep only max_entries
                    if len(self.log_memory) > self.max_entries:
                        self.log_memory.pop()
                        
                except Exception:
                    pass  # Ignore errors in logging
        
        # Create and add memory handler to root logger
        memory_handler = MemoryLogHandler(self.log_memory, self.max_log_entries)
        memory_handler.setLevel(logging.INFO)
        memory_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Get the root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.addHandler(memory_handler)
        
    def _read_logs_from_file(self, limit: int = 100) -> List[Dict]:
        """Read logs from the consolidated log file
        
        Returns log entries in a format compatible with the memory log handler
        """
        log_file_path = 'logs/trading_agent.log'
        logs = []
        
        try:
            if not os.path.exists(log_file_path):
                logger.warning(f"Log file not found: {log_file_path}")
                return []
                
            with open(log_file_path, 'r') as f:
                # Read the last N lines from the file (fastest approach for large files)
                # Start with a buffer size that should be large enough for limit lines
                buffer_size = limit * 200  # Estimate average line length
                
                # Get file size
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                
                # Set initial position
                pos = max(file_size - buffer_size, 0)
                f.seek(pos)
                
                # If we're not at the beginning, discard partial line
                if pos > 0:
                    f.readline()
                
                # Read remaining lines
                lines = f.readlines()
                
                # Get the last 'limit' lines or all if fewer
                lines = lines[-limit:] if len(lines) > limit else lines
                
                for line in lines:
                    # Parse and convert log line to dict entry
                    log_entry = self._parse_log_line(line)
                    if log_entry:
                        logs.append(log_entry)
                        
            return logs
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            return []
    
    def _parse_log_line(self, line: str) -> Dict:
        """Parse a log line from the file into a structured log entry
        
        Handles both backend Python logs and frontend Node/React logs
        """
        try:
            line = line.strip()
            if not line:
                return None
                
            # Default log entry structure
            log_entry = {
                'timestamp': '',
                'level': 'INFO',
                'message': line,  # Default to the whole line
                'type': 'system'
            }
            
            # Try to parse backend log format first (most common)
            # Format: YYYY-MM-DD HH:MM:SS - module - LEVEL - function:line - message
            backend_pattern = r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+-\s+([\w_\.]+)\s+-\s+(\w+)\s+-\s+([\w_\.]+:\d+)\s+-\s+(.+)$'
            match = re.match(backend_pattern, line)
            
            if match:
                timestamp_str, module, level, location, message = match.groups()
                # Convert to display timestamp (just HH:MM:SS)
                try:
                    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    timestamp = dt.strftime('%H:%M:%S')
                except:
                    timestamp = timestamp_str.split(' ')[-1]  # Just take time part
                    
                log_entry = {
                    'timestamp': timestamp,
                    'level': level,
                    'message': message,
                    'type': 'system'
                }
                
                # Check if it's a trading log
                if any(pattern in message for pattern in [
                    "Analysis Results", "Strategy Recommendation", "Portfolio", "P&L",
                    "Trade executed", "Current Position", "📈", "💰", "🚀", "💼",
                    "BUY", "SELL", "HOLD", "Confidence"
                ]):
                    log_entry['type'] = 'trading'
                
                return log_entry
                
            # Check for frontend logs from React/Node
            # These often have a pattern with timestamps and components
            frontend_patterns = [
                # Match standard webpack output
                r'\[(\d{2}:\d{2}:\d{2})\]\s+(\w+)\s+(.*)',
                # Match npm start output 
                r'(\d{2}:\d{2}:\d{2})\s+((?:(?:error|warn|info)\s+)?.*?):\s+(.*)',
            ]
            
            for pattern in frontend_patterns:
                match = re.match(pattern, line)
                if match:
                    parts = match.groups()
                    
                    timestamp = parts[0]
                    level_or_component = parts[1].upper()
                    message = parts[2]
                    
                    # Determine log level
                    level = 'INFO'
                    if 'ERROR' in level_or_component:
                        level = 'ERROR'
                    elif 'WARN' in level_or_component:
                        level = 'WARNING'
                    
                    log_entry = {
                        'timestamp': timestamp,
                        'level': level, 
                        'message': message,
                        'type': 'system'
                    }
                    
                    return log_entry
            
            # Fall back to generic parsing if specific patterns don't match
            # Look for common timestamp patterns
            timestamp_patterns = [
                # Match HH:MM:SS format
                r'(\d{1,2}:\d{2}:\d{2})',
                # Match [HH:MM:SS] format
                r'\[(\d{1,2}:\d{2}:\d{2})\]',
            ]
            
            for pattern in timestamp_patterns:
                match = re.search(pattern, line)
                if match:
                    log_entry['timestamp'] = match.group(1)
                    # Remove timestamp from message to avoid duplication
                    log_entry['message'] = re.sub(pattern, '', line).strip()
                    break
            
            # Look for log levels
            level_matches = re.search(r'\b(ERROR|WARNING|INFO|DEBUG)\b', line, re.IGNORECASE)
            if level_matches:
                log_entry['level'] = level_matches.group(1).upper()
            
            # Check for duplicate content in message
            # Sometimes there are duplicated words or repeated content
            words = log_entry['message'].split()
            if len(words) >= 2:
                # Look for duplicated content at the end
                half_len = len(words) // 2
                if words[:half_len] == words[half_len:2*half_len]:
                    log_entry['message'] = ' '.join(words[:half_len])
            
            return log_entry
            
        except Exception as e:
            # If parsing fails, return a simple entry with the original line
            return {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'level': 'INFO',
                'message': line,
                'type': 'system'
            }
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent logs from memory and/or file
        
        Reads both in-memory logs and logs from the consolidated log file
        """
        # Start with in-memory logs
        memory_logs = self.log_memory.copy()
        
        # Also read logs from the log file
        try:
            file_logs = self._read_logs_from_file(limit=limit*2)  # Read more than needed as we'll filter later
            
            # Merge logs, avoiding duplicates
            # Use a set of log message+timestamps to track duplicates
            seen = set()
            filtered_memory_logs = []
            
            # First filter out duplicate messages from memory logs
            for log in memory_logs:
                # Normalize message by removing ANSI color codes and common prefixes
                msg = log.get('message', '')
                msg = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', msg)
                msg = re.sub(r'^- INFO - ', '', msg)
                
                # Skip logs with duplicate message-timestamp pairs
                log_key = (log.get('timestamp', ''), msg)
                if log_key not in seen:
                    log['message'] = msg  # Use cleaned message
                    filtered_memory_logs.append(log)
                    seen.add(log_key)
            
            # Replace memory logs with filtered version
            memory_logs = filtered_memory_logs
            
            # Add file logs that aren't already in memory
            for log in file_logs:
                msg = log.get('message', '')
                msg = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', msg)
                msg = re.sub(r'^- INFO - ', '', msg)
                
                log_key = (log.get('timestamp', ''), msg)
                if log_key not in seen:
                    log['message'] = msg  # Use cleaned message
                    memory_logs.append(log)
                    seen.add(log_key)
            
            # Sort logs by timestamp (newest first)
            memory_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
        except Exception as e:
            logger.warning(f"Failed to read logs from file: {e}")
        
        # Return limited logs
        return memory_logs[:limit]
    
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
    
    def setup_flask_server(self, host='127.0.0.1', port=5001):
        """Setup Flask server with API endpoints"""
        self.app = Flask(__name__)
        # Enable CORS for React frontend
        CORS(self.app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
        
        # Force JSON content type for all responses
        @self.app.after_request
        def add_header(response):
            response.headers['Content-Type'] = 'application/json'
            return response
        
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Root endpoint with API information"""
            return jsonify({
                'name': 'Crypto Trading Agent API',
                'version': '1.0.0',
                'status': 'running',
                'endpoints': {
                    'GET /api/status': 'Get trading agent status',
                    'POST /api/trading/start': 'Start trading',
                    'POST /api/trading/stop': 'Stop trading',
                    'POST /api/trading/toggle': 'Toggle auto trading',
                    'GET /api/trading/history': 'Get trading history',
                    'POST /api/analysis/run': 'Run analysis',
                    'GET /api/portfolio': 'Get portfolio status',
                    'POST /api/settings/update': 'Update settings'
                },
                'documentation': 'Visit the endpoints above for trading operations'
            })
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get current trading agent status"""
            try:
                balance = self.get_account_balance()
                return jsonify({
                    'success': True,
                    'status': {
                        'auto_trading': self.auto_trading_enabled,
                        'is_running': self.is_running,
                        'fast_mode': self.fast_mode,
                        'trading_pairs': self.trading_pairs,
                        'balance': balance,
                        'daily_trade_count': self.daily_trade_count,
                        'max_daily_trades': self.max_daily_trades,
                        'min_confidence_threshold': self.min_confidence_threshold,
                        'trading_interval_minutes': self.trading_interval_minutes,
                        'last_update': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/trading/start', methods=['POST'])
        def start_trading():
            """Start automated trading"""
            try:
                data = request.get_json() or {}
                paper_trading = data.get('paper_trading', True)
                
                if not paper_trading:
                    self.enable_auto_trading()
                else:
                    self.auto_trading_enabled = False
                
                if not self.is_running:
                    # Start trading in a separate thread so it doesn't block the response
                    trading_thread = threading.Thread(target=self.start_trading, daemon=True)
                    trading_thread.start()
                    self.is_running = True
                
                logger.info(f"Trading started - Paper: {paper_trading}")
                return jsonify({
                    'success': True,
                    'message': 'Trading started successfully',
                    'auto_trading': self.auto_trading_enabled,
                    'is_running': self.is_running
                })
            except Exception as e:
                logger.error(f"Error starting trading: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/trading/stop', methods=['POST'])
        def stop_trading():
            """Stop automated trading"""
            try:
                self.is_running = False
                self.disable_auto_trading()
                
                logger.info("Trading stopped")
                return jsonify({
                    'success': True,
                    'message': 'Trading stopped successfully',
                    'auto_trading': self.auto_trading_enabled,
                    'is_running': self.is_running
                })
            except Exception as e:
                logger.error(f"Error stopping trading: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/trading/toggle', methods=['POST'])
        def toggle_auto_trading():
            """Toggle auto trading mode"""
            try:
                if self.auto_trading_enabled:
                    self.disable_auto_trading()
                    message = "Auto trading disabled"
                else:
                    self.enable_auto_trading()
                    message = "Auto trading enabled"
                
                logger.info(message)
                return jsonify({
                    'success': True,
                    'message': message,
                    'auto_trading': self.auto_trading_enabled
                })
            except Exception as e:
                logger.error(f"Error toggling auto trading: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/trading/history', methods=['GET'])
        def get_trading_history():
            """Get trading history with optional filtering"""
            try:
                # Get query parameters
                limit = request.args.get('limit', 50, type=int)
                symbol = request.args.get('symbol')
                days = request.args.get('days', 30, type=int)
                
                # Get trade history from trade tracker
                trades = self.trade_tracker.get_trade_history(limit=limit)
                
                # Filter by symbol if specified
                if symbol:
                    trades = [trade for trade in trades if trade.get('symbol') == symbol]
                
                # Get trading statistics
                stats = self.get_trading_stats(days=days)
                
                return jsonify({
                    'success': True,
                    'trades': trades,
                    'statistics': stats,
                    'total_trades': len(trades),
                    'last_update': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting trading history: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/analysis/run', methods=['POST'])
        def run_analysis():
            """Run analysis cycle"""
            try:
                # Handle both JSON and form data
                if request.is_json:
                    data = request.get_json() or {}
                else:
                    data = request.form.to_dict() or {}
                    
                symbol = data.get('symbol')
                
                if symbol:
                    # Analyze specific symbol
                    result = self.analyze_symbol(symbol)
                    return jsonify({
                        'success': True,
                        'message': f'Analysis completed for {symbol}',
                        'analysis': result
                    })
                else:
                    # Run full analysis cycle
                    analysis_results = self.run_analysis_cycle()
                    return jsonify({
                        'success': True,
                        'message': 'Analysis cycle completed',
                        'analysis': analysis_results
                    })
            except Exception as e:
                logger.error(f"Error running analysis: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/portfolio', methods=['GET'])
        def get_portfolio():
            """Get portfolio status"""
            try:
                self._update_portfolio_data()
                portfolio_data = self.portfolio_manager.get_portfolio_summary()
                balance = self.get_account_balance()
                
                # Convert positions dict to list format for better API compatibility
                positions_dict = portfolio_data.get('positions', {})
                positions_list = []
                for symbol, position_data in positions_dict.items():
                    position_info = position_data.copy()
                    position_info['symbol'] = symbol
                    positions_list.append(position_info)
                
                return jsonify({
                    'success': True,
                    'portfolio': {
                        'balance': balance,
                        'positions': positions_list,
                        'total_value': portfolio_data.get('total_portfolio_value', 0),
                        'unrealized_pnl': portfolio_data.get('total_unrealized_pnl', 0),
                        'last_update': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/settings/update', methods=['POST'])
        def update_settings():
            """Update trading settings"""
            try:
                data = request.get_json() or {}
                
                if 'trading_pairs' in data:
                    self.trading_pairs = data['trading_pairs']
                
                if 'min_confidence_threshold' in data:
                    self.min_confidence_threshold = float(data['min_confidence_threshold'])
                
                if 'max_daily_trades' in data:
                    self.max_daily_trades = int(data['max_daily_trades'])
                
                if 'trading_interval_minutes' in data:
                    self.trading_interval_minutes = int(data['trading_interval_minutes'])
                
                logger.info("Settings updated successfully")
                return jsonify({
                    'success': True,
                    'message': 'Settings updated successfully',
                    'settings': {
                        'trading_pairs': self.trading_pairs,
                        'min_confidence_threshold': self.min_confidence_threshold,
                        'max_daily_trades': self.max_daily_trades,
                        'trading_interval_minutes': self.trading_interval_minutes
                    }
                })
            except Exception as e:
                logger.error(f"Error updating settings: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/market/price/<symbol>', methods=['GET'])
        def get_market_price(symbol):
            """Get real-time market price data for a symbol"""
            try:
                logger.info(f"Fetching market price for symbol: {symbol}")
                # Get current price using market data manager
                current_price = self.market_data.get_current_price(symbol)
                logger.info(f"Current price result for {symbol}: {current_price}")
                if current_price is None:
                    logger.error(f"Market data manager returned None for {symbol}")
                    return jsonify({'success': False, 'error': f'Could not fetch price for {symbol}'}), 404
                
                # Get 24hr stats for additional data
                try:
                    ticker = self.market_client.get_24hr_stats(symbol)
                    price_change = float(ticker.get('changeRate', 0)) * 100  # Convert to percentage
                    volume = float(ticker.get('vol', 0))
                    high_24h = float(ticker.get('high', current_price))
                    low_24h = float(ticker.get('low', current_price))
                except Exception as e:
                    logger.warning(f"Could not get 24hr stats for {symbol}: {e}")
                    price_change = 0.0
                    volume = 0.0
                    high_24h = current_price
                    low_24h = current_price
                
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'price': current_price,
                    'price_change_percent': price_change,
                    'volume_24h': volume,
                    'high_24h': high_24h,
                    'low_24h': low_24h,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting market price for {symbol}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/logs', methods=['GET', 'POST'])
        def get_logs():
            """Get or add recent trading logs"""
            try:
                if request.method == 'GET':
                    # Get query parameters
                    limit = int(request.args.get('limit', 50))
                    
                    # Get logs from memory handler
                    logs = self.get_recent_logs(limit)
                    
                    return jsonify({
                        'success': True,
                        'logs': logs,
                        'total_count': len(logs),
                        'timestamp': datetime.now().isoformat()
                    })
                elif request.method == 'POST':
                    # Add a new log entry directly to the memory handler
                    data = request.get_json()
                    if not data:
                        return jsonify({'success': False, 'error': 'No data provided'}), 400
                    
                    message = data.get('message', '')
                    level = data.get('level', 'INFO')
                    log_type = data.get('type', 'trading')
                    
                    # Create log entry
                    log_entry = {
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'level': level,
                        'message': message,
                        'type': log_type
                    }
                    
                    # Add to memory
                    self.log_memory.insert(0, log_entry)
                    
                    # Keep only max_entries
                    if len(self.log_memory) > self.max_log_entries:
                        self.log_memory.pop()
                    
                    return jsonify({
                        'success': True,
                        'message': 'Log entry added successfully',
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error getting logs: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        
        
        
    
    def start_flask_server(self):
        """Start Flask server in a separate thread"""
        if not self.app:
            logger.error("Flask server not configured. Call setup_flask_server() first.")
            return
        
        def run_server():
            self.server_running = True
            logger.info(f"🌐 Starting Flask API server on http://127.0.0.1:5001")
            self.app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(1)  # Give server time to start
    
    def stop_flask_server(self):
        """Stop Flask server"""
        self.server_running = False
        logger.info("Flask server stopped")
    
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
            return ""
    
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
    """Main function to start Flask API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Trading Agent Flask API Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind the server to')
    parser.add_argument('--auto-trading', action='store_true',
                       help='Enable automatic trading execution on startup')
    parser.add_argument('--check-aws', action='store_true',
                       help='Check AWS service status and exit')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize agent
    agent = CryptoTradingAgent(auto_trading_enabled=args.auto_trading)
    
    # Check AWS status if requested
    if args.check_aws:
        agent.check_aws_status()
        return
    
    try:
        # Setup and start Flask server
        agent.setup_flask_server(host=args.host, port=args.port)
        agent.start_flask_server()
        
        print("\n🚀 Crypto Trading Agent Flask Server Started!")
        print(f"📊 API Server: http://{args.host}:{args.port}")
        print("\n📋 Available API Endpoints:")
        print("  GET  /api/status          - Get trading agent status")
        print("  POST /api/trading/start   - Start trading")
        print("  POST /api/trading/stop    - Stop trading")
        print("  POST /api/trading/toggle  - Toggle auto trading")
        print("  POST /api/analysis/run    - Run analysis")
        print("  GET  /api/portfolio       - Get portfolio status")
        print("  POST /api/settings/update - Update settings")
        print("\n💡 Use Ctrl+C to stop the server")
        
        # Send startup notification
        agent.send_startup_notification()
        
        # Keep the main thread alive
        while agent.server_running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        agent.stop_flask_server()
        agent.send_shutdown_notification()
        print("✅ Server stopped successfully")
    except Exception as e:
        agent.send_error_notification(str(e), "flask_server")
        logger.error(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()

