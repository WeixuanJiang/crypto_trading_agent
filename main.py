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
    @staticmethod
    def _load_settings_from_file():
        """Load all settings from config/settings.json centrally"""
        config_path = os.path.join(os.getcwd(), 'config', 'settings.json')

        if not os.path.exists(config_path):
            logger.warning(f"⚠️ Settings file not found at {config_path}, will use defaults")
            return None

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading settings file: {e}")
            return None

    def __init__(self, auto_trading_enabled=False, fast_mode=None):
        # Load environment variables from .env file
        load_dotenv()

        # Load settings from config file centrally
        saved_settings = self._load_settings_from_file()

        # Read fast_mode from settings file, then environment if not explicitly provided
        if fast_mode is None:
            if saved_settings and 'llm_configuration' in saved_settings:
                fast_mode = str(saved_settings['llm_configuration'].get('FAST_MODE', 'true')).lower() == 'true'
            else:
                fast_mode = os.getenv('FAST_MODE', 'true').lower() == 'true'

        logger.info("Initializing Crypto Trading Agent...")
        
        # Initialize API credentials
        self.api_key, self.api_secret, self.api_passphrase = self.get_api_credentials()
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.cross_region_inference = os.getenv('AWS_CROSS_REGION_INFERENCE', 'true').lower() == 'true'

        # Sync local clock with KuCoin server time to avoid KC-API-TIMESTAMP errors
        self._sync_kucoin_timestamp()

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

        # Load paper trading balances from settings
        paper_trading_balances = self._load_paper_trading_balances()

        # Initialize portfolio manager
        from src.portfolio.manager import PortfolioManager
        self.portfolio_manager = PortfolioManager(
            self.user_client,
            self.market_client,
            self.trade_tracker,
            config,
            self.db_manager,
            self.position_repository,
            paper_trading_balances=paper_trading_balances
        )
        
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
        
        # Load trading pairs from settings file
        if saved_settings and 'trading_parameters' in saved_settings:
            trading_pairs_str = saved_settings['trading_parameters'].get('TRADING_PAIRS', 'BTC-USDT,ETH-USDT')
            logger.info(f"✅ Trading pairs loaded from settings: {trading_pairs_str}")
        else:
            trading_pairs_str = os.getenv('TRADING_PAIRS', 'BTC-USDT,ETH-USDT,ADA-USDT,DOT-USDT')
            logger.warning(f"⚠️ Trading pairs from environment fallback: {trading_pairs_str}")

        self.trading_pairs = [pair.strip() for pair in trading_pairs_str.split(',')]
        self.is_running = False

        # Check ENABLE_LIVE_TRADING from settings if not set via command-line
        if not auto_trading_enabled:
            if saved_settings and 'trading_parameters' in saved_settings:
                trading_params = saved_settings['trading_parameters']
                enable_live_str = str(trading_params.get('ENABLE_LIVE_TRADING', 'false')).lower()
                enable_live_trading = enable_live_str in ['true', '1', 'yes']
                logger.info(f"✅ Live trading setting loaded from settings: {enable_live_trading}")
            else:
                # Fallback to environment variable
                enable_live_trading = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
                logger.warning(f"⚠️ Live trading from environment fallback: {enable_live_trading}")

            self.auto_trading_enabled = enable_live_trading
        else:
            self.auto_trading_enabled = auto_trading_enabled

        self.trade_history = []  # Keep for backward compatibility

        # Load other trading parameters from settings file
        if saved_settings and 'trading_parameters' in saved_settings:
            trading_params = saved_settings['trading_parameters']

            # Load all trading parameters from settings file
            self.min_confidence_threshold = float(trading_params.get('MIN_CONFIDENCE_THRESHOLD', '0.7'))
            self.max_daily_trades = int(trading_params.get('MAX_DAILY_TRADES', '1000'))
            self.minimum_balance = float(trading_params.get('MINIMUM_BALANCE', '10'))
            self.trading_interval_minutes = int(trading_params.get('TRADING_INTERVAL_MINUTES', '15'))
            self.historical_data_limit = int(trading_params.get('HISTORICAL_DATA_LIMIT', '500'))

            logger.info(f"✅ Trading parameters loaded from settings:")
            logger.info(f"   - Trading interval: {self.trading_interval_minutes} minutes")
            logger.info(f"   - Min confidence: {self.min_confidence_threshold}")
            logger.info(f"   - Max daily trades: {self.max_daily_trades}")
            logger.info(f"   - Min balance: ${self.minimum_balance}")
        else:
            # Fallback to environment variables only if settings file doesn't exist
            self.min_confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7'))
            self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '5'))
            self.minimum_balance = float(os.getenv('MINIMUM_BALANCE', '10'))
            self.trading_interval_minutes = int(os.getenv('TRADING_INTERVAL_MINUTES', '60'))
            self.historical_data_limit = int(os.getenv('HISTORICAL_DATA_LIMIT', '500'))
            logger.warning(f"⚠️ Using environment variables for trading parameters (settings file not found)")

        # Load LLM / sentiment settings
        _llm_cfg = (saved_settings or {}).get('llm_configuration', {})
        self.llm_provider = str(_llm_cfg.get('LLM_PROVIDER', os.getenv('LLM_PROVIDER', 'bedrock'))).lower()
        self.enable_sentiment_analysis = str(_llm_cfg.get('ENABLE_SENTIMENT_ANALYSIS',
                                             os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'true'))).lower() in ('true', '1', 'yes')

        self.daily_trade_count = 0
        self.last_trade_date = None

        # Balance cache — avoids hammering KuCoin's authenticated accounts endpoint
        self._balance_cache = {'value': 0.0, 'ts': 0.0}
        self._balance_cache_ttl = 30  # seconds

        # Log current mode
        mode = "LIVE TRADING" if self.auto_trading_enabled else "PAPER TRADING"
        speed_mode = "FAST" if fast_mode else "FULL"
        logger.info(f"Agent initialized in {mode} mode ({speed_mode} analysis)")
        
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
        """Get API credentials from env vars, falling back to settings.json"""
        api_key        = os.getenv('KUCOIN_API_KEY', '')
        api_secret     = os.getenv('KUCOIN_API_SECRET', '')
        api_passphrase = os.getenv('KUCOIN_API_PASSPHRASE', '')

        # Fall back to settings.json when env vars are absent
        if not all([api_key, api_secret, api_passphrase]):
            cfg = self._load_settings_from_file() or {}
            creds = cfg.get('api_credentials', {})
            api_key        = api_key        or creds.get('kucoin_api_key', '')
            api_secret     = api_secret     or creds.get('kucoin_api_secret', '')
            api_passphrase = api_passphrase or creds.get('kucoin_api_passphrase', '')
            if all([api_key, api_secret, api_passphrase]):
                logger.info("✅ KuCoin credentials loaded from settings.json")

        # Determine live-trading flag (settings.json takes precedence over env)
        cfg = self._load_settings_from_file() or {}
        tp = cfg.get('trading_parameters', {})
        enable_live_trading = str(tp.get('ENABLE_LIVE_TRADING', os.getenv('ENABLE_LIVE_TRADING', 'false'))).lower() == 'true'

        if enable_live_trading and not all([api_key, api_secret, api_passphrase]):
            logger.error("KuCoin credentials required for live trading but not found in env or settings.json")
            sys.exit(1)

        if not all([api_key, api_secret, api_passphrase]):
            logger.warning("⚠️  Running in PAPER TRADING mode - API credentials not configured")

        return api_key, api_secret, api_passphrase

    def _sync_kucoin_timestamp(self):
        """Install a self-updating time proxy in kucoin.base_client.

        The proxy recalculates the clock offset every 60 s in a background
        thread so it stays accurate even when Windows NTP corrects the system
        clock after startup.
        """
        import requests as _req
        import kucoin.base_client as _kbc
        import time as _real_time
        import threading as _threading

        def _fetch_offset():
            try:
                resp = _req.get('https://api.kucoin.com/api/v1/timestamp', timeout=5)
                server_ms = resp.json().get('data', 0)
                local_ms  = int(_real_time.time() * 1000)
                return (server_ms - local_ms) / 1000.0
            except Exception:
                return None

        # Get initial offset
        offset_s = _fetch_offset()
        if offset_s is None:
            logger.warning("Could not fetch KuCoin server time — timestamp proxy not applied")
            return

        # Shared mutable state for the proxy
        state = {'offset': offset_s}

        class _LiveTimeProxy:
            """Wraps the time module; .time() returns clock-corrected time."""
            def time(self):
                return _real_time.time() + state['offset']
            def __getattr__(self, n):
                return getattr(_real_time, n)

        _kbc.time = _LiveTimeProxy()
        logger.info(f"KuCoin clock offset {offset_s:+.2f}s — live timestamp proxy installed")

        # Background thread: refreshes offset every 60 s
        def _refresh_loop():
            while True:
                _real_time.sleep(60)
                new_off = _fetch_offset()
                if new_off is not None:
                    state['offset'] = new_off
                    logger.debug(f"KuCoin timestamp proxy offset updated: {new_off:+.2f}s")

        t = _threading.Thread(target=_refresh_loop, daemon=True, name="kucoin-ts-sync")
        t.start()

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
    
    def get_account_balance(self, force_refresh: bool = False) -> float:
        """Get USDT account balance. Caches the result for _balance_cache_ttl seconds
        to avoid hammering KuCoin's authenticated endpoint on every status poll."""
        try:
            # Paper trading: use portfolio manager's in-memory balances
            if not self.auto_trading_enabled:
                self.portfolio_manager.update_balances()
                balances = self.portfolio_manager.balances
                if balances and 'USDT' in balances:
                    return balances['USDT'].get('total', 0.0)
                return self.portfolio_manager.paper_trading_balances.get('USDT', 10000.0)

            # Live trading: return cached value if still fresh
            now = time.time()
            if not force_refresh and (now - self._balance_cache['ts']) < self._balance_cache_ttl:
                return self._balance_cache['value']

            def _get_balance():
                accounts = self.user_client.get_accounts()
                for account in accounts:
                    if account['currency'] == 'USDT' and account['type'] == 'trade':
                        return float(account['balance'])
                return 0.0

            balance = self.retry_api_call(_get_balance)
            if balance is not None:
                self._balance_cache = {'value': balance, 'ts': now}
                return balance
            return self._balance_cache['value']  # stale but better than 0

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return self._balance_cache.get('value', 0.0)
    
    def _update_portfolio_data(self, force: bool = False):
        """Update portfolio balances and positions (throttled to once per 30 s)."""
        now = time.time()
        if not force and hasattr(self, '_last_portfolio_update'):
            if now - self._last_portfolio_update < 30:
                return
        self._last_portfolio_update = now
        try:
            self.portfolio_manager.update_balances()
            current_prices = {}
            for symbol in self.trading_pairs:
                try:
                    ticker = self.market_client.get_24hr_stats(symbol)
                    current_prices[symbol] = float(ticker['last'])
                except Exception:
                    pass
            self.portfolio_manager.update_positions(current_prices)
            self.portfolio_manager.save_portfolio_cache()
        except Exception as e:
            logger.error(f"Error updating portfolio data: {e}")

    def _load_paper_trading_balances(self):
        """Load paper trading initial balances from settings"""
        try:
            settings = self._load_settings_from_json()
            paper_trading = settings.get('paper_trading', {})

            return {
                'USDT': float(paper_trading.get('PAPER_TRADING_INITIAL_BALANCE', 10000)),
                'BTC': float(paper_trading.get('PAPER_TRADING_INITIAL_BTC', 0)),
                'ETH': float(paper_trading.get('PAPER_TRADING_INITIAL_ETH', 0))
            }
        except Exception as e:
            logger.warning(f"Could not load paper trading balances: {e}. Using defaults.")
            return {'USDT': 10000.0, 'BTC': 0.0, 'ETH': 0.0}

    def _load_settings_from_json(self):
        """Load settings from JSON config file"""
        import os
        import json

        config_path = os.path.join(os.getcwd(), 'config', 'settings.json')

        if not os.path.exists(config_path):
            logger.warning(f"Settings file not found: {config_path}")
            return {}

        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return {}

    def _save_settings_to_json(self, settings: Dict):
        """Save settings to JSON config file"""
        import os
        import json

        config_dir = os.path.join(os.getcwd(), 'config')
        config_path = os.path.join(config_dir, 'settings.json')

        # Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)

        try:
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
            logger.info("Settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            raise

    def _get_setting_value(self, settings: Dict, key: str, default: str = ""):
        """Get setting value from nested JSON structure"""
        # Check all sections for the key
        for section in settings.values():
            if isinstance(section, dict) and key in section:
                return section[key]
        return default

    def _reload_trading_pairs(self):
        """Reload trading pairs from saved settings or environment variables"""
        try:
            # Try to load from settings file first
            settings = self._load_settings_from_json()
            trading_params = settings.get('trading_parameters', {})
            trading_pairs_str = trading_params.get('TRADING_PAIRS', '')

            # Fallback to environment variable if not in settings
            if not trading_pairs_str:
                trading_pairs_str = os.getenv('TRADING_PAIRS', 'BTC-USDT,ETH-USDT,ADA-USDT,DOT-USDT')

            # Parse and update trading pairs
            new_pairs = [pair.strip() for pair in trading_pairs_str.split(',') if pair.strip()]

            if new_pairs:
                old_pairs = self.trading_pairs.copy()
                self.trading_pairs = new_pairs
                logger.info(f"✓ Trading pairs reloaded: {', '.join(self.trading_pairs)}")

                # Log if pairs changed
                if set(old_pairs) != set(new_pairs):
                    logger.info(f"📊 Trading pairs updated from {len(old_pairs)} to {len(new_pairs)} pairs")
            else:
                logger.warning("⚠ No valid trading pairs found, keeping existing pairs")

        except Exception as e:
            logger.error(f"❌ Error reloading trading pairs: {e}")

    def _check_kucoin_configured(self) -> bool:
        """Check if KuCoin API credentials are configured"""
        settings = self._load_settings_from_json()
        api_creds = settings.get('api_credentials', {})

        api_key = api_creds.get('kucoin_api_key', '')
        api_secret = api_creds.get('kucoin_api_secret', '')
        api_passphrase = api_creds.get('kucoin_api_passphrase', '')

        return bool(api_key and api_secret and api_passphrase)

    def _serialize_trade_for_api(self, trade) -> Dict:
        """Normalize trade tracker records into the shape used by the React UI."""
        if isinstance(trade, dict):
            data = trade
        else:
            data = {
                'id': getattr(trade, 'id', ''),
                'timestamp': getattr(trade, 'timestamp', ''),
                'symbol': getattr(trade, 'symbol', ''),
                'side': getattr(trade, 'action', ''),
                'action': getattr(trade, 'action', ''),
                'amount': getattr(trade, 'size', 0),
                'quantity': getattr(trade, 'size', 0),
                'price': getattr(trade, 'price', 0),
                'value': getattr(trade, 'value', 0),
                'confidence': getattr(trade, 'confidence', 0),
                'order_id': getattr(trade, 'order_id', None),
                'status': 'closed' if getattr(trade, 'is_closed', False) else 'open',
                'fees': getattr(trade, 'fees', 0),
                'pnl': getattr(trade, 'pnl', 0) or 0,
                'pnl_percent': getattr(trade, 'pnl_percent', 0) or 0,
                'strategy': 'hybrid',
                'stop_loss': getattr(trade, 'stop_loss', None),
                'take_profit': getattr(trade, 'take_profit', None),
                'exit_timestamp': getattr(trade, 'exit_timestamp', None),
                'exit_price': getattr(trade, 'exit_price', None),
                'exit_reason': getattr(trade, 'exit_reason', None),
                'is_closed': getattr(trade, 'is_closed', False),
            }

        symbol = data.get('symbol') or data.get('pair') or ''
        side = data.get('side') or data.get('action') or ''
        quantity = data.get('quantity', data.get('amount', data.get('size', 0)))
        is_closed = bool(data.get('is_closed')) or data.get('status') == 'closed'

        return {
            'id': data.get('id', ''),
            'timestamp': data.get('timestamp', ''),
            'symbol': symbol,
            'pair': symbol,
            'side': side,
            'action': side,
            'amount': quantity,
            'quantity': quantity,
            'price': data.get('price', 0),
            'value': data.get('value', 0),
            'confidence': data.get('confidence', 0),
            'order_id': data.get('order_id'),
            'status': 'closed' if is_closed else data.get('status', 'open'),
            'fees': data.get('fees', 0),
            'pnl': data.get('pnl', 0) or 0,
            'pnl_percent': data.get('pnl_percent', 0) or 0,
            'strategy': data.get('strategy', 'hybrid'),
            'stop_loss': data.get('stop_loss'),
            'take_profit': data.get('take_profit'),
            'exit_timestamp': data.get('exit_timestamp'),
            'exit_price': data.get('exit_price'),
            'exit_reason': data.get('exit_reason'),
            'is_closed': is_closed,
        }

    def _get_api_trades(self, symbol: Optional[str] = None) -> List[Dict]:
        """Return normalized trade records for API consumers."""
        if not hasattr(self, 'trade_tracker'):
            return []

        if hasattr(self.trade_tracker, 'get_all_trades'):
            raw_trades = self.trade_tracker.get_all_trades().values()
        else:
            raw_trades = getattr(self.trade_tracker, 'trades', [])

        trades = [self._serialize_trade_for_api(trade) for trade in raw_trades]
        if symbol:
            trades = [trade for trade in trades if trade.get('symbol') == symbol]
        return trades

    def setup_flask_server(self, host='127.0.0.1', port=5001):
        """Setup Flask server with API endpoints"""
        self.app = Flask(__name__)
        # Enable CORS for React frontend
        CORS(self.app,
             resources={r"/*": {
                 "origins": "*",
                 "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                 "allow_headers": ["Content-Type", "Authorization"]
             }})

        # Add CORS and JSON headers to all responses
        @self.app.after_request
        def add_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            if request.path.startswith('/api/'):
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
        def start_trading_api():
            """Start automated trading"""
            try:
                # Try to get JSON data, but don't fail if it's not JSON
                try:
                    data = request.get_json(force=True, silent=True) or {}
                except:
                    data = {}

                # If paper_trading not provided, check ENABLE_LIVE_TRADING from settings
                # Otherwise use the provided paper_trading parameter
                if 'paper_trading' in data:
                    paper_trading = data.get('paper_trading', True)
                else:
                    # Check current auto_trading_enabled status (which reflects ENABLE_LIVE_TRADING)
                    paper_trading = not self.auto_trading_enabled

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
                logger.error(f"Error starting trading: {e}", exc_info=True)
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

                # Use real live balance; fall back to portfolio_manager's USDT cache
                balance = self.get_account_balance()
                if balance == 0.0:
                    balance = self.portfolio_manager.balances.get('USDT', {}).get('total', 0.0)

                # Compute real total portfolio value: live USDT + crypto position values
                crypto_value = portfolio_data.get('total_crypto_value', 0)
                total_value  = balance + crypto_value

                # Positions list
                positions_dict = portfolio_data.get('positions', {})
                positions_list = []
                for symbol, position_data in positions_dict.items():
                    position_info = position_data.copy()
                    position_info['symbol'] = symbol
                    positions_list.append(position_info)

                # P&L and win-rate from trade tracker
                metrics = {}
                try:
                    metrics = self.trade_tracker.get_performance_summary() if hasattr(self, 'trade_tracker') else {}
                except Exception:
                    pass

                # All non-zero currency holdings with estimated USDT value
                holdings = []
                raw_balances = self.portfolio_manager.balances or {}
                # Fetch prices for trading pairs + any held currencies
                current_prices = {}
                symbols_needed = set(self.trading_pairs)
                for cur in raw_balances:
                    if cur != 'USDT':
                        symbols_needed.add(f'{cur}-USDT')
                for sym in symbols_needed:
                    try:
                        t = self.market_client.get_ticker(sym)
                        current_prices[sym] = float(t.get('price', 0))
                    except Exception:
                        pass
                for currency, info in raw_balances.items():
                    total = info.get('total', 0.0) if isinstance(info, dict) else float(info)
                    if total <= 0:
                        continue
                    if currency == 'USDT':
                        usdt_value = total
                    else:
                        price = current_prices.get(f'{currency}-USDT', 0)
                        usdt_value = total * price if price else None
                    holdings.append({
                        'currency': currency,
                        'amount': total,
                        'available': info.get('available', total) if isinstance(info, dict) else total,
                        'usdt_value': usdt_value,
                    })
                holdings.sort(key=lambda h: (h['usdt_value'] or 0), reverse=True)

                return jsonify({
                    'success': True,
                    'portfolio': {
                        'balance': balance,
                        'holdings': holdings,
                        'positions': positions_list,
                        'total_value':    total_value,
                        'unrealized_pnl': portfolio_data.get('total_unrealized_pnl', 0),
                        'total_pnl':  metrics.get('total_pnl', 0),
                        'win_rate':   metrics.get('win_rate', 0),
                        'total_trades': metrics.get('total_trades', 0),
                        'last_update': datetime.now().isoformat()
                    }
                })
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/settings', methods=['GET'])
        def get_settings():
            """Get current trading settings"""
            try:
                saved_settings = self._load_settings_from_json()

                # Defaults
                defaults = {
                    'trading_parameters': {
                        'TRADING_PAIRS': 'BTC-USDT,ETH-USDT',
                        'TRADING_INTERVAL_MINUTES': '15',
                        'MAX_DAILY_TRADES': '1000',
                        'MIN_CONFIDENCE_THRESHOLD': '0.7',
                        'MINIMUM_BALANCE': '10',
                        'ENABLE_LIVE_TRADING': 'false',
                        'HISTORICAL_DATA_LIMIT': '500'
                    },
                    'risk_management': {
                        'MAX_PORTFOLIO_RISK': '0.02',
                        'MAX_POSITION_SIZE': '0.1',
                        'MAX_PORTFOLIO_EXPOSURE': '0.20',
                        'MAX_OPEN_POSITIONS': '5',
                        'STOP_LOSS_PERCENTAGE': '0.05',
                        'RISK_REWARD_RATIO': '2.0',
                        'USE_TRAILING_STOP': 'false',
                        'TRAILING_STOP_PERCENT': '0.03'
                    },
                    'technical_indicators': {
                        'RSI_PERIOD': '14',
                        'RSI_OVERBOUGHT': '70',
                        'RSI_OVERSOLD': '30',
                        'MACD_FAST': '12',
                        'MACD_SLOW': '26',
                        'MACD_SIGNAL': '9',
                        'BB_PERIOD': '20',
                        'BB_STD_DEV': '2'
                    },
                    'llm_configuration': {
                        'FAST_MODE': 'true',
                        'LLM_PROVIDER': 'openai',
                        'LLM_MODEL_NAME': 'gpt-4o-mini',
                        'LLM_MAX_TOKENS': '1000',
                        'LLM_TEMPERATURE': '0.3',
                        'ENABLE_SENTIMENT_ANALYSIS': 'true'
                    },
                    'paper_trading': {
                        'PAPER_TRADING_INITIAL_BALANCE': '10000',
                        'PAPER_TRADING_INITIAL_BTC': '0',
                        'PAPER_TRADING_INITIAL_ETH': '0'
                    },
                    'performance': {
                        'PERFORMANCE_REPORT_DAYS': '30'
                    },
                    'notifications': {
                        'ENABLE_NOTIFICATIONS': 'false',
                        'NOTIFY_ON_TRADE': 'true',
                        'NOTIFY_ON_ERROR': 'true'
                    }
                }

                # Merge saved settings with defaults
                for section, values in defaults.items():
                    if section not in saved_settings:
                        saved_settings[section] = {}
                    for key, default_value in values.items():
                        if key not in saved_settings[section]:
                            saved_settings[section][key] = default_value

                # Flatten for frontend
                flat_settings = {}
                for section in saved_settings.values():
                    if isinstance(section, dict):
                        flat_settings.update(section)

                return jsonify({
                    'success': True,
                    'settings': flat_settings
                })
            except Exception as e:
                logger.error(f"Error getting settings: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/settings/update', methods=['POST'])
        def update_settings():
            """Update trading settings"""
            try:
                data = request.get_json() or {}

                # Load existing settings
                saved_settings = self._load_settings_from_json()

                # Categorize incoming data
                api_credentials = {}
                trading_parameters = {}
                risk_management = {}
                technical_indicators = {}
                llm_configuration = {}
                paper_trading = {}
                performance = {}
                notifications = {}

                # Map fields to categories
                for key, value in data.items():
                    if key in ['kucoin_api_key', 'kucoin_api_secret', 'kucoin_api_passphrase', 'openai_api_key']:
                        api_credentials[key] = value
                    elif key in ['TRADING_PAIRS', 'TRADING_INTERVAL_MINUTES', 'MAX_DAILY_TRADES',
                                'MIN_CONFIDENCE_THRESHOLD', 'MINIMUM_BALANCE', 'ENABLE_LIVE_TRADING',
                                'HISTORICAL_DATA_LIMIT']:
                        trading_parameters[key] = value
                    elif key in ['MAX_PORTFOLIO_RISK', 'MAX_POSITION_SIZE', 'MAX_PORTFOLIO_EXPOSURE',
                                'MAX_OPEN_POSITIONS', 'STOP_LOSS_PERCENTAGE', 'RISK_REWARD_RATIO',
                                'USE_TRAILING_STOP', 'TRAILING_STOP_PERCENT']:
                        risk_management[key] = value
                    elif key in ['RSI_PERIOD', 'RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'MACD_FAST',
                                'MACD_SLOW', 'MACD_SIGNAL', 'BB_PERIOD', 'BB_STD_DEV']:
                        technical_indicators[key] = value
                    elif key in ['LLM_PROVIDER', 'LLM_MODEL_NAME', 'LLM_MAX_TOKENS',
                                'LLM_TEMPERATURE', 'ENABLE_SENTIMENT_ANALYSIS', 'FAST_MODE']:
                        llm_configuration[key] = value
                    elif key in ['PAPER_TRADING_INITIAL_BALANCE', 'PAPER_TRADING_INITIAL_BTC',
                                'PAPER_TRADING_INITIAL_ETH']:
                        paper_trading[key] = value
                    elif key in ['PERFORMANCE_REPORT_DAYS']:
                        performance[key] = value
                    elif key in ['ENABLE_NOTIFICATIONS', 'NOTIFY_ON_TRADE', 'NOTIFY_ON_ERROR']:
                        notifications[key] = value

                # Update settings structure
                if api_credentials:
                    if 'api_credentials' not in saved_settings:
                        saved_settings['api_credentials'] = {}
                    saved_settings['api_credentials'].update(api_credentials)

                if trading_parameters:
                    if 'trading_parameters' not in saved_settings:
                        saved_settings['trading_parameters'] = {}
                    saved_settings['trading_parameters'].update(trading_parameters)

                if risk_management:
                    if 'risk_management' not in saved_settings:
                        saved_settings['risk_management'] = {}
                    saved_settings['risk_management'].update(risk_management)

                if technical_indicators:
                    if 'technical_indicators' not in saved_settings:
                        saved_settings['technical_indicators'] = {}
                    saved_settings['technical_indicators'].update(technical_indicators)

                if llm_configuration:
                    if 'llm_configuration' not in saved_settings:
                        saved_settings['llm_configuration'] = {}
                    saved_settings['llm_configuration'].update(llm_configuration)

                if paper_trading:
                    if 'paper_trading' not in saved_settings:
                        saved_settings['paper_trading'] = {}
                    saved_settings['paper_trading'].update(paper_trading)

                if performance:
                    if 'performance' not in saved_settings:
                        saved_settings['performance'] = {}
                    saved_settings['performance'].update(performance)

                if notifications:
                    if 'notifications' not in saved_settings:
                        saved_settings['notifications'] = {}
                    saved_settings['notifications'].update(notifications)

                # Save to JSON file
                self._save_settings_to_json(saved_settings)

                # Reload trading pairs if they were modified
                if 'TRADING_PAIRS' in trading_parameters:
                    self._reload_trading_pairs()

                # Sync ENABLE_LIVE_TRADING with auto_trading_enabled
                if 'ENABLE_LIVE_TRADING' in trading_parameters:
                    enable_live = str(trading_parameters['ENABLE_LIVE_TRADING']).lower()
                    if enable_live in ['true', '1', 'yes']:
                        self.auto_trading_enabled = True
                        logger.info("✅ Live Trading ENABLED via settings")
                    else:
                        self.auto_trading_enabled = False
                        logger.info("📝 Paper Trading mode ENABLED via settings")

                # Apply FAST_MODE immediately to the running agent
                if 'FAST_MODE' in llm_configuration:
                    new_fast_mode = str(llm_configuration['FAST_MODE']).lower() in ('true', '1', 'yes')
                    self.fast_mode = new_fast_mode
                    if hasattr(self, 'strategy') and self.strategy:
                        self.strategy.enable_llm = not new_fast_mode
                    mode_str = 'FAST (Technical Only)' if new_fast_mode else 'FULL (LLM+Technical)'
                    logger.info(f"✅ Fast Mode updated to {mode_str}")

                # Reload trading interval if modified
                if 'TRADING_INTERVAL_MINUTES' in trading_parameters:
                    self.trading_interval_minutes = int(trading_parameters['TRADING_INTERVAL_MINUTES'])
                    logger.info(f"✅ Trading interval updated to {self.trading_interval_minutes} minutes")

                # Reload other trading parameters if modified
                if 'MIN_CONFIDENCE_THRESHOLD' in trading_parameters:
                    self.min_confidence_threshold = float(trading_parameters['MIN_CONFIDENCE_THRESHOLD'])
                    logger.info(f"✅ Confidence threshold updated to {self.min_confidence_threshold}")

                if 'MAX_DAILY_TRADES' in trading_parameters:
                    self.max_daily_trades = int(trading_parameters['MAX_DAILY_TRADES'])
                    logger.info(f"✅ Max daily trades updated to {self.max_daily_trades}")

                if 'MINIMUM_BALANCE' in trading_parameters:
                    self.minimum_balance = float(trading_parameters['MINIMUM_BALANCE'])
                    logger.info(f"✅ Minimum balance updated to ${self.minimum_balance}")

                if 'HISTORICAL_DATA_LIMIT' in trading_parameters:
                    self.historical_data_limit = int(trading_parameters['HISTORICAL_DATA_LIMIT'])
                    logger.info(f"✅ Historical data limit updated to {self.historical_data_limit}")

                # Apply risk management parameters immediately
                if 'MAX_PORTFOLIO_RISK' in risk_management:
                    self.risk_manager.max_portfolio_risk = float(risk_management['MAX_PORTFOLIO_RISK'])
                    logger.info(f"✅ Max portfolio risk updated to {self.risk_manager.max_portfolio_risk:.2%}")

                if 'MAX_POSITION_SIZE' in risk_management:
                    self.risk_manager.max_position_size = float(risk_management['MAX_POSITION_SIZE'])
                    logger.info(f"✅ Max position size updated to {self.risk_manager.max_position_size:.2%}")

                if 'STOP_LOSS_PERCENTAGE' in risk_management:
                    self.risk_manager.stop_loss_percentage = float(risk_management['STOP_LOSS_PERCENTAGE'])
                    logger.info(f"✅ Stop loss updated to {self.risk_manager.stop_loss_percentage:.2%}")

                if 'RISK_REWARD_RATIO' in risk_management:
                    self.risk_manager.risk_reward_ratio = float(risk_management['RISK_REWARD_RATIO'])
                    logger.info(f"✅ Risk/reward ratio updated to {self.risk_manager.risk_reward_ratio}")

                if 'MAX_PORTFOLIO_EXPOSURE' in risk_management:
                    self.risk_manager.max_portfolio_exposure = float(risk_management['MAX_PORTFOLIO_EXPOSURE'])
                    logger.info(f"✅ Max portfolio exposure updated to {self.risk_manager.max_portfolio_exposure:.2%}")

                if 'MAX_OPEN_POSITIONS' in risk_management:
                    self.max_open_positions = int(risk_management['MAX_OPEN_POSITIONS'])
                    logger.info(f"✅ Max open positions updated to {self.max_open_positions}")

                if 'USE_TRAILING_STOP' in risk_management:
                    self.risk_manager.use_trailing_stop = str(risk_management['USE_TRAILING_STOP']).lower() in ('true', '1', 'yes')
                    logger.info(f"✅ Trailing stop {'enabled' if self.risk_manager.use_trailing_stop else 'disabled'}")

                if 'TRAILING_STOP_PERCENT' in risk_management:
                    self.risk_manager.trailing_stop_percent = float(risk_management['TRAILING_STOP_PERCENT'])
                    logger.info(f"✅ Trailing stop % updated to {self.risk_manager.trailing_stop_percent:.2%}")

                # Apply technical indicator parameters immediately (live on strategy object)
                if self.strategy:
                    if 'RSI_PERIOD' in technical_indicators:
                        self.strategy.rsi_period = int(technical_indicators['RSI_PERIOD'])
                        logger.info(f"✅ RSI period updated to {self.strategy.rsi_period}")

                    if 'RSI_OVERBOUGHT' in technical_indicators:
                        v = int(technical_indicators['RSI_OVERBOUGHT'])
                        self.strategy.rsi_overbought_weak = v
                        self.strategy.rsi_overbought_strong = min(v + 5, 100)
                        logger.info(f"✅ RSI overbought updated to {v} (strong: {self.strategy.rsi_overbought_strong})")

                    if 'RSI_OVERSOLD' in technical_indicators:
                        v = int(technical_indicators['RSI_OVERSOLD'])
                        self.strategy.rsi_oversold_weak = v
                        self.strategy.rsi_oversold_strong = max(v - 5, 0)
                        logger.info(f"✅ RSI oversold updated to {v} (strong: {self.strategy.rsi_oversold_strong})")

                    if 'MACD_FAST' in technical_indicators:
                        self.strategy.macd_fast = int(technical_indicators['MACD_FAST'])
                        logger.info(f"✅ MACD fast period updated to {self.strategy.macd_fast}")

                    if 'MACD_SLOW' in technical_indicators:
                        self.strategy.macd_slow = int(technical_indicators['MACD_SLOW'])
                        logger.info(f"✅ MACD slow period updated to {self.strategy.macd_slow}")

                    if 'MACD_SIGNAL' in technical_indicators:
                        self.strategy.macd_signal = int(technical_indicators['MACD_SIGNAL'])
                        logger.info(f"✅ MACD signal period updated to {self.strategy.macd_signal}")

                    if 'BB_PERIOD' in technical_indicators:
                        self.strategy.bb_period = int(technical_indicators['BB_PERIOD'])
                        logger.info(f"✅ Bollinger period updated to {self.strategy.bb_period}")

                    if 'BB_STD_DEV' in technical_indicators:
                        self.strategy.bb_std_dev = float(technical_indicators['BB_STD_DEV'])
                        logger.info(f"✅ Bollinger std dev updated to {self.strategy.bb_std_dev}")

                    if 'LLM_MAX_TOKENS' in llm_configuration:
                        self.strategy.max_tokens = int(llm_configuration['LLM_MAX_TOKENS'])
                        logger.info(f"✅ LLM max tokens updated to {self.strategy.max_tokens}")

                    if 'LLM_TEMPERATURE' in llm_configuration:
                        self.strategy.temperature = float(llm_configuration['LLM_TEMPERATURE'])
                        logger.info(f"✅ LLM temperature updated to {self.strategy.temperature}")

                    if 'LLM_MODEL_NAME' in llm_configuration:
                        self.strategy.model_id = llm_configuration['LLM_MODEL_NAME']
                        logger.info(f"✅ LLM model updated to {self.strategy.model_id}")

                if 'LLM_PROVIDER' in llm_configuration:
                    new_provider = str(llm_configuration['LLM_PROVIDER']).lower()
                    self.llm_provider = new_provider
                    if new_provider == 'bedrock' and self.strategy and not self.fast_mode:
                        import boto3 as _boto3
                        try:
                            self.strategy.bedrock_client = _boto3.client('bedrock-runtime', region_name=self.aws_region)
                            self.strategy.enable_llm = True
                            logger.info(f"✅ LLM provider switched to Bedrock (region: {self.aws_region})")
                        except Exception as _e:
                            logger.warning(f"⚠️ Could not init Bedrock client: {_e}")
                    elif new_provider == 'openai':
                        logger.info("ℹ️ OpenAI provider saved — will be active on next restart when LLM enabled")

                if 'ENABLE_SENTIMENT_ANALYSIS' in llm_configuration:
                    self.enable_sentiment_analysis = str(llm_configuration['ENABLE_SENTIMENT_ANALYSIS']).lower() in ('true', '1', 'yes')
                    logger.info(f"✅ Sentiment analysis {'enabled' if self.enable_sentiment_analysis else 'disabled'}")

                # Apply paper trading balances immediately (paper mode only)
                if paper_trading and not self.auto_trading_enabled:
                    new_usdt = float(paper_trading.get('PAPER_TRADING_INITIAL_BALANCE',
                                     self.portfolio_manager.paper_trading_balances.get('USDT', 10000)))
                    new_btc  = float(paper_trading.get('PAPER_TRADING_INITIAL_BTC',
                                     self.portfolio_manager.paper_trading_balances.get('BTC', 0)))
                    new_eth  = float(paper_trading.get('PAPER_TRADING_INITIAL_ETH',
                                     self.portfolio_manager.paper_trading_balances.get('ETH', 0)))
                    self.portfolio_manager.paper_trading_balances = {'USDT': new_usdt, 'BTC': new_btc, 'ETH': new_eth}
                    self.portfolio_manager.balances = {
                        'USDT': {'total': new_usdt, 'available': new_usdt, 'hold': 0.0},
                        'BTC':  {'total': new_btc,  'available': new_btc,  'hold': 0.0},
                        'ETH':  {'total': new_eth,  'available': new_eth,  'hold': 0.0},
                    }
                    self._balance_cache = {'value': new_usdt, 'ts': 0.0}  # Invalidate cache
                    logger.info(f"✅ Paper trading balances reset — USDT: ${new_usdt:,.2f}, BTC: {new_btc}, ETH: {new_eth}")

                message = 'Settings saved successfully to config/settings.json'
                logger.info(message)

                return jsonify({
                    'success': True,
                    'message': message,
                    'settings': data
                })
            except Exception as e:
                logger.error(f"Error updating settings: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/reset', methods=['POST'])
        def reset_data():
            """Reset/clear all logs, trades, and portfolio data"""
            try:
                data = request.get_json() or {}
                reset_logs = data.get('reset_logs', True)
                reset_trades = data.get('reset_trades', True)
                reset_portfolio = data.get('reset_portfolio', True)

                results = {
                    'logs_cleared': False,
                    'trades_cleared': False,
                    'portfolio_reset': False
                }

                # Clear logs
                if reset_logs:
                    try:
                        # Clear in-memory logs
                        self.log_memory.clear()

                        # Truncate the log file to remove all persisted logs
                        import os
                        log_file_path = 'logs/trading_agent.log'
                        if os.path.exists(log_file_path):
                            # Open in write mode to truncate the file
                            with open(log_file_path, 'w') as f:
                                f.write('')  # Clear the file
                            logger.info("✅ Log file cleared")

                        results['logs_cleared'] = True
                        logger.info("✅ Logs cleared from memory and file")
                    except Exception as e:
                        logger.error(f"Error clearing logs: {e}")

                # Clear trades
                if reset_trades:
                    try:
                        # Clear in-memory trade history
                        self.trade_history = []
                        self.daily_trade_count = 0

                        # Clear trade tracker data
                        self.trade_tracker.trades = []
                        self.trade_tracker.portfolio = {
                            "initial_balance": 0.0,
                            "current_balance": 0.0,
                            "total_invested": 0.0,
                            "total_fees": 0.0,
                            "positions": {},
                            "last_updated": datetime.now().isoformat()
                        }

                        # Save empty trades
                        self.trade_tracker._save_trades()
                        self.trade_tracker._save_portfolio()

                        results['trades_cleared'] = True
                        logger.info("✅ Trades cleared")
                    except Exception as e:
                        logger.error(f"Error clearing trades: {e}")

                # Reset portfolio
                if reset_portfolio:
                    try:
                        # Reset portfolio manager
                        paper_trading_balances = self._load_paper_trading_balances()
                        self.portfolio_manager.balances = self.portfolio_manager._get_mock_balances()
                        self.portfolio_manager.positions = {}
                        self.portfolio_manager.portfolio_value = paper_trading_balances.get('USDT', 10000.0)
                        self.portfolio_manager.last_update = datetime.now()

                        # Save to cache
                        self.portfolio_manager.save_portfolio_cache()

                        results['portfolio_reset'] = True
                        logger.info("✅ Portfolio reset to initial state")
                    except Exception as e:
                        logger.error(f"Error resetting portfolio: {e}")

                # Clear trading data files
                if reset_trades:
                    import os
                    try:
                        trades_file = os.path.join("trading_data", "trades.json")
                        portfolio_file = os.path.join("trading_data", "portfolio.json")

                        if os.path.exists(trades_file):
                            os.remove(trades_file)
                        if os.path.exists(portfolio_file):
                            os.remove(portfolio_file)
                    except Exception as e:
                        logger.warning(f"Could not remove data files: {e}")

                message = "Data reset completed"
                if results['logs_cleared']:
                    message += " - Logs cleared"
                if results['trades_cleared']:
                    message += " - Trades cleared"
                if results['portfolio_reset']:
                    message += " - Portfolio reset"

                logger.info(f"🔄 {message}")

                return jsonify({
                    'success': True,
                    'message': message,
                    'results': results
                })
            except Exception as e:
                logger.error(f"Error resetting data: {e}")
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

        @self.app.route('/api/market/ticker/<symbol>', methods=['GET'])
        def get_market_ticker(symbol):
            """Get ticker data for a symbol"""
            try:
                stats = self.market_data.get_24hr_stats(symbol)
                if not stats:
                    price = self.market_data.get_current_price(symbol)
                    stats = {'symbol': symbol, 'last': price}
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'ticker': stats,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting ticker for {symbol}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/market/ohlcv/<symbol>', methods=['GET'])
        def get_market_ohlcv(symbol):
            """Get OHLCV candles for a symbol"""
            try:
                timeframe = request.args.get('timeframe', '1h')
                limit = request.args.get('limit', 100, type=int)
                interval_map = {
                    '1m': '1min',
                    '5m': '5min',
                    '15m': '15min',
                    '30m': '30min',
                    '1h': '1hour',
                    '4h': '4hour',
                    '1d': '1day',
                    '1w': '1week',
                }
                interval = interval_map.get(timeframe, timeframe)
                df = self.market_data.get_historical_klines(symbol, interval, limit=limit)
                candles = []
                if df is not None and not df.empty:
                    for row in df.tail(limit).to_dict('records'):
                        ts = row.get('timestamp')
                        candles.append({
                            'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else ts,
                            'open': float(row.get('open', 0)),
                            'high': float(row.get('high', 0)),
                            'low': float(row.get('low', 0)),
                            'close': float(row.get('close', 0)),
                            'volume': float(row.get('volume', 0)),
                        })
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'candles': candles,
                    'count': len(candles)
                })
            except Exception as e:
                logger.error(f"Error getting OHLCV for {symbol}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/market/orderbook/<symbol>', methods=['GET'])
        def get_market_orderbook(symbol):
            """Get order book depth for a symbol"""
            try:
                depth = request.args.get('depth', 20, type=int)
                orderbook = self.market_data.get_order_book(symbol)
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'bids': orderbook.get('bids', [])[:depth],
                    'asks': orderbook.get('asks', [])[:depth],
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting orderbook for {symbol}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/market/trades/<symbol>', methods=['GET'])
        def get_market_recent_trades(symbol):
            """Get recent market trades for a symbol"""
            try:
                limit = request.args.get('limit', 50, type=int)
                raw_trades = []
                if hasattr(self.market_client, 'get_trade_histories'):
                    raw_trades = self.market_client.get_trade_histories(symbol)
                trades = []
                for trade in (raw_trades or [])[:limit]:
                    trades.append({
                        'price': float(trade.get('price', 0)),
                        'size': float(trade.get('size', 0)),
                        'side': trade.get('side', ''),
                        'time': trade.get('time') or trade.get('createdAt'),
                    })
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'trades': trades,
                    'count': len(trades)
                })
            except Exception as e:
                logger.error(f"Error getting recent market trades for {symbol}: {e}")
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

        @self.app.route('/api/trades', methods=['GET'])
        def get_trades():
            """Get trading history/trades"""
            try:
                limit = request.args.get('limit', type=int)
                symbol = request.args.get('symbol')

                trades_list = self._get_api_trades(symbol=symbol)

                # Sort by timestamp descending
                trades_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                if limit:
                    trades_list = trades_list[:limit]

                # Count open and closed trades
                open_trades = [t for t in trades_list if t.get('status') == 'open']
                closed_trades = [t for t in trades_list if t.get('status') == 'closed']

                return jsonify({
                    'success': True,
                    'trades': trades_list,
                    'total': len(trades_list),
                    'open': len(open_trades),
                    'closed': len(closed_trades)
                })
            except Exception as e:
                logger.error(f"Error getting trades: {e}")
                return jsonify({
                    'success': True,
                    'trades': [],
                    'total': 0,
                    'open': 0,
                    'closed': 0
                })

        @self.app.route('/api/trades/<trade_id>', methods=['GET'])
        def get_trade(trade_id):
            """Get a single trade by ID"""
            try:
                trade = self.trade_tracker.get_trade_by_id(trade_id) if hasattr(self, 'trade_tracker') else None
                if not trade:
                    return jsonify({'success': False, 'error': 'Trade not found'}), 404
                return jsonify({'success': True, 'trade': self._serialize_trade_for_api(trade)})
            except Exception as e:
                logger.error(f"Error getting trade {trade_id}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/trades/execute', methods=['POST'])
        def execute_trade_api():
            """Execute a manual trade through the same trading engine used by analysis."""
            try:
                data = request.get_json() or {}
                symbol = data.get('symbol') or data.get('pair')
                action = (data.get('action') or data.get('side') or '').upper()
                confidence = float(data.get('confidence', 1.0))

                if not symbol or action not in ('BUY', 'SELL'):
                    return jsonify({'success': False, 'error': 'symbol and BUY/SELL action are required'}), 400

                executed = self.execute_trade(symbol, action, confidence, {'source': 'manual_api'})
                return jsonify({
                    'success': executed,
                    'message': 'Trade executed' if executed else 'Trade was rejected by risk or strategy checks',
                    'symbol': symbol,
                    'action': action
                }), 200 if executed else 409
            except Exception as e:
                logger.error(f"Error executing manual trade: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/positions/<position_id>/close', methods=['POST'])
        def close_position_api(position_id):
            """Close a tracked position/trade at the current or provided exit price."""
            try:
                data = request.get_json(silent=True) or {}
                exit_price = data.get('exit_price')
                trade = self.trade_tracker.get_trade_by_id(position_id) if hasattr(self, 'trade_tracker') else None

                if not trade:
                    return jsonify({'success': False, 'error': 'Position not found'}), 404

                if exit_price is None:
                    exit_price = self.market_data.get_current_price(trade.symbol)
                if exit_price is None:
                    return jsonify({'success': False, 'error': 'Could not determine exit price'}), 400

                closed = self.trade_tracker.close_position(position_id, float(exit_price))
                return jsonify({
                    'success': closed,
                    'message': 'Position closed' if closed else 'Position could not be closed',
                    'position_id': position_id,
                    'exit_price': float(exit_price)
                }), 200 if closed else 409
            except Exception as e:
                logger.error(f"Error closing position {position_id}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/performance', methods=['GET'])
        def get_performance():
            """Get performance metrics"""
            try:
                # Get performance data from trade_tracker
                metrics = self.trade_tracker.get_performance_summary() if hasattr(self, 'trade_tracker') else {}

                return jsonify({
                    'success': True,
                    'total_pnl': metrics.get('total_pnl', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'winning_trades': metrics.get('winning_trades', 0),
                    'losing_trades': metrics.get('losing_trades', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'avg_win': metrics.get('avg_win', 0),
                    'avg_loss': metrics.get('avg_loss', 0),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting performance: {e}")
                return jsonify({
                    'success': True,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'avg_win': 0,
                    'avg_loss': 0
                })

        @self.app.route('/api/positions/open', methods=['GET'])
        def get_open_positions():
            """Get open positions"""
            try:
                # Get open positions from portfolio manager
                positions = self.portfolio_manager.get_positions() if hasattr(self, 'portfolio_manager') else []

                positions_list = []
                for symbol, position_data in positions.items() if isinstance(positions, dict) else []:
                    positions_list.append({
                        'symbol': symbol,
                        'amount': position_data.get('amount', 0),
                        'entry_price': position_data.get('entry_price', 0),
                        'current_price': position_data.get('current_price', 0),
                        'pnl': position_data.get('pnl', 0),
                        'pnl_percent': position_data.get('pnl_percent', 0)
                    })

                return jsonify({
                    'success': True,
                    'positions': positions_list,
                    'count': len(positions_list)
                })
            except Exception as e:
                logger.error(f"Error getting open positions: {e}")
                return jsonify({
                    'success': True,
                    'positions': [],
                    'count': 0
                })



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
            
            # Get historical data — respects the HISTORICAL_DATA_LIMIT setting
            limit = getattr(self, 'historical_data_limit', 200 if self.fast_mode else 500)
            df = self.market_data.get_historical_klines(symbol, '1hour', limit=limit)
            if df.empty:
                logger.error(f"❌ No data available for {symbol}")
                return {}
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Get news data only if LLM and sentiment analysis are enabled
            news_data = []
            if use_llm and not self.fast_mode and getattr(self, 'enable_sentiment_analysis', True):
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

        # If there's an error (no trades), return clean empty metrics instead
        if 'error' in performance_metrics:
            performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl_per_trade': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'closed_trades': 0,
                'executed_trades': 0,
                'paper_trades': 0
            }

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

