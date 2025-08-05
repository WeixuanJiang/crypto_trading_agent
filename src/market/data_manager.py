"""Enhanced market data manager with caching and retry mechanisms"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from kucoin.client import Client

from ..core.config import Config
from ..core.logger import get_logger, TradeLoggerMixin
from ..core.exceptions import MarketDataError, APIError, ValidationError
from ..data.models import MarketData
from ..data.repositories import MarketDataRepository


@dataclass
class DataRequest:
    """Market data request configuration"""
    symbol: str
    interval: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 500
    use_cache: bool = True
    max_age_minutes: int = 5


class MarketDataCache:
    """In-memory cache for market data"""
    
    def __init__(self, max_size: int = 1000, ttl_minutes: int = 5):
        self.cache: Dict[str, Tuple[datetime, List[MarketData]]] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.logger = get_logger('market_cache')
    
    def get(self, key: str) -> Optional[List[MarketData]]:
        """Get cached data if valid"""
        if key in self.cache:
            timestamp, data = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                self.logger.debug(f"Cache hit for {key}")
                return data
            else:
                # Remove expired entry
                del self.cache[key]
                self.logger.debug(f"Cache expired for {key}")
        
        return None
    
    def set(self, key: str, data: List[MarketData]) -> None:
        """Cache data with timestamp"""
        # Clean up if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup_oldest()
        
        self.cache[key] = (datetime.now(), data)
        self.logger.debug(f"Cached data for {key}")
    
    def _cleanup_oldest(self) -> None:
        """Remove oldest cache entries"""
        if not self.cache:
            return
        
        # Remove 20% of oldest entries
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1][0])
        remove_count = max(1, len(sorted_items) // 5)
        
        for key, _ in sorted_items[:remove_count]:
            del self.cache[key]
        
        self.logger.debug(f"Cleaned up {remove_count} cache entries")
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = datetime.now()
        valid_entries = sum(1 for _, (ts, _) in self.cache.items() if now - ts < self.ttl)
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'max_size': self.max_size,
            'ttl_minutes': self.ttl.total_seconds() / 60
        }


class MarketDataManager(TradeLoggerMixin):
    """Enhanced market data manager with caching, retry, and validation"""
    
    def __init__(self, config: Config, repository: Optional[MarketDataRepository] = None):
        super().__init__()
        self.config = config
        self.repository = repository
        self.logger = get_logger('market_data')
        
        # Initialize KuCoin client
        self.client = Client(
            api_key=config.api.kucoin_api_key,
            api_secret=config.api.kucoin_api_secret,
            api_passphrase=config.api.kucoin_passphrase,
            sandbox=config.api.kucoin_sandbox
        )
        
        # Initialize cache
        self.cache = MarketDataCache(
            max_size=config.market_data.cache_size,
            ttl_minutes=config.market_data.cache_ttl_minutes
        )
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0 / config.api.rate_limit_per_second
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.log_info("MarketDataManager initialized")
    
    def get_klines(self, request: DataRequest) -> List[MarketData]:
        """Get kline data with caching and retry"""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Try cache first
            if request.use_cache:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    self.log_debug(f"Returning cached data for {request.symbol}")
                    return cached_data
            
            # Fetch from API
            data = self._fetch_klines_with_retry(request)
            
            # Cache the result
            if request.use_cache and data:
                self.cache.set(cache_key, data)
            
            # Store in database if repository is available
            if self.repository and data:
                self._store_market_data(data)
            
            self.log_info(f"Retrieved {len(data)} klines for {request.symbol}")
            return data
        
        except Exception as e:
            self.log_error(f"Failed to get klines for {request.symbol}: {e}")
            raise MarketDataError(f"Failed to get market data: {e}")
    
    def get_latest_price(self, symbol: str) -> Decimal:
        """Get latest price for symbol"""
        try:
            self._rate_limit()
            
            # Get ticker data
            ticker = self.client.get_ticker(symbol)
            if not ticker or 'price' not in ticker:
                raise MarketDataError(f"No price data for {symbol}")
            
            price = Decimal(str(ticker['price']))
            self.log_debug(f"Latest price for {symbol}: {price}")
            return price
        
        except Exception as e:
            self.log_error(f"Failed to get latest price for {symbol}: {e}")
            raise MarketDataError(f"Failed to get latest price: {e}")
    
    def get_24hr_stats(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr statistics for symbol"""
        try:
            self._rate_limit()
            
            stats = self.client.get_24hr_stats(symbol)
            if not stats:
                raise MarketDataError(f"No 24hr stats for {symbol}")
            
            # Convert to proper types
            result = {
                'symbol': symbol,
                'price_change': Decimal(str(stats.get('changePrice', '0'))),
                'price_change_percent': float(stats.get('changeRate', '0')) * 100,
                'high_price': Decimal(str(stats.get('high', '0'))),
                'low_price': Decimal(str(stats.get('low', '0'))),
                'volume': Decimal(str(stats.get('vol', '0'))),
                'quote_volume': Decimal(str(stats.get('volValue', '0'))),
                'open_price': Decimal(str(stats.get('open', '0'))),
                'close_price': Decimal(str(stats.get('close', '0'))),
                'last_price': Decimal(str(stats.get('last', '0')))
            }
            
            self.log_debug(f"24hr stats for {symbol}: {result['price_change_percent']:.2f}%")
            return result
        
        except Exception as e:
            self.log_error(f"Failed to get 24hr stats for {symbol}: {e}")
            raise MarketDataError(f"Failed to get 24hr stats: {e}")
    
    def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get order book data"""
        try:
            self._rate_limit()
            
            order_book = self.client.get_order_book(symbol)
            if not order_book:
                raise MarketDataError(f"No order book data for {symbol}")
            
            # Process bids and asks
            bids = [[Decimal(price), Decimal(size)] for price, size in order_book.get('bids', [])[:depth]]
            asks = [[Decimal(price), Decimal(size)] for price, size in order_book.get('asks', [])[:depth]]
            
            result = {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now()
            }
            
            if bids and asks:
                result['spread'] = asks[0][0] - bids[0][0]
                result['spread_percent'] = float(result['spread'] / bids[0][0] * 100)
            
            self.log_debug(f"Order book for {symbol}: {len(bids)} bids, {len(asks)} asks")
            return result
        
        except Exception as e:
            self.log_error(f"Failed to get order book for {symbol}: {e}")
            raise MarketDataError(f"Failed to get order book: {e}")
    
    def get_symbols(self) -> List[Dict[str, Any]]:
        """Get all available trading symbols"""
        try:
            self._rate_limit()
            
            symbols = self.client.get_symbols()
            if not symbols:
                raise MarketDataError("No symbols data available")
            
            # Filter and format symbols
            result = []
            for symbol in symbols:
                if symbol.get('enableTrading', False):
                    result.append({
                        'symbol': symbol['symbol'],
                        'base_currency': symbol['baseCurrency'],
                        'quote_currency': symbol['quoteCurrency'],
                        'base_min_size': Decimal(str(symbol.get('baseMinSize', '0'))),
                        'quote_min_size': Decimal(str(symbol.get('quoteMinSize', '0'))),
                        'base_increment': Decimal(str(symbol.get('baseIncrement', '0'))),
                        'quote_increment': Decimal(str(symbol.get('quoteIncrement', '0'))),
                        'price_increment': Decimal(str(symbol.get('priceIncrement', '0'))),
                        'fee_rate': Decimal(str(symbol.get('feeCurrency', '0')))
                    })
            
            self.log_info(f"Retrieved {len(result)} trading symbols")
            return result
        
        except Exception as e:
            self.log_error(f"Failed to get symbols: {e}")
            raise MarketDataError(f"Failed to get symbols: {e}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is tradeable"""
        try:
            symbols = self.get_symbols()
            return any(s['symbol'] == symbol for s in symbols)
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear market data cache"""
        self.cache.clear()
        self.log_info("Market data cache cleared")
    
    def get_historical_data(self, symbol: str, interval: str = '1hour', limit: int = 500) -> pd.DataFrame:
        """Get historical data as pandas DataFrame for strategy analysis"""
        try:
            # Create data request
            request = DataRequest(
                symbol=symbol,
                interval=interval,
                limit=limit,
                use_cache=True
            )
            
            # Get market data
            market_data = self.get_klines(request)
            
            if not market_data:
                self.log_warning(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_list = []
            for md in market_data:
                data_list.append({
                    'timestamp': md.timestamp,
                    'open': float(md.open_price),
                    'high': float(md.high_price),
                    'low': float(md.low_price),
                    'close': float(md.close_price),
                    'volume': float(md.volume)
                })
            
            df = pd.DataFrame(data_list)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            self.log_info(f"Retrieved {len(df)} historical data points for {symbol}")
            return df
            
        except Exception as e:
            self.log_error(f"Failed to get historical data for {symbol}: {e}")
            # Return empty DataFrame instead of raising exception to prevent crashes
            return pd.DataFrame()
    
    def _fetch_klines_with_retry(self, request: DataRequest, max_retries: int = 3) -> List[MarketData]:
        """Fetch klines with retry mechanism"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Prepare parameters
                params = {
                    'symbol': request.symbol,
                    'type': request.interval
                }
                
                if request.start_time:
                    params['startAt'] = int(request.start_time.timestamp())
                if request.end_time:
                    params['endAt'] = int(request.end_time.timestamp())
                
                # Fetch data
                klines = self.client.get_kline_data(**params)
                
                if not klines:
                    raise MarketDataError(f"No kline data returned for {request.symbol}")
                
                # Convert to MarketData objects
                market_data = []
                for kline in klines:
                    try:
                        data = MarketData(
                            symbol=request.symbol,
                            timestamp=datetime.fromtimestamp(int(kline[0])),
                            open_price=Decimal(str(kline[1])),
                            close_price=Decimal(str(kline[2])),
                            high_price=Decimal(str(kline[3])),
                            low_price=Decimal(str(kline[4])),
                            volume=Decimal(str(kline[5])),
                            quote_volume=Decimal(str(kline[6]))
                        )
                        market_data.append(data)
                    except (ValueError, IndexError) as e:
                        self.log_warning(f"Invalid kline data: {kline}, error: {e}")
                        continue
                
                if not market_data:
                    raise MarketDataError(f"No valid kline data for {request.symbol}")
                
                # Sort by timestamp
                market_data.sort(key=lambda x: x.timestamp)
                
                self.log_debug(f"Fetched {len(market_data)} klines for {request.symbol} (attempt {attempt + 1})")
                return market_data
            
            except Exception as e:
                last_exception = e
                self.log_warning(f"Attempt {attempt + 1} failed for {request.symbol}: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 1.0
                    self.log_debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise APIError(f"Failed to fetch klines after {max_retries} attempts: {last_exception}")
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request"""
        key_parts = [request.symbol, request.interval]
        
        if request.start_time:
            key_parts.append(request.start_time.isoformat())
        if request.end_time:
            key_parts.append(request.end_time.isoformat())
        
        key_parts.append(str(request.limit))
        
        return '|'.join(key_parts)
    
    def _rate_limit(self) -> None:
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _store_market_data(self, data: List[MarketData]) -> None:
        """Store market data in database"""
        if not self.repository:
            return
        
        try:
            for market_data in data:
                self.repository.create(market_data)
        except Exception as e:
            self.log_warning(f"Failed to store market data: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)