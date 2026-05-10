"""
Yahoo Finance Data Fetcher

Provides fallback market data from Yahoo Finance when KuCoin API is not available.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from decimal import Decimal

from ..core.logger import get_logger

logger = get_logger(__name__)


class YahooFinanceDataFetcher:
    """Fetches cryptocurrency market data from Yahoo Finance"""

    # Mapping from KuCoin symbols to Yahoo Finance tickers
    SYMBOL_MAP = {
        'BTC-USDT': 'BTC-USD',
        'ETH-USDT': 'ETH-USD',
        'ADA-USDT': 'ADA-USD',
        'DOT-USDT': 'DOT-USD',
        'SOL-USDT': 'SOL-USD',
        'XRP-USDT': 'XRP-USD',
        'DOGE-USDT': 'DOGE-USD',
        'MATIC-USDT': 'MATIC-USD',
        'AVAX-USDT': 'AVAX-USD',
        'LINK-USDT': 'LINK-USD',
    }

    # Interval mapping from KuCoin to Yahoo Finance
    INTERVAL_MAP = {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '1hour': '1h',
        '1day': '1d',
    }

    def __init__(self):
        """Initialize Yahoo Finance data fetcher"""
        self.cache = {}
        self.cache_duration = 60  # Cache for 60 seconds
        logger.info("✅ Yahoo Finance data fetcher initialized")

    def _convert_symbol(self, kucoin_symbol: str) -> str:
        """Convert KuCoin symbol to Yahoo Finance ticker"""
        yf_symbol = self.SYMBOL_MAP.get(kucoin_symbol)
        if not yf_symbol:
            # Try to auto-convert if not in map
            if '-USDT' in kucoin_symbol:
                base = kucoin_symbol.replace('-USDT', '')
                yf_symbol = f"{base}-USD"
            else:
                yf_symbol = kucoin_symbol

        logger.debug(f"Converting {kucoin_symbol} -> {yf_symbol}")
        return yf_symbol

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            yf_symbol = self._convert_symbol(symbol)

            # Check cache
            cache_key = f"price_{yf_symbol}"
            if cache_key in self.cache:
                cached_time, cached_price = self.cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_duration:
                    return cached_price

            # Fetch from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period='1d', interval='1m')

            if data.empty:
                logger.warning(f"No data available for {yf_symbol}")
                return None

            price = float(data['Close'].iloc[-1])

            # Update cache
            self.cache[cache_key] = (datetime.now(), price)

            logger.debug(f"Yahoo Finance price for {symbol}: ${price:.2f}")
            return price

        except Exception as e:
            logger.error(f"Error fetching price from Yahoo Finance for {symbol}: {e}")
            return None

    def get_24hr_stats(self, symbol: str) -> Optional[Dict]:
        """Get 24-hour statistics for a symbol"""
        try:
            yf_symbol = self._convert_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)

            # Get 2 days of data to calculate 24hr change
            data = ticker.history(period='2d', interval='1h')

            if data.empty or len(data) < 2:
                logger.warning(f"Insufficient data for 24hr stats: {yf_symbol}")
                return None

            current_price = float(data['Close'].iloc[-1])
            yesterday_price = float(data['Close'].iloc[0])
            high_24h = float(data['High'].tail(24).max())
            low_24h = float(data['Low'].tail(24).min())
            volume_24h = float(data['Volume'].tail(24).sum())

            change_24h = current_price - yesterday_price
            change_percent = (change_24h / yesterday_price * 100) if yesterday_price > 0 else 0

            stats = {
                'last': current_price,
                'high': high_24h,
                'low': low_24h,
                'vol': volume_24h,
                'changePrice': change_24h,
                'changeRate': change_percent / 100,  # As decimal
                'symbol': symbol,
            }

            logger.debug(f"Yahoo Finance 24hr stats for {symbol}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error fetching 24hr stats from Yahoo Finance for {symbol}: {e}")
            return None

    def get_klines(self, symbol: str, interval: str = '1hour', limit: int = 100) -> pd.DataFrame:
        """Get historical candlestick data"""
        try:
            yf_symbol = self._convert_symbol(symbol)
            yf_interval = self.INTERVAL_MAP.get(interval, '1h')

            # Calculate period based on limit
            # Yahoo Finance doesn't support exact limit, so we estimate
            if yf_interval == '1m':
                period = f"{min(limit, 7)}d"  # Max 7 days for 1min data
            elif yf_interval == '1h':
                period = f"{min(limit // 24 + 1, 730)}d"  # Max 2 years
            else:
                period = 'max'

            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=yf_interval)

            if data.empty:
                logger.warning(f"No kline data available for {yf_symbol}")
                return pd.DataFrame()

            # Limit to requested number of candles
            data = data.tail(limit)

            # Convert to expected format
            df = pd.DataFrame({
                'time': data.index.astype('int64') // 10**9,  # Unix timestamp
                'open': data['Open'],
                'high': data['High'],
                'low': data['Low'],
                'close': data['Close'],
                'volume': data['Volume'],
            })

            logger.debug(f"Yahoo Finance klines for {symbol}: {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"Error fetching klines from Yahoo Finance for {symbol}: {e}")
            return pd.DataFrame()

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker information (similar to KuCoin format)"""
        try:
            price = self.get_current_price(symbol)
            if price is None:
                return None

            return {
                'symbol': symbol,
                'price': str(price),
                'time': int(datetime.now().timestamp() * 1000)
            }

        except Exception as e:
            logger.error(f"Error fetching ticker from Yahoo Finance for {symbol}: {e}")
            return None

    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        try:
            # Try to fetch BTC price as a test
            test_price = self.get_current_price('BTC-USDT')
            return test_price is not None
        except Exception as e:
            logger.error(f"Yahoo Finance availability check failed: {e}")
            return False
