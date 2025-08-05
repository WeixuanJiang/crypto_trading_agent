import pandas as pd
import numpy as np
from kucoin.client import Client
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import requests
from ..utils.config_manager import config

API_CONFIG = config.api

class MarketDataManager:
    def __init__(self, market_client: Client):
        self.client = market_client
    
    def retry_api_call(self, func, *args, **kwargs):
        """Retry API calls with exponential backoff"""
        max_retries = getattr(API_CONFIG, 'max_retries', 5)
        retry_delay = getattr(API_CONFIG, 'retry_delay_seconds', 2)
        backoff_factor = getattr(API_CONFIG, 'backoff_factor', 2.0)
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, Exception) as e:
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (backoff_factor ** attempt)
                        print(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                raise e
        return None
        
    def get_historical_klines(self, symbol: str, interval: str = '1hour', 
                             start_time: Optional[str] = None, 
                             end_time: Optional[str] = None,
                             limit: int = 1000) -> pd.DataFrame:
        """Get historical kline data from KuCoin"""
        try:
            # If no start time provided, get data from 30 days ago
            if not start_time:
                start_time = int((datetime.now() - timedelta(days=30)).timestamp())
            if not end_time:
                end_time = int(datetime.now().timestamp())
                
            klines = self.client.get_kline_data(
                symbol=symbol,
                kline_type=interval,
                startAt=start_time,
                endAt=end_time
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='s')
            numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'turnover']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, interval: str = '1hour', limit: int = 500) -> pd.DataFrame:
        """Get historical data as pandas DataFrame for strategy analysis"""
        try:
            # Use the existing get_historical_klines method
            df = self.get_historical_klines(symbol, interval, limit=limit)
            
            if df.empty:
                print(f"⚠️ No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Keep timestamp as a column instead of index to avoid pandas_ta comparison issues
            # Reset index to use numeric index for pandas_ta compatibility
            if df.index.name == 'timestamp':
                df.reset_index(inplace=True)
            
            print(f"📊 Retrieved {len(df)} historical data points for {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Failed to get historical data for {symbol}: {e}")
            # Return empty DataFrame instead of raising exception to prevent crashes
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            print(f"DEBUG: Attempting to get ticker for {symbol}")
            ticker = self.retry_api_call(self.client.get_ticker, symbol)
            print(f"DEBUG: Ticker result for {symbol}: {ticker}")
            if ticker:
                price = float(ticker['price'])
                print(f"DEBUG: Extracted price for {symbol}: {price}")
                return price
            print(f"DEBUG: No ticker data returned for {symbol}")
            return None
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_order_book(self, symbol: str) -> Dict:
        """Get order book data"""
        try:
            order_book = self.client.get_part_order(symbol)
            return {
                'bids': [[float(price), float(size)] for price, size in order_book['bids']],
                'asks': [[float(price), float(size)] for price, size in order_book['asks']],
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return {'bids': [], 'asks': [], 'timestamp': datetime.now()}
    
    def get_24hr_stats(self, symbol: str) -> Dict:
        """Get 24hr statistics"""
        try:
            stats = self.client.get_24h_stats(symbol)
            return {
                'symbol': stats['symbol'],
                'high': float(stats['high']),
                'low': float(stats['low']),
                'vol': float(stats['vol']),
                'volValue': float(stats['volValue']),
                'last': float(stats['last']),
                'buy': float(stats['buy']),
                'sell': float(stats['sell']),
                'changePrice': float(stats['changePrice']),
                'changeRate': float(stats['changeRate']),
                'time': datetime.fromtimestamp(int(stats['time']) / 1000)
            }
        except Exception as e:
            print(f"Error fetching 24h stats for {symbol}: {e}")
            return {}
    
    def get_market_list(self) -> List[str]:
        """Get list of available trading pairs"""
        try:
            symbols = self.client.get_symbols()
            return [symbol['symbol'] for symbol in symbols if symbol['enableTrading']]
        except Exception as e:
            print(f"Error fetching trading pairs: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed symbol information including increments"""
        try:
            symbols = self.retry_api_call(self.client.get_symbols)
            if symbols:
                for s in symbols:
                    if s['symbol'] == symbol:
                        return {
                            'symbol': s['symbol'],
                            'baseIncrement': float(s['baseIncrement']),
                            'quoteIncrement': float(s['quoteIncrement']),
                            'priceIncrement': float(s['priceIncrement']),
                            'baseMinSize': float(s['baseMinSize']),
                            'quoteMinSize': float(s['quoteMinSize']),
                            'baseMaxSize': float(s['baseMaxSize']),
                            'quoteMaxSize': float(s['quoteMaxSize']),
                            'minFunds': float(s['minFunds']),
                            'enableTrading': s['enableTrading']
                        }
            return {}
        except Exception as e:
            print(f"Error fetching symbol info for {symbol}: {e}")
            return {}
    
    def get_news_data(self, symbol: str, limit: int = 10) -> List[str]:
        """Fetch news data (placeholder - you would integrate with news APIs)"""
        # This is a placeholder. In a real implementation, you would integrate with:
        # - CoinGecko API
        # - CryptoNews API
        # - Twitter API
        # - Reddit API
        # - News aggregators
        
        try:
            # Example using CoinGecko API (free tier)
            coin_id = symbol.split('-')[0].lower()  # Extract base currency
            
            # This is a simplified example - you'd need proper news API integration
            news_items = [
                f"Market analysis for {symbol}: Technical indicators showing mixed signals",
                f"{symbol} trading volume increased by 15% in the last 24 hours",
                f"Institutional interest in {coin_id} continues to grow",
                f"Recent price action in {symbol} suggests consolidation phase",
                f"Market sentiment for {coin_id} remains cautiously optimistic"
            ]
            
            return news_items[:limit]
            
        except Exception as e:
            print(f"Error fetching news data: {e}")
            return []
    
    def calculate_market_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate additional market metrics"""
        if df.empty:
            return {}
        
        try:
            latest = df.iloc[-1]
            
            # Price metrics
            price_change_24h = (latest['close'] - df.iloc[-24]['close']) / df.iloc[-24]['close'] * 100 if len(df) >= 24 else 0
            price_change_7d = (latest['close'] - df.iloc[-168]['close']) / df.iloc[-168]['close'] * 100 if len(df) >= 168 else 0
            
            # Volume metrics
            avg_volume_24h = df['volume'].tail(24).mean() if len(df) >= 24 else df['volume'].mean()
            volume_ratio = latest['volume'] / avg_volume_24h if avg_volume_24h > 0 else 1
            
            # Volatility metrics
            returns = df['close'].pct_change().dropna()
            volatility_24h = returns.tail(24).std() * np.sqrt(24) * 100 if len(returns) >= 24 else 0
            
            # Support and resistance levels
            high_24h = df['high'].tail(24).max() if len(df) >= 24 else latest['high']
            low_24h = df['low'].tail(24).min() if len(df) >= 24 else latest['low']
            
            return {
                'current_price': latest['close'],
                'price_change_24h': price_change_24h,
                'price_change_7d': price_change_7d,
                'volume_24h': df['volume'].tail(24).sum() if len(df) >= 24 else latest['volume'],
                'avg_volume_24h': avg_volume_24h,
                'volume_ratio': volume_ratio,
                'volatility_24h': volatility_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'support_level': low_24h,
                'resistance_level': high_24h,
                'timestamp': latest['timestamp']
            }
            
        except Exception as e:
            print(f"Error calculating market metrics: {e}")
            return {}
    
    def get_top_gainers_losers(self, limit: int = 10) -> Dict:
        """Get top gainers and losers"""
        try:
            tickers = self.client.get_all_tickers()['ticker']
            
            # Filter and sort by change rate
            tickers_data = []
            for ticker in tickers:
                if ticker['symbol'].endswith('-USDT'):  # Focus on USDT pairs
                    tickers_data.append({
                        'symbol': ticker['symbol'],
                        'price': float(ticker['last']),
                        'change_rate': float(ticker['changeRate']) * 100,
                        'volume': float(ticker['vol'])
                    })
            
            # Sort by change rate
            tickers_data.sort(key=lambda x: x['change_rate'], reverse=True)
            
            return {
                'top_gainers': tickers_data[:limit],
                'top_losers': tickers_data[-limit:],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching gainers/losers: {e}")
            return {'top_gainers': [], 'top_losers': [], 'timestamp': datetime.now()}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and is tradeable"""
        try:
            symbols = self.retry_api_call(self.client.get_symbols)
            if symbols:
                for s in symbols:
                    if s['symbol'] == symbol and s['enableTrading']:
                        return True
            return False
        except Exception as e:
            print(f"Error validating symbol: {e}")
            return False
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """Get comprehensive real-time data for a symbol"""
        try:
            current_price = self.get_current_price(symbol)
            stats_24hr = self.get_24hr_stats(symbol)
            order_book = self.get_order_book(symbol)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'stats_24hr': stats_24hr,
                'order_book': order_book,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            return {}