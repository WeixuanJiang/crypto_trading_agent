import boto3
import pandas as pd
import numpy as np
import pandas_ta as ta
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from botocore.exceptions import ClientError
from dotenv import load_dotenv

class LLMTradingStrategy:
    def __init__(self, aws_region: str = None, enable_llm: bool = None, cross_region_inference: bool = None):
        # Load environment variables
        load_dotenv()
        
        # Use environment variables with fallback to defaults
        aws_region = aws_region or os.getenv('AWS_REGION', 'us-east-1')
        enable_llm = enable_llm if enable_llm is not None else (os.getenv('FAST_MODE', 'true').lower() != 'true')
        cross_region_inference = cross_region_inference if cross_region_inference is not None else (os.getenv('AWS_CROSS_REGION_INFERENCE', 'true').lower() == 'true')
        # Initialize AWS Bedrock client with cross-region inference support
        self.bedrock_client = None
        self.enable_llm = enable_llm
        self.cross_region_inference = cross_region_inference
        
        # Technical Analysis Parameters
        self.rsi_period = int(os.getenv('RSI_PERIOD', '14'))
        self.rsi_short_period = int(os.getenv('RSI_SHORT_PERIOD', '7'))
        self.rsi_oversold_strong = int(os.getenv('RSI_OVERSOLD_STRONG', '25'))
        self.rsi_oversold_weak = int(os.getenv('RSI_OVERSOLD_WEAK', '35'))
        self.rsi_overbought_weak = int(os.getenv('RSI_OVERBOUGHT_WEAK', '65'))
        self.rsi_overbought_strong = int(os.getenv('RSI_OVERBOUGHT_STRONG', '75'))
        self.rsi_short_oversold = int(os.getenv('RSI_SHORT_OVERSOLD', '20'))
        self.rsi_short_overbought = int(os.getenv('RSI_SHORT_OVERBOUGHT', '80'))
        
        self.macd_fast = int(os.getenv('MACD_FAST', '12'))
        self.macd_slow = int(os.getenv('MACD_SLOW', '26'))
        self.macd_signal = int(os.getenv('MACD_SIGNAL', '9'))
        
        self.bb_period = int(os.getenv('BB_PERIOD', '20'))
        self.bb_std_dev = int(os.getenv('BB_STD_DEV', '2'))
        self.bb_min_width = float(os.getenv('BB_MIN_WIDTH', '0.04'))
        
        self.stoch_k_period = int(os.getenv('STOCH_K_PERIOD', '14'))
        self.stoch_d_period = int(os.getenv('STOCH_D_PERIOD', '3'))
        self.stoch_oversold_strong = int(os.getenv('STOCH_OVERSOLD_STRONG', '15'))
        self.stoch_oversold_weak = int(os.getenv('STOCH_OVERSOLD_WEAK', '25'))
        self.stoch_overbought_weak = int(os.getenv('STOCH_OVERBOUGHT_WEAK', '75'))
        self.stoch_overbought_strong = int(os.getenv('STOCH_OVERBOUGHT_STRONG', '85'))
        
        self.atr_period = int(os.getenv('ATR_PERIOD', '14'))
        self.sma_20_period = int(os.getenv('SMA_20_PERIOD', '20'))
        self.sma_50_period = int(os.getenv('SMA_50_PERIOD', '50'))
        self.sma_200_period = int(os.getenv('SMA_200_PERIOD', '200'))
        self.ema_12_period = int(os.getenv('EMA_12_PERIOD', '12'))
        self.ema_26_period = int(os.getenv('EMA_26_PERIOD', '26'))
        
        self.volume_sma_period = int(os.getenv('VOLUME_SMA_PERIOD', '20'))
        self.support_resistance_window = int(os.getenv('SUPPORT_RESISTANCE_WINDOW', '20'))
        
        self.williams_r_period = int(os.getenv('WILLIAMS_R_PERIOD', '14'))
        self.williams_r_oversold_strong = int(os.getenv('WILLIAMS_R_OVERSOLD_STRONG', '-85'))
        self.williams_r_oversold_weak = int(os.getenv('WILLIAMS_R_OVERSOLD_WEAK', '-80'))
        self.williams_r_overbought_weak = int(os.getenv('WILLIAMS_R_OVERBOUGHT_WEAK', '-20'))
        self.williams_r_overbought_strong = int(os.getenv('WILLIAMS_R_OVERBOUGHT_STRONG', '-15'))
        
        self.cci_period = int(os.getenv('CCI_PERIOD', '20'))
        self.cci_oversold_strong = int(os.getenv('CCI_OVERSOLD_STRONG', '-150'))
        self.cci_oversold_weak = int(os.getenv('CCI_OVERSOLD_WEAK', '-100'))
        self.cci_overbought_weak = int(os.getenv('CCI_OVERBOUGHT_WEAK', '100'))
        self.cci_overbought_strong = int(os.getenv('CCI_OVERBOUGHT_STRONG', '150'))
        
        self.roc_period = int(os.getenv('ROC_PERIOD', '10'))
        self.mfi_period = int(os.getenv('MFI_PERIOD', '14'))
        self.mfi_oversold = int(os.getenv('MFI_OVERSOLD', '20'))
        self.mfi_overbought = int(os.getenv('MFI_OVERBOUGHT', '80'))
        
        # Enhanced indicators parameters
        self.adx_period = int(os.getenv('ADX_PERIOD', '14'))
        self.aroon_period = int(os.getenv('AROON_PERIOD', '14'))
        self.psar_af = float(os.getenv('PSAR_AF', '0.02'))
        self.psar_max_af = float(os.getenv('PSAR_MAX_AF', '0.2'))
        self.vwap_period = int(os.getenv('VWAP_PERIOD', '20'))
        self.keltner_period = int(os.getenv('KELTNER_PERIOD', '20'))
        self.keltner_multiplier = float(os.getenv('KELTNER_MULTIPLIER', '2.0'))
        
        # Multi-timeframe analysis weights
        self.timeframe_weights = {
            'short': 0.3,   # 5m-15m for entry timing
            'medium': 0.4,  # 1h-4h for trend confirmation
            'long': 0.3     # Daily for overall direction
        }
        
        # Pattern recognition cache
        self.pattern_cache = {}
        self.market_regime_cache = {}
        
        # Advanced signal processing
        self.signal_history = {}
        self.confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7'))
        
        # Risk-adjusted position sizing
        self.volatility_lookback = int(os.getenv('VOLATILITY_LOOKBACK', '20'))
        self.max_position_risk = float(os.getenv('MAX_POSITION_RISK', '0.02'))
        
        if enable_llm:
            try:
                # Primary region client
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=aws_region
                )
                
                # Cross-region fallback clients for better availability
                if cross_region_inference:
                    self.fallback_regions = ['us-west-2', 'eu-west-1', 'ap-southeast-1']
                    self.fallback_clients = {}
                    for region in self.fallback_regions:
                        if region != aws_region:
                            try:
                                self.fallback_clients[region] = boto3.client(
                                    'bedrock-runtime',
                                    region_name=region
                                )
                            except Exception as e:
                                print(f"Warning: Could not initialize fallback client for {region}: {e}")
                
            except Exception as e:
                print(f"Warning: Could not initialize AWS Bedrock client: {e}")
                self.bedrock_client = None
                self.enable_llm = False
        
        self.sentiment_cache = {}  # Cache LLM results
        self.model_id = os.getenv('LLM_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '1000'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.3'))
        
    def analyze_market_sentiment(self, symbol: str, news_data: List[str]) -> Dict:
        """Use AWS Bedrock Claude 3.5 Haiku to analyze market sentiment from news and social media"""
        if not self.enable_llm or not self.bedrock_client:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "key_factors": ["LLM analysis disabled"],
                "risk_level": "medium",
                "recommendation": "hold"
            }
        
        # Create cache key from news data
        news_key = hash(' '.join(news_data[:3])) if news_data else 0
        cache_key = f"{symbol}_{news_key}"
        
        # Check cache first
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        prompt = f"""
        Analyze the market sentiment for {symbol} based on the following news and information:
        
        {' '.join(news_data[:3])}  # Limit to avoid token limits
        
        Provide a JSON response with:
        1. sentiment_score: float between -1 (very bearish) and 1 (very bullish)
        2. confidence: float between 0 and 1
        3. key_factors: list of main factors affecting sentiment
        4. risk_level: string (low, medium, high)
        5. recommendation: string (buy, sell, hold)
        
        Response format: {{"sentiment_score": 0.5, "confidence": 0.8, "key_factors": [...], "risk_level": "medium", "recommendation": "buy"}}
        """
        
        # Try primary region first, then fallback regions if cross-region inference is enabled
        clients_to_try = [self.bedrock_client]
        if self.cross_region_inference and hasattr(self, 'fallback_clients'):
            clients_to_try.extend(self.fallback_clients.values())
        
        for client in clients_to_try:
            try:
                # Prepare the request body for Claude 3.5 Haiku
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are an expert cryptocurrency market analyst. Provide accurate, data-driven analysis.\n\n{prompt}"
                        }
                    ]
                })
                
                # Invoke the model
                response = client.invoke_model(
                    modelId=self.model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json"
                )
                
                # Parse the response
                response_body = json.loads(response['body'].read())
                content = response_body['content'][0]['text']
                
                # Extract JSON from the response
                try:
                    # Try to find JSON in the response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        json_str = content[start_idx:end_idx]
                        analysis = json.loads(json_str)
                    else:
                        # Fallback parsing if no clear JSON structure
                        analysis = json.loads(content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a basic response
                    analysis = {
                        "sentiment_score": 0.0,
                        "confidence": 0.5,
                        "key_factors": ["Unable to parse LLM response"],
                        "risk_level": "medium",
                        "recommendation": "hold"
                    }
                
                # Cache the result
                self.sentiment_cache[cache_key] = analysis
                return analysis
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                print(f"AWS Bedrock error ({error_code}): {e}")
                if client != clients_to_try[-1]:  # Not the last client to try
                    continue
            except Exception as e:
                print(f"LLM analysis error: {e}")
                if client != clients_to_try[-1]:  # Not the last client to try
                    continue
        
        # If all clients failed, return fallback
        fallback = {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "key_factors": ["LLM analysis failed"],
            "risk_level": "medium",
            "recommendation": "hold"
        }
        # Cache the fallback result too
        self.sentiment_cache[cache_key] = fallback
        return fallback
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for enhanced buy low, sell high strategy"""
        # Ensure proper data types to avoid dtype warnings
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')
        
        # Basic price-based indicators
        df['sma_20'] = ta.sma(df['close'], length=self.sma_20_period)
        df['sma_50'] = ta.sma(df['close'], length=self.sma_50_period)
        df['sma_200'] = ta.sma(df['close'], length=self.sma_200_period)  # Long-term trend
        df['ema_12'] = ta.ema(df['close'], length=self.ema_12_period)
        df['ema_26'] = ta.ema(df['close'], length=self.ema_26_period)
        
        # Additional EMAs for better trend analysis
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # MACD with enhanced analysis
        macd_result = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df['macd'] = macd_result['MACD_12_26_9']
        df['macd_signal'] = macd_result['MACDs_12_26_9']
        df['macd_hist'] = macd_result['MACDh_12_26_9']
        df['macd_momentum'] = df['macd_hist'].diff()  # MACD momentum
        
        # RSI with multiple timeframes for better low/high detection
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        df['rsi_short'] = ta.rsi(df['close'], length=self.rsi_short_period)  # More sensitive
        df['rsi_long'] = ta.rsi(df['close'], length=21)  # Less sensitive
        
        # RSI divergence detection
        df['rsi_divergence'] = self._detect_rsi_divergence(df)
        
        # Bollinger Bands with enhanced metrics
        bb_result = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
        df['bb_upper'] = bb_result['BBU_20_2.0']
        df['bb_middle'] = bb_result['BBM_20_2.0']
        df['bb_lower'] = bb_result['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic with enhanced signals
        stoch_result = ta.stoch(df['high'], df['low'], df['close'], k=self.stoch_k_period, d=self.stoch_d_period)
        df['stoch_k'] = stoch_result['STOCHk_14_3_3']
        df['stoch_d'] = stoch_result['STOCHd_14_3_3']
        df['stoch_momentum'] = df['stoch_k'] - df['stoch_d']
        
        # Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=self.volume_sma_period)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = self._calculate_vwap(df)
        
        # Volatility indicators
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        df['volatility'] = df['close'].rolling(window=self.volatility_lookback).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
        
        # Advanced trend indicators
        df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)['ADX_14']
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=self.adx_period)
        df['plus_di'] = adx_result['DMP_14']
        df['minus_di'] = adx_result['DMN_14']
        
        # Aroon Oscillator
        aroon_result = ta.aroon(df['high'], df['low'], length=self.aroon_period)
        df['aroon_up'] = aroon_result['AROONU_14']
        df['aroon_down'] = aroon_result['AROOND_14']
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        
        # Parabolic SAR
        df['psar'] = ta.psar(df['high'], df['low'], af=self.psar_af, max_af=self.psar_max_af)['PSARl_0.02_0.2']
        df['psar_trend'] = np.where(df['close'] > df['psar'], 1, -1)
        
        # Keltner Channels
        df['keltner_upper'], df['keltner_middle'], df['keltner_lower'] = self._calculate_keltner_channels(df)
        df['keltner_position'] = (df['close'] - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'])
        
        # Buy Low, Sell High specific indicators
        # Enhanced Support and Resistance levels
        df['support'] = df['low'].rolling(window=self.support_resistance_window).min()
        df['resistance'] = df['high'].rolling(window=self.support_resistance_window).max()
        
        # Dynamic support/resistance based on volume
        df['volume_support'], df['volume_resistance'] = self._calculate_volume_levels(df)
        
        # Price position within recent range (0 = at low, 1 = at high)
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        df['price_position'] = df['price_position'].fillna(0.5)  # Handle division by zero
        
        # Williams %R for overbought/oversold
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=self.williams_r_period)
        
        # Commodity Channel Index
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=self.cci_period)
        
        # Rate of Change for momentum
        df['roc'] = ta.roc(df['close'], length=self.roc_period)
        
        # Money Flow Index - calculate with robust data type handling
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Round volume values and convert to int64 for MFI calculation
                volume_converted = df['volume'].round().astype('int64')
                df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], volume_converted, length=self.mfi_period)
        except Exception as e:
            # Fallback: try with float64 and suppress all warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    volume_converted = df['volume'].astype('float64')
                    df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], volume_converted, length=self.mfi_period)
            except Exception:
                # Final fallback: use NaN values
                df['mfi'] = np.nan
        
        # Market structure analysis
        df['market_structure'] = self._analyze_market_structure(df)
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Pattern recognition
        df['patterns'] = self._detect_candlestick_patterns(df)
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).rolling(window=self.vwap_period).sum() / df['volume'].rolling(window=self.vwap_period).sum()
    
    def _calculate_keltner_channels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        middle = df['ema_20'] if 'ema_20' in df.columns else ta.ema(df['close'], length=20)
        atr_mult = df['atr'] * self.keltner_multiplier
        upper = middle + atr_mult
        lower = middle - atr_mult
        return upper, middle, lower
    
    def _calculate_volume_levels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate volume-weighted support and resistance levels"""
        window = self.support_resistance_window
        
        # Volume-weighted support (where high volume occurred at low prices)
        volume_support = df.apply(lambda x: self._get_volume_level(df, x.name, window, 'support'), axis=1)
        
        # Volume-weighted resistance (where high volume occurred at high prices)
        volume_resistance = df.apply(lambda x: self._get_volume_level(df, x.name, window, 'resistance'), axis=1)
        
        return volume_support, volume_resistance
    
    def _get_volume_level(self, df: pd.DataFrame, idx: int, window: int, level_type: str) -> float:
        """Get volume-weighted support or resistance level"""
        if idx < window:
            return df['close'].iloc[idx]
        
        window_data = df.iloc[idx-window:idx]
        
        if level_type == 'support':
            # Find price level with highest volume in lower 30% of price range
            price_threshold = window_data['low'].min() + (window_data['high'].max() - window_data['low'].min()) * 0.3
            relevant_data = window_data[window_data['close'] <= price_threshold]
        else:  # resistance
            # Find price level with highest volume in upper 30% of price range
            price_threshold = window_data['low'].min() + (window_data['high'].max() - window_data['low'].min()) * 0.7
            relevant_data = window_data[window_data['close'] >= price_threshold]
        
        if len(relevant_data) == 0:
            return df['close'].iloc[idx]
        
        # Return volume-weighted average price
        return (relevant_data['close'] * relevant_data['volume']).sum() / relevant_data['volume'].sum()
    
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Detect RSI divergence patterns"""
        divergence = pd.Series(0, index=df.index)
        
        if len(df) < 20:
            return divergence
        
        # Look for bullish divergence (price makes lower low, RSI makes higher low)
        for i in range(10, len(df)-1):
            if i < 20:
                continue
                
            # Find recent lows in price and RSI
            price_window = df['close'].iloc[i-10:i+1]
            rsi_window = df['rsi'].iloc[i-10:i+1]
            
            if len(price_window) < 5 or len(rsi_window) < 5:
                continue
            
            current_price = df['close'].iloc[i]
            current_rsi = df['rsi'].iloc[i]
            
            # Check for bullish divergence
            if (current_price < price_window.min() * 1.01 and  # Near recent low
                current_rsi > rsi_window.min() + 5):  # RSI higher than recent low
                divergence.iloc[i] = 1  # Bullish divergence
            
            # Check for bearish divergence
            elif (current_price > price_window.max() * 0.99 and  # Near recent high
                  current_rsi < rsi_window.max() - 5):  # RSI lower than recent high
                divergence.iloc[i] = -1  # Bearish divergence
        
        return divergence
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> pd.Series:
        """Analyze market structure (higher highs, lower lows, etc.)"""
        structure = pd.Series('sideways', index=df.index)
        
        if len(df) < 20:
            return structure
        
        for i in range(10, len(df)):
            recent_highs = df['high'].iloc[i-10:i]
            recent_lows = df['low'].iloc[i-10:i]
            
            # Uptrend: higher highs and higher lows
            if (df['high'].iloc[i] > recent_highs.max() * 0.995 and
                df['low'].iloc[i] > recent_lows.min() * 1.005):
                structure.iloc[i] = 'uptrend'
            
            # Downtrend: lower highs and lower lows
            elif (df['high'].iloc[i] < recent_highs.max() * 0.995 and
                  df['low'].iloc[i] < recent_lows.min() * 0.995):
                structure.iloc[i] = 'downtrend'
            
            # Otherwise sideways
            else:
                structure.iloc[i] = 'sideways'
        
        return structure
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength based on multiple factors"""
        if len(df) < 20:
            return pd.Series(0.5, index=df.index)
        
        # Combine ADX, price momentum, and volume
        adx_strength = df['adx'] / 100  # Normalize ADX
        price_momentum = abs(df['roc']) / 10  # Normalize ROC
        volume_strength = np.minimum(df['volume_ratio'], 3) / 3  # Cap and normalize volume ratio
        
        # Weighted combination
        trend_strength = (adx_strength * 0.4 + price_momentum * 0.3 + volume_strength * 0.3)
        
        return np.minimum(trend_strength, 1.0)  # Cap at 1.0
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect key candlestick patterns using pandas_ta and custom logic"""
        patterns = pd.Series('none', index=df.index)
        
        if len(df) < 5:
            return patterns
        
        # Calculate basic candle metrics
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        candle_range = df['high'] - df['low']
        
        # Avoid division by zero
        candle_range = candle_range.replace(0, 0.0001)
        
        # Doji pattern: small body relative to range
        doji_mask = body < (candle_range * 0.1)
        
        # Hammer pattern: small body, long lower shadow, small upper shadow, bullish
        hammer_mask = (
            (lower_shadow > body * 2) & 
            (upper_shadow < body * 0.5) & 
            (df['close'] > df['open']) &
            (body < candle_range * 0.3)
        )
        
        # Bullish engulfing: current candle engulfs previous bearish candle
        prev_bearish = (df['close'].shift(1) < df['open'].shift(1))
        current_bullish = (df['close'] > df['open'])
        engulfing = (
            prev_bearish &
            current_bullish &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        )
        
        # Morning star pattern (simplified): gap down, small body, gap up
        gap_down = df['high'].shift(1) < df['low'].shift(2)
        small_body = body.shift(1) < candle_range.shift(1) * 0.3
        gap_up = df['low'] > df['high'].shift(1)
        morning_star_mask = gap_down & small_body & gap_up & current_bullish
        
        # Evening star pattern (simplified): gap up, small body, gap down
        gap_up_prev = df['low'].shift(1) > df['high'].shift(2)
        gap_down_current = df['high'] < df['low'].shift(1)
        current_bearish = (df['close'] < df['open'])
        evening_star_mask = gap_up_prev & small_body & gap_down_current & current_bearish
        
        # Bearish engulfing: current bearish candle engulfs previous bullish candle
        prev_bullish = (df['close'].shift(1) > df['open'].shift(1))
        bearish_engulfing = (
            prev_bullish &
            current_bearish &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        )
        
        # Apply patterns in order of priority
        patterns.loc[morning_star_mask] = 'morning_star'
        patterns.loc[evening_star_mask] = 'evening_star'
        patterns.loc[engulfing] = 'bullish_engulfing'
        patterns.loc[bearish_engulfing] = 'bearish_engulfing'
        patterns.loc[hammer_mask] = 'bullish_hammer'
        patterns.loc[doji_mask] = 'doji'
        
        return patterns
    
    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced technical analysis signals for buy low, sell high strategy"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Enhanced RSI signals with multiple timeframes and divergence
        rsi_signal = 0
        rsi_divergence_bonus = 0
        
        # Check for RSI divergence
        if hasattr(latest, 'rsi_divergence') and latest['rsi_divergence'] == 1:
            rsi_divergence_bonus = 1  # Bullish divergence
        elif hasattr(latest, 'rsi_divergence') and latest['rsi_divergence'] == -1:
            rsi_divergence_bonus = -1  # Bearish divergence
        
        if latest['rsi'] < self.rsi_oversold_strong and latest['rsi_short'] < self.rsi_short_oversold:
            rsi_signal = 2 + rsi_divergence_bonus  # Very oversold with potential divergence
        elif latest['rsi'] < self.rsi_oversold_weak:
            rsi_signal = 1 + max(0, rsi_divergence_bonus)  # Oversold
        elif latest['rsi'] > self.rsi_overbought_strong and latest['rsi_short'] > self.rsi_short_overbought:
            rsi_signal = -2 + rsi_divergence_bonus  # Very overbought
        elif latest['rsi'] > self.rsi_overbought_weak:
            rsi_signal = -1 + min(0, rsi_divergence_bonus)  # Overbought
        
        # Enhanced MACD signals with momentum and histogram
        macd_signal = 0
        macd_momentum = latest.get('macd_momentum', 0)
        
        if (latest['macd'] > latest['macd_signal'] and 
            latest['macd_hist'] > prev['macd_hist'] and 
            macd_momentum > 0):
            macd_signal = 2  # Strong bullish with momentum
        elif latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
            macd_signal = 1  # Bullish crossover
        elif (latest['macd'] < latest['macd_signal'] and 
              latest['macd_hist'] < prev['macd_hist'] and 
              macd_momentum < 0):
            macd_signal = -2  # Strong bearish with momentum
        elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
            macd_signal = -1  # Bearish crossover
        
        # Enhanced moving average signals with trend strength
        ma_signal = 0
        trend_strength = latest.get('trend_strength', 0.5)
        
        # Buy low: Price below short MA but above long-term trend
        if (latest['close'] < latest['sma_20'] and 
            latest['close'] > latest['sma_200'] and 
            latest['sma_20'] > latest['sma_50']):
            ma_signal = 1 + int(trend_strength > 0.7)  # Stronger signal with high trend strength
        # Sell high: Price above short MA in strong uptrend
        elif (latest['close'] > latest['sma_20'] > latest['sma_50'] > latest['sma_200']):
            ma_signal = -1 - int(trend_strength > 0.7)  # Stronger sell signal
        # Bearish trend
        elif latest['close'] < latest['sma_20'] < latest['sma_50'] < latest['sma_200']:
            ma_signal = -1 - int(trend_strength > 0.7)
        
        # Enhanced Bollinger Bands with Keltner Channel confirmation
        bb_signal = 0
        bb_width = latest.get('bb_width', 0)
        keltner_position = latest.get('keltner_position', 0.5)
        
        if (latest['close'] <= latest['bb_lower'] and 
            bb_width > self.bb_min_width and 
            keltner_position < 0.2):  # Confirmed by Keltner
            bb_signal = 2  # Very strong buy
        elif latest['bb_position'] < 0.2:  # Near lower band
            bb_signal = 1  # Buy signal
        elif (latest['close'] >= latest['bb_upper'] and 
              bb_width > self.bb_min_width and 
              keltner_position > 0.8):  # Confirmed by Keltner
            bb_signal = -2  # Very strong sell
        elif latest['bb_position'] > 0.8:  # Near upper band
            bb_signal = -1  # Sell signal
        
        # Enhanced Stochastic with momentum
        stoch_signal = 0
        stoch_momentum = latest.get('stoch_momentum', 0)
        
        if (latest['stoch_k'] < self.stoch_oversold_strong and 
            latest['stoch_d'] < self.stoch_oversold_strong and 
            stoch_momentum > 0):  # Bullish momentum
            stoch_signal = 2
        elif latest['stoch_k'] < self.stoch_oversold_weak:
            stoch_signal = 1
        elif (latest['stoch_k'] > self.stoch_overbought_strong and 
              latest['stoch_d'] > self.stoch_overbought_strong and 
              stoch_momentum < 0):  # Bearish momentum
            stoch_signal = -2
        elif latest['stoch_k'] > self.stoch_overbought_weak:
            stoch_signal = -1
        
        # Enhanced price position with volume confirmation
        position_signal = 0
        volume_confirmation = latest.get('volume_ratio', 1) > 1.2  # Above average volume
        
        if latest['price_position'] < 0.15:  # Very low price position
            position_signal = 2 + int(volume_confirmation)  # Stronger with volume
        elif latest['price_position'] < 0.35:
            position_signal = 1 + int(volume_confirmation)
        elif latest['price_position'] > 0.85:  # Very high price position
            position_signal = -2 - int(volume_confirmation)  # Stronger with volume
        elif latest['price_position'] > 0.65:
            position_signal = -1 - int(volume_confirmation)
        
        # Advanced oscillator signals
        williams_signal = self._calculate_williams_signal(latest)
        cci_signal = self._calculate_cci_signal(latest)
        mfi_signal = self._calculate_mfi_signal(latest)
        
        # New advanced signals
        adx_signal = self._calculate_adx_signal(latest)
        aroon_signal = self._calculate_aroon_signal(latest)
        psar_signal = self._calculate_psar_signal(latest, prev)
        vwap_signal = self._calculate_vwap_signal(latest)
        pattern_signal = self._calculate_pattern_signal(latest)
        
        # Market structure signal
        structure_signal = self._calculate_structure_signal(latest)
        
        # Volume analysis signal
        volume_signal = self._calculate_volume_signal(latest)
        
        return {
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'ma_signal': ma_signal,
            'bb_signal': bb_signal,
            'stoch_signal': stoch_signal,
            'position_signal': position_signal,  # Key for buy low, sell high
            'williams_signal': williams_signal,
            'cci_signal': cci_signal,
            'mfi_signal': mfi_signal,
            'adx_signal': adx_signal,
            'aroon_signal': aroon_signal,
            'psar_signal': psar_signal,
            'vwap_signal': vwap_signal,
            'pattern_signal': pattern_signal,
            'structure_signal': structure_signal,
            'volume_signal': volume_signal,
            'price_position': latest['price_position'],
            'trend_strength': latest.get('trend_strength', 0.5),
            'volatility_ratio': latest.get('volatility_ratio', 1.0)
        }
    
    def _calculate_williams_signal(self, latest) -> int:
        """Calculate Williams %R signal"""
        if latest['williams_r'] < self.williams_r_oversold_strong:
            return 2  # Very oversold
        elif latest['williams_r'] < self.williams_r_oversold_weak:
            return 1  # Oversold
        elif latest['williams_r'] > self.williams_r_overbought_strong:
            return -2  # Very overbought
        elif latest['williams_r'] > self.williams_r_overbought_weak:
            return -1  # Overbought
        return 0
    
    def _calculate_cci_signal(self, latest) -> int:
        """Calculate CCI signal"""
        if latest['cci'] < self.cci_oversold_strong:
            return 2  # Very oversold
        elif latest['cci'] < self.cci_oversold_weak:
            return 1  # Oversold
        elif latest['cci'] > self.cci_overbought_strong:
            return -2  # Very overbought
        elif latest['cci'] > self.cci_overbought_weak:
            return -1  # Overbought
        return 0
    
    def _calculate_mfi_signal(self, latest) -> int:
        """Calculate Money Flow Index signal"""
        if latest['mfi'] < self.mfi_oversold:
            return 1  # Oversold
        elif latest['mfi'] > self.mfi_overbought:
            return -1  # Overbought
        return 0
    
    def _calculate_adx_signal(self, latest) -> int:
        """Calculate ADX trend strength signal"""
        adx = latest.get('adx', 25)
        plus_di = latest.get('plus_di', 25)
        minus_di = latest.get('minus_di', 25)
        
        if adx > 25:  # Strong trend
            if plus_di > minus_di:
                return 1  # Strong uptrend
            else:
                return -1  # Strong downtrend
        return 0  # Weak trend
    
    def _calculate_aroon_signal(self, latest) -> int:
        """Calculate Aroon oscillator signal"""
        aroon_osc = latest.get('aroon_osc', 0)
        
        if aroon_osc > 50:
            return 1  # Bullish
        elif aroon_osc < -50:
            return -1  # Bearish
        return 0
    
    def _calculate_psar_signal(self, latest, prev) -> int:
        """Calculate Parabolic SAR signal"""
        psar_trend = latest.get('psar_trend', 0)
        prev_psar_trend = prev.get('psar_trend', 0)
        
        if psar_trend == 1 and prev_psar_trend == -1:
            return 1  # Bullish reversal
        elif psar_trend == -1 and prev_psar_trend == 1:
            return -1  # Bearish reversal
        return 0
    
    def _calculate_vwap_signal(self, latest) -> int:
        """Calculate VWAP signal"""
        vwap = latest.get('vwap', latest['close'])
        
        if latest['close'] > vwap * 1.01:
            return -1  # Price above VWAP - potential sell
        elif latest['close'] < vwap * 0.99:
            return 1  # Price below VWAP - potential buy
        return 0
    
    def _calculate_pattern_signal(self, latest) -> int:
        """Calculate candlestick pattern signal"""
        pattern = latest.get('patterns', 'none')
        
        bullish_patterns = ['bullish_hammer', 'bullish_engulfing', 'morning_star']
        bearish_patterns = ['bearish_engulfing', 'evening_star']
        
        if pattern in bullish_patterns:
            return 1
        elif pattern in bearish_patterns:
            return -1
        return 0
    
    def _calculate_structure_signal(self, latest) -> int:
        """Calculate market structure signal"""
        structure = latest.get('market_structure', 'sideways')
        
        if structure == 'uptrend':
            return 1
        elif structure == 'downtrend':
            return -1
        return 0
    
    def _calculate_volume_signal(self, latest) -> int:
        """Calculate volume-based signal"""
        volume_ratio = latest.get('volume_ratio', 1)
        obv_trend = latest.get('obv', 0)  # Would need to calculate trend
        
        if volume_ratio > 1.5:  # High volume
            # High volume supports the price movement
            if latest['close'] > latest.get('vwap', latest['close']):
                return 1  # High volume upward movement
            else:
                return -1  # High volume downward movement
        return 0
    
    def generate_trading_strategy(self, symbol: str, df: pd.DataFrame, 
                                news_data: List[str] = None, use_llm: bool = False) -> Dict:
        """Generate comprehensive enhanced trading strategy"""
        # Calculate enhanced technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Detect market regime
        market_regime = self.detect_market_regime(df)
        
        # Get adaptive strategy configuration
        strategy_config = self.adaptive_strategy_selection(df, market_regime)
        
        # Get enhanced technical analysis signals
        technical_signals = self.calculate_technical_signals(df)
        
        # Get LLM sentiment analysis (optional for speed)
        llm_analysis = {"sentiment_score": 0.0, "confidence": 0.0, "recommendation": "hold"}
        if use_llm and news_data and self.enable_llm:
            print("🧠 Running LLM sentiment analysis...")
            llm_analysis = self.analyze_market_sentiment(symbol, news_data)
        
        # Combine signals with enhanced logic
        strategy = self._combine_signals(df, technical_signals, llm_analysis)
        
        # Add regime and configuration information
        strategy.update({
            'market_regime': market_regime,
            'strategy_config': strategy_config,
            'enhanced_features': {
                'regime_detection': True,
                'pattern_recognition': True,
                'multi_timeframe': True,
                'volume_analysis': True,
                'volatility_adjustment': True,
                'dynamic_thresholds': True
            }
        })
        
        # Calculate dynamic position sizing
        if 'account_balance' in strategy:  # If account balance is available
            position_info = self.calculate_dynamic_position_size(
                strategy['account_balance'],
                strategy['confidence'],
                strategy['volatility_ratio'],
                strategy['risk_level']
            )
            strategy['position_sizing'] = position_info
        
        return strategy
    
    def analyze_symbol(self, symbol: str, market_data_manager=None, use_llm: bool = None) -> Dict:
        """Analyze a symbol with enhanced strategy (backward compatibility method)"""
        if use_llm is None:
            use_llm = not self.enable_llm  # Use opposite of fast mode
        
        try:
            # Get market data (this would need to be passed in or retrieved)
            if market_data_manager:
                df = market_data_manager.get_historical_data(symbol)
            else:
                # Fallback - create dummy data structure for testing
                print(f"⚠️ No market data manager provided for {symbol}")
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'error': 'No market data available'
                }
            
            # Generate strategy
            strategy = self.generate_trading_strategy(symbol, df, use_llm=use_llm)
            
            return strategy
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _combine_signals(self, df: pd.DataFrame, technical_signals: Dict, 
                        llm_analysis: Dict) -> Dict:
        """Enhanced signal combination for buy low, sell high strategy with advanced features"""
        latest = df.iloc[-1]
        
        # Dynamic weight adjustment based on market conditions
        volatility_ratio = technical_signals.get('volatility_ratio', 1.0)
        trend_strength = technical_signals.get('trend_strength', 0.5)
        
        # Adjust weights based on market volatility and trend strength
        base_weights = {
            'llm': 0.25,
            'position_signal': 0.20,  # Core buy low, sell high signal
            'rsi_signal': 0.12,
            'bb_signal': 0.12,
            'williams_signal': 0.08,
            'cci_signal': 0.08,
            'stoch_signal': 0.06,
            'mfi_signal': 0.05,
            'macd_signal': 0.04,
            'ma_signal': 0.04,
            'adx_signal': 0.03,
            'aroon_signal': 0.03,
            'psar_signal': 0.03,
            'vwap_signal': 0.03,
            'pattern_signal': 0.04,
            'structure_signal': 0.04,
            'volume_signal': 0.05
        }
        
        # Adjust weights for high volatility (favor mean reversion)
        if volatility_ratio > 1.5:
            base_weights['position_signal'] *= 1.3
            base_weights['bb_signal'] *= 1.2
            base_weights['rsi_signal'] *= 1.2
        
        # Adjust weights for strong trends (favor trend following)
        if trend_strength > 0.7:
            base_weights['ma_signal'] *= 1.4
            base_weights['adx_signal'] *= 1.3
            base_weights['structure_signal'] *= 1.2
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        weights = {k: v/total_weight for k, v in base_weights.items()}
        
        # Calculate weighted technical score with all signals
        technical_score = sum(
            weights.get(signal_name, 0) * technical_signals.get(signal_name, 0)
            for signal_name in weights.keys() if signal_name != 'llm'
        )
        
        # Convert LLM recommendation to numeric with confidence weighting
        llm_score = {
            'buy': 1,
            'sell': -1,
            'hold': 0
        }.get(llm_analysis['recommendation'], 0)
        
        # Combined score with confidence weighting
        combined_score = (
            weights['llm'] * llm_score * llm_analysis['confidence'] +
            technical_score
        )
        
        # Advanced decision logic with multiple confirmation layers
        price_position = technical_signals.get('price_position', 0.5)
        
        # Multi-factor confirmation system
        buy_factors = self._calculate_buy_factors(technical_signals, latest)
        sell_factors = self._calculate_sell_factors(technical_signals, latest)
        
        # Risk-adjusted confidence calculation
        base_confidence = min(abs(combined_score), 1.0)
        
        # Confidence adjustments
        confidence_adjustments = []
        
        # Volume confirmation
        if technical_signals.get('volume_signal', 0) != 0:
            confidence_adjustments.append(0.1)
        
        # Pattern confirmation
        if technical_signals.get('pattern_signal', 0) != 0:
            confidence_adjustments.append(0.1)
        
        # Multiple oscillator agreement
        oscillator_signals = [
            technical_signals.get('rsi_signal', 0),
            technical_signals.get('williams_signal', 0),
            technical_signals.get('cci_signal', 0),
            technical_signals.get('stoch_signal', 0)
        ]
        
        if len([s for s in oscillator_signals if s > 0]) >= 3:  # 3+ bullish oscillators
            confidence_adjustments.append(0.15)
        elif len([s for s in oscillator_signals if s < 0]) >= 3:  # 3+ bearish oscillators
            confidence_adjustments.append(0.15)
        
        # Trend alignment bonus
        if (technical_signals.get('structure_signal', 0) > 0 and 
            technical_signals.get('ma_signal', 0) > 0):
            confidence_adjustments.append(0.1)
        elif (technical_signals.get('structure_signal', 0) < 0 and 
              technical_signals.get('ma_signal', 0) < 0):
            confidence_adjustments.append(0.1)
        
        # Apply confidence adjustments
        final_confidence = min(base_confidence + sum(confidence_adjustments), 1.0)
        
        # Enhanced decision thresholds with dynamic adjustment
        strong_threshold = 0.5 - (volatility_ratio - 1) * 0.1  # Lower threshold in high volatility
        moderate_threshold = 0.2 - (volatility_ratio - 1) * 0.05
        
        # Generate final recommendation with enhanced logic
        if buy_factors >= 3 or combined_score > strong_threshold:
            action = 'BUY'
            confidence = max(final_confidence, 0.7)  # Minimum confidence for strong signals
        elif sell_factors >= 3 or combined_score < -strong_threshold:
            action = 'SELL'
            confidence = max(final_confidence, 0.7)
        elif combined_score > moderate_threshold:
            action = 'BUY'
            confidence = min(final_confidence, 0.65)  # Cap moderate signals
        elif combined_score < -moderate_threshold:
            action = 'SELL'
            confidence = min(final_confidence, 0.65)
        else:
            action = 'HOLD'
            confidence = max(0.2, 0.5 - abs(combined_score))  # Higher confidence for neutral signals
        
        # Risk assessment
        risk_level = self._assess_risk_level(technical_signals, volatility_ratio, trend_strength)
        
        # Calculate stop loss and take profit levels
        stop_loss, take_profit = self._calculate_levels(latest, action, technical_signals)
        
        return {
            'action': action,
            'confidence': confidence,
            'combined_score': combined_score,
            'technical_score': technical_score,
            'llm_analysis': llm_analysis,
            'technical_signals': technical_signals,
            'buy_factors': buy_factors,
            'sell_factors': sell_factors,
            'risk_level': risk_level,
            'current_price': latest['close'],
            'price_position': price_position,
            'support_level': latest.get('support', latest['close'] * 0.95),
            'resistance_level': latest.get('resistance', latest['close'] * 1.05),
            'volume_support': latest.get('volume_support', latest['close'] * 0.95),
            'volume_resistance': latest.get('volume_resistance', latest['close'] * 1.05),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'volatility_ratio': volatility_ratio,
            'trend_strength': trend_strength,
            'rsi': latest['rsi'],
            'rsi_short': latest.get('rsi_short', latest['rsi']),
            'rsi_long': latest.get('rsi_long', latest['rsi']),
            'macd': latest['macd'],
            'bb_position': latest.get('bb_position', 0.5),
            'keltner_position': latest.get('keltner_position', 0.5),
            'adx': latest.get('adx', 25),
            'vwap': latest.get('vwap', latest['close']),
            'market_structure': latest.get('market_structure', 'sideways'),
            'candlestick_pattern': latest.get('patterns', 'none'),
            'strategy_type': 'enhanced_buy_low_sell_high',
            'timestamp': datetime.now().isoformat(),
            'signal_weights': weights
        }
    
    def _calculate_buy_factors(self, signals: Dict, latest) -> int:
        """Calculate number of bullish factors present"""
        factors = 0
        
        # Price position factor
        if signals.get('position_signal', 0) >= 1:
            factors += 1
        
        # Oversold oscillators
        oversold_count = sum(1 for signal in ['rsi_signal', 'williams_signal', 'cci_signal', 'stoch_signal'] 
                           if signals.get(signal, 0) >= 1)
        if oversold_count >= 2:
            factors += 1
        
        # Bollinger Bands support
        if signals.get('bb_signal', 0) >= 1:
            factors += 1
        
        # Volume confirmation
        if signals.get('volume_signal', 0) > 0:
            factors += 1
        
        # Pattern confirmation
        if signals.get('pattern_signal', 0) > 0:
            factors += 1
        
        # Trend alignment (buying in uptrend)
        if (signals.get('structure_signal', 0) > 0 and 
            signals.get('ma_signal', 0) >= 0):
            factors += 1
        
        return factors
    
    def _calculate_sell_factors(self, signals: Dict, latest) -> int:
        """Calculate number of bearish factors present"""
        factors = 0
        
        # Price position factor
        if signals.get('position_signal', 0) <= -1:
            factors += 1
        
        # Overbought oscillators
        overbought_count = sum(1 for signal in ['rsi_signal', 'williams_signal', 'cci_signal', 'stoch_signal'] 
                             if signals.get(signal, 0) <= -1)
        if overbought_count >= 2:
            factors += 1
        
        # Bollinger Bands resistance
        if signals.get('bb_signal', 0) <= -1:
            factors += 1
        
        # Volume confirmation
        if signals.get('volume_signal', 0) < 0:
            factors += 1
        
        # Pattern confirmation
        if signals.get('pattern_signal', 0) < 0:
            factors += 1
        
        # Trend alignment (selling in downtrend)
        if (signals.get('structure_signal', 0) < 0 and 
            signals.get('ma_signal', 0) <= 0):
            factors += 1
        
        return factors
    
    def _assess_risk_level(self, signals: Dict, volatility_ratio: float, trend_strength: float) -> str:
        """Assess overall risk level of the trade"""
        risk_score = 0
        
        # Volatility risk
        if volatility_ratio > 2.0:
            risk_score += 2
        elif volatility_ratio > 1.5:
            risk_score += 1
        
        # Trend uncertainty
        if trend_strength < 0.3:
            risk_score += 1
        
        # Conflicting signals
        signal_values = [v for k, v in signals.items() if k.endswith('_signal')]
        positive_signals = sum(1 for v in signal_values if v > 0)
        negative_signals = sum(1 for v in signal_values if v < 0)
        
        if abs(positive_signals - negative_signals) < 2:  # Conflicting signals
            risk_score += 1
        
        # Market structure risk
        if signals.get('structure_signal', 0) == 0:  # Sideways market
            risk_score += 1
        
        if risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_levels(self, latest, action: str, signals: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        current_price = latest['close']
        atr = latest.get('atr', current_price * 0.02)
        
        if action == 'BUY':
            # Stop loss below support or using ATR
            support = latest.get('support', current_price * 0.95)
            stop_loss = min(support * 0.98, current_price - (atr * 2))
            
            # Take profit at resistance or using risk-reward ratio
            resistance = latest.get('resistance', current_price * 1.05)
            risk_amount = current_price - stop_loss
            take_profit = max(resistance * 1.02, current_price + (risk_amount * 2))
            
        else:  # SELL
            # Stop loss above resistance or using ATR
            resistance = latest.get('resistance', current_price * 1.05)
            stop_loss = max(resistance * 1.02, current_price + (atr * 2))
            
            # Take profit at support or using risk-reward ratio
            support = latest.get('support', current_price * 0.95)
            risk_amount = stop_loss - current_price
            take_profit = min(support * 0.98, current_price - (risk_amount * 2))
        
        return stop_loss, take_profit
    
    def detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Detect current market regime for adaptive strategy selection"""
        if len(df) < 50:
            return {
                'regime': 'unknown',
                'confidence': 0.5,
                'volatility': 'medium',
                'trend_strength': 0.5,
                'volume_profile': 'normal'
            }
        
        latest = df.iloc[-1]
        recent_data = df.tail(20)
        
        # Volatility analysis
        volatility = recent_data['close'].std() / recent_data['close'].mean()
        if volatility > 0.05:
            volatility_level = 'high'
        elif volatility > 0.02:
            volatility_level = 'medium'
        else:
            volatility_level = 'low'
        
        # Trend analysis
        trend_strength = latest.get('trend_strength', 0.5)
        adx = latest.get('adx', 25)
        
        # Price momentum
        price_change_20 = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        # Volume analysis
        avg_volume = recent_data['volume'].mean()
        recent_volume = recent_data['volume'].tail(5).mean()
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.5:
            volume_profile = 'high'
        elif volume_ratio > 0.8:
            volume_profile = 'normal'
        else:
            volume_profile = 'low'
        
        # Regime classification
        if adx > 30 and abs(price_change_20) > 0.1:
            if price_change_20 > 0:
                regime = 'bull_trend'
                confidence = min(adx / 50 + abs(price_change_20), 1.0)
            else:
                regime = 'bear_trend'
                confidence = min(adx / 50 + abs(price_change_20), 1.0)
        elif volatility_level == 'high' and adx < 25:
            regime = 'volatile_sideways'
            confidence = volatility * 10
        elif volatility_level == 'low' and adx < 20:
            regime = 'quiet_sideways'
            confidence = 0.7
        else:
            regime = 'transitional'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': min(confidence, 1.0),
            'volatility': volatility_level,
            'trend_strength': trend_strength,
            'volume_profile': volume_profile,
            'price_momentum': price_change_20,
            'adx': adx,
            'volatility_ratio': volatility
        }
    
    def adaptive_strategy_selection(self, df: pd.DataFrame, market_regime: Dict) -> Dict:
        """Select optimal strategy based on market regime"""
        regime = market_regime['regime']
        
        strategy_config = {
            'bull_trend': {
                'strategy_type': 'trend_following',
                'buy_threshold': 0.3,
                'sell_threshold': 0.6,
                'position_sizing': 'aggressive',
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 3.0
            },
            'bear_trend': {
                'strategy_type': 'short_bias',
                'buy_threshold': 0.7,
                'sell_threshold': 0.2,
                'position_sizing': 'conservative',
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 2.0
            },
            'volatile_sideways': {
                'strategy_type': 'mean_reversion',
                'buy_threshold': 0.15,
                'sell_threshold': 0.85,
                'position_sizing': 'moderate',
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 1.5
            },
            'quiet_sideways': {
                'strategy_type': 'range_trading',
                'buy_threshold': 0.25,
                'sell_threshold': 0.75,
                'position_sizing': 'moderate',
                'stop_loss_multiplier': 1.8,
                'take_profit_multiplier': 2.0
            },
            'transitional': {
                'strategy_type': 'conservative',
                'buy_threshold': 0.4,
                'sell_threshold': 0.6,
                'position_sizing': 'conservative',
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0
            }
        }
        
        return strategy_config.get(regime, strategy_config['transitional'])
    
    def calculate_dynamic_position_size(self, account_balance: float, confidence: float, 
                                      volatility_ratio: float, risk_level: str) -> Dict:
        """Calculate dynamic position size based on multiple factors"""
        base_risk = self.max_position_risk
        
        # Adjust risk based on confidence
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2.0x
        
        # Adjust for volatility
        if volatility_ratio > 2.0:
            volatility_multiplier = 0.5
        elif volatility_ratio > 1.5:
            volatility_multiplier = 0.7
        elif volatility_ratio < 0.5:
            volatility_multiplier = 1.2
        else:
            volatility_multiplier = 1.0
        
        # Adjust for risk level
        risk_multipliers = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.6
        }
        risk_multiplier = risk_multipliers.get(risk_level, 1.0)
        
        # Calculate final position size
        adjusted_risk = base_risk * confidence_multiplier * volatility_multiplier * risk_multiplier
        adjusted_risk = max(0.005, min(adjusted_risk, 0.05))  # Cap between 0.5% and 5%
        
        position_value = account_balance * adjusted_risk
        
        return {
            'risk_percentage': adjusted_risk * 100,
            'position_value': position_value,
            'confidence_multiplier': confidence_multiplier,
            'volatility_multiplier': volatility_multiplier,
            'risk_multiplier': risk_multiplier,
            'max_position_value': account_balance * 0.1  # Never exceed 10% of account
        }
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 0.02,
                              stop_loss_pct: float = 0.05) -> float:
        """Calculate position size based on risk management"""
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        return min(position_size, account_balance * 0.1)  # Max 10% of balance per trade