// API Configuration
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';
export const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:5001';

// Chart Timeframes
export const TIMEFRAMES = [
  { value: '1m', label: '1 Minute', seconds: 60 },
  { value: '5m', label: '5 Minutes', seconds: 300 },
  { value: '15m', label: '15 Minutes', seconds: 900 },
  { value: '30m', label: '30 Minutes', seconds: 1800 },
  { value: '1h', label: '1 Hour', seconds: 3600 },
  { value: '4h', label: '4 Hours', seconds: 14400 },
  { value: '1d', label: '1 Day', seconds: 86400 },
  { value: '1w', label: '1 Week', seconds: 604800 }
];

// Date Range Presets
export const DATE_RANGES = [
  { value: 'today', label: 'Today', days: 0 },
  { value: '7d', label: 'Last 7 Days', days: 7 },
  { value: '30d', label: 'Last 30 Days', days: 30 },
  { value: '90d', label: 'Last 90 Days', days: 90 },
  { value: 'ytd', label: 'Year to Date', days: null },
  { value: 'all', label: 'All Time', days: null }
];

// Trading Pairs
export const POPULAR_PAIRS = [
  'BTC-USDT',
  'ETH-USDT',
  'BNB-USDT',
  'SOL-USDT',
  'XRP-USDT',
  'ADA-USDT',
  'DOGE-USDT',
  'MATIC-USDT',
  'DOT-USDT',
  'AVAX-USDT'
];

// Chart Colors
export const CHART_COLORS = {
  primary: '#00d4ff',
  secondary: '#b388ff',
  success: '#00e676',
  error: '#ff1744',
  warning: '#ffab00',
  info: '#00b0ff',
  bull: '#00e676',
  bear: '#ff1744',
  volume: '#00b0ff',
  grid: '#1e2235',
  text: '#b0bec5'
};

// Chart Indicators
export const INDICATORS = [
  { value: 'sma', label: 'Simple Moving Average (SMA)', periods: [20, 50, 200] },
  { value: 'ema', label: 'Exponential Moving Average (EMA)', periods: [12, 26, 50] },
  { value: 'rsi', label: 'Relative Strength Index (RSI)', period: 14 },
  { value: 'macd', label: 'MACD', periods: [12, 26, 9] },
  { value: 'bb', label: 'Bollinger Bands', period: 20 },
  { value: 'volume', label: 'Volume', enabled: true }
];

// Trade Statuses
export const TRADE_STATUS = {
  PENDING: 'pending',
  OPEN: 'open',
  CLOSED: 'closed',
  CANCELLED: 'cancelled',
  FAILED: 'failed'
};

export const TRADE_STATUS_LABELS = {
  pending: 'Pending',
  open: 'Open',
  closed: 'Closed',
  cancelled: 'Cancelled',
  failed: 'Failed'
};

export const TRADE_STATUS_COLORS = {
  pending: 'warning',
  open: 'info',
  closed: 'success',
  cancelled: 'default',
  failed: 'error'
};

// Trade Sides
export const TRADE_SIDES = {
  BUY: 'BUY',
  SELL: 'SELL'
};

// Confidence Levels
export const CONFIDENCE_LEVELS = {
  VERY_HIGH: { min: 0.8, max: 1.0, label: 'Very High', color: 'success' },
  HIGH: { min: 0.6, max: 0.8, label: 'High', color: 'info' },
  MEDIUM: { min: 0.4, max: 0.6, label: 'Medium', color: 'warning' },
  LOW: { min: 0.2, max: 0.4, label: 'Low', color: 'error' },
  VERY_LOW: { min: 0.0, max: 0.2, label: 'Very Low', color: 'error' }
};

// Refresh Intervals (in milliseconds)
export const REFRESH_INTERVALS = {
  FAST: 5000,      // 5 seconds
  NORMAL: 10000,   // 10 seconds
  SLOW: 30000,     // 30 seconds
  VERY_SLOW: 60000 // 1 minute
};

// Pagination
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 25,
  PAGE_SIZE_OPTIONS: [10, 25, 50, 100],
  MAX_PAGE_SIZE: 100
};

// Chart Settings
export const CHART_SETTINGS = {
  DEFAULT_HEIGHT: 400,
  MOBILE_HEIGHT: 300,
  CANDLESTICK_HEIGHT: 500,
  INDICATOR_HEIGHT: 150,
  ANIMATION_DURATION: 300,
  TOOLTIP_DELAY: 0
};

// Risk Levels
export const RISK_LEVELS = {
  CONSERVATIVE: { value: 'conservative', label: 'Conservative', max_risk: 0.01, color: 'success' },
  MODERATE: { value: 'moderate', label: 'Moderate', max_risk: 0.02, color: 'info' },
  AGGRESSIVE: { value: 'aggressive', label: 'Aggressive', max_risk: 0.05, color: 'warning' },
  VERY_AGGRESSIVE: { value: 'very_aggressive', label: 'Very Aggressive', max_risk: 0.10, color: 'error' }
};

// Strategy Types
export const STRATEGY_TYPES = {
  TECHNICAL: 'technical',
  LLM: 'llm',
  HYBRID: 'hybrid'
};

export const STRATEGY_LABELS = {
  technical: 'Technical Analysis',
  llm: 'LLM-Powered',
  hybrid: 'Hybrid Strategy'
};

// LLM Models
export const LLM_MODELS = {
  BEDROCK_CLAUDE: 'bedrock_claude',
  OPENAI_GPT4: 'openai_gpt4'
};

export const LLM_MODEL_LABELS = {
  bedrock_claude: 'AWS Bedrock (Claude)',
  openai_gpt4: 'OpenAI GPT-4'
};

// Notification Types
export const NOTIFICATION_TYPES = {
  TRADE_EXECUTED: 'trade_executed',
  TRADE_CLOSED: 'trade_closed',
  RISK_ALERT: 'risk_alert',
  SYSTEM_ERROR: 'system_error',
  DAILY_SUMMARY: 'daily_summary'
};

// WebSocket Events
export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  PRICE_UPDATE: 'price_update',
  TRADE_UPDATE: 'trade_update',
  PORTFOLIO_UPDATE: 'portfolio_update',
  MARKET_DATA: 'market_data',
  SYSTEM_STATUS: 'system_status'
};

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your connection.',
  SERVER_ERROR: 'Server error. Please try again later.',
  INVALID_INPUT: 'Invalid input. Please check your data.',
  UNAUTHORIZED: 'Unauthorized. Please log in again.',
  NOT_FOUND: 'Resource not found.',
  RATE_LIMIT: 'Rate limit exceeded. Please try again later.',
  UNKNOWN: 'An unknown error occurred.'
};

// Success Messages
export const SUCCESS_MESSAGES = {
  SETTINGS_SAVED: 'Settings saved successfully',
  TRADE_EXECUTED: 'Trade executed successfully',
  CREDENTIALS_UPDATED: 'Credentials updated successfully',
  DATA_REFRESHED: 'Data refreshed successfully'
};

// Local Storage Keys
export const STORAGE_KEYS = {
  THEME_MODE: 'theme_mode',
  SELECTED_PAIRS: 'selected_pairs',
  CHART_SETTINGS: 'chart_settings',
  DASHBOARD_LAYOUT: 'dashboard_layout',
  TABLE_PREFERENCES: 'table_preferences'
};

// Table Column Widths
export const TABLE_COLUMNS = {
  TIMESTAMP: 180,
  PAIR: 120,
  SIDE: 80,
  PRICE: 120,
  QUANTITY: 120,
  PNL: 120,
  STATUS: 100,
  ACTIONS: 100
};

// Animation Variants (for Framer Motion)
export const ANIMATION_VARIANTS = {
  fadeIn: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 }
  },
  slideUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  },
  slideDown: {
    initial: { opacity: 0, y: -20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 20 }
  },
  scale: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 }
  }
};

// Breakpoints (matches MUI breakpoints)
export const BREAKPOINTS = {
  xs: 0,
  sm: 600,
  md: 900,
  lg: 1200,
  xl: 1536
};

// Z-Index Levels
export const Z_INDEX = {
  DRAWER: 1200,
  MODAL: 1300,
  SNACKBAR: 1400,
  TOOLTIP: 1500
};

// Default Settings
export const DEFAULT_SETTINGS = {
  trading_pairs: ['BTC-USDT'],
  min_confidence_threshold: 0.7,
  max_daily_trades: 10,
  trading_interval_minutes: 60,
  risk_level: 'moderate',
  strategy_type: 'hybrid',
  llm_model: 'bedrock_claude',
  enable_notifications: true,
  dark_mode: true
};

// Performance Metrics
export const PERFORMANCE_METRICS = [
  { key: 'total_pnl', label: 'Total P&L', format: 'currency' },
  { key: 'win_rate', label: 'Win Rate', format: 'percent' },
  { key: 'sharpe_ratio', label: 'Sharpe Ratio', format: 'number' },
  { key: 'max_drawdown', label: 'Max Drawdown', format: 'percent' },
  { key: 'profit_factor', label: 'Profit Factor', format: 'number' },
  { key: 'total_trades', label: 'Total Trades', format: 'number' },
  { key: 'avg_win', label: 'Avg Win', format: 'currency' },
  { key: 'avg_loss', label: 'Avg Loss', format: 'currency' }
];

// Export all as a single object for convenience
export default {
  API_BASE_URL,
  WS_BASE_URL,
  TIMEFRAMES,
  DATE_RANGES,
  POPULAR_PAIRS,
  CHART_COLORS,
  INDICATORS,
  TRADE_STATUS,
  TRADE_STATUS_LABELS,
  TRADE_STATUS_COLORS,
  TRADE_SIDES,
  CONFIDENCE_LEVELS,
  REFRESH_INTERVALS,
  PAGINATION,
  CHART_SETTINGS,
  RISK_LEVELS,
  STRATEGY_TYPES,
  STRATEGY_LABELS,
  LLM_MODELS,
  LLM_MODEL_LABELS,
  NOTIFICATION_TYPES,
  WS_EVENTS,
  ERROR_MESSAGES,
  SUCCESS_MESSAGES,
  STORAGE_KEYS,
  TABLE_COLUMNS,
  ANIMATION_VARIANTS,
  BREAKPOINTS,
  Z_INDEX,
  DEFAULT_SETTINGS,
  PERFORMANCE_METRICS
};
