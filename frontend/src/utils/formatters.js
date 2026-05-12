import { format, formatDistanceToNow, parseISO, isValid } from 'date-fns';
import numeral from 'numeral';

/**
 * Format a number as currency
 * @param {number} value - The value to format
 * @param {string} currency - Currency symbol (default: '$')
 * @returns {string} Formatted currency string
 */
export const formatCurrency = (value, currency = '$') => {
  if (value === null || value === undefined || isNaN(value)) return `${currency}0.00`;

  const absValue = Math.abs(value);
  let formatted;

  if (absValue >= 1000000) {
    formatted = numeral(value).format('$0.00a').toUpperCase();
  } else if (absValue >= 1000) {
    formatted = numeral(value).format('$0,0.00');
  } else if (absValue >= 1) {
    formatted = numeral(value).format('$0,0.00');
  } else if (absValue > 0) {
    formatted = numeral(value).format('$0,0.0000');
  } else {
    formatted = `${currency}0.00`;
  }

  return formatted.replace('$', currency);
};

/**
 * Format a number as percentage
 * @param {number} value - The value to format (0.05 = 5%)
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted percentage string
 */
export const formatPercent = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) return '0.00%';

  const percent = value * 100;
  return numeral(percent).format(`0,0.${'0'.repeat(decimals)}`) + '%';
};

/**
 * Format a number with appropriate precision
 * @param {number} value - The value to format
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted number string
 */
export const formatNumber = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) return '0';

  const absValue = Math.abs(value);

  if (absValue >= 1000000) {
    return numeral(value).format('0.00a').toUpperCase();
  } else if (absValue >= 1000) {
    return numeral(value).format('0,0');
  } else if (absValue >= 1) {
    return numeral(value).format(`0,0.${'0'.repeat(decimals)}`);
  } else {
    return numeral(value).format(`0.${'0'.repeat(decimals)}`);
  }
};

/**
 * Format a crypto amount with appropriate precision
 * @param {number} value - The crypto amount
 * @param {string} symbol - Crypto symbol (e.g., 'BTC', 'ETH')
 * @returns {string} Formatted crypto amount
 */
export const formatCrypto = (value, symbol = '') => {
  if (value === null || value === undefined || isNaN(value)) {
    return symbol ? `0 ${symbol}` : '0';
  }

  const absValue = Math.abs(value);
  let formatted;

  if (absValue >= 1) {
    formatted = numeral(value).format('0,0.00000000');
  } else {
    formatted = numeral(value).format('0.00000000');
  }

  // Remove trailing zeros
  formatted = formatted.replace(/\.?0+$/, '');

  return symbol ? `${formatted} ${symbol}` : formatted;
};

/**
 * Format a date in a human-readable format
 * @param {string|Date} date - The date to format
 * @param {string} formatString - The format string (default: 'MMM dd, yyyy HH:mm')
 * @returns {string} Formatted date string
 */
export const formatDate = (date, formatString = 'MMM dd, yyyy HH:mm') => {
  if (!date) return 'N/A';

  try {
    const parsedDate = typeof date === 'string' ? parseISO(date) : date;
    if (!isValid(parsedDate)) return 'Invalid Date';
    return format(parsedDate, formatString);
  } catch (error) {
    console.error('Error formatting date:', error);
    return 'Invalid Date';
  }
};

/**
 * Format a date as relative time (e.g., "2 hours ago")
 * @param {string|Date} date - The date to format
 * @returns {string} Relative time string
 */
export const formatRelativeTime = (date) => {
  if (!date) return 'N/A';

  try {
    const parsedDate = typeof date === 'string' ? parseISO(date) : date;
    if (!isValid(parsedDate)) return 'Invalid Date';
    return formatDistanceToNow(parsedDate, { addSuffix: true });
  } catch (error) {
    console.error('Error formatting relative time:', error);
    return 'Invalid Date';
  }
};

/**
 * Format a timestamp in short format
 * @param {string|Date} date - The date to format
 * @returns {string} Short timestamp (HH:mm:ss)
 */
export const formatTimestamp = (date) => {
  return formatDate(date, 'HH:mm:ss');
};

/**
 * Format a trade side with color indicator
 * @param {string} side - Trade side ('BUY' or 'SELL')
 * @returns {object} Object with text and color
 */
export const formatTradeSide = (side) => {
  const upperSide = side?.toUpperCase();
  return {
    text: upperSide,
    color: upperSide === 'BUY' ? 'success' : 'error'
  };
};

/**
 * Format profit/loss with sign and color
 * @param {number} value - P&L value
 * @param {boolean} includeSign - Whether to include + sign for positive values
 * @returns {object} Object with text and color
 */
export const formatPnL = (value, includeSign = true) => {
  if (value === null || value === undefined || isNaN(value)) {
    return { text: '$0.00', color: 'text.secondary' };
  }

  const formatted = formatCurrency(Math.abs(value));
  const sign = value >= 0 ? '+' : '-';

  return {
    text: includeSign ? `${sign}${formatted}` : formatted,
    color: value >= 0 ? 'success.main' : 'error.main'
  };
};

/**
 * Format percentage change with sign and color
 * @param {number} value - Percentage value (0.05 = 5%)
 * @returns {object} Object with text and color
 */
export const formatPercentChange = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return { text: '0.00%', color: 'text.secondary' };
  }

  const formatted = formatPercent(Math.abs(value));
  const sign = value >= 0 ? '+' : '-';

  return {
    text: `${sign}${formatted}`,
    color: value >= 0 ? 'success.main' : 'error.main'
  };
};

/**
 * Format a confidence score
 * @param {number} value - Confidence value (0-1)
 * @returns {object} Object with text and color
 */
export const formatConfidence = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return { text: 'N/A', color: 'text.secondary' };
  }

  const percent = formatPercent(value);
  let color;

  if (value >= 0.8) {
    color = 'success.main';
  } else if (value >= 0.6) {
    color = 'info.main';
  } else if (value >= 0.4) {
    color = 'warning.main';
  } else {
    color = 'error.main';
  }

  return { text: percent, color };
};

/**
 * Format file size in human-readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
};

/**
 * Truncate text with ellipsis
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength = 50) => {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - 3) + '...';
};

/**
 * Format a trading pair
 * @param {string} pair - Trading pair (e.g., 'BTC-USDT')
 * @returns {object} Object with base and quote currencies
 */
export const formatTradingPair = (pair) => {
  if (!pair) return { base: '', quote: '', display: '' };

  const parts = pair.split('-');
  return {
    base: parts[0] || '',
    quote: parts[1] || '',
    display: pair
  };
};
