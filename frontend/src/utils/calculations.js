/**
 * Calculate profit and loss
 * @param {number} entryPrice - Entry price
 * @param {number} exitPrice - Exit price
 * @param {number} quantity - Trade quantity
 * @param {string} side - Trade side ('BUY' or 'SELL')
 * @returns {number} P&L value
 */
export const calculatePnL = (entryPrice, exitPrice, quantity, side) => {
  if (!entryPrice || !exitPrice || !quantity) return 0;

  const priceDiff = exitPrice - entryPrice;
  const multiplier = side?.toUpperCase() === 'BUY' ? 1 : -1;

  return priceDiff * quantity * multiplier;
};

/**
 * Calculate percentage profit and loss
 * @param {number} entryPrice - Entry price
 * @param {number} exitPrice - Exit price
 * @param {string} side - Trade side ('BUY' or 'SELL')
 * @returns {number} P&L percentage (0.05 = 5%)
 */
export const calculatePnLPercent = (entryPrice, exitPrice, side) => {
  if (!entryPrice || !exitPrice) return 0;

  const priceDiff = exitPrice - entryPrice;
  const multiplier = side?.toUpperCase() === 'BUY' ? 1 : -1;

  return (priceDiff / entryPrice) * multiplier;
};

/**
 * Calculate return on investment (ROI)
 * @param {number} initialValue - Initial investment
 * @param {number} finalValue - Final value
 * @returns {number} ROI percentage (0.05 = 5%)
 */
export const calculateROI = (initialValue, finalValue) => {
  if (!initialValue || initialValue === 0) return 0;
  return (finalValue - initialValue) / initialValue;
};

/**
 * Calculate win rate
 * @param {number} wins - Number of winning trades
 * @param {number} total - Total number of trades
 * @returns {number} Win rate percentage (0.05 = 5%)
 */
export const calculateWinRate = (wins, total) => {
  if (!total || total === 0) return 0;
  return wins / total;
};

/**
 * Calculate Sharpe Ratio
 * @param {number[]} returns - Array of returns
 * @param {number} riskFreeRate - Risk-free rate (default: 0.02 = 2%)
 * @returns {number} Sharpe ratio
 */
export const calculateSharpeRatio = (returns, riskFreeRate = 0.02) => {
  if (!returns || returns.length === 0) return 0;

  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);

  if (stdDev === 0) return 0;

  // Annualize assuming daily returns
  const annualizedReturn = avgReturn * 252;
  const annualizedStdDev = stdDev * Math.sqrt(252);

  return (annualizedReturn - riskFreeRate) / annualizedStdDev;
};

/**
 * Calculate maximum drawdown
 * @param {number[]} equityCurve - Array of portfolio values over time
 * @returns {number} Maximum drawdown percentage (negative value, e.g., -0.15 = -15%)
 */
export const calculateMaxDrawdown = (equityCurve) => {
  if (!equityCurve || equityCurve.length === 0) return 0;

  let maxDrawdown = 0;
  let peak = equityCurve[0];

  for (const value of equityCurve) {
    if (value > peak) {
      peak = value;
    }

    const drawdown = (value - peak) / peak;
    if (drawdown < maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return maxDrawdown;
};

/**
 * Calculate average trade duration
 * @param {Array} trades - Array of trade objects with entry_time and exit_time
 * @returns {number} Average duration in milliseconds
 */
export const calculateAvgTradeDuration = (trades) => {
  if (!trades || trades.length === 0) return 0;

  const durations = trades
    .filter(t => t.entry_time && t.exit_time)
    .map(t => new Date(t.exit_time) - new Date(t.entry_time));

  if (durations.length === 0) return 0;

  return durations.reduce((sum, d) => sum + d, 0) / durations.length;
};

/**
 * Calculate profit factor
 * @param {Array} trades - Array of trade objects with pnl field
 * @returns {number} Profit factor (gross profit / gross loss)
 */
export const calculateProfitFactor = (trades) => {
  if (!trades || trades.length === 0) return 0;

  const grossProfit = trades.reduce((sum, t) => sum + (t.pnl > 0 ? t.pnl : 0), 0);
  const grossLoss = Math.abs(trades.reduce((sum, t) => sum + (t.pnl < 0 ? t.pnl : 0), 0));

  if (grossLoss === 0) return grossProfit > 0 ? Infinity : 0;

  return grossProfit / grossLoss;
};

/**
 * Calculate position size based on risk
 * @param {number} accountBalance - Total account balance
 * @param {number} riskPercent - Risk percentage per trade (0.02 = 2%)
 * @param {number} entryPrice - Entry price
 * @param {number} stopLoss - Stop loss price
 * @returns {number} Position size in base currency units
 */
export const calculatePositionSize = (accountBalance, riskPercent, entryPrice, stopLoss) => {
  if (!accountBalance || !riskPercent || !entryPrice || !stopLoss) return 0;

  const riskAmount = accountBalance * riskPercent;
  const priceRisk = Math.abs(entryPrice - stopLoss);

  if (priceRisk === 0) return 0;

  return riskAmount / priceRisk;
};

/**
 * Calculate expected value
 * @param {number} winRate - Win rate (0.6 = 60%)
 * @param {number} avgWin - Average winning trade amount
 * @param {number} avgLoss - Average losing trade amount (positive value)
 * @returns {number} Expected value per trade
 */
export const calculateExpectedValue = (winRate, avgWin, avgLoss) => {
  if (winRate === null || avgWin === null || avgLoss === null) return 0;

  const lossRate = 1 - winRate;
  return (winRate * avgWin) - (lossRate * avgLoss);
};

/**
 * Calculate compound annual growth rate (CAGR)
 * @param {number} beginValue - Beginning portfolio value
 * @param {number} endValue - Ending portfolio value
 * @param {number} years - Number of years
 * @returns {number} CAGR percentage (0.15 = 15%)
 */
export const calculateCAGR = (beginValue, endValue, years) => {
  if (!beginValue || !endValue || !years || beginValue === 0) return 0;

  return Math.pow(endValue / beginValue, 1 / years) - 1;
};

/**
 * Calculate moving average
 * @param {number[]} data - Array of numbers
 * @param {number} period - Period for moving average
 * @returns {number[]} Array of moving average values
 */
export const calculateMovingAverage = (data, period) => {
  if (!data || data.length < period) return [];

  const result = [];
  for (let i = period - 1; i < data.length; i++) {
    const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
    result.push(sum / period);
  }

  return result;
};

/**
 * Calculate exponential moving average (EMA)
 * @param {number[]} data - Array of numbers
 * @param {number} period - Period for EMA
 * @returns {number[]} Array of EMA values
 */
export const calculateEMA = (data, period) => {
  if (!data || data.length === 0) return [];

  const multiplier = 2 / (period + 1);
  const result = [];

  // Start with SMA
  const firstSum = data.slice(0, period).reduce((a, b) => a + b, 0);
  result.push(firstSum / period);

  // Calculate EMA for remaining values
  for (let i = period; i < data.length; i++) {
    const ema = (data[i] - result[result.length - 1]) * multiplier + result[result.length - 1];
    result.push(ema);
  }

  return result;
};

/**
 * Calculate RSI (Relative Strength Index)
 * @param {number[]} prices - Array of prices
 * @param {number} period - Period for RSI (default: 14)
 * @returns {number[]} Array of RSI values (0-100)
 */
export const calculateRSI = (prices, period = 14) => {
  if (!prices || prices.length < period + 1) return [];

  const changes = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  const result = [];
  let avgGain = 0;
  let avgLoss = 0;

  // Calculate initial averages
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) {
      avgGain += changes[i];
    } else {
      avgLoss += Math.abs(changes[i]);
    }
  }

  avgGain /= period;
  avgLoss /= period;

  // Calculate RSI for remaining values
  for (let i = period; i < changes.length; i++) {
    const rs = avgGain / (avgLoss || 1);
    const rsi = 100 - (100 / (1 + rs));
    result.push(rsi);

    // Update averages
    const change = changes[i];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = ((avgGain * (period - 1)) + gain) / period;
    avgLoss = ((avgLoss * (period - 1)) + loss) / period;
  }

  return result;
};

/**
 * Calculate volatility (standard deviation of returns)
 * @param {number[]} prices - Array of prices
 * @returns {number} Annualized volatility percentage
 */
export const calculateVolatility = (prices) => {
  if (!prices || prices.length < 2) return 0;

  // Calculate returns
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }

  // Calculate standard deviation
  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);

  // Annualize (assuming daily data)
  return stdDev * Math.sqrt(252);
};

/**
 * Calculate trade statistics
 * @param {Array} trades - Array of completed trades
 * @returns {Object} Trade statistics
 */
export const calculateTradeStats = (trades) => {
  if (!trades || trades.length === 0) {
    return {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      totalPnL: 0,
      avgWin: 0,
      avgLoss: 0,
      largestWin: 0,
      largestLoss: 0,
      profitFactor: 0,
      expectedValue: 0
    };
  }

  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl < 0);

  const totalPnL = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const avgWin = winningTrades.length > 0
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length > 0
    ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length)
    : 0;

  const largestWin = winningTrades.length > 0
    ? Math.max(...winningTrades.map(t => t.pnl))
    : 0;
  const largestLoss = losingTrades.length > 0
    ? Math.min(...losingTrades.map(t => t.pnl))
    : 0;

  const winRate = calculateWinRate(winningTrades.length, trades.length);
  const profitFactor = calculateProfitFactor(trades);
  const expectedValue = calculateExpectedValue(winRate, avgWin, avgLoss);

  return {
    totalTrades: trades.length,
    winningTrades: winningTrades.length,
    losingTrades: losingTrades.length,
    winRate,
    totalPnL,
    avgWin,
    avgLoss,
    largestWin,
    largestLoss,
    profitFactor,
    expectedValue
  };
};
