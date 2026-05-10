import { useQuery, useQueries } from '@tanstack/react-query';
import axios from 'axios';
import { API_BASE_URL, REFRESH_INTERVALS } from '../utils/constants';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000
});

/**
 * Fetch current price for a trading pair
 */
const fetchPrice = async (pair) => {
  const { data } = await api.get(`/api/market/price/${pair}`);
  return data;
};

/**
 * Fetch OHLCV data for a trading pair
 */
const fetchOHLCV = async (pair, timeframe = '1h', limit = 100) => {
  const { data } = await api.get(`/api/market/ohlcv/${pair}`, {
    params: { timeframe, limit }
  });
  return data;
};

/**
 * Fetch ticker data for a trading pair
 */
const fetchTicker = async (pair) => {
  const { data } = await api.get(`/api/market/ticker/${pair}`);
  return data;
};

/**
 * Fetch orderbook for a trading pair
 */
const fetchOrderbook = async (pair, depth = 20) => {
  const { data } = await api.get(`/api/market/orderbook/${pair}`, {
    params: { depth }
  });
  return data;
};

/**
 * Fetch recent trades for a trading pair
 */
const fetchRecentTrades = async (pair, limit = 50) => {
  const { data } = await api.get(`/api/market/trades/${pair}`, {
    params: { limit }
  });
  return data;
};

/**
 * Hook to fetch current price for a trading pair
 */
export const usePrice = (pair, options = {}) => {
  return useQuery({
    queryKey: ['price', pair],
    queryFn: () => fetchPrice(pair),
    enabled: !!pair,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.FAST,
    staleTime: 3000,
    ...options
  });
};

/**
 * Hook to fetch OHLCV data for a trading pair
 */
export const useOHLCV = (pair, timeframe = '1h', limit = 100, options = {}) => {
  return useQuery({
    queryKey: ['ohlcv', pair, timeframe, limit],
    queryFn: () => fetchOHLCV(pair, timeframe, limit),
    enabled: !!pair,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.NORMAL,
    staleTime: 5000,
    ...options
  });
};

/**
 * Hook to fetch ticker data for a trading pair
 */
export const useTicker = (pair, options = {}) => {
  return useQuery({
    queryKey: ['ticker', pair],
    queryFn: () => fetchTicker(pair),
    enabled: !!pair,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.FAST,
    staleTime: 3000,
    ...options
  });
};

/**
 * Hook to fetch orderbook for a trading pair
 */
export const useOrderbook = (pair, depth = 20, options = {}) => {
  return useQuery({
    queryKey: ['orderbook', pair, depth],
    queryFn: () => fetchOrderbook(pair, depth),
    enabled: !!pair,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.FAST,
    staleTime: 2000,
    ...options
  });
};

/**
 * Hook to fetch recent trades for a trading pair
 */
export const useRecentTrades = (pair, limit = 50, options = {}) => {
  return useQuery({
    queryKey: ['recentTrades', pair, limit],
    queryFn: () => fetchRecentTrades(pair, limit),
    enabled: !!pair,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.NORMAL,
    staleTime: 5000,
    ...options
  });
};


/**
 * Hook to fetch prices for multiple pairs
 */
export const usePrices = (pairs = [], options = {}) => {
  return useQueries({
    queries: pairs.map(pair => ({
      queryKey: ['price', pair],
      queryFn: () => fetchPrice(pair),
      enabled: !!pair,
      refetchInterval: options.refetchInterval || REFRESH_INTERVALS.FAST,
      staleTime: 3000,
      ...options
    }))
  });
};

/**
 * Hook to fetch tickers for multiple pairs
 */
export const useTickers = (pairs = [], options = {}) => {
  return useQueries({
    queries: pairs.map(pair => ({
      queryKey: ['ticker', pair],
      queryFn: () => fetchTicker(pair),
      enabled: !!pair,
      refetchInterval: options.refetchInterval || REFRESH_INTERVALS.FAST,
      staleTime: 3000,
      ...options
    }))
  });
};

/**
 * Hook to fetch complete market data for a pair
 */
export const useMarketData = (pair, options = {}) => {
  const price = usePrice(pair, { enabled: options.includePrice !== false });
  const ticker = useTicker(pair, { enabled: options.includeTicker !== false });
  const ohlcv = useOHLCV(
    pair,
    options.timeframe || '1h',
    options.limit || 100,
    { enabled: options.includeOHLCV !== false }
  );
  const orderbook = useOrderbook(pair, options.depth || 20, {
    enabled: options.includeOrderbook === true
  });
  const recentTrades = useRecentTrades(pair, options.tradesLimit || 50, {
    enabled: options.includeRecentTrades === true
  });

  const isLoading =
    price.isLoading ||
    ticker.isLoading ||
    ohlcv.isLoading ||
    (options.includeOrderbook && orderbook.isLoading) ||
    (options.includeRecentTrades && recentTrades.isLoading);

  const isError =
    price.isError ||
    ticker.isError ||
    ohlcv.isError ||
    (options.includeOrderbook && orderbook.isError) ||
    (options.includeRecentTrades && recentTrades.isError);

  const refetch = () => {
    price.refetch();
    ticker.refetch();
    ohlcv.refetch();
    if (options.includeOrderbook) orderbook.refetch();
    if (options.includeRecentTrades) recentTrades.refetch();
  };

  return {
    price: price.data,
    ticker: ticker.data,
    ohlcv: ohlcv.data,
    orderbook: orderbook.data,
    recentTrades: recentTrades.data,
    isLoading,
    isError,
    refetch,
    queries: {
      price,
      ticker,
      ohlcv,
      orderbook,
      recentTrades
    }
  };
};

export default useMarketData;
