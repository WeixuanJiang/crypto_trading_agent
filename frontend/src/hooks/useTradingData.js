import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { API_BASE_URL, REFRESH_INTERVALS } from '../utils/constants';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000
});

/**
 * Fetch trading history
 */
const fetchTrades = async (params = {}) => {
  const { data } = await api.get('/api/trades', { params });
  return data;
};

/**
 * Fetch a specific trade by ID
 */
const fetchTrade = async (tradeId) => {
  const { data } = await api.get(`/api/trades/${tradeId}`);
  return data;
};

/**
 * Fetch portfolio data
 */
const fetchPortfolio = async () => {
  const { data } = await api.get('/api/portfolio');
  return data;
};

/**
 * Fetch performance metrics
 */
const fetchPerformance = async (params = {}) => {
  const { data } = await api.get('/api/performance', { params });
  return data;
};

/**
 * Fetch open positions
 */
const fetchOpenPositions = async () => {
  const { data } = await api.get('/api/positions/open');
  return data;
};

/**
 * Fetch settings
 */
const fetchSettings = async () => {
  const { data } = await api.get('/api/settings');
  return data;
};

/**
 * Update settings
 */
const updateSettings = async (settings) => {
  const { data } = await api.post('/api/settings/update', settings);
  return data;
};

/**
 * Execute a trade
 */
const executeTrade = async (tradeData) => {
  const { data } = await api.post('/api/trades/execute', tradeData);
  return data;
};

/**
 * Close a position
 */
const closePosition = async (positionId) => {
  const { data } = await api.post(`/api/positions/${positionId}/close`);
  return data;
};

/**
 * Fetch system status
 */
const fetchSystemStatus = async () => {
  const { data } = await api.get('/api/status');
  return data;
};

/**
 * Hook to fetch trading history
 */
export const useTrades = (params = {}, options = {}) => {
  return useQuery({
    queryKey: ['trades', params],
    queryFn: () => fetchTrades(params),
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.NORMAL,
    staleTime: 10000,
    ...options
  });
};

/**
 * Hook to fetch a specific trade
 */
export const useTrade = (tradeId, options = {}) => {
  return useQuery({
    queryKey: ['trade', tradeId],
    queryFn: () => fetchTrade(tradeId),
    enabled: !!tradeId,
    staleTime: 30000,
    ...options
  });
};

/**
 * Hook to fetch portfolio data
 */
export const usePortfolio = (options = {}) => {
  return useQuery({
    queryKey: ['portfolio'],
    queryFn: fetchPortfolio,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.NORMAL,
    staleTime: 10000,
    ...options
  });
};

/**
 * Hook to fetch performance metrics
 */
export const usePerformance = (params = {}, options = {}) => {
  return useQuery({
    queryKey: ['performance', params],
    queryFn: () => fetchPerformance(params),
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.SLOW,
    staleTime: 30000,
    ...options
  });
};

/**
 * Hook to fetch open positions
 */
export const useOpenPositions = (options = {}) => {
  return useQuery({
    queryKey: ['openPositions'],
    queryFn: fetchOpenPositions,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.NORMAL,
    staleTime: 10000,
    ...options
  });
};

/**
 * Hook to fetch settings
 */
export const useSettings = (options = {}) => {
  return useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
    staleTime: 60000,
    ...options
  });
};

/**
 * Hook to update settings
 */
export const useUpdateSettings = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: updateSettings,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      return data;
    }
  });
};

/**
 * Hook to execute a trade
 */
export const useExecuteTrade = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: executeTrade,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trades'] });
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      queryClient.invalidateQueries({ queryKey: ['openPositions'] });
      queryClient.invalidateQueries({ queryKey: ['performance'] });
    }
  });
};

/**
 * Hook to close a position
 */
export const useClosePosition = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: closePosition,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trades'] });
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      queryClient.invalidateQueries({ queryKey: ['openPositions'] });
      queryClient.invalidateQueries({ queryKey: ['performance'] });
    }
  });
};

/**
 * Hook to fetch system status
 */
export const useSystemStatus = (options = {}) => {
  return useQuery({
    queryKey: ['systemStatus'],
    queryFn: fetchSystemStatus,
    refetchInterval: options.refetchInterval || REFRESH_INTERVALS.SLOW,
    staleTime: 30000,
    ...options
  });
};

/**
 * Hook to fetch all trading data
 */
export const useTradingData = (options = {}) => {
  const trades = useTrades({}, { enabled: options.includeTrades !== false });
  const portfolio = usePortfolio({ enabled: options.includePortfolio !== false });
  const performance = usePerformance({}, { enabled: options.includePerformance !== false });
  const openPositions = useOpenPositions({ enabled: options.includeOpenPositions !== false });
  const settings = useSettings({ enabled: options.includeSettings !== false });
  const systemStatus = useSystemStatus({ enabled: options.includeSystemStatus !== false });

  const isLoading =
    trades.isLoading ||
    portfolio.isLoading ||
    performance.isLoading ||
    openPositions.isLoading ||
    settings.isLoading ||
    systemStatus.isLoading;

  const isError =
    trades.isError ||
    portfolio.isError ||
    performance.isError ||
    openPositions.isError ||
    settings.isError ||
    systemStatus.isError;

  const refetchAll = () => {
    trades.refetch();
    portfolio.refetch();
    performance.refetch();
    openPositions.refetch();
    settings.refetch();
    systemStatus.refetch();
  };

  return {
    trades: trades.data,
    portfolio: portfolio.data,
    performance: performance.data,
    openPositions: openPositions.data,
    settings: settings.data,
    systemStatus: systemStatus.data,
    isLoading,
    isError,
    refetchAll,
    queries: {
      trades,
      portfolio,
      performance,
      openPositions,
      settings,
      systemStatus
    }
  };
};

/**
 * Hook for real-time trading statistics
 */
export const useTradingStats = (options = {}) => {
  const { data: trades } = useTrades({}, options);
  const { data: performance } = usePerformance({}, options);
  const { data: portfolio } = usePortfolio(options);

  const stats = {
    totalTrades: trades?.total || 0,
    openTrades: trades?.open || 0,
    closedTrades: trades?.closed || 0,
    winRate: performance?.win_rate || 0,
    totalPnL: performance?.total_pnl || 0,
    portfolioValue: portfolio?.total_value || 0,
    availableBalance: portfolio?.available_balance || 0,
    sharpeRatio: performance?.sharpe_ratio || 0,
    maxDrawdown: performance?.max_drawdown || 0
  };

  return {
    stats,
    isLoading: !trades || !performance || !portfolio
  };
};

export default useTradingData;
