import { useEffect, useRef, useCallback, useState } from 'react';
import useWebSocketLib from 'react-use-websocket';
import { WS_BASE_URL, WS_EVENTS } from '../utils/constants';

/**
 * Custom hook for WebSocket connection management
 * @param {string} endpoint - WebSocket endpoint (e.g., '/ws/prices')
 * @param {Object} options - Configuration options
 * @returns {Object} WebSocket state and methods
 */
const useWebSocket = (endpoint, options = {}) => {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    shouldConnect = true,
    protocols,
    share = false
  } = options;

  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [lastMessage, setLastMessage] = useState(null);
  const [messageHistory, setMessageHistory] = useState([]);
  const reconnectCount = useRef(0);

  const url = endpoint ? `${WS_BASE_URL}${endpoint}` : null;

  const {
    sendMessage,
    sendJsonMessage,
    lastMessage: wsLastMessage,
    readyState,
    getWebSocket
  } = useWebSocketLib(
    url,
    {
      onOpen: (event) => {
        console.log(`WebSocket connected: ${endpoint}`);
        setConnectionStatus('connected');
        reconnectCount.current = 0;
        onOpen?.(event);
      },
      onClose: (event) => {
        console.log(`WebSocket disconnected: ${endpoint}`);
        setConnectionStatus('disconnected');
        onClose?.(event);

        // Handle reconnection
        if (reconnectCount.current < reconnectAttempts) {
          reconnectCount.current += 1;
          console.log(`Reconnecting... (${reconnectCount.current}/${reconnectAttempts})`);
          setTimeout(() => {
            setConnectionStatus('reconnecting');
          }, reconnectInterval);
        } else {
          setConnectionStatus('failed');
        }
      },
      onError: (event) => {
        console.error(`WebSocket error: ${endpoint}`, event);
        setConnectionStatus('error');
        onError?.(event);
      },
      shouldReconnect: (closeEvent) => {
        return reconnectCount.current < reconnectAttempts;
      },
      reconnectAttempts,
      reconnectInterval,
      share,
      protocols
    },
    shouldConnect
  );

  // Process incoming messages
  useEffect(() => {
    if (wsLastMessage) {
      try {
        const data = JSON.parse(wsLastMessage.data);
        setLastMessage(data);
        setMessageHistory(prev => [...prev.slice(-99), data]); // Keep last 100 messages
        onMessage?.(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
        setLastMessage(wsLastMessage.data);
        setMessageHistory(prev => [...prev.slice(-99), wsLastMessage.data]);
        onMessage?.(wsLastMessage.data);
      }
    }
  }, [wsLastMessage, onMessage]);

  // Convert readyState to string status
  useEffect(() => {
    const statusMap = {
      0: 'connecting',
      1: 'connected',
      2: 'closing',
      3: 'disconnected'
    };
    setConnectionStatus(statusMap[readyState] || 'unknown');
  }, [readyState]);

  const clearHistory = useCallback(() => {
    setMessageHistory([]);
  }, []);

  const isConnected = readyState === 1;
  const isConnecting = readyState === 0;
  const isDisconnected = readyState === 3;

  return {
    sendMessage,
    sendJsonMessage,
    lastMessage,
    messageHistory,
    connectionStatus,
    isConnected,
    isConnecting,
    isDisconnected,
    readyState,
    getWebSocket,
    clearHistory
  };
};

/**
 * Hook for subscribing to price updates
 */
export const usePriceUpdates = (tradingPairs = [], options = {}) => {
  const [prices, setPrices] = useState({});

  const handleMessage = useCallback((data) => {
    if (data.event === WS_EVENTS.PRICE_UPDATE && data.pair && data.price) {
      setPrices(prev => ({
        ...prev,
        [data.pair]: {
          price: data.price,
          change: data.change,
          changePercent: data.changePercent,
          timestamp: data.timestamp || new Date().toISOString()
        }
      }));
    }
  }, []);

  const ws = useWebSocket('/ws/prices', {
    ...options,
    onMessage: handleMessage,
    shouldConnect: false  // Disable WebSocket - backend doesn't support it yet
  });

  // Subscribe to trading pairs
  useEffect(() => {
    if (ws.isConnected && tradingPairs.length > 0) {
      ws.sendJsonMessage({
        event: 'subscribe',
        pairs: tradingPairs
      });
    }
  }, [ws.isConnected, tradingPairs, ws]);

  return {
    prices,
    ...ws
  };
};

/**
 * Hook for subscribing to trade updates
 */
export const useTradeUpdates = (options = {}) => {
  const [trades, setTrades] = useState([]);
  const [latestTrade, setLatestTrade] = useState(null);

  const handleMessage = useCallback((data) => {
    if (data.event === WS_EVENTS.TRADE_UPDATE && data.trade) {
      setLatestTrade(data.trade);
      setTrades(prev => [data.trade, ...prev.slice(0, 99)]); // Keep last 100 trades
    }
  }, []);

  const ws = useWebSocket('/ws/trades', {
    ...options,
    onMessage: handleMessage
  });

  const clearTrades = useCallback(() => {
    setTrades([]);
    setLatestTrade(null);
  }, []);

  return {
    trades,
    latestTrade,
    clearTrades,
    ...ws
  };
};

/**
 * Hook for subscribing to portfolio updates
 */
export const usePortfolioUpdates = (options = {}) => {
  const [portfolio, setPortfolio] = useState(null);

  const handleMessage = useCallback((data) => {
    if (data.event === WS_EVENTS.PORTFOLIO_UPDATE && data.portfolio) {
      setPortfolio(data.portfolio);
    }
  }, []);

  const ws = useWebSocket('/ws/portfolio', {
    ...options,
    onMessage: handleMessage
  });

  return {
    portfolio,
    ...ws
  };
};

/**
 * Hook for subscribing to market data
 */
export const useMarketDataStream = (pair, timeframe = '1m', options = {}) => {
  const [candles, setCandles] = useState([]);
  const [latestCandle, setLatestCandle] = useState(null);

  const handleMessage = useCallback((data) => {
    if (data.event === WS_EVENTS.MARKET_DATA && data.candle) {
      setLatestCandle(data.candle);
      setCandles(prev => {
        const existing = prev.findIndex(c => c.timestamp === data.candle.timestamp);
        if (existing >= 0) {
          // Update existing candle
          const updated = [...prev];
          updated[existing] = data.candle;
          return updated;
        } else {
          // Add new candle
          return [...prev, data.candle].slice(-500); // Keep last 500 candles
        }
      });
    }
  }, []);

  const ws = useWebSocket('/ws/market', {
    ...options,
    onMessage: handleMessage
  });

  // Subscribe to pair and timeframe
  useEffect(() => {
    if (ws.isConnected && pair && timeframe) {
      ws.sendJsonMessage({
        event: 'subscribe',
        pair,
        timeframe
      });
    }
  }, [ws.isConnected, pair, timeframe, ws]);

  const clearCandles = useCallback(() => {
    setCandles([]);
    setLatestCandle(null);
  }, []);

  return {
    candles,
    latestCandle,
    clearCandles,
    ...ws
  };
};

/**
 * Hook for subscribing to system status
 */
export const useSystemStatus = (options = {}) => {
  const [status, setStatus] = useState(null);

  const handleMessage = useCallback((data) => {
    if (data.event === WS_EVENTS.SYSTEM_STATUS && data.status) {
      setStatus(data.status);
    }
  }, []);

  const ws = useWebSocket('/ws/status', {
    ...options,
    onMessage: handleMessage
  });

  return {
    status,
    ...ws
  };
};

export default useWebSocket;
