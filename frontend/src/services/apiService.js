import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false,
});

// Interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response || error.message);
    return Promise.reject(error);
  }
);

export const tradingService = {
  // Status endpoints
  getStatus: () => api.get('/api/status'),
  
  // Trading control endpoints
  startTrading: (data) => api.post('/api/trading/start', data),
  stopTrading: () => api.post('/api/trading/stop'),
  toggleAutoTrading: () => api.post('/api/trading/toggle'),
  
  // Trading history
  getTradingHistory: (params) => api.get('/api/trading/history', { params }),
  
  // Analysis
  runAnalysis: (data = {}) => axios({
    method: 'post',
    url: `${API_URL}/api/analysis/run`,
    data: data,
    headers: {
      'Content-Type': 'application/json'
    },
    timeout: 300000  // 5 minutes timeout specifically for analysis
  }),
  analyzeSymbol: (symbol) => axios({
    method: 'post',
    url: `${API_URL}/api/analysis/run`,
    data: { symbol },
    headers: {
      'Content-Type': 'application/json'
    },
    timeout: 300000  // 5 minutes timeout specifically for symbol analysis
  }),
  
  // Portfolio
  getPortfolio: () => api.get('/api/portfolio'),
  
  // Market data
  getMarketPrice: (symbol) => api.get(`/api/market/price/${symbol}`),
  
  // Settings
  updateSettings: (settings) => api.post('/api/settings/update', settings),
  
  // Logs
  getLogs: (limit) => api.get('/api/logs', { params: { limit } }),
};

export default api;