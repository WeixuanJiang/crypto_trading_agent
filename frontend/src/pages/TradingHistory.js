import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper,
  Tab,
  Tabs,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Divider
} from '@mui/material';
import { 
  BarChart as ChartIcon,
  History as HistoryIcon
} from '@mui/icons-material';

import TradeHistoryTable from '../components/TradeHistoryTable';
import PerformanceMetrics from '../components/PerformanceMetrics';
import { tradingService } from '../services/apiService';

export default function TradingHistory() {
  const [trades, setTrades] = useState([]);
  const [statistics, setStatistics] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tabIndex, setTabIndex] = useState(0);
  const [timeRange, setTimeRange] = useState(30);
  const [tradingPairs, setTradingPairs] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState('');

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Get trading pairs from API
      const statusResponse = await tradingService.getStatus();
      if (statusResponse.data && statusResponse.data.success) {
        setTradingPairs(statusResponse.data.status.trading_pairs || []);
      }
      
      // Get trading history
      const params = { days: timeRange };
      if (selectedSymbol) {
        params.symbol = selectedSymbol;
      }
      
      const response = await tradingService.getTradingHistory(params);
      if (response.data && response.data.success) {
        setTrades(response.data.trades || []);
        setStatistics(response.data.statistics || {});
        setError(null);
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err) {
      console.error('Failed to fetch trading history:', err);
      setError('Failed to load trading history. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [timeRange, selectedSymbol]);

  const handleTabChange = (event, newValue) => {
    setTabIndex(newValue);
  };

  const handleTimeRangeChange = (event) => {
    setTimeRange(event.target.value);
  };

  const handleSymbolChange = (event) => {
    setSelectedSymbol(event.target.value);
  };

  return (
    <>
      <Box mb={3}>
        <Typography variant="h5" gutterBottom>
          Trading History
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={tabIndex}
            onChange={handleTabChange}
            aria-label="trading history tabs"
            sx={{ px: 2, pt: 1 }}
          >
            <Tab icon={<HistoryIcon />} label="HISTORY" id="tab-0" />
            <Tab icon={<ChartIcon />} label="PERFORMANCE" id="tab-1" />
          </Tabs>
        </Box>
        
        <Box sx={{ p: 3 }}>
          <Grid container spacing={2} alignItems="center" sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel id="time-range-label">Time Range</InputLabel>
                <Select
                  labelId="time-range-label"
                  value={timeRange}
                  label="Time Range"
                  onChange={handleTimeRangeChange}
                >
                  <MenuItem value={7}>Last 7 Days</MenuItem>
                  <MenuItem value={14}>Last 14 Days</MenuItem>
                  <MenuItem value={30}>Last 30 Days</MenuItem>
                  <MenuItem value={90}>Last 90 Days</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel id="symbol-label">Trading Pair</InputLabel>
                <Select
                  labelId="symbol-label"
                  value={selectedSymbol}
                  label="Trading Pair"
                  onChange={handleSymbolChange}
                >
                  <MenuItem value="">All Pairs</MenuItem>
                  {tradingPairs.map((pair) => (
                    <MenuItem key={pair} value={pair}>{pair}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          
          <Divider sx={{ mb: 3 }} />

          {loading ? (
            <Box display="flex" justifyContent="center" py={5}>
              <CircularProgress />
            </Box>
          ) : trades.length === 0 ? (
            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              py={8}
            >
              <HistoryIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No Trading History Yet
              </Typography>
              <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ maxWidth: 400 }}>
                Your trading history will appear here once you start trading.
                Execute your first trade to see performance metrics and trade details.
              </Typography>
            </Box>
          ) : (
            <Box>
              {tabIndex === 0 && (
                <TradeHistoryTable trades={trades} statistics={statistics} />
              )}
              {tabIndex === 1 && (
                <PerformanceMetrics statistics={statistics} />
              )}
            </Box>
          )}
        </Box>
      </Paper>
    </>
  );
}