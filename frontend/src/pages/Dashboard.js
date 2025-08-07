import React, { useState, useEffect } from 'react';
import { Grid, Box, Typography, Alert, CircularProgress, FormControl, InputLabel, Select, MenuItem, Card, CardContent } from '@mui/material';
import { 
  AccountBalanceWallet as WalletIcon,
  TrendingUp as TrendingUpIcon,
  PriceChange as PriceIcon,
  Settings as SettingsIcon,
  FilterAlt as FilterIcon
} from '@mui/icons-material';

import StatusCard from '../components/StatusCard';
import CryptoPrice from '../components/CryptoPrice';
import TradeControls from '../components/TradeControls';
import PortfolioSummary from '../components/PortfolioSummary';
import RecentLogs from '../components/RecentLogs';
import { tradingService } from '../services/apiService';

export default function Dashboard() {
  const [status, setStatus] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tradingPairs, setTradingPairs] = useState([]);
  const [selectedPairs, setSelectedPairs] = useState([]);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Fetch status data
      const statusResponse = await tradingService.getStatus();
      if (statusResponse.data && statusResponse.data.success) {
        setStatus(statusResponse.data.status);
        setTradingPairs(statusResponse.data.status.trading_pairs || []);
      }
      
      // Fetch portfolio data
      const portfolioResponse = await tradingService.getPortfolio();
      if (portfolioResponse.data && portfolioResponse.data.success) {
        setPortfolio(portfolioResponse.data.portfolio);
      }
      
      setError(null);
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
      setError('Failed to load dashboard data. Please check your connection to the backend server.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000);
    
    return () => clearInterval(interval);
  }, []);

  // Set selectedPairs to all trading pairs when tradingPairs change
  useEffect(() => {
    setSelectedPairs(tradingPairs);
  }, [tradingPairs]);
  
  const handlePairSelection = (event) => {
    setSelectedPairs(event.target.value);
  };

  const handleTradeAction = async (action, data) => {
    // Refresh data after trade action
    fetchData();
  };

  if (loading && !status) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <>
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap">
        <Typography variant="h5" gutterBottom>
          Trading Dashboard
        </Typography>
        
        <FormControl sx={{ minWidth: 200, mb: 1 }} size="small">
          <InputLabel id="trading-pairs-label">Trading Pairs</InputLabel>
          <Select
            labelId="trading-pairs-label"
            multiple
            value={selectedPairs}
            onChange={handlePairSelection}
            renderValue={(selected) => selected.join(', ')}
            label="Trading Pairs"
          >
            {tradingPairs.map((pair) => (
              <MenuItem key={pair} value={pair}>
                {pair}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {/* Status Cards */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} lg={3}>
          <StatusCard
            title="Account Balance"
            value={`$${Number(status?.balance || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}`}
            icon={<WalletIcon />}
            color="primary.main"
            subValue="Available for trading"
          />
        </Grid>
        <Grid item xs={12} sm={6} lg={3}>
          <StatusCard
            title="Trading Status"
            value={status?.auto_trading ? "Auto Trading" : "Paper Trading"}
            icon={<SettingsIcon />}
            color={status?.auto_trading ? "success.main" : "warning.main"}
            chipText={status?.is_running ? "Running" : "Stopped"}
            chipColor={status?.is_running ? "success" : "error"}
          />
        </Grid>
        <Grid item xs={12} sm={6} lg={3}>
          <StatusCard
            title="Daily Trades"
            value={`${status?.daily_trade_count || 0}/${status?.max_daily_trades || 10}`}
            icon={<PriceIcon />}
            color="info.main"
            subValue="Trades executed today"
          />
        </Grid>
        <Grid item xs={12} sm={6} lg={3}>
          <StatusCard
            title="Min. Confidence"
            value={`${(status?.min_confidence_threshold || 0) * 100}%`}
            icon={<TrendingUpIcon />}
            color="secondary.main"
            subValue="Required for trade execution"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Price Cards */}
        {selectedPairs.length > 0 ? (
          selectedPairs.map((pair) => (
            <Grid item xs={12} sm={6} md={3} key={pair}>
              <CryptoPrice symbol={pair} />
            </Grid>
          ))
        ) : (
          <Grid item xs={12}>
            <Card sx={{ boxShadow: 3 }}>
              <CardContent>
                <Box display="flex" alignItems="center">
                  <FilterIcon sx={{ mr: 1 }} color="action" />
                  <Typography color="text.secondary">
                    Please select trading pairs to display
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        )}
        
        {/* Main content row */}
        <Grid item xs={12} md={4}>
          <PortfolioSummary portfolio={portfolio} loading={loading} />
        </Grid>
        
        {/* Trading Controls */}
        <Grid item xs={12} md={8}>
          <TradeControls status={status} onAction={handleTradeAction} />
        </Grid>
        
        {/* Logs */}
        <Grid item xs={12}>
          <RecentLogs />
        </Grid>
      </Grid>
    </>
  );
}