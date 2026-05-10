import React, { useState, useEffect } from 'react';
import { Grid, Box, Typography, Alert, CircularProgress, FormControl, InputLabel, Select, MenuItem, Card, CardContent, Chip } from '@mui/material';
import {
  AccountBalanceWallet as WalletIcon,
  TrendingUp as TrendingUpIcon,
  PriceChange as PriceIcon,
  Settings as SettingsIcon,
  FilterAlt as FilterIcon,
  AccountBalance as PaperTradingIcon,
  MonetizationOn as LiveTradingIcon
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
      
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h5">
            Trading Dashboard
          </Typography>
          {status && (
            status.auto_trading ? (
              <Chip
                icon={<LiveTradingIcon />}
                label="LIVE TRADING"
                color="error"
                sx={{
                  fontWeight: 'bold',
                  fontSize: '0.875rem',
                  animation: 'pulse 2s infinite',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                    '100%': { opacity: 1 }
                  }
                }}
              />
            ) : (
              <Chip
                icon={<PaperTradingIcon />}
                label="PAPER TRADING"
                color="warning"
                sx={{
                  fontWeight: 'bold',
                  fontSize: '0.875rem'
                }}
              />
            )
          )}
        </Box>

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

      {/* Paper Trading Mode Info Banner */}
      {status && !status.auto_trading && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Paper Trading Mode:</strong> You are currently in simulation mode. All trades are virtual and no real money is at risk.
            Balances and positions shown are simulated based on your paper trading configuration.
            {status.fast_mode && <span> Running in <strong>Fast Mode</strong> (Technical Analysis only).</span>}
          </Typography>
        </Alert>
      )}

      {/* Live Trading Warning Banner */}
      {status && status.auto_trading && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>⚠️ LIVE TRADING ACTIVE:</strong> Real money is at risk! All trades will be executed with actual funds on your KuCoin account.
            Monitor your positions carefully and ensure you have proper risk management settings configured.
          </Typography>
        </Alert>
      )}

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