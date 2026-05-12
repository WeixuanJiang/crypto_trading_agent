import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Box,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  TrendingUp as BuyIcon,
  TrendingDown as SellIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import axios from 'axios';
import { API_BASE_URL } from '../utils/constants';
import { formatCurrency } from '../utils/formatters';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Trading Controls Component
 */
export default function TradingControls({ status, onRefresh }) {
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);
  const [manualTradeDialog, setManualTradeDialog] = useState(false);

  const isRunning = status?.is_running || false;
  const isAutoTrading = status?.auto_trading || false;

  const handleStartTrading = async () => {
    try {
      setLoading(true);
      const response = await api.post('/api/trading/start', {});

      if (response.data.success) {
        enqueueSnackbar('AI Trading started successfully', { variant: 'success' });
        onRefresh?.();
      } else {
        enqueueSnackbar(response.data.message || 'Failed to start trading', { variant: 'error' });
      }
    } catch (error) {
      console.error('Error starting trading:', error);
      enqueueSnackbar('Error starting trading: ' + (error.response?.data?.error || error.message), {
        variant: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStopTrading = async () => {
    try {
      setLoading(true);
      const response = await api.post('/api/trading/stop', {});

      if (response.data.success) {
        enqueueSnackbar('Trading stopped successfully', { variant: 'info' });
        onRefresh?.();
      } else {
        enqueueSnackbar(response.data.message || 'Failed to stop trading', { variant: 'error' });
      }
    } catch (error) {
      console.error('Error stopping trading:', error);
      enqueueSnackbar('Error stopping trading: ' + (error.response?.data?.error || error.message), {
        variant: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRunAnalysis = async () => {
    try {
      setLoading(true);
      const response = await api.post('/api/analysis/run', {});

      if (response.data.success) {
        enqueueSnackbar('Analysis completed successfully', { variant: 'success' });
        onRefresh?.();
      } else {
        enqueueSnackbar(response.data.message || 'Analysis failed', { variant: 'error' });
      }
    } catch (error) {
      console.error('Error running analysis:', error);
      enqueueSnackbar('Error running analysis: ' + (error.response?.data?.error || error.message), {
        variant: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h6" fontWeight={600}>
              AI Trading Controls
            </Typography>
            <Tooltip title="Refresh status">
              <IconButton onClick={onRefresh} size="small" disabled={loading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>

          {/* Status Display */}
          <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
            <Box>
              <Typography variant="caption" color="text.secondary" display="block">
                Trading Status
              </Typography>
              <Chip
                label={isRunning ? 'RUNNING' : 'STOPPED'}
                color={isRunning ? 'success' : 'default'}
                size="small"
                sx={{ mt: 0.5, fontWeight: 600 }}
              />
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary" display="block">
                Mode
              </Typography>
              <Chip
                label={isAutoTrading ? 'LIVE TRADING' : 'PAPER TRADING'}
                color={isAutoTrading ? 'error' : 'warning'}
                size="small"
                sx={{ mt: 0.5, fontWeight: 600 }}
              />
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary" display="block">
                Daily Trades
              </Typography>
              <Typography variant="body2" fontWeight={600} sx={{ mt: 0.5 }}>
                {status?.daily_trade_count || 0} / {status?.max_daily_trades || 10}
              </Typography>
            </Box>
          </Stack>

          {/* Warning for Live Trading */}
          {isAutoTrading && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <strong>Live Trading Active!</strong> Real money is at risk. Monitor your positions carefully.
            </Alert>
          )}

          {/* Control Buttons */}
          <Stack spacing={2}>
            {!isRunning ? (
              <Button
                variant="contained"
                color="success"
                size="large"
                startIcon={loading ? <CircularProgress size={20} /> : <StartIcon />}
                onClick={handleStartTrading}
                disabled={loading}
                fullWidth
              >
                {loading ? 'Starting...' : 'Start AI Trading'}
              </Button>
            ) : (
              <Button
                variant="contained"
                color="error"
                size="large"
                startIcon={loading ? <CircularProgress size={20} /> : <StopIcon />}
                onClick={handleStopTrading}
                disabled={loading}
                fullWidth
              >
                {loading ? 'Stopping...' : 'Stop Trading'}
              </Button>
            )}

            <Button
              variant="outlined"
              size="large"
              startIcon={loading ? <CircularProgress size={20} /> : <RefreshIcon />}
              onClick={handleRunAnalysis}
              disabled={loading || isRunning}
              fullWidth
            >
              {loading ? 'Running...' : 'Run Analysis Now'}
            </Button>
          </Stack>

          {/* Info Box */}
          <Box
            sx={{
              mt: 3,
              p: 2,
              borderRadius: 2,
              bgcolor: 'action.hover',
              display: 'flex',
              gap: 1
            }}
          >
            <InfoIcon fontSize="small" color="info" />
            <Typography variant="caption" color="text.secondary">
              {isRunning ? (
                <>AI is actively analyzing markets and will execute trades based on your settings.
                Use "Stop Trading" to pause all automated activity.</>
              ) : (
                <>Click "Start AI Trading" to begin automated trading. The AI will analyze market
                conditions every {status?.trading_interval_minutes || 15} minutes and execute trades
                when confidence threshold is met.</>
              )}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </>
  );
}
