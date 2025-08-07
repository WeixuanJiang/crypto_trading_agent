import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  LinearProgress,
  Card,
  CardContent,
  Divider
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ShowChart as ShowChartIcon,
  AccountBalanceWallet as WalletIcon,
  Functions as FunctionsIcon
} from '@mui/icons-material';

const MetricCard = ({ title, value, icon, color = 'primary', subtitle = null }) => (
  <Card variant="outlined" sx={{ height: '100%' }}>
    <CardContent>
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {title}
          </Typography>
          <Typography variant="h5" component="div" fontWeight="500">
            {value}
          </Typography>
          {subtitle && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {subtitle}
            </Typography>
          )}
        </Box>
        <Box 
          sx={{ 
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: `${color}.main`,
            color: `${color}.contrastText`,
            borderRadius: '50%',
            width: 40,
            height: 40
          }}
        >
          {icon}
        </Box>
      </Box>
    </CardContent>
  </Card>
);

export default function PerformanceMetrics({ statistics }) {
  if (!statistics || Object.keys(statistics).length === 0) {
    return (
      <Box py={3} textAlign="center">
        <Typography variant="body1" color="text.secondary">
          No performance data available
        </Typography>
      </Box>
    );
  }

  // Format numbers
  const formatCurrency = (value) => {
    if (value === null || value === undefined) return '$0.00';
    return `$${Number(value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  };

  const formatPercent = (value) => {
    if (value === null || value === undefined) return '0%';
    return `${Number(value).toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1})}%`;
  };
  
  // Calculate winning percentage for progress bar
  const winRate = statistics.win_rate || 0;
  
  return (
    <Box>
      {/* Overview Section */}
      <Box mb={4}>
        <Typography variant="h6" gutterBottom>Performance Overview</Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={4}>
            <MetricCard 
              title="Total P&L" 
              value={formatCurrency(statistics.total_pnl)} 
              icon={<ShowChartIcon />} 
              color={statistics.total_pnl > 0 ? 'success' : statistics.total_pnl < 0 ? 'error' : 'primary'}
              subtitle={`${statistics.closed_trades || 0} closed positions`}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <MetricCard 
              title="Win Rate" 
              value={formatPercent(statistics.win_rate)} 
              icon={<TimelineIcon />} 
              subtitle={`${statistics.winning_trades || 0} wins / ${statistics.losing_trades || 0} losses`}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <MetricCard 
              title="Profit Factor" 
              value={statistics.profit_factor === "N/A" ? "N/A" : Number(statistics.profit_factor).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} 
              icon={<FunctionsIcon />} 
              color={statistics.profit_factor > 1 ? 'success' : 'warning'}
            />
          </Grid>
        </Grid>
      </Box>
      
      {/* Win Rate Progress Bar */}
      <Box mb={4}>
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Win/Loss Distribution
          </Typography>
          <Box display="flex" alignItems="center" mb={1}>
            <Box sx={{ flexGrow: 1 }}>
              <LinearProgress
                variant="determinate"
                value={winRate}
                sx={{
                  height: 20,
                  borderRadius: 1,
                  backgroundColor: 'error.light',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: 'success.main',
                    borderRadius: '4px 0 0 4px'
                  }
                }}
              />
            </Box>
          </Box>
          <Box display="flex" justifyContent="space-between">
            <Box display="flex" alignItems="center">
              <TrendingUpIcon color="success" fontSize="small" sx={{ mr: 0.5 }} />
              <Typography variant="body2">
                {statistics.winning_trades || 0} Wins ({formatPercent(statistics.win_rate)})
              </Typography>
            </Box>
            <Box display="flex" alignItems="center">
              <Typography variant="body2">
                {statistics.losing_trades || 0} Losses ({formatPercent(100 - statistics.win_rate)})
              </Typography>
              <TrendingDownIcon color="error" fontSize="small" sx={{ ml: 0.5 }} />
            </Box>
          </Box>
        </Paper>
      </Box>
      
      {/* Trade Metrics */}
      <Box mb={4}>
        <Typography variant="h6" gutterBottom>Trade Metrics</Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={3}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">Average Win</Typography>
              <Typography variant="h6" color="success.main">
                {formatCurrency(statistics.avg_win)}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">Average Loss</Typography>
              <Typography variant="h6" color="error.main">
                {formatCurrency(statistics.avg_loss)}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">Largest Win</Typography>
              <Typography variant="h6" color="success.main">
                {formatCurrency(statistics.largest_win)}
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">Largest Loss</Typography>
              <Typography variant="h6" color="error.main">
                {formatCurrency(statistics.largest_loss)}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </Box>
      
      {/* Activity Summary */}
      <Box>
        <Typography variant="h6" gutterBottom>Activity Summary</Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6} md={4}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">Total Trades</Typography>
              <Typography variant="h6">
                {statistics.total_trades || 0}
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">Executed:</Typography>
                <Typography variant="body2">{statistics.executed_trades || 0}</Typography>
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">Paper:</Typography>
                <Typography variant="body2">{statistics.paper_trades || 0}</Typography>
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">Trading Period</Typography>
              <Typography variant="h6">
                {statistics.period_days || 30} Days
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">Daily Average:</Typography>
                <Typography variant="body2">
                  {statistics.total_trades && statistics.period_days 
                    ? (statistics.total_trades / statistics.period_days).toFixed(1) 
                    : '0'} trades
                </Typography>
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="body2" color="text.secondary">Total Fees</Typography>
              <Typography variant="h6">
                {formatCurrency(statistics.total_fees)}
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body2" color="text.secondary">Avg Fee per Trade:</Typography>
                <Typography variant="body2">
                  {statistics.total_trades && statistics.total_fees 
                    ? formatCurrency(statistics.total_fees / statistics.total_trades)
                    : '$0.00'}
                </Typography>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
}