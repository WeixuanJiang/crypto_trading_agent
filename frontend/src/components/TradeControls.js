import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  FormControlLabel,
  Switch,
  Alert,
  CircularProgress,
  CardActions,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Paper,
  Grid,
  Chip,
  Avatar
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Autorenew as AutorenewIcon,
  BarChart as AnalyzeIcon,
  ExpandMore as ExpandMoreIcon,
  ArrowUpward as ArrowUpIcon,
  ArrowDownward as ArrowDownIcon,
  HorizontalRule as HoldIcon,
  Favorite as SentimentPositiveIcon,
  SentimentNeutral as SentimentNeutralIcon,
  SentimentVeryDissatisfied as SentimentNegativeIcon,
} from '@mui/icons-material';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, BarElement } from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { tradingService } from '../services/apiService';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Helper component for displaying sentiment chart
const SentimentChart = ({ sentiment }) => {
  if (!sentiment) return null;
  
  const labels = Object.keys(sentiment).filter(key => 
    !['overall', 'summary', 'reasoning'].includes(key)
  );
  
  const sentimentData = labels.map(key => {
    const value = sentiment[key];
    return typeof value === 'number' ? value : 
           typeof value === 'object' && value.score ? value.score : 0;
  });
  
  const data = {
    labels,
    datasets: [
      {
        label: 'Sentiment Score',
        data: sentimentData,
        backgroundColor: sentimentData.map(value => 
          value > 0.6 ? 'rgba(75, 192, 75, 0.8)' : 
          value > 0.4 ? 'rgba(255, 205, 86, 0.8)' : 
          'rgba(255, 99, 132, 0.8)'
        ),
        borderColor: 'rgba(75, 99, 132, 1)',
        borderWidth: 1
      }
    ]
  };
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 1.0
      }
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: 'Sentiment Analysis'
      }
    }
  };
  
  return (
    <Box sx={{ height: 200, mb: 2 }}>
      <Bar data={data} options={options} />
    </Box>
  );
};

// Helper component for price indicators
const PriceIndicator = ({ result }) => {
  if (!result?.market_metrics) return null;
  
  const metrics = result.market_metrics || {};
  const getTrendColor = (value, threshold = 0) => {
    if (!value && value !== 0) return 'warning.main';
    if (value > threshold) return 'success.main';
    if (value < threshold) return 'error.main';
    return 'warning.main';
  };
  
  return (
    <Grid container spacing={2} sx={{ mt: 1 }}>
      <Grid item xs={6} sm={3}>
        <Paper 
          elevation={2} 
          sx={{ p: 1.5, textAlign: 'center', bgcolor: '#252525', borderRadius: 2 }}
        >
          <Typography variant="caption" display="block" color="grey.300">RSI</Typography>
          <Typography 
            variant="h6" 
            color={getTrendColor(metrics.rsi, 50)}
            sx={{ fontSize: '1.3rem', fontWeight: 'medium' }}
          >
            {(metrics.rsi != null) ? metrics.rsi.toFixed(1) : 'N/A'}
          </Typography>
        </Paper>
      </Grid>
      
      <Grid item xs={6} sm={3}>
        <Paper 
          elevation={2} 
          sx={{ p: 1.5, textAlign: 'center', bgcolor: '#252525', borderRadius: 2 }}
        >
          <Typography variant="caption" display="block" color="grey.300">Volatility</Typography>
          <Typography variant="h6" color="white" sx={{ fontSize: '1.3rem', fontWeight: 'medium' }}>
            {(metrics.volatility != null) ? (metrics.volatility * 100).toFixed(1) + '%' : 'N/A'}
          </Typography>
        </Paper>
      </Grid>
      
      <Grid item xs={6} sm={3}>
        <Paper 
          elevation={2} 
          sx={{ p: 1.5, textAlign: 'center', bgcolor: '#252525', borderRadius: 2 }}
        >
          <Typography variant="caption" display="block" color="grey.300">24h Change</Typography>
          <Typography 
            variant="h6" 
            color={getTrendColor(metrics.price_change_24h)}
            sx={{ fontSize: '1.3rem', fontWeight: 'medium' }}
          >
            {(metrics.price_change_24h != null) ? metrics.price_change_24h.toFixed(1) + '%' : 'N/A'}
          </Typography>
        </Paper>
      </Grid>
      
      <Grid item xs={6} sm={3}>
        <Paper 
          elevation={2} 
          sx={{ p: 1.5, textAlign: 'center', bgcolor: '#252525', borderRadius: 2 }}
        >
          <Typography variant="caption" display="block" color="grey.300">Trend</Typography>
          <Typography 
            variant="h6" 
            color={getTrendColor(metrics.trend_strength)}
            sx={{ fontSize: '1.3rem', fontWeight: 'medium' }}
          >
            {(metrics.trend_strength != null) ? (metrics.trend_strength * 100).toFixed(0) + '%' : 'N/A'}
          </Typography>
        </Paper>
      </Grid>
    </Grid>
  );
};

// Helper component for action recommendation
const ActionBadge = ({ action, confidence }) => {
  let color, icon;
  switch(action) {
    case 'BUY':
      color = 'success';
      icon = <ArrowUpIcon />;
      break;
    case 'SELL':
      color = 'error';
      icon = <ArrowDownIcon />;
      break;
    default:
      color = 'warning';
      icon = <HoldIcon />;
  }
  
  return (
    <Chip 
      color={color}
      icon={icon}
      label={`${action} (${(confidence * 100).toFixed(1)}%)`}
      variant="filled"
      sx={{ 
        fontWeight: 'bold', 
        mt: 1, 
        fontSize: '1.2rem', 
        py: 2,
        height: 'auto',
        '& .MuiChip-icon': { fontSize: '1.5rem' }
      }}
    />
  );
};

export default function TradeControls({ status, onAction }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isPaperTrading, setIsPaperTrading] = useState(!status?.auto_trading);

  // Update isPaperTrading when status changes
  useEffect(() => {
    if (status) {
      setIsPaperTrading(!status.auto_trading);
    }
  }, [status]);

  const handleTradingModeToggle = () => {
    setIsPaperTrading(!isPaperTrading);
  };

  const handleAction = async (action) => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      let response;
      
      switch (action) {
        case 'start':
          response = await tradingService.startTrading({ paper_trading: isPaperTrading });
          setSuccess(`${isPaperTrading ? 'Paper' : 'Real'} trading started successfully`);
          break;
        case 'stop':
          response = await tradingService.stopTrading();
          setSuccess('Trading stopped successfully');
          break;
        case 'toggle':
          response = await tradingService.toggleAutoTrading();
          setSuccess(`Auto trading ${status.auto_trading ? 'disabled' : 'enabled'} successfully`);
          break;
        case 'toggle-mode':
          // Mode is toggled locally, actual switch happens when starting trading
          setSuccess(`Switched to ${isPaperTrading ? 'paper' : 'real'} trading mode`);
          break;
        case 'analyze':
          try {
            response = await tradingService.runAnalysis();
            setSuccess('Analysis cycle completed successfully');
            setAnalysisResults(response.data.analysis || response.data);
          } catch (analysisError) {
            console.error('Analysis error:', analysisError);
            setError(`Failed to analyze trading: ${analysisError.message || 'Unknown error'}`);
            throw analysisError; // Re-throw to be caught by outer catch
          }
          break;
        default:
          throw new Error(`Unknown action: ${action}`);
      }
      
      if (onAction) {
        onAction(action, response.data);
      }
    } catch (err) {
      setError(`Failed to ${action} trading: ${err.message || 'Unknown error'}`);
      console.error(`Error during ${action}:`, err);
    } finally {
      setLoading(false);
      
      // Clear success/error messages after 5 seconds
      setTimeout(() => {
        setSuccess(null);
        setError(null);
      }, 5000);
      
      // Don't clear analysis results
    }
  };

  return (
    <Card sx={{ boxShadow: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Trading Controls
        </Typography>
        
        <Box mt={2} sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={status?.auto_trading || false}
                onChange={() => handleAction('toggle')}
                color="primary"
                disabled={loading}
              />
            }
            label={`Auto Trading: ${status?.auto_trading ? 'Enabled' : 'Disabled'}`}
          />

          <FormControlLabel
            control={
              <Switch
                checked={!isPaperTrading}
                onChange={() => {
                  handleTradingModeToggle();
                  handleAction('toggle-mode');
                }}
                color="warning"
                disabled={loading || status?.is_running}
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography>
                  Trading Mode: {isPaperTrading ? 'Paper Trading' : 'Real Trading'}
                </Typography>
                {!isPaperTrading && (
                  <Chip 
                    size="small" 
                    color="error" 
                    label="LIVE" 
                    sx={{ ml: 1, height: 20 }}
                  />
                )}
              </Box>
            }
          />
        </Box>
        
        <Box mt={2} sx={{ bgcolor: 'background.paper', p: 1.5, borderRadius: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Status: {status?.is_running ? 'Running' : 'Stopped'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Mode: {isPaperTrading ? 'Paper Trading' : 'Real Trading'} | {status?.fast_mode ? 'Fast (no LLM)' : 'Full (with LLM)'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Daily Trades: {status?.daily_trade_count || 0}/{status?.max_daily_trades || 10}
          </Typography>
        </Box>

        {error && (
          <Box mt={2}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}
        
        {success && (
          <Box mt={2}>
            <Alert severity="success">{success}</Alert>
          </Box>
        )}
        
        {analysisResults && (
          <Box mt={3}>
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="h5" color="white" fontWeight="bold">Analysis Results</Typography>
              </AccordionSummary>
              <AccordionDetails>
                {Array.isArray(analysisResults) ? (
                  <Box sx={{ maxHeight: 800, overflow: 'auto' }}>
                    <Grid container spacing={2}>
                      {analysisResults.map((result, index) => (
                        <Grid item xs={12} md={12} key={index}>
                          <Card 
                            elevation={3}
                            sx={{ 
                              height: '100%',
                              display: 'flex',
                              flexDirection: 'column',
                              backgroundColor: '#1a1a1a'
                            }}
                          >
                            <CardContent sx={{ flexGrow: 1, p: 4 }}>
                              {/* Symbol Header with Price */}
                              <Box 
                                sx={{ 
                                  display: 'flex', 
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                  mb: 1,
                                  pb: 1,
                                  borderBottom: '1px solid #333'
                                }}
                              >
                                <Typography variant="h6" fontWeight="bold" color="white">
                                  {result.symbol}
                                </Typography>
                                <Box>
                                  <Typography variant="h6" color="primary.light" fontWeight="bold" sx={{ fontSize: '1.5rem' }}>
                                    ${result.current_price?.toFixed(4)}
                                  </Typography>
                                </Box>
                              </Box>
                              
                              {/* Action Recommendation Badge */}
                              <Box 
                                sx={{ 
                                  display: 'flex', 
                                  justifyContent: 'center',
                                  mb: 2
                                }}
                              >
                                <ActionBadge 
                                  action={result.strategy?.action || 'HOLD'} 
                                  confidence={result.strategy?.confidence || 0}
                                />
                                <Typography sx={{ mt: 1, color: 'grey.400', fontStyle: 'italic', fontSize: '0.9rem', textAlign: 'center' }}>
                                  Signal strength: {(result.strategy?.signal_strength * 100).toFixed(1) || 0}%
                                </Typography>
                              </Box>
                              
                              {/* Price Metrics Dashboard */}
                              <PriceIndicator result={result} />
                              
                              {/* Market Regime */}
                              {result.strategy?.market_regime && (
                                <Box sx={{ mt: 2 }}>
                                  <Typography variant="subtitle1" color="white" sx={{ mb: 1, fontWeight: 'medium' }}>
                                    Market Regime
                                  </Typography>
                                  <Paper 
                                    elevation={0}
                                    sx={{ 
                                      p: 2, 
                                      bgcolor: '#252525',
                                      display: 'flex',
                                      justifyContent: 'center',
                                      alignItems: 'center',
                                      borderRadius: 2
                                    }}
                                  >
                                    <Chip 
                                      label={result.strategy.market_regime.regime}
                                      color={
                                        result.strategy.market_regime.regime.toLowerCase().includes('bull') ? 'success' :
                                        result.strategy.market_regime.regime.toLowerCase().includes('bear') ? 'error' : 'warning'
                                      }
                                      sx={{ fontWeight: 'bold', fontSize: '1.1rem', py: 1 }}
                                    />
                                    <Typography variant="body2" sx={{ ml: 2, color: 'white', fontSize: '1rem' }}>
                                      Confidence: {(result.strategy.market_regime.confidence * 100).toFixed(1)}%
                                    </Typography>
                                  </Paper>
                                </Box>
                              )}
                              
                              {/* Sentiment Analysis */}
                              {result.strategy?.sentiment && (
                                <Box sx={{ mt: 2 }}>
                                  <Typography variant="subtitle2" color="text.secondary">
                                    LLM Sentiment Analysis
                                  </Typography>
                                  
                                  {/* Sentiment Chart */}
                                  <SentimentChart sentiment={result.strategy.sentiment} />
                                  
                                  {/* Sentiment Summary */}
                                  <Paper 
                                    elevation={0}
                                    sx={{ 
                                      p: 1, 
                                      bgcolor: 'background.default',
                                      borderLeft: '4px solid',
                                      borderColor: (
                                        (result.strategy.sentiment.overall || 0) > 0.6 ? 'success.main' :
                                        (result.strategy.sentiment.overall || 0) < 0.4 ? 'error.main' : 'warning.main'
                                      )
                                    }}
                                  >
                                    <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                                      {result.strategy.sentiment.summary || 'No sentiment summary available'}
                                    </Typography>
                                  </Paper>
                                </Box>
                              )}
                              
                              {/* Position Info */}
                              {result.position_info && result.position_info.has_position && (
                                <Box sx={{ mt: 2 }}>
                                  <Typography variant="subtitle2" color="text.secondary">
                                    Current Position
                                  </Typography>
                                  <Paper 
                                    elevation={0}
                                    sx={{ 
                                      p: 1, 
                                      bgcolor: 'background.default',
                                      display: 'flex',
                                      justifyContent: 'space-between'
                                    }}
                                  >
                                    <Typography variant="body2">
                                      Quantity: {result.position_info.quantity.toFixed(6)}
                                    </Typography>
                                    <Typography 
                                      variant="body2" 
                                      color={result.position_info.unrealized_pnl > 0 ? 'success.main' : 'error.main'}
                                      fontWeight="bold"
                                    >
                                      {result.position_info.unrealized_pnl > 0 ? '+' : ''}
                                      ${result.position_info.unrealized_pnl.toFixed(2)} 
                                      ({result.position_info.unrealized_pnl_percent.toFixed(1)}%)
                                    </Typography>
                                  </Paper>
                                </Box>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                ) : (
                  <Paper 
                    variant="outlined" 
                    sx={{ 
                      p: 2, 
                      maxHeight: 300, 
                      overflow: 'auto',
                      backgroundColor: '#f5f5f5'
                    }}
                  >
                    <Typography component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                      {JSON.stringify(analysisResults, null, 2)}
                    </Typography>
                  </Paper>
                )}
              </AccordionDetails>
            </Accordion>
          </Box>
        )}
      </CardContent>
      
      <Divider />
      
      <CardActions sx={{ padding: 2, justifyContent: 'center', mt: 'auto' }}>
        <Box display="flex" flexWrap="wrap" gap={1} width="100%" justifyContent="space-around">
          <Button
            variant="contained"
            color={isPaperTrading ? "primary" : "error"}
            startIcon={<PlayIcon />}
            onClick={() => handleAction('start')}
            disabled={loading || (status?.is_running)}
            sx={{ 
              position: 'relative',
              '&:after': !isPaperTrading ? {
                content: '"REAL"',
                position: 'absolute',
                top: '-8px',
                right: '-8px',
                fontSize: '0.6rem',
                backgroundColor: 'error.main',
                color: 'white',
                padding: '2px 4px',
                borderRadius: '4px',
                fontWeight: 'bold'
              } : {}
            }}
          >
            Start {isPaperTrading ? "Paper" : "Real"} Trading
          </Button>
          
          <Button
            variant="contained"
            color="error"
            startIcon={<StopIcon />}
            onClick={() => handleAction('stop')}
            disabled={loading || (!status?.is_running)}
          >
            Stop
          </Button>
          
          <Button
            variant="contained"
            color="secondary"
            startIcon={<AutorenewIcon />}
            onClick={() => handleAction('toggle')}
            disabled={loading}
          >
            Toggle Auto
          </Button>
          
          <Button
            variant="contained"
            color="info"
            startIcon={loading ? <CircularProgress size={24} /> : <AnalyzeIcon />}
            onClick={() => handleAction('analyze')}
            disabled={loading}
            sx={{ mt: { xs: 1, sm: 0 } }}
          >
            Run Analysis
          </Button>
        </Box>
      </CardActions>
    </Card>
  );
}