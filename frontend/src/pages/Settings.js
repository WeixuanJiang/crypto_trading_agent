import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Alert,
  CircularProgress,
  Chip,
  Card,
  CardContent,
  CardHeader,
  Divider,
  IconButton,
  Tooltip,
  InputAdornment
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { tradingService } from '../services/apiService';

export default function Settings() {
  const [settings, setSettings] = useState({
    trading_pairs: [],
    min_confidence_threshold: 0.7,
    max_daily_trades: 5,
    trading_interval_minutes: 60
  });
  
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [newPair, setNewPair] = useState('');

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      setLoading(true);
      const response = await tradingService.getStatus();
      
      if (response.data && response.data.success) {
        const { status } = response.data;
        setSettings({
          trading_pairs: status.trading_pairs || [],
          min_confidence_threshold: status.min_confidence_threshold || 0.7,
          max_daily_trades: status.max_daily_trades || 5,
          trading_interval_minutes: status.trading_interval_minutes || 60
        });
        setError(null);
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err) {
      console.error('Failed to fetch settings:', err);
      setError('Failed to load settings. Please check your connection to the backend server.');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const response = await tradingService.updateSettings(settings);
      
      if (response.data && response.data.success) {
        setSuccess('Settings saved successfully!');
        setError(null);
        
        // Update settings with values from the server
        setSettings(response.data.settings);
        
        // Clear success message after 3 seconds
        setTimeout(() => {
          setSuccess(null);
        }, 3000);
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (err) {
      console.error('Failed to save settings:', err);
      setError(`Failed to save settings: ${err.message || 'Unknown error'}`);
      
      // Clear error message after 5 seconds
      setTimeout(() => {
        setError(null);
      }, 5000);
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (field) => (event) => {
    setSettings({
      ...settings,
      [field]: event.target.value
    });
  };

  const handleSliderChange = (field) => (event, newValue) => {
    setSettings({
      ...settings,
      [field]: newValue
    });
  };

  const handleAddPair = () => {
    if (newPair && !settings.trading_pairs.includes(newPair)) {
      setSettings({
        ...settings,
        trading_pairs: [...settings.trading_pairs, newPair]
      });
      setNewPair('');
    }
  };

  const handleDeletePair = (pair) => {
    setSettings({
      ...settings,
      trading_pairs: settings.trading_pairs.filter(p => p !== pair)
    });
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <>
      <Box mb={3} display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h5">
          Trading Settings
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={fetchSettings}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Trading Pairs Section */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', boxShadow: 3 }}>
            <CardHeader 
              title="Trading Pairs" 
              subheader="Configure the trading pairs to monitor"
            />
            <CardContent>
              <Box mb={3}>
                <TextField
                  label="New Trading Pair"
                  value={newPair}
                  onChange={(e) => setNewPair(e.target.value)}
                  placeholder="e.g., BTC-USDT"
                  fullWidth
                  size="small"
                  InputProps={{
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton 
                          onClick={handleAddPair}
                          disabled={!newPair}
                          size="small"
                        >
                          <AddIcon />
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
              </Box>

              <Box 
                sx={{ 
                  display: 'flex', 
                  flexWrap: 'wrap', 
                  gap: 1,
                  minHeight: '100px'
                }}
              >
                {settings.trading_pairs.map((pair) => (
                  <Chip
                    key={pair}
                    label={pair}
                    onDelete={() => handleDeletePair(pair)}
                    color="primary"
                    variant="outlined"
                  />
                ))}
                
                {settings.trading_pairs.length === 0 && (
                  <Typography variant="body2" color="text.secondary">
                    No trading pairs configured
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Trading Parameters Section */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', boxShadow: 3 }}>
            <CardHeader 
              title="Trading Parameters" 
              subheader="Configure trading behavior"
            />
            <CardContent>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Typography gutterBottom>
                    Minimum Confidence Threshold ({Math.round(settings.min_confidence_threshold * 100)}%)
                  </Typography>
                  <Box display="flex" alignItems="center">
                    <Slider
                      value={settings.min_confidence_threshold}
                      onChange={handleSliderChange('min_confidence_threshold')}
                      min={0}
                      max={1}
                      step={0.01}
                      sx={{ flexGrow: 1, mr: 2 }}
                    />
                    <Tooltip title="Minimum confidence required to execute a trade">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Max Daily Trades"
                    type="number"
                    value={settings.max_daily_trades}
                    onChange={handleChange('max_daily_trades')}
                    fullWidth
                    InputProps={{
                      inputProps: { min: 1, max: 100 }
                    }}
                  />
                </Grid>
                
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Trading Interval (minutes)"
                    type="number"
                    value={settings.trading_interval_minutes}
                    onChange={handleChange('trading_interval_minutes')}
                    fullWidth
                    InputProps={{
                      inputProps: { min: 5, max: 1440 }
                    }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Save Button */}
        <Grid item xs={12}>
          <Box display="flex" justifyContent="flex-end" mt={2}>
            <Button
              variant="contained"
              color="primary"
              startIcon={saving ? <CircularProgress size={20} /> : <SaveIcon />}
              onClick={handleSave}
              disabled={saving}
              size="large"
            >
              Save Settings
            </Button>
          </Box>
        </Grid>
      </Grid>
    </>
  );
}