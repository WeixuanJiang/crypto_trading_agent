import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  Button,
  IconButton,
  CircularProgress,
  Divider
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { tradingService } from '../services/apiService';

export default function RecentLogs() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterMode, setFilterMode] = useState('all'); // 'all' or 'trading'
  
  // Calculate trading logs count
  const getTradingLogsCount = () => {
    return logs.filter(log => log.type === 'trading').length;
  };

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const response = await tradingService.getLogs(50);
      if (response.data && response.data.success) {
        setLogs(response.data.logs || []);
        setError(null);
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err) {
      console.error('Failed to fetch logs:', err);
      setError('Failed to fetch logs. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
    
    // Refresh logs every 30 seconds
    const interval = setInterval(fetchLogs, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getLevelColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR':
        return 'error.main';
      case 'WARNING':
        return 'warning.main';
      case 'INFO':
        return 'info.main';
      case 'DEBUG':
        return 'text.secondary';
      default:
        return 'text.primary';
    }
  };

  return (
    <Card sx={{ height: '100%', boxShadow: 3 }}>
      <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="h6">
            Recent Logs
          </Typography>
          <Box display="flex" alignItems="center">
            <Button 
              size="small" 
              onClick={() => setFilterMode('all')} 
              color={filterMode === 'all' ? 'primary' : 'inherit'}
              sx={{ mr: 1, minWidth: 'auto', px: 1 }}
            >
              All
            </Button>
            <Button 
              size="small" 
              onClick={() => setFilterMode('trading')} 
              color={filterMode === 'trading' ? 'primary' : 'inherit'}
              sx={{ mr: 1, minWidth: 'auto', px: 1 }}
              endIcon={
                <Box 
                  component="span" 
                  sx={{ 
                    bgcolor: 'primary.main', 
                    color: 'white',
                    borderRadius: '50%',
                    width: 20,
                    height: 20,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.75rem',
                    ml: 0.5
                  }}
                >
                  {getTradingLogsCount()}
                </Box>
              }
            >
              Trading
            </Button>
            <IconButton onClick={fetchLogs} size="small" disabled={loading}>
              {loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            </IconButton>
          </Box>
        </Box>

        {error && (
          <Typography color="error" variant="body2">
            {error}
          </Typography>
        )}

        <Box sx={{ 
          flexGrow: 1, 
          overflow: 'auto', 
          maxHeight: '400px',
          bgcolor: 'background.paper',
          borderRadius: 1,
          border: '1px solid',
          borderColor: 'divider'
        }}>
          {loading && logs.length === 0 ? (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <CircularProgress />
            </Box>
          ) : logs.length > 0 ? (
            <List dense disablePadding>
              {logs
                .filter(log => filterMode === 'all' || log.type === 'trading')
                .map((log, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <Divider variant="middle" component="li" />}
                  <ListItem alignItems="flex-start" sx={{ py: 0.5 }}>
                    <ListItemText
                      primaryTypographyProps={{ 
                        variant: 'body2',
                        color: getLevelColor(log.level),
                        fontFamily: 'monospace'
                      }}
                      primary={
                        <>
                          <Box component="span" sx={{ color: 'text.secondary', mr: 1 }}>
                            [{log.timestamp}]
                          </Box>
                          <Box 
                            component="span" 
                            sx={{ 
                              color: getLevelColor(log.level),
                              fontWeight: log.type === 'trading' || log.level?.toUpperCase() === 'ERROR' ? 'bold' : 'normal',
                              bgcolor: log.type === 'trading' ? 'rgba(25, 118, 210, 0.08)' : 'transparent',
                              py: log.type === 'trading' ? 0.5 : 0,
                              px: log.type === 'trading' ? 1 : 0,
                              borderRadius: log.type === 'trading' ? 1 : 0,
                              display: 'inline-block',
                              width: log.type === 'trading' ? '100%' : 'auto',
                            }}
                          >
                            {log.message}
                          </Box>
                        </>
                      }
                    />
                  </ListItem>
                </React.Fragment>
              ))}
            </List>
          ) : (
            <Box display="flex" justifyContent="center" alignItems="center" height="100px">
              <Typography variant="body2" color="text.secondary">
                No logs available
              </Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}