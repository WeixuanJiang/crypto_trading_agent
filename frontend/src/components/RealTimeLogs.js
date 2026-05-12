import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Chip,
  Switch,
  FormControlLabel,
  TextField,
  MenuItem,
  Tooltip,
  CircularProgress
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { getLogs } from '../services/apiService';

/**
 * Real-Time Logs Component
 * Displays live backend logs with auto-refresh and filtering
 */
export default function RealTimeLogs() {
  const { enqueueSnackbar } = useSnackbar();
  const [logs, setLogs] = useState([]);
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [levelFilter, setLevelFilter] = useState('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const logsEndRef = useRef(null);
  const containerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Fetch logs on mount and set up auto-refresh
  useEffect(() => {
    fetchLogs();

    if (autoRefresh) {
      const interval = setInterval(fetchLogs, 5000); // Refresh every 5 seconds
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // Filter logs when filter criteria changes
  useEffect(() => {
    applyFilters();
  }, [logs, levelFilter, searchTerm]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const data = await getLogs(200); // Fetch last 200 logs
      setLogs(data.logs || []);
    } catch (error) {
      console.error('Error fetching logs:', error);
      // Don't show error notification on every fetch to avoid spam
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...logs];

    // Filter by level
    if (levelFilter !== 'ALL') {
      filtered = filtered.filter(log => log.level === levelFilter);
    }

    // Filter by search term
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(log =>
        log.message?.toLowerCase().includes(term) ||
        log.timestamp?.toLowerCase().includes(term)
      );
    }

    setFilteredLogs(filtered);
  };

  const handleRefresh = () => {
    fetchLogs();
    enqueueSnackbar('Logs refreshed', { variant: 'info' });
  };

  const handleDownloadLogs = () => {
    const logText = filteredLogs.map(log =>
      `[${log.timestamp}] [${log.level}] ${log.message}`
    ).join('\n');

    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trading-logs-${new Date().toISOString()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    enqueueSnackbar('Logs downloaded', { variant: 'success' });
  };

  const getLevelColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR':
      case 'CRITICAL':
        return 'error';
      case 'WARNING':
        return 'warning';
      case 'INFO':
        return 'info';
      case 'SUCCESS':
        return 'success';
      default:
        return 'default';
    }
  };

  const getLevelIcon = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR':
      case 'CRITICAL':
        return '❌';
      case 'WARNING':
        return '⚠️';
      case 'INFO':
        return 'ℹ️';
      case 'SUCCESS':
        return '✅';
      case 'DEBUG':
        return '🔍';
      default:
        return '📝';
    }
  };

  return (
    <Paper sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6" component="h2">
          Real-Time Logs
        </Typography>
        <Box display="flex" gap={1} alignItems="center">
          <FormControlLabel
            control={
              <Switch
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                size="small"
              />
            }
            label="Auto-scroll"
            sx={{ mr: 1 }}
          />
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                size="small"
              />
            }
            label={autoRefresh ? 'Live' : 'Paused'}
            sx={{ mr: 1 }}
          />
          <Tooltip title="Refresh">
            <IconButton onClick={handleRefresh} size="small" disabled={loading}>
              {loading ? <CircularProgress size={20} /> : <RefreshIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Download Logs">
            <IconButton onClick={handleDownloadLogs} size="small">
              <DownloadIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Filters */}
      <Box display="flex" gap={2} mb={2}>
        <TextField
          select
          size="small"
          label="Level"
          value={levelFilter}
          onChange={(e) => setLevelFilter(e.target.value)}
          sx={{ minWidth: 120 }}
        >
          <MenuItem value="ALL">All Levels</MenuItem>
          <MenuItem value="DEBUG">Debug</MenuItem>
          <MenuItem value="INFO">Info</MenuItem>
          <MenuItem value="SUCCESS">Success</MenuItem>
          <MenuItem value="WARNING">Warning</MenuItem>
          <MenuItem value="ERROR">Error</MenuItem>
          <MenuItem value="CRITICAL">Critical</MenuItem>
        </TextField>

        <TextField
          size="small"
          label="Search logs..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ flexGrow: 1 }}
          placeholder="Filter by message or timestamp"
        />
      </Box>

      {/* Log Count */}
      <Box mb={1}>
        <Typography variant="caption" color="text.secondary">
          Showing {filteredLogs.length} of {logs.length} logs
          {autoRefresh && (
            <Chip
              label="LIVE"
              size="small"
              color="success"
              sx={{ ml: 1, height: 20, fontSize: '0.7rem' }}
            />
          )}
        </Typography>
      </Box>

      {/* Logs Container */}
      <Box
        ref={containerRef}
        sx={{
          flexGrow: 1,
          overflowY: 'auto',
          bgcolor: '#050914',
          borderRadius: 1,
          p: 2,
          fontFamily: 'monospace',
          fontSize: '0.85rem',
          maxHeight: '600px'
        }}
      >
        {filteredLogs.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height="100%">
            <Typography color="text.secondary">
              {loading ? 'Loading logs...' : 'No logs to display'}
            </Typography>
          </Box>
        ) : (
          filteredLogs.map((log, index) => (
            <Box
              key={index}
              sx={{
                mb: 1,
                p: 1,
                borderRadius: 0.5,
                bgcolor: 'rgba(255,255,255,0.02)',
                '&:hover': {
                  bgcolor: 'rgba(255,255,255,0.05)'
                },
                borderLeft: `3px solid ${
                  log.level === 'ERROR' ? '#ef4444' :
                  log.level === 'WARNING' ? '#eab308' :
                  log.level === 'SUCCESS' ? '#22c55e' :
                  log.level === 'INFO' ? '#3b82f6' :
                  '#64748b'
                }`
              }}
            >
              <Box display="flex" alignItems="flex-start" gap={1}>
                <span>{getLevelIcon(log.level)}</span>
                <Box flexGrow={1}>
                  <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                    <Chip
                      label={log.level || 'INFO'}
                      size="small"
                      color={getLevelColor(log.level)}
                      sx={{ height: 18, fontSize: '0.7rem' }}
                    />
                    <Typography
                      variant="caption"
                      sx={{ color: 'text.secondary', fontFamily: 'monospace' }}
                    >
                      {log.timestamp}
                    </Typography>
                  </Box>
                  <Typography
                    sx={{
                      color: '#fff',
                      fontFamily: 'monospace',
                      fontSize: '0.85rem',
                      wordBreak: 'break-word',
                      whiteSpace: 'pre-wrap'
                    }}
                  >
                    {log.message}
                  </Typography>
                </Box>
              </Box>
            </Box>
          ))
        )}
        <div ref={logsEndRef} />
      </Box>

      {/* Status Bar */}
      <Box
        mt={2}
        pt={2}
        sx={{ borderTop: '1px solid', borderColor: 'divider' }}
      >
        <Typography variant="caption" color="text.secondary">
          Last updated: {logs.length > 0 ? new Date().toLocaleTimeString() : 'Never'}
          {autoRefresh && ' • Auto-refreshing every 5 seconds'}
        </Typography>
      </Box>
    </Paper>
  );
}
