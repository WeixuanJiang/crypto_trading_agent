import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Menu,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
  useTheme
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  Refresh as RefreshIcon,
  Fullscreen as FullscreenIcon
} from '@mui/icons-material';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { format } from 'date-fns';
import { formatCurrency, formatNumber } from '../../utils/formatters';
import { TIMEFRAMES, CHART_SETTINGS } from '../../utils/constants';
import { ChartSkeleton } from '../common/LoadingSkeleton';

/**
 * Custom candlestick component for Recharts
 */
const Candlestick = (props) => {
  const { x, y, width, height, open, close, high, low, fill } = props;
  const isRising = close > open;
  const color = isRising ? '#00e676' : '#ff1744';
  const ratio = Math.abs(height / (high - low));

  return (
    <g>
      {/* Wick (high-low line) */}
      <line
        x1={x + width / 2}
        y1={y}
        x2={x + width / 2}
        y2={y + height}
        stroke={color}
        strokeWidth={1}
      />
      {/* Body (open-close rectangle) */}
      <rect
        x={x}
        y={isRising ? y + (high - close) * ratio : y + (high - open) * ratio}
        width={width}
        height={Math.abs((close - open) * ratio)}
        fill={color}
        stroke={color}
        strokeWidth={1}
      />
    </g>
  );
};

/**
 * Custom tooltip for candlestick chart
 */
const CustomTooltip = ({ active, payload }) => {
  const theme = useTheme();

  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload;

  return (
    <Card
      sx={{
        bgcolor: 'background.paper',
        border: `1px solid ${theme.palette.divider}`,
        boxShadow: 3,
        minWidth: 200
      }}
    >
      <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
        <Box sx={{ mb: 1 }}>
          <Box sx={{ fontSize: '0.75rem', color: 'text.secondary', mb: 0.5 }}>
            {format(new Date(data.timestamp), 'MMM dd, yyyy HH:mm')}
          </Box>
        </Box>
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.5, fontSize: '0.75rem' }}>
          <Box sx={{ color: 'text.secondary' }}>Open:</Box>
          <Box sx={{ textAlign: 'right', fontWeight: 600 }}>{formatCurrency(data.open)}</Box>

          <Box sx={{ color: 'text.secondary' }}>High:</Box>
          <Box sx={{ textAlign: 'right', fontWeight: 600, color: 'success.main' }}>
            {formatCurrency(data.high)}
          </Box>

          <Box sx={{ color: 'text.secondary' }}>Low:</Box>
          <Box sx={{ textAlign: 'right', fontWeight: 600, color: 'error.main' }}>
            {formatCurrency(data.low)}
          </Box>

          <Box sx={{ color: 'text.secondary' }}>Close:</Box>
          <Box
            sx={{
              textAlign: 'right',
              fontWeight: 600,
              color: data.close >= data.open ? 'success.main' : 'error.main'
            }}
          >
            {formatCurrency(data.close)}
          </Box>

          <Box sx={{ color: 'text.secondary' }}>Volume:</Box>
          <Box sx={{ textAlign: 'right', fontWeight: 600 }}>{formatNumber(data.volume)}</Box>
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * Candlestick Chart Component
 */
const CandlestickChart = ({
  data = [],
  title = 'Price Chart',
  height = CHART_SETTINGS.CANDLESTICK_HEIGHT,
  showVolume = true,
  showMA = true,
  timeframe = '1h',
  onTimeframeChange,
  onRefresh,
  loading = false
}) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [selectedTimeframe, setSelectedTimeframe] = React.useState(timeframe);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleTimeframeChange = (event, newTimeframe) => {
    if (newTimeframe) {
      setSelectedTimeframe(newTimeframe);
      onTimeframeChange?.(newTimeframe);
    }
  };

  // Calculate moving averages
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const processedData = [...data];

    // Calculate 20-period MA
    if (showMA && processedData.length >= 20) {
      for (let i = 19; i < processedData.length; i++) {
        const sum = processedData
          .slice(i - 19, i + 1)
          .reduce((acc, d) => acc + d.close, 0);
        processedData[i].ma20 = sum / 20;
      }
    }

    return processedData;
  }, [data, showMA]);

  const maxPrice = useMemo(
    () => Math.max(...chartData.map((d) => d.high || 0)),
    [chartData]
  );
  const minPrice = useMemo(
    () => Math.min(...chartData.map((d) => d.low || Infinity)),
    [chartData]
  );

  if (loading) {
    return <ChartSkeleton height={height} />;
  }

  return (
    <Card>
      <CardHeader
        title={title}
        action={
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <ToggleButtonGroup
              value={selectedTimeframe}
              exclusive
              onChange={handleTimeframeChange}
              size="small"
            >
              {TIMEFRAMES.slice(0, 6).map((tf) => (
                <ToggleButton key={tf.value} value={tf.value} sx={{ px: 1.5, py: 0.5 }}>
                  {tf.value}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            {onRefresh && (
              <IconButton onClick={onRefresh} size="small">
                <RefreshIcon />
              </IconButton>
            )}

            <IconButton onClick={handleMenuOpen} size="small">
              <MoreVertIcon />
            </IconButton>

            <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleMenuClose}>
              <MenuItem onClick={handleMenuClose}>
                <FullscreenIcon sx={{ mr: 1, fontSize: 20 }} />
                Fullscreen
              </MenuItem>
              <MenuItem onClick={handleMenuClose}>Export Data</MenuItem>
              <MenuItem onClick={handleMenuClose}>Settings</MenuItem>
            </Menu>
          </Box>
        }
      />
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={theme.palette.divider}
              opacity={0.3}
            />

            <XAxis
              dataKey="timestamp"
              tickFormatter={(timestamp) => format(new Date(timestamp), 'MMM dd HH:mm')}
              stroke={theme.palette.text.secondary}
              style={{ fontSize: '0.75rem' }}
            />

            <YAxis
              yAxisId="price"
              domain={[minPrice * 0.995, maxPrice * 1.005]}
              tickFormatter={(value) => formatCurrency(value, '$')}
              stroke={theme.palette.text.secondary}
              style={{ fontSize: '0.75rem' }}
            />

            {showVolume && (
              <YAxis
                yAxisId="volume"
                orientation="right"
                tickFormatter={(value) => formatNumber(value)}
                stroke={theme.palette.text.secondary}
                style={{ fontSize: '0.75rem' }}
              />
            )}

            <Tooltip content={<CustomTooltip />} />

            <Legend
              wrapperStyle={{ fontSize: '0.875rem' }}
              iconType="line"
            />

            {showVolume && (
              <Bar
                yAxisId="volume"
                dataKey="volume"
                fill={theme.palette.info.main}
                opacity={0.3}
                name="Volume"
              />
            )}

            {/* Candlesticks rendered as custom shape */}
            <Bar
              yAxisId="price"
              dataKey="high"
              fill="transparent"
              shape={<Candlestick />}
              name="Price"
            />

            {showMA && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="ma20"
                stroke={theme.palette.secondary.main}
                strokeWidth={2}
                dot={false}
                name="MA(20)"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default CandlestickChart;
