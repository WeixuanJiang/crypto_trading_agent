import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  useTheme
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { format } from 'date-fns';
import { formatCurrency, formatPercent } from '../../utils/formatters';
import { DATE_RANGES, CHART_SETTINGS } from '../../utils/constants';
import { ChartSkeleton } from '../common/LoadingSkeleton';

/**
 * Custom tooltip for performance chart
 */
const CustomTooltip = ({ active, payload }) => {
  const theme = useTheme();

  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload;
  const value = data.value || 0;
  const change = data.change || 0;

  return (
    <Card
      sx={{
        bgcolor: 'background.paper',
        border: `1px solid ${theme.palette.divider}`,
        boxShadow: 3,
        minWidth: 180
      }}
    >
      <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
        <Box sx={{ fontSize: '0.75rem', color: 'text.secondary', mb: 1 }}>
          {format(new Date(data.timestamp), 'MMM dd, yyyy HH:mm')}
        </Box>
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.5, fontSize: '0.75rem' }}>
          <Box sx={{ color: 'text.secondary' }}>Value:</Box>
          <Box sx={{ textAlign: 'right', fontWeight: 600 }}>{formatCurrency(value)}</Box>

          <Box sx={{ color: 'text.secondary' }}>Change:</Box>
          <Box
            sx={{
              textAlign: 'right',
              fontWeight: 600,
              color: change >= 0 ? 'success.main' : 'error.main'
            }}
          >
            {change >= 0 ? '+' : ''}
            {formatPercent(change)}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * Performance Chart Component
 */
const PerformanceChart = ({
  data = [],
  title = 'Portfolio Performance',
  height = CHART_SETTINGS.DEFAULT_HEIGHT,
  showGrid = true,
  dateRange = '30d',
  onDateRangeChange,
  onRefresh,
  loading = false
}) => {
  const theme = useTheme();
  const [selectedRange, setSelectedRange] = React.useState(dateRange);

  const handleRangeChange = (event, newRange) => {
    if (newRange) {
      setSelectedRange(newRange);
      onDateRangeChange?.(newRange);
    }
  };

  // Calculate min and max for chart domain
  const { minValue, maxValue, initialValue } = useMemo(() => {
    if (!data || data.length === 0) {
      return { minValue: 0, maxValue: 100, initialValue: 100 };
    }

    const values = data.map((d) => d.value || 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const initial = data[0]?.value || 0;

    return {
      minValue: min * 0.95,
      maxValue: max * 1.05,
      initialValue: initial
    };
  }, [data]);

  // Calculate total change
  const totalChange = useMemo(() => {
    if (!data || data.length < 2) return 0;
    const first = data[0]?.value || 0;
    const last = data[data.length - 1]?.value || 0;
    if (first === 0) return 0;
    return (last - first) / first;
  }, [data]);

  // Determine if overall trend is positive
  const isPositive = totalChange >= 0;

  if (loading) {
    return <ChartSkeleton height={height} />;
  }

  return (
    <Card>
      <CardHeader
        title={title}
        subheader={
          <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
            <Box component="span" sx={{ fontSize: '0.875rem', color: 'text.secondary' }}>
              Total Change:
            </Box>
            <Box
              component="span"
              sx={{
                fontSize: '0.875rem',
                fontWeight: 600,
                color: isPositive ? 'success.main' : 'error.main'
              }}
            >
              {isPositive ? '+' : ''}
              {formatPercent(totalChange)}
            </Box>
          </Box>
        }
        action={
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <ToggleButtonGroup
              value={selectedRange}
              exclusive
              onChange={handleRangeChange}
              size="small"
            >
              {DATE_RANGES.slice(0, 4).map((range) => (
                <ToggleButton key={range.value} value={range.value} sx={{ px: 1.5, py: 0.5 }}>
                  {range.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            {onRefresh && (
              <IconButton onClick={onRefresh} size="small">
                <RefreshIcon />
              </IconButton>
            )}
          </Box>
        }
      />
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            {showGrid && (
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={theme.palette.divider}
                opacity={0.3}
              />
            )}

            <XAxis
              dataKey="timestamp"
              tickFormatter={(timestamp) => format(new Date(timestamp), 'MMM dd')}
              stroke={theme.palette.text.secondary}
              style={{ fontSize: '0.75rem' }}
            />

            <YAxis
              domain={[minValue, maxValue]}
              tickFormatter={(value) => formatCurrency(value, '$')}
              stroke={theme.palette.text.secondary}
              style={{ fontSize: '0.75rem' }}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Reference line at initial value */}
            <ReferenceLine
              y={initialValue}
              stroke={theme.palette.text.secondary}
              strokeDasharray="3 3"
              strokeOpacity={0.5}
            />

            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={isPositive ? theme.palette.success.main : theme.palette.error.main}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={isPositive ? theme.palette.success.main : theme.palette.error.main}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>

            <Area
              type="monotone"
              dataKey="value"
              stroke={isPositive ? theme.palette.success.main : theme.palette.error.main}
              strokeWidth={2}
              fill="url(#colorValue)"
              name="Portfolio Value"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default PerformanceChart;
