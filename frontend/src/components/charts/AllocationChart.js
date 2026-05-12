import React from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Typography,
  useTheme
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { formatCurrency, formatPercent } from '../../utils/formatters';
import { CHART_SETTINGS } from '../../utils/constants';
import { ChartSkeleton } from '../common/LoadingSkeleton';

// Color palette for different assets
const COLORS = [
  '#a855f7', // Purple (primary)
  '#3b82f6', // Blue (secondary)
  '#22c55e', // Green (success)
  '#eab308', // Yellow (warning)
  '#c084fc', // Purple light
  '#ef4444', // Red (error)
  '#4ade80', // Light green
  '#60a5fa', // Blue light
  '#f59e0b', // Amber
  '#818cf8'  // Indigo
];

/**
 * Custom label for pie chart
 */
const renderCustomLabel = ({
  cx,
  cy,
  midAngle,
  innerRadius,
  outerRadius,
  percent
}) => {
  if (percent < 0.05) return null; // Don't show label if less than 5%

  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      style={{ fontSize: '0.875rem', fontWeight: 600 }}
    >
      {formatPercent(percent)}
    </text>
  );
};

/**
 * Custom tooltip for allocation chart
 */
const CustomTooltip = ({ active, payload }) => {
  const theme = useTheme();

  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0];

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
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          {data.name}
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.5, fontSize: '0.75rem' }}>
          <Box sx={{ color: 'text.secondary' }}>Value:</Box>
          <Box sx={{ textAlign: 'right', fontWeight: 600 }}>{formatCurrency(data.value)}</Box>

          <Box sx={{ color: 'text.secondary' }}>Allocation:</Box>
          <Box sx={{ textAlign: 'right', fontWeight: 600 }}>{formatPercent(data.payload.percent)}</Box>

          {data.payload.quantity && (
            <>
              <Box sx={{ color: 'text.secondary' }}>Quantity:</Box>
              <Box sx={{ textAlign: 'right', fontWeight: 600 }}>{data.payload.quantity}</Box>
            </>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * Asset Allocation Chart Component (Pie Chart)
 */
const AllocationChart = ({
  data = [],
  title = 'Asset Allocation',
  height = CHART_SETTINGS.DEFAULT_HEIGHT,
  showLegend = true,
  showList = true,
  onRefresh,
  loading = false
}) => {
  const theme = useTheme();

  // Calculate total value
  const totalValue = React.useMemo(() => {
    return data.reduce((sum, item) => sum + (item.value || 0), 0);
  }, [data]);

  // Add percentage to each item
  const chartData = React.useMemo(() => {
    return data.map((item, index) => ({
      ...item,
      percent: totalValue > 0 ? item.value / totalValue : 0,
      color: COLORS[index % COLORS.length]
    }));
  }, [data, totalValue]);

  if (loading) {
    return <ChartSkeleton height={height} />;
  }

  if (data.length === 0) {
    return (
      <Card>
        <CardHeader
          title={title}
          action={
            onRefresh && (
              <IconButton onClick={onRefresh} size="small">
                <RefreshIcon />
              </IconButton>
            )
          }
        />
        <CardContent>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: height,
              color: 'text.secondary'
            }}
          >
            <Typography>No allocation data available</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title={title}
        subheader={`Total Value: ${formatCurrency(totalValue)}`}
        action={
          onRefresh && (
            <IconButton onClick={onRefresh} size="small">
              <RefreshIcon />
            </IconButton>
          )
        }
      />
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={renderCustomLabel}
              outerRadius={Math.min(height * 0.35, 120)}
              fill="#a855f7"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            {showLegend && (
              <Legend
                verticalAlign="bottom"
                height={36}
                wrapperStyle={{ fontSize: '0.875rem' }}
              />
            )}
          </PieChart>
        </ResponsiveContainer>

        {showList && (
          <List dense sx={{ mt: 2 }}>
            {chartData.map((item, index) => (
              <ListItem
                key={index}
                sx={{
                  py: 0.5,
                  borderBottom: index < chartData.length - 1 ? `1px solid ${theme.palette.divider}` : 'none'
                }}
              >
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    bgcolor: item.color,
                    mr: 1.5,
                    flexShrink: 0
                  }}
                />
                <ListItemText
                  primary={item.name}
                  secondary={item.quantity ? `${item.quantity} units` : undefined}
                  primaryTypographyProps={{ fontSize: '0.875rem' }}
                  secondaryTypographyProps={{ fontSize: '0.75rem' }}
                />
                <Box sx={{ textAlign: 'right' }}>
                  <Typography variant="body2" fontWeight={600}>
                    {formatCurrency(item.value)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatPercent(item.percent)}
                  </Typography>
                </Box>
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

/**
 * Donut Chart variant
 */
export const DonutAllocationChart = (props) => {
  return (
    <Card>
      <CardHeader
        title={props.title || 'Asset Allocation'}
        subheader={props.totalValue ? `Total: ${formatCurrency(props.totalValue)}` : undefined}
        action={
          props.onRefresh && (
            <IconButton onClick={props.onRefresh} size="small">
              <RefreshIcon />
            </IconButton>
          )
        }
      />
      <CardContent>
        <ResponsiveContainer width="100%" height={props.height || CHART_SETTINGS.DEFAULT_HEIGHT}>
          <PieChart>
            <Pie
              data={props.data}
              cx="50%"
              cy="50%"
              innerRadius="60%"
              outerRadius="80%"
              fill="#a855f7"
              paddingAngle={2}
              dataKey="value"
            >
              {props.data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            {props.showLegend && (
              <Legend
                verticalAlign="bottom"
                height={36}
                wrapperStyle={{ fontSize: '0.875rem' }}
              />
            )}
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default AllocationChart;
