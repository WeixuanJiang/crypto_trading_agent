import React from 'react';
import { Box, Card, CardContent, CardHeader, Skeleton, Grid, Stack } from '@mui/material';

/**
 * Generic loading skeleton component
 */
export const LoadingSkeleton = ({ variant = 'text', width, height, ...props }) => {
  return (
    <Skeleton
      variant={variant}
      width={width}
      height={height}
      animation="wave"
      {...props}
    />
  );
};

/**
 * Loading skeleton for cards
 */
export const CardSkeleton = ({ hasHeader = true, lines = 3 }) => {
  return (
    <Card>
      {hasHeader && (
        <CardHeader
          avatar={<Skeleton variant="circular" width={40} height={40} />}
          title={<Skeleton variant="text" width="60%" />}
          subheader={<Skeleton variant="text" width="40%" />}
        />
      )}
      <CardContent>
        <Stack spacing={1}>
          {[...Array(lines)].map((_, i) => (
            <Skeleton key={i} variant="text" width={i === lines - 1 ? '80%' : '100%'} />
          ))}
        </Stack>
      </CardContent>
    </Card>
  );
};

/**
 * Loading skeleton for tables
 */
export const TableSkeleton = ({ rows = 5, columns = 4 }) => {
  return (
    <Box>
      {/* Table Header */}
      <Grid container spacing={2} sx={{ mb: 2, px: 2 }}>
        {[...Array(columns)].map((_, i) => (
          <Grid item xs={12 / columns} key={`header-${i}`}>
            <Skeleton variant="text" width="80%" height={40} />
          </Grid>
        ))}
      </Grid>

      {/* Table Rows */}
      {[...Array(rows)].map((_, rowIndex) => (
        <Grid container spacing={2} key={`row-${rowIndex}`} sx={{ mb: 1, px: 2 }}>
          {[...Array(columns)].map((_, colIndex) => (
            <Grid item xs={12 / columns} key={`cell-${rowIndex}-${colIndex}`}>
              <Skeleton variant="text" width="90%" />
            </Grid>
          ))}
        </Grid>
      ))}
    </Box>
  );
};

/**
 * Loading skeleton for charts
 */
export const ChartSkeleton = ({ height = 400 }) => {
  return (
    <Card>
      <CardHeader
        title={<Skeleton variant="text" width="40%" />}
        action={<Skeleton variant="rectangular" width={100} height={36} />}
      />
      <CardContent>
        <Skeleton variant="rectangular" width="100%" height={height} />
      </CardContent>
    </Card>
  );
};

/**
 * Loading skeleton for stat cards
 */
export const StatCardSkeleton = () => {
  return (
    <Card>
      <CardContent>
        <Stack spacing={2}>
          <Skeleton variant="text" width="60%" height={20} />
          <Skeleton variant="text" width="80%" height={40} />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Skeleton variant="text" width="40%" height={20} />
            <Skeleton variant="circular" width={24} height={24} />
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
};

/**
 * Loading skeleton for the dashboard
 */
export const DashboardSkeleton = () => {
  return (
    <Box>
      {/* Stats Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {[...Array(4)].map((_, i) => (
          <Grid item xs={12} sm={6} md={3} key={`stat-${i}`}>
            <StatCardSkeleton />
          </Grid>
        ))}
      </Grid>

      {/* Charts Row */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <ChartSkeleton height={400} />
        </Grid>
        <Grid item xs={12} md={4}>
          <ChartSkeleton height={400} />
        </Grid>
      </Grid>

      {/* Table */}
      <Card>
        <CardHeader
          title={<Skeleton variant="text" width="30%" />}
          action={<Skeleton variant="rectangular" width={120} height={36} />}
        />
        <CardContent>
          <TableSkeleton rows={5} columns={5} />
        </CardContent>
      </Card>
    </Box>
  );
};

/**
 * Loading skeleton for trade history
 */
export const TradeHistorySkeleton = () => {
  return (
    <Card>
      <CardHeader
        title={<Skeleton variant="text" width="40%" />}
        action={
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Skeleton variant="rectangular" width={100} height={36} />
            <Skeleton variant="rectangular" width={100} height={36} />
          </Box>
        }
      />
      <CardContent>
        <TableSkeleton rows={10} columns={7} />
      </CardContent>
    </Card>
  );
};

/**
 * Loading skeleton for settings page
 */
export const SettingsSkeleton = () => {
  return (
    <Box>
      <Grid container spacing={3}>
        {[...Array(3)].map((_, i) => (
          <Grid item xs={12} key={`setting-${i}`}>
            <CardSkeleton hasHeader={true} lines={4} />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

/**
 * Loading skeleton for list items
 */
export const ListItemSkeleton = ({ items = 5, hasAvatar = true, hasSecondary = true }) => {
  return (
    <Box>
      {[...Array(items)].map((_, i) => (
        <Box
          key={i}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            py: 2,
            borderBottom: i < items - 1 ? '1px solid' : 'none',
            borderColor: 'divider'
          }}
        >
          {hasAvatar && <Skeleton variant="circular" width={40} height={40} />}
          <Box sx={{ flex: 1 }}>
            <Skeleton variant="text" width="60%" />
            {hasSecondary && <Skeleton variant="text" width="40%" />}
          </Box>
          <Skeleton variant="rectangular" width={80} height={32} />
        </Box>
      ))}
    </Box>
  );
};

export default LoadingSkeleton;
