import React from 'react';
import { Box, Typography, Button, Stack } from '@mui/material';
import {
  Inbox as InboxIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
  Settings as SettingsIcon,
  Error as ErrorIcon,
  Search as SearchIcon
} from '@mui/icons-material';

const ICONS = {
  inbox: InboxIcon,
  trending: TrendingUpIcon,
  assessment: AssessmentIcon,
  settings: SettingsIcon,
  error: ErrorIcon,
  search: SearchIcon
};

/**
 * Empty state component for when there's no data to display
 */
const EmptyState = ({
  icon = 'inbox',
  title = 'No data available',
  description,
  action,
  actionLabel,
  onAction,
  secondaryAction,
  secondaryActionLabel,
  onSecondaryAction,
  iconColor = 'text.secondary',
  maxWidth = 400,
  ...props
}) => {
  const IconComponent = typeof icon === 'string' ? ICONS[icon] || InboxIcon : icon;

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        py: 6,
        px: 3,
        minHeight: 300,
        ...props.sx
      }}
      {...props}
    >
      <IconComponent
        sx={{
          fontSize: 80,
          color: iconColor,
          mb: 2,
          opacity: 0.5
        }}
      />

      <Typography variant="h5" gutterBottom sx={{ maxWidth, fontWeight: 600 }}>
        {title}
      </Typography>

      {description && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ mb: 3, maxWidth }}
        >
          {description}
        </Typography>
      )}

      {(action || onAction) && (
        <Stack direction="row" spacing={2}>
          <Button
            variant="contained"
            onClick={onAction}
            sx={{ minWidth: 120 }}
          >
            {actionLabel || action}
          </Button>

          {(secondaryAction || onSecondaryAction) && (
            <Button
              variant="outlined"
              onClick={onSecondaryAction}
              sx={{ minWidth: 120 }}
            >
              {secondaryActionLabel || secondaryAction}
            </Button>
          )}
        </Stack>
      )}
    </Box>
  );
};

/**
 * Empty state for no trades
 */
export const NoTradesEmptyState = ({ onStartTrading }) => (
  <EmptyState
    icon="trending"
    title="No trades yet"
    description="Start trading to see your trade history and performance metrics here."
    actionLabel="Configure Settings"
    onAction={onStartTrading}
    iconColor="primary.main"
  />
);

/**
 * Empty state for no data/results
 */
export const NoDataEmptyState = ({ title, description, onRefresh }) => (
  <EmptyState
    icon="inbox"
    title={title || 'No data available'}
    description={description || 'There is no data to display at the moment.'}
    actionLabel={onRefresh ? 'Refresh' : undefined}
    onAction={onRefresh}
  />
);

/**
 * Empty state for search results
 */
export const NoSearchResultsEmptyState = ({ searchTerm, onClear }) => (
  <EmptyState
    icon="search"
    title="No results found"
    description={
      searchTerm
        ? `No results found for "${searchTerm}". Try adjusting your search criteria.`
        : 'No results found. Try adjusting your filters or search criteria.'
    }
    actionLabel="Clear Search"
    onAction={onClear}
  />
);

/**
 * Empty state for errors
 */
export const ErrorEmptyState = ({ title, description, onRetry }) => (
  <EmptyState
    icon="error"
    title={title || 'Something went wrong'}
    description={description || 'An error occurred while loading data. Please try again.'}
    actionLabel="Retry"
    onAction={onRetry}
    iconColor="error.main"
  />
);

/**
 * Empty state for analytics/charts with no data
 */
export const NoAnalyticsEmptyState = ({ onViewTrades }) => (
  <EmptyState
    icon="assessment"
    title="No analytics available"
    description="Complete some trades to see your performance analytics and insights."
    actionLabel="View Settings"
    onAction={onViewTrades}
    iconColor="info.main"
  />
);

/**
 * Empty state for configuration needed
 */
export const ConfigurationNeededEmptyState = ({ title, description, onConfigure }) => (
  <EmptyState
    icon="settings"
    title={title || 'Configuration required'}
    description={description || 'Please configure your settings before proceeding.'}
    actionLabel="Go to Settings"
    onAction={onConfigure}
    iconColor="warning.main"
  />
);

/**
 * Empty state for filters with no results
 */
export const NoFilterResultsEmptyState = ({ onClearFilters, onViewAll }) => (
  <EmptyState
    icon="search"
    title="No matching results"
    description="No items match your current filters. Try adjusting your filter criteria."
    actionLabel="Clear Filters"
    onAction={onClearFilters}
    secondaryActionLabel="View All"
    onSecondaryAction={onViewAll}
  />
);

export default EmptyState;
