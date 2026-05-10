import React, { Component } from 'react';
import { Box, Button, Card, CardContent, Typography, Stack } from '@mui/material';
import { Error as ErrorIcon, Refresh as RefreshIcon } from '@mui/icons-material';

/**
 * Error Boundary component to catch JavaScript errors anywhere in the component tree
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    this.setState(prevState => ({
      error,
      errorInfo,
      errorCount: prevState.errorCount + 1
    }));

    // You can also log the error to an error reporting service here
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });

    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: this.props.fullScreen ? '100vh' : 400,
            p: 3,
            backgroundColor: this.props.fullScreen ? 'background.default' : 'transparent'
          }}
        >
          <Card sx={{ maxWidth: 600, width: '100%' }}>
            <CardContent>
              <Stack spacing={3} alignItems="center" textAlign="center">
                <ErrorIcon
                  sx={{
                    fontSize: 80,
                    color: 'error.main',
                    opacity: 0.8
                  }}
                />

                <Typography variant="h4" fontWeight={600}>
                  {this.props.title || 'Oops! Something went wrong'}
                </Typography>

                <Typography variant="body1" color="text.secondary">
                  {this.props.message ||
                    'An unexpected error occurred. Please try refreshing the page or contact support if the problem persists.'}
                </Typography>

                {this.props.showDetails && this.state.error && (
                  <Box
                    sx={{
                      width: '100%',
                      p: 2,
                      backgroundColor: 'action.hover',
                      borderRadius: 1,
                      textAlign: 'left'
                    }}
                  >
                    <Typography variant="caption" component="div" sx={{ mb: 1, fontWeight: 600 }}>
                      Error Details:
                    </Typography>
                    <Typography
                      variant="caption"
                      component="pre"
                      sx={{
                        overflow: 'auto',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        fontFamily: 'monospace',
                        fontSize: '0.7rem'
                      }}
                    >
                      {this.state.error.toString()}
                      {this.state.errorInfo && (
                        <>
                          {'\n\n'}
                          {this.state.errorInfo.componentStack}
                        </>
                      )}
                    </Typography>
                  </Box>
                )}

                {this.state.errorCount > 2 && (
                  <Typography variant="body2" color="warning.main">
                    Multiple errors detected. Consider reloading the page.
                  </Typography>
                )}

                <Stack direction="row" spacing={2}>
                  <Button
                    variant="contained"
                    startIcon={<RefreshIcon />}
                    onClick={this.handleReset}
                  >
                    Try Again
                  </Button>

                  {this.props.showReload && (
                    <Button
                      variant="outlined"
                      onClick={this.handleReload}
                    >
                      Reload Page
                    </Button>
                  )}
                </Stack>

                {this.props.supportEmail && (
                  <Typography variant="caption" color="text.secondary">
                    Need help? Contact us at{' '}
                    <a
                      href={`mailto:${this.props.supportEmail}`}
                      style={{ color: 'inherit', textDecoration: 'underline' }}
                    >
                      {this.props.supportEmail}
                    </a>
                  </Typography>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Box>
      );
    }

    return this.props.children;
  }
}

ErrorBoundary.defaultProps = {
  fullScreen: false,
  showDetails: process.env.NODE_ENV === 'development',
  showReload: true
};

/**
 * Higher-order component to wrap components with error boundary
 */
export const withErrorBoundary = (Component, errorBoundaryProps = {}) => {
  const WrappedComponent = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name || 'Component'})`;

  return WrappedComponent;
};

export default ErrorBoundary;
