import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SnackbarProvider } from 'notistack';
import Dashboard from './pages/DashboardNew';
import Analytics from './pages/Analytics';
import TradingHistory from './pages/TradingHistoryNew';
import Settings from './pages/Settings';
import TradingSettings from './pages/TradingSettings';
import TradingControl from './pages/TradingControl';
import Layout from './components/Layout';
import ErrorBoundary from './components/common/ErrorBoundary';
import darkTheme from './theme/darkTheme';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5000,
    },
  },
});

function App() {
  return (
    <ErrorBoundary fullScreen>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={darkTheme}>
          <SnackbarProvider
            maxSnack={3}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            autoHideDuration={4000}
          >
            <CssBaseline />
            <Layout>
              <Routes>
                <Route path="/"                element={<Dashboard />} />
                <Route path="/analytics"       element={<Analytics />} />
                <Route path="/history"         element={<TradingHistory />} />
                <Route path="/control"         element={<TradingControl />} />
                <Route path="/trading-settings" element={<TradingSettings />} />
                <Route path="/settings"        element={<Settings />} />
              </Routes>
            </Layout>
          </SnackbarProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
