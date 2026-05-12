import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Chip
} from '@mui/material';
import { 
  Wallet as WalletIcon,
  ArrowDropUp as ArrowUpIcon,
  ArrowDropDown as ArrowDownIcon
} from '@mui/icons-material';

export default function PortfolioSummary({ portfolio, loading }) {
  if (loading) {
    return (
      <Card sx={{ boxShadow: 3, height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Portfolio Summary
          </Typography>
          <LinearProgress />
        </CardContent>
      </Card>
    );
  }

  const positions = portfolio?.positions || [];
  
  return (
    <Card sx={{ boxShadow: 3, height: '100%', overflow: 'auto' }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={1}>
          <WalletIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Portfolio Summary
          </Typography>
        </Box>
        
        <Box display="flex" flexWrap="wrap" gap={2} mb={2}>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Total Value
            </Typography>
            <Typography variant="h6">
              ${Number(portfolio?.total_value || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="body2" color="text.secondary">
              USDT Balance
            </Typography>
            <Typography variant="h6">
              ${Number(portfolio?.balance || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}
            </Typography>
          </Box>
          
          <Box>
            <Typography variant="body2" color="text.secondary">
              Unrealized P&L
            </Typography>
            <Box display="flex" alignItems="center">
              <Typography 
                variant="h6" 
                color={portfolio?.unrealized_pnl > 0 ? "success.main" : portfolio?.unrealized_pnl < 0 ? "error.main" : "text.primary"}
              >
                ${Number(portfolio?.unrealized_pnl || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}
              </Typography>
              {portfolio?.unrealized_pnl > 0 ? <ArrowUpIcon color="success" /> : 
               portfolio?.unrealized_pnl < 0 ? <ArrowDownIcon color="error" /> : null}
            </Box>
          </Box>
        </Box>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="subtitle2" gutterBottom>
          Current Positions
        </Typography>
        
        {positions.length > 0 ? (
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell align="right">Quantity</TableCell>
                  <TableCell align="right">Value</TableCell>
                  <TableCell align="right">P&L</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((position) => (
                  <TableRow key={position.symbol}>
                    <TableCell component="th" scope="row">
                      {position.symbol}
                    </TableCell>
                    <TableCell align="right">
                      {position.quantity.toLocaleString(undefined, {maximumFractionDigits: 6})}
                    </TableCell>
                    <TableCell align="right">
                      ${position.market_value.toLocaleString(undefined, {minimumFractionDigits: 2})}
                    </TableCell>
                    <TableCell align="right">
                      <Box display="flex" alignItems="center" justifyContent="flex-end">
                        <Chip 
                          label={`${position.unrealized_pnl_percent.toFixed(2)}%`}
                          size="small"
                          color={position.unrealized_pnl > 0 ? "success" : position.unrealized_pnl < 0 ? "error" : "default"}
                          variant="outlined"
                        />
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : (
          <Typography variant="body2" color="text.secondary" align="center" sx={{ py: 2 }}>
            No open positions
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}