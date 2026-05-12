import React, { useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Chip,
  TablePagination,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  ArrowDropUp as ArrowUpIcon,
  ArrowDropDown as ArrowDownIcon,
  CheckCircle as CheckCircleIcon,
  Edit as EditIcon
} from '@mui/icons-material';

export default function TradeHistoryTable({ trades = [], statistics = {} }) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Format date string
  const formatDate = (dateString) => {
    if (!dateString) return '';
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (e) {
      return dateString;
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Trading Performance
        </Typography>
        <Box 
          sx={{ 
            display: 'flex', 
            flexWrap: 'wrap', 
            gap: 2,
            '& > *': { minWidth: { xs: '100%', sm: '48%', md: '30%', lg: '23%' } }
          }}
        >
          <Paper sx={{ p: 2 }}>
            <Typography variant="body2" color="text.secondary">Win Rate</Typography>
            <Typography variant="h6">{statistics.win_rate?.toFixed(1)}%</Typography>
          </Paper>
          
          <Paper sx={{ p: 2 }}>
            <Typography variant="body2" color="text.secondary">Total P&L</Typography>
            <Box display="flex" alignItems="center">
              <Typography 
                variant="h6" 
                color={statistics.total_pnl > 0 ? 'success.main' : statistics.total_pnl < 0 ? 'error.main' : 'inherit'}
              >
                ${Number(statistics.total_pnl || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}
              </Typography>
              {statistics.total_pnl > 0 ? <ArrowUpIcon color="success" /> : 
               statistics.total_pnl < 0 ? <ArrowDownIcon color="error" /> : null}
            </Box>
          </Paper>
          
          <Paper sx={{ p: 2 }}>
            <Typography variant="body2" color="text.secondary">Total Trades</Typography>
            <Typography variant="h6">{statistics.total_trades || 0}</Typography>
          </Paper>
          
          <Paper sx={{ p: 2 }}>
            <Typography variant="body2" color="text.secondary">Avg P&L per Trade</Typography>
            <Typography 
              variant="h6" 
              color={statistics.avg_pnl_per_trade > 0 ? 'success.main' : statistics.avg_pnl_per_trade < 0 ? 'error.main' : 'inherit'}
            >
              ${Number(statistics.avg_pnl_per_trade || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}
            </Typography>
          </Paper>
        </Box>
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table sx={{ minWidth: 650 }} size="small">
          <TableHead>
            <TableRow>
              <TableCell>Time</TableCell>
              <TableCell>Symbol</TableCell>
              <TableCell>Action</TableCell>
              <TableCell align="right">Price</TableCell>
              <TableCell align="right">Size</TableCell>
              <TableCell align="right">P&L</TableCell>
              <TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {trades
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((trade, index) => (
                <TableRow key={trade.id || index} hover>
                  <TableCell component="th" scope="row">
                    {formatDate(trade.timestamp)}
                  </TableCell>
                  <TableCell>{trade.symbol}</TableCell>
                  <TableCell>
                    <Chip 
                      label={trade.action}
                      size="small"
                      color={trade.action === 'BUY' ? 'success' : trade.action === 'SELL' ? 'error' : 'default'}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="right">
                    ${Number(trade.price).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 6})}
                  </TableCell>
                  <TableCell align="right">
                    {Number(trade.size).toLocaleString(undefined, {maximumFractionDigits: 8})}
                  </TableCell>
                  <TableCell align="right">
                    <Box display="flex" alignItems="center" justifyContent="flex-end">
                      {trade.pnl === 'N/A' ? (
                        <Typography variant="body2" color="text.secondary">N/A</Typography>
                      ) : (
                        <>
                          <Typography 
                            variant="body2" 
                            color={trade.pnl && trade.pnl.includes('-') ? 'error.main' : 'success.main'}
                            sx={{ mr: 0.5 }}
                          >
                            {trade.pnl}
                          </Typography>
                        </>
                      )}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center">
                      {trade.status === 'executed' ? (
                        <Tooltip title="Trade executed">
                          <CheckCircleIcon color="success" fontSize="small" />
                        </Tooltip>
                      ) : trade.status === 'paper_trade' ? (
                        <Tooltip title="Paper trade (simulation)">
                          <EditIcon color="info" fontSize="small" />
                        </Tooltip>
                      ) : (
                        <Chip 
                          label={trade.status} 
                          size="small"
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </TableCell>
                </TableRow>
            ))}
            {trades.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                    No trade history available
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      
      <TablePagination
        rowsPerPageOptions={[10, 25, 50]}
        component="div"
        count={trades.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Box>
  );
}