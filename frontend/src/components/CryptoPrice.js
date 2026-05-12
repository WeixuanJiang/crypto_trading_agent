import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Divider, 
  Skeleton,
  Chip 
} from '@mui/material';
import { 
  TrendingUp as TrendingUpIcon, 
  TrendingDown as TrendingDownIcon 
} from '@mui/icons-material';
import { tradingService } from '../services/apiService';

export default function CryptoPrice({ symbol }) {
  const [priceData, setPriceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        setLoading(true);
        const response = await tradingService.getMarketPrice(symbol);
        setPriceData(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching price for', symbol, err);
        setError(`Could not fetch price for ${symbol}`);
      } finally {
        setLoading(false);
      }
    };

    fetchPrice();
    const interval = setInterval(fetchPrice, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [symbol]);

  if (loading) {
    return (
      <Card sx={{ height: '100%', boxShadow: 3 }}>
        <CardContent>
          <Typography variant="h6">{symbol}</Typography>
          <Box mt={1}>
            <Skeleton variant="text" width="80%" height={40} />
            <Skeleton variant="text" width="40%" />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ height: '100%', boxShadow: 3 }}>
        <CardContent>
          <Typography variant="h6">{symbol}</Typography>
          <Typography color="error">{error}</Typography>
        </CardContent>
      </Card>
    );
  }

  const isPositive = priceData?.price_change_percent >= 0;
  const priceColor = isPositive ? 'success.main' : 'error.main';
  const PriceIcon = isPositive ? TrendingUpIcon : TrendingDownIcon;

  return (
    <Card sx={{ height: '100%', boxShadow: 3 }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">{symbol}</Typography>
          <Chip 
            label={`Vol: ${(priceData?.volume_24h || 0).toLocaleString()}`} 
            size="small" 
            variant="outlined"
          />
        </Box>
        
        <Box mt={2} mb={1}>
          <Typography variant="h4" component="div">
            ${Number(priceData?.price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
          </Typography>
          
          <Box display="flex" alignItems="center" mt={1}>
            <PriceIcon fontSize="small" sx={{ color: priceColor, mr: 0.5 }} />
            <Typography variant="body1" sx={{ color: priceColor }}>
              {priceData?.price_change_percent.toFixed(2)}%
            </Typography>
          </Box>
        </Box>
        
        <Divider sx={{ my: 1 }} />
        
        <Box display="flex" justifyContent="space-between">
          <Typography variant="body2" color="text.secondary">
            24h High
          </Typography>
          <Typography variant="body2">
            ${Number(priceData?.high_24h).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
          </Typography>
        </Box>
        
        <Box display="flex" justifyContent="space-between" mt={0.5}>
          <Typography variant="body2" color="text.secondary">
            24h Low
          </Typography>
          <Typography variant="body2">
            ${Number(priceData?.low_24h).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}