import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';

export default function StatusCard({ title, value, icon, color, subValue, chipText, chipColor }) {
  return (
    <Card sx={{ height: '100%', boxShadow: 3 }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={1}>
          {icon && (
            <Box mr={1} sx={{ color: color || 'primary.main' }}>
              {icon}
            </Box>
          )}
          <Typography variant="h6" component="div" color="text.secondary">
            {title}
          </Typography>
        </Box>
        
        <Typography variant="h4" component="div" sx={{ mb: 1, color: color }}>
          {value}
        </Typography>
        
        {subValue && (
          <Typography variant="body2" color="text.secondary">
            {subValue}
          </Typography>
        )}
        
        {chipText && (
          <Box mt={1}>
            <Chip 
              label={chipText}
              size="small"
              color={chipColor || "default"}
              variant="outlined"
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );
}