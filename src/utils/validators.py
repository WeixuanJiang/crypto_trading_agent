"""
Input validation utilities for the Crypto Trading Agent

This module provides comprehensive validation for all user inputs,
API responses, and configuration values to ensure data integrity
and prevent security vulnerabilities.
"""

import re
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal, InvalidOperation
from datetime import datetime

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class InputValidator:
    """Comprehensive input validation utilities"""
    
    # Valid trading pair pattern (e.g., BTC-USDT, ETH-BTC)
    TRADING_PAIR_PATTERN = re.compile(r'^[A-Z0-9]{2,10}-[A-Z0-9]{2,10}$')
    
    # Valid order actions
    VALID_ACTIONS = {'BUY', 'SELL', 'HOLD'}
    
    # Valid order statuses
    VALID_STATUSES = {'executed', 'paper_trade', 'cancelled', 'pending', 'failed'}
    
    @staticmethod
    def validate_trading_pair(symbol: str) -> str:
        """
        Validate trading pair symbol format
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            
        Returns:
            Validated and normalized symbol
            
        Raises:
            ValidationError: If symbol format is invalid
        """
        if not isinstance(symbol, str):
            raise ValidationError(f"Trading pair must be a string, got {type(symbol)}")
        
        symbol = symbol.upper().strip()
        
        if not InputValidator.TRADING_PAIR_PATTERN.match(symbol):
            raise ValidationError(f"Invalid trading pair format: {symbol}")
        
        return symbol
    
    @staticmethod
    def validate_price(price: Union[str, int, float, Decimal]) -> float:
        """
        Validate and convert price to float
        
        Args:
            price: Price value
            
        Returns:
            Validated price as float
            
        Raises:
            ValidationError: If price is invalid
        """
        try:
            price_decimal = Decimal(str(price))
            price_float = float(price_decimal)
            
            if price_float <= 0:
                raise ValidationError(f"Price must be positive, got {price_float}")
            
            if price_float > 1e10:  # Reasonable upper limit
                raise ValidationError(f"Price too large: {price_float}")
            
            return price_float
            
        except (InvalidOperation, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid price format: {price} - {e}")
    
    @staticmethod
    def validate_quantity(quantity: Union[str, int, float, Decimal]) -> float:
        """
        Validate and convert quantity to float
        
        Args:
            quantity: Quantity value
            
        Returns:
            Validated quantity as float
            
        Raises:
            ValidationError: If quantity is invalid
        """
        try:
            quantity_decimal = Decimal(str(quantity))
            quantity_float = float(quantity_decimal)
            
            if quantity_float <= 0:
                raise ValidationError(f"Quantity must be positive, got {quantity_float}")
            
            if quantity_float > 1e10:  # Reasonable upper limit
                raise ValidationError(f"Quantity too large: {quantity_float}")
            
            return quantity_float
            
        except (InvalidOperation, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid quantity format: {quantity} - {e}")
    
    @staticmethod
    def validate_confidence(confidence: Union[str, int, float]) -> float:
        """
        Validate confidence score
        
        Args:
            confidence: Confidence value (0-1)
            
        Returns:
            Validated confidence as float
            
        Raises:
            ValidationError: If confidence is invalid
        """
        try:
            conf_float = float(confidence)
            
            if not (0 <= conf_float <= 1):
                raise ValidationError(f"Confidence must be between 0 and 1, got {conf_float}")
            
            return conf_float
            
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid confidence format: {confidence} - {e}")
    
    @staticmethod
    def validate_action(action: str) -> str:
        """
        Validate trading action
        
        Args:
            action: Trading action
            
        Returns:
            Validated action
            
        Raises:
            ValidationError: If action is invalid
        """
        if not isinstance(action, str):
            raise ValidationError(f"Action must be a string, got {type(action)}")
        
        action = action.upper().strip()
        
        if action not in InputValidator.VALID_ACTIONS:
            raise ValidationError(f"Invalid action: {action}. Must be one of {InputValidator.VALID_ACTIONS}")
        
        return action
    
    @staticmethod
    def validate_status(status: str) -> str:
        """
        Validate trade status
        
        Args:
            status: Trade status
            
        Returns:
            Validated status
            
        Raises:
            ValidationError: If status is invalid
        """
        if not isinstance(status, str):
            raise ValidationError(f"Status must be a string, got {type(status)}")
        
        status = status.lower().strip()
        
        if status not in InputValidator.VALID_STATUSES:
            raise ValidationError(f"Invalid status: {status}. Must be one of {InputValidator.VALID_STATUSES}")
        
        return status
    
    @staticmethod
    def validate_percentage(percentage: Union[str, int, float], min_val: float = 0, max_val: float = 100) -> float:
        """
        Validate percentage value
        
        Args:
            percentage: Percentage value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated percentage as float
            
        Raises:
            ValidationError: If percentage is invalid
        """
        try:
            pct_float = float(percentage)
            
            if not (min_val <= pct_float <= max_val):
                raise ValidationError(f"Percentage must be between {min_val} and {max_val}, got {pct_float}")
            
            return pct_float
            
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid percentage format: {percentage} - {e}")
    
    @staticmethod
    def validate_timestamp(timestamp: Union[str, datetime]) -> datetime:
        """
        Validate and convert timestamp
        
        Args:
            timestamp: Timestamp string or datetime object
            
        Returns:
            Validated datetime object
            
        Raises:
            ValidationError: If timestamp is invalid
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try common formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
                        try:
                            return datetime.strptime(timestamp, fmt)
                        except ValueError:
                            continue
                    raise ValueError("No matching format found")
                except ValueError as e:
                    raise ValidationError(f"Invalid timestamp format: {timestamp} - {e}")
        
        raise ValidationError(f"Timestamp must be string or datetime, got {type(timestamp)}")
    
    @staticmethod
    def validate_api_response(response: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
        """
        Validate API response structure
        
        Args:
            response: API response dictionary
            required_fields: List of required field names
            
        Returns:
            Validated response
            
        Raises:
            ValidationError: If response is invalid
        """
        if not isinstance(response, dict):
            raise ValidationError(f"API response must be a dictionary, got {type(response)}")
        
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            raise ValidationError(f"Missing required fields in API response: {missing_fields}")
        
        return response
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 255) -> str:
        """
        Sanitize string input
        
        Args:
            value: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            ValidationError: If string is invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"Value must be a string, got {type(value)}")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', value.strip())
        
        if len(sanitized) > max_length:
            raise ValidationError(f"String too long: {len(sanitized)} > {max_length}")
        
        return sanitized
    
    @staticmethod
    def validate_trade_data(trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete trade data structure
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            Validated trade data
            
        Raises:
            ValidationError: If trade data is invalid
        """
        required_fields = ['symbol', 'action', 'price', 'size', 'confidence']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in trade_data]
        if missing_fields:
            raise ValidationError(f"Missing required trade fields: {missing_fields}")
        
        # Validate individual fields
        validated_data = {}
        validated_data['symbol'] = InputValidator.validate_trading_pair(trade_data['symbol'])
        validated_data['action'] = InputValidator.validate_action(trade_data['action'])
        validated_data['price'] = InputValidator.validate_price(trade_data['price'])
        validated_data['size'] = InputValidator.validate_quantity(trade_data['size'])
        validated_data['confidence'] = InputValidator.validate_confidence(trade_data['confidence'])
        
        # Optional fields
        if 'status' in trade_data:
            validated_data['status'] = InputValidator.validate_status(trade_data['status'])
        
        if 'timestamp' in trade_data:
            validated_data['timestamp'] = InputValidator.validate_timestamp(trade_data['timestamp'])
        
        if 'fees' in trade_data:
            validated_data['fees'] = InputValidator.validate_price(trade_data['fees'])
        
        return validated_data

# Convenience functions for common validations
def validate_trading_pair(symbol: str) -> str:
    """Validate trading pair symbol"""
    return InputValidator.validate_trading_pair(symbol)

def validate_price(price: Union[str, int, float]) -> float:
    """Validate price value"""
    return InputValidator.validate_price(price)

def validate_confidence(confidence: Union[str, int, float]) -> float:
    """Validate confidence score"""
    return InputValidator.validate_confidence(confidence)