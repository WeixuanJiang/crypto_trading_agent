"""
AWS Trade Tracker Wrapper - Non-AWS Version

This module provides a wrapper around the local TradeTracker to maintain
compatibility with the main application while removing AWS dependencies
(except for Bedrock which is kept for LLM functionality).
"""

from .tracker import TradeTracker
from ..notifications.service import NotificationService

class EnhancedTradeTracker(TradeTracker):
    """
    Enhanced TradeTracker with notification capabilities
    
    This class extends the base TradeTracker with notification methods
    to maintain compatibility with the main application.
    """
    
    def __init__(self, data_dir: str = "trading_data"):
        super().__init__(data_dir)
        self.notification_service = NotificationService()
    
    def send_system_alert(self, message: str, subject: str = "Crypto Trading Agent Alert"):
        """
        Send system alert notification
        
        Args:
            message: Alert message content
            subject: Alert subject/title
        """
        try:
            return self.notification_service.send_system_notification(message, subject)
        except Exception as e:
            print(f"⚠️ Failed to send system alert: {e}")
            # Still log to console as fallback
            print(f"\n{subject}")
            print(message)
            return None
    
    def add_trade(self, **kwargs):
        """
        Add trade with notification support
        
        This method extends the base log_trade method to include notifications.
        """
        # Use the parent class method to log the trade
        trade_id = self.log_trade(
            symbol=kwargs.get('symbol'),
            action=kwargs.get('action'),
            price=kwargs.get('price'),
            size=kwargs.get('size'),
            confidence=kwargs.get('confidence', 0.0),
            order_id=kwargs.get('order_id'),
            status=kwargs.get('status', 'executed'),
            fees=kwargs.get('fees', 0.0)
        )
        
        # Send trade notification if enabled
        try:
            trade_data = {
                'symbol': kwargs.get('symbol'),
                'action': kwargs.get('action'),
                'quantity': kwargs.get('size'),
                'price': kwargs.get('price'),
                'total_value': kwargs.get('value', kwargs.get('price', 0) * kwargs.get('size', 0)),
                'confidence': kwargs.get('confidence', 0.0),
                'pnl': kwargs.get('pnl', 0.0),
                'portfolio_pnl': self.calculate_portfolio_pnl()
            }
            
            self.notification_service.send_trade_notification(trade_data)
        except Exception as e:
            print(f"⚠️ Failed to send trade notification: {e}")
        
        return trade_id

def create_trade_tracker():
    """
    Create and return an EnhancedTradeTracker instance
    
    This function maintains compatibility with the existing codebase
    while using the local TradeTracker with notification capabilities.
    """
    return EnhancedTradeTracker()
