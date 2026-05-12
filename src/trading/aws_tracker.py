"""
AWS Trade Tracker Wrapper - Non-AWS Version

This module provides a wrapper around the local TradeTracker to maintain
compatibility with the main application while removing AWS dependencies
(except for Bedrock which is kept for LLM functionality).
"""

from typing import Dict
from .tracker import TradeTracker
from ..notifications.service import NotificationService

class EnhancedTradeTracker(TradeTracker):
    """
    Enhanced TradeTracker with notification capabilities
    
    This class extends the base TradeTracker with notification methods
    to maintain compatibility with the main application.
    """
    
    def __init__(self, data_dir: str = "trading_data", db_manager=None, trade_repository=None):
        super().__init__(data_dir, db_manager, trade_repository)
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

    def get_all_trades(self) -> Dict:
        """
        Get all trades as a dictionary

        Returns:
            Dict: Dictionary of all trades indexed by trade ID
        """
        trades_dict = {}
        for trade in self.trades:
            # Convert Trade dataclass to dict
            trade_dict = {
                'id': trade.id,
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.action,
                'action': trade.action,
                'amount': trade.size,
                'price': trade.price,
                'value': trade.value,
                'confidence': trade.confidence,
                'order_id': trade.order_id,
                'status': 'closed' if trade.is_closed else 'open',
                'fees': trade.fees,
                'pnl': trade.pnl if trade.pnl is not None else 0.0,
                'pnl_percent': trade.pnl_percent if trade.pnl_percent is not None else 0.0,
                'strategy': 'hybrid',  # Default strategy
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'exit_timestamp': trade.exit_timestamp,
                'exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason
            }
            trades_dict[trade.id] = trade_dict

        return trades_dict

    def get_performance_summary(self, days: int = 30) -> Dict:
        """
        Alias for get_performance_metrics for API compatibility

        Args:
            days: Number of days to include in performance calculation

        Returns:
            Dict: Performance metrics and statistics
        """
        return self.get_performance_metrics(days)

def create_trade_tracker(db_manager=None, trade_repository=None):
    """
    Create and return an EnhancedTradeTracker instance
    
    This function maintains compatibility with the existing codebase
    while using the local TradeTracker with notification capabilities.
    """
    return EnhancedTradeTracker(db_manager=db_manager, trade_repository=trade_repository)
