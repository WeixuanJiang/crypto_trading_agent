import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class NotificationService:
    """
    Local notification service without AWS dependencies
    
    This service provides local logging and console notifications
    for trade and system events.
    """
    
    def __init__(self):
        self.notification_log = "notifications.log"
        logger.info("✅ Local notification service initialized")
    
    def upload_trades_locally(self, file_path: str) -> str:
        """
        Return local file path for trade data
        
        This maintains compatibility with existing code while
        keeping files local.
        """
        logger.info(f"📁 Trades file available locally: {file_path}")
        return f"local://{file_path}"
    
    def send_trade_notification(self, trade_data: Dict, file_url: Optional[str] = None) -> str:
        """Send trade notification to console and log file"""
        try:
            message = self._format_trade_message(trade_data, file_url)
            subject = f"🚀 Crypto Trade Executed: {trade_data.get('symbol', 'Unknown')}"
            
            # Log to console
            print(f"\n{subject}")
            print(message)
            
            # Log to file
            self._log_notification(subject, message)
            
            notification_id = f"local_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"✅ Trade notification logged: {notification_id}")
            return notification_id
            
        except Exception as e:
            logger.error(f"❌ Failed to send trade notification: {e}")
            raise
    
    def _format_trade_message(self, trade_data: Dict, file_url: Optional[str] = None) -> str:
        """Format trade data into notification message"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        message = f"""
🤖 Crypto Trading Agent - Trade Executed

📊 Trade Details:
• Symbol: {trade_data.get('symbol', 'N/A')}
• Action: {trade_data.get('action', 'N/A').upper()}
• Quantity: {trade_data.get('quantity', 'N/A')}
• Price: ${trade_data.get('price', 'N/A')}
• Total Value: ${trade_data.get('total_value', 'N/A')}
• Confidence: {trade_data.get('confidence', 'N/A')}%

💰 P&L Information:
• Trade P&L: ${trade_data.get('pnl', 'N/A')}
• Portfolio P&L: ${trade_data.get('portfolio_pnl', 'N/A')}

🕐 Timestamp: {timestamp}

📈 Trading Data: {file_url if file_url else 'Available locally'}

---
Crypto Trading Agent v1.0
Running locally
        """.strip()
        
        return message
    
    def send_system_notification(self, message: str, subject: str = "Crypto Trading Agent Alert") -> str:
        """Send general system notification to console and log"""
        try:
            # Log to console
            print(f"\n{subject}")
            print(message)
            
            # Log to file
            self._log_notification(subject, message)
            
            notification_id = f"local_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"✅ System notification logged: {notification_id}")
            return notification_id
            
        except Exception as e:
            logger.error(f"❌ Failed to send system notification: {e}")
            raise
    
    def _log_notification(self, subject: str, message: str):
        """Log notification to file"""
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            log_entry = {
                "timestamp": timestamp,
                "subject": subject,
                "message": message
            }
            
            with open(self.notification_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"❌ Failed to log notification: {e}")