"""Repository pattern implementation for data access"""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from decimal import Decimal

from ..core.logger import get_logger
from ..core.exceptions import DatabaseError, ValidationError
from .database import DatabaseManager
from .models import Trade, Position, MarketData, PerformanceMetrics, TradeAction, TradeStatus, PositionStatus


class BaseRepository(ABC):
    """Base repository class"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = get_logger(self.__class__.__name__.lower())


class TradeRepository(BaseRepository):
    """Repository for trade data operations"""
    
    def create(self, trade: Trade) -> str:
        """Create a new trade record"""
        try:
            query = '''
                INSERT INTO trades (
                    id, timestamp, symbol, action, price, quantity, value, confidence,
                    order_id, status, fees, stop_loss, take_profit, strategy_name,
                    analysis_data, execution_time, pnl, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                trade.id,
                trade.timestamp.isoformat(),
                trade.symbol,
                trade.action.value,
                float(trade.price),
                float(trade.quantity),
                float(trade.value),
                trade.confidence,
                trade.order_id,
                trade.status.value,
                float(trade.fees),
                float(trade.stop_loss) if trade.stop_loss else None,
                float(trade.take_profit) if trade.take_profit else None,
                trade.strategy_name,
                json.dumps(trade.analysis_data) if trade.analysis_data else None,
                trade.execution_time.isoformat() if trade.execution_time else None,
                float(trade.pnl) if trade.pnl else None,
                trade.notes
            )
            
            self.db.execute_update(query, params)
            self.logger.info(f"Created trade: {trade.id}")
            return trade.id
        
        except Exception as e:
            self.logger.error(f"Failed to create trade: {e}")
            raise DatabaseError(f"Failed to create trade: {e}")
    
    def get_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID"""
        try:
            query = "SELECT * FROM trades WHERE id = ?"
            rows = self.db.execute_query(query, (trade_id,))
            
            if not rows:
                return None
            
            return self._row_to_trade(rows[0])
        
        except Exception as e:
            self.logger.error(f"Failed to get trade {trade_id}: {e}")
            raise DatabaseError(f"Failed to get trade: {e}")
    
    def get_by_symbol(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        """Get trades by symbol"""
        try:
            query = "SELECT * FROM trades WHERE symbol = ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            rows = self.db.execute_query(query, (symbol,))
            return [self._row_to_trade(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"Failed to get trades for {symbol}: {e}")
            raise DatabaseError(f"Failed to get trades: {e}")
    
    def get_by_date_range(self, start_date: datetime, end_date: datetime, 
                         symbol: Optional[str] = None) -> List[Trade]:
        """Get trades within date range"""
        try:
            query = "SELECT * FROM trades WHERE timestamp BETWEEN ? AND ?"
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY timestamp DESC"
            
            rows = self.db.execute_query(query, tuple(params))
            return [self._row_to_trade(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"Failed to get trades by date range: {e}")
            raise DatabaseError(f"Failed to get trades: {e}")
    
    def get_by_status(self, status: TradeStatus, limit: Optional[int] = None) -> List[Trade]:
        """Get trades by status"""
        try:
            query = "SELECT * FROM trades WHERE status = ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            rows = self.db.execute_query(query, (status.value,))
            return [self._row_to_trade(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"Failed to get trades by status {status}: {e}")
            raise DatabaseError(f"Failed to get trades: {e}")
    
    def update(self, trade: Trade) -> bool:
        """Update trade record"""
        try:
            query = '''
                UPDATE trades SET
                    timestamp = ?, symbol = ?, action = ?, price = ?, quantity = ?,
                    value = ?, confidence = ?, order_id = ?, status = ?, fees = ?,
                    stop_loss = ?, take_profit = ?, strategy_name = ?, analysis_data = ?,
                    execution_time = ?, pnl = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            '''
            
            params = (
                trade.timestamp.isoformat(),
                trade.symbol,
                trade.action.value,
                float(trade.price),
                float(trade.quantity),
                float(trade.value),
                trade.confidence,
                trade.order_id,
                trade.status.value,
                float(trade.fees),
                float(trade.stop_loss) if trade.stop_loss else None,
                float(trade.take_profit) if trade.take_profit else None,
                trade.strategy_name,
                json.dumps(trade.analysis_data) if trade.analysis_data else None,
                trade.execution_time.isoformat() if trade.execution_time else None,
                float(trade.pnl) if trade.pnl else None,
                trade.notes,
                trade.id
            )
            
            rows_affected = self.db.execute_update(query, params)
            success = rows_affected > 0
            
            if success:
                self.logger.info(f"Updated trade: {trade.id}")
            else:
                self.logger.warning(f"No trade found to update: {trade.id}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Failed to update trade {trade.id}: {e}")
            raise DatabaseError(f"Failed to update trade: {e}")
    
    def delete(self, trade_id: str) -> bool:
        """Delete trade record"""
        try:
            query = "DELETE FROM trades WHERE id = ?"
            rows_affected = self.db.execute_update(query, (trade_id,))
            success = rows_affected > 0
            
            if success:
                self.logger.info(f"Deleted trade: {trade_id}")
            else:
                self.logger.warning(f"No trade found to delete: {trade_id}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Failed to delete trade {trade_id}: {e}")
            raise DatabaseError(f"Failed to delete trade: {e}")
    
    def get_recent(self, limit: int = 50) -> List[Trade]:
        """Get recent trades"""
        try:
            query = "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?"
            rows = self.db.execute_query(query, (limit,))
            return [self._row_to_trade(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"Failed to get recent trades: {e}")
            raise DatabaseError(f"Failed to get recent trades: {e}")
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for specified period"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    MAX(pnl) as largest_win,
                    MIN(pnl) as largest_loss,
                    SUM(fees) as total_fees
                FROM trades 
                WHERE timestamp >= ? AND status = 'executed' AND pnl IS NOT NULL
            '''
            
            rows = self.db.execute_query(query, (start_date.isoformat(),))
            result = rows[0] if rows else None
            
            if not result or result['total_trades'] == 0:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'total_fees': 0.0
                }
            
            total_trades = result['total_trades']
            winning_trades = result['winning_trades'] or 0
            gross_profit = result['gross_profit'] or 0
            gross_loss = result['gross_loss'] or 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': result['losing_trades'] or 0,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_pnl': result['total_pnl'] or 0,
                'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0,
                'avg_win': result['avg_win'] or 0,
                'avg_loss': result['avg_loss'] or 0,
                'largest_win': result['largest_win'] or 0,
                'largest_loss': result['largest_loss'] or 0,
                'total_fees': result['total_fees'] or 0
            }
        
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            raise DatabaseError(f"Failed to get performance summary: {e}")
    
    def _row_to_trade(self, row) -> Trade:
        """Convert database row to Trade object"""
        return Trade(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            symbol=row['symbol'],
            action=TradeAction(row['action']),
            price=Decimal(str(row['price'])),
            quantity=Decimal(str(row['quantity'])),
            value=Decimal(str(row['value'])),
            confidence=row['confidence'],
            status=TradeStatus(row['status']),
            order_id=row['order_id'],
            fees=Decimal(str(row['fees'])) if row['fees'] else Decimal('0'),
            stop_loss=Decimal(str(row['stop_loss'])) if row['stop_loss'] else None,
            take_profit=Decimal(str(row['take_profit'])) if row['take_profit'] else None,
            strategy_name=row['strategy_name'],
            analysis_data=json.loads(row['analysis_data']) if row['analysis_data'] else {},
            execution_time=datetime.fromisoformat(row['execution_time']) if row['execution_time'] else None,
            pnl=Decimal(str(row['pnl'])) if row['pnl'] else None,
            notes=row['notes']
        )


class PositionRepository(BaseRepository):
    """Repository for position data operations"""
    
    def create(self, position: Position) -> str:
        """Create a new position record"""
        try:
            query = '''
                INSERT INTO positions (
                    id, symbol, quantity, average_price, current_price, market_value,
                    unrealized_pnl, unrealized_pnl_percent, status, opened_at,
                    last_updated, stop_loss, take_profit, trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                position.id,
                position.symbol,
                float(position.quantity),
                float(position.average_price),
                float(position.current_price),
                float(position.market_value),
                float(position.unrealized_pnl),
                position.unrealized_pnl_percent,
                position.status.value,
                position.opened_at.isoformat(),
                position.last_updated.isoformat(),
                float(position.stop_loss) if position.stop_loss else None,
                float(position.take_profit) if position.take_profit else None,
                json.dumps(position.trades)
            )
            
            self.db.execute_update(query, params)
            self.logger.info(f"Created position: {position.id}")
            return position.id
        
        except Exception as e:
            self.logger.error(f"Failed to create position: {e}")
            raise DatabaseError(f"Failed to create position: {e}")
    
    def get_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        try:
            query = "SELECT * FROM positions WHERE symbol = ? AND status = 'open'"
            rows = self.db.execute_query(query, (symbol,))
            
            if not rows:
                return None
            
            return self._row_to_position(rows[0])
        
        except Exception as e:
            self.logger.error(f"Failed to get position for {symbol}: {e}")
            raise DatabaseError(f"Failed to get position: {e}")
    
    def get_all_open(self) -> List[Position]:
        """Get all open positions"""
        try:
            query = "SELECT * FROM positions WHERE status = 'open' ORDER BY opened_at DESC"
            rows = self.db.execute_query(query)
            return [self._row_to_position(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"Failed to get open positions: {e}")
            raise DatabaseError(f"Failed to get open positions: {e}")
    
    def update(self, position: Position) -> bool:
        """Update position record"""
        try:
            query = '''
                UPDATE positions SET
                    quantity = ?, average_price = ?, current_price = ?, market_value = ?,
                    unrealized_pnl = ?, unrealized_pnl_percent = ?, status = ?,
                    last_updated = ?, stop_loss = ?, take_profit = ?, trades = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            '''
            
            params = (
                float(position.quantity),
                float(position.average_price),
                float(position.current_price),
                float(position.market_value),
                float(position.unrealized_pnl),
                position.unrealized_pnl_percent,
                position.status.value,
                position.last_updated.isoformat(),
                float(position.stop_loss) if position.stop_loss else None,
                float(position.take_profit) if position.take_profit else None,
                json.dumps(position.trades),
                position.id
            )
            
            rows_affected = self.db.execute_update(query, params)
            success = rows_affected > 0
            
            if success:
                self.logger.info(f"Updated position: {position.id}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Failed to update position {position.id}: {e}")
            raise DatabaseError(f"Failed to update position: {e}")
    
    def _row_to_position(self, row) -> Position:
        """Convert database row to Position object"""
        return Position(
            id=row['id'],
            symbol=row['symbol'],
            quantity=Decimal(str(row['quantity'])),
            average_price=Decimal(str(row['average_price'])),
            current_price=Decimal(str(row['current_price'])),
            market_value=Decimal(str(row['market_value'])),
            unrealized_pnl=Decimal(str(row['unrealized_pnl'])),
            unrealized_pnl_percent=row['unrealized_pnl_percent'],
            status=PositionStatus(row['status']),
            opened_at=datetime.fromisoformat(row['opened_at']),
            last_updated=datetime.fromisoformat(row['last_updated']),
            stop_loss=Decimal(str(row['stop_loss'])) if row['stop_loss'] else None,
            take_profit=Decimal(str(row['take_profit'])) if row['take_profit'] else None,
            trades=json.loads(row['trades']) if row['trades'] else []
        )


class MarketDataRepository(BaseRepository):
    """Repository for market data operations"""
    
    def create(self, market_data: MarketData) -> int:
        """Create market data record"""
        try:
            query = '''
                INSERT OR REPLACE INTO market_data (
                    symbol, timestamp, open_price, high_price, low_price,
                    close_price, volume, quote_volume, indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                market_data.symbol,
                market_data.timestamp.isoformat(),
                float(market_data.open_price),
                float(market_data.high_price),
                float(market_data.low_price),
                float(market_data.close_price),
                float(market_data.volume),
                float(market_data.quote_volume),
                json.dumps(market_data.indicators) if market_data.indicators else None
            )
            
            self.db.execute_update(query, params)
            return 1  # Success indicator
        
        except Exception as e:
            self.logger.error(f"Failed to create market data: {e}")
            raise DatabaseError(f"Failed to create market data: {e}")
    
    def get_latest(self, symbol: str, limit: int = 100) -> List[MarketData]:
        """Get latest market data for symbol"""
        try:
            query = '''
                SELECT * FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            rows = self.db.execute_query(query, (symbol, limit))
            return [self._row_to_market_data(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            raise DatabaseError(f"Failed to get market data: {e}")
    
    def _row_to_market_data(self, row) -> MarketData:
        """Convert database row to MarketData object"""
        return MarketData(
            symbol=row['symbol'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            open_price=Decimal(str(row['open_price'])),
            high_price=Decimal(str(row['high_price'])),
            low_price=Decimal(str(row['low_price'])),
            close_price=Decimal(str(row['close_price'])),
            volume=Decimal(str(row['volume'])),
            quote_volume=Decimal(str(row['quote_volume'])),
            indicators=json.loads(row['indicators']) if row['indicators'] else {}
        )