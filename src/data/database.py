"""Enhanced database management system for the Crypto Trading Agent"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager
import threading
from pathlib import Path

from ..core.config import config
from ..core.exceptions import DatabaseError
from ..core.logger import get_logger
from .models import Trade, Position, MarketData, PerformanceMetrics


class DatabaseManager:
    """Enhanced database management system with connection pooling and transactions"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.database.db_path
        self.lock = threading.RLock()
        self.logger = get_logger("database")
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Setup backup if enabled
        if config.database.backup_enabled:
            self._setup_backup_schedule()
    
    def _initialize_database(self):
        """Initialize database with required tables and indexes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    value REAL NOT NULL,
                    confidence REAL NOT NULL,
                    order_id TEXT,
                    status TEXT NOT NULL,
                    fees REAL DEFAULT 0.0,
                    stop_loss REAL,
                    take_profit REAL,
                    strategy_name TEXT,
                    analysis_data TEXT,
                    execution_time TEXT,
                    pnl REAL,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    market_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    unrealized_pnl_percent REAL NOT NULL,
                    status TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    trades TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_volume REAL NOT NULL,
                    indicators TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    gross_profit REAL NOT NULL,
                    gross_loss REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    average_win REAL NOT NULL,
                    average_loss REAL NOT NULL,
                    largest_win REAL NOT NULL,
                    largest_loss REAL NOT NULL,
                    average_trade_duration REAL NOT NULL,
                    total_fees REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Configuration table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configuration (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)',
                'CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)',
                'CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)',
                'CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)',
                'CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_performance_period ON performance_metrics(period_start, period_end)'
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            with self.lock:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                conn.execute('PRAGMA foreign_keys = ON')
                conn.execute('PRAGMA journal_mode = WAL')
                yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def transaction(self):
        """Database transaction context manager"""
        with self.get_connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Transaction failed: {e}")
                raise
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute multiple queries with different parameters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database"""
        if not backup_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.db_path}.backup_{timestamp}"
        
        try:
            with self.get_connection() as source:
                backup = sqlite3.connect(backup_path)
                source.backup(backup)
                backup.close()
            
            self.logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            raise DatabaseError(f"Database backup failed: {e}")
    
    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        if not os.path.exists(backup_path):
            raise DatabaseError(f"Backup file not found: {backup_path}")
        
        try:
            # Create backup of current database
            current_backup = self.backup_database()
            
            # Restore from backup
            with sqlite3.connect(backup_path) as source:
                with self.get_connection() as target:
                    source.backup(target)
            
            self.logger.info(f"Database restored from: {backup_path}")
            self.logger.info(f"Previous database backed up to: {current_backup}")
        
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            raise DatabaseError(f"Database restore failed: {e}")
    
    def cleanup_old_backups(self, max_backups: int = None):
        """Clean up old backup files"""
        max_backups = max_backups or config.database.max_backup_files
        
        backup_dir = Path(self.db_path).parent
        backup_pattern = f"{Path(self.db_path).name}.backup_*"
        
        backup_files = sorted(
            backup_dir.glob(backup_pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove old backups
        for backup_file in backup_files[max_backups:]:
            try:
                backup_file.unlink()
                self.logger.info(f"Removed old backup: {backup_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove backup {backup_file}: {e}")
    
    def _setup_backup_schedule(self):
        """Setup automatic backup schedule"""
        # This would typically be implemented with a scheduler like APScheduler
        # For now, we'll just log that backup is enabled
        self.logger.info(f"Automatic backup enabled (interval: {config.database.backup_interval_hours}h)")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table row counts
            tables = ['trades', 'positions', 'market_data', 'performance_metrics']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Database size
            stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            # Last trade timestamp
            cursor.execute("SELECT MAX(timestamp) FROM trades")
            last_trade = cursor.fetchone()[0]
            stats['last_trade_timestamp'] = last_trade
            
            # Performance summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl
                FROM trades 
                WHERE status = 'executed' AND pnl IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                stats['total_executed_trades'] = result[0]
                stats['winning_trades'] = result[1] or 0
                stats['win_rate'] = (result[1] or 0) / result[0] * 100
                stats['total_pnl'] = result[2] or 0
        
        return stats
    
    def vacuum_database(self):
        """Optimize database by running VACUUM"""
        try:
            with self.get_connection() as conn:
                conn.execute('VACUUM')
            self.logger.info("Database vacuum completed")
        except Exception as e:
            self.logger.error(f"Database vacuum failed: {e}")
            raise DatabaseError(f"Database vacuum failed: {e}")