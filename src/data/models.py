"""Data models for the Crypto Trading Agent"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
from enum import Enum


class TradeAction(Enum):
    """Trade action enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PAPER_TRADE = "paper_trade"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class Trade:
    """Trade data model"""
    id: str
    timestamp: datetime
    symbol: str
    action: TradeAction
    price: Decimal
    quantity: Decimal
    value: Decimal
    confidence: float
    status: TradeStatus
    order_id: Optional[str] = None
    fees: Decimal = Decimal('0')
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    strategy_name: Optional[str] = None
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[datetime] = None
    pnl: Optional[Decimal] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'action': self.action.value,
            'price': float(self.price),
            'quantity': float(self.quantity),
            'value': float(self.value),
            'confidence': self.confidence,
            'status': self.status.value,
            'order_id': self.order_id,
            'fees': float(self.fees),
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'strategy_name': self.strategy_name,
            'analysis_data': self.analysis_data,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None,
            'pnl': float(self.pnl) if self.pnl else None,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create trade from dictionary"""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbol=data['symbol'],
            action=TradeAction(data['action']),
            price=Decimal(str(data['price'])),
            quantity=Decimal(str(data['quantity'])),
            value=Decimal(str(data['value'])),
            confidence=data['confidence'],
            status=TradeStatus(data['status']),
            order_id=data.get('order_id'),
            fees=Decimal(str(data.get('fees', 0))),
            stop_loss=Decimal(str(data['stop_loss'])) if data.get('stop_loss') else None,
            take_profit=Decimal(str(data['take_profit'])) if data.get('take_profit') else None,
            strategy_name=data.get('strategy_name'),
            analysis_data=data.get('analysis_data', {}),
            execution_time=datetime.fromisoformat(data['execution_time']) if data.get('execution_time') else None,
            pnl=Decimal(str(data['pnl'])) if data.get('pnl') else None,
            notes=data.get('notes')
        )


@dataclass
class Position:
    """Position data model"""
    id: str
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_percent: float
    status: PositionStatus
    opened_at: datetime
    last_updated: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    trades: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'average_price': float(self.average_price),
            'current_price': float(self.current_price),
            'market_value': float(self.market_value),
            'unrealized_pnl': float(self.unrealized_pnl),
            'unrealized_pnl_percent': self.unrealized_pnl_percent,
            'status': self.status.value,
            'opened_at': self.opened_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'stop_loss': float(self.stop_loss) if self.stop_loss else None,
            'take_profit': float(self.take_profit) if self.take_profit else None,
            'trades': self.trades
        }


@dataclass
class MarketData:
    """Market data model"""
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    indicators: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open_price': float(self.open_price),
            'high_price': float(self.high_price),
            'low_price': float(self.low_price),
            'close_price': float(self.close_price),
            'volume': float(self.volume),
            'quote_volume': float(self.quote_volume),
            'indicators': self.indicators
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics model"""
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    average_trade_duration: float
    total_fees: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary"""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': float(self.total_pnl),
            'gross_profit': float(self.gross_profit),
            'gross_loss': float(self.gross_loss),
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'average_win': float(self.average_win),
            'average_loss': float(self.average_loss),
            'largest_win': float(self.largest_win),
            'largest_loss': float(self.largest_loss),
            'average_trade_duration': self.average_trade_duration,
            'total_fees': float(self.total_fees)
        }


@dataclass
class AnalysisResult:
    """Analysis result model"""
    symbol: str
    timestamp: datetime
    action: TradeAction
    confidence: float
    price: Decimal
    technical_signals: Dict[str, Any]
    llm_analysis: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    strategy_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action.value,
            'confidence': self.confidence,
            'price': float(self.price),
            'technical_signals': self.technical_signals,
            'llm_analysis': self.llm_analysis,
            'risk_assessment': self.risk_assessment,
            'strategy_name': self.strategy_name
        }