#!/usr/bin/env python3
"""
Trade Tracker Module for Crypto Trading Agent

This module provides comprehensive trade logging, profit/loss calculation,
and performance analytics for the crypto trading agent.

Features:
- Persistent trade history storage
- Real-time P&L calculation
- Performance metrics and analytics
- Trade statistics and reporting
- Portfolio tracking
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass, asdict

@dataclass
class Trade:
    """Trade record data structure"""
    id: str
    timestamp: str
    symbol: str
    action: str  # BUY, SELL
    price: float
    size: float
    value: float
    confidence: float
    order_id: Optional[str]
    status: str  # executed, paper_trade, cancelled
    fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_timestamp: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    is_closed: bool = False

class TradeTracker:
    """Comprehensive trade tracking and P&L calculation system with database support"""
    
    def __init__(self, data_dir: str = "trading_data", db_manager=None, trade_repository=None):
        self.data_dir = data_dir
        self.trades_file = os.path.join(data_dir, "trades.json")
        self.portfolio_file = os.path.join(data_dir, "portfolio.json")
        
        # Database components
        self.db_manager = db_manager
        self.trade_repository = trade_repository
        self.use_database = db_manager is not None and trade_repository is not None
        
        # Create data directory if it doesn't exist (fallback for JSON)
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        if self.use_database:
            self.trades: List[Trade] = self._load_trades_from_db()
        else:
            self.trades: List[Trade] = self._load_trades()
        
        self.portfolio = self._load_portfolio()
        
        # Initialize portfolio if empty
        if not self.portfolio:
            self.portfolio = {
                "initial_balance": 0.0,
                "current_balance": 0.0,
                "total_invested": 0.0,
                "total_fees": 0.0,
                "positions": {},
                "last_updated": datetime.now().isoformat()
            }
    
    def _load_trades_from_db(self) -> List[Trade]:
        """Load trades from database"""
        try:
            if self.trade_repository:
                db_trades = self.trade_repository.get_all()
                # Convert database trades to TradeTracker Trade format
                trades = []
                for db_trade in db_trades:
                    trade = Trade(
                        id=db_trade.id,
                        timestamp=db_trade.timestamp.isoformat(),
                        symbol=db_trade.symbol,
                        action=db_trade.action.value,
                        price=float(db_trade.price),
                        size=float(db_trade.quantity),
                        value=float(db_trade.value),
                        confidence=db_trade.confidence,
                        order_id=db_trade.order_id,
                        status=db_trade.status.value,
                        fees=float(db_trade.fees),
                        stop_loss=float(db_trade.stop_loss) if db_trade.stop_loss else None,
                        take_profit=float(db_trade.take_profit) if db_trade.take_profit else None,
                        pnl=float(db_trade.pnl) if db_trade.pnl else None
                    )
                    trades.append(trade)
                return trades
            return []
        except Exception as e:
            print(f"Error loading trades from database: {e}")
            return []
    
    def _load_trades(self) -> List[Trade]:
        """Load trades from JSON file (fallback)"""
        if not os.path.exists(self.trades_file):
            return []
        
        try:
            with open(self.trades_file, 'r') as f:
                trades_data = json.load(f)
            return [Trade(**trade) for trade in trades_data]
        except Exception as e:
            print(f"Error loading trades: {e}")
            return []
    
    def _save_trades(self):
        """Save trades to database or JSON file"""
        if self.use_database:
            # Database saves are handled in log_trade method
            return
        
        # Fallback to JSON file
        try:
            trades_data = [asdict(trade) for trade in self.trades]
            with open(self.trades_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
        except Exception as e:
            print(f"Error saving trades: {e}")
    
    def _load_portfolio(self) -> Dict:
        """Load portfolio from JSON file"""
        if not os.path.exists(self.portfolio_file):
            return {}
        
        try:
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return {}
    
    def _save_portfolio(self):
        """Save portfolio to JSON file"""
        try:
            self.portfolio["last_updated"] = datetime.now().isoformat()
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio, f, indent=2)
        except Exception as e:
            print(f"Error saving portfolio: {e}")
    
    def log_trade(self, symbol: str, action: str, price: float, size: float, 
                  confidence: float, order_id: Optional[str] = None, 
                  status: str = "executed", fees: float = 0.0,
                  stop_loss: Optional[float] = None, 
                  take_profit: Optional[float] = None) -> str:
        """Log a new trade"""
        
        trade_id = f"{symbol}_{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        value = price * size
        
        trade = Trade(
            id=trade_id,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            action=action,
            price=price,
            size=size,
            value=value,
            confidence=confidence,
            order_id=order_id,
            status=status,
            fees=fees,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Save to database if available
        if self.use_database and self.trade_repository:
            try:
                from src.data.models import Trade as DbTrade, TradeAction, TradeStatus
                from decimal import Decimal
                
                db_trade = DbTrade(
                    id=trade_id,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action=TradeAction(action),
                    price=Decimal(str(price)),
                    quantity=Decimal(str(size)),
                    value=Decimal(str(value)),
                    confidence=confidence,
                    status=TradeStatus(status.upper()),
                    order_id=order_id,
                    fees=Decimal(str(fees)),
                    stop_loss=Decimal(str(stop_loss)) if stop_loss else None,
                    take_profit=Decimal(str(take_profit)) if take_profit else None
                )
                
                self.trade_repository.create(db_trade)
                print(f"✅ Trade saved to database: {trade_id}")
            except Exception as e:
                print(f"⚠️ Failed to save trade to database: {e}")
                # Fall back to JSON storage
                self.trades.append(trade)
                self._save_trades()
        else:
            # Use JSON storage
            self.trades.append(trade)
            self._save_trades()
        
        self._update_portfolio(trade)
        self._save_portfolio()
        
        return trade_id
    
    def _update_portfolio(self, trade: Trade):
        """Update portfolio with new trade"""
        symbol = trade.symbol
        
        # Initialize position if doesn't exist
        if symbol not in self.portfolio["positions"]:
            self.portfolio["positions"][symbol] = {
                "size": 0.0,
                "avg_price": 0.0,
                "total_cost": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0
            }
        
        position = self.portfolio["positions"][symbol]
        
        if trade.action == "BUY":
            # Add to position
            new_total_cost = position["total_cost"] + trade.value + trade.fees
            new_size = position["size"] + trade.size
            
            if new_size > 0:
                position["avg_price"] = new_total_cost / new_size
            
            position["size"] = new_size
            position["total_cost"] = new_total_cost
            
        elif trade.action == "SELL":
            # Reduce position and calculate realized P&L
            if position["size"] >= trade.size:
                # Calculate realized P&L for this sale
                cost_basis = position["avg_price"] * trade.size
                realized_pnl = (trade.price * trade.size) - cost_basis - trade.fees
                
                position["realized_pnl"] += realized_pnl
                position["size"] -= trade.size
                position["total_cost"] -= cost_basis
                
                # Update trade with P&L
                trade.pnl = realized_pnl
                trade.pnl_percent = (realized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
                trade.is_closed = True
                trade.exit_timestamp = trade.timestamp
                trade.exit_price = trade.price
                trade.exit_reason = "manual_sell"
        
        # Update total fees
        self.portfolio["total_fees"] += trade.fees
    
    def close_position(self, trade_id: str, exit_price: float, exit_reason: str = "manual_close"):
        """Close a position and calculate final P&L"""
        trade = self.get_trade_by_id(trade_id)
        if not trade or trade.is_closed:
            return False
        
        trade.exit_timestamp = datetime.now().isoformat()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.is_closed = True
        
        # Calculate P&L
        if trade.action == "BUY":
            trade.pnl = (exit_price - trade.price) * trade.size - trade.fees
        else:  # SELL
            trade.pnl = (trade.price - exit_price) * trade.size - trade.fees
        
        trade.pnl_percent = (trade.pnl / trade.value) * 100 if trade.value > 0 else 0
        
        self._save_trades()
        return True
    
    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get trade by ID"""
        for trade in self.trades:
            if trade.id == trade_id:
                return trade
        return None
    
    def calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate unrealized P&L for open positions"""
        unrealized_pnl = {}
        
        for symbol, position in self.portfolio["positions"].items():
            if position["size"] > 0 and symbol in current_prices:
                current_value = position["size"] * current_prices[symbol]
                unrealized_pnl[symbol] = current_value - position["total_cost"]
                position["unrealized_pnl"] = unrealized_pnl[symbol]
        
        self._save_portfolio()
        return unrealized_pnl
    
    def get_performance_metrics(self, days: int = 30) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {"error": "No trades found"}
        
        # Filter trades by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            trade for trade in self.trades 
            if datetime.fromisoformat(trade.timestamp) >= cutoff_date
        ]
        
        if not recent_trades:
            return {"error": f"No trades found in the last {days} days"}
        
        # Calculate metrics
        total_trades = len(recent_trades)
        executed_trades = [t for t in recent_trades if t.status == "executed"]
        paper_trades = [t for t in recent_trades if t.status == "paper_trade"]
        closed_trades = [t for t in recent_trades if t.is_closed and t.pnl is not None]
        
        # P&L calculations
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)
        winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Trading frequency
        buy_trades = [t for t in recent_trades if t.action == "BUY"]
        sell_trades = [t for t in recent_trades if t.action == "SELL"]
        
        return {
            "period_days": days,
            "total_trades": total_trades,
            "executed_trades": len(executed_trades),
            "paper_trades": len(paper_trades),
            "closed_trades": len(closed_trades),
            "open_positions": total_trades - len(closed_trades),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "N/A",
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_fees": round(self.portfolio["total_fees"], 2)
        }
    
    def get_trade_history(self, limit: int = 50, symbol: Optional[str] = None) -> List[Dict]:
        """Get formatted trade history"""
        trades = self.trades
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # Sort by timestamp (newest first)
        trades = sorted(trades, key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        trades = trades[:limit]
        
        # Format for display
        formatted_trades = []
        for trade in trades:
            formatted_trade = {
                "id": trade.id,
                "timestamp": trade.timestamp[:19],  # Remove microseconds
                "symbol": trade.symbol,
                "action": trade.action,
                "price": f"${trade.price:.4f}",
                "size": f"{trade.size:.6f}",
                "value": f"${trade.value:.2f}",
                "confidence": f"{trade.confidence:.2f}",
                "status": trade.status,
                "pnl": f"${trade.pnl:.2f}" if trade.pnl is not None else "N/A",
                "pnl_percent": f"{trade.pnl_percent:.2f}%" if trade.pnl_percent is not None else "N/A",
                "is_closed": trade.is_closed
            }
            formatted_trades.append(formatted_trade)
        
        return formatted_trades
    
    def export_trades_to_csv(self, filename: Optional[str] = None) -> str:
        """Export trades to CSV file"""
        if not filename:
            filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Convert trades to DataFrame
        trades_data = [asdict(trade) for trade in self.trades]
        df = pd.DataFrame(trades_data)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_trades_to_excel(self, filename: Optional[str] = None) -> str:
        """Export trades to Excel file with multiple sheets and formatting"""
        if not filename:
            filename = "trades_export.xlsx"
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Convert trades to DataFrame
        trades_data = [asdict(trade) for trade in self.trades]
        df = pd.DataFrame(trades_data)
        
        # Check if file exists and load existing data to append
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_excel(filepath, sheet_name='All Trades')
                # Remove duplicates based on order_id and append new trades
                if not existing_df.empty and not df.empty:
                    # Get order IDs that already exist
                    existing_order_ids = set(existing_df['order_id'].tolist()) if 'order_id' in existing_df.columns else set()
                    # Filter out trades that already exist (based on order_id)
                    new_trades = df[~df['order_id'].isin(existing_order_ids)]
                    # Combine existing and new trades
                    df = pd.concat([existing_df, new_trades], ignore_index=True)
                elif not existing_df.empty:
                    df = existing_df
            except Exception as e:
                print(f"Warning: Could not read existing Excel file, will overwrite: {e}")
        
        # Create Excel writer object
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main trades sheet
            df.to_excel(writer, sheet_name='All Trades', index=False)
            
            # Executed trades only
            executed_df = df[df['status'] == 'executed']
            if not executed_df.empty:
                executed_df.to_excel(writer, sheet_name='Executed Trades', index=False)
            
            # Paper trades only
            paper_df = df[df['status'] == 'paper_trade']
            if not paper_df.empty:
                paper_df.to_excel(writer, sheet_name='Paper Trades', index=False)
            
            # Closed trades (with P&L)
            closed_df = df[df['is_closed'] == True]
            if not closed_df.empty:
                closed_df.to_excel(writer, sheet_name='Closed Trades', index=False)
            
            # Summary statistics
            summary_data = self.get_performance_metrics()
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Performance by symbol
            if not df.empty:
                symbol_performance = []
                for symbol in df['symbol'].unique():
                    symbol_trades = df[df['symbol'] == symbol]
                    closed_symbol_trades = symbol_trades[symbol_trades['is_closed'] == True]
                    
                    if not closed_symbol_trades.empty:
                        total_pnl = closed_symbol_trades['pnl'].fillna(0).sum()
                        win_rate = (closed_symbol_trades['pnl'] > 0).mean() * 100
                        total_trades = len(symbol_trades)
                        executed_trades = len(symbol_trades[symbol_trades['status'] == 'executed'])
                        
                        symbol_performance.append({
                            'Symbol': symbol,
                            'Total Trades': total_trades,
                            'Executed Trades': executed_trades,
                            'Closed Trades': len(closed_symbol_trades),
                            'Total P&L': round(total_pnl, 4),
                            'Win Rate (%)': round(win_rate, 2),
                            'Avg Trade Value': round(symbol_trades['value'].mean(), 4)
                        })
                
                if symbol_performance:
                    symbol_df = pd.DataFrame(symbol_performance)
                    symbol_df.to_excel(writer, sheet_name='Performance by Symbol', index=False)
        
        return filepath
    
    def auto_export_to_excel(self) -> str:
        """Automatically export trades to Excel with timestamp"""
        return self.export_trades_to_excel()
    
    def calculate_portfolio_pnl(self, symbol: str = None) -> float:
        """Calculate current portfolio P&L for specific symbol or total"""
        if symbol:
            # Calculate P&L for specific symbol
            if symbol in self.portfolio["positions"]:
                position = self.portfolio["positions"][symbol]
                return position["realized_pnl"] + position["unrealized_pnl"]
            return 0.0
        else:
            # Calculate total portfolio P&L
            total_realized = sum(
                pos["realized_pnl"] for pos in self.portfolio["positions"].values()
            )
            total_unrealized = sum(
                pos["unrealized_pnl"] for pos in self.portfolio["positions"].values()
            )
            return total_realized + total_unrealized
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_realized_pnl = sum(
            pos["realized_pnl"] for pos in self.portfolio["positions"].values()
        )
        
        total_unrealized_pnl = sum(
            pos["unrealized_pnl"] for pos in self.portfolio["positions"].values()
        )
        
        open_positions = {
            symbol: pos for symbol, pos in self.portfolio["positions"].items()
            if pos["size"] > 0
        }
        
        return {
            "total_realized_pnl": round(total_realized_pnl, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_pnl": round(total_realized_pnl + total_unrealized_pnl, 2),
            "total_fees": round(self.portfolio["total_fees"], 2),
            "open_positions_count": len(open_positions),
            "open_positions": open_positions,
            "last_updated": self.portfolio["last_updated"]
        }
    
    def print_performance_report(self, days: int = 30):
        """Print a comprehensive performance report"""
        print(f"\n{'='*60}")
        print(f"TRADING PERFORMANCE REPORT ({days} DAYS)")
        print(f"{'='*60}")
        
        metrics = self.get_performance_metrics(days)
        
        if "error" in metrics:
            print(f"❌ {metrics['error']}")
            return
        
        # Trading Activity
        print(f"\n📊 TRADING ACTIVITY")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Executed Trades: {metrics['executed_trades']}")
        print(f"Paper Trades: {metrics['paper_trades']}")
        print(f"Closed Positions: {metrics['closed_trades']}")
        print(f"Open Positions: {metrics['open_positions']}")
        
        # P&L Performance
        print(f"\n💰 PROFIT & LOSS")
        print(f"Total P&L: ${metrics['total_pnl']}")
        print(f"Win Rate: {metrics['win_rate']}%")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Average Win: ${metrics['avg_win']}")
        print(f"Average Loss: ${metrics['avg_loss']}")
        print(f"Profit Factor: {metrics['profit_factor']}")
        
        # Trading Breakdown
        print(f"\n📈 TRADING BREAKDOWN")
        print(f"Buy Trades: {metrics['buy_trades']}")
        print(f"Sell Trades: {metrics['sell_trades']}")
        print(f"Total Fees: ${metrics['total_fees']}")
        
        # Portfolio Summary
        portfolio = self.get_portfolio_summary()
        print(f"\n💼 PORTFOLIO SUMMARY")
        print(f"Total Realized P&L: ${portfolio['total_realized_pnl']}")
        print(f"Total Unrealized P&L: ${portfolio['total_unrealized_pnl']}")
        print(f"Net P&L: ${portfolio['total_pnl']}")
        print(f"Open Positions: {portfolio['open_positions_count']}")
        
        if portfolio['open_positions']:
            print(f"\n🔓 OPEN POSITIONS")
            for symbol, pos in portfolio['open_positions'].items():
                print(f"{symbol}: {pos['size']:.6f} @ ${pos['avg_price']:.4f} | "
                      f"Unrealized P&L: ${pos['unrealized_pnl']:.2f}")
        
        print(f"\n{'='*60}")