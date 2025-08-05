import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

class RiskManager:
    def __init__(self, max_portfolio_risk: float = None, max_position_size: float = None):
        # Load environment variables
        load_dotenv()
        
        # Use environment variables with fallback to defaults
        self.max_portfolio_risk = max_portfolio_risk or float(os.getenv('MAX_PORTFOLIO_RISK', '0.02'))
        self.max_position_size = max_position_size or float(os.getenv('MAX_POSITION_SIZE', '0.1'))
        self.stop_loss_percentage = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.05'))
        self.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', '2.0'))
        self.atr_multiplier = float(os.getenv('ATR_MULTIPLIER', '2.0'))
        
        self.positions = {}  # Track open positions
        self.trade_history = []  # Track trade history
        
    def calculate_position_size(self, account_balance: float, entry_price: float,
                              stop_loss_price: float, risk_per_trade: float = None) -> Dict:
        """Calculate optimal position size based on risk management rules"""
        if risk_per_trade is None:
            risk_per_trade = self.max_portfolio_risk
            
        # Calculate risk per share/unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return {'size': 0, 'risk_amount': 0, 'error': 'Invalid stop loss price'}
        
        # Calculate position size based on risk
        risk_amount = account_balance * risk_per_trade
        position_size = risk_amount / risk_per_unit
        
        # Apply maximum position size constraint
        max_size_by_balance = (account_balance * self.max_position_size) / entry_price
        position_size = min(position_size, max_size_by_balance)
        
        # Calculate actual risk amount
        actual_risk = position_size * risk_per_unit
        
        return {
            'size': position_size,
            'risk_amount': actual_risk,
            'risk_percentage': (actual_risk / account_balance) * 100,
            'position_value': position_size * entry_price,
            'position_percentage': ((position_size * entry_price) / account_balance) * 100
        }
    
    def calculate_stop_loss(self, entry_price: float, direction: str, 
                          atr: float = None, method: str = 'percentage') -> Dict:
        """Calculate stop loss levels using different methods"""
        stop_loss_levels = {}
        
        if method == 'percentage':
            # Fixed percentage stop loss from environment variable
            stop_loss_pct = self.stop_loss_percentage
            if direction.upper() == 'BUY':
                stop_loss_levels['percentage'] = entry_price * (1 - stop_loss_pct)
            else:
                stop_loss_levels['percentage'] = entry_price * (1 + stop_loss_pct)
        
        if method == 'atr' and atr:
            # ATR-based stop loss using environment variable
            atr_multiplier = self.atr_multiplier
            if direction.upper() == 'BUY':
                stop_loss_levels['atr'] = entry_price - (atr * atr_multiplier)
            else:
                stop_loss_levels['atr'] = entry_price + (atr * atr_multiplier)
        
        if method == 'support_resistance':
            # This would require support/resistance levels from technical analysis
            # Placeholder for now
            stop_loss_levels['support_resistance'] = entry_price * 0.95 if direction.upper() == 'BUY' else entry_price * 1.05
        
        return stop_loss_levels
    
    def calculate_take_profit(self, entry_price: float, stop_loss_price: float,
                            direction: str, risk_reward_ratio: float = None) -> float:
        """Calculate take profit level based on risk-reward ratio"""
        if risk_reward_ratio is None:
            risk_reward_ratio = self.risk_reward_ratio
        
        risk = abs(entry_price - stop_loss_price)
        reward = risk * risk_reward_ratio
        
        if direction.upper() == 'BUY':
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
            
        return take_profit
    
    def validate_trade(self, symbol: str, direction: str, size: float, 
                      entry_price: float, account_balance: float) -> Dict:
        """Validate if a trade meets risk management criteria"""
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check position size limits
        position_value = size * entry_price
        position_percentage = (position_value / account_balance) * 100
        
        if position_percentage > self.max_position_size * 100:
            validation_result['errors'].append(
                f"Position size ({position_percentage:.2f}%) exceeds maximum allowed ({self.max_position_size * 100}%)"
            )
            validation_result['valid'] = False
        
        # Check if symbol already has a position
        if symbol in self.positions:
            validation_result['warnings'].append(
                f"Already have an open position in {symbol}"
            )
        
        # Check portfolio concentration
        total_exposure = sum([pos['value'] for pos in self.positions.values()])
        total_exposure += position_value
        
        if total_exposure > account_balance * 0.8:  # Max 80% portfolio exposure
            validation_result['warnings'].append(
                "High portfolio exposure - consider reducing position sizes"
            )
        
        return validation_result
    
    def add_position(self, symbol: str, direction: str, size: float, 
                    entry_price: float, stop_loss: float, take_profit: float,
                    timestamp: datetime = None) -> str:
        """Add a new position to tracking"""
        if timestamp is None:
            timestamp = datetime.now()
            
        position_id = f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        self.positions[position_id] = {
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'value': size * entry_price,
            'timestamp': timestamp,
            'status': 'open'
        }
        
        return position_id
    
    def update_position(self, position_id: str, current_price: float) -> Dict:
        """Update position with current market price and check exit conditions"""
        if position_id not in self.positions:
            return {'error': 'Position not found'}
        
        position = self.positions[position_id]
        
        # Calculate current P&L
        if position['direction'].upper() == 'BUY':
            pnl = (current_price - position['entry_price']) * position['size']
            pnl_percentage = ((current_price / position['entry_price']) - 1) * 100
        else:
            pnl = (position['entry_price'] - current_price) * position['size']
            pnl_percentage = ((position['entry_price'] / current_price) - 1) * 100
        
        # Check exit conditions
        exit_signal = None
        
        if position['direction'].upper() == 'BUY':
            if current_price <= position['stop_loss']:
                exit_signal = 'stop_loss'
            elif current_price >= position['take_profit']:
                exit_signal = 'take_profit'
        else:
            if current_price >= position['stop_loss']:
                exit_signal = 'stop_loss'
            elif current_price <= position['take_profit']:
                exit_signal = 'take_profit'
        
        # Update position
        position.update({
            'current_price': current_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'last_update': datetime.now()
        })
        
        return {
            'position_id': position_id,
            'current_price': current_price,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'exit_signal': exit_signal
        }
    
    def close_position(self, position_id: str, exit_price: float, 
                      exit_reason: str = 'manual') -> Dict:
        """Close a position and record the trade"""
        if position_id not in self.positions:
            return {'error': 'Position not found'}
        
        position = self.positions[position_id]
        
        # Calculate final P&L
        if position['direction'].upper() == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        pnl_percentage = (pnl / position['value']) * 100
        
        # Record trade in history
        trade_record = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'direction': position['direction'],
            'size': position['size'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['timestamp'],
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'exit_reason': exit_reason
        }
        
        self.trade_history.append(trade_record)
        
        # Remove from active positions
        del self.positions[position_id]
        
        return trade_record
    
    def get_portfolio_metrics(self, account_balance: float) -> Dict:
        """Calculate portfolio-level risk metrics"""
        if not self.positions:
            return {
                'total_exposure': 0,
                'exposure_percentage': 0,
                'number_of_positions': 0,
                'largest_position': 0,
                'portfolio_risk': 0
            }
        
        total_exposure = sum([pos['value'] for pos in self.positions.values()])
        position_values = [pos['value'] for pos in self.positions.values()]
        
        return {
            'total_exposure': total_exposure,
            'exposure_percentage': (total_exposure / account_balance) * 100,
            'number_of_positions': len(self.positions),
            'largest_position': max(position_values),
            'largest_position_percentage': (max(position_values) / account_balance) * 100,
            'average_position_size': np.mean(position_values),
            'portfolio_risk': self.calculate_portfolio_var()
        }
    
    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Portfolio Value at Risk (simplified)"""
        if not self.trade_history:
            return 0.0
        
        # Get returns from trade history
        returns = [trade['pnl_percentage'] for trade in self.trade_history[-100:]]  # Last 100 trades
        
        if len(returns) < 10:
            return 0.0
        
        # Calculate VaR using historical simulation
        returns_array = np.array(returns)
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_array, var_percentile)
        
        return abs(var)
    
    def get_trade_statistics(self) -> Dict:
        """Calculate trading performance statistics"""
        if not self.trade_history:
            return {}
        
        trades = pd.DataFrame(self.trade_history)
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L statistics
        total_pnl = trades['pnl'].sum()
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        max_drawdown = self.calculate_max_drawdown(trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': self.calculate_sharpe_ratio(trades)
        }
    
    def calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if trades_df.empty:
            return 0.0
        
        cumulative_returns = trades_df['pnl'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        
        return abs(drawdown.min())
    
    def calculate_sharpe_ratio(self, trades_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if trades_df.empty or len(trades_df) < 2:
            return 0.0
        
        returns = trades_df['pnl_percentage'] / 100  # Convert to decimal
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        
        if returns.std() == 0:
            return 0.0
        
        return excess_returns / returns.std() * np.sqrt(252)  # Annualized
    
    def save_state(self, filepath: str):
        """Save risk manager state to file"""
        state = {
            'positions': self.positions,
            'trade_history': self.trade_history,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_position_size': self.max_position_size
        }
        
        # Convert datetime objects to strings for JSON serialization
        for pos in state['positions'].values():
            pos['timestamp'] = pos['timestamp'].isoformat()
            if 'last_update' in pos:
                pos['last_update'] = pos['last_update'].isoformat()
        
        for trade in state['trade_history']:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load risk manager state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.max_portfolio_risk = state['max_portfolio_risk']
            self.max_position_size = state['max_position_size']
            self.positions = state['positions']
            self.trade_history = state['trade_history']
            
            # Convert string timestamps back to datetime objects
            for pos in self.positions.values():
                pos['timestamp'] = datetime.fromisoformat(pos['timestamp'])
                if 'last_update' in pos:
                    pos['last_update'] = datetime.fromisoformat(pos['last_update'])
            
            for trade in self.trade_history:
                trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
                trade['exit_time'] = datetime.fromisoformat(trade['exit_time'])
                
        except Exception as e:
            print(f"Error loading state: {e}")