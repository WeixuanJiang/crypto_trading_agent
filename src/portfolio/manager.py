"""
Portfolio Manager for Crypto Trading Agent

This module manages portfolio balances, positions, and provides
portfolio-aware trading decisions considering existing holdings.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
from dataclasses import dataclass

@dataclass
class Position:
    """Represents a cryptocurrency position"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    last_updated: datetime

class PortfolioManager:
    """Manages cryptocurrency portfolio with balance-aware trading"""
    
    def __init__(self, user_client=None, market_client=None, trade_tracker=None, config=None, 
                 db_manager=None, position_repository=None):
        self.user_client = user_client
        self.market_client = market_client
        self.trade_tracker = trade_tracker
        self.config = config
        self.db_manager = db_manager
        self.position_repository = position_repository
        self.use_database = db_manager is not None and position_repository is not None
        
        self.positions = {}
        self.balances = {}
        self.portfolio_value = 0.0
        self.last_update = None
        
        # Portfolio limits
        self.max_position_percentage = 0.25  # Max 25% per asset
        self.min_trade_size_usdt = config.trading.min_trade_size_usdt if config else 2.0  # Minimum trade size
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalancing consideration
        
        # Load cached positions
        self._load_cached_positions()
    
    def update_balances(self) -> Dict[str, float]:
        """Update account balances from exchange"""
        try:
            if not self.user_client:
                return self._get_mock_balances()
            
            # Get account balances
            accounts = self.user_client.get_accounts()
            
            balances = {}
            for account in accounts:
                if account['type'] == 'trade':  # Trading account
                    currency = account['currency']
                    balance = float(account['balance'])
                    available = float(account['available'])
                    
                    # Include all currencies, even with 0 balance for transparency
                    balances[currency] = {
                        'total': balance,
                        'available': available,
                        'hold': balance - available
                    }
            
            # Ensure USDT is always included for portfolio value calculation
            if 'USDT' not in balances:
                balances['USDT'] = {
                    'total': 0.0,
                    'available': 0.0,
                    'hold': 0.0
                }
            
            self.balances = balances
            self.last_update = datetime.now()
            
            print(f"📊 Updated balances for {len(balances)} currencies")
            print(f"💰 USDT Balance: ${balances.get('USDT', {}).get('total', 0.0):.2f}")
            return balances
            
        except Exception as e:
            print(f"❌ Error updating balances: {e}")
            return self._get_mock_balances()
    
    def _get_mock_balances(self) -> Dict[str, float]:
        """Get mock balances for testing"""
        return {
            'USDT': {'total': 1000.0, 'available': 800.0, 'hold': 200.0},
            'BTC': {'total': 0.05, 'available': 0.05, 'hold': 0.0},
            'ETH': {'total': 0.8, 'available': 0.8, 'hold': 0.0},
            'ADA': {'total': 500.0, 'available': 500.0, 'hold': 0.0},
            'DOT': {'total': 20.0, 'available': 20.0, 'hold': 0.0}
        }
    
    def update_positions(self, current_prices: Dict[str, float]) -> Dict[str, Position]:
        """Update position information with current market prices"""
        positions = {}
        
        for currency, balance_info in self.balances.items():
            if currency == 'USDT' or balance_info['total'] <= 0:
                continue
            
            symbol = f"{currency}-USDT"
            current_price = current_prices.get(symbol, 0.0)
            
            if current_price > 0:
                quantity = balance_info['total']
                market_value = quantity * current_price
                
                # Get average price from trade history or use current price
                avg_price = self._get_average_price(currency, current_price)
                
                # Calculate P&L
                unrealized_pnl = (current_price - avg_price) * quantity
                unrealized_pnl_percent = (unrealized_pnl / (avg_price * quantity)) * 100 if avg_price > 0 else 0
                
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=avg_price,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                    last_updated=datetime.now()
                )
                
                positions[symbol] = position
        
        self.positions = positions
        self.portfolio_value = sum(pos.market_value for pos in positions.values())
        if 'USDT' in self.balances:
            self.portfolio_value += self.balances['USDT']['total']
        
        return positions
    
    def _get_average_price(self, currency: str, fallback_price: float) -> float:
        """Get average purchase price for a currency from trade tracker"""
        try:
            # Try to get from trade tracker if available
            if hasattr(self, 'trade_tracker') and self.trade_tracker:
                symbol = f"{currency}-USDT"
                portfolio_summary = self.trade_tracker.get_portfolio_summary()
                if 'open_positions' in portfolio_summary and symbol in portfolio_summary['open_positions']:
                    return portfolio_summary['open_positions'][symbol].get('avg_price', fallback_price)
            
            # Try to load from cached data
            cache_file = f"trading_data/avg_prices.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    avg_prices = json.load(f)
                    return avg_prices.get(currency, fallback_price)
        except Exception as e:
            print(f"Warning: Could not get average price for {currency}: {e}")
        
        return fallback_price
    
    def get_position_size_recommendation(self, symbol: str, action: str, 
                                       confidence: float, current_price: float,
                                       account_balance: float = None) -> Dict:
        """Get position size recommendation considering existing positions"""
        
        if account_balance is None:
            account_balance = self.portfolio_value
        
        currency = symbol.split('-')[0]
        current_position = self.positions.get(symbol)
        usdt_balance = self.balances.get('USDT', {}).get('available', 0)
        
        recommendation = {
            'action': action,
            'recommended_size': 0.0,
            'recommended_value': 0.0,
            'reason': '',
            'current_position': current_position.quantity if current_position else 0.0,
            'current_allocation': 0.0,
            'target_allocation': 0.0,
            'can_execute': False
        }
        
        # Calculate current allocation
        if current_position and self.portfolio_value > 0:
            current_allocation = current_position.market_value / self.portfolio_value
        else:
            current_allocation = 0.0
        
        recommendation['current_allocation'] = current_allocation * 100
        
        if action == 'BUY':
            return self._calculate_buy_recommendation(
                symbol, confidence, current_price, current_allocation, 
                usdt_balance, recommendation
            )
        elif action == 'SELL':
            return self._calculate_sell_recommendation(
                symbol, confidence, current_price, current_position, 
                recommendation
            )
        else:  # HOLD
            recommendation['reason'] = 'Hold signal - no action recommended'
            return recommendation
    
    def _calculate_buy_recommendation(self, symbol: str, confidence: float, 
                                    current_price: float, current_allocation: float,
                                    usdt_balance: float, recommendation: Dict) -> Dict:
        """Calculate buy recommendation considering portfolio balance"""
        
        # Target allocation based on confidence
        base_allocation = 0.15  # 15% base allocation
        confidence_bonus = confidence * 0.10  # Up to 10% bonus for high confidence
        target_allocation = min(base_allocation + confidence_bonus, self.max_position_percentage)
        
        recommendation['target_allocation'] = target_allocation * 100
        
        # Calculate how much more we can buy
        current_value = current_allocation * self.portfolio_value
        target_value = target_allocation * self.portfolio_value
        additional_value_needed = target_value - current_value
        
        # Check constraints
        if additional_value_needed <= self.min_trade_size_usdt:
            recommendation['reason'] = f'Position already near target allocation ({current_allocation*100:.1f}%)'
            return recommendation
        
        if usdt_balance < self.min_trade_size_usdt:
            recommendation['reason'] = f'Insufficient USDT balance (${usdt_balance:.2f})'
            return recommendation
        
        if usdt_balance < additional_value_needed:
            additional_value_needed = usdt_balance * 0.95  # Leave some buffer
            recommendation['reason'] = f'Limited by available USDT balance'
        
        # Calculate recommended size
        recommended_quantity = additional_value_needed / current_price
        
        recommendation.update({
            'recommended_size': recommended_quantity,
            'recommended_value': additional_value_needed,
            'can_execute': True,
            'reason': f'Buy to reach target allocation of {target_allocation*100:.1f}%'
        })
        
        return recommendation
    
    def _calculate_sell_recommendation(self, symbol: str, confidence: float,
                                     current_price: float, current_position: Optional[Position],
                                     recommendation: Dict) -> Dict:
        """Calculate sell recommendation considering current position"""
        
        if not current_position or current_position.quantity <= 0:
            recommendation['reason'] = 'No position to sell'
            return recommendation
        
        # Determine sell percentage based on confidence and P&L
        base_sell_pct = 0.3  # Sell 30% base
        confidence_bonus = confidence * 0.4  # Up to 40% more for high confidence
        
        # Adjust based on P&L
        if current_position.unrealized_pnl_percent > 20:  # Big gains
            pnl_bonus = 0.3  # Sell more to take profits
        elif current_position.unrealized_pnl_percent < -10:  # Losses
            pnl_bonus = 0.2  # Sell more to cut losses
        else:
            pnl_bonus = 0.0
        
        sell_percentage = min(base_sell_pct + confidence_bonus + pnl_bonus, 0.8)  # Max 80%
        
        recommended_quantity = current_position.quantity * sell_percentage
        recommended_value = recommended_quantity * current_price
        
        if recommended_value < self.min_trade_size_usdt:
            recommendation['reason'] = f'Recommended sell amount too small (${recommended_value:.2f})'
            return recommendation
        
        recommendation.update({
            'recommended_size': recommended_quantity,
            'recommended_value': recommended_value,
            'can_execute': True,
            'reason': f'Sell {sell_percentage*100:.1f}% of position (P&L: {current_position.unrealized_pnl_percent:.1f}%)',
            'current_pnl': current_position.unrealized_pnl,
            'current_pnl_percent': current_position.unrealized_pnl_percent
        })
        
        return recommendation
    
    def check_rebalancing_needed(self) -> List[Dict]:
        """Check if portfolio rebalancing is needed"""
        rebalancing_suggestions = []
        
        if not self.positions or self.portfolio_value <= 0:
            return rebalancing_suggestions
        
        target_allocation = 1.0 / len(self.positions)  # Equal weight
        
        for symbol, position in self.positions.items():
            current_allocation = position.market_value / self.portfolio_value
            deviation = abs(current_allocation - target_allocation)
            
            if deviation > self.rebalance_threshold:
                if current_allocation > target_allocation:
                    action = 'SELL'
                    amount = (current_allocation - target_allocation) * self.portfolio_value
                else:
                    action = 'BUY'
                    amount = (target_allocation - current_allocation) * self.portfolio_value
                
                rebalancing_suggestions.append({
                    'symbol': symbol,
                    'action': action,
                    'current_allocation': current_allocation * 100,
                    'target_allocation': target_allocation * 100,
                    'deviation': deviation * 100,
                    'suggested_amount': amount
                })
        
        return rebalancing_suggestions
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        usdt_balance = self.balances.get('USDT', {}).get('total', 0)
        
        summary = {
            'total_portfolio_value': self.portfolio_value,  # This includes both crypto + USDT
            'total_crypto_value': total_market_value,       # Only crypto positions
            'usdt_balance': usdt_balance,                   # USDT balance
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_unrealized_pnl_percent': (total_unrealized_pnl / total_market_value * 100) if total_market_value > 0 else 0,
            'number_of_positions': len(self.positions),
            'positions': {},
            'last_updated': self.last_update.isoformat() if self.last_update else None
        }
        
        for symbol, position in self.positions.items():
            allocation = (position.market_value / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
            summary['positions'][symbol] = {
                'quantity': position.quantity,
                'average_price': position.average_price,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'allocation_percent': allocation,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_percent': position.unrealized_pnl_percent
            }
        
        return summary
    
    def _load_cached_positions(self):
        """Load cached position data from database or JSON file"""
        try:
            if self.use_database:
                self._load_positions_from_db()
            else:
                self._load_positions_from_json()
        except Exception as e:
            print(f"⚠️ Could not load cached positions: {e}")
    
    def _load_positions_from_db(self):
        """Load position data from database"""
        try:
            positions = self.position_repository.get_all_open()
            print(f"📁 Loaded {len(positions)} positions from database")
        except Exception as e:
            print(f"⚠️ Error loading positions from database: {e}")
            # Fallback to JSON
            self._load_positions_from_json()
    
    def _load_positions_from_json(self):
        """Load position data from JSON file (fallback)"""
        try:
            cache_file = "trading_data/portfolio_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Load any cached data here
                    print("📁 Loaded cached portfolio data from JSON")
        except Exception as e:
            print(f"⚠️ Could not load cached positions from JSON: {e}")
    
    def save_portfolio_cache(self):
        """Save portfolio data to database or cache file"""
        try:
            if self.use_database:
                self._save_positions_to_db()
            else:
                self._save_positions_to_json()
        except Exception as e:
            print(f"⚠️ Could not save portfolio cache: {e}")
    
    def _save_positions_to_db(self):
        """Save position data to database"""
        try:
            # Import the Position model from data.models
            from ..data.models import Position as DBPosition, PositionStatus
            from decimal import Decimal
            import uuid
            
            # Save positions to database
            for symbol, position in self.positions.items():
                # Check if position already exists
                existing = self.position_repository.get_by_symbol(symbol)
                
                if existing:
                    # Update existing position
                    existing.quantity = Decimal(str(position.quantity))
                    existing.average_price = Decimal(str(position.average_price))
                    existing.current_price = Decimal(str(position.current_price))
                    existing.market_value = Decimal(str(position.market_value))
                    existing.unrealized_pnl = Decimal(str(position.unrealized_pnl))
                    existing.unrealized_pnl_percent = position.unrealized_pnl_percent
                    existing.last_updated = position.last_updated
                    self.position_repository.update(existing)
                else:
                    # Create new position
                    db_position = DBPosition(
                        id=str(uuid.uuid4()),
                        symbol=position.symbol,
                        quantity=Decimal(str(position.quantity)),
                        average_price=Decimal(str(position.average_price)),
                        current_price=Decimal(str(position.current_price)),
                        market_value=Decimal(str(position.market_value)),
                        unrealized_pnl=Decimal(str(position.unrealized_pnl)),
                        unrealized_pnl_percent=position.unrealized_pnl_percent,
                        status=PositionStatus.OPEN,
                        opened_at=position.last_updated,
                        last_updated=position.last_updated
                    )
                    self.position_repository.create(db_position)
            print("💾 Saved portfolio positions to database")
        except Exception as e:
            print(f"⚠️ Error saving positions to database: {e}")
            # Fallback to JSON
            self._save_positions_to_json()
    
    def _save_positions_to_json(self):
        """Save portfolio data to JSON file (fallback)"""
        try:
            os.makedirs("trading_data", exist_ok=True)
            cache_data = {
                'balances': self.balances,
                'portfolio_value': self.portfolio_value,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'positions_summary': {
                    symbol: {
                        'quantity': pos.quantity,
                        'average_price': pos.average_price,
                        'last_price': pos.current_price
                    } for symbol, pos in self.positions.items()
                }
            }
            
            with open("trading_data/portfolio_cache.json", 'w') as f:
                json.dump(cache_data, f, indent=2)
            print("💾 Saved portfolio cache to JSON")
                
        except Exception as e:
            print(f"⚠️ Could not save portfolio cache to JSON: {e}")
