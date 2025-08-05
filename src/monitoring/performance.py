"""
Performance monitoring system for the Crypto Trading Agent

This module provides comprehensive performance tracking, metrics collection,
and alerting capabilities to monitor the health and performance of the trading system.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import os

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    api_response_time: float
    trades_per_minute: float
    error_rate: float
    active_connections: int
    system_load: float

@dataclass
class TradingMetrics:
    """Trading-specific metrics"""
    timestamp: datetime
    total_trades: int
    successful_trades: int
    failed_trades: int
    average_confidence: float
    win_rate: float
    total_pnl: float
    sharpe_ratio: Optional[float]
    max_drawdown: float

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, monitoring_interval: int = 60):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.performance_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.trading_history: deque = deque(maxlen=1440)
        self.api_response_times: deque = deque(maxlen=100)
        self.error_counts: defaultdict = defaultdict(int)
        self.trade_counts: defaultdict = defaultdict(int)
        
        # Alerting
        self.alert_callbacks: List[Callable] = []
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'api_response_time': 5.0,
            'error_rate': 10.0,
            'max_drawdown': 15.0
        }
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_metrics_time = datetime.now()
    
    def start_monitoring(self):
        """Start the performance monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("📊 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect trading metrics
                self._collect_trading_metrics()
                
                # Check alerts
                self._check_alerts()
                
                # Sleep until next collection
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # System load
            system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
            # API response time (average of recent calls)
            avg_response_time = sum(self.api_response_times) / len(self.api_response_times) if self.api_response_times else 0.0
            
            # Calculate error rate
            total_errors = sum(self.error_counts.values())
            total_operations = sum(self.trade_counts.values()) + total_errors
            error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0.0
            
            # Trades per minute
            current_time = datetime.now()
            minute_key = current_time.strftime('%Y-%m-%d %H:%M')
            trades_per_minute = self.trade_counts[minute_key]
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=current_time,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                api_response_time=avg_response_time,
                trades_per_minute=trades_per_minute,
                error_rate=error_rate,
                active_connections=self._get_active_connections(),
                system_load=system_load
            )
            
            # Store metrics
            self.performance_history.append(metrics)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    def _collect_trading_metrics(self):
        """Collect trading performance metrics"""
        # This would integrate with the trade tracker
        # For now, we'll create a placeholder
        pass
    
    def _get_active_connections(self) -> int:
        """Get number of active network connections"""
        try:
            connections = psutil.net_connections()
            return len([conn for conn in connections if conn.status == 'ESTABLISHED'])
        except:
            return 0
    
    def _check_alerts(self):
        """Check if any metrics exceed alert thresholds"""
        if not self.performance_history:
            return
        
        latest_metrics = self.performance_history[-1]
        alerts = []
        
        # Check each threshold
        if latest_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {latest_metrics.cpu_usage:.1f}%")
        
        if latest_metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {latest_metrics.memory_usage:.1f}%")
        
        if latest_metrics.api_response_time > self.alert_thresholds['api_response_time']:
            alerts.append(f"Slow API response: {latest_metrics.api_response_time:.2f}s")
        
        if latest_metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {latest_metrics.error_rate:.1f}%")
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, message: str):
        """Trigger an alert"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': 'WARNING'
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                print(f"Error in alert callback: {e}")
        
        # Log alert
        print(f"🚨 ALERT: {message}")
    
    def record_api_call(self, response_time: float, success: bool = True):
        """Record API call metrics"""
        self.api_response_times.append(response_time)
        
        minute_key = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        if success:
            self.trade_counts[minute_key] += 1
        else:
            self.error_counts[minute_key] += 1
    
    def record_trade(self, success: bool = True):
        """Record trade execution"""
        minute_key = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        if success:
            self.trade_counts[minute_key] += 1
        else:
            self.error_counts[minute_key] += 1
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics"""
        return self.performance_history[-1] if self.performance_history else None
    
    def get_metrics_summary(self, hours: int = 1) -> Dict:
        """Get performance metrics summary for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'avg_cpu_usage': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_memory_usage': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'avg_api_response_time': sum(m.api_response_time for m in recent_metrics) / len(recent_metrics),
            'max_cpu_usage': max(m.cpu_usage for m in recent_metrics),
            'max_memory_usage': max(m.memory_usage for m in recent_metrics),
            'max_api_response_time': max(m.api_response_time for m in recent_metrics),
            'total_trades': sum(m.trades_per_minute for m in recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        }
    
    def export_metrics(self, filename: str, hours: int = 24):
        """Export metrics to JSON file"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        # Convert to serializable format
        metrics_data = []
        for metric in recent_metrics:
            data = asdict(metric)
            data['timestamp'] = metric.timestamp.isoformat()
            metrics_data.append(data)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'metrics_count': len(metrics_data),
            'metrics': metrics_data,
            'summary': self.get_metrics_summary(hours)
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Metrics exported to {filename}")
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a specific metric"""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            print(f"Alert threshold for {metric} set to {threshold}")
        else:
            print(f"Unknown metric: {metric}")
    
    def get_uptime(self) -> timedelta:
        """Get system uptime"""
        return datetime.now() - self.start_time
    
    def get_health_status(self) -> Dict:
        """Get overall system health status"""
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return {'status': 'UNKNOWN', 'message': 'No metrics available'}
        
        # Determine health status based on thresholds
        issues = []
        
        if current_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            issues.append('High CPU usage')
        
        if current_metrics.memory_usage > self.alert_thresholds['memory_usage']:
            issues.append('High memory usage')
        
        if current_metrics.api_response_time > self.alert_thresholds['api_response_time']:
            issues.append('Slow API responses')
        
        if current_metrics.error_rate > self.alert_thresholds['error_rate']:
            issues.append('High error rate')
        
        if issues:
            status = 'WARNING' if len(issues) <= 2 else 'CRITICAL'
            message = f"Issues detected: {', '.join(issues)}"
        else:
            status = 'HEALTHY'
            message = 'All systems operating normally'
        
        return {
            'status': status,
            'message': message,
            'uptime': str(self.get_uptime()),
            'last_update': current_metrics.timestamp.isoformat(),
            'issues': issues
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()