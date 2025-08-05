"""Notifications Module

This module provides notification capabilities including:
- Trade execution notifications
- System alerts and warnings
- Performance threshold notifications
- Multi-channel notification support
"""

from .service import NotificationService

__all__ = [
    'NotificationService'
]