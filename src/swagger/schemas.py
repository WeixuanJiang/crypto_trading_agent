"""
Schemas for API request and response validation using Marshmallow
"""

from marshmallow import Schema, fields


# Request Schemas
class StartTradingRequestSchema(Schema):
    paper_trading = fields.Boolean(required=False, default=True)


class AnalysisRequestSchema(Schema):
    symbol = fields.String(required=False)


class SettingsRequestSchema(Schema):
    trading_pairs = fields.List(fields.String(), required=False)
    min_confidence_threshold = fields.Float(required=False)
    max_daily_trades = fields.Integer(required=False)
    trading_interval_minutes = fields.Integer(required=False)


# Response Schemas
class ApiInfoSchema(Schema):
    name = fields.String()
    version = fields.String()
    status = fields.String()
    endpoints = fields.Dict(keys=fields.String(), values=fields.String())
    documentation = fields.String()


class StatusSchema(Schema):
    auto_trading = fields.Boolean()
    is_running = fields.Boolean()
    fast_mode = fields.Boolean()
    trading_pairs = fields.List(fields.String())
    balance = fields.Float()
    daily_trade_count = fields.Integer()
    max_daily_trades = fields.Integer()
    min_confidence_threshold = fields.Float()
    trading_interval_minutes = fields.Integer()
    last_update = fields.DateTime(format="iso")


class StatusResponseSchema(Schema):
    success = fields.Boolean()
    status = fields.Nested(StatusSchema)


class TradingResponseSchema(Schema):
    success = fields.Boolean()
    message = fields.String()
    auto_trading = fields.Boolean()
    is_running = fields.Boolean()


class AnalysisResponseSchema(Schema):
    success = fields.Boolean()
    message = fields.String()
    analysis = fields.Dict()


class SettingsSchema(Schema):
    trading_pairs = fields.List(fields.String())
    min_confidence_threshold = fields.Float()
    max_daily_trades = fields.Integer()
    trading_interval_minutes = fields.Integer()


class SettingsResponseSchema(Schema):
    success = fields.Boolean()
    message = fields.String()
    settings = fields.Nested(SettingsSchema)


class PositionSchema(Schema):
    symbol = fields.String()
    quantity = fields.Float()
    market_value = fields.Float()
    unrealized_pnl = fields.Float()
    unrealized_pnl_percent = fields.Float()


class PortfolioSchema(Schema):
    balance = fields.Float()
    positions = fields.List(fields.Nested(PositionSchema))
    total_value = fields.Float()
    unrealized_pnl = fields.Float()
    last_update = fields.DateTime(format="iso")


class PortfolioResponseSchema(Schema):
    success = fields.Boolean()
    portfolio = fields.Nested(PortfolioSchema)


class TradeSchema(Schema):
    id = fields.String()
    timestamp = fields.DateTime(format="iso")
    symbol = fields.String()
    action = fields.String()
    price = fields.Float()
    size = fields.Float()
    value = fields.Float()
    confidence = fields.Float()
    pnl = fields.Float(allow_none=True)
    status = fields.String()
    order_id = fields.String(allow_none=True)


class TradeStatisticsSchema(Schema):
    total_trades = fields.Integer()
    executed_trades = fields.Integer()
    paper_trades = fields.Integer()
    win_rate = fields.Float()
    total_pnl = fields.Float()


class TradeHistoryResponseSchema(Schema):
    success = fields.Boolean()
    trades = fields.List(fields.Nested(TradeSchema))
    statistics = fields.Nested(TradeStatisticsSchema)
    total_trades = fields.Integer()
    last_update = fields.DateTime(format="iso")


class MarketPriceResponseSchema(Schema):
    success = fields.Boolean()
    symbol = fields.String()
    price = fields.Float()
    price_change_percent = fields.Float()
    volume_24h = fields.Float()
    high_24h = fields.Float()
    low_24h = fields.Float()
    timestamp = fields.DateTime(format="iso")


class LogEntrySchema(Schema):
    timestamp = fields.String()
    level = fields.String()
    message = fields.String()


class LogsResponseSchema(Schema):
    success = fields.Boolean()
    logs = fields.List(fields.Nested(LogEntrySchema))
    total_count = fields.Integer()
    timestamp = fields.DateTime(format="iso")


class ErrorResponseSchema(Schema):
    success = fields.Boolean()
    error = fields.String()