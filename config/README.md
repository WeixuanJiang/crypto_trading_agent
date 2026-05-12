# Configuration Directory

This directory contains the application settings stored in JSON format.

## Files

- **settings.json.template** - Template file with default settings (committed to git)
- **settings.json** - Actual settings file with your configuration (ignored by git)

## Setup

The `settings.json` file is automatically created when you save settings through the web UI. If you need to manually create it, copy the template:

```bash
cp config/settings.json.template config/settings.json
```

## Configuration via Web UI

All settings should be configured through the web interface:

1. **Account Settings** - `http://localhost:8501/settings`
   - KuCoin API credentials
   - OpenAI API key

2. **Trading Settings** - `http://localhost:8501/trading-settings`
   - Trading parameters (pairs, intervals, confidence thresholds)
   - Risk management (position sizing, stop losses)
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - LLM configuration (provider, model, temperature)
   - Performance and notification settings

## File Structure

```json
{
  "api_credentials": {
    "kucoin_api_key": "your-key",
    "kucoin_api_secret": "your-secret",
    "kucoin_api_passphrase": "your-passphrase",
    "openai_api_key": "sk-..."
  },
  "trading_parameters": { ... },
  "risk_management": { ... },
  "technical_indicators": { ... },
  "llm_configuration": { ... },
  "performance": { ... },
  "notifications": { ... }
}
```

## Security

- **Never commit `settings.json`** to version control (it's already in `.gitignore`)
- API keys and secrets are stored in this file
- Keep backups of your `settings.json` in a secure location
- For production deployments, ensure proper file permissions (e.g., `chmod 600 config/settings.json`)

## Persistence

All changes made through the web UI are automatically persisted to `config/settings.json`. No need to manually edit this file unless you're doing batch updates or migrations.
