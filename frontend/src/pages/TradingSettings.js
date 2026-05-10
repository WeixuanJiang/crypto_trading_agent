import React, { useState, useEffect } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { useSnackbar } from 'notistack';
import { useSettings, useUpdateSettings } from '../hooks/useTradingData';
import { tradingService } from '../services/apiService';
import { T } from '../theme/terminal';

// ── Primitives ─────────────────────────────────────────────────────────────────

function PanelHeader({ children, right }) {
  return (
    <Box sx={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      px: '14px', py: '8px', borderBottom: `1px solid ${T.border}`,
      fontSize: '10px', color: T.dim, letterSpacing: '.1em', textTransform: 'uppercase',
      fontFamily: T.font, background: T.status, flexShrink: 0,
    }}>
      <span>{children}</span>
      {right && <span>{right}</span>}
    </Box>
  );
}

function TermBtn({ children, onClick, disabled, variant = 'default', loading }) {
  const styles = {
    save:    { bg: '#001a12', border: T.green,  color: T.green },
    reset:   { bg: 'transparent', border: T.dim, color: T.text2 },
    danger:  { bg: '#1a0008', border: T.red,    color: T.red },
    default: { bg: 'transparent', border: T.border, color: T.text2 },
  };
  const s = styles[variant] || styles.default;
  return (
    <Box
      component="button"
      onClick={disabled ? undefined : onClick}
      sx={{
        display: 'inline-flex', alignItems: 'center', gap: '6px',
        px: '14px', py: '7px', fontFamily: T.font, fontSize: '11px', fontWeight: 700,
        letterSpacing: '.08em', textTransform: 'uppercase',
        cursor: disabled ? 'not-allowed' : 'pointer',
        background: s.bg, border: `1px solid ${disabled ? T.border : s.border}`,
        color: disabled ? T.vdim : s.color, opacity: disabled ? 0.5 : 1,
        transition: 'opacity .15s',
      }}
    >
      {loading && <CircularProgress size={10} sx={{ color: 'inherit' }} />}
      {children}
    </Box>
  );
}

function Field({ label, description, children }) {
  return (
    <Box sx={{ mb: '14px' }}>
      <Box sx={{ fontSize: '10px', color: T.dim, textTransform: 'uppercase', letterSpacing: '.08em', mb: '4px', fontFamily: T.font }}>
        {label}
      </Box>
      {children}
      {description && (
        <Box sx={{ fontSize: '10px', color: T.vdim, mt: '3px', fontFamily: T.font }}>{description}</Box>
      )}
    </Box>
  );
}

function TermInput({ value, onChange, type = 'text', placeholder }) {
  return (
    <Box
      component="input"
      type={type}
      value={value ?? ''}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      sx={{
        width: '100%', background: T.status, border: `1px solid ${T.border}`,
        color: T.text1, fontFamily: T.font, fontSize: '12px',
        px: '10px', py: '7px', outline: 'none',
        '&:focus': { borderColor: T.green },
        '&::placeholder': { color: T.vdim },
      }}
    />
  );
}

function TermToggle({ value, onChange, label }) {
  const on = value === 'true' || value === true;
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      <Box
        onClick={() => onChange((!on).toString())}
        sx={{
          width: '36px', height: '18px', borderRadius: '9px',
          background: on ? T.green : T.border,
          position: 'relative', cursor: 'pointer', flexShrink: 0, transition: 'background .2s',
        }}
      >
        <Box sx={{
          position: 'absolute', top: '2px',
          left: on ? '20px' : '2px',
          width: '14px', height: '14px', borderRadius: '7px',
          background: on ? '#000' : T.dim, transition: 'left .2s',
        }} />
      </Box>
      <Box sx={{ fontSize: '11px', color: on ? T.green : T.dim, fontFamily: T.font, fontWeight: 600 }}>
        {on ? 'ENABLED' : 'DISABLED'}
      </Box>
    </Box>
  );
}

function TermSelect({ value, onChange, options }) {
  return (
    <Box
      component="select"
      value={value ?? ''}
      onChange={e => onChange(e.target.value)}
      sx={{
        width: '100%', background: T.status, border: `1px solid ${T.border}`,
        color: T.text1, fontFamily: T.font, fontSize: '12px',
        px: '10px', py: '7px', outline: 'none', cursor: 'pointer',
        '&:focus': { borderColor: T.green },
      }}
    >
      {options.map(o => (
        <option key={o.value} value={o.value} style={{ background: T.panel }}>{o.label}</option>
      ))}
    </Box>
  );
}

function Section({ title, children }) {
  const [open, setOpen] = useState(true);
  return (
    <Box sx={{ borderBottom: `1px solid ${T.border}` }}>
      <Box
        onClick={() => setOpen(o => !o)}
        sx={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          px: '16px', py: '10px', cursor: 'pointer', background: T.panel,
          borderBottom: open ? `1px solid ${T.border}` : 'none',
          fontSize: '11px', fontWeight: 700, letterSpacing: '.08em',
          textTransform: 'uppercase', color: T.text2, fontFamily: T.font,
          '&:hover': { background: T.hover },
        }}
      >
        <span>{title}</span>
        <Box component="span" sx={{ color: T.dim }}>{open ? '▼' : '▶'}</Box>
      </Box>
      {open && (
        <Box sx={{ px: '16px', pt: '16px', pb: '8px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 24px' }}>
          {children}
        </Box>
      )}
    </Box>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────────

export default function TradingSettings() {
  const { enqueueSnackbar } = useSnackbar();
  const { data: settingsData, isLoading, refetch } = useSettings();
  const updateMutation = useUpdateSettings();

  const [settings, setSettings] = useState(() => settingsData?.settings || null);

  useEffect(() => {
    if (settingsData?.settings) setSettings(settingsData.settings);
  }, [settingsData]);

  const set = field => value => setSettings(s => ({ ...s, [field]: value }));

  const handleSave = async () => {
    try {
      const pairsChanged = settingsData?.settings?.TRADING_PAIRS !== settings.TRADING_PAIRS;
      await updateMutation.mutateAsync(settings);
      window.dispatchEvent(new Event('trading-status-changed'));
      enqueueSnackbar(pairsChanged ? 'Settings saved — trading pairs updated' : 'Settings saved', { variant: 'success' });
    } catch {
      enqueueSnackbar('Failed to save settings', { variant: 'error' });
    }
  };

  const handleReset = () => { refetch(); enqueueSnackbar('Settings reset', { variant: 'info' }); };

  const handleClearData = async () => {
    if (!window.confirm('Clear all logs, trades, and reset portfolio? This cannot be undone.')) return;
    try {
      const res = await tradingService.resetData({
        reset_logs: true,
        reset_trades: true,
        reset_portfolio: true,
      });
      const data = res.data;
      if (data.success) {
        enqueueSnackbar(data.message, { variant: 'success' });
        setTimeout(() => window.location.reload(), 1500);
      } else {
        enqueueSnackbar('Failed: ' + data.error, { variant: 'error' });
      }
    } catch (e) {
      enqueueSnackbar('Error: ' + e.message, { variant: 'error' });
    }
  };

  if (isLoading || !settings) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress size={20} sx={{ color: T.green }} />
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', fontFamily: T.font, overflow: 'hidden' }}>

      {/* Toolbar */}
      <PanelHeader
        right={
          <Box sx={{ display: 'flex', gap: '8px' }}>
            <TermBtn variant="danger" onClick={handleClearData}>✕ Clear Data</TermBtn>
            <TermBtn variant="reset"  onClick={handleReset}>↺ Reset</TermBtn>
            <TermBtn variant="save"   onClick={handleSave} disabled={updateMutation.isLoading} loading={updateMutation.isLoading}>
              ✓ Save Changes
            </TermBtn>
          </Box>
        }
      >
        Trading Settings
      </PanelHeader>

      <Box sx={{ flex: 1, overflow: 'auto' }}>

        <Section title="Trading Parameters">
          <Field label="Trading Pairs" description="Comma-separated pairs, e.g. BTC-USDT,ETH-USDT">
            <TermInput value={settings.TRADING_PAIRS} onChange={set('TRADING_PAIRS')} />
          </Field>
          <Field label="Interval (minutes)" description="Time between trading cycles">
            <TermInput value={settings.TRADING_INTERVAL_MINUTES} onChange={set('TRADING_INTERVAL_MINUTES')} type="number" />
          </Field>
          <Field label="Max Daily Trades">
            <TermInput value={settings.MAX_DAILY_TRADES} onChange={set('MAX_DAILY_TRADES')} type="number" />
          </Field>
          <Field label="Min Confidence Threshold" description="0.0–1.0, minimum to execute a trade">
            <TermInput value={settings.MIN_CONFIDENCE_THRESHOLD} onChange={set('MIN_CONFIDENCE_THRESHOLD')} type="number" />
          </Field>
          <Field label="Minimum Balance ($)" description="Minimum USDT balance to trade">
            <TermInput value={settings.MINIMUM_BALANCE} onChange={set('MINIMUM_BALANCE')} type="number" />
          </Field>
          <Field label="Historical Data Limit">
            <TermInput value={settings.HISTORICAL_DATA_LIMIT} onChange={set('HISTORICAL_DATA_LIMIT')} type="number" />
          </Field>
          <Field label="Enable Live Trading" description="WARNING: real funds will be used">
            <TermToggle value={settings.ENABLE_LIVE_TRADING} onChange={set('ENABLE_LIVE_TRADING')} />
          </Field>
        </Section>

        <Section title="Risk Management">
          <Field label="Max Portfolio Risk" description="Max fraction of portfolio to risk per trade">
            <TermInput value={settings.MAX_PORTFOLIO_RISK} onChange={set('MAX_PORTFOLIO_RISK')} type="number" />
          </Field>
          <Field label="Max Position Size" description="Max fraction per position">
            <TermInput value={settings.MAX_POSITION_SIZE} onChange={set('MAX_POSITION_SIZE')} type="number" />
          </Field>
          <Field label="Max Portfolio Exposure">
            <TermInput value={settings.MAX_PORTFOLIO_EXPOSURE} onChange={set('MAX_PORTFOLIO_EXPOSURE')} type="number" />
          </Field>
          <Field label="Max Open Positions">
            <TermInput value={settings.MAX_OPEN_POSITIONS} onChange={set('MAX_OPEN_POSITIONS')} type="number" />
          </Field>
          <Field label="Stop Loss (%)">
            <TermInput value={settings.STOP_LOSS_PERCENTAGE} onChange={set('STOP_LOSS_PERCENTAGE')} type="number" />
          </Field>
          <Field label="Risk/Reward Ratio">
            <TermInput value={settings.RISK_REWARD_RATIO} onChange={set('RISK_REWARD_RATIO')} type="number" />
          </Field>
          <Field label="Trailing Stop">
            <TermToggle value={settings.USE_TRAILING_STOP} onChange={set('USE_TRAILING_STOP')} />
          </Field>
          <Field label="Trailing Stop (%)">
            <TermInput value={settings.TRAILING_STOP_PERCENT} onChange={set('TRAILING_STOP_PERCENT')} type="number" />
          </Field>
        </Section>

        <Section title="Technical Indicators">
          <Field label="RSI Period">
            <TermInput value={settings.RSI_PERIOD} onChange={set('RSI_PERIOD')} type="number" />
          </Field>
          <Field label="RSI Overbought">
            <TermInput value={settings.RSI_OVERBOUGHT} onChange={set('RSI_OVERBOUGHT')} type="number" />
          </Field>
          <Field label="RSI Oversold">
            <TermInput value={settings.RSI_OVERSOLD} onChange={set('RSI_OVERSOLD')} type="number" />
          </Field>
          <Field label="MACD Fast">
            <TermInput value={settings.MACD_FAST} onChange={set('MACD_FAST')} type="number" />
          </Field>
          <Field label="MACD Slow">
            <TermInput value={settings.MACD_SLOW} onChange={set('MACD_SLOW')} type="number" />
          </Field>
          <Field label="MACD Signal">
            <TermInput value={settings.MACD_SIGNAL} onChange={set('MACD_SIGNAL')} type="number" />
          </Field>
          <Field label="Bollinger Period">
            <TermInput value={settings.BB_PERIOD} onChange={set('BB_PERIOD')} type="number" />
          </Field>
          <Field label="Bollinger Std Dev">
            <TermInput value={settings.BB_STD_DEV} onChange={set('BB_STD_DEV')} type="number" />
          </Field>
        </Section>

        <Section title="LLM Configuration">
          <Field label="Fast Mode (Technical Only)">
            <TermToggle value={settings.FAST_MODE} onChange={set('FAST_MODE')} />
          </Field>
          <Field label="LLM Provider">
            <TermSelect
              value={settings.LLM_PROVIDER}
              onChange={set('LLM_PROVIDER')}
              options={[{ value: 'openai', label: 'OpenAI' }, { value: 'bedrock', label: 'AWS Bedrock' }]}
            />
          </Field>
          <Field label="Model Name">
            <TermInput value={settings.LLM_MODEL_NAME} onChange={set('LLM_MODEL_NAME')} />
          </Field>
          <Field label="Max Tokens">
            <TermInput value={settings.LLM_MAX_TOKENS} onChange={set('LLM_MAX_TOKENS')} type="number" />
          </Field>
          <Field label="Temperature" description="0.0–1.0, lower = more deterministic">
            <TermInput value={settings.LLM_TEMPERATURE} onChange={set('LLM_TEMPERATURE')} type="number" />
          </Field>
          <Field label="Sentiment Analysis">
            <TermToggle value={settings.ENABLE_SENTIMENT_ANALYSIS} onChange={set('ENABLE_SENTIMENT_ANALYSIS')} />
          </Field>
        </Section>

        <Section title="Paper Trading">
          <Field label="Initial Balance (USDT)">
            <TermInput value={settings.PAPER_TRADING_INITIAL_BALANCE} onChange={set('PAPER_TRADING_INITIAL_BALANCE')} type="number" />
          </Field>
          <Field label="Initial BTC">
            <TermInput value={settings.PAPER_TRADING_INITIAL_BTC} onChange={set('PAPER_TRADING_INITIAL_BTC')} type="number" />
          </Field>
          <Field label="Initial ETH">
            <TermInput value={settings.PAPER_TRADING_INITIAL_ETH} onChange={set('PAPER_TRADING_INITIAL_ETH')} type="number" />
          </Field>
          <Field label="Performance Report Days">
            <TermInput value={settings.PERFORMANCE_REPORT_DAYS} onChange={set('PERFORMANCE_REPORT_DAYS')} type="number" />
          </Field>
        </Section>
      </Box>
    </Box>
  );
}
