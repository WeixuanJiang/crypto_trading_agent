import React, { useState } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { tradingService } from '../services/apiService';
import { T } from '../theme/terminal';

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

function Field({ label, description, children }) {
  return (
    <Box sx={{ mb: '16px' }}>
      <Box sx={{ fontSize: '10px', color: T.dim, textTransform: 'uppercase', letterSpacing: '.08em', mb: '4px', fontFamily: T.font }}>
        {label}
      </Box>
      {children}
      {description && (
        <Box sx={{ fontSize: '10px', color: T.vdim, mt: '4px', fontFamily: T.font }}>{description}</Box>
      )}
    </Box>
  );
}

function TermInput({ value, onChange, type = 'text', placeholder, masked }) {
  const [show, setShow] = useState(false);
  return (
    <Box sx={{ display: 'flex', gap: '0' }}>
      <Box
        component="input"
        type={masked && !show ? 'password' : type}
        value={value ?? ''}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder || (masked ? '••••••••••••••••' : '')}
        sx={{
          flex: 1, background: T.status, border: `1px solid ${T.border}`,
          borderRight: masked ? 'none' : `1px solid ${T.border}`,
          color: T.text1, fontFamily: T.font, fontSize: '12px',
          px: '10px', py: '8px', outline: 'none',
          '&:focus': { borderColor: T.green },
          '&::placeholder': { color: T.vdim },
        }}
      />
      {masked && (
        <Box
          component="button"
          onClick={() => setShow(s => !s)}
          sx={{
            px: '10px', background: T.panel, border: `1px solid ${T.border}`,
            color: T.dim, cursor: 'pointer', fontSize: '11px', fontFamily: T.font,
            '&:hover': { color: T.text2 },
          }}
        >
          {show ? '🙈' : '👁'}
        </Box>
      )}
    </Box>
  );
}

function TermBtn({ children, onClick, disabled, variant = 'default', loading }) {
  const styles = {
    save:    { bg: 'rgba(168,85,247,0.08)', border: T.green,  color: T.green },
    default: { bg: 'transparent', border: T.border, color: T.text2 },
  };
  const s = styles[variant] || styles.default;
  return (
    <Box
      component="button"
      onClick={disabled ? undefined : onClick}
      sx={{
        display: 'inline-flex', alignItems: 'center', gap: '6px',
        px: '16px', py: '8px', fontFamily: T.font, fontSize: '11px', fontWeight: 700,
        letterSpacing: '.08em', textTransform: 'uppercase',
        cursor: disabled ? 'not-allowed' : 'pointer',
        background: s.bg, border: `1px solid ${disabled ? T.border : s.border}`,
        color: disabled ? T.vdim : s.color, opacity: disabled ? 0.5 : 1,
      }}
    >
      {loading && <CircularProgress size={10} sx={{ color: 'inherit' }} />}
      {children}
    </Box>
  );
}

function InfoBox({ children, variant = 'info' }) {
  const color = { info: T.blue, warn: T.yellow, error: T.red }[variant] || T.blue;
  return (
    <Box sx={{
      p: '10px', mb: '20px', border: `1px solid ${color}44`,
      background: `${color}0a`, fontSize: '11px', color, fontFamily: T.font,
    }}>
      {children}
    </Box>
  );
}

export default function Settings() {
  const [settings, setSettings] = useState({
    kucoin_api_key: '',
    kucoin_api_secret: '',
    kucoin_api_passphrase: '',
    openai_api_key: '',
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const set = field => value => setSettings(s => ({ ...s, [field]: value }));

  const handleSave = async () => {
    try {
      setSaving(true);
      const res = await tradingService.updateSettings(settings);
      if (res.data?.success) {
        setSuccess('Credentials saved');
        setError(null);
        if (res.data.settings) setSettings(res.data.settings);
        setTimeout(() => setSuccess(null), 3000);
      } else {
        throw new Error('Save failed');
      }
    } catch (err) {
      setError(err.message || 'Failed to save');
      setSuccess(null);
    } finally { setSaving(false); }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', fontFamily: T.font, overflow: 'hidden' }}>

      <PanelHeader
        right={
          <TermBtn variant="save" onClick={handleSave} disabled={saving} loading={saving}>
            ✓ Save Credentials
          </TermBtn>
        }
      >
        Account &amp; API Keys
      </PanelHeader>

      <Box sx={{ flex: 1, overflow: 'auto', p: '24px', maxWidth: '700px' }}>

        {error   && <InfoBox variant="error">{error}</InfoBox>}
        {success && <InfoBox variant="info">✓ {success}</InfoBox>}

        <InfoBox variant="warn">
          ⚠ API keys are stored locally. Never share them. KuCoin keys only need trade + withdrawal permissions for live trading, or read-only for paper trading.
        </InfoBox>

        {/* KuCoin */}
        <Box sx={{ mb: '28px' }}>
          <Box sx={{ fontSize: '11px', fontWeight: 700, color: T.text2, letterSpacing: '.08em', textTransform: 'uppercase', mb: '16px', pb: '8px', borderBottom: `1px solid ${T.border}` }}>
            KuCoin API
          </Box>
          <Field label="API Key" description="Your KuCoin API key from the API Management page">
            <TermInput value={settings.kucoin_api_key} onChange={set('kucoin_api_key')} masked />
          </Field>
          <Field label="API Secret">
            <TermInput value={settings.kucoin_api_secret} onChange={set('kucoin_api_secret')} masked />
          </Field>
          <Field label="Passphrase" description="The passphrase you set when creating the API key">
            <TermInput value={settings.kucoin_api_passphrase} onChange={set('kucoin_api_passphrase')} masked />
          </Field>
        </Box>

        {/* OpenAI */}
        <Box sx={{ mb: '28px' }}>
          <Box sx={{ fontSize: '11px', fontWeight: 700, color: T.text2, letterSpacing: '.08em', textTransform: 'uppercase', mb: '16px', pb: '8px', borderBottom: `1px solid ${T.border}` }}>
            OpenAI API
          </Box>
          <Field label="OpenAI API Key" description="Required when LLM_PROVIDER=openai in Trading Settings">
            <TermInput value={settings.openai_api_key} onChange={set('openai_api_key')} masked />
          </Field>
        </Box>

        {/* Info */}
        <Box sx={{ fontSize: '11px', color: T.vdim, lineHeight: 1.8 }}>
          <div>• AWS Bedrock credentials are managed via IAM roles / environment variables</div>
          <div>• Keys are used only for KuCoin API calls and LLM analysis</div>
          <div>• Leave fields blank to continue using environment variable defaults</div>
        </Box>
      </Box>
    </Box>
  );
}
