import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Box } from '@mui/material';
import axios from 'axios';
import { T } from '../theme/terminal';
import { API_BASE_URL } from '../utils/constants';

const TABS = [
  { label: 'DASHBOARD', path: '/' },
  { label: 'TRADE LOG',  path: '/history' },
  { label: 'ANALYTICS', path: '/analytics' },
  { label: 'CONTROLS',  path: '/control' },
  { label: 'SETTINGS',  path: '/trading-settings' },
];


function StatusItem({ label, value, color }) {
  return (
    <Box component="span" sx={{ fontFamily: T.font, fontSize: '11px', whiteSpace: 'nowrap' }}>
      <Box component="span" sx={{ color: T.dim }}>{label} </Box>
      <Box component="span" sx={{ color }}>{value}</Box>
    </Box>
  );
}

function Sep() {
  return <Box component="span" sx={{ color: T.border, mx: '4px' }}>│</Box>;
}

function LiveDot() {
  return (
    <Box component="span" sx={{
      display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
      background: T.green, animation: 'blink 1.2s infinite', mr: '6px',
      verticalAlign: 'middle',
    }} />
  );
}

export default function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [status, setStatus] = useState(null);
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/status`, { timeout: 5000 });
        if (res.data?.success) setStatus(res.data.status);
      } catch {}
    };
    fetchStatus();
    const si = setInterval(fetchStatus, 30000);
    const ti = setInterval(() => setNow(new Date()), 1000);
    window.addEventListener('trading-status-changed', fetchStatus);
    return () => {
      clearInterval(si);
      clearInterval(ti);
      window.removeEventListener('trading-status-changed', fetchStatus);
    };
  }, []);

  const mode = status?.auto_trading ? 'LIVE' : 'PAPER';
  const modeColor = mode === 'LIVE' ? T.red : T.yellow;
  const isOnline = !!status;
  const pairs = status?.trading_pairs?.join(' ') || 'BTC-USDT ETH-USDT ADA-USDT DOT-USDT';
  const llmLabel = status?.fast_mode === false ? 'BEDROCK/HAIKU' : 'FAST-MODE';
  const utcStr = now.toISOString().slice(0, 19).replace('T', ' ');

  return (
    <Box sx={{
      display: 'flex', flexDirection: 'column', height: '100vh',
      background: T.bg, fontFamily: T.font, color: T.text2, overflow: 'hidden',
    }}>

      {/* ── Tab bar ── */}
      <Box sx={{
        display: 'flex', alignItems: 'stretch',
        background: T.panel, borderBottom: `1px solid ${T.border}`, flexShrink: 0,
      }}>
        {TABS.map(tab => {
          const active = location.pathname === tab.path;
          return (
            <Box
              key={tab.path}
              onClick={() => navigate(tab.path)}
              sx={{
                px: '22px', py: '10px', fontSize: '11px', fontWeight: 600,
                letterSpacing: '.08em', cursor: 'pointer', userSelect: 'none',
                color: active ? T.green : T.dim,
                borderRight: `1px solid ${T.border}`,
                borderBottom: active ? `2px solid ${T.green}` : '2px solid transparent',
                background: active ? T.bg : 'transparent',
                transition: 'color .15s',
                '&:hover': { color: active ? T.green : T.text2 },
              }}
            >
              {tab.label}
            </Box>
          );
        })}
        <Box sx={{ flex: 1 }} />
        {/* Live indicator */}
        <Box sx={{
          display: 'flex', alignItems: 'center', px: 2,
          fontSize: '11px', color: isOnline ? T.green : T.dim,
        }}>
          {isOnline && <LiveDot />}
          {isOnline ? 'AGENT ONLINE' : 'CONNECTING...'}
        </Box>
      </Box>

      {/* ── Status row ── */}
      <Box sx={{
        display: 'flex', alignItems: 'center', gap: 0,
        background: T.status, borderBottom: `1px solid ${T.border}`,
        px: 2, py: '6px', fontSize: '11px', flexShrink: 0, flexWrap: 'wrap', rowGap: '4px',
      }}>
        <StatusItem label="SYS"   value={isOnline ? 'ONLINE' : 'OFFLINE'} color={isOnline ? T.green : T.red} />
        <Sep />
        <StatusItem label="MODE"  value={mode}     color={modeColor} />
        <Sep />
        <StatusItem label="LLM"   value={llmLabel} color={T.green} />
        <Sep />
        <StatusItem label="PAIRS" value={pairs}    color={T.text2} />
        <Sep />
        <StatusItem label="TRADES TODAY" value={status?.daily_trade_count ?? '--'} color={T.text2} />
        <Sep />
        <StatusItem label="UTC" value={utcStr} color={T.green} />
      </Box>

      {/* ── Body ── */}
      <Box sx={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
        {children}
      </Box>

      {/* ── Ticker bar ── */}
      <TickerBar status={status} />
    </Box>
  );
}

function TickerBar({ status }) {
  const pairs = status?.trading_pairs || ['BTC-USDT', 'ETH-USDT', 'ADA-USDT', 'DOT-USDT'];
  const [prices, setPrices] = useState({});

  useEffect(() => {
    const fetchPrices = async () => {
      const results = {};
      for (const pair of pairs) {
        try {
          const res = await axios.get(`${API_BASE_URL}/api/market/price/${pair}`, { timeout: 3000 });
          if (res.data?.success) results[pair] = res.data;
        } catch {}
      }
      if (Object.keys(results).length) setPrices(results);
    };
    fetchPrices();
    const id = setInterval(fetchPrices, 15000);
    return () => clearInterval(id);
  }, [pairs.join(',')]);

  return (
    <Box sx={{
      display: 'flex', alignItems: 'center', gap: '28px', overflow: 'hidden',
      background: T.sidebar, borderTop: `1px solid ${T.border}`,
      px: 2, py: '6px', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0,
      fontFamily: T.font,
    }}>
      {pairs.map(pair => {
        const d = prices[pair];
        const price = d?.price ?? '--';
        const change = d?.change_percent;
        const isPos = change >= 0;
        const sym = pair.replace('-USDT', '');
        return (
          <Box key={pair} sx={{ display: 'flex', gap: '8px' }}>
            <Box component="span" sx={{ color: T.dim, fontWeight: 700 }}>{sym}</Box>
            <Box component="span" sx={{ color: T.text1 }}>
              {typeof price === 'number' ? `$${price.toLocaleString()}` : price}
            </Box>
            {change !== undefined && (
              <Box component="span" sx={{ color: isPos ? T.green : T.red }}>
                {isPos ? '▲+' : '▼'}{Math.abs(change).toFixed(2)}%
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
