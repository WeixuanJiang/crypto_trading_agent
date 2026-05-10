import React, { useState, useEffect, useRef } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { useSnackbar } from 'notistack';
import { startTrading, stopTrading, getSystemStatus, getLogs } from '../services/apiService';
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

function TermBtn({ children, onClick, disabled, variant = 'default', loading }) {
  const colors = {
    start:   { bg: '#001a12', border: T.green,  color: T.green },
    stop:    { bg: '#1a0008', border: T.red,    color: T.red },
    default: { bg: 'transparent', border: T.border, color: T.text2 },
  };
  const c = colors[variant] || colors.default;
  return (
    <Box
      component="button"
      onClick={disabled ? undefined : onClick}
      sx={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
        px: '20px', py: '9px', fontFamily: T.font, fontSize: '11px', fontWeight: 700,
        letterSpacing: '.08em', textTransform: 'uppercase', cursor: disabled ? 'not-allowed' : 'pointer',
        background: disabled ? 'transparent' : c.bg,
        border: `1px solid ${disabled ? T.border : c.border}`,
        color: disabled ? T.vdim : c.color,
        opacity: disabled ? 0.5 : 1, transition: 'all .15s',
        '&:hover': disabled ? {} : { background: c.bg, opacity: .85 },
        width: '100%',
      }}
    >
      {loading && <CircularProgress size={12} sx={{ color: 'inherit' }} />}
      {children}
    </Box>
  );
}

function StatusRow({ label, value, color }) {
  return (
    <Box sx={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      py: '8px', borderBottom: `1px solid ${T.border}`, fontSize: '11px', fontFamily: T.font,
    }}>
      <Box sx={{ color: T.dim, textTransform: 'uppercase', letterSpacing: '.06em', fontSize: '10px' }}>{label}</Box>
      <Box sx={{ color: color || T.text1, fontWeight: 600 }}>{value ?? '--'}</Box>
    </Box>
  );
}

function LiveDot({ active }) {
  return (
    <Box component="span" sx={{
      display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
      background: active ? T.green : T.red,
      animation: active ? 'blink 1.2s infinite' : 'none',
      mr: '8px', verticalAlign: 'middle',
    }} />
  );
}

export default function TradingControl() {
  const { enqueueSnackbar } = useSnackbar();
  const [status, setStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [logFilter, setLogFilter] = useState('ALL');
  const endRef = useRef(null);

  useEffect(() => {
    fetchStatus();
    fetchLogs();
    const si = setInterval(fetchStatus, 30000);
    const li = setInterval(fetchLogs, 5000);
    return () => { clearInterval(si); clearInterval(li); };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const fetchStatus = async () => {
    try {
      const data = await getSystemStatus();
      setStatus(data.status || data);
    } catch {}
  };

  const fetchLogs = async () => {
    try {
      const data = await getLogs(200);
      setLogs(data.logs || []);
    } catch {}
  };

  const notifyLayout = () => window.dispatchEvent(new Event('trading-status-changed'));

  const handleStart = async () => {
    try {
      setLoading(true);
      await startTrading();
      enqueueSnackbar('Trading started', { variant: 'success' });
      await fetchStatus();
      notifyLayout();
    } catch (e) {
      enqueueSnackbar(e.message || 'Failed to start', { variant: 'error' });
    } finally { setLoading(false); }
  };

  const handleStop = async () => {
    try {
      setLoading(true);
      await stopTrading();
      enqueueSnackbar('Trading stopped', { variant: 'info' });
      await fetchStatus();
      notifyLayout();
    } catch (e) {
      enqueueSnackbar(e.message || 'Failed to stop', { variant: 'error' });
    } finally { setLoading(false); }
  };

  const isRunning = status?.is_running || false;
  const isAuto = status?.auto_trading || false;
  const mode = isAuto ? 'LIVE' : 'PAPER';

  const lvColor = lv => {
    if (!lv) return T.dim;
    const u = lv.toUpperCase();
    if (u === 'ERROR' || u === 'CRITICAL') return T.red;
    if (u === 'WARNING' || u === 'WARN') return T.yellow;
    if (u === 'INFO') return T.green;
    return T.dim;
  };

  const filteredLogs = logFilter === 'ALL'
    ? logs
    : logs.filter(l => (l.level || '').toUpperCase().startsWith(logFilter));

  const FILTERS = ['ALL', 'INFO', 'WARN', 'ERROR'];

  return (
    <Box sx={{ display: 'flex', height: '100%', fontFamily: T.font, overflow: 'hidden' }}>

      {/* Left: controls */}
      <Box sx={{
        width: '300px', flexShrink: 0, display: 'flex', flexDirection: 'column',
        borderRight: `1px solid ${T.border}`,
      }}>
        <PanelHeader>AI Trading Controls</PanelHeader>

        <Box sx={{ flex: 1, overflow: 'auto', p: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* Status block */}
          <Box>
            <StatusRow
              label="Agent Status"
              value={<><LiveDot active={isRunning} />{isRunning ? 'RUNNING' : 'STOPPED'}</>}
              color={isRunning ? T.green : T.red}
            />
            <StatusRow label="Trading Mode" value={mode} color={isAuto ? T.red : T.yellow} />
            <StatusRow label="LLM Mode" value={status?.fast_mode === false ? 'FULL (Bedrock)' : 'FAST (Technical)'} />
            <StatusRow label="Active Pairs" value={status?.trading_pairs?.length ?? '--'} />
            <StatusRow label="Uptime" value={status?.uptime || '--'} />
            <StatusRow label="Last Analysis" value={status?.last_analysis ? new Date(status.last_analysis).toLocaleTimeString() : '--'} />
            <StatusRow label="Trades Today" value={status?.trades_today ?? '--'} />
            <StatusRow label="Open Positions" value={status?.active_positions ?? '--'} />
          </Box>

          {/* Warning */}
          {isAuto && (
            <Box sx={{
              p: '10px', border: `1px solid ${T.red}44`, background: '#1a0008',
              fontSize: '11px', color: T.red,
            }}>
              ⚠ LIVE TRADING ACTIVE — Real funds at risk
            </Box>
          )}

          {/* Buttons */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: '10px', mt: 'auto' }}>
            <TermBtn
              variant="start"
              onClick={handleStart}
              disabled={loading || isRunning}
              loading={loading && !isRunning}
            >
              ▶ Start AI Trading
            </TermBtn>
            <TermBtn
              variant="stop"
              onClick={handleStop}
              disabled={loading || !isRunning}
              loading={loading && isRunning}
            >
              ■ Stop Trading
            </TermBtn>
            <TermBtn onClick={fetchStatus} disabled={loading}>
              ↻ Refresh Status
            </TermBtn>
          </Box>
        </Box>

        {/* Session info footer */}
        <Box sx={{ borderTop: `1px solid ${T.border}`, p: '12px', fontSize: '10px', color: T.vdim, lineHeight: 1.8 }}>
          <div>Mode: {status?.fast_mode === false ? 'Full LLM+Technical' : 'Fast Technical Only'}</div>
          <div>Pairs: {status?.trading_pairs?.join(', ') || '--'}</div>
        </Box>
      </Box>

      {/* Right: log panel */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <PanelHeader
          right={
            <Box sx={{ display: 'flex', gap: '6px' }}>
              {FILTERS.map(f => (
                <Box
                  key={f}
                  onClick={() => setLogFilter(f)}
                  sx={{
                    px: '8px', py: '2px', fontSize: '10px', fontWeight: 600, cursor: 'pointer',
                    border: `1px solid ${logFilter === f ? T.green : T.border}`,
                    color: logFilter === f ? T.green : T.dim,
                    background: logFilter === f ? '#001e28' : 'transparent',
                  }}
                >
                  {f}
                </Box>
              ))}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: '6px', ml: '8px', color: T.green, fontSize: '10px' }}>
                <Box sx={{ width: 6, height: 6, borderRadius: '50%', background: T.green, animation: 'blink 1.2s infinite' }} />
                LIVE
              </Box>
            </Box>
          }
        >
          System Log
        </PanelHeader>

        <Box sx={{ flex: 1, overflow: 'auto', p: '10px', background: T.status }}>
          {filteredLogs.length === 0 ? (
            <Box sx={{ fontSize: '10px', color: T.vdim }}>No logs available</Box>
          ) : (
            filteredLogs.map((log, i) => {
              const ts = log.timestamp || '--:--:--';
              const lv = (log.level || 'INFO').toUpperCase();
              return (
                <Box key={i} sx={{ fontSize: '10px', lineHeight: 1.7, fontFamily: T.font }}>
                  <Box component="span" sx={{ color: T.vdim }}>[{ts}]</Box>
                  {' '}
                  <Box component="span" sx={{ color: lvColor(lv), display: 'inline-block', minWidth: '5ch' }}>{lv.slice(0, 4)}</Box>
                  {' '}
                  <Box component="span" sx={{ color: T.text2 }}>{log.message}</Box>
                </Box>
              );
            })
          )}
          <div ref={endRef} />
        </Box>
      </Box>
    </Box>
  );
}
