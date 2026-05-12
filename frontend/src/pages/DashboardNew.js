import React, { useState, useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import axios from 'axios';
import { T } from '../theme/terminal';
import { API_BASE_URL } from '../utils/constants';
import { useSubNav } from '../hooks/useSubNav';

// ── Formatters ─────────────────────────────────────────────────────────────────

const fmtUSD = v => { const n = Number(v); return (v == null || isNaN(n)) ? '--' : `$${n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`; };
const fmtNum = (v, d = 6) => { const n = Number(v); return (v == null || isNaN(n)) ? '--' : n.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: d }); };
const fmtPct = v => { const n = Number(v); return (v == null || isNaN(n)) ? '--' : `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`; };

// ── Shared primitives ──────────────────────────────────────────────────────────

function PanelHeader({ children, right }) {
  return (
    <Box sx={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      px: '14px', py: '7px', borderBottom: `1px solid ${T.border}`,
      fontSize: '10px', color: T.dim, letterSpacing: '.1em', textTransform: 'uppercase',
      fontFamily: T.font, background: T.status, flexShrink: 0,
    }}>
      <span>{children}</span>
      {right && <span>{right}</span>}
    </Box>
  );
}

function LiveDot() {
  return (
    <Box component="span" sx={{
      display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
      background: T.green, animation: 'blink 1.2s infinite',
    }} />
  );
}

// ── KPI strip ──────────────────────────────────────────────────────────────────

function KpiRow({ portfolio, status }) {
  const kpis = [
    {
      label: 'Portfolio Value',
      value: fmtUSD(portfolio?.total_value ?? status?.balance),
      delta: null,
    },
    {
      label: 'Total P&L',
      value: fmtUSD(portfolio?.total_pnl),
      valueColor: (portfolio?.total_pnl ?? 0) >= 0 ? T.green : T.red,
    },
    {
      label: 'Win Rate',
      value: portfolio?.win_rate != null ? `${Number(portfolio.win_rate).toFixed(1)}%` : '--',
    },
    {
      label: 'Trades Today / Max',
      value: `${status?.daily_trade_count ?? '--'} / ${status?.max_daily_trades ?? '--'}`,
    },
  ];

  return (
    <Box sx={{ display: 'flex', borderBottom: `1px solid ${T.border}`, flexShrink: 0 }}>
      {kpis.map((k, i) => (
        <Box key={k.label} sx={{
          flex: 1, px: '16px', py: '12px',
          borderRight: i < kpis.length - 1 ? `1px solid ${T.border}` : 'none',
        }}>
          <Box sx={{ fontSize: '10px', color: T.dim, letterSpacing: '.1em', textTransform: 'uppercase', mb: '3px', fontFamily: T.font }}>
            {k.label}
          </Box>
          <Box sx={{ fontSize: '20px', fontWeight: 700, color: k.valueColor || T.text1, fontFamily: T.font }}>
            {k.value}
          </Box>
          {k.delta != null && (
            <Box sx={{ fontSize: '11px', color: k.delta >= 0 ? T.green : T.red, mt: '2px', fontFamily: T.font }}>
              {k.delta >= 0 ? '▲' : '▼'} {Math.abs(k.delta).toFixed(2)}%
            </Box>
          )}
        </Box>
      ))}
    </Box>
  );
}

// ── Holdings table ─────────────────────────────────────────────────────────────

function HoldingsPanel({ holdings, totalValue }) {
  const rows = holdings || [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <PanelHeader right={<Box sx={{ color: T.green, display: 'flex', alignItems: 'center', gap: '6px' }}><LiveDot /> LIVE</Box>}>
        Holdings
      </PanelHeader>
      <Box sx={{ overflow: 'auto', flex: 1 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: T.font }}>
          <thead>
            <tr>
              {['Currency', 'Amount', 'Value (USDT)', 'Allocation'].map((col, i) => (
                <th key={col} style={{
                  padding: '6px 12px', fontSize: '10px', color: T.dim,
                  letterSpacing: '.08em', textTransform: 'uppercase',
                  borderBottom: `1px solid ${T.border}`, background: T.status,
                  textAlign: i > 1 ? 'right' : 'left',
                  position: 'sticky', top: 0,
                }}>
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={4} style={{ padding: '20px 12px', color: T.vdim, fontSize: '11px', textAlign: 'center' }}>
                  No holdings data
                </td>
              </tr>
            ) : (
              rows.map((h, i) => {
                const pct = (totalValue > 0 && h.usdt_value != null)
                  ? (h.usdt_value / totalValue * 100) : null;
                return (
                  <tr key={h.currency} style={{ background: i % 2 === 0 ? T.status : 'transparent' }}>
                    <td style={{ padding: '8px 12px', color: T.text1, fontWeight: 700, fontSize: '12px' }}>
                      {h.currency}
                    </td>
                    <td style={{ padding: '8px 12px', color: T.text2, fontSize: '11px' }}>
                      {fmtNum(h.amount)}
                    </td>
                    <td style={{ padding: '8px 12px', color: T.text1, fontSize: '11px', textAlign: 'right' }}>
                      {h.usdt_value != null ? fmtUSD(h.usdt_value) : <span style={{ color: T.vdim }}>--</span>}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                      {pct != null ? (
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px' }}>
                          <Box sx={{
                            height: '4px', width: `${Math.min(pct, 100)}px`, maxWidth: '80px',
                            background: T.green, opacity: 0.7,
                          }} />
                          <Box sx={{ fontSize: '10px', color: T.dim, minWidth: '36px', textAlign: 'right' }}>
                            {pct.toFixed(1)}%
                          </Box>
                        </Box>
                      ) : <span style={{ color: T.vdim, fontSize: '10px' }}>--</span>}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </Box>
    </Box>
  );
}

// ── Open positions ─────────────────────────────────────────────────────────────

function PositionsPanel({ positions }) {
  const rows = positions || [];
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <PanelHeader right={<Box sx={{ fontSize: '10px', color: T.dim }}>{rows.length} open</Box>}>
        Open Positions
      </PanelHeader>
      <Box sx={{ overflow: 'auto', flex: 1 }}>
        {rows.length === 0 ? (
          <Box sx={{ p: '14px 12px', fontSize: '11px', color: T.vdim, fontFamily: T.font }}>
            No open positions
          </Box>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: T.font }}>
            <thead>
              <tr>
                {['Symbol', 'Qty', 'Avg Price', 'Current', 'Value', 'P&L'].map((col, i) => (
                  <th key={col} style={{
                    padding: '6px 12px', fontSize: '10px', color: T.dim,
                    letterSpacing: '.08em', textTransform: 'uppercase',
                    borderBottom: `1px solid ${T.border}`, background: T.status,
                    textAlign: i > 1 ? 'right' : 'left',
                    position: 'sticky', top: 0,
                  }}>
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((p, i) => {
                const pnl = p.unrealized_pnl ?? p.pnl ?? 0;
                return (
                  <tr key={p.symbol || i} style={{ background: i % 2 === 0 ? T.status : 'transparent' }}>
                    <td style={{ padding: '8px 12px', color: T.text1, fontWeight: 700, fontSize: '12px' }}>{p.symbol}</td>
                    <td style={{ padding: '8px 12px', color: T.text2, fontSize: '11px' }}>{fmtNum(p.quantity, 8)}</td>
                    <td style={{ padding: '8px 12px', color: T.text2, fontSize: '11px', textAlign: 'right' }}>{fmtUSD(p.average_price)}</td>
                    <td style={{ padding: '8px 12px', color: T.text1, fontSize: '11px', textAlign: 'right' }}>{fmtUSD(p.current_price)}</td>
                    <td style={{ padding: '8px 12px', color: T.text1, fontSize: '11px', textAlign: 'right' }}>{fmtUSD(p.market_value)}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right', fontSize: '11px', fontWeight: 600, color: pnl >= 0 ? T.green : T.red }}>
                      {pnl >= 0 ? '+' : ''}{fmtUSD(pnl)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </Box>
    </Box>
  );
}

// ── Market overview ────────────────────────────────────────────────────────────

function MarketPanel({ pairs }) {
  const [prices, setPrices] = useState({});
  const api = axios.create({ baseURL: API_BASE_URL, timeout: 8000 });

  useEffect(() => {
    if (!pairs?.length) return;
    const fetch = async () => {
      const results = {};
      await Promise.allSettled(pairs.map(async p => {
        try {
          const r = await api.get(`/api/market/price/${p}`);
          if (r.data?.success) results[p] = r.data;
        } catch {}
      }));
      setPrices(results);
    };
    fetch();
    const id = setInterval(fetch, 15000);
    return () => clearInterval(id);
  }, [pairs?.join(',')]); // eslint-disable-line react-hooks/exhaustive-deps

  const displayPairs = pairs?.length ? pairs : ['BTC-USDT', 'ETH-USDT'];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <PanelHeader>Market Prices</PanelHeader>
      <Box sx={{ overflow: 'auto', flex: 1 }}>
        {displayPairs.map((pair, i) => {
          const d = prices[pair];
          const chg = d?.price_change_percent;
          return (
            <Box key={pair} sx={{
              display: 'grid',
              gridTemplateColumns: '110px 1fr 80px 80px 80px',
              gap: '8px', px: '12px', py: '10px',
              borderBottom: `1px solid ${T.border}`,
              fontSize: '11px', fontFamily: T.font,
              background: i % 2 === 0 ? T.status : 'transparent',
              alignItems: 'center',
            }}>
              <Box sx={{ color: T.text1, fontWeight: 700 }}>{pair}</Box>
              <Box sx={{ color: d ? (chg >= 0 ? T.green : T.red) : T.dim, fontWeight: 700, fontSize: '13px' }}>
                {d ? fmtUSD(d.price) : '--'}
              </Box>
              <Box sx={{ color: chg != null ? (chg >= 0 ? T.green : T.red) : T.dim, textAlign: 'right' }}>
                {chg != null ? fmtPct(chg) : '--'}
              </Box>
              <Box sx={{ color: T.dim, textAlign: 'right' }}>
                H: {d ? fmtUSD(d.high_24h) : '--'}
              </Box>
              <Box sx={{ color: T.dim, textAlign: 'right' }}>
                L: {d ? fmtUSD(d.low_24h) : '--'}
              </Box>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

// ── Trade feed ─────────────────────────────────────────────────────────────────

function TradeFeed({ trades }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1 }}>
      <PanelHeader right={<Box sx={{ display: 'flex', alignItems: 'center', gap: '6px', color: T.green }}><LiveDot />LIVE</Box>}>
        Recent Trades
      </PanelHeader>
      <Box sx={{ overflow: 'auto', flex: 1 }}>
        {!trades?.length ? (
          <Box sx={{ p: '14px 12px', fontSize: '11px', color: T.vdim, fontFamily: T.font }}>No trades yet</Box>
        ) : (
          trades.slice(0, 10).map((tr, i) => {
            const isBuy = (tr.side || tr.action || '').toUpperCase() === 'BUY';
            const pnl = tr.pnl ?? tr.profit_loss;
            return (
              <Box key={i} sx={{
                display: 'grid', gridTemplateColumns: '44px 90px 1fr auto',
                gap: '8px', px: '12px', py: '8px',
                borderBottom: `1px solid ${T.panel}`,
                fontSize: '11px', fontFamily: T.font,
                background: i % 2 === 0 ? '#060d11' : 'transparent',
                alignItems: 'center',
              }}>
                <Box sx={{ color: isBuy ? T.green : T.red, fontWeight: 700 }}>
                  {isBuy ? 'BUY' : 'SELL'}
                </Box>
                <Box sx={{ color: T.text2 }}>{tr.pair || tr.symbol || '--'}</Box>
                <Box sx={{ color: T.text1 }}>
                  {tr.price ? fmtUSD(tr.price) : '--'}
                </Box>
                {pnl !== undefined && pnl !== 'N/A' && (
                  <Box sx={{ color: Number(pnl) >= 0 ? T.green : T.red, textAlign: 'right' }}>
                    {Number(pnl) >= 0 ? '+' : ''}{fmtUSD(pnl)}
                  </Box>
                )}
              </Box>
            );
          })
        )}
      </Box>
    </Box>
  );
}

// ── Log panel ──────────────────────────────────────────────────────────────────

function LogPanel({ logs }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  const lvColor = lv => {
    const u = (lv || '').toUpperCase();
    if (u === 'ERROR' || u === 'CRITICAL') return T.red;
    if (u === 'WARNING' || u === 'WARN') return T.yellow;
    if (u === 'INFO') return T.green;
    return T.dim;
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1, borderTop: `1px solid ${T.border}` }}>
      <PanelHeader>System Log</PanelHeader>
      <Box sx={{ overflow: 'auto', flex: 1, p: '8px 10px', background: T.status }}>
        {!logs?.length ? (
          <Box sx={{ fontSize: '10px', color: T.vdim, fontFamily: T.font }}>No logs yet</Box>
        ) : (
          logs.slice(-60).map((log, i) => (
            <Box key={i} sx={{ fontSize: '10px', lineHeight: 1.7, fontFamily: T.font }}>
              <Box component="span" sx={{ color: T.vdim }}>[{log.timestamp || '--:--:--'}] </Box>
              <Box component="span" sx={{ color: lvColor(log.level), minWidth: '5ch', display: 'inline-block' }}>
                {(log.level || 'INFO').slice(0, 4).toUpperCase()}
              </Box>
              <Box component="span" sx={{ color: T.text2 }}> {log.message}</Box>
            </Box>
          ))
        )}
        <div ref={endRef} />
      </Box>
    </Box>
  );
}

// ── Portfolio view ─────────────────────────────────────────────────────────────

function PortfolioView({ portfolio, status, trades }) {
  const totalValue = portfolio?.total_value ?? 0;
  const holdings   = portfolio?.holdings ?? [];
  const positions  = portfolio?.positions ?? [];
  const totalPnl   = portfolio?.total_pnl ?? 0;
  const winRate    = portfolio?.win_rate ?? 0;
  const totalTrades = portfolio?.total_trades ?? 0;

  // Summary metrics row
  const metrics = [
    { label: 'Total Value',    value: fmtUSD(totalValue),          color: T.text1 },
    { label: 'Available USDT', value: fmtUSD(portfolio?.balance),  color: T.green },
    { label: 'Total P&L',      value: fmtUSD(totalPnl),            color: totalPnl >= 0 ? T.green : T.red },
    { label: 'Win Rate',       value: `${Number(winRate).toFixed(1)}%`, color: T.text1 },
    { label: 'Total Trades',   value: totalTrades,                  color: T.text1 },
    { label: 'Open Positions', value: positions.length,             color: positions.length > 0 ? T.yellow : T.dim },
  ];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Summary strip */}
      <Box sx={{ display: 'flex', borderBottom: `1px solid ${T.border}`, flexShrink: 0 }}>
        {metrics.map((m, i) => (
          <Box key={m.label} sx={{
            flex: 1, px: '14px', py: '12px',
            borderRight: i < metrics.length - 1 ? `1px solid ${T.border}` : 'none',
          }}>
            <Box sx={{ fontSize: '10px', color: T.dim, letterSpacing: '.08em', textTransform: 'uppercase', mb: '3px', fontFamily: T.font }}>
              {m.label}
            </Box>
            <Box sx={{ fontSize: '16px', fontWeight: 700, color: m.color, fontFamily: T.font }}>
              {m.value}
            </Box>
          </Box>
        ))}
      </Box>

      {/* Body: two columns */}
      <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* Left column */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: `1px solid ${T.border}`, overflow: 'hidden' }}>

          {/* Holdings */}
          <Box sx={{ flex: holdings.length > 0 ? '0 0 auto' : '0 0 auto', maxHeight: '45%', display: 'flex', flexDirection: 'column', borderBottom: `1px solid ${T.border}`, overflow: 'hidden' }}>
            <PanelHeader right={<Box sx={{ color: T.green, display: 'flex', alignItems: 'center', gap: '6px' }}><LiveDot /> LIVE</Box>}>
              Account Holdings
            </PanelHeader>
            <Box sx={{ overflow: 'auto', flex: 1 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: T.font }}>
                <thead>
                  <tr>
                    {['Currency', 'Balance', 'Available', 'Value (USDT)', 'Allocation'].map((col, i) => (
                      <th key={col} style={{
                        padding: '7px 14px', fontSize: '10px', color: T.dim,
                        letterSpacing: '.08em', textTransform: 'uppercase',
                        borderBottom: `1px solid ${T.border}`, background: T.status,
                        textAlign: i > 2 ? 'right' : 'left', position: 'sticky', top: 0,
                      }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {holdings.length === 0 ? (
                    <tr><td colSpan={5} style={{ padding: '20px', textAlign: 'center', color: T.vdim, fontSize: '11px' }}>No holdings data</td></tr>
                  ) : holdings.map((h, i) => {
                    const pct = (totalValue > 0 && h.usdt_value != null) ? (h.usdt_value / totalValue * 100) : null;
                    const barW = pct ? Math.min(pct, 100) : 0;
                    return (
                      <tr key={h.currency} style={{ background: i % 2 === 0 ? T.status : 'transparent' }}>
                        <td style={{ padding: '9px 14px', fontWeight: 700, color: T.text1, fontSize: '13px' }}>
                          {h.currency}
                        </td>
                        <td style={{ padding: '9px 14px', color: T.text2, fontSize: '11px' }}>
                          {fmtNum(h.amount)}
                        </td>
                        <td style={{ padding: '9px 14px', color: h.available < h.amount ? T.yellow : T.dim, fontSize: '11px' }}>
                          {fmtNum(h.available)}
                        </td>
                        <td style={{ padding: '9px 14px', color: T.text1, fontSize: '12px', textAlign: 'right', fontWeight: 600 }}>
                          {h.usdt_value != null ? fmtUSD(h.usdt_value) : <span style={{ color: T.vdim }}>--</span>}
                        </td>
                        <td style={{ padding: '9px 14px', textAlign: 'right' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px' }}>
                            <Box sx={{ height: '3px', width: `${barW}px`, maxWidth: '80px', background: T.green, opacity: 0.6 }} />
                            <Box sx={{ fontSize: '10px', color: T.dim, minWidth: '40px', textAlign: 'right' }}>
                              {pct != null ? `${pct.toFixed(1)}%` : '--'}
                            </Box>
                          </Box>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Box>
          </Box>

          {/* Open positions */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <PanelHeader right={<Box sx={{ fontSize: '10px', color: T.dim }}>{positions.length} open</Box>}>
              Open Positions
            </PanelHeader>
            <Box sx={{ overflow: 'auto', flex: 1 }}>
              {positions.length === 0 ? (
                <Box sx={{ p: '18px 14px', fontSize: '11px', color: T.vdim, fontFamily: T.font }}>
                  No open positions — waiting for trade signals
                </Box>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: T.font }}>
                  <thead>
                    <tr>
                      {['Symbol', 'Qty', 'Avg Price', 'Current', 'Value', 'Unr. P&L', 'P&L %'].map((col, i) => (
                        <th key={col} style={{
                          padding: '7px 14px', fontSize: '10px', color: T.dim,
                          letterSpacing: '.08em', textTransform: 'uppercase',
                          borderBottom: `1px solid ${T.border}`, background: T.status,
                          textAlign: i > 1 ? 'right' : 'left', position: 'sticky', top: 0,
                        }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((p, i) => {
                      const pnl = p.unrealized_pnl ?? 0;
                      const pnlPct = p.unrealized_pnl_percent ?? 0;
                      return (
                        <tr key={p.symbol || i} style={{ background: i % 2 === 0 ? T.status : 'transparent' }}>
                          <td style={{ padding: '9px 14px', color: T.text1, fontWeight: 700, fontSize: '12px' }}>{p.symbol}</td>
                          <td style={{ padding: '9px 14px', color: T.text2, fontSize: '11px' }}>{fmtNum(p.quantity, 8)}</td>
                          <td style={{ padding: '9px 14px', color: T.dim, fontSize: '11px', textAlign: 'right' }}>{fmtUSD(p.average_price)}</td>
                          <td style={{ padding: '9px 14px', color: T.text1, fontSize: '11px', textAlign: 'right' }}>{fmtUSD(p.current_price)}</td>
                          <td style={{ padding: '9px 14px', color: T.text1, fontSize: '11px', textAlign: 'right' }}>{fmtUSD(p.market_value)}</td>
                          <td style={{ padding: '9px 14px', textAlign: 'right', fontSize: '11px', fontWeight: 600, color: pnl >= 0 ? T.green : T.red }}>
                            {pnl >= 0 ? '+' : ''}{fmtUSD(pnl)}
                          </td>
                          <td style={{ padding: '9px 14px', textAlign: 'right', fontSize: '11px', color: pnlPct >= 0 ? T.green : T.red }}>
                            {fmtPct(pnlPct)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </Box>
          </Box>
        </Box>

        {/* Right column: recent trades */}
        <Box sx={{ width: '340px', flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <PanelHeader right={<Box sx={{ color: T.dim, fontSize: '10px' }}>{trades?.length ?? 0} total</Box>}>
            Trade History
          </PanelHeader>
          <Box sx={{ overflow: 'auto', flex: 1 }}>
            {!trades?.length ? (
              <Box sx={{ p: '18px 14px', fontSize: '11px', color: T.vdim, fontFamily: T.font }}>No trades recorded</Box>
            ) : trades.map((tr, i) => {
              const isBuy = (tr.side || tr.action || '').toUpperCase() === 'BUY';
              const pnl = tr.pnl ?? tr.profit_loss;
              const ts = tr.timestamp || tr.entry_time || tr.created_at;
              return (
                <Box key={i} sx={{
                  px: '14px', py: '10px', borderBottom: `1px solid ${T.panel}`,
                  background: i % 2 === 0 ? '#060d11' : 'transparent',
                  fontFamily: T.font,
                }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: '3px' }}>
                    <Box sx={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                      <Box sx={{ fontSize: '11px', fontWeight: 700, color: isBuy ? T.green : T.red }}>
                        {isBuy ? 'BUY' : 'SELL'}
                      </Box>
                      <Box sx={{ fontSize: '12px', color: T.text1, fontWeight: 600 }}>
                        {tr.pair || tr.symbol || '--'}
                      </Box>
                    </Box>
                    {pnl != null && pnl !== 'N/A' && (
                      <Box sx={{ fontSize: '11px', fontWeight: 700, color: Number(pnl) >= 0 ? T.green : T.red }}>
                        {Number(pnl) >= 0 ? '+' : ''}{fmtUSD(pnl)}
                      </Box>
                    )}
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: T.dim }}>
                    <span>{tr.price ? fmtUSD(tr.price) : '--'}</span>
                    <span>{ts ? new Date(ts).toLocaleString() : '--'}</span>
                  </Box>
                  {tr.confidence && (
                    <Box sx={{ fontSize: '10px', color: T.vdim, mt: '2px' }}>
                      Confidence: {(Number(tr.confidence) * 100).toFixed(0)}%
                    </Box>
                  )}
                </Box>
              );
            })}
          </Box>
        </Box>

      </Box>
    </Box>
  );
}

// ── Overview view ──────────────────────────────────────────────────────────────

function OverviewView({ portfolio, status, trades, logs }) {
  const tradingPairs = status?.trading_pairs || ['BTC-USDT', 'ETH-USDT'];
  return (
    <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: `1px solid ${T.border}`, overflow: 'hidden' }}>
        <Box sx={{ flex: '0 0 auto', maxHeight: '35%', display: 'flex', flexDirection: 'column', borderBottom: `1px solid ${T.border}`, overflow: 'hidden' }}>
          <HoldingsPanel holdings={portfolio?.holdings} totalValue={portfolio?.total_value} />
        </Box>
        <Box sx={{ flex: '0 0 auto', maxHeight: '28%', display: 'flex', flexDirection: 'column', borderBottom: `1px solid ${T.border}`, overflow: 'hidden' }}>
          <PositionsPanel positions={portfolio?.positions} />
        </Box>
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <MarketPanel pairs={tradingPairs} />
        </Box>
      </Box>
      <Box sx={{ width: '320px', flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <TradeFeed trades={trades} />
        <LogPanel logs={logs} />
      </Box>
    </Box>
  );
}

// ── Dashboard ──────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const { active } = useSubNav();
  const [status,    setStatus]    = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [trades,    setTrades]    = useState([]);
  const [logs,      setLogs]      = useState([]);

  const api = axios.create({ baseURL: API_BASE_URL, timeout: 10000 });

  const fetchAll = async () => {
    try {
      const [st, po, hi, lg] = await Promise.allSettled([
        api.get('/api/status'),
        api.get('/api/portfolio'),
        api.get('/api/trades', { params: { limit: 10 } }),
        api.get('/api/logs', { params: { limit: 60 } }),
      ]);
      if (st.status === 'fulfilled' && st.value.data?.success) setStatus(st.value.data.status);
      if (po.status === 'fulfilled' && po.value.data?.success) setPortfolio(po.value.data.portfolio);
      if (hi.status === 'fulfilled' && hi.value.data?.success) setTrades(hi.value.data.trades || []);
      if (lg.status === 'fulfilled') setLogs(lg.value.data?.logs || []);
    } catch {}
  };

  useEffect(() => {
    fetchAll(); // eslint-disable-line react-hooks/exhaustive-deps
    const id = setInterval(fetchAll, 30000);
    return () => clearInterval(id);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', fontFamily: T.font }}>
      <KpiRow portfolio={portfolio} status={status} />
      {active === 'Portfolio'
        ? <PortfolioView portfolio={portfolio} status={status} trades={trades} />
        : <OverviewView portfolio={portfolio} status={status} trades={trades} logs={logs} />
      }
    </Box>
  );
}
