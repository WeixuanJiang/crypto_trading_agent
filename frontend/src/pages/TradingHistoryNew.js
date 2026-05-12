import React, { useState, useMemo } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { T } from '../theme/terminal';
import { useTrades } from '../hooks/useTradingData';
import { formatCurrency } from '../utils/formatters';

function TermInput({ value, onChange, placeholder }) {
  return (
    <Box
      component="input"
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      sx={{
        background: T.status, border: `1px solid ${T.border}`, color: T.text1,
        fontFamily: T.font, fontSize: '11px', px: '10px', py: '5px',
        outline: 'none', width: '180px',
        '&:focus': { borderColor: T.green },
        '&::placeholder': { color: T.vdim },
      }}
    />
  );
}

function TermSelect({ value, onChange, options }) {
  return (
    <Box
      component="select"
      value={value}
      onChange={e => onChange(e.target.value)}
      sx={{
        background: T.status, border: `1px solid ${T.border}`, color: T.text1,
        fontFamily: T.font, fontSize: '11px', px: '8px', py: '5px', outline: 'none',
        cursor: 'pointer', '&:focus': { borderColor: T.green },
      }}
    >
      {options.map(o => (
        <option key={o.value} value={o.value} style={{ background: T.panel }}>
          {o.label}
        </option>
      ))}
    </Box>
  );
}

const COLS = [
  { key: 'timestamp', label: 'Timestamp',  width: '160px' },
  { key: 'pair',      label: 'Pair',        width: '100px' },
  { key: 'side',      label: 'Side',        width: '70px'  },
  { key: 'price',     label: 'Price',       width: '110px', align: 'right' },
  { key: 'quantity',  label: 'Qty',         width: '100px', align: 'right' },
  { key: 'pnl',       label: 'P&L',         width: '100px', align: 'right' },
  { key: 'confidence',label: 'Confidence',  width: '90px',  align: 'right' },
  { key: 'status',    label: 'Status',      width: '90px'  },
];

const PAGE_SIZE = 25;

export default function TradingHistory() {
  const { data: tradesData, isLoading } = useTrades({ limit: 500 });
  const [search, setSearch] = useState('');
  const [sideFilter, setSideFilter] = useState('ALL');
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState('timestamp');
  const [sortDir, setSortDir] = useState('desc');

  const trades = useMemo(() => tradesData?.trades || [], [tradesData]);

  const filtered = useMemo(() => {
    let rows = [...trades];
    if (search) {
      const q = search.toLowerCase();
      rows = rows.filter(r =>
        (r.pair || '').toLowerCase().includes(q) ||
        (r.symbol || '').toLowerCase().includes(q)
      );
    }
    if (sideFilter !== 'ALL') {
      rows = rows.filter(r => (r.side || r.action || '').toUpperCase() === sideFilter);
    }
    rows.sort((a, b) => {
      let av = a[sortKey], bv = b[sortKey];
      if (typeof av === 'string') av = av.toLowerCase();
      if (typeof bv === 'string') bv = bv.toLowerCase();
      if (av < bv) return sortDir === 'asc' ? -1 : 1;
      if (av > bv) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return rows;
  }, [trades, search, sideFilter, sortKey, sortDir]);

  const pageRows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);

  const handleSort = key => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  // Stats
  const stats = useMemo(() => {
    const wins = trades.filter(t => (t.pnl ?? 0) > 0).length;
    const totalPnl = trades.reduce((s, t) => s + (t.pnl ?? 0), 0);
    return { total: trades.length, wins, winRate: trades.length ? (wins / trades.length * 100).toFixed(1) : 0, totalPnl };
  }, [trades]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', fontFamily: T.font, overflow: 'hidden' }}>

      {/* Stats strip */}
      <Box sx={{
        display: 'flex', borderBottom: `1px solid ${T.border}`,
        background: T.status, flexShrink: 0,
      }}>
        {[
          { label: 'Total Trades', value: stats.total },
          { label: 'Win Rate',     value: `${stats.winRate}%` },
          { label: 'Total P&L',    value: formatCurrency(stats.totalPnl), color: stats.totalPnl >= 0 ? T.green : T.red },
          { label: 'Wins',         value: stats.wins, color: T.green },
          { label: 'Losses',       value: stats.total - stats.wins, color: T.red },
        ].map((s, i, arr) => (
          <Box key={s.label} sx={{
            px: '16px', py: '12px', borderRight: i < arr.length - 1 ? `1px solid ${T.border}` : 'none',
          }}>
            <Box sx={{ fontSize: '10px', color: T.dim, letterSpacing: '.08em', textTransform: 'uppercase', mb: '2px' }}>{s.label}</Box>
            <Box sx={{ fontSize: '18px', fontWeight: 700, color: s.color || T.text1 }}>{s.value}</Box>
          </Box>
        ))}
      </Box>

      {/* Toolbar */}
      <Box sx={{
        display: 'flex', alignItems: 'center', gap: '10px',
        px: '14px', py: '8px', borderBottom: `1px solid ${T.border}`,
        background: T.panel, flexShrink: 0,
      }}>
        <TermInput value={search} onChange={v => { setSearch(v); setPage(0); }} placeholder="Search pair..." />
        <TermSelect
          value={sideFilter}
          onChange={v => { setSideFilter(v); setPage(0); }}
          options={[{ value: 'ALL', label: 'ALL SIDES' }, { value: 'BUY', label: 'BUY' }, { value: 'SELL', label: 'SELL' }]}
        />
        <Box sx={{ marginLeft: 'auto', fontSize: '11px', color: T.dim }}>
          {filtered.length} rows · page {page + 1}/{totalPages || 1}
        </Box>
      </Box>

      {/* Table */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><CircularProgress size={20} /></Box>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: T.font }}>
            <thead>
              <tr>
                {COLS.map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    style={{
                      width: col.width, padding: '8px 12px', textAlign: col.align || 'left',
                      fontSize: '10px', color: sortKey === col.key ? T.green : T.dim,
                      letterSpacing: '.08em', textTransform: 'uppercase',
                      borderBottom: `1px solid ${T.border}`, background: T.status,
                      cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap',
                      position: 'sticky', top: 0,
                    }}
                  >
                    {col.label}
                    {sortKey === col.key && (sortDir === 'asc' ? ' ▲' : ' ▼')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pageRows.length === 0 ? (
                <tr>
                  <td colSpan={COLS.length} style={{ padding: '24px', textAlign: 'center', color: T.vdim, fontSize: '12px' }}>
                    No trades found
                  </td>
                </tr>
              ) : (
                pageRows.map((tr, i) => {
                  const isBuy = (tr.side || tr.action || '').toUpperCase() === 'BUY';
                  const pnl = tr.pnl ?? tr.profit_loss;
                  const pair = tr.pair || tr.symbol || '--';
                  const ts = tr.timestamp || tr.entry_time || tr.created_at;
                  const status = tr.status || (tr.exit_price ? 'CLOSED' : 'OPEN');
                  return (
                    <tr key={i} style={{ background: i % 2 === 0 ? T.status : 'transparent' }}>
                      <td style={{ padding: '8px 12px', color: T.dim, fontSize: '11px', whiteSpace: 'nowrap' }}>
                        {ts ? (() => { const d = new Date(ts); return isNaN(d) ? ts : d.toLocaleString(); })() : '--'}
                      </td>
                      <td style={{ padding: '8px 12px', color: T.text1, fontWeight: 600 }}>{pair}</td>
                      <td style={{ padding: '8px 12px', color: isBuy ? T.green : T.red, fontWeight: 700 }}>
                        {isBuy ? 'BUY' : 'SELL'}
                      </td>
                      <td style={{ padding: '8px 12px', color: T.text1, textAlign: 'right' }}>
                        {tr.price ? `$${Number(tr.price).toLocaleString()}` : '--'}
                      </td>
                      <td style={{ padding: '8px 12px', color: T.text2, textAlign: 'right' }}>
                        {tr.quantity ?? tr.amount ?? '--'}
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: pnl == null ? T.dim : pnl >= 0 ? T.green : T.red, fontWeight: 600 }}>
                        {pnl == null ? '--' : `${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}`}
                      </td>
                      <td style={{ padding: '8px 12px', color: T.text2, textAlign: 'right' }}>
                        {tr.confidence != null ? `${(tr.confidence * 100).toFixed(0)}%` : '--'}
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <Box component="span" sx={{
                          px: '6px', py: '2px', fontSize: '10px', fontWeight: 600,
                          border: '1px solid',
                          color: status === 'CLOSED' ? T.text2 : T.green,
                          borderColor: status === 'CLOSED' ? T.border : T.green,
                          background: status === 'CLOSED' ? 'transparent' : 'rgba(168,85,247,0.1)',
                        }}>
                          {status.toUpperCase()}
                        </Box>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        )}
      </Box>

      {/* Pagination */}
      <Box sx={{
        display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px',
        px: '14px', py: '8px', borderTop: `1px solid ${T.border}`,
        background: T.status, flexShrink: 0, fontSize: '11px', fontFamily: T.font,
      }}>
        {[
          { label: '« First', fn: () => setPage(0),            disabled: page === 0 },
          { label: '‹ Prev',  fn: () => setPage(p => p - 1),  disabled: page === 0 },
          { label: 'Next ›',  fn: () => setPage(p => p + 1),  disabled: page >= totalPages - 1 },
          { label: 'Last »',  fn: () => setPage(totalPages - 1), disabled: page >= totalPages - 1 },
        ].map(btn => (
          <Box
            key={btn.label}
            onClick={btn.disabled ? undefined : btn.fn}
            sx={{
              px: '10px', py: '4px', cursor: btn.disabled ? 'not-allowed' : 'pointer',
              border: `1px solid ${T.border}`, color: btn.disabled ? T.vdim : T.text2,
              '&:hover': btn.disabled ? {} : { borderColor: T.green, color: T.green },
            }}
          >
            {btn.label}
          </Box>
        ))}
      </Box>
    </Box>
  );
}
