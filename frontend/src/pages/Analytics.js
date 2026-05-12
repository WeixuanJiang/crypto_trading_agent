import React, { useMemo } from 'react';
import { Box, CircularProgress } from '@mui/material';
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip as RechartTooltip,
  ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar,
} from 'recharts';
import { T } from '../theme/terminal';
import { useTrades, usePerformance } from '../hooks/useTradingData';
import { formatCurrency, formatPercent, formatDate } from '../utils/formatters';
import { calculateTradeStats } from '../utils/calculations';

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

function MetricRow({ label, value, color }) {
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

const rechartStyle = {
  axis:    { stroke: T.border, tick: { fill: T.dim, fontSize: 10, fontFamily: T.font } },
  grid:    { stroke: T.border, strokeDasharray: '2 4' },
  tooltip: { contentStyle: { background: T.panel, border: `1px solid ${T.border}`, borderRadius: 2, fontFamily: T.font, fontSize: 11, color: T.text2 }, labelStyle: { color: T.dim } },
};

export default function Analytics() {
  const { data: tradesData, isLoading: tradesLoading } = useTrades({ limit: 500 });
  const { data: perfData,   isLoading: perfLoading }   = usePerformance();

  const trades = tradesData?.trades || [];
  const stats  = useMemo(() => calculateTradeStats(trades), [trades]);

  // P&L by day
  const dailyPnl = useMemo(() => {
    const map = {};
    trades.forEach(t => {
      const d = (t.timestamp || t.entry_time || '').slice(0, 10);
      if (!d) return;
      map[d] = (map[d] || 0) + (t.pnl ?? 0);
    });
    return Object.entries(map).sort(([a], [b]) => a.localeCompare(b)).slice(-14).map(([date, pnl]) => ({
      date: date.slice(5), pnl: parseFloat(pnl.toFixed(2)),
    }));
  }, [trades]);

  // Trades per pair
  const pairVolume = useMemo(() => {
    const map = {};
    trades.forEach(t => {
      const p = t.pair || t.symbol || 'Unknown';
      if (!map[p]) map[p] = { pair: p, count: 0, pnl: 0 };
      map[p].count += 1;
      map[p].pnl   += t.pnl ?? 0;
    });
    return Object.values(map).sort((a, b) => b.count - a.count);
  }, [trades]);

  // Radar chart data
  const radarData = [
    { subject: 'Win Rate',      A: (stats?.winRate || 0) * 100 },
    { subject: 'Profit Factor', A: Math.min((stats?.profitFactor || 0) * 20, 100) },
    { subject: 'Avg Win',       A: Math.min((stats?.avgWin || 0) / 10, 100) },
    { subject: 'Consistency',   A: 60 },
    { subject: 'Risk/Reward',   A: Math.min((stats?.riskRewardRatio || 0) * 30, 100) },
  ];

  if (tradesLoading || perfLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress size={20} sx={{ color: T.green }} />
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', fontFamily: T.font, overflow: 'auto' }}>

      {/* KPI strip */}
      <Box sx={{ display: 'flex', borderBottom: `1px solid ${T.border}`, background: T.status, flexShrink: 0 }}>
        {[
          { label: 'Total P&L',      value: formatCurrency(stats?.totalPnl),     color: (stats?.totalPnl ?? 0) >= 0 ? T.green : T.red },
          { label: 'Win Rate',       value: formatPercent(stats?.winRate),        color: T.green },
          { label: 'Total Trades',   value: stats?.totalTrades ?? '--' },
          { label: 'Profit Factor',  value: stats?.profitFactor?.toFixed(2) ?? '--' },
          { label: 'Max Drawdown',   value: formatPercent(stats?.maxDrawdown),    color: T.red },
          { label: 'Sharpe Ratio',   value: stats?.sharpeRatio?.toFixed(2) ?? '--' },
        ].map((k, i, arr) => (
          <Box key={k.label} sx={{
            flex: 1, px: '16px', py: '12px',
            borderRight: i < arr.length - 1 ? `1px solid ${T.border}` : 'none',
          }}>
            <Box sx={{ fontSize: '10px', color: T.dim, textTransform: 'uppercase', letterSpacing: '.08em', mb: '2px' }}>{k.label}</Box>
            <Box sx={{ fontSize: '18px', fontWeight: 700, color: k.color || T.text1 }}>{k.value}</Box>
          </Box>
        ))}
      </Box>

      {/* Charts row */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', borderBottom: `1px solid ${T.border}` }}>

        {/* Daily P&L bar chart */}
        <Box sx={{ borderRight: `1px solid ${T.border}` }}>
          <PanelHeader>Daily P&L — Last 14 Days</PanelHeader>
          <Box sx={{ p: '16px', height: '220px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dailyPnl} barSize={12}>
                <CartesianGrid {...rechartStyle.grid} />
                <XAxis dataKey="date" {...rechartStyle.axis} />
                <YAxis {...rechartStyle.axis} />
                <RechartTooltip {...rechartStyle.tooltip} formatter={v => [`$${v.toFixed(2)}`, 'P&L']} />
                <Bar dataKey="pnl" fill={T.green}
                  label={false}
                  cell={entry => <rect fill={entry.pnl >= 0 ? T.green : T.red} />}
                />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Box>

        {/* Strategy radar */}
        <Box>
          <PanelHeader>Strategy Performance Profile</PanelHeader>
          <Box sx={{ p: '16px', height: '220px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke={T.border} />
                <PolarAngleAxis dataKey="subject" tick={{ fill: T.dim, fontSize: 10, fontFamily: T.font }} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name="Score" dataKey="A" stroke={T.green} fill={T.green} fillOpacity={0.15} />
              </RadarChart>
            </ResponsiveContainer>
          </Box>
        </Box>
      </Box>

      {/* Bottom row */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 280px', flex: 1 }}>

        {/* Pair breakdown table */}
        <Box sx={{ borderRight: `1px solid ${T.border}`, overflow: 'auto' }}>
          <PanelHeader>Volume by Pair</PanelHeader>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: T.font }}>
            <thead>
              <tr>
                {['Pair', 'Trades', 'Total P&L', 'Avg P&L'].map(h => (
                  <th key={h} style={{
                    padding: '8px 12px', textAlign: h === 'Pair' ? 'left' : 'right',
                    fontSize: '10px', color: T.dim, textTransform: 'uppercase', letterSpacing: '.08em',
                    borderBottom: `1px solid ${T.border}`, background: T.status,
                    position: 'sticky', top: 0,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {pairVolume.length === 0 ? (
                <tr><td colSpan={4} style={{ padding: '20px', textAlign: 'center', color: T.vdim, fontSize: '11px' }}>No data</td></tr>
              ) : pairVolume.map((row, i) => (
                <tr key={row.pair} style={{ background: i % 2 === 0 ? T.status : 'transparent' }}>
                  <td style={{ padding: '8px 12px', color: T.text1, fontWeight: 600, fontSize: '11px' }}>{row.pair}</td>
                  <td style={{ padding: '8px 12px', color: T.text2, textAlign: 'right', fontSize: '11px' }}>{row.count}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', fontSize: '11px', color: row.pnl >= 0 ? T.green : T.red }}>
                    {row.pnl >= 0 ? '+' : ''}${row.pnl.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'right', fontSize: '11px', color: T.text2 }}>
                    ${(row.pnl / row.count).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Box>

        {/* Metrics panel */}
        <Box sx={{ overflow: 'auto' }}>
          <PanelHeader>Risk Metrics</PanelHeader>
          <Box sx={{ px: '14px', pb: '14px' }}>
            <MetricRow label="Total Trades"    value={stats?.totalTrades} />
            <MetricRow label="Winning Trades"  value={stats?.winningTrades} color={T.green} />
            <MetricRow label="Losing Trades"   value={stats?.losingTrades}  color={T.red} />
            <MetricRow label="Win Rate"        value={formatPercent(stats?.winRate)} color={T.green} />
            <MetricRow label="Avg Win"         value={formatCurrency(stats?.avgWin)}  color={T.green} />
            <MetricRow label="Avg Loss"        value={formatCurrency(stats?.avgLoss)} color={T.red} />
            <MetricRow label="Max Drawdown"    value={formatPercent(stats?.maxDrawdown)} color={T.red} />
            <MetricRow label="Profit Factor"   value={stats?.profitFactor?.toFixed(2)} />
            <MetricRow label="Sharpe Ratio"    value={stats?.sharpeRatio?.toFixed(2)} />
            <MetricRow label="Risk/Reward"     value={stats?.riskRewardRatio?.toFixed(2)} />
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
