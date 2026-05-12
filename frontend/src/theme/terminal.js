// Shared terminal design tokens
export const T = {
  bg:      '#050914',
  sidebar: '#0b1121',
  panel:   '#111827',
  status:  '#020617',
  border:  '#1e293b',
  hover:   '#0b1121',
  green:   '#a855f7',
  red:     '#ef4444',
  yellow:  '#eab308',
  blue:    '#3b82f6',
  text1:   '#f1f5f9',
  text2:   '#cbd5e1',
  dim:     '#64748b',
  vdim:    '#475569',
  font:    "'Fira Code', 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace",
};

// Common sx helpers
export const panelSx = {
  background: T.panel,
  border: `1px solid ${T.border}`,
  borderRadius: 0,
  fontFamily: T.font,
};

export const labelSx = {
  fontSize: '10px',
  color: T.dim,
  letterSpacing: '.1em',
  textTransform: 'uppercase',
  fontFamily: T.font,
};

export const valueSx = {
  fontFamily: T.font,
  color: T.text1,
};
