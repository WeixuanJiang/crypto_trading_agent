// Shared terminal design tokens
export const T = {
  bg:      '#080c0f',
  sidebar: '#0a0f13',
  panel:   '#0e1519',
  status:  '#060a0d',
  border:  '#1c2830',
  hover:   '#0d1a22',
  green:   '#00d4aa',
  red:     '#ff4d6a',
  yellow:  '#f5c842',
  blue:    '#4db8ff',
  text1:   '#e0e8f0',
  text2:   '#c0c8d0',
  dim:     '#4a6070',
  vdim:    '#2a4050',
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
