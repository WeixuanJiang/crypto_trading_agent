import { createTheme } from '@mui/material/styles';

const terminalFont = "'Fira Code', 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace";

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary:    { main: '#a855f7', light: '#c084fc', dark: '#7e22ce', contrastText: '#fff' },
    secondary:  { main: '#3b82f6', light: '#60a5fa', dark: '#2563eb', contrastText: '#fff' },
    success:    { main: '#22c55e', light: '#4ade80', dark: '#16a34a' },
    error:      { main: '#ef4444', light: '#f87171', dark: '#dc2626' },
    warning:    { main: '#eab308', light: '#facc15', dark: '#a16207' },
    info:       { main: '#3b82f6', light: '#60a5fa', dark: '#2563eb' },
    background: { default: '#050914', paper: '#111827' },
    text:       { primary: '#f1f5f9', secondary: '#cbd5e1', disabled: '#64748b' },
    divider:    '#1e293b',
  },
  typography: {
    fontFamily: terminalFont,
    fontSize: 12,
    h1: { fontFamily: terminalFont, fontWeight: 700 },
    h2: { fontFamily: terminalFont, fontWeight: 700 },
    h3: { fontFamily: terminalFont, fontWeight: 600 },
    h4: { fontFamily: terminalFont, fontWeight: 600 },
    h5: { fontFamily: terminalFont, fontWeight: 600 },
    h6: { fontFamily: terminalFont, fontWeight: 600 },
    body1: { fontFamily: terminalFont, fontSize: '0.8rem' },
    body2: { fontFamily: terminalFont, fontSize: '0.75rem' },
    caption: { fontFamily: terminalFont, fontSize: '0.7rem' },
    button: { fontFamily: terminalFont, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.08em' },
    overline: { fontFamily: terminalFont, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.1em' },
  },
  shape: { borderRadius: 2 },
  shadows: Array(25).fill('none'),
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: { fontFamily: terminalFont, backgroundColor: '#050914', color: '#cbd5e1' },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 2,
          border: '1px solid #1e293b',
          background: '#111827',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 2,
          border: '1px solid #1e293b',
          background: '#111827',
          boxShadow: 'none',
          '&:hover': { border: '1px solid #a855f744' },
        },
      },
    },
    MuiCardContent: {
      styleOverrides: { root: { padding: '14px 16px', '&:last-child': { paddingBottom: '14px' } } },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 2, fontFamily: terminalFont, fontWeight: 600,
          textTransform: 'uppercase', letterSpacing: '.08em',
          fontSize: '11px', boxShadow: 'none',
          '&:hover': { boxShadow: 'none' },
        },
        contained: { border: '1px solid transparent' },
        outlined: { borderWidth: '1px' },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { borderRadius: 2, fontFamily: terminalFont, fontSize: '10px', fontWeight: 600, letterSpacing: '.06em' },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid #1e293b', fontFamily: terminalFont,
          fontSize: '11px', color: '#cbd5e1', padding: '8px 12px',
        },
        head: {
          fontWeight: 600, fontSize: '10px', textTransform: 'uppercase',
          letterSpacing: '.08em', color: '#64748b', background: '#020617',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: { '&:hover': { background: '#0b1121' } },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 2, fontFamily: terminalFont, fontSize: '12px',
            background: '#020617',
            '& fieldset': { borderColor: '#1e293b' },
            '&:hover fieldset': { borderColor: '#334155' },
            '&.Mui-focused fieldset': { borderColor: '#a855f7', borderWidth: 1 },
          },
          '& .MuiInputLabel-root': {
            fontFamily: terminalFont, fontSize: '11px', color: '#64748b',
            '&.Mui-focused': { color: '#a855f7' },
          },
          '& .MuiInputBase-input': { color: '#f1f5f9' },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: { fontFamily: terminalFont, fontSize: '12px' },
        icon: { color: '#64748b' },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          fontFamily: terminalFont, fontSize: '12px', color: '#cbd5e1',
          '&:hover': { background: '#0b1121' },
          '&.Mui-selected': { background: 'rgba(168,85,247,0.1)', color: '#a855f7' },
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        switchBase: { '&.Mui-checked': { color: '#a855f7' } },
        track: { backgroundColor: '#1e293b' },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: { borderRadius: 2, fontFamily: terminalFont, fontSize: '11px', border: '1px solid' },
        standardSuccess: { background: '#071810', borderColor: '#22c55e44', color: '#22c55e' },
        standardError: { background: '#1a0606', borderColor: '#ef444444', color: '#ef4444' },
        standardWarning: { background: '#1a1500', borderColor: '#eab30844', color: '#eab308' },
        standardInfo: { background: '#06101e', borderColor: '#3b82f644', color: '#3b82f6' },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          background: '#111827', border: '1px solid #1e293b',
          borderRadius: 2, fontFamily: terminalFont, fontSize: '11px', color: '#cbd5e1',
        },
      },
    },
    MuiDivider: {
      styleOverrides: { root: { borderColor: '#1e293b' } },
    },
    MuiAccordion: {
      styleOverrides: {
        root: { background: '#111827', border: '1px solid #1e293b', borderRadius: '2px !important', boxShadow: 'none' },
      },
    },
    MuiAccordionSummary: {
      styleOverrides: {
        root: { fontFamily: terminalFont, fontSize: '11px', color: '#64748b', '&:hover': { background: '#0b1121' } },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: { backgroundColor: '#1e293b', borderRadius: 0 },
        bar: { backgroundColor: '#a855f7' },
      },
    },
    MuiDrawer: {
      styleOverrides: { paper: { background: '#0b1121', border: 'none' } },
    },
    MuiAppBar: {
      styleOverrides: { root: { background: '#111827', boxShadow: 'none', borderBottom: '1px solid #1e293b' } },
    },
    MuiCircularProgress: {
      styleOverrides: { root: { color: '#a855f7' } },
    },
    MuiFormControlLabel: {
      styleOverrides: {
        label: { fontFamily: terminalFont, fontSize: '11px', color: '#cbd5e1' },
      },
    },
  },
});

export default darkTheme;
