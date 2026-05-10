import { createTheme } from '@mui/material/styles';

const terminalFont = "'Fira Code', 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace";

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary:    { main: '#00d4aa', light: '#4dffda', dark: '#009e7e', contrastText: '#000' },
    secondary:  { main: '#4db8ff', light: '#80ccff', dark: '#2a8fcc', contrastText: '#000' },
    success:    { main: '#00d4aa', light: '#4dffda', dark: '#009e7e' },
    error:      { main: '#ff4d6a', light: '#ff8096', dark: '#cc2244' },
    warning:    { main: '#f5c842', light: '#ffd966', dark: '#c49a00' },
    info:       { main: '#4db8ff', light: '#80ccff', dark: '#2a8fcc' },
    background: { default: '#080c0f', paper: '#0e1519' },
    text:       { primary: '#e0e8f0', secondary: '#c0c8d0', disabled: '#4a6070' },
    divider:    '#1c2830',
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
        body: { fontFamily: terminalFont, backgroundColor: '#080c0f', color: '#c0c8d0' },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 2,
          border: '1px solid #1c2830',
          background: '#0e1519',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 2,
          border: '1px solid #1c2830',
          background: '#0e1519',
          boxShadow: 'none',
          '&:hover': { border: '1px solid #00d4aa44' },
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
          borderBottom: '1px solid #1c2830', fontFamily: terminalFont,
          fontSize: '11px', color: '#c0c8d0', padding: '8px 12px',
        },
        head: {
          fontWeight: 600, fontSize: '10px', textTransform: 'uppercase',
          letterSpacing: '.08em', color: '#4a6070', background: '#060a0d',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: { '&:hover': { background: '#0d1a22' } },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 2, fontFamily: terminalFont, fontSize: '12px',
            background: '#060a0d',
            '& fieldset': { borderColor: '#1c2830' },
            '&:hover fieldset': { borderColor: '#2a4050' },
            '&.Mui-focused fieldset': { borderColor: '#00d4aa', borderWidth: 1 },
          },
          '& .MuiInputLabel-root': {
            fontFamily: terminalFont, fontSize: '11px', color: '#4a6070',
            '&.Mui-focused': { color: '#00d4aa' },
          },
          '& .MuiInputBase-input': { color: '#e0e8f0' },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: { fontFamily: terminalFont, fontSize: '12px' },
        icon: { color: '#4a6070' },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          fontFamily: terminalFont, fontSize: '12px', color: '#c0c8d0',
          '&:hover': { background: '#0d1a22' },
          '&.Mui-selected': { background: '#001e28', color: '#00d4aa' },
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        switchBase: { '&.Mui-checked': { color: '#00d4aa' } },
        track: { backgroundColor: '#1c2830' },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: { borderRadius: 2, fontFamily: terminalFont, fontSize: '11px', border: '1px solid' },
        standardSuccess: { background: '#001a12', borderColor: '#00d4aa44', color: '#00d4aa' },
        standardError: { background: '#1a0008', borderColor: '#ff4d6a44', color: '#ff4d6a' },
        standardWarning: { background: '#1a1400', borderColor: '#f5c84244', color: '#f5c842' },
        standardInfo: { background: '#001222', borderColor: '#4db8ff44', color: '#4db8ff' },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          background: '#0e1519', border: '1px solid #1c2830',
          borderRadius: 2, fontFamily: terminalFont, fontSize: '11px', color: '#c0c8d0',
        },
      },
    },
    MuiDivider: {
      styleOverrides: { root: { borderColor: '#1c2830' } },
    },
    MuiAccordion: {
      styleOverrides: {
        root: { background: '#0e1519', border: '1px solid #1c2830', borderRadius: '2px !important', boxShadow: 'none' },
      },
    },
    MuiAccordionSummary: {
      styleOverrides: {
        root: { fontFamily: terminalFont, fontSize: '11px', color: '#4a6070', '&:hover': { background: '#0d1a22' } },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: { backgroundColor: '#1c2830', borderRadius: 0 },
        bar: { backgroundColor: '#00d4aa' },
      },
    },
    MuiDrawer: {
      styleOverrides: { paper: { background: '#0a0f13', border: 'none' } },
    },
    MuiAppBar: {
      styleOverrides: { root: { background: '#0e1519', boxShadow: 'none', borderBottom: '1px solid #1c2830' } },
    },
    MuiCircularProgress: {
      styleOverrides: { root: { color: '#00d4aa' } },
    },
    MuiFormControlLabel: {
      styleOverrides: {
        label: { fontFamily: terminalFont, fontSize: '11px', color: '#c0c8d0' },
      },
    },
  },
});

export default darkTheme;
