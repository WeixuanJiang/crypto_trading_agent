import { createTheme } from '@mui/material/styles';

const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0066cc',
      light: '#3d8ddb',
      dark: '#004c99',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#7c4dff',
      light: '#b47cff',
      dark: '#5624cc',
      contrastText: '#ffffff',
    },
    success: {
      main: '#00c853',
      light: '#5df073',
      dark: '#009624',
    },
    error: {
      main: '#d32f2f',
      light: '#ef5350',
      dark: '#c62828',
    },
    warning: {
      main: '#ff9800',
      light: '#ffb74d',
      dark: '#f57c00',
    },
    info: {
      main: '#0288d1',
      light: '#03a9f4',
      dark: '#01579b',
    },
    background: {
      default: '#f5f7fa',
      paper: '#ffffff',
      gradient: 'linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%)',
    },
    text: {
      primary: '#1a202c',
      secondary: '#4a5568',
      disabled: '#a0aec0',
    },
    divider: 'rgba(0, 0, 0, 0.08)',
    chart: {
      bull: '#00c853',
      bear: '#d32f2f',
      volume: '#0288d1',
      grid: '#e2e8f0',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
    subtitle1: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.75,
    },
    subtitle2: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.57,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.43,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.66,
    },
    overline: {
      fontSize: '0.75rem',
      fontWeight: 600,
      textTransform: 'uppercase',
      lineHeight: 2.5,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.08)',
    '0px 8px 16px rgba(0, 0, 0, 0.12)',
    '0px 12px 24px rgba(0, 0, 0, 0.15)',
    '0px 16px 32px rgba(0, 0, 0, 0.18)',
    ...Array(19).fill('0px 20px 40px rgba(0, 0, 0, 0.2)'),
  ],
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 12,
          border: '1px solid rgba(0, 0, 0, 0.06)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        elevation1: {
          boxShadow: '0px 4px 16px rgba(0, 0, 0, 0.08)',
        },
        elevation2: {
          boxShadow: '0px 8px 24px rgba(0, 0, 0, 0.12)',
        },
        elevation3: {
          boxShadow: '0px 12px 32px rgba(0, 0, 0, 0.15)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          border: '1px solid rgba(0, 0, 0, 0.06)',
          background: '#ffffff',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            border: '1px solid rgba(0, 102, 204, 0.3)',
            boxShadow: '0px 12px 32px rgba(0, 102, 204, 0.15)',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 24px',
          fontSize: '0.875rem',
          fontWeight: 600,
          textTransform: 'none',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0px 8px 24px rgba(0, 102, 204, 0.3)',
            transform: 'translateY(-2px)',
          },
        },
        outlined: {
          border: '2px solid',
          '&:hover': {
            border: '2px solid',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            backgroundColor: '#ffffff',
            '&:hover': {
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(0, 102, 204, 0.5)',
              },
            },
            '&.Mui-focused': {
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: '#0066cc',
                borderWidth: 2,
              },
            },
          },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid rgba(0, 0, 0, 0.06)',
        },
        head: {
          fontWeight: 600,
          backgroundColor: 'rgba(0, 102, 204, 0.05)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          background: '#ffffff',
          borderRight: '1px solid rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: '#ffffff',
          color: '#1a202c',
          borderBottom: '1px solid rgba(0, 0, 0, 0.08)',
          boxShadow: '0px 4px 24px rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: 'rgba(26, 32, 44, 0.95)',
          backdropFilter: 'blur(10px)',
          borderRadius: 8,
          fontSize: '0.75rem',
        },
      },
    },
  },
});

export default lightTheme;
