import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  IconButton,
  Box,
  Typography
} from '@mui/material';
import {
  Close as CloseIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  CheckCircle as SuccessIcon,
  Help as QuestionIcon
} from '@mui/icons-material';

const SEVERITY_ICONS = {
  warning: WarningIcon,
  error: ErrorIcon,
  info: InfoIcon,
  success: SuccessIcon,
  question: QuestionIcon
};

const SEVERITY_COLORS = {
  warning: 'warning.main',
  error: 'error.main',
  info: 'info.main',
  success: 'success.main',
  question: 'primary.main'
};

/**
 * Confirmation dialog component
 */
const ConfirmDialog = ({
  open,
  onClose,
  onConfirm,
  title = 'Confirm Action',
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  severity = 'warning',
  showIcon = true,
  confirmColor = 'primary',
  cancelColor = 'inherit',
  maxWidth = 'sm',
  fullWidth = true,
  loading = false,
  disableBackdropClick = false,
  disableEscapeKeyDown = false,
  children,
  ...props
}) => {
  const IconComponent = SEVERITY_ICONS[severity];
  const iconColor = SEVERITY_COLORS[severity];

  const handleClose = (event, reason) => {
    if (disableBackdropClick && reason === 'backdropClick') {
      return;
    }
    if (disableEscapeKeyDown && reason === 'escapeKeyDown') {
      return;
    }
    onClose?.();
  };

  const handleConfirm = () => {
    onConfirm?.();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth={maxWidth}
      fullWidth={fullWidth}
      {...props}
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {showIcon && IconComponent && (
            <IconComponent sx={{ color: iconColor, fontSize: 28 }} />
          )}
          <Typography variant="h6" component="span">
            {title}
          </Typography>
        </Box>
        <IconButton
          edge="end"
          onClick={onClose}
          disabled={loading}
          sx={{ ml: 1 }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent>
        {message && (
          <DialogContentText>
            {message}
          </DialogContentText>
        )}
        {children}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button
          onClick={onClose}
          color={cancelColor}
          disabled={loading}
        >
          {cancelText}
        </Button>
        <Button
          onClick={handleConfirm}
          color={confirmColor}
          variant="contained"
          disabled={loading}
          autoFocus
        >
          {loading ? 'Processing...' : confirmText}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

/**
 * Delete confirmation dialog
 */
export const DeleteConfirmDialog = ({
  open,
  onClose,
  onConfirm,
  itemName,
  title = 'Confirm Delete',
  ...props
}) => (
  <ConfirmDialog
    open={open}
    onClose={onClose}
    onConfirm={onConfirm}
    title={title}
    message={
      itemName
        ? `Are you sure you want to delete "${itemName}"? This action cannot be undone.`
        : 'Are you sure you want to delete this item? This action cannot be undone.'
    }
    confirmText="Delete"
    confirmColor="error"
    severity="error"
    {...props}
  />
);

/**
 * Discard changes confirmation dialog
 */
export const DiscardChangesDialog = ({
  open,
  onClose,
  onConfirm,
  ...props
}) => (
  <ConfirmDialog
    open={open}
    onClose={onClose}
    onConfirm={onConfirm}
    title="Discard Changes?"
    message="You have unsaved changes. Are you sure you want to discard them?"
    confirmText="Discard"
    confirmColor="error"
    cancelText="Keep Editing"
    severity="warning"
    {...props}
  />
);

/**
 * Execute trade confirmation dialog
 */
export const TradeConfirmDialog = ({
  open,
  onClose,
  onConfirm,
  tradeDetails,
  ...props
}) => (
  <ConfirmDialog
    open={open}
    onClose={onClose}
    onConfirm={onConfirm}
    title="Confirm Trade"
    confirmText="Execute Trade"
    confirmColor="primary"
    severity="question"
    {...props}
  >
    {tradeDetails && (
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" gutterBottom>
          <strong>Pair:</strong> {tradeDetails.pair}
        </Typography>
        <Typography variant="body2" gutterBottom>
          <strong>Side:</strong> {tradeDetails.side}
        </Typography>
        <Typography variant="body2" gutterBottom>
          <strong>Price:</strong> ${tradeDetails.price}
        </Typography>
        <Typography variant="body2" gutterBottom>
          <strong>Quantity:</strong> {tradeDetails.quantity}
        </Typography>
        <Typography variant="body2" gutterBottom>
          <strong>Total:</strong> ${tradeDetails.total}
        </Typography>
      </Box>
    )}
  </ConfirmDialog>
);

/**
 * Reset settings confirmation dialog
 */
export const ResetSettingsDialog = ({
  open,
  onClose,
  onConfirm,
  ...props
}) => (
  <ConfirmDialog
    open={open}
    onClose={onClose}
    onConfirm={onConfirm}
    title="Reset Settings"
    message="Are you sure you want to reset all settings to their default values? This action cannot be undone."
    confirmText="Reset"
    confirmColor="error"
    severity="warning"
    {...props}
  />
);

/**
 * Logout confirmation dialog
 */
export const LogoutConfirmDialog = ({
  open,
  onClose,
  onConfirm,
  ...props
}) => (
  <ConfirmDialog
    open={open}
    onClose={onClose}
    onConfirm={onConfirm}
    title="Confirm Logout"
    message="Are you sure you want to logout?"
    confirmText="Logout"
    confirmColor="primary"
    severity="question"
    {...props}
  />
);

/**
 * Stop trading confirmation dialog
 */
export const StopTradingDialog = ({
  open,
  onClose,
  onConfirm,
  hasOpenPositions = false,
  ...props
}) => (
  <ConfirmDialog
    open={open}
    onClose={onClose}
    onConfirm={onConfirm}
    title="Stop Trading"
    message={
      hasOpenPositions
        ? 'You have open positions. Are you sure you want to stop trading? Your open positions will remain active.'
        : 'Are you sure you want to stop trading?'
    }
    confirmText="Stop Trading"
    confirmColor="error"
    severity={hasOpenPositions ? 'error' : 'warning'}
    {...props}
  />
);

export default ConfirmDialog;
