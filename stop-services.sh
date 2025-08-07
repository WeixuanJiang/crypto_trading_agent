#!/bin/bash

echo "Stopping Crypto Trading Agent services..."
echo "=========================================="

# Function to stop a service
stop_service() {
  SERVICE_NAME=$1
  PID_FILE="logs/$SERVICE_NAME.pid"
  
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p "$PID" > /dev/null; then
      echo "Stopping $SERVICE_NAME service (PID: $PID)..."
      kill "$PID"
      sleep 1
      
      # Check if process still exists
      if ps -p "$PID" > /dev/null; then
        echo "Process still running, forcing termination..."
        kill -9 "$PID"
      fi
      
      echo "$SERVICE_NAME service stopped successfully"
    else
      echo "$SERVICE_NAME service is not running (PID: $PID)"
    fi
    
    rm -f "$PID_FILE"
  else
    echo "No PID file found for $SERVICE_NAME service"
    
    # Try to find and kill processes by name
    if [ "$SERVICE_NAME" = "backend" ]; then
      PIDS=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $2}')
    elif [ "$SERVICE_NAME" = "frontend" ]; then
      PIDS=$(ps aux | grep "npm start" | grep -v grep | awk '{print $2}')
    fi
    
    if [ -n "$PIDS" ]; then
      echo "Found $SERVICE_NAME processes: $PIDS"
      for PID in $PIDS; do
        echo "Killing process $PID..."
        kill "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
      done
      echo "$SERVICE_NAME processes terminated"
    else
      echo "No $SERVICE_NAME processes found"
    fi
  fi
}

# Stop backend and frontend services
stop_service backend
stop_service frontend

echo ""
echo "All services have been stopped."
echo "=========================================="