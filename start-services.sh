#!/bin/bash

echo "Starting Crypto Trading Agent Backend and Frontend in background..."
echo "=========================================="

# Create required directories if they don't exist
mkdir -p logs trading_data data

# Check if .env file exists
if [ ! -f .env ]; then
  echo "Warning: .env file not found. Creating example file..."
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
    echo "Please update the .env file with your actual API keys and settings."
  else
    echo "Warning: No .env.example file found. You may need to create a .env file manually."
  fi
fi

# Start the backend in the background with logging
echo "Starting backend server in background..."
python3 main.py >> logs/trading_agent.log 2>&1 &
BACKEND_PID=$!
if [ $? -eq 0 ]; then
  echo "Backend server started with PID: $BACKEND_PID"
  echo "Logs are being written to: logs/trading_agent.log"
  echo "API is available at: http://localhost:5001"
else
  echo "Failed to start backend server!"
fi

# Wait a moment to ensure backend starts
sleep 2

# Start the frontend in the background with logging
echo "Starting frontend server in background..."
cd frontend && PORT=3000 npm start >> ../logs/trading_agent.log 2>&1 &
FRONTEND_PID=$!
if [ $? -eq 0 ]; then
  echo "Frontend server started with PID: $FRONTEND_PID"
  echo "Logs are being written to: logs/trading_agent.log"
  echo "Frontend is available at: http://localhost:3000"
else
  echo "Failed to start frontend server!"
fi

# Save PIDs for future reference
cd ..
# Ensure logs directory exists
mkdir -p logs
echo "$BACKEND_PID" > logs/backend.pid
echo "$FRONTEND_PID" > logs/frontend.pid

echo ""
echo "Services started in background mode!"
echo "=========================================="
echo "Backend API URL: http://localhost:5001"
echo "Frontend URL: http://localhost:3000"
echo ""
echo "To stop the services, run: ./stop-services.sh"
echo "To view logs: tail -f logs/trading_agent.log"
echo "=========================================="