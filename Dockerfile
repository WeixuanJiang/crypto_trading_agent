# Use Python 3.11 slim image (3.13 has compatibility issues with pandas)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DOCKER_CONTAINER=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs trading_data

# Set permissions
RUN chmod +x main.py

# Create non-root user for security
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# Default command
CMD ["python", "main.py", "--auto-trading"]
