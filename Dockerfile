# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy models directory
COPY models/ ./models/

# Set environment variables
ENV PYTHONPATH=/app
ENV AZURE_STORAGE_CONNECTION_STRING=""
ENV AZURE_SERVICE_BUS_CONNECTION_STRING=""
ENV AZURE_COSMOS_CONNECTION_STRING=""

# Expose port
EXPOSE 8080

# Run the application
CMD ["python", "processor_main.py"]
