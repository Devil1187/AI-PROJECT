# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: The requirements include PyTorch with CUDA 12.1
# For CPU-only deployment, you can modify the --extra-index-url and torch version
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Create a directory for models (if not already present)
RUN mkdir -p Models

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
