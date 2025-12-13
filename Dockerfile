# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install build essentials for scikit-learn, xgboost, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port (match your uvicorn port)
EXPOSE 10000

# Start command
CMD ["uvicorn", "src.ml_api:app", "--host", "0.0.0.0", "--port", "10000"]

