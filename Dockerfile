# Use Python 3.11 base image
FROM python:3.11-slim

# Prevents interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for scikit-learn to build C extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the app port
EXPOSE 10000

# Start the app
CMD ["uvicorn", "src.ml_api:app", "--host", "0.0.0.0", "--port", "10000"]


