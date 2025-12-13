# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy everything else
COPY . .

# Expose port and run app
EXPOSE 10000
CMD ["uvicorn", "src.ml_api:app", "--host", "0.0.0.0", "--port", "10000"]
