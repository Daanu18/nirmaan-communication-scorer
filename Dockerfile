FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Expose port (Railway sets PORT env var)
EXPOSE $PORT

# Start command
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

