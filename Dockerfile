# Push-up Analysis API Docker Container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set non-root user for security
RUN useradd -m -u 1000 appuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# The app imports `src.*`, so the repo root (not /app/src) must be importable.
ENV PYTHONPATH=/app

# Create checkpoints directory
RUN mkdir -p checkpoints

# Change ownership of checkpoints to non-root user
RUN chown -R appuser:appuser checkpoints

# Expose port 8000 for the API
EXPOSE 8000

# Healthcheck configuration for Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Switch to non-root user
USER appuser

# Default command: run the API
CMD ["python", "-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative command with gunicorn for production (commented out)
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "src.serving.app:app"]