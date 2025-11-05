FROM python:3.9-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
FROM base AS builder

# Copy requirements files
COPY requirements.txt .
COPY api/requirements.txt ./api/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r api/requirements.txt

# Stage 3: Runtime stage
FROM base AS runtime

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY api/ /app/api/
COPY src/ /app/src/
COPY models/ /app/models/
COPY config/ /app/config/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check - simple TCP connection test
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('localhost', 8000), timeout=2)" || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]