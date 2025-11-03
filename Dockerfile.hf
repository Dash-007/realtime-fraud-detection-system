FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ /app/api
COPY dashboard /app/dashboard
COPY src /app/src/
COPY models/ /app/models/
COPY config/ /app/config/

# COPY configuration files
COPY deployment/nginx.conf /etc/nginx/nginx.conf
COPY deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deployment/start.sh /app/start.sh

# Make satrtup script executable
RUN chmod +x /app/start.sh

# Create necessary directories
RUN mkdir -p /var/log/nginx /var/log/supervisor

# Expose Hugging Face required port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Start services
CMD ["/app/start.sh"]