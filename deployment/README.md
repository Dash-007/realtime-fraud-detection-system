# Deployment Configuration

This directory contains configuration files for Hugging Face Docker Space deployment.

## Files:

- **nginx.conf** - Reverse proxy configuration
- **supervisord.conf** - Process manager for running multiple services
- **start.sh** - Startup script

## Architecture:
```
Port 7860 (Hugging Face)
    ↓
  Nginx
    ↓
  ├─→ Streamlit (localhost:8501) - Dashboard
  └─→ FastAPI (localhost:8000) - API
```

## Logs:

- FastAPI: `/var/log/supervisor/fastapi.out.log`
- Streamlit: `/var/log/supervisor/streamlit.out.log`
- Nginx: `/var/log/nginx/access.log`