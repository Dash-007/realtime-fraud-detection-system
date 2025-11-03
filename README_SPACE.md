---
title: Real-Time Fraud Detection System
emoji: 🚨
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# 🚨 Real-Time Fraud Detection System

**Production-grade ML system with FastAPI backend + Streamlit dashboard**

## 🎯 Live Demo

- **📊 Dashboard**: [Main Page](/)
- **📚 API Docs**: [Interactive Documentation](/docs)
- **🏥 Health Check**: [System Status](/api/health)

## ⚡ Quick API Test
```bash
# Test the live API
curl -X POST "https://Dash-007.hf.space/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "Amount": 100.0,
    "V1": 0.144, "V2": 0.358, "V3": 1.220,
    ... (add all V features)
  }'
```

## 🏗️ Architecture

This Space runs a complete production system:
```
┌─────────────────────────────────────┐
│   Docker Container (Port 7860)      │
│                                      │
│   ┌──────────────────────────────┐  │
│   │  Nginx Reverse Proxy         │  │
│   └────┬──────────────────┬──────┘  │
│        │                  │          │
│   ┌────▼────────┐   ┌────▼──────┐  │
│   │  FastAPI    │   │ Streamlit │  │
│   │  Port 8000  │   │ Port 8501 │  │
│   └─────────────┘   └───────────┘  │
└─────────────────────────────────────┘
```

### Components:

1. **FastAPI Backend** (`/api/*`)
   - Real-time fraud prediction
   - Batch processing
   - SHAP analysis
   - Health monitoring

2. **Streamlit Dashboard** (`/`)
   - Interactive UI
   - SHAP visualizations
   - Batch uploads
   - Performance metrics

3. **Nginx**
   - Route management
   - Load balancing
   - WebSocket support

## 📊 Model Performance

- **Precision**: 91.9%
- **Recall**: 80.6%
- **F1-Score**: 85.9%
- **Response Time**: ~40ms

## 🛠️ Technology Stack

- **ML**: Scikit-learn, XGBoost, SHAP
- **Backend**: FastAPI, Pydantic
- **Frontend**: Streamlit, Plotly
- **Deployment**: Docker, Nginx, Supervisor
- **Testing**: Pytest (50+ tests)
- **CI/CD**: GitHub Actions

## 🚀 Features

### API Endpoints

- `POST /api/predict` - Single transaction prediction
- `POST /api/predict/batch` - Batch predictions
- `POST /api/analyze` - Detailed SHAP analysis
- `GET /api/health` - Health check
- `GET /api/model/info` - Model information

### Dashboard Pages

- **Home** - System overview and metrics
- **Single Prediction** - Interactive fraud detection
- **SHAP Explainer** - Model interpretability
- **Batch Analysis** - CSV upload and processing
- **Monitoring** - Performance tracking

## 💻 Local Development
```bash
# Clone repository
git clone https://github.com/Dash-007/realtime-fraud-detection-system

# Build and run
docker-compose up --build

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000/docs
```

## 📖 Documentation

- **GitHub**: [Full Documentation](https://github.com/Dash-007/realtime-fraud-detection-system)
- **API Docs**: [/docs](/docs)
- **Model Details**: See repository README

## 🎓 ML Engineering Showcase

This project demonstrates:
- ✅ End-to-end ML pipeline
- ✅ Production API design
- ✅ Model explainability (SHAP)
- ✅ Docker containerization
- ✅ Automated testing & CI/CD
- ✅ Interactive dashboards
- ✅ Multi-service architecture

---

**Built with ❤️ for ML Engineering**

[GitHub](https://github.com/Dash-007/realtime-fraud-detection-system) • [LinkedIn](https://linkedin.com/in/dakshina-perera)