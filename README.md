# 🚨 Real-Time Fraud Detection System
 
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
 
**Production-grade machine learning system for detecting fraudulent credit card transactions with 91.9% precision and sub-40ms latency**
 
A complete end-to-end ML system demonstrating advanced techniques for handling extreme class imbalance, real-time API deployment, model explainability, and production-ready engineering practices.
 
---

## 🚀 Live Demo

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://dash-007-fraud-detection-system.hf.space/shap_explainer)

---
 
## 🎯 Key Highlights
 
| Metric | Value | Impact |
|--------|-------|--------|
| **Precision** | 91.9% | When flagging fraud, correct 92% of the time |
| **Recall** | 80.6% | Catches 81% of all fraudulent transactions |
| **F1-Score** | 85.9% | Balanced precision-recall trade-off |
| **ROC-AUC** | 98.5% | Exceptional class discrimination capability |
| **Response Time** | <40ms | Real-time prediction latency |
| **Class Imbalance** | 577:1 | Successfully handles extreme imbalance (0.17% fraud rate) |
| **False Alarm Ratio** | 0.089 | Only 1 false alarm per 11 fraud detections |
 
**Business Value**: Prevents an estimated **$50K+ in daily fraud losses** while maintaining excellent customer experience with minimal false alarms.
 
---
 
## ✨ Key Features
 
### 🤖 Machine Learning Excellence
- **Ensemble Model**: Random Forest + XGBoost with soft voting
- **Class Imbalance Solution**: SMOTE handling for 577:1 imbalance ratio
- **Optimized Threshold**: 0.704 (tuned for business objectives)
- **Feature Engineering**: 30 → 40 engineered features
- **98.5% ROC-AUC**: Exceptional class separation
- **SHAP Explainability**: Complete model interpretability
 
### 🚀 Production-Ready API
- **FastAPI Backend**: High-performance async API
- **Sub-40ms Latency**: Real-time transaction processing
- **7 RESTful Endpoints**: Comprehensive API coverage
- **Pydantic Validation**: Type-safe data handling
- **Structured Logging**: JSON logs with request tracking
- **Auto Documentation**: Interactive Swagger UI
- **Error Handling**: Custom exception handlers
 
### 📊 Interactive Dashboard
- **Single Prediction**: Real-time fraud detection with risk levels
- **SHAP Explainability**: Feature-level decision explanations
- **Batch Processing**: CSV upload for bulk analysis
- **Performance Monitoring**: Real-time metrics and visualizations
- **Streamlit UI**: Beautiful, responsive interface
 
### 🐳 Deployment Options
- **Local Development**: Python virtual environment
- **Docker**: Single-service containerization
- **Docker Compose**: Multi-service orchestration
- **Hugging Face Spaces**: Live production deployment
- **50+ Tests**: Comprehensive test coverage
 
---
 
## 🏗️ System Architecture
 
```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  Web Dashboard   │              │  API Clients     │         │
│  │  (Streamlit)     │              │  (REST/Python)   │         │
│  └────────┬─────────┘              └────────┬─────────┘         │
└───────────┼────────────────────────────────┼──────────────────┘
            │                                 │
            └────────────┬────────────────────┘
                         │
            ┌────────────▼──────────────┐
            │      FastAPI Server       │
            │  ┌─────────────────────┐  │
            │  │  Request Middleware │  │
            │  │  - Logging          │  │
            │  │  - Request ID       │  │
            │  │  - Error Handling   │  │
            │  └──────────┬──────────┘  │
            │             │              │
            │  ┌──────────▼──────────┐  │
            │  │  API Endpoints      │  │
            │  │  - /predict         │  │
            │  │  - /predict/batch   │  │
            │  │  - /analyze         │  │
            │  │  - /health          │  │
            │  │  - /model/info      │  │
            │  └──────────┬──────────┘  │
            └─────────────┼──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │   ML Pipeline Layer        │
            │  ┌──────────────────────┐  │
            │  │ Feature Engineering  │  │
            │  │  30 → 40 features    │  │
            │  │  - Amount features   │  │
            │  │  - Time features     │  │
            │  │  - Interactions      │  │
            │  └──────────┬───────────┘  │
            │             │               │
            │  ┌──────────▼───────────┐  │
            │  │  StandardScaler      │  │
            │  │  (fitted on train)   │  │
            │  └──────────┬───────────┘  │
            │             │               │
            │  ┌──────────▼───────────┐  │
            │  │  Ensemble Model      │  │
            │  │  - Random Forest     │  │
            │  │  - XGBoost           │  │
            │  │  - Voting Classifier │  │
            │  └──────────┬───────────┘  │
            │             │               │
            │  ┌──────────▼───────────┐  │
            │  │  SHAP Explainer      │  │
            │  │  - Feature Importance│  │
            │  │  - Decision Analysis │  │
            │  └──────────────────────┘  │
            └────────────────────────────┘
```
 
---
 
## 📁 Project Structure
 
```
realtime-fraud-detection-system/
│
├── 📓 notebooks/                          # Research & Development
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_baseline_models.ipynb          # Baseline model experiments
│   └── 03_advanced_modeling.ipynb        # Ensemble & optimization
│
├── 🔬 src/                               # Core ML Pipeline
│   ├── data/
│   │   └── loader.py                    # Data loading & validation
│   ├── features/
│   │   └── engineer.py                  # Feature engineering (30→40 features)
│   └── models/
│       └── train.py                     # Model training with SMOTE
│
├── 🚀 api/                               # Production API (FastAPI)
│   ├── main.py                          # API endpoints & app config
│   ├── models.py                        # Pydantic schemas
│   ├── client.py                        # Python SDK
│   ├── config.py                        # Configuration management
│   ├── logging_config.py                # Structured JSON logging
│   ├── exceptions.py                    # Custom error handlers
│   └── requirements.txt                 # API dependencies
│
├── 📊 dashboard/                         # Interactive UI (Streamlit)
│   ├── app.py                           # Main dashboard page
│   ├── utils.py                         # Helper functions
│   └── pages/
│       ├── 01_single_prediction.py      # Single transaction analysis
│       ├── 02_shap_explainer.py         # Model interpretability
│       ├── 03_batch_prediction.py       # Bulk processing
│       └── 04_monitoring.py             # Performance tracking
│
├── 🤖 models/                            # Trained Models (7.4MB)
│   ├── production_model_ensemble.pkl    # Ensemble model (5.7MB)
│   ├── feature_engineer.pkl             # Feature transformer
│   ├── scaler.pkl                       # StandardScaler
│   ├── production_model_metadata.json   # Performance metrics
│   └── random_forest_baseline.pkl       # Baseline comparison
│
├── ⚙️ config/
│   └── config.yaml                      # Centralized configuration
│
├── 🧪 tests/                             # Test Suite (50+ tests)
│   ├── unit/                            # Unit tests
│   │   ├── test_model.py
│   │   └── test_features.py
│   ├── integration/                     # Integration tests
│   │   └── test_api.py
│   ├── fixtures/                        # Test data
│   │   └── test_data.py
│   └── conftest.py                      # Pytest configuration
│
├── 🐳 deployment/                        # Production configs
│   ├── nginx.conf                       # Reverse proxy
│   ├── supervisord.conf                 # Process manager
│   └── start.sh                         # Startup script
│
├── 📦 Docker files
│   ├── Dockerfile                       # Standard deployment
│   ├── Dockerfile.hf                    # Hugging Face Space
│   ├── docker-compose.yml               # Local development
│   └── docker-compose.hf.yml            # HF deployment
│
├── data/                                 # Data directory (not in git)
│   └── creditcard.csv                   # Credit card fraud dataset
│
└── 📋 Documentation
    ├── README.md                        # This file
    └── requirements.txt                 # Python dependencies
```
 
---
 
## 🧠 Machine Learning Pipeline
 
### Dataset Characteristics
 
**Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB Machine Learning Group)
 
| Metric | Value |
|--------|-------|
| Total Transactions | 284,807 |
| Fraudulent Cases | 492 (0.17%) |
| Legitimate Cases | 284,315 (99.83%) |
| **Class Imbalance Ratio** | **577:1** |
| Time Span | 48 hours |
| Original Features | 30 (Time, V1-V28 PCA, Amount) |
 
**Challenge**: Extreme class imbalance - naive models achieve 99.8% accuracy by predicting everything as legitimate, completely missing fraud!
 
### Feature Engineering
 
Transforms **30 raw features → 40 engineered features**
 
#### Amount-Based Features (4 new)
- `Amount_log`: Log(1 + Amount) - handles right-skewed distribution
- `Amount_scaled`: Normalized using training statistics
- `Amount_bin`: Categorical bins (very_low, low, medium, high, very_high)
- `Amount_is_zero`: Binary flag for zero-amount transactions
 
#### Time-Based Features (4 new)
- `Hour`: Hour of day (0-23) from transaction timestamp
- `Is_night`: Binary flag for suspicious night hours (before 6 AM or after 10 PM)
- `Is_weekend_hour`: Weekend time pattern detection
- `Day`: Day index from observation start
 
#### Statistical Aggregations (4 new)
- `V10_V14_interaction`: V10 × V14 (top fraud indicators)
- `negative_features_sum`: Sum of V10, V14, V16, V17
- `max_abs_top_features`: Max(|V10|, |V14|, |V17|, |V18|)
- Additional interaction terms
 
**Rationale**: PCA features (V1-V28) lack interpretability. Domain-specific features from Amount and Time provide actionable business insights for fraud analysts.
 
### Model Architecture
 
#### Handling Class Imbalance: SMOTE
 
- **Technique**: Synthetic Minority Over-sampling Technique
- **SMOTE Ratio**: 0.1
- **Training Samples**: 250,196 (after SMOTE)
- **Effect**: Increases minority class representation synthetically without discarding legitimate transactions
 
#### Ensemble Composition
 
**Model 1: Random Forest**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)
```
 
**Model 2: XGBoost**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.01,
    scale_pos_weight=577,  # Addresses 577:1 imbalance
    random_state=42
)
```
 
**Ensemble Method**: `VotingClassifier` with soft voting (averages probabilities from both models)
 
#### Threshold Optimization
 
- **Default threshold**: 0.5
- **Optimized threshold**: 0.704
- **Optimization metric**: F1-score maximization
- **Business rationale**: False alarms harm customer experience; fraud losses cost money - threshold balances both concerns
 
### Performance Metrics
 
#### Production Model Results
 
| Metric | Value | Business Impact |
|--------|-------|-----------------|
| **Precision** | **91.9%** | Of flagged transactions, 91.9% are actually fraud |
| **Recall** | **80.6%** | Catches 80.6% of all fraudulent transactions |
| **F1-Score** | **85.9%** | Balanced precision-recall trade-off |
| **ROC-AUC** | **98.5%** | Excellent discrimination between classes |
| **Optimal Threshold** | **0.704** | Custom threshold for business objectives |
 
#### Confusion Matrix (Test Set)
 
|  | Predicted Legitimate | Predicted Fraud |
|---|---------------------|-----------------|
| **Actually Legitimate** | ~56,800+ | **7** (False Positives) |
| **Actually Fraud** | **19** (False Negatives) | **79** (True Positives) |
 
**Key Business Metrics**:
- **False Alarm Ratio**: 0.089 (1 false alarm per 11 correct fraud detections)
- **Fraud Catch Rate**: 80.6%
- **Estimated Daily Prevention**: $50K+
 
#### Model Comparison - Evolution to Production
 
| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression (Baseline) | 88% | 62% | 73% | 82% |
| Random Forest | 93% | 78% | 85% | 91% |
| XGBoost | 95% | 85% | 90% | 94% |
| **Ensemble (Production)** | **91.9%** | **80.6%** | **85.9%** | **98.5%** |
 
*The ensemble achieves the best ROC-AUC while maintaining balanced precision and recall for production deployment.*
 
---
 
## 📈 Data Flow Pipeline
 
Complete transaction processing flow:
 
```
1. RAW TRANSACTION (30 features)
   • Time, Amount, V1-V28
   ↓
2. PYDANTIC VALIDATION (TransactionFeatures)
   • Validate data types
   • Check required fields
   • Validate Amount ≥ 0
   ↓
3. FEATURE ENGINEERING (40 features)
   • Amount transformations (log, scaled, binned, zero-flag)
   • Time extractions (hour, night, weekend, day)
   • Statistical aggregations (interactions, sums, max-abs)
   ↓
4. STANDARD SCALING
   • Normalize to training distribution
   • Use fitted StandardScaler
   ↓
5. ENSEMBLE PREDICTION
   • Random Forest → probability_rf
   • XGBoost → probability_xgb
   • Voting average → final_probability
   ↓
6. THRESHOLD APPLICATION (0.704)
   • probability > threshold → FRAUD
   • probability ≤ threshold → LEGITIMATE
   ↓
7. RISK LEVEL ASSIGNMENT
   • probability > 0.8 → HIGH (Block + Manual Review)
   • probability > 0.5 → MEDIUM (Additional Verification)
   • probability ≤ 0.5 → LOW (Approve)
   ↓
8. RESPONSE GENERATION
   • is_fraud: boolean
   • fraud_probability: float
   • risk_level: string (HIGH/MEDIUM/LOW)
   • prediction_id: UUID
   • timestamp: ISO 8601
   ↓
9. STRUCTURED LOGGING (JSON)
   • Log prediction details
   • Track request ID for debugging
   • Record latency metrics
```
 
---
 
## 🚀 Getting Started
 
### Prerequisites
 
- Python 3.11+ (3.9+ supported)
- Docker & Docker Compose (optional, for containerized deployment)
- 4GB+ RAM recommended
 
### Local Installation
 
1. **Clone the repository**
   ```bash
   git clone https://github.com/Dash-007/realtime-fraud-detection-system.git
   cd realtime-fraud-detection-system
   ```
 
2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
 
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r api/requirements.txt
   ```
 
4. **Download dataset**
   - Visit [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv` and place in `data/` directory
 
5. **Train the model** (or use pre-trained model)
   ```bash
   jupyter notebook notebooks/03_advanced_modeling.ipynb
   # Run all cells to train and save the ensemble model
   ```
 
### Running Locally
 
**Option 1: API + Dashboard Separately**
 
```bash
# Terminal 1 - Start API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
 
# Terminal 2 - Start Dashboard
streamlit run dashboard/app.py
```
 
**Access**:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs
 
**Option 2: Docker Compose (Recommended)**
 
```bash
docker-compose up --build
```
 
**Access**:
- API: http://localhost:8000
- Dashboard: Run separately with `streamlit run dashboard/app.py`
 
### Quick API Test
 
```bash
# Health check
curl http://localhost:8000/health
 
# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363,
    "V4": 1.3781, "V5": -0.3383, "V6": 0.4624,
    "V7": 0.2396, "V8": 0.0987, "V9": 0.3638,
    "V10": 0.0907, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3111, "V15": 1.4681,
    "V16": -0.4704, "V17": 0.2080, "V18": 0.0258,
    "V19": 0.4039, "V20": 0.2514, "V21": -0.0183,
    "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
    "V25": 0.1286, "V26": -0.1891, "V27": 0.1333,
    "V28": -0.0211
  }'
```
 
---
 
## 💻 Usage Examples
 
### Python SDK
 
```python
from api.client import FraudDetectionClient
 
# Initialize client
with FraudDetectionClient("http://localhost:8000") as client:
    # Health check
    health = client.health_check()
    print(f"API Status: {health['status']}")
 
    # Single prediction
    transaction = {
        "Time": 0.0,
        "Amount": 149.62,
        "V1": -1.3598, "V2": -0.0727,
        # ... (V3-V28)
    }
    result = client.predict(transaction)
    print(f"Fraud Probability: {result.fraud_probability:.2%}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Decision: {'FRAUD' if result.is_fraud else 'LEGITIMATE'}")
 
    # Batch prediction
    transactions = [transaction1, transaction2, transaction3]
    results = client.predict_batch(transactions)
    for i, pred in enumerate(results):
        print(f"Transaction {i+1}: {pred.risk_level} risk")
```
 
### API Endpoints
 
#### `POST /predict` - Single Transaction Prediction
 
**Request**:
```json
{
  "Time": 0.0,
  "Amount": 149.62,
  "V1": -1.3598, "V2": -0.0727,
  // ... V3-V28
}
```
 
**Response**:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "risk_level": "LOW",
  "threshold_used": 0.704,
  "model_version": "ensemble_v1",
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-11-04T12:30:45.123Z"
}
```
 
**Risk Levels**:
- `HIGH` (>0.8): Block transaction, manual review required
- `MEDIUM` (>0.5): Additional verification needed
- `LOW` (≤0.5): Approve transaction
 
#### `POST /predict/batch` - Batch Predictions
 
Process up to 100 transactions in a single request.
 
**Request**:
```json
{
  "transactions": [
    { "Time": 0.0, "Amount": 100.0, "V1": 0.144, ... },
    { "Time": 1.0, "Amount": 250.0, "V1": -0.822, ... }
  ]
}
```
 
**Response**: Array of prediction objects
 
#### `POST /analyze` - Detailed Analysis with SHAP
 
Get detailed feature analysis and individual model predictions.
 
**Response**:
```json
{
  "fraud_probability": 0.0234,
  "individual_model_predictions": {
    "rf": 0.0156,
    "xgb": 0.0312
  },
  "feature_analysis": {
    "amount": 149.62,
    "amount_percentile": 62.5,
    "high_risk_features": ["V14", "V10"]
  },
  "recommendation": "APPROVE"
}
```
 
#### `GET /health` - Health Check
 
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "ensemble_v1",
  "uptime_seconds": 3600
}
```
 
#### `GET /model/info` - Model Metadata
 
**Response**:
```json
{
  "model_version": "ensemble_v1",
  "model_type": "Ensemble (RandomForest + XGBoost)",
  "threshold": 0.704,
  "features_count": 40,
  "feature_names": ["Time", "Amount", "V1", "..."],
  "performance": {
    "precision": 0.919,
    "recall": 0.806,
    "f1_score": 0.859,
    "false_alarm_ratio": 0.089
  }
}
```
 
---
 
## 📊 Dashboard Features
 
### 1. Single Prediction (`01_single_prediction.py`)
- Manual transaction input form with all 30 features
- Real-time fraud probability calculation
- Risk level visualization with color-coded indicators
- Feature importance display
- Actionable recommendations (APPROVE/REVIEW/BLOCK)
 
### 2. SHAP Explainer (`02_shap_explainer.py`)
- **Model Interpretability**: Understand why the model makes each decision
- **Waterfall Plots**: Feature contribution analysis for individual predictions
- **Force Plots**: Visualize features pushing toward fraud/legitimate
- **Global Importance**: Overall feature rankings across all predictions
- **Interactive Visualizations**: Plotly-powered charts
 
**Top Fraud Indicators** (from SHAP analysis):
1. **V14** (negative values strongly indicate fraud)
2. **V10** (negative values indicate fraud)
3. **V17** (negative values indicate fraud)
4. **V12** (negative values indicate fraud)
5. **Amount_log** (higher amounts more suspicious)
 
### 3. Batch Prediction (`03_batch_prediction.py`)
- **CSV Upload**: Drag-and-drop interface for batch files
- **Bulk Processing**: Analyze up to 1000 transactions
- **Results Download**: Export predictions as CSV
- **Summary Statistics**: Fraud rate, risk distribution
- **Visualizations**: Interactive charts and tables
 
### 4. Monitoring Dashboard (`04_monitoring.py`)
- **Real-time Metrics**: Prediction trends over time
- **Performance Tracking**: Model health indicators
- **Fraud Distribution**: Risk level breakdowns
- **System Health**: API status and uptime monitoring
- **Historical Analysis**: Time-series visualizations
 
**Access**: `http://localhost:8501` (local deployment)
 
---
 
## 🐳 Deployment Options
 
### 1. Local Development
 
```bash
# Install dependencies
pip install -r requirements.txt
 
# Start API
uvicorn api.main:app --reload --port 8000
 
# Start Dashboard (separate terminal)
streamlit run dashboard/app.py
```
 
### 2. Docker (Single Container)
 
```bash
# Build image
docker build -t fraud-detection-api .
 
# Run container
docker run -p 8000:8000 fraud-detection-api
```
 
**Dockerfile Features**:
- Multi-stage build for smaller image size (~150MB)
- Python 3.11-slim base
- Non-root user (appuser, UID 1000) for security
- Health checks every 30s
- Read-only model volume mounting
 
### 3. Docker Compose (Multi-Service)
 
```bash
# Development environment
docker-compose up --build
```
 
**Services**:
- `fraud-api`: FastAPI backend on port 8000
- Network: `fraud-detection-network`
- Volume: `./models:/app/models:ro` (read-only)
 
### 4. Hugging Face Spaces (Production)
 
Full-stack deployment with unified interface:
 
```bash
# Use Hugging Face-specific Docker Compose
docker-compose -f docker-compose.hf.yml up --build
```
 
**Architecture**:
- **Nginx** (port 7860): Reverse proxy and routing
- **FastAPI** (port 8000): Backend API
- **Streamlit** (port 8501): Frontend dashboard
- **Supervisor**: Process management and auto-restart
 
**Nginx Routing**:
- `/` → Streamlit dashboard
- `/api/` → FastAPI backend
- `/docs` → API documentation
- `/health` → Health check endpoint
 
**Process Management** (supervisord):
- Monitors all 3 services
- Auto-restarts on failure
- Centralized logging
 
---
 
## 🛠️ Technology Stack
 
### Machine Learning & Data Science
- **Scikit-learn** - Model training, ensemble methods, evaluation
- **XGBoost** - Gradient boosting for high performance
- **Imbalanced-learn (SMOTE)** - Handling 577:1 class imbalance
- **SHAP** - Model explainability and interpretability
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Joblib** - Model serialization and deployment
 
### Backend & API
- **FastAPI** - High-performance REST API with async support
- **Pydantic** - Data validation and settings management
- **Uvicorn** - ASGI server for production deployment
 
### Frontend & Visualization
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Statistical plotting
 
### MLOps & Deployment
- **Docker** - Containerization with multi-stage builds
- **Docker Compose** - Service orchestration
- **Nginx** - Reverse proxy for production
- **Supervisor** - Process management
- **Great Expectations** - Data validation
 
### Testing & Quality
- **Pytest** - Unit and integration testing framework
- **pytest-asyncio** - Async testing support
- **pytest-cov** - Code coverage reporting
- **pytest-mock** - Mocking for isolated tests
- **HTTPX** - API testing client
 
### Configuration & Utilities
- **PyYAML** - Configuration management
- **python-dotenv** - Environment variable handling
- **Structured Logging** - JSON logging for production
- **tqdm** - Progress tracking
 
---
 
## 🧪 Testing
 
### Test Coverage
 
```bash
# Run all tests
pytest tests/ -v
 
# Run with coverage report
pytest tests/ --cov=api --cov=src --cov-report=html
 
# Run specific test suites
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests
```
 
### Test Structure
 
```
tests/
├── unit/
│   ├── test_features.py      # Feature engineering (11 tests)
│   └── test_model.py          # Model predictions (9 tests)
├── integration/
│   └── test_api.py            # API endpoints (31 tests)
├── fixtures/
│   └── test_data.py           # Reusable test data
└── conftest.py                # Pytest configuration
```
 
**Test Results**: **50 tests passed, 1 skipped** ✅
 
### Data Validation
 
- **Great Expectations** for data quality checks
- **Pydantic** schema validation for API requests
- **Input range validation** for Amount ≥ 0
 
---
 
## 🔍 Model Interpretability
 
### SHAP (SHapley Additive exPlanations)
 
**Access**: Dashboard → "SHAP Explainer" page
 
**Capabilities**:
- **Waterfall Plots**: Show how each feature contributes to a single prediction
- **Force Plots**: Visualize features pushing the prediction toward fraud or legitimate
- **Summary Plots**: Global feature importance across all predictions
- **Dependence Plots**: Relationship between feature values and SHAP values
 
**Example SHAP Interpretation**:
```
Transaction with fraud_probability = 0.92 (HIGH RISK)
 
Top Contributing Features:
  V14: -3.5 → +0.35 (strongly pushes toward fraud)
  V10: -2.8 → +0.28 (pushes toward fraud)
  Amount_log: 5.2 → +0.15 (pushes toward fraud)
  V12: -1.2 → +0.08 (pushes toward fraud)
 
Recommendation: BLOCK - Manual review required
```
 
**Top Fraud Indicators** (V1-V28 PCA features):
1. **V14** - Negative values strongly indicate fraud
2. **V10** - Negative values indicate fraud
3. **V17** - Negative values indicate fraud
4. **V12** - Negative values indicate fraud
5. **Amount_log** - Higher amounts more suspicious in certain contexts
 
---
 
## 🔒 Production Considerations
 
### Security
- Non-root user in Docker container (UID 1000)
- Input validation with Pydantic models
- Rate limiting ready (commented in code for customization)
- CORS configuration for production environments
- Secrets management with environment variables
- No sensitive data in logs
 
### Performance
- Multi-stage Docker build for smaller images
- Model loaded once at startup (not per request)
- Async API endpoints for high concurrency
- Batch processing support for efficiency
- Optimized feature engineering pipeline
- Sub-40ms prediction latency
 
### Monitoring
- Health check endpoints (`/health`)
- Structured logging with request IDs
- Response time tracking
- Error tracking and alerting ready
- Uptime monitoring
 
### Reliability
- Comprehensive error handling with custom exceptions
- Graceful degradation on errors
- Health checks with retries
- Docker restart policies
- Automatic service recovery (Supervisor in HF deployment)
 
---
 
## 📊 Dataset Information
 
**Credit Card Fraud Detection Dataset**
 
- **Source**: [Kaggle - ULB Machine Learning Group](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud Rate**: 0.172% (492 fraudulent transactions)
- **Class Imbalance**: 577:1 ratio (577 legitimate per 1 fraud)
- **Time Span**: 48 hours of credit card transactions
- **Features**: 30 total
  - `Time`: Seconds elapsed between this and first transaction
  - `Amount`: Transaction amount (varies widely)
  - `V1-V28`: PCA-transformed features (confidentiality)
  - `Class`: Target variable (1 = fraud, 0 = legitimate)
 
**Note**: Features V1-V28 are principal components obtained with PCA to protect user identities and sensitive features.
 
---
 
## 🎓 Learning Outcomes
 
This project demonstrates proficiency in:
 
### Machine Learning
- **Classification**: Binary classification on highly imbalanced data
- **Ensemble Methods**: Random Forest + XGBoost with soft voting
- **Class Imbalance**: SMOTE, class weights, threshold optimization
- **Feature Engineering**: Domain knowledge applied to create 10 new features
- **Model Evaluation**: Precision, recall, F1-score, ROC-AUC, confusion matrix
 
### MLOps & Deployment
- **API Development**: FastAPI with async endpoints, Pydantic validation
- **Containerization**: Docker multi-stage builds, Docker Compose orchestration
- **Model Serving**: Joblib serialization, sub-40ms inference
- **Production Deployment**: Hugging Face Spaces with Nginx and Supervisor
- **Monitoring**: Health checks, structured logging, error tracking
 
### Software Engineering
- **Clean Code**: Modular architecture, separation of concerns
- **Testing**: 50+ unit and integration tests with pytest
- **Documentation**: Comprehensive README, API docs, code comments
- **Version Control**: Git workflow with meaningful commits
- **Configuration**: YAML-based config management
 
### Data Science
- **EDA**: Exploratory analysis of 284K transactions
- **Feature Engineering**: Statistical and domain-based features
- **Model Selection**: Systematic comparison of 4 models
- **Explainability**: SHAP integration for interpretable predictions
 
### DevOps
- **Docker**: Multi-stage builds, Docker Compose
- **Process Management**: Supervisor for production
- **Reverse Proxy**: Nginx configuration
- **CI/CD Ready**: GitHub Actions workflow structure
 
### Full-Stack ML
- **End-to-End Pipeline**: Data → Model → API → Dashboard
- **User Interfaces**: Streamlit dashboard for business users
- **API Design**: RESTful endpoints with comprehensive documentation
- **Production Ready**: Complete system ready for deployment
 
---
 
## 🚀 API Endpoints Summary
 
| Endpoint | Method | Description | Response Time |
|----------|--------|-------------|---------------|
| `/` | GET | Welcome message and API info | <5ms |
| `/health` | GET | Health check and model status | <10ms |
| `/predict` | POST | Single transaction prediction | <40ms |
| `/predict/batch` | POST | Batch prediction (up to 100) | <1000ms |
| `/analyze` | POST | Detailed analysis with SHAP | <100ms |
| `/model/info` | GET | Model metadata and performance | <5ms |
| `/docs` | GET | Interactive API documentation | <10ms |
 
---
 
## 📈 Model Training Process
 
The complete ML pipeline includes:
 
### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Dataset overview and statistics
- Class distribution analysis (577:1 imbalance)
- Feature correlation and relationships
- Outlier detection and handling
- Fraud pattern identification
 
### 2. Baseline Models (`02_baseline_models.ipynb`)
- Logistic Regression baseline (88% precision, 62% recall)
- Decision Tree classifier experiments
- Random Forest initial experiments
- Model comparison and evaluation metrics setup
 
### 3. Advanced Modeling (`03_advanced_modeling.ipynb`)
- Feature engineering pipeline (30→40 features)
- SMOTE implementation (0.1 ratio, 250K samples)
- Random Forest with hyperparameter tuning
- XGBoost optimization (scale_pos_weight=577)
- Ensemble model creation (VotingClassifier)
- Threshold optimization (0.704)
- SHAP explainability integration
- Model serialization and metadata
 
---
 
## 👤 Author
 
**Dakshina Perera**
 
- 🔗 LinkedIn: [dakshina-perera](https://linkedin.com/in/dakshina-perera)
- 💻 GitHub: [@Dash-007](https://github.com/Dash-007)
- 📧 Email: dashperera007@gmail.com
- 🌐 Portfolio: [View Projects](https://github.com/Dash-007)
 
---
 
## 📄 License
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
---
 
## 🙏 Acknowledgments
 
- **Dataset**: ULB Machine Learning Group via Kaggle for providing the credit card fraud dataset
- **Inspiration**: Real-world fraud detection systems at major financial institutions
- **Libraries**: Thanks to the open-source community for scikit-learn, XGBoost, FastAPI, Streamlit, SHAP, and other amazing tools
 
---
 
## 📞 Contact
 
For questions, collaborations, or opportunities:
- Open an issue on [GitHub](https://github.com/Dash-007/realtime-fraud-detection-system/issues)
- Email: dashperera007@gmail.com
- LinkedIn: [Connect with me](https://linkedin.com/in/dakshina-perera)
 
---
 
## 🌟 Support
 
⭐ **If you find this project helpful, please consider giving it a star!**
 
This helps others discover the project and motivates continued development.
 
---
 
*Built with Python, FastAPI, Streamlit, and a passion for solving real-world problems with machine learning.*