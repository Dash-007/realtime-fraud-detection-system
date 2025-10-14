# Collection of curl commands for API testing

BASE_URL="http://localhost:8000"

echo "======================================"
echo "Fraud Detection API - curl Examples"
echo "======================================"

# Health Check
echo -e "\n1. Health Check:"
curl -X GET "${BASE_URL}/health" \
  -H "Content-Type: application/json" \
  | json_pp  # Pretty print JSON (or use jq if available)

# Single Prediction
echo -e "\n\n2. Single Prediction (Normal Transaction):"
curl -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 5000.0,
    "Amount": 125.50,
    "V1": 0.144, "V2": 0.358, "V3": 1.220, "V4": 0.331, "V5": -0.273,
    "V6": 0.635, "V7": 0.463, "V8": -0.215, "V9": -0.168, "V10": 0.125,
    "V11": -0.322, "V12": 0.179, "V13": 0.507, "V14": -0.287, "V15": -0.631,
    "V16": 0.123, "V17": -0.294, "V18": 0.661, "V19": 0.738, "V20": 0.868,
    "V21": 0.255, "V22": 0.662, "V23": -0.103, "V24": 0.502, "V25": 0.401,
    "V26": -0.128, "V27": -0.018, "V28": 0.028
  }' \
  | json_pp

# Fraudulent Transaction
echo -e "\n\n3. Single Prediction (Fraudulent Transaction):"
curl -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 12000.0,
    "Amount": 1.99,
    "V1": -1.359, "V2": -0.073, "V3": 2.536, "V4": 1.378, "V5": -0.338,
    "V6": 0.462, "V7": 0.240, "V8": 0.099, "V9": 0.364, "V10": -5.5,
    "V11": -0.552, "V12": -0.618, "V13": -0.991, "V14": -7.2, "V15": 1.468,
    "V16": -3.8, "V17": -2.5, "V18": 0.026, "V19": 0.404, "V20": 0.251,
    "V21": -0.018, "V22": 0.278, "V23": -0.110, "V24": 0.067, "V25": 0.129,
    "V26": -0.189, "V27": 0.134, "V28": -0.021
  }' \
  | json_pp

# Model Info
echo -e "\n\n4. Model Information:"
curl -X GET "${BASE_URL}/model/info" \
  -H "Content-Type: application/json" \
  | json_pp

echo -e "\n\nExamples complete!"
echo "For more examples, visit: ${BASE_URL}/docs"