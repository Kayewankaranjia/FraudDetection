# Fraud Detection API (FastAPI + PyTorch)

This project provides a REST API for fraud detection using a PyTorch model and FastAPI.

## Features

- `/predict`: Single prediction endpoint (requires authentication)
- `/batch_predict`: Batch prediction endpoint (requires authentication)
- TensorBoard logging for inference statistics

---

## Quick Start: Run with Docker

### 1. Set environment variables

Create a `.env` file in the project root with:

```
MODEL_PATH=fraud_prevention_model.pt
API_USERNAME=your_username
API_PASSWORD=your_password
```
### 2. Build the Docker image

```bash
docker build -t fraud-detection-api .
```

### 3. Run the container

```bash
docker run -d \
  --name fraud-detection-api \
  -p 8000:8000 \
  fraud-detection-api
```


---

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -u your_username:your_password \
  -H "Content-Type: application/json" \
  -d "[100.0, 5.0, 1, 6]"
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -u your_username:your_password \
  -H "Content-Type: application/json" \
  -d "[[0.2, 1.0, 0.0, 5.2], [1.1, 0.3, 1.0, 2.9]]"
```

---

**Access TensorBoard in your browser:**

   ```
   http://localhost:6006
   ```

---

## Development

To run locally (without Docker):

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

---