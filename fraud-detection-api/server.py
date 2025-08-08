from fastapi import FastAPI, Body
import torch
import numpy as np
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from torch.utils.tensorboard import SummaryWriter
import secrets
import os
from dotenv import load_dotenv
import time
load_dotenv()

# --- Load TorchScript model ---
model = torch.jit.load(os.getenv("MODEL_PATH"))
model.eval()

# Create a logs directory
os.makedirs("runs", exist_ok=True)
writer = SummaryWriter(log_dir="runs/fraud_inference")
stats = {
    "request_count": 0,
    "fraud_predictions": 0,
    "total_inference_time": 0.0
}

NUM_FEATURES = 4  
USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")

# --- FastAPI setup ---
app = FastAPI(title="Fraud Detection API")
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

@app.post("/predict")
async def predict(
    features: list = Body(..., description="List of features for prediction", example=[100.0, 5.0, 1, 6]),
    credentials: HTTPBasicCredentials = Depends(authenticate)
):
    features = np.array(features).astype(np.float32)
    if features.shape[0] != NUM_FEATURES:
        return {"error": f"Expected {NUM_FEATURES} features, got {features.shape[0]}"}
    tensor = torch.tensor(features).unsqueeze(0) 
    start_time = time.time()
    with torch.no_grad():
        logits = model.forward(tensor)
        # Since model outputs a logit, we apply sigmoid for probability
        prob = torch.sigmoid(logits).item()
        prediction = int(prob > 0.05)
    inference_time = time.time() - start_time

    # Update local stats
    stats["request_count"] += 1
    stats["fraud_predictions"] += prediction
    stats["total_inference_time"] += inference_time

    # Log to TensorBoard
    writer.add_scalar("Inference/Probability", prob, stats["request_count"])
    writer.add_scalar("Inference/Prediction", prediction, stats["fraud_predictions"] )
    writer.add_scalar("Inference/InferenceTime", inference_time, stats["total_inference_time"])
    writer.add_scalar("inference/fraud_rate", stats["fraud_predictions"] / stats["request_count"], stats["request_count"])
    writer.add_scalar("inference/avg_inference_time", stats["total_inference_time"] / stats["request_count"], stats["request_count"])


    
    return {
        "input": features.tolist(),
        "prediction": prediction,
        "confidence": prob
    }

@app.post("/batch_predict")
async def batch_predict(
    features_batch: list = Body(..., description="Batch of feature sets for prediction", example=[[0.2, 1.0, 0.0, 5.2], [1.1, 0.3, 1.0, 2.9]]),
    credentials: HTTPBasicCredentials = Depends(authenticate)
):
    features_batch = np.array(features_batch).astype(np.float32)
    if features_batch.ndim != 2 or features_batch.shape[1] != NUM_FEATURES:
        return {"error": f"Each input must have {NUM_FEATURES} features. Received shape: {features_batch.shape}"}
    tensor = torch.tensor(features_batch)
    start_time = time.time()
    with torch.no_grad():
        logits = model.forward(tensor)
        probs = torch.sigmoid(logits).squeeze().numpy()
        preds = (probs > 0.05).astype(int).tolist()
    inference_time = time.time() - start_time

    batch_size = len(features_batch)
    fraud_count = sum(preds)

    # Update stats
    stats["request_count"] += batch_size
    stats["fraud_predictions"] += fraud_count
    stats["total_inference_time"] += inference_time

    # Log to TensorBoard for each prediction in the batch sequentially
    for iterator, (prob, pred) in enumerate(zip(probs.tolist(), preds)):
        writer.add_scalar("Inference/Probability", prob, stats["request_count"] - batch_size + iterator + 1)
        writer.add_scalar("Inference/Prediction", pred, stats["fraud_predictions"])
        writer.add_scalar("Inference/InferenceTime", inference_time / batch_size, stats["total_inference_time"])
        writer.add_scalar("inference/fraud_rate", stats["fraud_predictions"] / stats["request_count"], stats["request_count"] - batch_size + iterator + 1)
        writer.add_scalar("inference/avg_inference_time", stats["total_inference_time"] / stats["request_count"], stats["request_count"] - batch_size + iterator + 1)

    return {
        "input_features": features_batch.tolist(),
        "predictions": preds,
        "confidence": probs.tolist()
    }


