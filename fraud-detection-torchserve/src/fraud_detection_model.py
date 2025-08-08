from typing import List, Dict, Any
import torch

class FraudDetectionModel:
    def __init__(self):
        self.model = None

    def load_model(self, model_path: str):
        model_path = "model-store/fraud_detection_model.mar"
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def predict(self, features: List[float]) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model is not loaded. Please load the model before prediction.")
        
        tensor = torch.tensor(features).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.model.forward(tensor)
            prob = torch.sigmoid(logits).item()
            prediction = int(prob > 0.5)

        return {
            "prediction": prediction,
            "confidence": prob
        }
    
    def batch_predict(self, features_batch: List[List[float]]) -> Dict[str, Any]:
      if self.model is None:
          raise ValueError("Model is not loaded. Please load the model before prediction.")
      
      tensor = torch.tensor(features_batch).float()
      with torch.no_grad():
          logits = self.model.forward(tensor)
          probs = torch.sigmoid(logits).squeeze().numpy()
          preds = (probs > 0.5).astype(int).tolist()
      return {
          "predictions": preds,
          "confidence": probs.tolist()
      }