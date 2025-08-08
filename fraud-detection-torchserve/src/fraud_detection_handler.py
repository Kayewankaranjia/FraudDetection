from ts.torch_handler.base_handler import BaseHandler
import torch
import numpy as np

class FraudDetectionHandler(BaseHandler):
    def __init__(self):
        super(FraudDetectionHandler, self).__init__()
        self.model = None

    def load_model(self):
        model_path = self.manifest['model_file']
        self.model = torch.jit.load(model_path)
        self.model.eval()
        

    def preprocess(self, data):
        features = np.array(data[0]['data']).astype(np.float32)
        if features.shape[0] != 4:  # Assuming 4 features as per original code
            raise ValueError(f"Expected 4 features, got {features.shape[0]}")
        return torch.tensor(features).unsqueeze(0)

    def inference(self, data):
        with torch.no_grad():
            logits = self.model(data)
            prob = torch.sigmoid(logits).item()
            prediction = int(prob > 0.5)
        return prediction, prob

    def postprocess(self, inference_output):
        prediction, prob = inference_output
        return {
            "prediction": prediction,
            "confidence": prob
        }

    def handle(self, data, context):
        if self.model is None:
            self.load_model()
        input_tensor = self.preprocess(data)
        inference_output = self.inference(input_tensor)
        return self.postprocess(inference_output)