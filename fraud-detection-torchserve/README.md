# Fraud Detection TorchServe

This project provides a TorchServe compatible solution for a fraud detection model. It includes the necessary files and configurations to deploy the model using TorchServe.

## Project Structure

```
fraud-detection-torchserve
├── model-store
│   └── fraud_detection.mar          # Serialized model in TorchScript format
├── src
│   ├── fraud_detection_handler.py    # Custom handler for the fraud detection model
│   └── fraud_detection_model.py      # Model definition and processing functions
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── config.properties                 # Configuration settings for TorchServe
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fraud-detection-torchserve
   ```

2. **Install dependencies**:
   Ensure you have Python 3.6 or higher installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   torch-model-archiver --model-name fraud_detection --version 1.0 --serialized-file model-store/fraud_prevention_model.pt --handler src/fraud_detection_handler.py --export-path model-store --extra-files src/fraud_detection_model.py --force

3. **Model Deployment**:
   To deploy the model using TorchServe, run the following command:
   ```bash
   torchserve --start --model-store model-store --ts-config config.properties --disable-token-auth --models fraud_detection=fraud_detection.mar    
   ```

4. **Accessing the API**:
   Once the model is deployed, you can access the prediction API at:
   ```
   http://localhost:8080/predictions/fraud_detection
   ```

## Usage Example

To make a prediction, send a POST request to the prediction endpoint with the required input features. Here is an example using `curl`:

```bash
curl -X POST http://localhost:8080/predictions/fraud_detection -H "Content-Type: application/json" -d '{"data": [[100.0, 5.0, 1, 6]]}'
```

## Model Information

The fraud detection model is designed to predict the likelihood of fraudulent activity based on input features. The model has been trained on relevant datasets and is optimized for inference speed and accuracy.

## Configuration

The `config.properties` file contains settings for the TorchServe deployment, including model parameters and logging configurations. Adjust these settings as necessary for your deployment environment.
