FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY . .

# Expose port
EXPOSE 8000

# Tensorboard 
EXPOSE 6006

# Start FastAPI app
# CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
# # Tensorboard
# CMD ["tensorboard", "--logdir", "runs", "--host", "0.0.0.0", "--port", "6006"]

CMD bash -c "tensorboard --logdir runs --host 0.0.0.0 --port 6006 & uvicorn server:app --host 0.0.0.0 --port 8000"

