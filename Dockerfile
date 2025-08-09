FROM python:3.10-slim
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip first
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/api ./src/api

ENV MODEL_PATH=/app/models/best_model.pkl
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
