FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (build-essential, libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained models
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
