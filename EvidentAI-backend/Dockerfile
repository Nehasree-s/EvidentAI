FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for compilation if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip, setuptools, and wheel first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Force pip to prefer pre-built binary wheels to avoid building spacy/thinc from source
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]