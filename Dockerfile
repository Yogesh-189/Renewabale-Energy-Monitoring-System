# Renewable Energy Monitoring and Optimization System - container image
# Build:  docker build -t solar-monitoring .
# Run:    docker run -p 8000:8000 solar-monitoring     then open http://localhost:8000
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    OPEN_DOCS_ON_START=0

WORKDIR /app

# Install dependencies first so Docker can cache this layer between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/src
EXPOSE 8000

# Cloud platforms (Render, Railway, Cloud Run...) tell the app which port to use via $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
