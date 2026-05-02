FROM python:3.11-slim

WORKDIR /app

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY checkpoints/best.pt ./checkpoints/best.pt

RUN mkdir -p storage/heatmaps storage/overlays storage/base_images \
             storage/raw_heatmaps storage/lesion_masks

EXPOSE 8001

ENV EXPLAIN_API_KEY=""
ENV SERVICE_BASE_URL="http://localhost:8001"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
