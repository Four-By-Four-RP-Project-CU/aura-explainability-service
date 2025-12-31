# AURA Explainability Service

FastAPI microservice for SHAP explanations (tabular) and Grad-CAM heatmaps (images).

## Requirements
- Python 3.10+

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export EXPLAIN_API_KEY=dev-key
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## API Key
Send the key in `X-API-KEY` header. The server checks against `EXPLAIN_API_KEY`.

## Endpoints
- `GET /health`
- `POST /explain/shap`
- `POST /explain/gradcam`
- `GET /files/{path}`

## SHAP Example
```bash
curl -X POST http://localhost:8001/explain/shap \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: dev-key" \
  -d '{
    "caseId": "CU001",
    "features": {
      "age": 34,
      "CRP": 10.39,
      "IgE": 766.59,
      "VitD": 16.73,
      "itchingScore": 2,
      "uctTotal": 10,
      "aectTotal": 7,
      "angioedemaPresent": true,
      "whealsPresent": false
    }
  }'
```

## Grad-CAM Example (URL)
```bash
curl -X POST http://localhost:8001/explain/gradcam \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: dev-key" \
  -d '{
    "caseId": "CU001",
    "imageUrl": "https://example.com/image.jpg"
  }'
```

## Grad-CAM Example (Local path)
```bash
curl -X POST http://localhost:8001/explain/gradcam \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: dev-key" \
  -d '{
    "caseId": "CU001",
    "imagePath": "./samples/CU001.jpg"
  }'
```

## Notes
- Heatmaps are saved to `storage/heatmaps` and served via `/files/{path}`.
- No training code is included; a fixed model is used if no joblib model is found.
