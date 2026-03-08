import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.file_server import safe_file_response
from app.gradcam_service import generate_heatmap
from app.models import GradcamRequest, GradcamResponse, HealthResponse, ShapRequest, ShapResponse
from app.security import require_api_key
from app.shap_service import compute_shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura-explainability")

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="AURA Explainability Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/explain/shap", response_model=ShapResponse, dependencies=[Depends(require_api_key)])
def explain_shap(request: ShapRequest) -> ShapResponse:
    try:
        base_value, shap_scores = compute_shap(request.features)
        return ShapResponse(
            caseId=request.caseId,
            shapAvailable=True,
            baseValue=base_value,
            shapScores=shap_scores,
        )
    except Exception as exc:
        logger.error("SHAP error: %s", str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.post("/explain/gradcam", response_model=GradcamResponse, dependencies=[Depends(require_api_key)])
def explain_gradcam(request: GradcamRequest) -> GradcamResponse:
    try:
        artifacts = generate_heatmap(
            request.caseId,
            request.imageUrl,
            request.imagePath,
            target_class_index=request.targetClassIndex,
            method=(request.method or "gradcampp"),
            smooth_passes=(request.smoothPasses or 1),
            target_layer_name=request.targetLayer,
            target_layer_mode=(request.targetLayerMode or "penultimate"),
            cam_percentile_threshold=(request.camPercentileThreshold or 40),
            cam_blur_kernel=(request.camBlurKernel or 11),
        )
        relative_heatmap_path = artifacts.heatmap_path.relative_to(BASE_DIR)
        relative_overlay_path = artifacts.overlay_path.relative_to(BASE_DIR)
        relative_base_path = artifacts.base_image_path.relative_to(BASE_DIR)
        relative_raw_heatmap_path = artifacts.raw_heatmap_path.relative_to(BASE_DIR)
        relative_lesion_mask_path = artifacts.lesion_mask_path.relative_to(BASE_DIR)
        api_key = os.getenv("EXPLAIN_API_KEY", "")
        key_suffix = f"?apiKey={api_key}" if api_key else ""
        return GradcamResponse(
            caseId=request.caseId,
            gradCamAvailable=True,
            heatmapPath=str(relative_heatmap_path),
            heatmapUrl=f"http://localhost:8001/files/{relative_heatmap_path}{key_suffix}",
            overlayPath=str(relative_overlay_path),
            overlayUrl=f"http://localhost:8001/files/{relative_overlay_path}{key_suffix}",
            rawHeatmapPath=str(relative_raw_heatmap_path),
            rawHeatmapUrl=f"http://localhost:8001/files/{relative_raw_heatmap_path}{key_suffix}",
            lesionMaskPath=str(relative_lesion_mask_path),
            lesionMaskUrl=f"http://localhost:8001/files/{relative_lesion_mask_path}{key_suffix}",
            baseImagePath=str(relative_base_path),
            baseImageUrl=f"http://localhost:8001/files/{relative_base_path}{key_suffix}",
            predictedClass=artifacts.predicted_class,
            predictionConfidence=artifacts.prediction_confidence,
        )
    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", str(exc))
        return GradcamResponse(
            caseId=request.caseId,
            gradCamAvailable=False,
            error=str(exc),
        )


@app.get("/files/{path:path}")
def files(path: str, _: None = Depends(require_api_key)):
    return safe_file_response(path)
