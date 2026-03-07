from typing import Dict, Optional, List

from pydantic import BaseModel, Field


class ShapRequest(BaseModel):
    caseId: str
    features: Dict[str, float | int | bool]


class ShapScore(BaseModel):
    feature: str
    contribution: float


class ShapResponse(BaseModel):
    caseId: str
    shapAvailable: bool
    baseValue: float
    shapScores: List[ShapScore]
    error: Optional[str] = None


class GradcamRequest(BaseModel):
    caseId: str
    imageUrl: Optional[str] = None
    imagePath: Optional[str] = None
    targetClassIndex: Optional[int] = None
    method: Optional[str] = "gradcampp"
    smoothPasses: Optional[int] = 1
    targetLayer: Optional[str] = None
    targetLayerMode: Optional[str] = "penultimate"
    camPercentileThreshold: Optional[int] = 35
    camBlurKernel: Optional[int] = 9


class GradcamResponse(BaseModel):
    caseId: str
    gradCamAvailable: bool
    heatmapPath: Optional[str] = None
    heatmapUrl: Optional[str] = None
    overlayPath: Optional[str] = None
    overlayUrl: Optional[str] = None
    rawHeatmapPath: Optional[str] = None
    rawHeatmapUrl: Optional[str] = None
    lesionMaskPath: Optional[str] = None
    lesionMaskUrl: Optional[str] = None
    baseImagePath: Optional[str] = None
    baseImageUrl: Optional[str] = None
    predictedClass: Optional[str] = None
    predictionConfidence: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
