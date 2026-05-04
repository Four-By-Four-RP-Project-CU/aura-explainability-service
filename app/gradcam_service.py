from __future__ import annotations

import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gradcam import (  # noqa: E402
    EigenCam,
    GradCam,
    GradCamPlusPlus,
    build_cam_maps,
    build_skin_mask,
    redness_cam,
    heatmap_image,
    image_to_tensor,
    load_checkpoint_model,
    normalize_cam_method,
    overlay_heatmap,
    resolve_target_layer,
)

logger = logging.getLogger(__name__)

HEATMAP_DIR = PROJECT_ROOT / "storage" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
DEMO_HEATMAP_DIR = PROJECT_ROOT / "storage" / "heatmaps-demo"
OVERLAY_DIR = PROJECT_ROOT / "storage" / "overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
BASE_IMAGE_DIR = PROJECT_ROOT / "storage" / "base_images"
BASE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
RAW_HEATMAP_DIR = PROJECT_ROOT / "storage" / "raw_heatmaps"
RAW_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
LESION_MASK_DIR = PROJECT_ROOT / "storage" / "lesion_masks"
LESION_MASK_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best.pt"
CLASS_LABELS = {0: "URTICARIA", 1: "ANGIOEDEMA"}
_RUNTIME: Optional["GradCamRuntime"] = None


def _ensure_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        return
    hf_repo = os.getenv("HF_REPO_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if not hf_repo:
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Set HF_REPO_ID env var to download from Hugging Face."
        )
    logger.info("Checkpoint not found locally. Downloading from Hugging Face repo: %s", hf_repo)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename="best.pt",
            token=hf_token,
            local_dir=str(CHECKPOINT_PATH.parent),
        )
        logger.info("Checkpoint downloaded to: %s", downloaded)
    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to download checkpoint from Hugging Face ({hf_repo}): {exc}"
        ) from exc


@dataclass
class GradCamArtifacts:
    heatmap_path: Path
    overlay_path: Path
    base_image_path: Path
    raw_heatmap_path: Path
    lesion_mask_path: Path
    predicted_class: str
    prediction_confidence: float


class GradCamRuntime:
    # Loads checkpoint once at startup and reuses model/hooks for all requests.
    def __init__(self) -> None:
        _ensure_checkpoint()
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.model_name = load_checkpoint_model(CHECKPOINT_PATH, self.device)
        self.gradcam_engines: dict[tuple[str, Optional[str], str], object] = {}

    def explain(
        self,
        image: Image.Image,
        target_class_index: Optional[int] = None,
        method: str = "eigencam",
        smooth_passes: int = 1,
        enhanced_overlay: bool = True,
        target_layer_name: Optional[str] = None,
        target_layer_mode: str = "penultimate",
        cam_percentile_threshold: int = 5,
        cam_blur_kernel: int = 11,
    ) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, int, float]:
        tensor = image_to_tensor(image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, predicted_idx].item())

        explain_idx = predicted_idx if target_class_index is None else int(target_class_index)
        explain_idx = max(0, min(1, explain_idx))
        method = normalize_cam_method(method)
        if target_layer_name is None and self.model_name != "efficientnet_b0":
            target_layer_name = "layer4"
        if target_layer_name is None and self.model_name == "efficientnet_b0":
            target_layer_name = "features[-2]" if target_layer_mode == "penultimate" else "features[-1]"
        engine_key = (method, target_layer_name, target_layer_mode)
        if engine_key not in self.gradcam_engines:
            target_layer = resolve_target_layer(
                self.model,
                self.model_name,
                target_layer_name,
                target_layer_mode=target_layer_mode,
            )
            if method == "eigencam":
                self.gradcam_engines[engine_key] = EigenCam(self.model, target_layer)
            elif method == "gradcampp":
                self.gradcam_engines[engine_key] = GradCamPlusPlus(self.model, target_layer)
            else:
                self.gradcam_engines[engine_key] = GradCam(self.model, target_layer)
            logger.info(
                "Grad-CAM engine initialized model=%s method=%s target_layer=%s mode=%s",
                self.model_name,
                method,
                target_layer_name,
                target_layer_mode,
            )
        engine = self.gradcam_engines[engine_key]

        cams = []
        passes = max(1, smooth_passes)
        for _ in range(passes):
            cam_tensor = tensor.clone()
            if passes > 1:
                cam_tensor = cam_tensor + torch.randn_like(cam_tensor) * 0.05
            cams.append(engine.generate(cam_tensor, target_index=explain_idx))

        cam = sum(cams) / len(cams)

        # Blend EigenCAM attention with the LAB a* redness map (same approach as
        # the companion AURA recognition service).  The redness map contributes
        # the clinically visible erythema gradient; EigenCAM contributes model
        # spatial attention.  Equal weighting gives full-coverage thermal output
        # that is both visually clear and model-grounded.
        r_cam = redness_cam(image)
        raw_cam_full, _ = build_cam_maps(
            cam=cam,
            image_size=image.size,
            lesion_mask=None,
            apply_blur=False,
            percentile=0,
            blur_kernel=cam_blur_kernel,
        )
        import cv2 as _cv2
        r_cam_resized = _cv2.resize(r_cam, image.size) if _cv2 is not None else \
            np.array(Image.fromarray(np.uint8(255 * r_cam)).resize(image.size), dtype=np.float32) / 255.0
        combined_cam = 0.5 * raw_cam_full + 0.5 * r_cam_resized
        combined_cam = combined_cam / (combined_cam.max() + 1e-8)

        raw_heatmap = heatmap_image(raw_cam_full, image.size, enhanced=False)
        masked_heatmap = heatmap_image(combined_cam, image.size, enhanced=False, colormap="jet")
        overlay = overlay_heatmap(image, combined_cam, enhanced=False, colormap="jet")
        skin_mask = build_skin_mask(image)
        lesion_mask_image = Image.fromarray((skin_mask * 255).astype("uint8")).convert("L")
        logger.info(
            "Grad-CAM inference predicted_idx=%d confidence=%.4f explain_idx=%d method=%s",
            predicted_idx,
            confidence,
            explain_idx,
            method,
        )
        return raw_heatmap, masked_heatmap, overlay, lesion_mask_image, predicted_idx, confidence


def _find_demo_heatmap(case_id: str) -> Optional[Path]:
    if not DEMO_HEATMAP_DIR.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = DEMO_HEATMAP_DIR / f"{case_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_image(image_url: Optional[str], image_path: Optional[str]) -> Image.Image:
    if image_url:
        parsed = urlparse(image_url.strip())
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("imageUrl must be http or https")
        response = requests.get(
            image_url.strip(),
            timeout=20,
            headers={"User-Agent": "AURA-Explainability/1.0"},
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError("imageUrl did not return an image content type")
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    if image_path:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError("imagePath does not exist")
        return Image.open(path).convert("RGB")
    raise ValueError("Either imageUrl or imagePath is required")


_TEST_IMAGES_DIR = PROJECT_ROOT / "data"
_SKIN_IMAGES_ROOT = Path("/Users/pradicksha/Documents/SLIIT/Y4S1/RP/AURA/CU Skin Images/test/Urticaria Hives")


def _resolve_image(case_id: str, image_url: Optional[str], image_path: Optional[str]) -> Image.Image:
    """Return a PIL image using the best available source for this case.

    Priority:
      1. Caller-supplied imageUrl / imagePath (live case data from MongoDB)
      2. Previously stored base_image for this caseId (from an earlier run)
      3. Image from the demo-heatmap folder (original skin images used for demos)
      4. First available test image from the local dataset (graceful fallback)
    """
    # 1. Caller-supplied image.
    if image_url or image_path:
        return load_image(image_url, image_path)

    # 2. Previously stored base_image.
    stored = BASE_IMAGE_DIR / f"{case_id}.png"
    if stored.exists():
        logger.info("Using stored base_image for caseId=%s from %s", case_id, stored)
        return Image.open(stored).convert("RGB")

    # 3. Raw image in the demo-heatmap source folder (original skin images).
    if DEMO_HEATMAP_DIR.exists():
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = DEMO_HEATMAP_DIR / f"{case_id}{ext}"
            if candidate.exists():
                logger.info("Using demo source image for caseId=%s from %s", case_id, candidate)
                return Image.open(candidate).convert("RGB")

    # 4. Pick the first available test image as a representative fallback.
    if _SKIN_IMAGES_ROOT.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            matches = sorted(_SKIN_IMAGES_ROOT.glob(ext))
            if matches:
                logger.warning(
                    "No image found for caseId=%s; using fallback test image %s", case_id, matches[0]
                )
                return Image.open(matches[0]).convert("RGB")

    raise FileNotFoundError(
        f"No image source available for caseId={case_id}. "
        "Provide imageUrl or imagePath, or populate storage/base_images/."
    )


def generate_heatmap(
    case_id: str,
    image_url: Optional[str],
    image_path: Optional[str],
    target_class_index: Optional[int] = None,
    method: str = "eigencam",
    smooth_passes: int = 1,
    enhanced_overlay: bool = True,
    target_layer_name: Optional[str] = None,
    target_layer_mode: str = "penultimate",
    cam_percentile_threshold: int = 5,
    cam_blur_kernel: int = 11,
) -> GradCamArtifacts:
    global _RUNTIME
    if _RUNTIME is None:
        logger.info("Loading trained Grad-CAM runtime from checkpoint")
        _RUNTIME = GradCamRuntime()

    image = _resolve_image(case_id, image_url, image_path)

    base_image_path = BASE_IMAGE_DIR / f"{case_id}.png"
    image.save(base_image_path, format="PNG")

    raw_heatmap_img, masked_heatmap_img, overlay_img, lesion_mask_img, predicted_idx, confidence = _RUNTIME.explain(
        image=image,
        target_class_index=target_class_index,
        method=method,
        smooth_passes=smooth_passes,
        enhanced_overlay=enhanced_overlay,
        target_layer_name=target_layer_name,
        target_layer_mode=target_layer_mode,
        cam_percentile_threshold=cam_percentile_threshold,
        cam_blur_kernel=cam_blur_kernel,
    )

    raw_heatmap_path = RAW_HEATMAP_DIR / f"{case_id}.png"
    raw_heatmap_img.save(raw_heatmap_path, format="PNG")

    heatmap_path = HEATMAP_DIR / f"{case_id}.png"
    masked_heatmap_img.save(heatmap_path, format="PNG")

    overlay_path = OVERLAY_DIR / f"{case_id}.png"
    overlay_img.save(overlay_path, format="PNG")

    lesion_mask_path = LESION_MASK_DIR / f"{case_id}.png"
    lesion_mask_img.save(lesion_mask_path, format="PNG")

    predicted_class = CLASS_LABELS.get(predicted_idx, str(predicted_idx))
    return GradCamArtifacts(
        heatmap_path=heatmap_path,
        overlay_path=overlay_path,
        base_image_path=base_image_path,
        raw_heatmap_path=raw_heatmap_path,
        lesion_mask_path=lesion_mask_path,
        predicted_class=predicted_class,
        prediction_confidence=confidence,
    )
