import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
try:
    import matplotlib.cm as mpl_cm  # type: ignore
except Exception:
    mpl_cm = None

from model import build_model
from transforms import get_val_transforms


def build_transform(image_size: int = 224):
    return get_val_transforms(image_size)


def image_to_tensor(image: Image.Image, image_size: int = 224):
    transform = build_transform(image_size)
    tensor = transform(image).unsqueeze(0)
    return tensor


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor, target_index=None):
        output = self.model(tensor)
        if target_index is None:
            target_index = int(output.argmax(dim=1).item())
        score = output[0, target_index]
        self.model.zero_grad()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


class GradCamPlusPlus(GradCam):
    def generate(self, tensor, target_index=None):
        output = self.model(tensor)
        if target_index is None:
            target_index = int(output.argmax(dim=1).item())
        score = output[0, target_index]
        self.model.zero_grad()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM++ hooks failed")

        grads = self.gradients
        activations = self.activations
        grads_power_2 = grads**2
        grads_power_3 = grads**3
        eps = 1e-8

        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        weights = torch.sum(alpha * torch.relu(grads), dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


class EigenCam:
    """Class-agnostic CAM using SVD on feature map activations.

    Does NOT depend on classifier accuracy — safe to use even when the
    classification head predicts random outputs (e.g., under-trained model).
    The first right singular vector of the [C × H*W] feature matrix captures
    the dominant spatial activation pattern across all channels.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _input, output):
        self._activations = output

    def generate(self, tensor, target_index=None) -> np.ndarray:  # noqa: ARG002
        with torch.no_grad():
            self.model(tensor)

        if self._activations is None:
            raise RuntimeError("EigenCAM forward hook did not fire")

        # Shape: [1, C, H, W] → [C, H*W]
        acts = self._activations.squeeze(0)  # [C, H, W]
        c, h, w = acts.shape
        feature_matrix = acts.reshape(c, h * w).float()

        # SVD — first right singular vector captures the dominant spatial pattern.
        # Vt shape: [min(C,H*W), H*W]; Vt[0] is the first right singular vector,
        # already a spatial map of shape [H*W] that can be directly reshaped.
        _, _, Vt = torch.linalg.svd(feature_matrix, full_matrices=False)
        cam = Vt[0].reshape(h, w)  # [H, W]
        cam = cam.abs()
        cam = cam - cam.min()
        max_val = cam.max()
        if max_val > 1e-8:
            cam = cam / max_val
        return cam.cpu().numpy()

    def remove_hooks(self):
        self._handle.remove()


def normalize_cam_method(method: str) -> str:
    normalized = (method or "eigencam").strip().lower()
    if normalized in {"gradcam++", "gradcampp"}:
        return "gradcampp"
    if normalized == "gradcam":
        return "gradcam"
    return "eigencam"


def _postprocess_cam(cam: np.ndarray, enhanced: bool) -> np.ndarray:
    cam = np.clip(cam, 0, 1)
    if not enhanced or cv2 is None:
        return cam
    # Post-processing applied for visualization enhancement only.
    cam_blur = cv2.GaussianBlur(cam, (21, 21), 0)
    threshold = np.percentile(cam_blur, 50)
    cam_blur = np.where(cam_blur >= threshold, cam_blur, 0)
    max_val = cam_blur.max() if cam_blur.size else 0
    return cam_blur / max_val if max_val > 0 else cam_blur


def _resolve_cv2_colormap(colormap: str):
    if cv2 is None:
        return None
    key = (colormap or "jet").strip().lower()
    mapping = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
    }
    return mapping.get(key, cv2.COLORMAP_JET)


def build_lesion_mask(image: Image.Image) -> np.ndarray:
    """
    Detect inflamed/urticaria skin regions via colour analysis.

    Urticaria wheals range from deep red → salmon pink → pale-centred flares,
    so we cast a wider net than a simple red detector:
      1. HSV: broad red/pink hue band with low saturation floor
      2. LAB 'a': lowered threshold to capture lighter inflammation
      3. Relative redness: R channel significantly exceeds G and B
    Detected regions are dilated to include wheal borders, then the mask is
    blended with the original to avoid sharp edges on the heatmap.
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    if cv2 is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        # Wider hue range + lower saturation floor to catch pink/salmon wheals.
        mask_red_low = cv2.inRange(hsv, (0, 20, 50), (25, 255, 255))
        mask_red_high = cv2.inRange(hsv, (155, 20, 50), (179, 255, 255))
        mask_red = cv2.bitwise_or(mask_red_low, mask_red_high)

        # LAB 'a' channel — lowered threshold from 145 → 130 to pick up
        # lighter inflammatory tones.
        mask_lab = (lab[:, :, 1] > 130).astype(np.uint8) * 255

        # Relative redness: pixels where R significantly exceeds G and B.
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        mask_redness = (((r - g) > 10) & ((r - b) > 5) & (r.astype(np.uint16) > 80)).astype(np.uint8) * 255

        mask = cv2.bitwise_or(mask_red, mask_lab)
        mask = cv2.bitwise_or(mask, mask_redness)

        # Morphological clean-up: remove noise, then connect nearby lesion blobs.
        open_k = np.ones((3, 3), np.uint8)
        close_k = np.ones((9, 9), np.uint8)
        dilate_k = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        # Dilate to include wheal borders and surrounding erythema.
        mask = cv2.dilate(mask, dilate_k, iterations=1)

        lesion_mask = (mask > 0).astype(np.float32)

        # Soften mask edges so the heatmap blends naturally into clean skin.
        lesion_mask = cv2.GaussianBlur(lesion_mask, (21, 21), 0)
        if lesion_mask.max() > 0:
            lesion_mask = lesion_mask / lesion_mask.max()
    else:
        # Fallback without cv2: relative redness proxy.
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        lesion_mask = (((r - g) > 10) & ((r - b) > 5)).astype(np.float32)

    # Avoid degenerate fully-empty masks by falling back to full-image.
    if lesion_mask.mean() < 0.01:
        lesion_mask = np.ones((h, w), dtype=np.float32)
    return lesion_mask


def build_skin_mask(image: Image.Image) -> np.ndarray:
    """
    Detect the entire visible skin area (all skin tones, not just inflamed regions).

    Uses YCbCr + HSV skin-tone ranges so the heatmap covers the complete skin
    surface.  EigenCAM values then determine warm (lesion) vs cool (healthy skin)
    colouring, producing the full-coverage thermal gradient look.
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    if cv2 is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

        # YCbCr skin range — reliable across a wide range of skin tones.
        mask_ycc = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        # HSV skin range: warm hues (0-30°), low-medium saturation, not too dark.
        mask_hsv_low = cv2.inRange(hsv, (0, 8, 60), (30, 200, 255))
        # Include high-hue wrap (350-360°) for some skin tones.
        mask_hsv_high = cv2.inRange(hsv, (165, 8, 60), (179, 200, 255))
        mask_hsv = cv2.bitwise_or(mask_hsv_low, mask_hsv_high)

        mask = cv2.bitwise_or(mask_ycc, mask_hsv)

        # Fill holes, remove small specks, then expand to cover skin borders.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)

        skin_mask = (mask > 0).astype(np.float32)
        skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 0)
        if skin_mask.max() > 0:
            skin_mask = skin_mask / skin_mask.max()
    else:
        lum = rgb.mean(axis=2)
        skin_mask = (lum > 50).astype(np.float32)

    if skin_mask.mean() < 0.05:
        skin_mask = np.ones((h, w), dtype=np.float32)
    return skin_mask


def redness_cam(image: Image.Image) -> np.ndarray:
    """
    Erythema heatmap via CIE LAB a* channel — identical approach to the
    companion AURA recognition service (IT22577160/app/explain.py).

    The a* channel measures red-green opposition: the reddest/most inflamed
    skin area maps to 1.0, the least red area maps to 0.0.  This is
    class-agnostic and requires no model gradient, so it is robust even
    when the classifier has low accuracy.  For urticaria the resulting
    heatmap shows a clinically meaningful erythema gradient across the
    entire skin surface.
    """
    rgb = np.array(image.convert("RGB"))
    if cv2 is not None:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        a = lab[:, :, 1].astype(np.float32)
    else:
        # Approximate a* without cv2: use (R - G) as a redness proxy.
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        a = r - g + 128.0

    a_min, a_max = a.min(), a.max()
    cam = (a - a_min) / (a_max - a_min + 1e-6)
    return cam.astype(np.float32)


def _resize_cam(cam: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    if cv2 is not None:
        cam_resized = cv2.resize(cam, image_size)
    else:
        cam_resized = np.array(Image.fromarray(np.uint8(255 * cam)).resize(image_size), dtype=np.float32) / 255.0
    cam_resized = np.clip(cam_resized, 0, 1)
    return cam_resized


def build_cam_maps(
    cam: np.ndarray,
    image_size: tuple[int, int],
    lesion_mask: np.ndarray | None = None,
    apply_blur: bool = True,
    percentile: int = 75,
    blur_kernel: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    raw_cam = np.clip(cam, 0, 1)
    raw_cam = raw_cam / (raw_cam.max() + 1e-8)
    raw_resized = _resize_cam(raw_cam, image_size)

    masked_cam = raw_resized.copy()
    if lesion_mask is not None:
        if lesion_mask.shape != masked_cam.shape:
            lesion_mask = _resize_cam(lesion_mask, image_size)
        masked_cam = masked_cam * lesion_mask

    if apply_blur and cv2 is not None:
        kernel = max(3, int(blur_kernel))
        if kernel % 2 == 0:
            kernel += 1
        masked_cam = cv2.GaussianBlur(masked_cam, (kernel, kernel), 0)

    thr = np.percentile(masked_cam, percentile)
    masked_cam = np.where(masked_cam >= thr, masked_cam, 0)

    raw_resized = raw_resized / (raw_resized.max() + 1e-8)
    masked_cam = masked_cam / (masked_cam.max() + 1e-8)
    return raw_resized, masked_cam


def heatmap_image(
    cam: np.ndarray,
    image_size: tuple[int, int],
    enhanced: bool = True,
    colormap: str = "jet",
) -> Image.Image:
    cam = _postprocess_cam(cam, enhanced)
    if cv2 is not None:
        cam_resized = cv2.resize(cam, image_size)
        heatmap = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(heatmap, _resolve_cv2_colormap(colormap))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap)

    cam_resized = np.array(Image.fromarray(np.uint8(255 * cam)).resize(image_size), dtype=np.float32) / 255.0
    if mpl_cm is not None:
        cmap = mpl_cm.get_cmap(colormap if colormap in {"jet", "hot", "plasma", "inferno"} else "jet")
        heatmap = (cmap(cam_resized)[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(heatmap).convert("RGB")
    heatmap = np.stack([cam_resized * 255, np.zeros_like(cam_resized), (1.0 - cam_resized) * 255], axis=-1)
    return Image.fromarray(np.uint8(heatmap)).convert("RGB")


def overlay_heatmap(image: Image.Image, cam: np.ndarray, enhanced: bool = True, colormap: str = "jet") -> Image.Image:
    heatmap = np.array(heatmap_image(cam, image.size, enhanced=enhanced, colormap=colormap).convert("RGB"))
    base = np.array(image.convert("RGB"))
    if cv2 is not None:
        # 55% original + 45% heatmap — same ratio as AURA recognition service.
        # The dark background stays dark (original is already dark there);
        # skin regions get the full thermal gradient.
        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.addWeighted(base_bgr, 0.55, heatmap_bgr, 0.45, 0)
        return Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    return Image.blend(image.convert("RGB"), Image.fromarray(heatmap).convert("RGB"), alpha=0.45)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name", "resnet18")
    model = build_model(model_name, num_classes=2, pretrained=False).to(device)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("Missing model weights in checkpoint (state_dict/model_state_dict)")
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_name


def resolve_target_layer(
    model,
    model_name: str,
    target_layer: str | None = None,
    target_layer_mode: str = "penultimate",
):
    if model_name == "efficientnet_b0":
        explicit = (target_layer or "").strip()
        mode = (target_layer_mode or "penultimate").strip().lower()
        if explicit in {"features[-1]", "last"} or mode == "last":
            return model.features[-1]
        return model.features[-2]
    if target_layer == "layer3":
        return model.layer3[-1]
    return model.layer4[-1]


def apply_tta(image: Image.Image):
    return [
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap")
    parser.add_argument("--image", required=True, help="Absolute path to image")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", required=True, help="Output path for heatmap PNG")
    parser.add_argument("--method", default="eigencam", choices=["gradcam", "gradcampp", "gradcam++", "eigencam"])
    parser.add_argument("--tta", type=int, default=1, help="Use test-time augmentation (1=off, 2=flip)")
    parser.add_argument("--smooth", type=int, default=0, help="SmoothGrad-CAM++ passes (0=off)")
    parser.add_argument(
        "--target-layer",
        default=None,
        choices=["layer3", "layer4", "features[-2]", "features[-1]"],
        help="Target layer for Grad-CAM",
    )
    parser.add_argument(
        "--target-layer-mode",
        default="penultimate",
        choices=["penultimate", "last"],
        help="For EfficientNet, choose penultimate (features[-2]) or last (features[-1]) feature block",
    )
    parser.add_argument("--cam-percentile-threshold", type=int, default=75)
    parser.add_argument("--cam-blur-kernel", type=int, default=11)
    parser.add_argument(
        "--overlay",
        default="enhanced",
        choices=["raw", "enhanced"],
        help="Overlay style",
    )
    parser.add_argument(
        "--colormap",
        default="jet",
        choices=["jet", "hot", "plasma", "inferno", "turbo"],
        help="Colormap for heatmap coloring",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.out)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name", "resnet18")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=2, pretrained=False).to(device)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise SystemExit("Checkpoint missing model weights (state_dict/model_state_dict)")
    model.load_state_dict(state_dict)
    model.eval()

    method = normalize_cam_method(args.method)
    target_layer = resolve_target_layer(
        model,
        model_name,
        args.target_layer,
        target_layer_mode=args.target_layer_mode,
    )
    if method == "eigencam":
        cam_engine = EigenCam(model, target_layer)
    elif method == "gradcampp":
        cam_engine = GradCamPlusPlus(model, target_layer)
    else:
        cam_engine = GradCam(model, target_layer)

    base_image = Image.open(image_path).convert("RGB")
    images = apply_tta(base_image) if args.tta > 1 else [base_image]

    cams = []
    passes = max(1, args.smooth)
    for img in images[: args.tta]:
        for _ in range(passes):
            tensor = image_to_tensor(img).to(device)
            if args.smooth > 0:
                noise = torch.randn_like(tensor) * 0.05
                tensor = tensor + noise
            cams.append(cam_engine.generate(tensor))
    cam = np.mean(cams, axis=0)
    with torch.no_grad():
        logits = model(image_to_tensor(base_image).to(device))
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_idx].item())
    lesion_mask = build_lesion_mask(base_image)
    _, cam = build_cam_maps(
        cam=cam,
        image_size=base_image.size,
        lesion_mask=lesion_mask,
        apply_blur=args.overlay == "enhanced",
        percentile=args.cam_percentile_threshold,
        blur_kernel=args.cam_blur_kernel,
    )

    overlay = overlay_heatmap(base_image, cam, enhanced=args.overlay == "enhanced", colormap=args.colormap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path, format="PNG")
    print(f"Training architecture: {model_name}")
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Target layer used: {args.target_layer or args.target_layer_mode}")
    print(f"CAM method used: {method}")
    print(f"Predicted class index: {predicted_idx}")
    print(f"Prediction confidence: {confidence:.4f}")
    print(f"Saved Grad-CAM heatmap to {out_path}")


def test_gradcam(
    image_path: str,
    checkpoint_path: str = "checkpoints/best.pt",
    output_path: str = "outputs/debug_heatmap.jpg",
    method: str = "gradcampp",
    smooth: int = 1,
    colormap: str = "jet",
):
    image_file = Path(image_path)
    checkpoint_file = Path(checkpoint_path)
    out_file = Path(output_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = load_checkpoint_model(checkpoint_file, device)
    target_layer_label = "features[-2]" if model_name == "efficientnet_b0" else "layer4[-1]"
    target_layer = resolve_target_layer(model, model_name, target_layer_label, target_layer_mode="penultimate")

    normalized_method = normalize_cam_method(method)
    if normalized_method == "eigencam":
        engine = EigenCam(model, target_layer)
    elif normalized_method == "gradcampp":
        engine = GradCamPlusPlus(model, target_layer)
    else:
        engine = GradCam(model, target_layer)
    image = Image.open(image_file).convert("RGB")
    tensor = image_to_tensor(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_idx].item())

    cams = []
    for _ in range(max(1, smooth)):
        cam_tensor = tensor.clone()
        if smooth > 1:
            cam_tensor = cam_tensor + torch.randn_like(cam_tensor) * 0.05
        cams.append(engine.generate(cam_tensor, target_index=predicted_idx))
    cam = np.mean(cams, axis=0)
    lesion_mask = build_lesion_mask(image)
    _, cam = build_cam_maps(
        cam=cam,
        image_size=image.size,
        lesion_mask=lesion_mask,
        apply_blur=True,
        percentile=75,
        blur_kernel=11,
    )
    raw_heatmap = heatmap_image(cam, image.size, enhanced=False, colormap=colormap)
    overlay = overlay_heatmap(image, cam, enhanced=True, colormap=colormap)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_out = out_file.with_name(f"{out_file.stem}_raw{out_file.suffix}")
    raw_heatmap.save(raw_out)
    overlay.save(out_file)

    label = "ANGIOEDEMA" if predicted_idx == 1 else "URTICARIA"
    print(f"Training architecture: {model_name}")
    print(f"Checkpoint loaded: {checkpoint_file}")
    print(f"Target layer used: {target_layer_label}")
    print(f"CAM method used: {normalized_method}")
    print(f"Predicted class: {label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Raw heatmap saved: {raw_out}")
    print(f"Heatmap saved: {out_file}")


if __name__ == "__main__":
    main()
