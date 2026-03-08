# app/explainability.py
"""
GradCAM attention visualisation for YOLO detection models.

Uses EigenCAM — it computes principal components of the feature map activations
rather than gradients, making it compatible with multi-output detection heads
without needing to specify a differentiable target.
"""
import io
import base64
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Input resolution the model expects
_YOLO_IMGSZ = 640


def generate_gradcam_b64(yolo_model, img: Image.Image) -> str:
    """
    Generate an EigenCAM heatmap overlay for a YOLO prediction.

    Args:
        yolo_model: Loaded ultralytics YOLO instance (predictor.model).
        img:        PIL Image (any size/mode).

    Returns:
        Base64-encoded PNG string of the heatmap overlay.

    Raises:
        RuntimeError: If pytorch-grad-cam is not installed or the model
                      architecture is incompatible.
    """
    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "pytorch-grad-cam is required. Install with: pip install pytorch-grad-cam"
        ) from exc

    # ── Preprocess ────────────────────────────────────────────────────────────
    img_rgb = img.convert("RGB").resize((_YOLO_IMGSZ, _YOLO_IMGSZ))
    img_array = np.array(img_rgb).astype(np.float32) / 255.0  # [H, W, 3] in [0,1]

    # [1, 3, H, W] float32
    input_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

    # ── Target layer selection ─────────────────────────────────────────────────
    # Unwrap the ultralytics wrapper to get the raw PyTorch model.
    # model.model.model is the Sequential; [-2] is the last backbone conv block
    # before the detection head — a typical choice for spatial attention maps.
    torch_model = yolo_model.model
    try:
        target_layers = [torch_model.model[-2]]
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(
            f"Could not identify target layer for GradCAM: {exc}. "
            "The model architecture may differ from expected YOLO layout."
        ) from exc

    # ── EigenCAM ──────────────────────────────────────────────────────────────
    cam = EigenCAM(model=torch_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # [H, W] in [0, 1]

    # ── Overlay & encode ──────────────────────────────────────────────────────
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    pil_out = Image.fromarray(visualization)

    buf = io.BytesIO()
    pil_out.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    logger.info("GradCAM heatmap generated (%d bytes)", len(b64))
    return b64
