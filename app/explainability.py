# app/explainability.py
"""
D-RISE saliency visualisation for YOLO detection models.

D-RISE (Detector Randomized Input Sampling for Explanation) is a black-box
perturbation method that directly measures which image regions causally affect
the confidence and localisation of a specific detection.

Reference: Petsiuk et al., "Black-box Explanation of Object Detectors
via Saliency Maps", CVPR 2021 — https://arxiv.org/abs/2006.03204
"""
import io
import base64
import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

_YOLO_IMGSZ = 640


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def generate_drise_b64(
    yolo_model,
    img: Image.Image,
    n_masks: int = 200,
    mask_res: int = 8,
    batch_size: int = 8,
) -> str:
    """
    Generate a D-RISE saliency map for the highest-confidence detection.

    Args:
        yolo_model: Loaded ultralytics YOLO instance (predictor.model).
        img:        PIL Image (any size/mode).
        n_masks:    Number of random masks to sample (higher = better quality, slower).
        mask_res:   Low-resolution grid size for mask generation (upsampled bilinearly).
        batch_size: Number of masked images per YOLO inference call.

    Returns:
        Base64-encoded PNG string of the D-RISE saliency overlay.

    Raises:
        RuntimeError: If no detections are found or pytorch-grad-cam is missing.
    """
    try:
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise RuntimeError(
            "pytorch-grad-cam is required for heatmap overlay. "
            "Install with: pip install pytorch-grad-cam"
        ) from exc

    # ── Preprocess ────────────────────────────────────────────────────────────
    img_rgb  = img.convert("RGB").resize((_YOLO_IMGSZ, _YOLO_IMGSZ))
    img_arr  = np.array(img_rgb).astype(np.float32) / 255.0   # [H, W, 3]
    img_blur = np.array(
        img_rgb.filter(ImageFilter.GaussianBlur(radius=16))
    ).astype(np.float32) / 255.0

    # ── Reference detection ───────────────────────────────────────────────────
    ref_results = yolo_model.predict(img_rgb, imgsz=_YOLO_IMGSZ, verbose=False)
    ref = ref_results[0]
    if ref.boxes is None or len(ref.boxes) == 0:
        raise RuntimeError(
            "No detections found — upload an image with a detectable object first."
        )

    bi      = int(ref.boxes.conf.argmax())
    ref_box = ref.boxes.xyxy[bi].cpu().numpy().astype(np.float32)  # [x1,y1,x2,y2]
    ref_cls = int(ref.boxes.cls[bi].item())

    # ── Generate random upsampled masks ───────────────────────────────────────
    rng   = np.random.default_rng(42)
    low   = rng.integers(0, 2, (n_masks, mask_res, mask_res), dtype=np.uint8)
    masks = np.zeros((n_masks, _YOLO_IMGSZ, _YOLO_IMGSZ), dtype=np.float32)
    for i, m in enumerate(low):
        masks[i] = np.array(
            Image.fromarray(m * 255).resize((_YOLO_IMGSZ, _YOLO_IMGSZ), Image.BILINEAR)
        ).astype(np.float32) / 255.0

    # ── Score each masked image ───────────────────────────────────────────────
    weights = np.zeros(n_masks, dtype=np.float32)

    for start in range(0, n_masks, batch_size):
        batch_masks = masks[start : start + batch_size]
        batch_imgs  = []
        for m in batch_masks:
            m3     = m[:, :, np.newaxis]
            masked = img_arr * m3 + img_blur * (1.0 - m3)
            batch_imgs.append(Image.fromarray((masked * 255).astype(np.uint8)))

        batch_results = yolo_model.predict(batch_imgs, imgsz=_YOLO_IMGSZ, verbose=False)

        for j, res in enumerate(batch_results):
            best = 0.0
            if res.boxes is not None and len(res.boxes) > 0:
                for k in range(len(res.boxes)):
                    if int(res.boxes.cls[k].item()) != ref_cls:
                        continue
                    det_box = res.boxes.xyxy[k].cpu().numpy().astype(np.float32)
                    score   = _iou(ref_box, det_box) * float(res.boxes.conf[k].item())
                    best    = max(best, score)
            weights[start + j] = best

    # ── Saliency = weighted sum of masks, normalised to [0, 1] ───────────────
    saliency = np.einsum("n,nhw->hw", weights, masks)
    lo, hi   = saliency.min(), saliency.max()
    if hi > lo:
        saliency = (saliency - lo) / (hi - lo)

    # ── Overlay & encode ──────────────────────────────────────────────────────
    visualization = show_cam_on_image(img_arr, saliency, use_rgb=True)
    buf = io.BytesIO()
    Image.fromarray(visualization).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    logger.info(
        "D-RISE saliency map generated (%d bytes, %d masks, ref_cls=%d)",
        len(b64), n_masks, ref_cls,
    )
    return b64
