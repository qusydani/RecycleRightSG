# tests/test_predictor.py
"""Unit tests for the YOLO inference wrapper and NEA guidance logic."""
from unittest.mock import MagicMock, patch

import pytest

from src.predictor import NEA_ADVICE_MAP, FALLBACK_ADVICE, TrashnetPredictor


# ── NEA guidance tests ─────────────────────────────────────────────────────────

YOLO_CLASSES = ["aluminium", "cardboard", "clothing", "e-waste",
                "glass", "metal", "paper", "plastic", "styrofoam"]

@pytest.mark.parametrize("cls", YOLO_CLASSES)
def test_all_classes_have_advice(cls):
    """Every YOLO class must have an explicit NEA advice entry."""
    assert cls in NEA_ADVICE_MAP, f"'{cls}' missing from NEA_ADVICE_MAP"


def test_unknown_class_uses_fallback():
    assert NEA_ADVICE_MAP.get("unknown_item", FALLBACK_ADVICE) == FALLBACK_ADVICE


def test_contamination_classes_say_do_not():
    """Styrofoam, e-waste, and clothing must NOT be put in the blue bin."""
    for cls in ("styrofoam", "e-waste", "clothing"):
        assert "DO NOT" in NEA_ADVICE_MAP[cls], \
            f"'{cls}' should have 'DO NOT' advice"


def test_recyclable_classes_say_blue_bin_ok():
    """Standard recyclables must be marked as Blue Bin OK."""
    for cls in ("aluminium", "cardboard", "paper", "plastic", "glass", "metal"):
        assert "Blue Bin OK" in NEA_ADVICE_MAP[cls], \
            f"'{cls}' should say 'Blue Bin OK'"


# ── Predictor filtering tests ──────────────────────────────────────────────────

def _make_box_mock(conf: float, xyxy: list, class_id: int = 0):
    """Helper: create a mock ultralytics box result.

    box.xyxy[0] must have a .tolist() method to match the real ultralytics
    tensor interface used in predictor.py.
    """
    xyxy_mock = MagicMock()
    xyxy_mock.tolist.return_value = xyxy

    box = MagicMock()
    box.conf = [conf]
    box.xyxy = [xyxy_mock]
    box.cls = [class_id]
    return box


@patch("src.predictor.YOLO")
def test_confidence_threshold_filters_low_confidence(mock_yolo_cls):
    """Detections below the threshold should be excluded."""
    from PIL import Image

    low_conf_box = _make_box_mock(conf=0.25, xyxy=[0, 0, 50, 50], class_id=0)

    result = MagicMock()
    result.boxes = [low_conf_box]
    mock_instance = MagicMock()
    mock_instance.names = {0: "plastic"}
    mock_instance.return_value = [result]
    mock_yolo_cls.return_value = mock_instance

    predictor = TrashnetPredictor("fake.pt", abstain_threshold=0.40)
    img = Image.new("RGB", (640, 640))
    detections = predictor.predict_pil(img)

    assert detections == [], "Low-confidence detection should be filtered out"


@patch("src.predictor.YOLO")
def test_spatial_filter_removes_full_image_boxes(mock_yolo_cls):
    """Boxes covering >95% of the image area should be discarded."""
    from PIL import Image

    # A box covering the entire 640×640 image
    full_box = _make_box_mock(conf=0.95, xyxy=[0, 0, 640, 640], class_id=1)

    result = MagicMock()
    result.boxes = [full_box]
    mock_instance = MagicMock()
    mock_instance.names = {1: "paper"}
    mock_instance.return_value = [result]
    mock_yolo_cls.return_value = mock_instance

    predictor = TrashnetPredictor("fake.pt", abstain_threshold=0.40)
    img = Image.new("RGB", (640, 640))
    detections = predictor.predict_pil(img)

    assert detections == [], "Full-image box should be removed by spatial filter"
