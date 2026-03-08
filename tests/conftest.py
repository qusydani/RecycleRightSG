# tests/conftest.py
"""
Pytest configuration.

ultralytics must be mocked BEFORE any app module is imported, because
app/main.py creates a TrashnetPredictor (which calls YOLO()) at module level.
Injecting a mock into sys.modules here — at collection time — ensures it is
in place before app.main is ever imported.
"""
import io
import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

# ── Mock ultralytics at module level (before any app import) ───────────────────
_mock_yolo_instance = MagicMock()
_mock_yolo_instance.names = {0: "plastic", 1: "paper"}

_empty_result = MagicMock()
_empty_result.boxes = []
_mock_yolo_instance.return_value = [_empty_result]

_mock_yolo_cls = MagicMock(return_value=_mock_yolo_instance)
_mock_ultralytics = MagicMock()
_mock_ultralytics.YOLO = _mock_yolo_cls
sys.modules["ultralytics"] = _mock_ultralytics


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with the real app but a mocked YOLO model."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """A tiny valid JPEG image."""
    img = Image.new("RGB", (100, 100), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()
