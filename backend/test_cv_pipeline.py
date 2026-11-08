import cv2
import numpy as np

from core.cv_pipeline import ClinicalCVPreprocessor


def _text_lines_image(seed=7):
    rng = np.random.default_rng(seed)
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    for y in range(60, 560, 40):
        cv2.line(img, (60, y), (740, y + int(rng.integers(-4, 5))), (20, 20, 20), 2)
    noise = rng.integers(0, 30, size=img.shape, dtype=np.uint8)
    return cv2.subtract(img, noise)


def _encode_jpg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_hough_fallback_handles_native_line_shape():
    out = ClinicalCVPreprocessor.process_prescription_image(
        _encode_jpg(_text_lines_image())
    )
    assert out["base64_data_uri"].startswith("data:image/png;base64,")
    assert out["telemetry"]["geometry"]["method"] == "Hough_SE2_Affine_Rotation_Fallback"


def test_hough_fallback_accepts_both_line_shapes():
    img = _text_lines_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 80,
                            minLineLength=int(img.shape[1] * 0.15), maxLineGap=20)
    assert lines is not None and len(lines) > 0
    for shaped in (lines.reshape(-1, 1, 4), lines.reshape(-1, 4)):
        count = 0
        for line in shaped.reshape(-1, 4):
            x1, y1, x2, y2 = line
            count += 1
        assert count == len(lines)


def test_invalid_bytes_raise_value_error():
    try:
        ClinicalCVPreprocessor.process_prescription_image(b"not-an-image")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid image bytes")
